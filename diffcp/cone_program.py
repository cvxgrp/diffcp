import diffcp.cones as cone_lib

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scs


def pi(z, cones):
    """Projection onto R^n x K^* x R_+

    `cones` represents a convex cone K, and K^* is its dual cone.
    """
    u, v, w = z
    return np.concatenate(
        [u, cone_lib.pi(v, cones, dual=True), np.maximum(w, 0)])


def dpi(z, cones):
    """Derivative of projection onto R^n x K^* x R_+

     `cones` represents a conex cone K, and K^* is its dual cone.
    """
    u, v, w = z
    return cone_lib.as_block_diag_linear_operator([
        sparse.eye(np.prod(u.shape)),
        cone_lib.dpi(v, cones, dual=True),
        sparse.diags(.5 * (np.sign(w) + 1))
    ])


def solve_and_derivative(A, b, c, cone_dict, warm_start=None, **kwargs):
    """Solves a cone program, returns its derivative as an abstract linear map.

    This function solves a convex cone program, with primal-dual problems

        min.        c^T x                  min.        b^Ty
        subject to  Ax + s = b             subject to  A^Ty + c = 0
                    s \in K                            y \in K^*

    The problem data A, b, and c correspond to the arguments `A`, `b`, and `c`,
    and the convex cone `K` corresponds to `cone_dict`; x and s are the primal
    variables, and y is the dual variable.

    This function returns a solution (x, y, s) to the program. It also returns
    two functions that respectively represent application of the derivative
    (at A, b, and c) and its adjoint.

    The problem data must be formatted according to the SCS convention, see
    https://github.com/cvxgrp/scs.

    For background on derivatives of cone programs, see
    http://web.stanford.edu/~boyd/papers/diff_cone_prog.html.

    Args:
      A: A sparse SciPy matrix in CSC format; the first block of rows must
        correspondond to the zero cone, the next block to the positive orthant,
        then the second-order cone, the PSD cone, the exponential cone, and
        finally the exponential dual cone. PSD matrix variables must be
        vectorized by scaling the off-diagonal entries by sqrt(2) and stacking
        the lower triangular part in column-major order.
      b: A NumPy array representing the offset.
      c: A NumPy array representing the objective function.
      cone_dict: A dictionary with keys corresponding to cones, values
          corresponding to their dimensions. The keys must be a subset of
          diffcp.ZERO, diffcp.POS, diffcp.SOC, diffcp.PSD, diffcp.EXP;
          the values of diffcp.SOC, diffcp.PSD, and diffcp.EXP
          should be lists. A k-dimensional PSD cone corresponds to a k-by-k
          matrix variable; a value of k for diffcp.EXP corresponds to k / 3
          exponential cones. See SCS documentation for more details.
      warm_start: (optional) A tuple (x, y, s) at which to warm-start SCS.
      kwargs: (optional) Keyword arguments to forward to SCS.

    Returns:
        x: Optimal value of the primal variable x.
        y: Optimal value of the dual variable y.
        s: Optimal value of the slack variable s.
        derivative: A callable with signature
                derivative(dA, db, dc) -> dx, dy, ds
            that applies the derivative of the cone program at (A, b, and c)
            to the perturbations `dA`, `db`, `dc`. `dA` must be a SciPy sparse
            matrix in CSC format with the same sparsity pattern as `A`;
            `db` and `dc` are NumPy arrays.
        adjoint_derivative: A callable with signature
                adjoint_derivative(dx, dy, ds) -> dA, db, dc
            that applies the adjoint of the derivative of the cone program at
            (A, b, and c) to the perturbations `dx`, `dy`, `ds`, which must be
            NumPy arrays. The output `dA` matches the sparsity pattern of `A`.

    """
    data = {
        "A": A,
        "b": b,
        "c": c
    }
    if warm_start is not None:
        data["x"] = warm_start[0]
        data["y"] = warm_start[1]
        data["s"] = warm_start[2]

    kwargs.setdefault("verbose", False)
    result = scs.solve(data, cone_dict, **kwargs)

    x = result["x"]
    y = result["y"]
    s = result["s"]

    # pre-compute quantities for the derivative
    m, n = A.shape
    N = m + n + 1
    cones = cone_lib.parse_cone_dict(cone_dict)
    z = (x, y - s, np.array([1]))
    u, v, w = z
    D_proj_dual_cone = cone_lib.dpi(v, cones, dual=True)
    Q = sparse.bmat([
        [None, A.T, np.expand_dims(c, - 1)],
        [-A, None, np.expand_dims(b, -1)],
        [-np.expand_dims(c, -1).T, -np.expand_dims(b, -1).T, None]
    ])
    M = splinalg.aslinearoperator(Q - sparse.eye(N)) @ dpi(
        z, cones) + splinalg.aslinearoperator(sparse.eye(N))
    pi_z = pi(z, cones)
    rows, cols = A.nonzero()

    def derivative(dA, db, dc, **kwargs):
        """Applies derivative at (A, b, c) to perturbations dA, db, dc

        Args:
            dA: SciPy sparse matrix in CSC format; must have same sparsity
                pattern as the matrix `A` from the cone program
            db: NumPy array representing perturbation in `b`
            dc: NumPy array representing perturbation in `c`

        Returns:
           NumPy arrays dx, dy, ds, the result of applying the derivative
           to the perturbations.
        """
        dQ = sparse.bmat([
            [None, dA.T, np.expand_dims(dc, - 1)],
            [-dA, None, np.expand_dims(db, -1)],
            [-np.expand_dims(dc, -1).T, -np.expand_dims(db, -1).T, None]
        ])
        # can ignore w since w = 1
        rhs = dQ @ pi_z
        if np.allclose(rhs, 0):
            dz = np.zeros(rhs.size)
        else:
            dz = splinalg.lsqr(M, rhs, **kwargs)[0]
        du, dv, dw = np.split(dz, [n, n + m])
        dx = du - x * dw
        dy = D_proj_dual_cone@dv - y * dw
        ds = D_proj_dual_cone@dv - dv - s * dw
        return -dx, -dy, -ds

    def adjoint_derivative(dx, dy, ds, **kwargs):
        """Applies adjoint of derivative at (A, b, c) to perturbations dx, dy, ds

        Args:
            dx: NumPy array representing perturbation in `x`
            dy: NumPy array representing perturbation in `y`
            ds: NumPy array representing perturbation in `s`

        Returns:
            (`dA`, `db`, `dc`), the result of applying the adjoint to the
            perturbations; the sparsity pattern of `dA` matches that of `A`.
        """
        dw = -(x @ dx + y @ dy + s @ ds)
        dz = np.concatenate(
            [dx, D_proj_dual_cone.rmatvec(dy + ds) - ds, np.array([dw])])
        if np.allclose(dz, 0):
            r = np.zeros(dz.shape)
        else:
            r = splinalg.lsqr(
                cone_lib.transpose_linear_operator(M), dz, **kwargs)[0]

        # dQ is the outer product of pi_z and r. Instead of materializing this,
        # the code below only computes the entries needed to compute dA, db, dc
        values = pi_z[cols] * r[rows + n] - pi_z[n + rows] * r[cols]
        dA = sparse.csc_matrix((values, (rows, cols)), shape=A.shape)
        db = pi_z[n:n + m] * r[-1] - pi_z[-1] * r[n:n + m]
        dc = pi_z[:n] * r[-1] - pi_z[-1] * r[:n]
        return dA, db, dc

    return x, y, s, derivative, adjoint_derivative
