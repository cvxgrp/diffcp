import diffcp.cones as cone_lib

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scs

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import _diffcp

def pi(z, cones):
    """Projection onto R^n x K^* x R_+

    `cones` represents a convex cone K, and K^* is its dual cone.
    """
    u, v, w = z
    return np.concatenate(
        [u, cone_lib.pi(v, cones, dual=True), np.maximum(w, 0)])


def dpi_sparse_matrix(z, cones):
    """Derivative of projection onto R^n x K^* x R_+
     `cones` represents a conex cone K, and K^* is its dual cone.
    """
    u, v, w = z
    return sparse.block_diag([
        sparse.eye(np.prod(u.shape)),
        cone_lib.dpi_sparse_matrix(v, cones, dual=True),
        sparse.diags(.5 * (np.sign(w) + 1))
    ])



def solve_and_derivative_wrapper(A, b, c, cone_dict, warm_start, kwargs):
    return solve_and_derivative(
        A, b, c, cone_dict, warm_start=warm_start, **kwargs)


def solve_and_derivative_batch(As, bs, cs, cone_dicts, n_jobs=-1,
                               warm_starts=None, **kwargs):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    batch_size = len(As)
    pool = ThreadPool(processes=n_jobs)
    args = []
    for i in range(batch_size):
        args += [(As[i], bs[i], cs[i], cone_dicts[i],
                  None if warm_starts is None else warm_starts[i], kwargs)]
    return pool.starmap(solve_and_derivative_wrapper, args)


class SolverError(Exception):
    pass


def solve_and_derivative(A, b, c, cone_dict, warm_start=None, mode='lsqr', **kwargs):
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
      mode: (optional) Which mode to compute derivative with, options are ["dense", "sparse", "lsqr"].
      kwargs: (optional) Keyword arguments to send to SCS.
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
    Raises:
        SolverError: if the cone program is infeasible or unbounded.
    """
    if mode not in ["dense", "sparse", "lsqr"]:
        return NotImplementedError

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

    # check status
    status = result["info"]["status"]
    if status == "Solved/Innacurate":
        warnings.warn("Solved/Innacurate.")
    elif status != "Solved":
        raise SolverError("Solver scs returned status %s" % status)

    x = result["x"]
    y = result["y"]
    s = result["s"]

    # pre-compute quantities for the derivative
    m, n = A.shape
    N = m + n + 1
    cones = cone_lib.parse_cone_dict(cone_dict)
    cones_parsed = cone_lib.parse_cone_dict_cpp(cones)
    z = (x, y - s, np.array([1]))
    u, v, w = z

    Q = sparse.bmat([
        [None, A.T, np.expand_dims(c, - 1)],
        [-A, None, np.expand_dims(b, -1)],
        [-np.expand_dims(c, -1).T, -np.expand_dims(b, -1).T, None]
    ])
    Q_dense = Q.todense()

    D_proj_dual_cone = _diffcp.dprojection(v, cones_parsed, True)
    if mode == "dense":
        M = _diffcp.M_dense(Q_dense, cones_parsed, u, v, w)
        MT = M.T
    elif mode == "sparse":
        M = (Q - sparse.eye(N)) @ dpi_sparse_matrix(z, cones) + sparse.eye(N)
        MT = M.T

    if mode == "lsqr":
        M = _diffcp.M_operator(Q, cones_parsed, u, v, w)
        MT = M.transpose()

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
        rhs = dQ @ pi_z
        if np.allclose(rhs, 0):
            dz = np.zeros(rhs.size)
        elif mode == "dense":
            dz = _diffcp._solve_derivative_dense(M, MT, rhs)
        elif mode == "sparse":
            rho = kwargs.get("rho", 1e-6)
            it_ref_iters = kwargs.get("it_ref_iters", 10)
            M_iref = sparse.bmat([
                [-sparse.eye(N), M],
                [MT, rho * sparse.eye(N)]
            ])
            solve = splinalg.factorized(M_iref.tocsc())
            rhs = MT @ rhs
            dz = np.zeros(N)
            # iterative refinement
            for _ in range(it_ref_iters):
                residual = rhs - MT @ (M @ dz)
                if np.linalg.norm(residual) <= 1e-10:
                    break
                dz = dz + solve(np.append(np.zeros(N), residual))[N:]
        elif mode == "lsqr":
            dz = _diffcp.lsqr(M, rhs).solution

        du, dv, dw = np.split(dz, [n, n + m])
        dx = du - x * dw
        dy = D_proj_dual_cone.matvec(dv) - y * dw
        ds = D_proj_dual_cone.matvec(dv) - dv - s * dw
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
        elif mode == "dense":
            r = _diffcp._solve_adjoint_derivative_dense(M, MT, dz)
        elif mode == "sparse":
            rho = kwargs.get("rho", 1e-6)
            it_ref_iters = kwargs.get("it_ref_iters", 5)
            MT_iref = sparse.bmat([
                [-sparse.eye(N), MT],
                [M, rho * sparse.eye(N)]
            ])
            solve = splinalg.factorized(MT_iref.tocsc())
            rhs = M @ dz
            r = np.zeros(N)
            # iterative refinement
            for k in range(it_ref_iters):
                residual = rhs - M @ (MT @ r)
                if np.linalg.norm(residual) <= 1e-10:
                    break
                r = r + solve(np.append(np.zeros(N), residual))[N:]
        elif mode == "lsqr":
            r = _diffcp.lsqr(MT, dz).solution
        else:
            raise NotImplementedError(f'Unrecognized mode: {mode}')

        values = pi_z[cols] * r[rows + n] - pi_z[n + rows] * r[cols]
        dA = sparse.csc_matrix((values, (rows, cols)), shape=A.shape)
        db = pi_z[n:n + m] * r[-1] - pi_z[-1] * r[n:n + m]
        dc = pi_z[:n] * r[-1] - pi_z[-1] * r[:n]

        return dA, db, dc

    return x, y, s, derivative, adjoint_derivative
