import warnings

import diffcp.cones as cone_lib

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scs
from threadpoolctl import threadpool_limits

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


def solve_and_derivative_wrapper(A, b, c, cone_dict, warm_start, mode, kwargs):
    """A wrapper around solve_and_derivative for the batch function."""
    return solve_and_derivative(
        A, b, c, cone_dict, warm_start=warm_start, mode=mode, **kwargs)


def solve_and_derivative_batch(As, bs, cs, cone_dicts, n_jobs_forward=-1, n_jobs_backward=-1,
                               mode="lsqr", warm_starts=None, **kwargs):
    """
    Solves a batch of cone programs and returns a function that
    performs a batch of derivatives. Uses a ThreadPool to perform
    operations across the batch in parallel.

    For more information on the arguments and return values,
    see the docstring for `solve_and_derivative` function.

    Args:
        As - A list of A matrices.
        bs - A list of b arrays.
        cs - A list of c arrays.
        cone_dicts - A list of dictionaries describing the cone.
        n_jobs_forward - Number of jobs to use in the forward pass. n_jobs_forward = 1
            means serial and n_jobs_forward = -1 defaults to the number of CPUs (default=-1).
        n_jobs_backward - Number of jobs to use in the backward pass. n_jobs_backward = 1
            means serial and n_jobs_backward = -1 defaults to the number of CPUs (default=-1).
        mode - Differentiation mode in ["lsqr", "dense"].
        warm_starts - A list of warm starts.
        kwargs - kwargs sent to scs.

    Returns:
        xs: A list of x arrays.
        ys: A list of y arrays.
        ss: A list of s arrays.
        D_batch: A callable with signature
                D_batch(dAs, dbs, dcs) -> dxs, dys, dss
            This callable maps lists of problem data derivatives to lists of solution derivatives.
        DT_batch: A callable with signature
                DT_batch(dxs, dys, dss) -> dAs, dbs, dcs
            This callable maps lists of solution derivatives to lists of problem data derivatives.
    """
    batch_size = len(As)
    if warm_starts is None:
        warm_starts = [None]*batch_size
    if n_jobs_forward == -1:
        n_jobs_forward = mp.cpu_count()
    if n_jobs_backward == -1:
        n_jobs_backward = mp.cpu_count()
    n_jobs_forward = min(batch_size, n_jobs_forward)
    n_jobs_backward = min(batch_size, n_jobs_backward)

    if n_jobs_forward == 1:
        # serial
        xs, ys, ss, Ds, DTs = [], [], [], [], []
        for i in range(batch_size):
            x, y, s, D, DT = solve_and_derivative(As[i], bs[i], cs[i],
                    cone_dicts[i], warm_starts[i], mode=mode, **kwargs)
            xs += [x]
            ys += [y]
            ss += [s]
            Ds += [D]
            DTs += [DT]
    else:
        # thread pool
        pool = ThreadPool(processes=n_jobs_forward)
        args = [(A, b, c, cone_dict, warm_start, mode, kwargs) for A, b, c, cone_dict, warm_start in \
                    zip(As, bs, cs, cone_dicts, warm_starts)]
        with threadpool_limits(limits=1):
            results = pool.starmap(solve_and_derivative_wrapper, args)
        pool.close()
        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        ss = [r[2] for r in results]
        Ds = [r[3] for r in results]
        DTs = [r[4] for r in results]

    if n_jobs_backward == 1:
        def D_batch(dAs, dbs, dcs, **kwargs):
            dxs, dys, dss = [], [], []
            for i in range(batch_size):
                dx, dy, ds = Ds[i](dAs[i], dbs[i], dcs[i], **kwargs)
                dxs += [dx]
                dys += [dy]
                dss += [ds]
            return dxs, dys, dss
        
        def DT_batch(dxs, dys, dss, **kwargs):
            dAs, dbs, dcs = [], [], []
            for i in range(batch_size):
                dA, db, dc = DTs[i](dxs[i], dys[i], dss[i], **kwargs)
                dAs += [dA]
                dbs += [db]
                dcs += [dc]
            return dAs, dbs, dcs
    else:

        def D_batch(dAs, dbs, dcs, **kwargs):
            pool = ThreadPool(processes=n_jobs_backward)
            def Di(i):
                return Ds[i](dAs[i], dbs[i], dcs[i], **kwargs)
            results = pool.map(Di, range(batch_size))
            pool.close()
            dxs = [r[0] for r in results]
            dys = [r[1] for r in results]
            dss = [r[2] for r in results]
            return dxs, dys, dss

        def DT_batch(dxs, dys, dss, **kwargs):
            pool = ThreadPool(processes=n_jobs_backward)
            def DTi(i):
                return DTs[i](dxs[i], dys[i], dss[i], **kwargs)
            results = pool.map(DTi, range(batch_size))
            pool.close()
            dAs = [r[0] for r in results]
            dbs = [r[1] for r in results]
            dcs = [r[2] for r in results]
            return dAs, dbs, dcs

    return xs, ys, ss, D_batch, DT_batch


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
      mode: (optional) Which mode to compute derivative with, options are
          ["dense", "lsqr"].
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
    result = solve_and_derivative_internal(
        A, b, c, cone_dict, warm_start=warm_start, mode=mode, **kwargs)
    x = result["x"]
    y = result["y"]
    s = result["s"]
    D = result["D"]
    DT = result["DT"]
    return x, y, s, D, DT


def solve_and_derivative_internal(A, b, c, cone_dict, warm_start=None,
                                  mode='lsqr', raise_on_error=True, **kwargs):
    if mode not in ["dense", "lsqr"]:
        raise ValueError("Unsupported mode {}; the supported modes are "
                         "'dense' and 'lsqr'".format(mode))

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
    if status == "Solved/Inaccurate":
        warnings.warn("Solved/Inaccurate.")
    elif status != "Solved":
        if raise_on_error:
            raise SolverError("Solver scs returned status %s" % status)
        else:
            result["D"] = None
            result["DT"] = None
            return result

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

    D_proj_dual_cone = _diffcp.dprojection(v, cones_parsed, True)
    if mode == "dense":
        Q_dense = Q.todense()
        M = _diffcp.M_dense(Q_dense, cones_parsed, u, v, w)
        MT = M.T
    else:
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
        else:
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
        else:
            r = _diffcp.lsqr(MT, dz).solution

        values = pi_z[cols] * r[rows + n] - pi_z[n + rows] * r[cols]
        dA = sparse.csc_matrix((values, (rows, cols)), shape=A.shape)
        db = pi_z[n:n + m] * r[-1] - pi_z[-1] * r[n:n + m]
        dc = pi_z[:n] * r[-1] - pi_z[-1] * r[:n]

        return dA, db, dc

    result["D"] = derivative
    result["DT"] = adjoint_derivative
    return result
