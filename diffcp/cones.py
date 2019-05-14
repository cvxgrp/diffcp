import numpy as np
import _proj as proj_lib
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

ZERO = "f"
POS = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"
POWER = "p"

# The ordering of CONES matches SCS.
CONES = [ZERO, POS, SOC, PSD, EXP, EXP_DUAL, POWER]


def parse_cone_dict(cone_dict):
    """Parses SCS-style cone dictionary."""
    return [(cone, cone_dict[cone]) for cone in CONES if cone in cone_dict]


def as_block_diag_linear_operator(matrices):
    """Block diag of SciPy sparse matrices (or linear operators)."""
    linear_operators = [splinalg.aslinearoperator(
        op) if not isinstance(op, splinalg.LinearOperator) else op
        for op in matrices]
    num_operators = len(linear_operators)
    nrows = [op.shape[0] for op in linear_operators]
    ncols = [op.shape[1] for op in linear_operators]
    m, n = sum(nrows), sum(ncols)
    row_indices = np.append(0, np.cumsum(nrows))
    col_indices = np.append(0, np.cumsum(ncols))

    def matvec(x):
        output = np.zeros(m)
        for i, op in enumerate(linear_operators):
            z = x[col_indices[i]:col_indices[i + 1]].ravel()
            output[row_indices[i]:row_indices[i + 1]] = op.matvec(z)
        return output

    def rmatvec(y):
        output = np.zeros(n)
        for i, op in enumerate(linear_operators):
            z = y[row_indices[i]:row_indices[i + 1]].ravel()
            output[col_indices[i]:col_indices[i + 1]] = op.rmatvec(z)
        return output

    return splinalg.LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec)


def transpose_linear_operator(op):
    return splinalg.LinearOperator(reversed(op.shape), matvec=op.rmatvec,
                                   rmatvec=op.matvec)


def vec_psd_dim(dim):
    return int(dim * (dim + 1) / 2)


def psd_dim(x):
    return int(np.sqrt(2 * x.size))


def in_exp(x):
    return (x[0] <= 0 and np.isclose(x[1], 0) and x[2] >= 0) or (x[1] > 0 and
                                                                 x[1] * np.exp(x[0] / x[1]) <= x[2])


def in_exp_dual(x):
    return (np.isclose(x[0], 0) and x[1] >= 0 and x[2] >= 0) or (
        x[0] < 0 and -x[0] * np.exp(x[1] / x[0]) <= np.e * x[2])


def unvec_symm(x, dim):
    """Returns a dim-by-dim symmetric matrix corresponding to `x`.

    `x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
    matrix; the correspondence is as in SCS.
    X = [ X11 X12 ... X1k
          X21 X22 ... X2k
          ...
          Xk1 Xk2 ... Xkk ],
    where
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """
    X = np.zeros((dim, dim))
    # triu_indices gets indices of upper triangular matrix in row-major order
    col_idx, row_idx = np.triu_indices(dim)
    X[(row_idx, col_idx)] = x
    X = X + X.T
    X /= np.sqrt(2)
    X[np.diag_indices(dim)] = np.diagonal(X) * np.sqrt(2) / 2
    return X


def vec_symm(X):
    """Returns a vectorized representation of a symmetric matrix `X`.

    Vectorization (including scaling) as per SCS.
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """
    X = X.copy()
    X *= np.sqrt(2)
    X[np.diag_indices(X.shape[0])] = np.diagonal(X) / np.sqrt(2)
    col_idx, row_idx = np.triu_indices(X.shape[0])
    return X[(row_idx, col_idx)]


def _proj(x, cone, dual=False):
    """Returns the projection of x onto a cone or its dual cone."""
    if cone == ZERO:
        return x if dual else np.zeros(x.shape)
    elif cone == POS:
        return np.maximum(x, 0)
    elif cone == SOC:
        t = x[0]
        z = x[1:]
        norm_z = np.linalg.norm(z, 2)
        if norm_z <= t or np.isclose(norm_z, t, atol=1e-8):
            return x
        elif norm_z <= -t:
            return np.zeros(x.shape)
        else:
            return 0.5 * (1 + t / norm_z) * np.append(norm_z, z)
    elif cone == PSD:
        dim = psd_dim(x)
        X = unvec_symm(x, dim)
        lambd, Q = np.linalg.eig(X)
        return vec_symm(Q @ sparse.diags(np.maximum(lambd, 0)) @ Q.T)
    elif cone == EXP:
        num_cones = int(x.size / 3)
        out = np.zeros(x.size)
        offset = 0
        for _ in range(num_cones):
            x_i = x[offset:offset + 3]
            r, s, t, _ = proj_lib.proj_exp_cone(
                float(x_i[0]), float(x_i[1]), float(x_i[2]))
            out[offset:offset + 3] = np.array([r, s, t])
            offset += 3
        # via Moreau
        return x - out if dual else out
    else:
        raise NotImplementedError(f"{cone} not implemented")


def _dproj(x, cone, dual=False):
    """Returns the derivative of projecting onto a cone (or its dual cone) at x.

    The derivative is represented as either a sparse matrix or linear operator.
    """
    shape = (x.size, x.size)
    if cone == ZERO:
        return sparse.eye(*shape) if dual else sparse.csc_matrix(shape)
    elif cone == POS:
        return sparse.diags(.5 * (np.sign(x) + 1), format="csc")
    elif cone == SOC:
        t = x[0]
        z = x[1:]
        norm_z = np.linalg.norm(z, 2)
        if norm_z <= t:
            return sparse.eye(*shape)
        elif norm_z <= -t:
            return sparse.csc_matrix(shape)
        else:
            z = z.reshape(z.size)
            unit_z = z / norm_z
            scale_factor = 1.0 / (2 * norm_z)
            t_plus_norm_z = t + norm_z

            def matvec(y):
                t_in = y[0]
                z_in = y[1:]
                first = norm_z * t_in + np.dot(z, z_in)
                rest = z * t_in + t_plus_norm_z * z_in - \
                    t * unit_z * np.dot(unit_z, z_in)
                return scale_factor * np.append(first, rest)

            # derivative is symmetric
            return splinalg.LinearOperator(shape, matvec=matvec,
                                           rmatvec=matvec)
    elif cone == PSD:
        dim = psd_dim(x)
        X = unvec_symm(x, dim)
        lambd, Q = np.linalg.eig(X)
        if np.all(lambd >= 0):
            matvec = lambda y: y
            return splinalg.LinearOperator(shape, matvec=matvec, rmatvec=matvec)

        # Sort eigenvalues, eigenvectors in ascending order, so that
        # we can obtain the index k such that lambd[k-1] < 0 < lambd[k]
        idx = lambd.argsort()
        lambd = lambd[idx]
        Q = Q[:, idx]
        k = np.searchsorted(lambd, 0)

        B = np.zeros((dim, dim))
        pos_gt_k = np.outer(np.maximum(lambd, 0)[k:], np.ones(k))
        neg_lt_k = np.outer(np.ones(dim - k), np.minimum(lambd, 0)[:k])
        B[k:, :k] = pos_gt_k / (neg_lt_k + pos_gt_k)
        B[:k, k:] = B[k:, :k].T
        B[k:, k:] = 1
        matvec = lambda y: vec_symm(
            Q @ (B * (Q.T @ unvec_symm(y, dim) @ Q)) @ Q.T)
        return splinalg.LinearOperator(shape, matvec=matvec, rmatvec=matvec)
    elif cone == EXP:
        num_cones = int(x.size / 3)
        ops = []
        offset = 0
        for _ in range(num_cones):
            x_i = x[offset:offset + 3]
            offset += 3
            if in_exp(x_i):
                ops.append(splinalg.aslinearoperator(sparse.eye(3)))
            elif in_exp_dual(-x_i):
                ops.append(splinalg.aslinearoperator(
                    sparse.csc_matrix((3, 3))))
            elif x_i[0] < 0 and x_i[1] and not np.isclose(x_i[2], 0):
                matvec = lambda y: np.array([
                    y[0], 0, y[2] * 0.5 * (1 + np.sign(x_i[2]))])
                ops.append(splinalg.LinearOperator((3, 3), matvec=matvec,
                                                   rmatvec=matvec))
            else:
                # TODO(akshayka): Cache projection if this is a bottleneck
                # TODO(akshayka): y_st is sometimes zero ...
                x_st, y_st, _, mu = proj_lib.proj_exp_cone(x_i[0], x_i[1],
                                                           x_i[2])
                if np.equal(y_st, 0):
                    y_st = np.abs(x_st)
                exp_x_y = np.exp(x_st / y_st)
                mu_exp_x_y = mu * exp_x_y
                x_mu_exp_x_y = x_st * mu_exp_x_y
                M = np.zeros((4, 4))
                M[:, 0] = np.array([
                    1 + mu_exp_x_y / y_st, -x_mu_exp_x_y / (y_st ** 2),
                    0,
                    exp_x_y])
                M[:, 1] = np.array([
                    -x_mu_exp_x_y / (y_st ** 2),
                    1 + x_st * x_mu_exp_x_y / (y_st ** 3),
                    0, exp_x_y - x_st * exp_x_y / y_st])
                M[:, 2] = np.array([0, 0, 1, -1])
                M[:, 3] = np.array([
                    exp_x_y, exp_x_y - x_st * exp_x_y / y_st, -1, 0])
                ops.append(splinalg.aslinearoperator(np.linalg.inv(M)[:3, :3]))
        D = as_block_diag_linear_operator(ops)
        if dual:
            return splinalg.LinearOperator((x.size, x.size),
                                           matvec=lambda v: v - D.matvec(v),
                                           rmatvec=lambda v: v - D.rmatvec(v))
        else:
            return D
    else:
        raise NotImplementedError(f"{cone} not implemented")


def pi(x, cones, dual=False):
    """Projects x onto product of cones (or their duals)

    Args:
        x: NumPy array (with PSD data formatted in SCS convention)
        cones: list of (cone name, size)
        dual: whether to project onto the dual cone

    Returns:
        NumPy array that is the projection of `x` onto the (dual) cones
    """
    projection = np.zeros(x.shape)
    offset = 0
    for cone, sz in cones:
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for dim in sz:
            if cone == PSD:
                dim = vec_psd_dim(dim)
            elif cone == EXP:
                dim *= 3
            projection[offset:offset + dim] = _proj(
                x[offset:offset + dim], cone, dual=dual)
            offset += dim
    return projection


def dpi(x, cones, dual=False):
    """Derivative of projection onto product of cones (or their duals), at x

    Args:
        x: NumPy array
        cones: list of (cone name, size)
        dual: whether to project onto the dual cone

    Returns:
        An abstract linear map representing the derivative, with methods
        `matvec` and `rmatvec`
    """
    dprojections = []
    offset = 0
    for cone, sz in cones:
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for dim in sz:
            if cone == PSD:
                dim = vec_psd_dim(dim)
            elif cone == EXP:
                dim *= 3
            dprojections.append(
                _dproj(x[offset:offset + dim], cone, dual=dual))
            offset += dim
    return as_block_diag_linear_operator(dprojections)
