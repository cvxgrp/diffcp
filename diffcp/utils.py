import diffcp.cones as cone_lib
import numpy as np
from scipy import sparse
from copy import deepcopy


def scs_data_from_cvxpy_problem(problem):
    import cvxpy as cp
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


def least_squares_eq_scs_data(m, n, seed=0):
    """Generate a conic problem with unique solution."""
    import cvxpy as cp
    np.random.seed(seed)
    assert m >= n
    x = cp.Variable(n)
    b = np.random.randn(m)
    A = np.random.randn(m, n)
    assert np.linalg.matrix_rank(A) == n
    objective = cp.pnorm(A @ x - b, 1)
    constraints = [x >= 0, cp.sum(x) == 1.0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return scs_data_from_cvxpy_problem(problem)


def get_random_like(A, randomness):
    """Generate a random sparse matrix with the same sparsity
    pattern as A, using the function `randomness`.

    `randomness` is a function that returns a random vector
    with a prescribed length.
    """
    rows, cols = A.nonzero()
    values = randomness(A.nnz)
    return sparse.csc_matrix((values, (rows, cols)), shape=A.shape)


def regularize_P(P, rho, size):
    """Regularizes the matrix P by adding rho * I to it."""
    if rho > 0:
        reg = rho * sparse.eye(size, format='csc')
        if P is None:
            P_reg = reg
        else:
            P_reg = P + reg
    else:
        P_reg = P
    return P_reg

def embed_problem(A, b, c, P, cone_dict):   
    m = b.shape[0]         
    A_emb = sparse.bmat([
        [A, sparse.eye(m, format=A.format)],
        [None, -sparse.eye(m, format=A.format)]
    ]).tocsc()
    b_emb = np.hstack([b, np.zeros(m)])
    c_emb = np.hstack([c, np.zeros(m)])
    if P is not None:
        P_emb = sparse.bmat([
            [P, None],
            [None, np.zeros((m, m))]
        ]).tocsc()
    else:
        P_emb = None
    cone_dict_emb = deepcopy(cone_dict)
    if 'z' in cone_dict_emb:
        cone_dict_emb['z'] += m
    else:
        cone_dict_emb['z'] = m
    return A_emb, b_emb, c_emb, P_emb, cone_dict_emb
