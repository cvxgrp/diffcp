import diffcp.cones as cone_lib
import numpy as np
from scipy import sparse



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
