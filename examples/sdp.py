import cvxpy as cp
import diffcp
import numpy as np

import time


def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


def randn_symm(n):
    A = np.random.randn(n, n)
    return (A + A.T) / 2


def randn_psd(n):
    A = 1. / 10 * np.random.randn(n, n)
    return A@A.T


def main(n=3, p=3):
    # Generate problem data
    C = randn_psd(n)
    As = [randn_symm(n) for _ in range(p)]
    Bs = np.random.randn(p)

    # Extract problem data using cvxpy
    X = cp.Variable((n, n), PSD=True)
    objective = cp.trace(C@X)
    constraints = [cp.trace(As[i]@X) == Bs[i] for i in range(p)]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob)

    # Print problem size
    mn_plus_m_plus_n = A.size + b.size + c.size
    n_plus_2n = c.size + 2 * b.size
    entries_in_derivative = mn_plus_m_plus_n * n_plus_2n
    print(f"""n={n}, p={p}, A.shape={A.shape}, nnz in A={A.nnz}, derivative={mn_plus_m_plus_n}x{n_plus_2n} ({entries_in_derivative} entries)""")

    # Compute solution and derivative maps
    start = time.perf_counter()
    x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
        A, b, c, cone_dims, eps=1e-5)
    end = time.perf_counter()
    print("Compute solution and set up derivative: %.2f s." % (end - start))

    # Derivative
    lsqr_args = dict(atol=1e-5, btol=1e-5)
    start = time.perf_counter()
    dA, db, dc = adjoint_derivative(diffcp.cones.vec_symm(
        C), np.zeros(y.size), np.zeros(s.size), **lsqr_args)
    end = time.perf_counter()
    print("Evaluate derivative: %.2f s." % (end - start))

    # Adjoint of derivative
    start = time.perf_counter()
    dx, dy, ds = derivative(A, b, c, **lsqr_args)
    end = time.perf_counter()
    print("Evaluate adjoint of derivative: %.2f s." % (end - start))

if __name__ == '__main__':
    np.random.seed(0)
    main(50, 25)
