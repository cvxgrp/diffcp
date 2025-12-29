import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

import diffcp.cone_program as cone_prog
import diffcp.cones as cone_lib
import diffcp.utils as utils


def test_ecos_solve():
    np.random.seed(0)
    m = 20
    n = 10

    A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
    cone_dims.pop("q")
    cone_dims.pop("s")
    cone_dims.pop("ep")
    x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative(
        A, b, c, cone_dims, solve_method="ECOS")

    # check optimality conditions
    np.testing.assert_allclose(A @ x + s, b, atol=1e-8)
    np.testing.assert_allclose(A.T @ y + c, 0, atol=1e-8)
    np.testing.assert_allclose(s @ y, 0, atol=1e-8)
    np.testing.assert_allclose(s, cone_lib.pi(
        s, cone_lib.parse_cone_dict(cone_dims), dual=False), atol=1e-8)
    np.testing.assert_allclose(y, cone_lib.pi(
        y, cone_lib.parse_cone_dict(cone_dims), dual=True), atol=1e-8)

    x = cp.Variable(10)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(np.random.randn(5, 10) @ x) + np.random.randn(10) @ x),
                      [cp.norm2(x) <= 1, np.random.randn(2, 10) @ x == np.random.randn(2)])
    A, b, c, cone_dims = utils.scs_data_from_cvxpy_problem(prob)
    x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative(
        A, b, c, cone_dims, solve_method="ECOS")

    # check optimality conditions
    np.testing.assert_allclose(A @ x + s, b, atol=1e-8)
    np.testing.assert_allclose(A.T @ y + c, 0, atol=1e-8)
    np.testing.assert_allclose(s @ y, 0, atol=1e-8)
    np.testing.assert_allclose(s, cone_lib.pi(
        s, cone_lib.parse_cone_dict(cone_dims), dual=False), atol=1e-8)
    np.testing.assert_allclose(y, cone_lib.pi(
        y, cone_lib.parse_cone_dict(cone_dims), dual=True), atol=1e-8)


def test_infeasible():
    np.random.seed(0)
    c = np.ones(1)
    b = np.array([1.0, -1.0])
    A = sparse.csc_matrix(np.ones((2, 1)))
    cone_dims = {cone_lib.EQ_DIM: 2}
    with pytest.raises(cone_prog.SolverError, match=r"Solver ecos returned status Infeasible"):
        cone_prog.solve_and_derivative(A, b, c, cone_dims, solve_method="ECOS")


def test_expcone():
    np.random.seed(0)
    n = 10
    y = cp.Variable(n)
    obj = cp.Minimize(- cp.sum(cp.entr(y)))
    const = [cp.sum(y) == 1]
    prob = cp.Problem(obj, const)
    A, b, c, cone_dims = utils.scs_data_from_cvxpy_problem(prob)
    for mode in ["lsqr", "lsmr", "dense"]:
        x, y, s, D, DT = cone_prog.solve_and_derivative(A,
                                                        b,
                                                        c,
                                                        cone_dims,
                                                        solve_method="ECOS",
                                                        mode=mode,
                                                        feastol=1e-10,
                                                        abstol=1e-10,
                                                        reltol=1e-10)
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n))
        db = np.random.normal(0, 1e-6, size=b.size)
        dc = np.random.normal(0, 1e-6, size=c.size)
        dx, dy, ds = D(dA, db, dc)
        x_pert, y_pert, s_pert, _, _ = cone_prog.solve_and_derivative(A + dA,
                                                                      b + db,
                                                                      c + dc,
                                                                      cone_dims,
                                                                      solve_method="ECOS",
                                                                      mode=mode,
                                                                      feastol=1e-10,
                                                                      abstol=1e-10,
                                                                      reltol=1e-10)
        np.testing.assert_allclose(x_pert - x, dx, atol=1e-8)
        np.testing.assert_allclose(y_pert - y, dy, atol=1e-8)
        np.testing.assert_allclose(s_pert - s, ds, atol=1e-8)
