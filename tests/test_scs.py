import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse
from diffcp import ZERO

import diffcp.cone_program as cone_prog
import diffcp.utils as utils


@pytest.mark.parametrize("mode", ["lsqr", "lsmr"])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("limit", ["tolerance", "iterations"])
def test_derivative_solver_controls(mode, adjoint, limit):
    rng = np.random.default_rng(0)
    n = 32
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(n, n)))
    matrix = left @ np.diag(np.geomspace(1, 1e-4, n)) @ right.T
    A = sparse.csc_matrix(matrix)
    zero = np.zeros(n)
    _, _, _, D, DT = cone_prog.solve_and_derivative(
        A, zero, zero, {ZERO: n}, mode=mode, eps=1e-12)
    # The equality constraints uniquely determine x, with derivative solve(A, db).
    direction = rng.normal(size=n)
    if adjoint:
        reference = np.linalg.solve(matrix.T, direction)

        def apply(**options):
            return DT(direction, zero, zero, **options)[1]
    else:
        reference = np.linalg.solve(matrix, direction)

        def apply(**options):
            return D(sparse.csc_matrix(A.shape), direction, zero, **options)[0]

    iteration_key = "iter_lim" if mode == "lsqr" else "maxiter"
    accurate = apply(atol=1e-14, btol=1e-14, conlim=0, **{iteration_key: 1000})
    np.testing.assert_allclose(accurate, reference, atol=1e-7, rtol=1e-8)
    if limit == "tolerance":
        limited = apply(atol=1e-2, btol=1e-2)
    else:
        limited = apply(**{iteration_key: 1})
    assert np.linalg.norm(limited - reference) > 0.1 * np.linalg.norm(reference)


@pytest.mark.parametrize("mode", ["lsqr", "lsmr"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_derivative_solver_controls_batch(mode, n_jobs):
    matrix = np.diag([1., 0.1, 0.01])
    A = sparse.csc_matrix(matrix)
    zero = np.zeros(3)
    direction = np.ones(3)
    _, _, _, D, DT = cone_prog.solve_and_derivative_batch(
        [A, A], [zero, zero], [zero, zero], [{ZERO: 3}] * 2,
        n_jobs_forward=n_jobs, n_jobs_backward=n_jobs, mode=mode, eps=1e-12)
    dAs = [sparse.csc_matrix(A.shape)] * 2
    reference = np.linalg.solve(matrix, direction)
    options = dict(atol=1e-14, btol=1e-14)
    dxs, _, _ = D(dAs, [direction] * 2, [zero] * 2, **options)
    _, dbs, _ = DT([direction] * 2, [zero] * 2, [zero] * 2, **options)
    iteration_key = "iter_lim" if mode == "lsqr" else "maxiter"
    limited_dxs, _, _ = D(dAs, [direction] * 2, [zero] * 2, **{iteration_key: 1})
    _, limited_dbs, _ = DT([direction] * 2, [zero] * 2, [zero] * 2, **{iteration_key: 1})
    for actual, limited in zip(dxs + dbs, limited_dxs + limited_dbs):
        np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=1e-8)
        assert np.linalg.norm(limited - reference) > 0.1 * np.linalg.norm(reference)



def test_solve_and_derivative():
    np.random.seed(0)
    m = 20
    n = 10

    A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
    for mode in ["lsqr", "dense"]:
        x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, eps=1e-10, mode=mode, solve_method="SCS")

        dA = utils.get_random_like(
            A, lambda n: np.random.normal(0, 1e-6, size=n))
        db = np.random.normal(0, 1e-6, size=b.size)
        dc = np.random.normal(0, 1e-6, size=c.size)

        dx, dy, ds = derivative(dA, db, dc)

        x_pert, y_pert, s_pert, _, _ = cone_prog.solve_and_derivative(
            A + dA, b + db, c + dc, cone_dims, eps=1e-10, solve_method="SCS")

        np.testing.assert_allclose(x_pert - x, dx, atol=1e-8)
        np.testing.assert_allclose(y_pert - y, dy, atol=1e-8)
        np.testing.assert_allclose(s_pert - s, ds, atol=1e-8)

        objective = c.T @ x
        dA, db, dc = adjoint_derivative(
            c, np.zeros(y.size), np.zeros(s.size))

        x_pert, _, _, _, _ = cone_prog.solve_and_derivative(
            A + 1e-6 * dA, b + 1e-6 * db, c + 1e-6 * dc, cone_dims, eps=1e-10, solve_method="SCS")
        objective_pert = c.T @ x_pert

        np.testing.assert_allclose(
            objective_pert - objective,
            1e-6 * dA.multiply(dA).sum() + 1e-6 * db @ db + 1e-6 * dc @ dc, atol=1e-8)


def test_warm_start():
    np.random.seed(0)
    m = 20
    n = 10
    A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
    x, y, s, _, _ = cone_prog.solve_and_derivative(
        A, b, c, cone_dims, eps=1e-9, solve_method="SCS")
    x_p, y_p, s_p, _, _ = cone_prog.solve_and_derivative(
        A, b, c, cone_dims, warm_start=(x, y, s), max_iters=1, solve_method="SCS", eps=1e-9)

    np.testing.assert_allclose(x, x_p, atol=1e-7)
    np.testing.assert_allclose(y, y_p, atol=1e-7)
    np.testing.assert_allclose(s, s_p, atol=1e-7)


def test_threading():
    np.random.seed(0)
    test_rtol = 1e-3
    test_atol = 1e-8
    m = 20
    n = 10
    As, bs, cs, cone_dicts = [], [], [], []
    results = []

    for _ in range(50):
        A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
        As += [A]
        bs += [b]
        cs += [c]
        cone_dicts += [cone_dims]
        results.append(cone_prog.solve_and_derivative(A, b, c, cone_dims))

    for n_jobs in [1, -1]:
        xs, ys, ss, _, DT_batch = cone_prog.solve_and_derivative_batch(
            As, bs, cs, cone_dicts, n_jobs_forward=n_jobs, n_jobs_backward=n_jobs)

        for i in range(50):
            np.testing.assert_allclose(results[i][0], xs[i], rtol=test_rtol, atol=test_atol)
            np.testing.assert_allclose(results[i][1], ys[i], rtol=test_rtol, atol=test_atol)
            np.testing.assert_allclose(results[i][2], ss[i], rtol=test_rtol, atol=test_atol)

        dAs, dbs, dcs = DT_batch(xs, ys, ss)
        for i in range(50):
            dA, db, dc = results[
                i][-1](results[i][0], results[i][1], results[i][2])
            np.testing.assert_allclose(dA.todense(), dAs[i].todense(), rtol=test_rtol, atol=test_atol)
            np.testing.assert_allclose(dbs[i], db, rtol=test_rtol, atol=test_atol)
            np.testing.assert_allclose(dcs[i], dc, rtol=test_rtol, atol=test_atol)


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
                                                        solve_method="SCS",
                                                        mode=mode,
                                                        eps=1e-10)
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n))
        db = np.random.normal(0, 1e-6, size=b.size)
        dc = np.random.normal(0, 1e-6, size=c.size)
        dx, dy, ds = D(dA, db, dc)
        x_pert, y_pert, s_pert, _, _ = cone_prog.solve_and_derivative(A + dA,
                                                                      b + db,
                                                                      c + dc,
                                                                      cone_dims,
                                                                      solve_method="SCS",
                                                                      mode=mode,
                                                                      eps=1e-10)
        
        np.testing.assert_allclose(x_pert - x, dx, atol=1e-8)
        np.testing.assert_allclose(y_pert - y, dy, atol=1e-8)
        np.testing.assert_allclose(s_pert - s, ds, atol=1e-8)
