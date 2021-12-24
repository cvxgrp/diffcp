import numpy as np

import diffcp.cone_program as cone_prog
import diffcp.utils as utils


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

        x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, eps=1e-10, mode=mode, solve_method="SCS")

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
