import cvxpy as cp
import numpy as np

import diffcp.cone_program as cone_prog
from diffcp.cones import unvec_symm
import diffcp.utils as utils


def test_solve_and_derivative():
    np.random.seed(0)
    m = 20
    n = 10

    A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
    for mode in ["lsqr", "dense"]:
        x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, mode=mode, solve_method="Clarabel")

        dA = utils.get_random_like(
            A, lambda n: np.random.normal(0, 1e-6, size=n))
        db = np.random.normal(0, 1e-6, size=b.size)
        dc = np.random.normal(0, 1e-6, size=c.size)

        dx, dy, ds = derivative(dA, db, dc)

        x_pert, y_pert, s_pert, _, _ = cone_prog.solve_and_derivative(
            A + dA, b + db, c + dc, cone_dims, solve_method="Clarabel")

        np.testing.assert_allclose(x_pert - x, dx, atol=1e-8)
        np.testing.assert_allclose(y_pert - y, dy, atol=1e-8)
        np.testing.assert_allclose(s_pert - s, ds, atol=1e-8)

        objective = c.T @ x
        dA, db, dc = adjoint_derivative(
            c, np.zeros(y.size), np.zeros(s.size))

        x_pert, _, _, _, _ = cone_prog.solve_and_derivative(
            A + 1e-6 * dA, b + 1e-6 * db, c + 1e-6 * dc, cone_dims, solve_method="Clarabel")
        objective_pert = c.T @ x_pert

        np.testing.assert_allclose(
            objective_pert - objective,
            1e-6 * dA.multiply(dA).sum() + 1e-6 * db @ db + 1e-6 * dc @ dc, atol=1e-8)


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
        x, y, s, D, DT = cone_prog.solve_and_derivative(
            A,
            b,
            c,
            cone_dims,
            solve_method="Clarabel",
            mode=mode,
            tol_gap_abs=1e-13,
            tol_gap_rel=1e-13,
            tol_feas=1e-13,
            tol_ktratio=1e-13,
        )
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n))
        db = np.random.normal(0, 1e-6, size=b.size)
        dc = np.random.normal(0, 1e-6, size=c.size)
        dx, dy, ds = D(dA, db, dc)
        x_pert, y_pert, s_pert, _, _ = cone_prog.solve_and_derivative(
            A + dA,
            b + db,
            c + dc,
            cone_dims,
            solve_method="Clarabel",
            mode=mode,
            tol_gap_abs=1e-13,
            tol_gap_rel=1e-13,
            tol_feas=1e-13,
            tol_infeas_abs=1e-13,
            tol_infeas_rel=1e-13,
            tol_ktratio=1e-13,
        )

        np.testing.assert_allclose(x_pert - x, dx, atol=1e-8)
        np.testing.assert_allclose(y_pert - y, dy, atol=1e-8)
        np.testing.assert_allclose(s_pert - s, ds, atol=1e-8)

def test_psdcone():
    DIM = 5
    X = cp.Variable(shape=(DIM, DIM), PSD=True)
    C = np.zeros((DIM, DIM))
    C[0, 0] = 1
    C[4, 4] = -1
    objective = cp.Minimize(cp.trace(C @ X))
    constraint = cp.trace(X) == 1
    problem = cp.Problem(objective, [constraint])
    A, b, c, cone_dims = utils.scs_data_from_cvxpy_problem(problem)
    sol_vec, _, _, _, _ = cone_prog.solve_and_derivative(A, b, c, cone_dims, solve_method='Clarabel')

    sol = unvec_symm(sol_vec, DIM)

    assert np.abs(np.trace(sol) - 1.0) < 1e-6
    assert (np.linalg.eigvals(sol) >= -1e-6).all()


def test_solve_only_batch():
    """Test solve_only_batch with Ps=None (default)."""
    np.random.seed(0)
    m = 20
    n = 10
    batch_size = 5

    As, bs, cs, cone_dicts = [], [], [], []
    for _ in range(batch_size):
        A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
        As.append(A)
        bs.append(b)
        cs.append(c)
        cone_dicts.append(cone_dims)

    # Test serial path (n_jobs_forward=1) with Ps=None
    xs, ys, ss = cone_prog.solve_only_batch(
        As, bs, cs, cone_dicts, n_jobs_forward=1, solve_method='Clarabel')
    assert len(xs) == batch_size
    assert len(ys) == batch_size
    assert len(ss) == batch_size

    # Verify solutions satisfy optimality conditions
    for i in range(batch_size):
        np.testing.assert_allclose(As[i] @ xs[i] + ss[i], bs[i], atol=1e-7)

    # Test parallel path (n_jobs_forward=-1) with Ps=None
    xs_par, ys_par, ss_par = cone_prog.solve_only_batch(
        As, bs, cs, cone_dicts, n_jobs_forward=-1, solve_method='Clarabel')
    assert len(xs_par) == batch_size

    # Verify parallel solutions also satisfy optimality conditions
    for i in range(batch_size):
        np.testing.assert_allclose(As[i] @ xs_par[i] + ss_par[i], bs[i], atol=1e-7)


def test_derivative_batch_parallel():
    """Test that parallel D_batch works correctly."""
    np.random.seed(0)
    m = 20
    n = 10
    batch_size = 5

    As, bs, cs, cone_dicts = [], [], [], []
    for _ in range(batch_size):
        A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
        As.append(A)
        bs.append(b)
        cs.append(c)
        cone_dicts.append(cone_dims)

    # Solve with serial backward pass
    xs_ser, ys_ser, ss_ser, D_ser, DT_ser = cone_prog.solve_and_derivative_batch(
        As, bs, cs, cone_dicts, n_jobs_forward=1, n_jobs_backward=1, solve_method='Clarabel')

    # Solve with parallel backward pass
    xs_par, ys_par, ss_par, D_par, DT_par = cone_prog.solve_and_derivative_batch(
        As, bs, cs, cone_dicts, n_jobs_forward=-1, n_jobs_backward=-1, solve_method='Clarabel')

    # Create perturbations
    dAs = [utils.get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n)) for A in As]
    dbs = [np.random.normal(0, 1e-6, size=b.size) for b in bs]
    dcs = [np.random.normal(0, 1e-6, size=c.size) for c in cs]

    # Test D_batch (forward derivative)
    dxs_ser, dys_ser, dss_ser = D_ser(dAs, dbs, dcs)
    dxs_par, dys_par, dss_par = D_par(dAs, dbs, dcs)

    for i in range(batch_size):
        np.testing.assert_allclose(dxs_ser[i], dxs_par[i], rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(dys_ser[i], dys_par[i], rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(dss_ser[i], dss_par[i], rtol=1e-5, atol=1e-10)

    # Test DT_batch (adjoint derivative)
    dAs_ser, dbs_ser, dcs_ser = DT_ser(xs_ser, ys_ser, ss_ser)
    dAs_par, dbs_par, dcs_par = DT_par(xs_par, ys_par, ss_par)

    for i in range(batch_size):
        np.testing.assert_allclose(dAs_ser[i].todense(), dAs_par[i].todense(), rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(dbs_ser[i], dbs_par[i], rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(dcs_ser[i], dcs_par[i], rtol=1e-5, atol=1e-10)
