import unittest

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import diffcp.cone_program as cone_prog
import diffcp.cones as cone_lib
import diffcp.utils as utils
import _diffcp
from _diffcp import Cone, ConeType
import cvxpy as cp


CPP_CONES_TO_SCS = {
    ConeType.ZERO: "f",
    ConeType.POS: "l",
    ConeType.SOC: "q",
    ConeType.PSD: "s",
    ConeType.EXP: "ep",
    ConeType.EXP_DUAL: "ed"
}


class TestConeProgDiff(unittest.TestCase):

    def test_vec_psd_dim(self):
        self.assertEqual(cone_lib.vec_psd_dim(10), (10) * (10 + 1) / 2)

    def test_psd_dim(self):
        n = 4096
        self.assertEqual(cone_lib.psd_dim(cone_lib.vec_psd_dim(n)), n)

    def test_in_exp(self):
        self.assertTrue(_diffcp.in_exp(np.array([0, 0, 1])))
        self.assertTrue(_diffcp.in_exp(np.array([-1, 0, 0])))
        self.assertTrue(_diffcp.in_exp(np.array([1, 1, 5])))
        self.assertFalse(_diffcp.in_exp(np.array([1, 0, 0])))
        self.assertFalse(_diffcp.in_exp(np.array([-1, -1, 1])))
        self.assertFalse(_diffcp.in_exp(np.array([-1, 0, -1])))

    def test_in_exp_dual(self):
        self.assertTrue(_diffcp.in_exp_dual(np.array([0, 1, 1])))
        self.assertTrue(_diffcp.in_exp_dual(np.array([-1, 1, 5])))
        self.assertFalse(_diffcp.in_exp_dual(np.array([0, -1, 1])))
        self.assertFalse(_diffcp.in_exp_dual(np.array([0, 1, -1])))

    def test_unvec_symm(self):
        np.random.seed(0)
        n = 5
        x = np.random.randn(n, n)
        x = x + x.T
        np.testing.assert_allclose(
            cone_lib.unvec_symm(cone_lib.vec_symm(x), n), x)

    def test_vec_symm(self):
        np.random.seed(0)
        n = 5
        x = np.random.randn(cone_lib.vec_psd_dim(n))
        np.testing.assert_allclose(
            cone_lib.vec_symm(cone_lib.unvec_symm(x, n)), x)

    def test_proj_zero(self):
        np.random.seed(0)
        n = 100
        for _ in range(10):
            x = np.random.randn(n)
            np.testing.assert_allclose(
                cone_lib._proj(x, cone_lib.ZERO, dual=True), x)
            np.testing.assert_allclose(
                cone_lib._proj(x, cone_lib.ZERO, dual=False),
                np.zeros(n))

    def test_proj_pos(self):
        np.random.seed(0)
        n = 100
        for _ in range(15):
            x = np.random.randn(n)
            p = cone_lib._proj(x, cone_lib.POS, dual=False)
            np.testing.assert_allclose(p, np.maximum(x, 0))
            np.testing.assert_allclose(
                p, cone_lib._proj(x, cone_lib.POS, dual=True))

    def test_proj_soc(self):
        np.random.seed(0)
        n = 100
        for _ in range(15):
            x = np.random.randn(n)
            z = cp.Variable(n)
            objective = cp.Minimize(cp.sum_squares(z - x))
            constraints = [cp.norm(z[1:], 2) <= z[0]]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver="SCS", eps=1e-10)
            p = cone_lib._proj(x, cone_lib.SOC, dual=False)
            np.testing.assert_allclose(
                p, np.array(z.value))
            np.testing.assert_allclose(
                p, cone_lib._proj(x, cone_lib.SOC, dual=True))

    def test_proj_psd(self):
        np.random.seed(0)
        n = 10
        for _ in range(15):
            x = np.random.randn(n, n)
            x = x + x.T
            x_vec = cone_lib.vec_symm(x)
            z = cp.Variable((n, n), PSD=True)
            objective = cp.Minimize(cp.sum_squares(z - x))
            prob = cp.Problem(objective)
            prob.solve(solver="SCS", eps=1e-10)
            p = cone_lib.unvec_symm(
                cone_lib._proj(x_vec, cone_lib.PSD, dual=False), n)
            np.testing.assert_allclose(p, z.value, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(p, cone_lib.unvec_symm(
                cone_lib._proj(x_vec, cone_lib.PSD, dual=True), n))

    def test_proj_exp(self):
        np.random.seed(0)
        for _ in range(15):
            x = np.random.randn(9)
            var = cp.Variable(9)
            constr = [cp.constraints.ExpCone(var[0], var[1], var[2])]
            constr.append(cp.constraints.ExpCone(var[3], var[4], var[5]))
            constr.append(cp.constraints.ExpCone(var[6], var[7], var[8]))
            obj = cp.Minimize(cp.norm(var[0:3] - x[0:3]) +
                              cp.norm(var[3:6] - x[3:6]) +
                              cp.norm(var[6:9] - x[6:9]))
            prob = cp.Problem(obj, constr)
            prob.solve(solver="SCS", eps=1e-12)
            p = cone_lib._proj(x, cone_lib.EXP, dual=False)
            np.testing.assert_allclose(p, var.value, atol=1e-6)
            # x + Pi_{exp}(-x) = Pi_{exp_dual}(x)
            p_dual = cone_lib._proj(x, cone_lib.EXP_DUAL, dual=False)
            var = cp.Variable(9)
            constr = [cp.constraints.ExpCone(var[0], var[1], var[2])]
            constr.append(cp.constraints.ExpCone(var[3], var[4], var[5]))
            constr.append(cp.constraints.ExpCone(var[6], var[7], var[8]))
            obj = cp.Minimize(cp.norm(var[0:3] + x[0:3]) +
                              cp.norm(var[3:6] + x[3:6]) +
                              cp.norm(var[6:9] + x[6:9]))
            prob = cp.Problem(obj, constr)
            prob.solve(solver="SCS", eps=1e-12)
            np.testing.assert_allclose(
                p_dual, x + var.value, atol=1e-6)

    def _test_dproj(self, cone, dual, n, x=None, tol=1e-8):
        if x is None:
            x = np.random.randn(n)
        dx = 1e-6 * np.random.randn(n)
        proj_x = cone_lib._proj(x, CPP_CONES_TO_SCS[cone.type], dual)
        z = cone_lib._proj(x + dx, CPP_CONES_TO_SCS[cone.type], dual)

        Dpi = _diffcp.dprojection(x, [cone], dual)
        np.testing.assert_allclose(Dpi.matvec(dx), z - proj_x, atol=tol)

        Dpi_dense = _diffcp.dprojection_dense(x, [cone], dual)
        np.testing.assert_allclose(Dpi_dense @ dx, z - proj_x, atol=tol)

        # assure that dense and linear operator are the same.
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            np.testing.assert_allclose(Dpi.matvec(ei), Dpi_dense[:, i])

    def test_dproj_zero(self):
        np.random.seed(0)
        for _ in range(10):
            self._test_dproj(Cone(ConeType.ZERO, [55]), True, 55)
            self._test_dproj(Cone(ConeType.ZERO, [55]), False, 55)

    def test_dproj_pos(self):
        np.random.seed(0)
        for _ in range(10):
            self._test_dproj(Cone(ConeType.POS, [55]), True, 55)
            self._test_dproj(Cone(ConeType.POS, [55]), False, 55)

    def test_dproj_soc(self):
        np.random.seed(0)
        for _ in range(10):
            self._test_dproj(Cone(ConeType.SOC, [55]), True, 55)
            self._test_dproj(Cone(ConeType.SOC, [55]), False, 55)

    def test_dproj_psd(self):
        np.random.seed(0)
        for _ in range(10):
            # n=55 equals k * (k + 1) / 2
            self._test_dproj(
                Cone(ConeType.PSD, [cone_lib.psd_dim(55)]), True, 55)
            self._test_dproj(
                Cone(ConeType.PSD, [cone_lib.psd_dim(55)]), False, 55)

    def test_dproj_exp(self):
        np.random.seed(0)
        for _ in range(10):
            # dimension must be a multiple of 3
            self._test_dproj(Cone(ConeType.EXP, [18]), True, 54, tol=1e-5)
            self._test_dproj(Cone(ConeType.EXP, [18]), False, 54, tol=1e-5)

    def test_dproj_exp_dual(self):
        np.random.seed(0)
        for _ in range(10):
            # dimension must be a multiple of 3
            self._test_dproj(Cone(ConeType.EXP_DUAL, [18]), True, 54, tol=1e-5)
            self._test_dproj(
                Cone(ConeType.EXP_DUAL, [18]), False, 54, tol=1e-5)

    def test_pi(self):
        np.random.seed(0)
        for _ in range(10):
            zero_dim = np.random.randint(1, 10)
            pos_dim = np.random.randint(1, 10)
            soc_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            psd_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            exp_dim = np.random.randint(3, 18)
            cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                     (cone_lib.SOC, soc_dim), (cone_lib.PSD, psd_dim),
                     (cone_lib.EXP, exp_dim), (cone_lib.EXP_DUAL, exp_dim)]
            size = zero_dim + pos_dim + sum(soc_dim) + sum(
                [cone_lib.vec_psd_dim(d) for d in psd_dim]) + 2 * 3 * exp_dim
            x = np.random.randn(size)
            for dual in [False, True]:
                proj = cone_lib.pi(x, cones, dual=dual)

                offset = 0
                np.testing.assert_allclose(proj[:zero_dim],
                                           cone_lib._proj(x[:zero_dim], cone_lib.ZERO, dual=dual))
                offset += zero_dim

                np.testing.assert_allclose(proj[offset:offset + pos_dim],
                                           cone_lib._proj(x[offset:offset + pos_dim], cone_lib.POS,
                                                          dual=dual))
                offset += pos_dim

                for dim in soc_dim:
                    np.testing.assert_allclose(proj[offset:offset + dim],
                                               cone_lib._proj(x[offset:offset + dim], cone_lib.SOC,
                                                              dual=dual))
                    offset += dim

                for dim in psd_dim:
                    dim = cone_lib.vec_psd_dim(dim)
                    np.testing.assert_allclose(proj[offset:offset + dim],
                                               cone_lib._proj(x[offset:offset + dim], cone_lib.PSD,
                                                              dual=dual))
                    offset += dim

                dim = 3 * exp_dim
                np.testing.assert_allclose(proj[offset:offset + dim],
                                           cone_lib._proj(x[offset:offset + dim], cone_lib.EXP, dual=dual))
                offset += dim

                np.testing.assert_allclose(proj[offset:],
                                           cone_lib._proj(x[offset:], cone_lib.EXP_DUAL, dual=dual))

    def test_dpi(self):
        np.random.seed(0)
        for _ in range(10):
            zero_dim = np.random.randint(1, 10)
            pos_dim = np.random.randint(1, 10)
            soc_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            psd_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            exp_dim = np.random.randint(3, 18)
            cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                     (cone_lib.SOC, soc_dim), (cone_lib.PSD, psd_dim),
                     (cone_lib.EXP, exp_dim), (cone_lib.EXP_DUAL, exp_dim)]
            size = zero_dim + pos_dim + sum(soc_dim) + sum(
                [cone_lib.vec_psd_dim(d) for d in psd_dim]) + 2 * 3 * exp_dim
            x = np.random.randn(size)

            for dual in [False, True]:
                cone_list_cpp = cone_lib.parse_cone_dict_cpp(cones)
                proj_x = cone_lib.pi(x, cones, dual=dual)
                dx = 1e-7 * np.random.randn(size)
                z = cone_lib.pi(x + dx, cones, dual=dual)

                Dpi = _diffcp.dprojection(x, cone_list_cpp, dual)
                np.testing.assert_allclose(
                    Dpi.matvec(dx), z - proj_x, atol=1e-6)

                Dpi = _diffcp.dprojection_dense(x, cone_list_cpp, dual)
                np.testing.assert_allclose(Dpi @ dx, z - proj_x, atol=1e-6)

    def test_get_random_like(self):
        np.random.seed(0)
        A = sparse.eye(5)
        B = utils.get_random_like(A,
                                  lambda n: np.random.normal(0, 1e-6, size=n))
        A_r, A_c = [list(x) for x in A.nonzero()]
        B_r, B_c = [list(x) for x in B.nonzero()]
        self.assertListEqual(A_r, B_r)
        self.assertListEqual(A_c, B_c)

    def test_infeasible(self):
        np.random.seed(0)
        c = np.ones(1)
        b = np.array([1.0, -1.0])
        A = sparse.csc_matrix(np.ones((2, 1)))
        cone_dims = {"f": 2}
        with self.assertRaises(cone_prog.SolverError, msg='Solver scs returned status.*'):
            cone_prog.solve_and_derivative(A, b, c, cone_dims)

    def test_lsqr(self):
        np.random.seed(0)
        A = np.random.randn(20, 10)
        b = np.random.randn(20)

        b_copy = b.copy()
        X = _diffcp.lsqr_sparse(sparse.csc_matrix(A), b)
        np.testing.assert_equal(b_copy, b)

        svx = np.linalg.lstsq(A, b, rcond=None)[0]
        xo = X.solution
        np.testing.assert_allclose(svx, xo, err_msg=(
            "istop: %d, itn: %d" % (X.istop, X.itn)))

    def test_get_nonzeros(self):
        np.random.seed(0)
        A = sparse.csc_matrix(np.random.randn(4, 3))
        self.assertEqual(A.nnz, 4 * 3)
        A[1, 1] = 0.0
        self.assertEqual(A.nnz, 4 * 3)
        import copy
        A_copy = copy.deepcopy(A)
        A.data[A.data == 0.0] = np.nan
        rows, cols = A.nonzero()
        self.assertEqual(rows.size, 4 * 3)
        self.assertEqual(cols.size, 4 * 3)
        A.data[np.isnan(A.data)] = 0.0
        np.testing.assert_equal(A.data, A_copy.data)
        self.assertEqual(A.nnz, 4 * 3)
        self.assertEqual(A.nonzero()[0].size, 4 * 3 - 1)
        A.eliminate_zeros()
        self.assertEqual(A.nnz, 4 * 3 - 1)


class TestSCS(unittest.TestCase):

    def test_solve_and_derivative(self):
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
                1e-6 * dA.multiply(dA).sum() + 1e-6 * db@db + 1e-6 * dc@dc, atol=1e-8)

    def test_warm_start(self):
        np.random.seed(0)
        m = 20
        n = 10
        A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
        x, y, s, _, _ = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, eps=1e-11, solve_method="SCS")
        x_p, y_p, s_p, _, _ = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, warm_start=(x, y, s), max_iters=1, solve_method="SCS")

        np.testing.assert_allclose(x, x_p, atol=1e-7)
        np.testing.assert_allclose(y, y_p, atol=1e-7)
        np.testing.assert_allclose(s, s_p, atol=1e-7)

    def test_threading(self):
        np.random.seed(0)
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
                np.testing.assert_allclose(results[i][0], xs[i])
                np.testing.assert_allclose(results[i][1], ys[i])
                np.testing.assert_allclose(results[i][2], ss[i])

            dAs, dbs, dcs = DT_batch(xs, ys, ss)
            for i in range(50):
                dA, db, dc = results[
                    i][-1](results[i][0], results[i][1], results[i][2])
                np.testing.assert_allclose(dA.todense(), dAs[i].todense())
                np.testing.assert_allclose(dbs[i], db)
                np.testing.assert_allclose(dcs[i], dc)


class TestECOS(unittest.TestCase):

    def test_ecos_solve(self):
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
        prob = cp.Problem(cp.Minimize(cp.sum_squares(np.random.randn(5, 10) @ x) + np.random.randn(10) @ x), [cp.norm2(x) <= 1, np.random.randn(2, 10) @ x == np.random.randn(2)])
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

    def test_infeasible(self):
        np.random.seed(0)
        c = np.ones(1)
        b = np.array([1.0, -1.0])
        A = sparse.csc_matrix(np.ones((2, 1)))
        cone_dims = {"f": 2}
        with self.assertRaises(cone_prog.SolverError, msg='Solver ecos returned status Infeasible'):
            cone_prog.solve_and_derivative(A, b, c, cone_dims, solve_method="ECOS")

if __name__ == '__main__':
    unittest.main()
