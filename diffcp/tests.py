import unittest
import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import diffcp.cone_program as cone_prog
import diffcp.cones as cone_lib
import diffcp.utils as utils


class TestConeProgDiff(unittest.TestCase):

    def linear_operator_same_as_matrix(self, linop, A):
        self.assertEqual(linop.shape, A.shape)
        linop_dense = linop.matmat(np.eye(linop.shape[1]))
        return np.allclose(linop_dense, A.todense())

    def test_as_block_diag_linear_operator(self):
        As = [np.random.randn(2, 2) for _ in range(10)]
        A = sparse.block_diag(As)
        A_linop = cone_lib.as_block_diag_linear_operator(As)
        self.assertTrue(self.linear_operator_same_as_matrix(A_linop, A))

    def test_transpose_linear_operator(self):
        A = sparse.random(100, 100)
        self.assertTrue(self.linear_operator_same_as_matrix(
            cone_lib.transpose_linear_operator(
                splinalg.aslinearoperator(A)), A.T))

    def test_vec_psd_dim(self):
        self.assertEqual(cone_lib.vec_psd_dim(10), (10) * (10 + 1) / 2)

    def test_psd_dim(self):
        n = 4096
        self.assertEqual(cone_lib.psd_dim(np.ones(cone_lib.vec_psd_dim(n))), n)

    def test_in_exp(self):
        self.assertTrue(cone_lib.in_exp(np.array([0, 0, 1])))
        self.assertTrue(cone_lib.in_exp(np.array([-1, 0, 0])))
        self.assertTrue(cone_lib.in_exp(np.array([1, 1, 5])))
        self.assertFalse(cone_lib.in_exp(np.array([1, 0, 0])))
        self.assertFalse(cone_lib.in_exp(np.array([-1, -1, 1])))
        self.assertFalse(cone_lib.in_exp(np.array([-1, 0, -1])))

    def test_in_exp_dual(self):
        self.assertTrue(cone_lib.in_exp_dual(np.array([0, 1, 1])))
        self.assertTrue(cone_lib.in_exp_dual(np.array([-1, 1, 5])))
        self.assertFalse(cone_lib.in_exp_dual(np.array([0, -1, 1])))
        self.assertFalse(cone_lib.in_exp_dual(np.array([0, 1, -1])))

    def test_unvec_symm(self):
        n = 5
        x = np.random.randn(n, n)
        x = x + x.T
        np.testing.assert_allclose(
            cone_lib.unvec_symm(cone_lib.vec_symm(x), n), x)

    def test_vec_symm(self):
        n = 5
        x = np.random.randn(cone_lib.vec_psd_dim(n))
        np.testing.assert_allclose(
            cone_lib.vec_symm(cone_lib.unvec_symm(x, n)), x)

    def test_proj_zero(self):
        n = 100
        for _ in range(10):
            x = np.random.randn(n)
            np.testing.assert_allclose(
                cone_lib._proj(x, cone_lib.ZERO, dual=True), x)
            np.testing.assert_allclose(
                cone_lib._proj(x, cone_lib.ZERO, dual=False),
                np.zeros(n))

    def test_proj_pos(self):
        n = 100
        for _ in range(15):
            x = np.random.randn(n)
            p = cone_lib._proj(x, cone_lib.POS, dual=False)
            np.testing.assert_allclose(p, np.maximum(x, 0))
            np.testing.assert_allclose(
                p, cone_lib._proj(x, cone_lib.POS, dual=True))

    def test_proj_soc(self):
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
            prob.solve(solver="SCS", eps=1e-10)
            p = cone_lib._proj(x, cone_lib.EXP, dual=False)
            np.testing.assert_allclose(p, var.value, atol=1e-5, rtol=1e-5)
            p_dual = cone_lib._proj(x, cone_lib.EXP, dual=True)
            # x - Pi_{exp}(x) = Pi_{exp_dual}(x)
            np.testing.assert_allclose(
                p_dual, x - var.value, atol=1e-5, rtol=1e-5)

    def _test_dproj(self, cone, dual, n, x=None):
        if x is None:
            x = np.random.randn(n)
        Dpi = cone_lib._dproj(x, cone, dual)
        dx = 1e-6 * np.random.randn(n)
        proj_x = cone_lib._proj(x, cone, dual)
        z = cone_lib._proj(x + dx, cone, dual)
        np.testing.assert_allclose(Dpi@dx, z - proj_x, atol=1e-3, rtol=1e-4)

    def test_dproj_zero(self):
        for _ in range(10):
            self._test_dproj(cone_lib.ZERO, True, 55)
            self._test_dproj(cone_lib.ZERO, False, 55)

    def test_dproj_pos(self):
        for _ in range(10):
            self._test_dproj(cone_lib.POS, True, 55)
            self._test_dproj(cone_lib.POS, False, 55)

    def test_dproj_soc(self):
        for _ in range(10):
            self._test_dproj(cone_lib.SOC, True, 55)
            self._test_dproj(cone_lib.SOC, False, 55)

    def test_dproj_psd(self):
        for _ in range(10):
            # n=55 equals k * (k + 1) / 2
            self._test_dproj(cone_lib.PSD, True, 55)
            self._test_dproj(cone_lib.PSD, False, 55)

    def test_dproj_exp(self):
        for _ in range(10):
            # dimension must be a multiple of 3
            self._test_dproj(cone_lib.EXP, True, 54)
            self._test_dproj(cone_lib.EXP, False, 54)

    def test_pi(self):
        for _ in range(10):
            zero_dim = np.random.randint(1, 10)
            pos_dim = np.random.randint(1, 10)
            soc_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            psd_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            exp_dim = np.random.randint(3, 18)
            exp_dim -= (exp_dim % 3)
            cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                     (cone_lib.SOC, soc_dim), (cone_lib.PSD, psd_dim),
                     (cone_lib.EXP, exp_dim)]
            size = zero_dim + pos_dim + sum(soc_dim) + sum(
                [cone_lib.vec_psd_dim(d) for d in psd_dim]) + exp_dim
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

                np.testing.assert_allclose(proj[offset:],
                                           cone_lib._proj(x[offset:], cone_lib.EXP, dual=dual))

    def test_dpi(self):
        for _ in range(10):
            zero_dim = np.random.randint(1, 10)
            pos_dim = np.random.randint(1, 10)
            soc_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            psd_dim = [np.random.randint(1, 10) for _ in range(
                np.random.randint(1, 10))]
            exp_dim = np.random.randint(3, 18)
            exp_dim -= (exp_dim % 3)
            cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                     (cone_lib.SOC, soc_dim), (cone_lib.PSD, psd_dim),
                     (cone_lib.EXP, exp_dim)]
            size = zero_dim + pos_dim + sum(soc_dim) + sum(
                [cone_lib.vec_psd_dim(d) for d in psd_dim]) + exp_dim
            x = np.random.randn(size)

            for dual in [False, True]:
                Dpi = cone_lib.dpi(x, cones, dual=dual)
                proj_x = cone_lib.pi(x, cones, dual=dual)
                dx = 1e-6 * np.random.randn(size)
                z = cone_lib.pi(x + dx, cones, dual=dual)
                np.testing.assert_allclose(Dpi@dx, z - proj_x,
                                           atol=1e-3, rtol=1e-4)

    def test_get_random_like(self):
        A = sparse.eye(5)
        B = utils.get_random_like(A,
                                  lambda n: np.random.normal(0, 1e-6, size=n))
        A_r, A_c = [list(x) for x in A.nonzero()]
        B_r, B_c = [list(x) for x in B.nonzero()]
        self.assertListEqual(A_r, B_r)
        self.assertListEqual(A_c, B_c)

    def test_solve_and_derivative(self):
        m = 20
        n = 10
        A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)

        x, y, s, derivative, _ = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, eps=1e-8)

        dA = utils.get_random_like(
            A, lambda n: np.random.normal(0, 1e-6, size=n))
        db = np.random.normal(0, 1e-6, size=b.size)
        dc = np.random.normal(0, 1e-6, size=c.size)

        dx, dy, ds = derivative(dA, db, dc)

        x_pert, y_pert, s_pert, _, _ = cone_prog.solve_and_derivative(
            A + dA, b + db, c + dc, cone_dims, eps=1e-8)

        np.testing.assert_allclose(x_pert - x, dx, atol=1e-6, rtol=1e-6)

    def test_warm_start(self):
        m = 20
        n = 10
        A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
        x, y, s, _, _ = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, eps=1e-11)
        x_p, y_p, s_p, _, _ = cone_prog.solve_and_derivative(
            A, b, c, cone_dims, warm_start=(x, y, s), max_iters=1)

        np.testing.assert_allclose(x, x_p, atol=1e-7)
        np.testing.assert_allclose(y, y_p, atol=1e-7)
        np.testing.assert_allclose(s, s_p, atol=1e-7)


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
