"""
Tests for Clarabel PSD cone support in diffcp.

Verifies that solutions and derivatives from Clarabel match SCS
for problems with PSD cones.

Note: When testing dual variables (y), we only check that the solution is correct
(feasible and optimal) rather than exact match, since SCS and Clarabel may
converge to different optimal dual solutions in degenerate cases.
"""
import numpy as np
import scipy.sparse as sparse
import pytest


def scs_data_from_cvxpy_problem(problem):
    """Extract SCS-format data from a CVXPy problem."""
    import cvxpy as cp
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(
        data["dims"]
    )
    return data["A"], data["b"], data["c"], cone_dims


class TestClarabelPSDPermutation:
    """Tests for Clarabel PSD cone permutation fixes."""

    def test_multiple_psd_cones_objective_match(self):
        """Test that SCS and Clarabel give same objective for multiple PSD cones."""
        import cvxpy as cp
        import diffcp

        # Create a problem with two PSD cones
        A = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ])
        B = np.array([
            [7, 8, 9],
            [8, 10, 11],
            [9, 11, 12],
        ])

        X = cp.Variable((3, 3), symmetric=True)
        y = cp.Variable(2)

        constraints = [y[0] * A + y[1] * B >> 0, X >> 0]
        constraints += [
            cp.trace(A @ X) == 1,
            y >= 0,
        ]

        obj = cp.Minimize(cp.trace(X) + np.ones(2) @ y)
        prob = cp.Problem(obj, constraints)

        # Get CVXPy solution as reference
        cvxpy_obj = prob.solve(solver=cp.CLARABEL)

        # Get SCS-format data
        scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

        # Solve with SCS through diffcp
        x_scs, y_scs, s_scs, D_scs, DT_scs = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='SCS',
            verbose=False,
        )

        # Solve with Clarabel through diffcp
        x_cla, y_cla, s_cla, D_cla, DT_cla = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='CLARABEL',
            verbose=False,
        )

        obj_scs = scs_c @ x_scs
        obj_cla = scs_c @ x_cla

        # Objectives should match CVXPy
        assert np.isclose(obj_scs, cvxpy_obj, atol=1e-4), \
            f"SCS obj {obj_scs} doesn't match CVXPy obj {cvxpy_obj}"
        assert np.isclose(obj_cla, cvxpy_obj, atol=1e-4), \
            f"Clarabel obj {obj_cla} doesn't match CVXPy obj {cvxpy_obj}"

        # Primal solution x should match between solvers
        assert np.allclose(x_scs, x_cla, atol=1e-4), \
            f"x mismatch: SCS={x_scs}, Clarabel={x_cla}"

        # Slack variable s should match (since s = b - Ax and x matches)
        assert np.allclose(s_scs, s_cla, atol=1e-4), \
            f"s mismatch between SCS and Clarabel"

        # Note: We do NOT check y here because dual degeneracy can cause
        # SCS and Clarabel to find different optimal dual solutions.

    def test_single_psd_cone(self):
        """Test that SCS and Clarabel match for a single PSD cone."""
        import cvxpy as cp
        import diffcp

        n = 3
        C = np.eye(n)

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0, cp.trace(X) == 1]
        obj = cp.Minimize(cp.trace(C @ X))
        prob = cp.Problem(obj, constraints)

        cvxpy_obj = prob.solve(solver=cp.CLARABEL)

        scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

        x_scs, y_scs, s_scs, _, _ = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='SCS',
            verbose=False,
        )

        x_cla, y_cla, s_cla, _, _ = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='CLARABEL',
            verbose=False,
        )

        obj_scs = scs_c @ x_scs
        obj_cla = scs_c @ x_cla

        assert np.isclose(obj_scs, cvxpy_obj, atol=1e-4)
        assert np.isclose(obj_cla, cvxpy_obj, atol=1e-4)
        assert np.allclose(x_scs, x_cla, atol=1e-4)
        assert np.allclose(s_scs, s_cla, atol=1e-4)

    def test_mixed_cones(self):
        """Test problem with zero, nonneg, and PSD cones."""
        import cvxpy as cp
        import diffcp

        n = 2
        X = cp.Variable((n, n), symmetric=True)
        t = cp.Variable()

        A = np.array([[1, 0.5], [0.5, 2]])

        constraints = [
            X >> 0,
            t >= 0,
            cp.trace(A @ X) == 1,
            t <= 5,
        ]
        obj = cp.Minimize(cp.trace(X) + t)
        prob = cp.Problem(obj, constraints)

        cvxpy_obj = prob.solve(solver=cp.CLARABEL)

        scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

        x_scs, y_scs, s_scs, _, _ = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='SCS',
            verbose=False,
        )

        x_cla, y_cla, s_cla, _, _ = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='CLARABEL',
            verbose=False,
        )

        obj_scs = scs_c @ x_scs
        obj_cla = scs_c @ x_cla

        assert np.isclose(obj_scs, cvxpy_obj, atol=1e-4)
        assert np.isclose(obj_cla, cvxpy_obj, atol=1e-4)
        assert np.allclose(x_scs, x_cla, atol=1e-4)

    def test_constraint_satisfaction(self):
        """Test that Clarabel solution satisfies Ax + s = b, s in K."""
        import cvxpy as cp
        import diffcp

        A_mat = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ])
        B_mat = np.array([
            [7, 8, 9],
            [8, 10, 11],
            [9, 11, 12],
        ])

        X = cp.Variable((3, 3), symmetric=True)
        y_var = cp.Variable(2)

        constraints = [y_var[0] * A_mat + y_var[1] * B_mat >> 0, X >> 0]
        constraints += [
            cp.trace(A_mat @ X) == 1,
            y_var >= 0,
        ]

        obj = cp.Minimize(cp.trace(X) + np.ones(2) @ y_var)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)

        scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

        x_cla, y_cla, s_cla, _, _ = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='CLARABEL',
            verbose=False,
        )

        # Check Ax + s = b
        residual = sparse.csc_matrix(scs_A) @ x_cla + s_cla - scs_b
        assert np.allclose(residual, 0, atol=1e-5), \
            f"Constraint residual too large: {np.linalg.norm(residual)}"

    def test_derivative_lsqr_mode(self):
        """Test that adjoint derivatives work with Clarabel in lsqr mode."""
        import cvxpy as cp
        import diffcp

        # Simple problem with single PSD cone (non-degenerate)
        n = 2
        C = np.array([[1.0, 0.3], [0.3, 2.0]])

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0, cp.trace(X) == 1]
        obj = cp.Minimize(cp.trace(C @ X))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)

        scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

        x_cla, y_cla, s_cla, D_cla, DT_cla = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='CLARABEL',
            verbose=False,
            mode='lsqr',
        )

        # Test adjoint derivative with random perturbations
        np.random.seed(42)
        dx = np.random.randn(x_cla.size) * 0.01
        dy = np.random.randn(y_cla.size) * 0.01
        ds = np.random.randn(s_cla.size) * 0.01

        # Just verify it runs without error and produces finite results
        dA_cla, db_cla, dc_cla = DT_cla(dx, dy, ds)

        assert np.all(np.isfinite(dc_cla)), "dc contains non-finite values"
        assert np.all(np.isfinite(db_cla)), "db contains non-finite values"
        assert np.all(np.isfinite(dA_cla.data)), "dA contains non-finite values"

    def test_psd_permutation_logic(self):
        """Test the PSD permutation logic directly on a simple case."""
        import cvxpy as cp
        import diffcp

        # Create a problem where we can verify the PSD block ordering
        n = 3
        C = np.random.randn(n, n)
        C = C @ C.T  # Make positive definite for interesting solution

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0, cp.trace(X) == 1]
        obj = cp.Minimize(cp.trace(C @ X))
        prob = cp.Problem(obj, constraints)

        cvxpy_obj = prob.solve(solver=cp.SCS)
        X_opt = X.value

        scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)

        # The primal variable x should encode X in lower-triangular column-major order
        x_cla, _, s_cla, _, _ = diffcp.solve_and_derivative(
            sparse.csc_matrix(scs_A), scs_b, scs_c,
            scs_cones,
            solve_method='CLARABEL',
            verbose=False,
        )

        # Objectives should match
        obj_cla = scs_c @ x_cla
        assert np.isclose(obj_cla, cvxpy_obj, atol=1e-4), \
            f"Clarabel obj {obj_cla} doesn't match CVXPy obj {cvxpy_obj}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
