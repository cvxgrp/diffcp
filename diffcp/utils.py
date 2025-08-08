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
    """Embeds the problem into a larger problem that allows costs on the slack variables
    A_emb = [A, I; 0, -I]
    b_emb = [b; 0]
    c_emb = [c; 0]
    P_emb = [P, 0; 0, 0]
    cone_dict_emb = cone_dict with z shifted by m
    """
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


def compute_perturbed_solution(dA, db, dc, dP, tau, rho, A, b, c, P, cone_dict, x, y, s, solver_kwargs, solve_method, solve_internal):
    """
    Computes the perturbed solution x_right, y_right, s_right at (A, b, c) with 
    perturbations dA, db, dc and optional regularization rho.
    
    Args:
        dA: SciPy sparse matrix in CSC format; must have same sparsity
            pattern as the matrix `A` from the cone program
        db: NumPy array representing perturbation in `b`
        dc: NumPy array representing perturbation in `c`
        dP: (optional) SciPy sparse matrix in CSC format; must have same sparsity
            pattern as the matrix `P` from the cone program
        tau: Perturbation strength parameter
        rho: Regularization strength parameter
    Returns:
        x_pert: Perturbed solution of the primal variable x
        y_pert: Perturbed solution of the dual variable y
        s_pert: Perturbed solution of the slack variable s
    """
    m, n = A.shape

    # Perturb the problem
    A_pert = A + tau * dA
    b_pert = b + tau * db
    c_pert = c + tau * dc
    if dP is not None:
        P_pert = P + tau * dP
    else:
        P_pert = P

    # Regularize: Effectively adds a rho/2 |x-x^*|^2 term to the objective
    P_pert_reg = regularize_P(P_pert, rho=rho, size=n)
    c_pert_reg = c_pert - rho * x

    # Set warm start
    warm_start = (x, y, s) if solve_method not in ["ECOS", "Clarabel"] else None

    # Solve the perturbed problem
    result_pert = solve_internal(A=A_pert, b=b_pert, c=c_pert_reg, P=P_pert_reg, cone_dict=cone_dict, 
                                solve_method=solve_method, warm_start=warm_start, **solver_kwargs)
    # Extract the solutions
    x_pert, y_pert, s_pert = result_pert["x"], result_pert["y"], result_pert["s"]
    return x_pert, y_pert, s_pert


def compute_adjoint_perturbed_solution(dx, dy, ds, tau, rho, A, b, c, P, cone_dict, x, y, s, solver_kwargs, solve_method, solve_internal):
    """
    Computes the adjoint perturbed solution x_right, y_right, s_right (Lagrangian Proximal Map)
    by solving the following perturbed problem with perturbations dx, dy, ds 
    and perturbation/regularization parameters tau/rho:
    argmin.     <x,Px> + <c,x> + tau*(<dx,x> + <ds,s>) + rho/2 |x-x^*|^2
    subject to  Ax + s = b-tau*dy
                s \in K

    For ds=0 we solve
    argmin.     <x,(P+rho*I)x> + <c+tau*dx-rho*x^*, x>
    subject to  Ax + s = b-tau*dy
                s \in K
    This is just a perturbed instance of the forward optimization problem.

    For ds!=0 we rewrite the problem as an embedded problem.
    We add a constraint x'=s, replace all appearances of s with x' and solve
    argmin.     <[x, x'], [[P+rho*I,     0];  [x, x']> + <[c+tau*dx-rho*x^*, tau*ds-rho*s^*], [x, x']>
                           [      0, rho*I]]
    subject to  [[A,  I];  [x, x'] + [s'; s] = [b-tau*dy; 0]
                 [0, -I]]
                (s', s) \in (0 x K)
    Note that we also add a regularizer on s in this case (rho/2 |s-s^*|^2).
    
    Args:
        dx: NumPy array representing perturbation in `x`
        dy: NumPy array representing perturbation in `y`
        ds: NumPy array representing perturbation in `s`
        tau: Perturbation strength parameter
        rho: Regularization strength parameter
    Returns:
        x_pert: Perturbed solution of the primal variable x
        y_pert: Perturbed solution of the dual variable y
        s_pert: Perturbed solution of the slack variable s
    """
    m, n = A.shape

    # The cases ds = 0 and ds != 0 are handled separately, see docstring
    if np.isclose(np.sum(np.abs(ds)), 0):
        # Perturb problem (Note: perturb primal and dual linear terms in different directions)
        c_pert = c + tau * dx
        b_pert = b - tau * dy

        # Regularize: Effectively adds a rho/2 |x-x^*|^2 term to the objective
        P_reg = regularize_P(P, rho=rho, size=n)
        c_pert_reg = c_pert - rho * x

        # Set warm start
        warm_start = (x, y, s) if solve_method not in ["ECOS", "Clarabel"] else None

        # Solve the perturbed problem
        # Note: In special case solve_method=='SCS' and rho==0, this could be sped up strongly by using solver.update
        result_pert = solve_internal(A=A, b=b_pert, c=c_pert_reg, P=P_reg, cone_dict=cone_dict, 
                                        solve_method=solve_method, warm_start=warm_start, **solver_kwargs)
        # Extract the solutions
        x_pert, y_pert, s_pert = result_pert["x"], result_pert["y"], result_pert["s"]
    else:
        # Embed problem (see docstring)
        A_emb, b_emb, c_emb, P_emb, cone_dict_emb = embed_problem(A, b, c, P, cone_dict)

        # Perturb problem (Note: perturb primal and dual linear terms in different directions)
        b_emb_pert = b_emb - tau * np.hstack([dy, np.zeros(m)])
        c_emb_pert = c_emb + tau * np.hstack([dx, ds])
        
        # Regularize: Effectively adds a rho/2 (|x-x^*|^2 + |s-s^*|^2) term to the objective
        P_emb_reg = regularize_P(P_emb, rho=rho, size=n+m)
        c_emb_pert_reg = c_emb_pert - rho * np.hstack([x, s])

        # Set warm start
        if solve_method in ["ECOS", "Clarabel"]:
            warm_start = None
        else:
            warm_start = (np.hstack([x, s]), np.hstack([y, y]), np.hstack([s, s]))

        # Solve the embedded problem
        result_pert = solve_internal(A=A_emb, b=b_emb_pert, c=c_emb_pert_reg, P=P_emb_reg, cone_dict=cone_dict_emb, 
                                    solve_method=solve_method, warm_start=warm_start, **solver_kwargs)
        # Extract the solutions
        x_pert = result_pert['x'][:n]
        y_pert = result_pert['y'][:m]
        s_pert = result_pert['x'][n:n+m]
    return x_pert, y_pert, s_pert
