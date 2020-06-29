import diffcp

import numpy as np
import utils
np.set_printoptions(precision=5, suppress=True)


# We generate a random cone program with a cone
# defined as a product of a 3-d fixed cone, 3-d positive orthant cone,
# and a 5-d second order cone.
K = {
    'f': 3,
    'l': 3,
    'q': [5]
}

m = 3 + 3 + 5
n = 5

np.random.seed(0)

A, b, c = utils.random_cone_prog(m, n, K)

# We solve the cone program and get the derivative and its adjoint
x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
    A, b, c, K, solver="ECOS", verbose=False)

print("x =", x)
print("y =", y)
print("s =", s)

# We evaluate the gradient of the objective with respect to A, b and c.
dA, db, dc = adjoint_derivative(c, np.zeros(
    m), np.zeros(m), atol=1e-10, btol=1e-10)

# The gradient of the objective with respect to b should be
# equal to minus the dual variable y (see, e.g., page 268 of Convex Optimization by
# Boyd & Vandenberghe).
print("db =", db)
print("-y =", -y)
