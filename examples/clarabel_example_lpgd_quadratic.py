import diffcp

import numpy as np
import utils
from scipy import sparse

np.set_printoptions(precision=5, suppress=True)


# We generate a random cone program with a cone
# defined as a product of a 3-d zero cone, 3-d positive orthant cone,
# and a 5-d second order cone.
K = {
    'z': 3,
    'l': 3,
    'q': [5]
}

m = 3 + 3 + 5
n = 5

np.random.seed(0)

A, b, c = utils.random_cone_prog(m, n, K)
P = sparse.csc_matrix((c.size, c.size))
P = sparse.triu(P).tocsc()

# We solve the cone program and get the derivative and its adjoint
x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
    A, b, c, K, P=P, solve_method="Clarabel", verbose=False, mode="lpgd", derivative_kwargs=dict(tau=0.1, rho=0.0))

print("x =", x)
print("y =", y)
print("s =", s)

# Adjoint derivative
dA, db, dc, dP = adjoint_derivative(dx=c, dy=np.zeros(m), ds=np.zeros(m), return_dP=True)

# Derivative (dummy inputs)
dx, dy, ds = derivative(dA=A, db=b, dc=c, dP=P)
