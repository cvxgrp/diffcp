import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time

import diffcp.cone_program as cone_prog
import diffcp.cones as cone_lib
import diffcp.utils as utils


m = 100
n = 50

A, b, c, cone_dims = utils.least_squares_eq_scs_data(m, n)
for mode in ["lsqr", "dense"]:
    x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative(
        A, b, c, cone_dims, eps=1e-10, mode=mode)

    dA = utils.get_random_like(
        A, lambda n: np.random.normal(0, 1e-2, size=n))
    db = np.random.normal(0, 1e-2, size=b.size)
    dc = np.random.normal(0, 1e-2, size=c.size)

    derivative_time = 0.0
    for _ in range(10):
        tic = time.time()
        dx, dy, ds = derivative(dA, db, dc)
        toc = time.time()
        derivative_time += (toc - tic) / 10

    adjoint_derivative_time = 0.0
    for _ in range(10):
        tic = time.time()
        dA, db, dc = adjoint_derivative(
            c, np.zeros(y.size), np.zeros(s.size))
        toc = time.time()
        adjoint_derivative_time += (toc - tic) / 10

    print(mode, derivative_time, adjoint_derivative_time)
