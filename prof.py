import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time

import diffcp.cone_program as cone_prog
import diffcp.cones as cone_lib
import diffcp.utils as utils


modes = ['lsqr', 'dense']
n_batch = 256
m = 100
n = 50

data = []
for i in range(n_batch):
    data.append(utils.least_squares_eq_scs_data(m, n, seed=i))
A, b, c, cone_dims = zip(*data)

for mode in modes:
    x, y, s, derivative, adjoint_derivative = cone_prog.solve_and_derivative_batch(
        A, b, c, cone_dims, eps=1e-10, mode=mode)

    dA, db, dc = [], [], []
    for i in range(n_batch):
        dA.append(utils.get_random_like(
            A[0], lambda n: np.random.normal(0, 1e-2, size=n)))
        db.append(np.random.normal(0, 1e-2, size=b[0].size))
        dc.append(np.random.normal(0, 1e-2, size=c[0].size))

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
            c, [np.zeros(y[0].size)] * n_batch, [np.zeros(s[0].size)] * n_batch)
        toc = time.time()
        adjoint_derivative_time += (toc - tic) / 10

    print(mode, derivative_time, adjoint_derivative_time)
