import diffcp
import utils
import IPython as ipy
import time
import numpy as np

m = 100
n = 50

batch_size = 16
n_jobs = 1

As, bs, cs, Ks = [], [], [], []
for _ in range(batch_size):
    A, b, c, K = diffcp.utils.least_squares_eq_scs_data(m, n)
    As += [A]
    bs += [b]
    cs += [c]
    Ks += [K]


def time_function(f, N=1):
    result = []
    for i in range(N):
        tic = time.time()
        f()
        toc = time.time()
        result += [toc - tic]
    return np.mean(result), np.std(result)

for n_jobs in range(1, 8):
    def f_forward():
        return diffcp.solve_and_derivative_batch(As, bs, cs, Ks,
                                                 n_jobs_forward=n_jobs, n_jobs_backward=n_jobs, solver="ECOS", verbose=False)
    xs, ys, ss, D_batch, DT_batch = diffcp.solve_and_derivative_batch(As, bs, cs, Ks,
                                                                      n_jobs_forward=1, n_jobs_backward=n_jobs, solver="ECOS", verbose=False)

    def f_backward():
        DT_batch(xs, ys, ss, mode="lsqr")

    mean_forward, std_forward = time_function(f_forward)
    mean_backward, std_backward = time_function(f_backward)
    print("%03d | %4.4f +/- %2.2f | %4.4f +/- %2.2f" %
          (n_jobs, mean_forward, std_forward, mean_backward, std_backward))
