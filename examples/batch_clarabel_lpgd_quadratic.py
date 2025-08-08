import diffcp
import time
import numpy as np
import scipy.sparse as sparse

m = 100
n = 50

batch_size = 16
n_jobs = 2

As, bs, cs, Ks, Ps = [], [], [], [], []
for _ in range(batch_size):
    A, b, c, K = diffcp.utils.least_squares_eq_scs_data(m, n)
    P = sparse.csc_matrix((c.size, c.size))
    P = sparse.triu(P).tocsc()
    As += [A]
    bs += [b]
    cs += [c]
    Ks += [K]
    Ps += [P]

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
                                                 n_jobs_forward=n_jobs, n_jobs_backward=n_jobs, solve_method="Clarabel", verbose=False, mode="lpgd", derivative_kwargs=dict(tau=1e-3, rho=0.1),
                                                 Ps=Ps)
    xs, ys, ss, D_batch, DT_batch = diffcp.solve_and_derivative_batch(As, bs, cs, Ks,
                                                                      n_jobs_forward=1, n_jobs_backward=n_jobs, solve_method="Clarabel", verbose=False,
                                                                      mode="lpgd", derivative_kwargs=dict(tau=1e-3, rho=0.1), Ps=Ps)

    def f_backward():
        DT_batch(xs, ys, ss, return_dP=True)

    mean_forward, std_forward = time_function(f_forward)
    mean_backward, std_backward = time_function(f_backward)
    print("%03d | %4.4f +/- %2.2f | %4.4f +/- %2.2f" %
          (n_jobs, mean_forward, std_forward, mean_backward, std_backward))
