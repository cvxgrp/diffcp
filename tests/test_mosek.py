# pytest -s submodule/diffcp/tests/test_mosek.py
import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

import diffcp.cone_program as cone_prog
import diffcp.cones as cone_lib
import diffcp.utils as utils

def test_sdp_and_compare():
    np.random.seed(42)
    n = 3
    m = 2

    # 变量
    X = cp.Variable((n, n), PSD=True)      # PSD锥
    Y = cp.Variable((m, m), PSD=True)      # 另一个PSD锥
    z = cp.Variable(4)                     # 非负锥
    t = cp.Variable()                      # 标量
    soc_var = cp.Variable(3)               # 二阶锥

    # 目标
    obj = cp.Minimize(cp.trace(X) + cp.trace(Y) + cp.sum(z) + t + soc_var[0])

    # 约束
    constraints = [
        X[0, 0] + Y[1, 1] + z[0] == 1,
        cp.sum(z) + t >= 2,
        z >= 0,
        cp.norm(soc_var[1:]) <= soc_var[0],  # SOC
        X[1, 2] == 0.5,
        Y[0, 1] == -0.2,
        cp.trace(X) + cp.trace(Y) + t <= 10,
    ]

    prob = cp.Problem(obj, constraints)
    # prob.solve(solver=cp.MOSEK, eps=1e-8, verbose=False)
    prob.solve(solver=cp.SCS, eps=1e-9, verbose=False)
    x_cvx = X.value
    y_cvx = Y.value
    z_cvx = z.value
    t_cvx = t.value
    soc_cvx = soc_var.value
    obj_cvx_SCS = prob.value
    prob.solve(solver=cp.MOSEK, eps=1e-9, verbose=False)
    obj_cvx_MOSEK = prob.value

    # 转换为SCS数据
    A, b, c, cone_dims = utils.scs_data_from_cvxpy_problem(prob)

    # 用diffcp求解
    x_diffcp, y_diffcp, s_diffcp, *_ = cone_prog.solve_and_derivative(
        A, b, c, cone_dims, solve_method="mosek", eps=1e-8
    )

    # 还原变量（需根据utils.scs_data_from_cvxpy_problem的变量顺序拆分）
    # 这里只做简单对比目标值
    obj_diffcp = c.T @ x_diffcp

    # 对比目标值
    print(f"\n x diffcp: {x_diffcp}\nx cvx: {x_cvx}")
    np.testing.assert_allclose(obj_diffcp, obj_cvx_SCS, atol=1e-6)
    np.testing.assert_allclose(obj_diffcp, obj_cvx_MOSEK, atol=1e-6)
    # np.testing.assert_allclose(x_diffcp[:n*n].reshape(n, n), x_cvx, atol=1e-6)
    print(f"\nCVXPY SCS objective:{obj_cvx_SCS}")
    print("CVXPY MOSEK objective:", obj_cvx_MOSEK)
    print("diffcp objective:", obj_diffcp)
    print("Test passed: diffcp matches CVXPY on SDP+other cones.")

