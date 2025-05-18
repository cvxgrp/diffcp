import numpy as np

def vec_matrix_product(e):
    """
    mosek 使用numpy相同的行优先，
    从 展平 的矩阵e(N*N,1)映射到 列优先的下三角矩阵t(N*(N+1)/2,1)
    变换矩阵用 sparse matrix 稀疏矩阵表示
    每行挑选一个e_{i*N+j},系数为1或sqrt(2)
    s = vec(S)
    SCS:
    S = mat(s) in S_+
    e:MOSEK variable in Domain.inpsdCone(n=d) 
    """
    import mosek.fusion as mf
    N = e.getShape()[0]

    # each row of the t
    msubi =np.arange(N * (N + 1) // 2, dtype=np.int32)
    # 列优先下三角
    msubj,mcof = flatten_dense2csc_np(N)    
    S = mf.Matrix.sparse(N * (N + 1) // 2, N * N, msubi, msubj, mcof)

    return S @ mf.Expr.flatten(e)

def flatten_dense2csc_np(N):
    col_u = np.repeat(np.arange(N, dtype=np.int32), N - np.arange(N))
    row_u = np.concatenate([np.arange(k, N, dtype=np.int32) for k in range(N)])
    msubj_F_np = row_u * N + col_u
    mcof_F_np  = np.where(row_u == col_u, 1.0, 2.0**0.5)
    return msubj_F_np, mcof_F_np

def vec(e):
    # return vec_slice(e)
    return vec_matrix_product(e)

def mosek_(A, b, c, cone_dict,
           warm_start=None, raise_on_error=True, **kwargs):
    import sys
    import mosek.fusion as mf
    import mosek.fusion.pythonic        # 重载 python 符号
    import diffcp.cones as dp
    import cvxpy.settings as s
    variables_num_scs = c.size
    with mf.Model("SCS_reconstruct") as M:
        # 1) 打开 Solver 日志
        M.setLogHandler(sys.stdout)

        x = M.variable(f"x", variables_num_scs, mf.Domain.unbounded())   # primal scs variable
        slacks = []     # 构造 Ax + slacks = b
        """equalities
        (SCS)s.EQ_DIM  Ax+s=b|s=0   (MOSEK)"""
        eq_row_e = cone_dict.get(dp.EQ_DIM, 0)
        if cone_dict[dp.EQ_DIM] != 0:
            eq_row_e += cone_dict[dp.EQ_DIM]
            s_eq = M.variable(f"s_eq", cone_dict[dp.EQ_DIM], mf.Domain.equalsTo(0.0))
            slacks.append(s_eq)
            # print(f"eq_row_e={eq_row_e} Equalities: {A[:eq_row_e,:].toarray()} = {b[:eq_row_e]}")

        """inequalities | positive orthant
        (SCS)s.LEQ_DIM  Ax+s=b|s>=0   (MOSEK)"""
        leq_row_e=eq_row_e
        if cone_dict.get(s.LEQ_DIM, None) and cone_dict[s.LEQ_DIM] != 0:
            leq_row_e += cone_dict[s.LEQ_DIM]
            s_leq = M.variable(f"s_leq", cone_dict[s.LEQ_DIM], mf.Domain.greaterThan(0.0))
            slacks.append(s_leq)
            # print(f"leq_row_e={leq_row_e}"
                # f"Inequalities: {A[eq_row_e:leq_row_e,:].toarray()} <= {b[eq_row_e:leq_row_e]}")
        
        """box bound
        (SCS) (t,s)t*l <= s <= t*u, bl=[l],bu=[u]
        """
        box_row_e=leq_row_e
        if cone_dict.get('bu',None) is not None and  cone_dict.get('bl',None) is not None:
            bsize = max(cone_dict['bu'] + 1,cone_dict['bl'] + 1)
            box_row_e += bsize
            s_box = M.variable(f"s_box", bsize, mf.Domain.unbounded())
            bu = cone_dict['bu']
            bl = cone_dict['bl']
            if bu:
                M.constraint("box_upper", s_box[1:bsize] <= s_box[[0]] * bu)
            if bl:
                M.constraint("box_lower", s_box[1:bsize] >= s_box[[0]] * bl)
            slacks.append(s_box)
            # print(f"box_row_e={box_row_e}"
                # f"Box bound: {A[leq_row_e:box_row_e,:].toarray()} <= {b[leq_row_e:box_row_e]}")

        """second-order cones(SOC) constraints
        # (SCS)s.SOC_DIM [q_0,q_1,...] q_i 是第i个soc锥的变量数  A(v_{t}^T,...,v_{sq_i}^T)+s=b|t>=s_1^2+..s_{q_i-1}^T   
        # (MOSEK) [x_1,x_2,...,x_d]
        # https://docs.mosek.com/latest/pythonfusion/domain.html#domain.inqcone"""
        soc_row_e=box_row_e
        soc_cones = []
        if cone_dict.get(dp.SOC, None) is not None:
            soc_row_e += sum(cone_dict[dp.SOC])
            soc_cones = cone_dict[dp.SOC]
        if soc_cones:
            total_soc = sum(cone_dict[dp.SOC])
            s_soc = M.variable(f"s_soc", total_soc, mf.Domain.unbounded())   
            off =0
            for idx,cone_dim in enumerate(soc_cones):
                # https://docs.mosek.com/latest/pythonfusion/case-studies-portfolio.html#doc-portfolio-basic-markowitz
                M.constraint(f"soc_cone_{idx}",
                            s_soc[off:(off+cone_dim)] == mf.Domain.inQCone())
                off += cone_dim
            slacks.append(s_soc)
            # print(f"soc_row_e={soc_row_e}"
                # f"Second-order cones: {A[exp_row_e:soc_row_e,:].toarray()} <> {b[exp_row_e:soc_row_e]}")
        
        r"""positive semidefinite (PSD) constraints
        (SCS)s.PSD_DIM  <A,X>+s=b|X=vec(PSD)
        (MOSE) S \in PSD
        [vec(S)=s]
        """
        psd_row_e=soc_row_e
        if cone_dict.get(dp.PSD, None) is not None:
            psd_row_e += (np.array(cone_dict[dp.PSD]) * (np.array(cone_dict[dp.PSD]) + 1) // 2).sum()
            psd_cones = np.array(cone_dict[dp.PSD])
            # total_psd = psd_cone_dims.sum()
            col_u = lambda d: np.repeat(np.arange(d,dtype=int), d-np.arange(d))
            row_u = lambda d: np.concatenate([np.arange(k, d,dtype=int) for k in range(d)])
            # vec s
            total_soc = sum(psd_cones * (psd_cones + 1) // 2)
            s_psd = M.variable(f"s_psd", total_soc, mf.Domain.unbounded())
            off = int(0)
            for idx,d in enumerate(psd_cones):
                # 一维松弛变量 s idx^th 个PSD cone
                size = int(d*(d+1)//2)
                s_psd_idx = s_psd.slice(off,off+size)
                off += size
                # m_psd_idx:S
                m_psd_idx = M.variable(f"m_psd_{idx}",mf.Domain.inPSDCone(d))
                # vec(S) as (N*(N+1)/2,1) equal s_psd_idx
                M.constraint(f"psd_cone_{idx}",
                            vec(m_psd_idx)==s_psd_idx)

            slacks.append(s_psd)
            # print(f"psd_row_e={psd_row_e}"
                # f"Positive semidefinite cones: {A[soc_row_e:psd_row_e,:].toarray()} <> {b[soc_row_e:psd_row_e]}")
        
        """primal exponential cone
        # (SCS)s.EXP_DIM  A(v_x^T,v_y^T,v_z^T)+s=b,Av_x^T=x| y exp(x/y) <= z, y>0   
        # (MOSEK) [x_1,x_2,x_3] x_2*exp(x_3/x_2) ≤ x_1, x_2>0 (x_1>=x_2*R_+> 0)
        # [z,y,x] 
        # https://docs.mosek.com/latest/pythonfusion/domain.html#domain.inpexpcone"""
        exp_row_e=psd_row_e
        if cone_dict.get(dp.EXP, None) is not None:
            exp_row_e += 3 * cone_dict[dp.EXP]
            # TODO 注意SCS 与 MOSEK 的顺序不同
            s_exp = M.variable(f"s_exp", 3 * cone_dict[dp.EXP], mf.Domain.unbunded())
            for i in range(cone_dict[dp.EXP]):
                M.constraint(f"exp_cone_{i}",
                            # s_exp[i*3:i*3+3], 
                            # return 1-dim Expr 
                            # https://docs.mosek.com/latest/pythonfusion/pythonicfusion.html#indexing-and-slicing
                            s_exp[[i*3+2,i*3+1,i*3]] == mf.Domain.inPExpCone())
            slacks.append(s_exp)
            # print(f"exp_row_e={exp_row_e}"
                # f"Exponential cone_dict: {A[psd_row_e:exp_row_e,:].toarray()} <> {b[psd_row_e:exp_row_e]}")

        """ dual exponential cones !don't have in cvxpy-scs, only in cvxpy-mosek and scs 
        (SCS) [u,v,w] -u*exp(v/u) <= ew , u<0
        (MOSEK) [x_1,x_2,x_3] -x_3*e^{-1}*exp(x_2/x_3) <= x_1, x_3<0, x_1>0
        (w,v,u)"""
        exp_dual_row_e=exp_row_e
        if cone_dict.get('ed',None) is not None:
            exp_dual_row_e += 3 * cone_dict[dp.EXP_DUAL]
            s_de = s_exp_dual = M.variable(f"s_de", 3 * cone_dict['ed'], mf.Domain.unbounded())
            for i in range(cone_dict['ed']):
                M.constraint(f"exp__cone_{i}",
                            s_de[[i*3+2,i*3+1,i*3]] == mf.Domain.inDExpCone())
            slacks.append(s_de)
            # print(f"exp__row_e={exp_dual_row_e}"
                # f"Exponential  cone_dict: {A[exp_row_e:exp_dual_row_e,:].toarray()} <> {b[exp_row_e:exp_dual_row_e]}")
        
        r"""Power cone &  power cone | 3d power cones(P3D) constraints
        (SCS)'p3'  [x,y,z] x^p * y^{1-p} >= |z|,p\in[0,1], 
            positive entries for primal, negative entries for  power cones
            [u,v,w] (u/p)^p * (v/1-p)^{1-p} >= |w|,-p\in[0,1]
        (MOSEK) [x_1,x_2,x_i] x_1^p * x_2^{1-p} >= sqrt{x_3^2+x_i^2},p\in[-1,1]
            Primal: https://docs.mosek.com/latest/pythonfusion/domain.html#domain.inppowercone
            :   https://docs.mosek.com/latest/pythonfusion/domain.html#domain.indpowercone
        """
        p3_row_e=exp_dual_row_e
        if cone_dict.get('p',None) is not None:
            p3_dims = cone_dict['p']
            p3_row_e  += 3 * sum(p3_dims)
            tot_p3 = len(p3_dims)
            s_p3 = M.variable(f"s_p3", 3*tot_p3, mf.Domain.unbounded())
            for k in range(tot_p3):
                p = p3_dims[k]
                if p == 0:                  # y>=|z|
                    M.constraint(f"p3_{k}", s_p3[3*k+1] >= mf.Expr.abs(s_p3[3*k+2]))
                elif p == 1 or p == -1:     # x>=|z|
                    M.constraint(f"p3_{k}", s_p3[3*k]   >= mf.Expr.abs(s_p3[3*k+2]))
                elif p > 0:
                    M.constraint(f"p3_{k}", s_p3.slice(3*k, 3*k+3), mf.Domain.inPPowerCone(p3_dims[k]))
                else:
                    M.constraint(f"p3_{k}", s_p3.slice(3*k, 3*k+3), mf.Domain.inDPowerCone(-p3_dims[k]))
            slacks.append(s_p3)
            # print(f"p3_row_e={p3_row_e}"
                # f"3d power cones: {A[exp_dual_row_e:p3_row_e,:].toarray()} <> {b[exp_dual_row_e:p3_row_e]}")
        # SCS: S_all = (-A)x + b, -A is what cvxpy givns us
        S_all = mf.Expr.vstack([v.asExpr() for v in slacks])
        b_const = mf.Expr.constTerm(b)
        A_const = mf.Matrix.dense(A.toarray()) 
        M.constraint("affine_eq", 
                    (A_const @ x) + S_all ==  b_const
                    )
        # c_const = mf.Expr.constTerm(c_scs)
        M.objective("objective", mf.ObjectiveSense.Minimize, mf.Expr.dot(c, x))

        # print(f"mosek_question:\n",M)
        try:
            M.solve()
            x_opt = x.level()   # x  is {0}
            s_opt = np.concatenate([v.level() for v in slacks])
            y_opt = s__opt = np.concatenate([v.dual() for v in slacks])
            # print(f"Optimal value: {M.primalObjValue}")
            # print(f"Optimal x:\n {x_opt}",
            #       f"Optimal y:\n {y_opt}",
            #       f"Optimal slack variables:\n {s_opt}\nslacks={slacks}")
            # 获取 mosek 求解信息
            status = M.getPrimalSolutionStatus()
            if status == mf.SolutionStatus.Optimal:
                status = "Optimal"
            print(f"status={status}")
            solve_time = M.getSolverDoubleInfo("optimizerTime")
            iterations = M.getSolverIntInfo("intpntIter")
            presolve_time = M.getSolverDoubleInfo("presolveTime")   # https://docs.mosek.com/latest/pythonfusion/constants.html#mosek.dinfitem.presolve_time
            # print(f"solve_time={solve_time}")
            # print(f"iterations={iterations}")
            # print(f"presolve_time={presolve_time}")
            info = {
                "status": status,
                "solveTime": solve_time,
                "setupTime": presolve_time,
                "iter": iterations,
                "pobj": M.primalObjValue,
            }
            # print(f"info={info}")
            return {'x':x_opt, 'y':y_opt, 's':s_opt,'info': info}
        except mf.SolutionError as e:
            # print("=== Solve failed ===")
            # print(e)
            return {'info': {'status': str(e)}, 'x': None, 'y': None, 's': None}