# -*- coding: utf-8 -*-
"""COPT 求解器接口, 对应 MATLAB 中的 solve.m (原使用 Gurobi).

已按 coptpy 8.x 矩阵接口 (addMVar/MVar + numpy/scipy 矩阵运算) 适配并实测通过;
注意 coptpy 的 LinExpr/QuadExpr 构造参数顺序为 (vars, coeffs), 与 Gurobi 相反,
且二次约束必须用 addQConstr 而不是 addConstr。

用法:
    result = solve(model, tolerance_gap=0.001, timelimit=3600, input_params={...})

result 为 SimpleNamespace, 字段与 Gurobi 返回结构保持一致:
    x        : 最优解 (n,)
    objval   : 目标值
    runtime  : 求解耗时(秒)
    status   : 求解状态字符串

input_params 支持的键 (对应 MATLAB 中的 input_params):
    Threads    : 线程数
    Seed       : 随机种子 (COPT 无对应公开参数时自动忽略)
    outputflag : 是否输出日志 (0/1)
"""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .milp_model import MilpModel


def _set_param(model, param_map, name, value):
    """设置 COPT 参数, 不支持的参数自动忽略."""
    try:
        attr = param_map.get(name)
        if attr is None:
            return
        model.setParam(attr, value)
    except Exception:
        pass


def solve(solve_model: MilpModel, tolerance_gap: float = 0.001,
          timelimit: float = None, input_params: dict = None) -> SimpleNamespace:
    try:
        import coptpy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "未找到 coptpy (COPT Python 接口)。请安装 COPT 求解器并配置好 "
            "Python 接口 (pip 安装 coptpy 或配置 COPT 安装目录) 后再运行。"
        ) from exc

    env = coptpy.Envr()
    model = env.createModel("solve_model")

    param_map = {}
    for name in dir(coptpy.COPT.Param):
        param_map[name] = getattr(coptpy.COPT.Param, name)

    n = solve_model.num_vars()
    lb = np.zeros(n) if solve_model.lb is None else np.asarray(solve_model.lb, dtype=float).ravel()
    ub = np.full(n, np.inf) if solve_model.ub is None else np.asarray(solve_model.ub, dtype=float).ravel()

    # 变量 (coptpy 8 矩阵接口: addMVar 返回 MVar, 直接支持 numpy/scipy 矩阵
    # 运算; vtype 需传单字符列表, 纯字符串只支持单类型广播)
    vtype = list(solve_model.ctype) if solve_model.ctype is not None else 'C'
    x = model.addMVar(n, lb=lb, ub=ub, vtype=vtype)

    # 目标函数: min f'x + x'Qx  (MATLAB solve.m 中 Q 直接以 x'Qx/2? 注意:
    # Gurobi 的 model.Q 约定为目标 x'Qx (Q 为上三角存储, 实际贡献 0.5*x'(Q+Q')x)。
    # COPT 的二次目标同样写作 x'Qx。为保持与 Gurobi 语义一致, 这里使用
    # obj = f'x + x'Qx, 其中 Q 取 0.5*(Q+Q') 的上三角, 与 Gurobi 完全相同。)
    f = np.zeros(n) if solve_model.f is None else np.asarray(solve_model.f, dtype=float).ravel()
    Q = solve_model.Q if solve_model.Q is not None else solve_model.H
    obj = f @ x
    if Q is not None and Q.shape[0] > 0 and Q.nnz > 0:
        Qs = 0.5 * (Q + Q.T)
        Qs = sparse.triu(Qs, format="csr")
        if Qs.nnz > 0:
            obj = obj + x @ Qs @ x
    model.setObjective(obj, coptpy.COPT.MINIMIZE)

    # 线性约束
    if solve_model.Aineq is not None and solve_model.Aineq.shape[0] > 0:
        bineq = np.asarray(solve_model.bineq, dtype=float).ravel()
        model.addConstr(solve_model.Aineq.tocsr() @ x <= bineq)
    if solve_model.Aeq is not None and solve_model.Aeq.shape[0] > 0:
        beq = np.asarray(solve_model.beq, dtype=float).ravel()
        model.addConstr(solve_model.Aeq.tocsr() @ x == beq)

    # 二次约束 x'Qc x + l'x <= r
    if solve_model.Qc is not None:
        for i, Qc in enumerate(solve_model.Qc):
            if Qc is None or Qc.shape[0] == 0:
                continue
            Qs = 0.5 * (Qc + Qc.T)
            Qs = sparse.triu(Qs, format="csr")
            expr = x @ Qs @ x
            has_lin = False
            if solve_model.l is not None and solve_model.l.shape[0] > 0:
                lmat = solve_model.l
                li = (lmat[:, i].toarray().ravel() if sparse.issparse(lmat)
                      else np.asarray(lmat)[:, i].ravel())
                if np.any(li != 0):
                    expr = expr + li @ x
                    has_lin = True
            if Qs.nnz > 0 or has_lin:
                model.addQConstr(expr <= float(solve_model.r[i]))

    # 参数
    model.setParam(coptpy.COPT.Param.Logging, 0)
    model.setParam(coptpy.COPT.Param.RelGap, tolerance_gap)
    if timelimit is not None:
        model.setParam(coptpy.COPT.Param.TimeLimit, timelimit)
    if input_params:
        _set_param(model, param_map, "Threads", input_params.get("Threads"))
        _set_param(model, param_map, "Seed", input_params.get("Seed"))
        if "outputflag" in input_params and input_params["outputflag"]:
            model.setParam(coptpy.COPT.Param.Logging, 1)

    model.solve()

    result = SimpleNamespace()
    try:
        result.x = np.asarray(model.getValues())
    except coptpy.CoptError as exc:
        raise RuntimeError(
            f"COPT 未返回可行解 (model.status={model.status}; "
            "1=OPTIMAL 2=INFEASIBLE 3=UNBOUNDED 7/8=LIMIT 无 incumbent)"
        ) from exc
    result.objval = model.objval
    result.runtime = float(model.getAttr(coptpy.COPT.Attr.SolvingTime))
    result.status = model.status
    # 复现性: 记录 MIP best bound 与最终 gap (属性名随 coptpy 版本可能不同, 容错)
    result.best_bound = float('nan')
    result.final_gap = float('nan')
    for name in ('BestBound', 'BestObjBound', 'ObjBound'):
        try:
            result.best_bound = float(model.getAttr(getattr(coptpy.COPT.Attr, name)))
            break
        except Exception:
            continue
    try:
        result.final_gap = float(model.getAttr(coptpy.COPT.Attr.BestGap))
    except Exception:
        if np.isfinite(result.best_bound) and np.isfinite(result.objval):
            result.final_gap = abs(result.objval - result.best_bound) \
                / max(abs(result.objval), 1e-9)
    return result
