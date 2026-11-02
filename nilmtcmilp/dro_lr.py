# -*- coding: utf-8 -*-
"""分布鲁棒线性回归 (DRO-LR), 对应 function_DRO_LR.m / function_DRO_LR_fixB.m / DRO_v1.m / DRO_v2.m.

模型: min  a*r + 1/N*sum(b_i)
      s.t.  |y_i - x_i'*beta| <= b_i          (i=1..N)
            ||beta||^2 + 1 <= a^2            (二次锥约束)

返回 beta (斜率向量, MATLAB 中为 2x1: [截距; 斜率]).
"""

import numpy as np
from scipy import sparse

from .copt_solver import solve
from .milp_model import MilpModel


def _dro_lr(x, y, eta, B=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N = x.shape[1]                      # 样本数
    a_num = 1 if np.isscalar(eta) else len(eta)
    b_num = N
    beta_num = x.shape[0]
    x_num = a_num + b_num + beta_num

    # 变量位置 (0-based)
    a_location = np.arange(0, a_num)
    b_location = np.arange(a_num, a_num + b_num)
    beta_location = np.arange(a_num + b_num, x_num)

    # 目标
    f = np.zeros(x_num)
    f[a_location] = eta if not np.isscalar(eta) else eta
    f[b_location] = 1.0 / N

    # 二次约束: ||beta||^2 + 1 <= a^2
    q = sparse.lil_matrix((x_num, x_num))
    for i in a_location:
        q[i, i] = -1.0
    for i in beta_location:
        q[i, i] = 1.0
    l = np.zeros(x_num)
    r = -1.0

    # 不等式: -b_i - x_i'beta <= -y_i ;  -b_i + x_i'beta <= y_i
    row_index = np.arange(N)
    zero_a = sparse.csr_matrix((N, a_num))
    minus_I = -sparse.identity(N, format="csr")
    A1 = sparse.hstack([zero_a, minus_I, -x.T], format="csr")
    b1 = -y
    A2 = sparse.hstack([zero_a, minus_I, x.T], format="csr")
    b2 = y
    Aineq = sparse.vstack([A1, A2], format="csr")
    bineq = np.concatenate([b1, b2])

    lb = np.zeros(x_num)
    lb[beta_location] = -np.inf
    if B is not None:
        lb[beta_location[0]] = B
    ub = np.full(x_num, np.inf)
    if B is not None:
        ub[beta_location[0]] = B
    ctype = 'C' * x_num

    model = MilpModel(f=f, Aineq=Aineq, bineq=bineq, lb=lb, ub=ub, ctype=ctype,
                      Qc=[q.tocsr()], l=l.reshape(-1, 1), r=np.array([r]))
    result = solve(model, 0.005, 3600)
    return result.x[beta_location]


def dro_lr(x, y, eta):
    """function_DRO_LR"""
    return _dro_lr(x, y, eta, B=None)


def dro_lr_fixB(x, y, eta, B):
    """function_DRO_LR_fixB (截距固定为 B)"""
    return _dro_lr(x, y, eta, B=B)


def dro_v1(list_old, accuracy, eta, max_iter: int = 200):
    """DRO_v1: 截尾迭代 DRO 回归.

    注: MATLAB 原版在 rate > accuracy 时每次迭代都重新对 list_old 拟合,
    结果是确定性的死循环; 这里改为定点迭代 (beta_old <- beta_new)。
    定点映射可能进入周期循环 (实测会在两个截尾窗之间振荡), 因此增加:
    - max_iter 上限;
    - 周期检测 (beta0 重访即判定循环, 返回当前解);
    - 静默 (原实现每个迭代打印 rate, 上万次迭代会被 I/O 拖死)。
    """
    list_old = np.asarray(list_old, dtype=float).ravel()
    idx = np.arange(1, len(list_old) + 1)
    state_beta_old = dro_lr(np.vstack([np.ones_like(idx), idx]), list_old, eta)
    seen = {round(float(state_beta_old[0]), 6)}
    for _ in range(max_iter):
        state_min = state_beta_old[0]
        state_max = state_beta_old[0] + len(list_old) * state_beta_old[1]
        list_new = list_old[(list_old >= state_min) & (list_old <= state_max)]
        if list_new.size == 0:
            list_new = list_old
        state_beta_new = dro_lr(np.vstack([np.ones(len(list_new)), np.arange(1, len(list_new) + 1)]),
                                list_new, eta)
        rate = abs((state_beta_old[0] - state_beta_new[0]) / state_beta_old[0])
        if rate <= accuracy:
            return state_beta_new
        key = round(float(state_beta_new[0]), 6)
        if key in seen:            # 进入周期循环, 定点不存在 — 返回当前解
            return state_beta_new
        seen.add(key)
        state_beta_old = state_beta_new
    return state_beta_old          # 达到迭代上限, 返回当前最优近似


def dro_v2(list_old, accuracy, eta, max_iter: int = 200):
    """DRO_v2: 普通最小二乘 + DRO 截尾迭代 (regress 用 numpy lstsq 实现).

    与 dro_v1 相同的终止保护 (max_iter + 周期检测 + 静默)。
    """
    list_old = np.asarray(list_old, dtype=float).ravel()
    X = np.vstack([np.ones(len(list_old)), np.arange(1, len(list_old) + 1)]).T
    state_beta_old, *_ = np.linalg.lstsq(X, list_old, rcond=None)
    seen = {round(float(state_beta_old[0]), 6)}
    for _ in range(max_iter):
        state_min = state_beta_old[0]
        state_max = state_beta_old[0] + len(list_old) * state_beta_old[1]
        list_new = list_old[(list_old >= state_min) & (list_old <= state_max)]
        if list_new.size == 0:
            list_new = list_old
        state_beta_new = dro_lr(np.vstack([np.ones(len(list_new)), np.arange(1, len(list_new) + 1)]),
                                list_new, eta)
        rate = abs((state_beta_old[0] - state_beta_new[0]) / state_beta_old[0])
        if rate <= accuracy:
            return state_beta_new
        # MATLAB 循环中下次迭代重新用 regress 拟合 list_old
        state_beta_old, *_ = np.linalg.lstsq(X, list_old, rcond=None)
        key = round(float(state_beta_old[0]), 6)
        if key in seen:
            return state_beta_new
        seen.add(key)
    return state_beta_old
