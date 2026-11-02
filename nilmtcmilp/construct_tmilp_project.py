# -*- coding: utf-8 -*-
"""TMILP_1bin_project 模型构建, 对应 function_construction_TMILP_1bin_project.m.

与 TMILP_1bin 的区别: P/Q 变量归一化到 [0,1], 功率平衡用
P = P_bottom*x + (P_top-P_bottom)*Pijt 展开, ramping 上界同样归一化.

原 construct_emilp.py 移除后, EMILP/TMILP 家族共用的目标罚项
_penalty_vector 与主脚本的功率分解函数 decompose_power 也移到了这里。
"""

import numpy as np
from scipy import sparse

from .milp_model import MilpModel, RowAccumulator
from .utils import index_appliance

TOL = 1e-8


def _penalty_vector(state_size, T, C, test_day_num, penlty_s, penlty_l,
                    C_start, C_run):
    """目标函数罚项 (对应各构造函数中 f 的构造)."""
    C = np.asarray(C, dtype=float).ravel()
    penalty_temp1 = C_start * C                       # (state_size,)
    # x 块: t=1 为 +penalty_temp1, t=T 为 -penalty_temp1, 其余 0
    f_x = np.zeros(state_size * T)
    f_x[0:state_size] = penalty_temp1
    f_x[state_size * (T - 1):state_size * T] = -penalty_temp1
    # s 块: t=1 为 0, t>=2 为 2*penalty_temp1
    f_s = np.tile(penalty_temp1, T) * 2.0
    f_s[0:state_size] = 0.0

    f = np.concatenate([f_x, f_s,
                        np.zeros(2 * state_size * T),
                        np.ones(T), np.ones(T)])

    # penlty 2
    penlty_s = np.asarray(penlty_s, dtype=float)
    penlty_l = np.asarray(penlty_l, dtype=float).ravel()
    if penlty_s.ndim == 1:
        penlty_s = penlty_s.reshape(-1, 1)
    if test_day_num > 1:
        penlty_s = np.tile(penlty_s, (1, test_day_num))
    else:
        # MATLAB: penlty_s(:, 1:test_day_num*length(penlty_s)),
        # length() 取最大维度 (此处即列数 per_day), 不是行数
        penlty_s = penlty_s[:, 0:test_day_num * penlty_s.shape[1]]
    penlty_l_mat = np.tile(penlty_l.reshape(-1, 1), (1, T))
    penlty_s = (1 - penlty_s) * penlty_l_mat
    penlty_flat = (C_run * penlty_s).flatten(order='F')[:state_size * T]
    f[0:state_size * T] += penlty_flat
    return f


def _normalize(up, down, top, bottom):
    """P_up_wan / P_down_wan 归一化.

    MATLAB 原版对 top<bottom 直接 error; 但零功率状态的 DRO 拟合会出现
    beta0>0、beta1 微小负值的数值退化 (拟合端点被削到 0, 区间宽度 ~1e-5),
    这里按零宽度状态处理 (归一化 ramping 界被 max(...,0) 削为 0) 并告警。
    """
    denominator = np.asarray(top, dtype=float).ravel() - np.asarray(bottom, dtype=float).ravel()
    neg = denominator < -TOL
    if np.any(neg):
        print(f'warning: {int(np.count_nonzero(neg))} 个状态 P/Q_top < P/Q_bottom '
              '(零功率退化状态), 按零宽度处理。')
    up = np.asarray(up, dtype=float).ravel()
    down = np.asarray(down, dtype=float).ravel()
    up_wan = np.zeros_like(up, dtype=float)
    down_wan = np.zeros_like(down, dtype=float)
    valid = np.abs(denominator) >= TOL
    up_wan[valid] = up[valid] / denominator[valid]
    down_wan[valid] = down[valid] / denominator[valid]
    up_wan[valid] = np.maximum(up_wan[valid], 0)
    down_wan[valid] = np.maximum(down_wan[valid], 0)
    if np.any(up_wan[valid] > 1 + 1e-6):
        print('warning: Some normalized up values are greater than 1.')
    if np.any(down_wan[valid] > 1 + 1e-6):
        print('warning: Some normalized down values are greater than 1.')
    return up_wan, down_wan


def construct_tmilp_1bin_project(P_top, P_bottom, Q_top, Q_bottom, State_Limit,
                                 P_Sum, Q_Sum, P_up, P_down, Q_up, Q_down,
                                 Transition_model, C, T, S, test_day_num,
                                 penlty_s, penlty_l, C_start, C_run) -> MilpModel:
    S = np.asarray(S).ravel()
    P_top = np.asarray(P_top, dtype=float).ravel()
    P_bottom = np.asarray(P_bottom, dtype=float).ravel()
    P_Sum = np.asarray(P_Sum, dtype=float).ravel()
    State_Limit = np.asarray(State_Limit, dtype=float).ravel().copy()
    has_Q = Q_Sum is not None and len(np.asarray(Q_Sum).ravel()) > 0

    P_up_wan, P_down_wan = _normalize(P_up, P_down, P_top, P_bottom)
    if has_Q:
        Q_top = np.asarray(Q_top, dtype=float).ravel()
        Q_bottom = np.asarray(Q_bottom, dtype=float).ravel()
        Q_Sum = np.asarray(Q_Sum, dtype=float).ravel()
        Q_up_wan, Q_down_wan = _normalize(Q_up, Q_down, Q_top, Q_bottom)

    N = len(S)
    state_size = int(np.sum(S))
    index_app = index_appliance(S)

    n_x = state_size * T
    x_loc = np.arange(0, n_x)
    s_loc = np.arange(n_x, 2 * n_x)
    P_loc = np.arange(2 * n_x, 3 * n_x)
    Q_loc = np.arange(3 * n_x, 4 * n_x)
    kP_loc = np.arange(4 * n_x, 4 * n_x + T)
    kQ_loc = np.arange(4 * n_x + T, 4 * n_x + 2 * T)
    total_var = 4 * n_x + 2 * T
    f = _penalty_vector(state_size, T, C, test_day_num, penlty_s, penlty_l,
                        C_start, C_run)

    # ---- c2: 归一化有功平衡 (kesi) ----
    acc = RowAccumulator()
    acc.add(np.repeat(np.arange(T), state_size), x_loc, np.tile(P_bottom, T))
    acc.add(np.repeat(np.arange(T), state_size), P_loc, np.tile(P_top - P_bottom, T))
    acc.add(np.arange(T), kP_loc, -np.ones(T))
    A2_lower = acc.csr(T, total_var)
    b2_lower = P_Sum
    acc = RowAccumulator()
    acc.add(np.repeat(np.arange(T), state_size), x_loc, -np.tile(P_bottom, T))
    acc.add(np.repeat(np.arange(T), state_size), P_loc, -np.tile(P_top - P_bottom, T))
    acc.add(np.arange(T), kP_loc, -np.ones(T))
    A2_upper = acc.csr(T, total_var)
    b2_upper = -P_Sum

    # ---- c3: 归一化无功平衡 (kesi) ----
    if has_Q:
        acc = RowAccumulator()
        acc.add(np.repeat(np.arange(T), state_size), x_loc, np.tile(Q_bottom, T))
        acc.add(np.repeat(np.arange(T), state_size), Q_loc, np.tile(Q_top - Q_bottom, T))
        acc.add(np.arange(T), kQ_loc, -np.ones(T))
        A3_lower = acc.csr(T, total_var)
        b3_lower = Q_Sum
        acc = RowAccumulator()
        acc.add(np.repeat(np.arange(T), state_size), x_loc, -np.tile(Q_bottom, T))
        acc.add(np.repeat(np.arange(T), state_size), Q_loc, -np.tile(Q_top - Q_bottom, T))
        acc.add(np.arange(T), kQ_loc, -np.ones(T))
        A3_upper = acc.csr(T, total_var)
        b3_upper = -Q_Sum
    else:
        A3_lower = A3_upper = None
        b3_lower = b3_upper = None

    # ---- c4 ----
    m = state_size * (T - 1)
    r = np.arange(m)
    acc = RowAccumulator()
    acc.add(r, x_loc[state_size:], np.ones(m))
    acc.add(r, x_loc[:m], -np.ones(m))
    acc.add(r, s_loc[state_size:], -np.ones(m))
    A4 = acc.csr(m, total_var)
    b4 = np.zeros(m)

    # ---- TMILP 状态约束 ----
    acc = RowAccumulator()
    acc.add(r, x_loc[state_size:], -np.ones(m))
    acc.add(r, s_loc[state_size:], np.ones(m))
    acc.add(r, x_loc[:m], np.ones(m))
    acc.add(r, s_loc[state_size:], np.ones(m))
    A_state = acc.csr(2 * m, total_var)
    b_state = np.concatenate([np.zeros(m), np.ones(m)])

    # ---- c5: 最小运行时间: -x(i,t) + sum_{tau 最近 limit 个时段} s(i,tau) <= 0 ----
    # x / s 必须分开登记系数: 二者在 rows/cols 中按设备交错排列,
    # 用一个拼接数组按位置对齐会把符号搞错 (State_Limit 全为 1 时即暴露为
    # x + s <= 0, 与每个设备恰好一个状态的等式冲突导致模型不可行)。
    limit = np.minimum(State_Limit, T).astype(int)
    acc = RowAccumulator()
    xr, xc, sr, sc = [], [], [], []
    for i in range(state_size):
        base = i * T
        for j in range(T):
            xr.append(base + j)
            xc.append(x_loc[i + j * state_size])
        for j in range(T):
            tau0 = max(j - limit[i] + 1, 0)
            for tau in range(tau0, j + 1):
                sr.append(base + j)
                sc.append(s_loc[i + tau * state_size])
    acc.add(np.array(xr), np.array(xc), -np.ones(len(xr)))
    acc.add(np.array(sr), np.array(sc), np.ones(len(sc)))
    A5 = acc.csr(state_size * T, total_var)
    b5 = np.zeros(state_size * T)

    # ---- c7 ----
    acc = RowAccumulator()
    row = 0
    for t in range(T):
        for i in range(1, N + 1):
            states_i = np.flatnonzero(index_app == i)
            acc.add(np.full(len(states_i), row),
                    x_loc[states_i + t * state_size], np.ones(len(states_i)))
            row += 1
    A7 = acc.csr(N * T, total_var)
    b7 = np.ones(N * T)

    # ---- c8: 0 <= Pijt <= x ----
    m2 = n_x
    acc = RowAccumulator()
    acc.add(np.arange(m2), x_loc, -np.ones(m2))
    acc.add(np.arange(m2), P_loc, np.ones(m2))
    A8_1 = acc.csr(m2, total_var)
    acc = RowAccumulator()
    acc.add(np.arange(m2), P_loc, -np.ones(m2))
    A8_2 = acc.csr(m2, total_var)
    b8 = np.zeros(m2)

    if has_Q:
        acc = RowAccumulator()
        acc.add(np.arange(m2), x_loc, -np.ones(m2))
        acc.add(np.arange(m2), Q_loc, np.ones(m2))
        A9_1 = acc.csr(m2, total_var)
        acc = RowAccumulator()
        acc.add(np.arange(m2), Q_loc, -np.ones(m2))
        A9_2 = acc.csr(m2, total_var)
        b9 = np.zeros(m2)
    else:
        A9_1 = A9_2 = None
        b9 = None

    # ---- c12/c13: 归一化 ramping ----
    m3 = state_size * (T - 1)
    r3 = np.arange(m3)
    ii = r3 % state_size
    acc = RowAccumulator()
    acc.add(r3, x_loc[state_size:], -P_up_wan[ii])
    acc.add(r3, s_loc[state_size:], -np.ones(m3))
    acc.add(r3, P_loc[:m3], -np.ones(m3))
    acc.add(r3, P_loc[state_size:], np.ones(m3))
    A12 = acc.csr(m3, total_var)
    b12 = np.zeros(m3)

    P_top_sub = np.maximum(1 - P_down_wan, 0)
    acc = RowAccumulator()
    acc.add(r3, x_loc[state_size:], P_top_sub[ii])
    acc.add(r3, x_loc[:m3], -np.ones(m3))
    acc.add(r3, s_loc[state_size:], -P_top_sub[ii])
    acc.add(r3, P_loc[:m3], -np.ones(m3))
    acc.add(r3, P_loc[state_size:], np.ones(m3))
    A13 = acc.csr(m3, total_var)
    b13 = np.zeros(m3)

    if has_Q:
        acc = RowAccumulator()
        acc.add(r3, x_loc[state_size:], -Q_up_wan[ii])
        acc.add(r3, s_loc[state_size:], -np.ones(m3))
        acc.add(r3, Q_loc[:m3], -np.ones(m3))
        acc.add(r3, Q_loc[state_size:], np.ones(m3))
        A14 = acc.csr(m3, total_var)
        b14 = np.zeros(m3)
        Q_top_sub = np.maximum(1 - Q_down_wan, 0)
        acc = RowAccumulator()
        acc.add(r3, x_loc[state_size:], Q_top_sub[ii])
        acc.add(r3, x_loc[:m3], -np.ones(m3))
        acc.add(r3, s_loc[state_size:], -Q_top_sub[ii])
        acc.add(r3, Q_loc[:m3], -np.ones(m3))
        acc.add(r3, Q_loc[state_size:], np.ones(m3))
        A15 = acc.csr(m3, total_var)
        b15 = np.zeros(m3)
    else:
        A14 = A15 = None
        b14 = b15 = None

    # ---- 状态转换约束 ----
    acc = RowAccumulator()
    row = 0
    for i in range(N):
        tm = np.asarray(Transition_model[i])
        rr, cc = np.where(tm == 0)
        if len(rr) == 0:
            continue
        states_i = np.flatnonzero(index_app == i + 1)
        for t in range(T - 1):
            acc.add(np.full(len(rr), row), x_loc[states_i[rr] + t * state_size], np.ones(len(rr)))
            acc.add(np.full(len(rr), row), x_loc[states_i[cc] + (t + 1) * state_size], np.ones(len(rr)))
            row += 1
    A_trans = acc.csr(row, total_var)
    b_trans = np.ones(row)

    Aineq = sparse.vstack([b for b in [A2_lower, A2_upper, A3_lower, A3_upper, A4,
                                       A_state, A5, A8_1, A8_2, A9_1, A9_2,
                                       A12, A13, A14, A15, A_trans]
                           if b is not None and b.shape[0] > 0], format='csr')
    bineq = np.concatenate([b for b in [b2_lower, b2_upper, b3_lower, b3_upper, b4,
                                        b_state, b5, b8, b8, b9, b9, b12, b13,
                                        b14 if has_Q else None, b15 if has_Q else None,
                                        b_trans]
                            if b is not None])

    model = MilpModel(f=f, Aineq=Aineq, bineq=bineq, Aeq=A7, beq=b7,
                      lb=np.zeros(total_var),
                      ub=np.concatenate([np.ones(2 * state_size * T),
                                         np.full(2 * state_size * T + 2 * T, np.inf)]))
    model.ctype = 'B' * (state_size * T) + 'C' * (state_size * T * 3 + T * 2)
    return model


def decompose_power(device_state_vec, P_value_vec, S, T):
    """主函数中的 Decompose_data 段.

    device_state_vec / P_value_vec 为按 F 序排列的 state_size*T 向量,
    返回 list[ np.ndarray(T,) ] —— 各设备各时刻的分解功率.
    """
    S = np.asarray(S).ravel()
    state_size = int(np.sum(S))
    X = np.asarray(device_state_vec).reshape(T, state_size).T   # (state_size, T)
    P = np.asarray(P_value_vec).reshape(T, state_size).T
    out = []
    p1 = 0
    for i in range(len(S)):
        n_i = int(S[i])
        out.append(np.sum(X[p1:p1 + n_i, :] * P[p1:p1 + n_i, :], axis=0))
        p1 += n_i
    return out
