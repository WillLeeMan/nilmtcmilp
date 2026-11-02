# -*- coding: utf-8 -*-
"""数据预处理模块 (仅保留 TMILP-1bin-project 所需的 EMILP 流程).

- DataProcessing_EMILP.m -> data_processing_emilp()

原 MATLAB 函数返回大量输出参数, Python 版本统一用 dataclass 打包返回,
字段名与 MATLAB 输出名一致。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .dro_lr import dro_lr, dro_lr_fixB, dro_v1
from .utils import (all_equal_window, index_appliance, medfilt1)


@dataclass
class EmilpData:
    """DataProcessing_EMILP 的全部输出."""
    P: Optional[np.ndarray] = None
    P_top: Optional[np.ndarray] = None
    P_bottom: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    Q_top: Optional[np.ndarray] = None
    Q_bottom: Optional[np.ndarray] = None
    State_Limit: Optional[np.ndarray] = None
    P_test: Optional[np.ndarray] = None
    Q_test: Optional[np.ndarray] = None
    P_Sum: Optional[np.ndarray] = None
    Q_Sum: Optional[np.ndarray] = None
    P_up: Optional[np.ndarray] = None
    P_down: Optional[np.ndarray] = None
    Q_up: Optional[np.ndarray] = None
    Q_down: Optional[np.ndarray] = None
    Transition_model: List[np.ndarray] = field(default_factory=list)
    Omega: Optional[np.ndarray] = None
    P_train: Optional[np.ndarray] = None
    Q_train: Optional[np.ndarray] = None
    probabilities_On: Optional[np.ndarray] = None       # MATLAB 主函数中 penlty_s
    proportion_used_applicance: Optional[np.ndarray] = None  # 主函数中 penlty_l
    State_max_Limit: Optional[np.ndarray] = None


def _state_sort_and_split(P_train, Q_train, cluster_id, S):
    """聚类标签按平均功率排序, 并切分各状态的功率序列 (对应各 DataProcessing 的公共部分).

    返回 (P_State, P_State_sort_origin, P_State_unsort_origin,
           Q_State_sort_origin, Q_State_unsort_origin).
    """
    N = len(S)
    P_State, P_sort, P_unsort = [], [], []
    Q_sort, Q_unsort = [], []
    has_Q = Q_train is not None and Q_train.size > 0
    for i in range(N):
        result_idx = cluster_id[:, i]
        P_temp = np.array([np.mean(P_train[result_idx == j, i]) for j in range(1, S[i] + 1)])
        order = np.argsort(P_temp)                 # 平均功率升序对应的旧标签-1
        # relation(j) = 旧标签 j 的新编号
        relation = np.empty(S[i], dtype=int)
        relation[order] = np.arange(1, S[i] + 1)
        result_idx_new = relation[result_idx - 1]
        P_State.append(result_idx_new)
        for j in range(1, S[i] + 1):
            mask = result_idx_new == j
            P_sort.append(np.sort(P_train[mask, i]))
            P_unsort.append(P_train[mask, i])
            if has_Q:
                Q_sort.append(np.sort(Q_train[mask, i]))
                Q_unsort.append(Q_train[mask, i])
    return P_State, P_sort, P_unsort, Q_sort, Q_unsort


def _run_length_list(P_State_days, S):
    """窗口法统计各状态运行时长 (对应 MATLAB State_limit_list, 已修正原版错误).

    P_State_days: 每电器一个"逐日状态序列列表" (第 i 个电器 = 长度为
    train_day_num 的数组列表) —— 逐日独立统计, 不产生跨午夜的虚假运行段。

    相对 MATLAB 原版 DataProcessing_EMILP.m 的修正:
    - 运行段长度 = j - cur_index (0-based 半开区间, 原版 +1 多计);
    - 末尾未再转换的运行段现在被记录 (原版遗漏, 且该段往往是最长的一段);
    - 不再依赖占位 0, _compute_state_limits 使用全部记录 (原版的 (2:end)
      是跳过占位 0, Python 版此前无占位却仍跳过, 等于误删每状态第一段)。
    """
    state_size = int(np.sum(S))
    state_limit_list = [[] for _ in range(state_size)]
    p1 = 0
    for i in range(len(S)):
        for P_temp in P_State_days[i]:
            cur_state = P_temp[0]
            cur_index = 0
            for j in range(1, len(P_temp)):
                if cur_state != P_temp[j]:
                    state_limit_list[p1 + cur_state - 1].append(j - cur_index)
                    cur_state = P_temp[j]
                    cur_index = j
            state_limit_list[p1 + cur_state - 1].append(len(P_temp) - cur_index)
        p1 += S[i]
    return state_limit_list


def _build_transition_model(P_State_days, S, State_Limit, train_rows):
    """状态转换矩阵 (对应各 DataProcessing 中的 Transition_model 部分).

    逐日独立统计状态变化 (不产生跨午夜的虚假转移边), 各日结果取并集;
    W = train_rows / 全部状态变化次数 (与原版口径一致)。
    """
    index_app = index_appliance(S)
    Transition_model = []
    W = []
    for i in range(len(S)):
        trans = np.eye(S[i])
        idx_states = np.flatnonzero(index_app == i + 1)
        n_changes = 0
        for P_state_temp in P_State_days[i]:
            state_change_location = np.flatnonzero(np.diff(P_state_temp) != 0) + 1  # 0-based
            n_changes += len(state_change_location)
            for j in range(len(state_change_location)):
                p1 = state_change_location[j] - 1
                p2 = state_change_location[j]
                T1 = int(State_Limit[idx_states[P_state_temp[p1] - 1]])
                T2 = int(State_Limit[idx_states[P_state_temp[p2] - 1]])
                if p1 - T1 >= 0 and p2 + T2 <= len(P_state_temp):
                    window1 = P_state_temp[p1 - T1:p1]
                    window2 = P_state_temp[p2:p2 + T2]
                    if all_equal_window(window1) and all_equal_window(window2):
                        s1 = P_state_temp[p1]
                        s2 = P_state_temp[p2]
                        trans[s1 - 1, s2 - 1] = 1
                    else:
                        if not all_equal_window(window1) and not all_equal_window(window2):
                            window = medfilt1(np.concatenate([window1, window2]))
                            window1 = window[:len(window1)]
                            window2 = window[len(window1):]
                        else:
                            if not all_equal_window(window1):
                                window1 = medfilt1(window1)
                            if not all_equal_window(window2):
                                window2 = medfilt1(window2)
                        s1 = window1[-1]
                        s2 = window2[0]
                        trans[int(s1) - 1, int(s2) - 1] = 1
        W.append(train_rows / max(n_changes, 1))
        Transition_model.append(trans)
    return Transition_model, W


def _compute_state_limits(state_limit_list, eta, use_fixB=True):
    """DRO 回归估计最小/最大运行时间 (DataProcessing_EMILP 版本)."""
    state_size = len(state_limit_list)
    State_Limit = np.zeros(state_size)
    State_max_Limit = np.zeros(state_size)
    for i in range(state_size):
        temp = np.array(state_limit_list[i], dtype=float)   # 修正版列表无占位 0, 全部使用
        temp = np.sort(temp)
        if temp.size == 0:
            temp = np.array([1.0])
        if use_fixB:
            state_beta1 = dro_lr_fixB(np.vstack([np.ones(len(temp)), np.arange(1, len(temp) + 1)]),
                                      temp, eta, np.min(temp))
            max_value = len(temp) * state_beta1[1] + state_beta1[0]
            state_limit_new = temp[temp <= max_value]
        else:
            state_limit_new = temp
        if state_limit_new.size == 0:
            state_limit_new = temp
        state_beta = dro_v1(state_limit_new, 0.01, eta)
        state_min = state_beta[0]
        state_max = state_beta[0] + len(temp) * state_beta[1]
        if state_min <= 0:
            state_min = np.min(state_limit_new)
        State_Limit[i] = state_min
        State_max_Limit[i] = state_max
    return np.ceil(State_Limit), np.ceil(State_max_Limit)


def _on_state_counts(P_State, S, train_data_num, train_day_num, always_on):
    """On_state_num / On_appliance_num 统计 (EMILP 版本)."""
    per_day = train_data_num // train_day_num
    state_size = int(np.sum(S))
    N = len(S)
    On_state_num = np.zeros((state_size, per_day))
    On_appliance_num = np.zeros((N, per_day))
    for i in range(N):
        state_data_interval = P_State[i].reshape(train_day_num, per_day)
        for t in range(per_day):
            for j in range(1, S[i] + 1):
                idx = np.flatnonzero(state_data_interval[:, t] == j)
                row = (j - 1) if i == 0 else (int(np.sum(S[:i])) + j - 1)
                On_state_num[row, t] = len(idx)
            if always_on[i] == 1:
                idx = np.flatnonzero(state_data_interval[:, t] > 0)
            else:
                idx = np.flatnonzero(state_data_interval[:, t] > 1)
            On_appliance_num[i, t] = len(idx)
    return On_state_num, On_appliance_num


def data_processing_emilp(train_data, test_data, select_device, select_data_type,
                          select_data_set, S, cluster_id, train_data_num,
                          test_data_num, train_day_num, always_on_appliance) -> EmilpData:
    """DataProcessing_EMILP.m 的 Python 版本."""
    S = np.asarray(S).ravel()
    always_on_appliance = np.asarray(always_on_appliance).ravel()
    N = len(select_device)
    train_rows = train_data_num
    index_app = index_appliance(S)
    ds = select_data_set[0]

    P_train = Q_train = None
    P_test = Q_test = None
    P_Sum = np.array([])
    Q_Sum = np.array([])
    for dt in select_data_type:
        if dt == 'P':
            P_train = train_data[ds]['appliance_P']
            P_test = test_data[ds]['appliance_P']
            P_Sum = np.sum(P_test, axis=1)
        elif dt == 'Q':
            Q_train = train_data[ds]['appliance_Q']
            Q_test = test_data[ds]['appliance_Q']
            Q_Sum = np.sum(Q_test, axis=1)
        else:
            print('no exist this data type!')

    state_size = int(np.sum(S))
    P_State, P_sort, P_unsort, Q_sort, Q_unsort = _state_sort_and_split(
        P_train, Q_train, cluster_id, S)

    # 鲁棒回归求功率上下界
    eta = 0.5
    P = np.zeros(state_size); P_top = np.zeros(state_size); P_bottom = np.zeros(state_size)
    Q = np.zeros(state_size); Q_top = np.zeros(state_size); Q_bottom = np.zeros(state_size)
    for i in range(state_size):
        x = np.vstack([np.ones(len(P_sort[i])), np.arange(1, len(P_sort[i]) + 1)])
        P_beta = dro_lr(x, P_sort[i], eta)
        P_len = len(P_sort[i])
        P_min = P_beta[0]; P_max = P_beta[0] + P_len * P_beta[1]
        P_min = max(P_min, 0); P_max = max(P_max, 0)
        P_bottom[i] = P_min; P_top[i] = P_max; P[i] = (P_max + P_min) / 2
        if Q_train is not None and Q_train.size > 0:
            Q_beta = dro_lr(x, Q_sort[i], eta)
            Q_len = len(Q_sort[i])
            Q_min = Q_beta[0]; Q_max = Q_beta[0] + Q_len * Q_beta[1]
            Q_min = max(Q_min, 0); Q_max = max(Q_max, 0)
            Q_bottom[i] = Q_min; Q_top[i] = Q_max; Q[i] = (Q_max + Q_min) / 2

    # 最小运行时间 (逐日统计, 不产生跨午夜虚假运行段/转移边)
    per_day = train_data_num // train_day_num
    P_State_days = [np.array_split(P_State[i], train_day_num) for i in range(N)]
    state_limit_list = _run_length_list(P_State_days, S)
    State_Limit, State_max_Limit = _compute_state_limits(state_limit_list, eta=100)

    # 状态内功率波动 (ramping) 上界 — 逐日归集, 差分不跨午夜
    P_up = []; P_down = []; Q_up = []; Q_down = []
    for i in range(N):
        P_train_temp = P_train[:, i]
        P_state_temp = P_State[i]
        Q_train_temp = Q_train[:, i] if (Q_train is not None and Q_train.size > 0) else None
        for j in range(1, S[i] + 1):
            mask = P_state_temp == j
            P_state_j = P_train_temp[mask]
            # 逐日切分: 每天的差分只在该日连续段内计算
            pos_all, neg_all = [], []
            for d0 in range(0, train_data_num, per_day):
                seg = P_train_temp[d0:d0 + per_day][mask[d0:d0 + per_day]]
                if seg.size < 2:
                    continue
                sub = np.diff(seg)
                pos_all.append(sub[sub > 0])
                neg_all.append(-sub[sub < 0])
            P_pos = np.sort(np.concatenate(pos_all)) if pos_all else np.array([0.0])
            P_neg = np.sort(np.concatenate(neg_all)) if neg_all else np.array([0.0])
            if P_pos.size == 0:
                P_pos = np.array([0.0])
            if P_neg.size == 0:
                P_neg = np.array([0.0])
            beta_up = dro_lr(np.vstack([np.ones(len(P_pos)), np.arange(1, len(P_pos) + 1)]), P_pos, eta)
            beta_dn = dro_lr(np.vstack([np.ones(len(P_neg)), np.arange(1, len(P_neg) + 1)]), P_neg, eta)
            P_up.append(beta_up[0] + len(P_pos) * beta_up[1])
            P_down.append(beta_dn[0] + len(P_neg) * beta_dn[1])
            if Q_train_temp is not None:
                # 修正: MATLAB 原版此处误用 P_state_j 计算 Q ramping (已证实的
                # 源代码 bug), 现改用该状态的真实 Q 样本 (与 P 相同的逐日切分)
                pos_q, neg_q = [], []
                for d0 in range(0, train_data_num, per_day):
                    m = mask[d0:d0 + per_day]
                    seg = Q_train_temp[d0:d0 + per_day][m]
                    if seg.size < 2:
                        continue
                    sub = np.diff(seg)
                    pos_q.append(sub[sub > 0])
                    neg_q.append(-sub[sub < 0])
                Q_pos = np.sort(np.concatenate(pos_q)) if pos_q else np.array([0.0])
                Q_neg = np.sort(np.concatenate(neg_q)) if neg_q else np.array([0.0])
                if Q_pos.size == 0:
                    Q_pos = np.array([0.0])
                if Q_neg.size == 0:
                    Q_neg = np.array([0.0])
                beta_up_q = dro_lr(np.vstack([np.ones(len(Q_pos)), np.arange(1, len(Q_pos) + 1)]), Q_pos, eta)
                beta_dn_q = dro_lr(np.vstack([np.ones(len(Q_neg)), np.arange(1, len(Q_neg) + 1)]), Q_neg, eta)
                Q_up.append(beta_up_q[0] + len(Q_pos) * beta_up_q[1])
                Q_down.append(beta_dn_q[0] + len(Q_neg) * beta_dn_q[1])

    P_up = np.ceil(P_up).reshape(-1, 1)
    P_down = np.ceil(P_down).reshape(-1, 1)
    if Q_up:
        Q_up = np.ceil(Q_up).reshape(-1, 1)
        Q_down = np.ceil(Q_down).reshape(-1, 1)
    else:
        Q_up = np.array([]); Q_down = np.array([])

    # 状态转换矩阵 (逐日统计)
    Transition_model, W = _build_transition_model(P_State_days, S, State_Limit, train_rows)

    # 状态/设备开启概率统计
    On_state_num, On_appliance_num = _on_state_counts(
        P_State, S, train_data_num, train_day_num, always_on_appliance)
    probabilities_On = On_state_num / train_day_num
    proportion_used = train_data_num / np.sum(On_state_num, axis=1, keepdims=True)

    Omega = np.zeros((state_size, 1))
    p1 = 0
    for i in range(N):
        Omega[p1:p1 + S[i], 0] = W[i]
        p1 += S[i]

    print('State_Limit done!')
    return EmilpData(P=P, P_top=P_top, P_bottom=P_bottom, Q=Q if Q_train is not None else np.array([]),
                     Q_top=Q_top if Q_train is not None else np.array([]),
                     Q_bottom=Q_bottom if Q_train is not None else np.array([]),
                     State_Limit=State_Limit, P_test=P_test, Q_test=Q_test,
                     P_Sum=P_Sum, Q_Sum=Q_Sum,
                     P_up=P_up, P_down=P_down, Q_up=Q_up, Q_down=Q_down,
                     Transition_model=Transition_model, Omega=Omega,
                     P_train=P_train, Q_train=Q_train,
                     probabilities_On=probabilities_On,
                     proportion_used_applicance=proportion_used,
                     State_max_Limit=State_max_Limit)
