# -*- coding: utf-8 -*-
"""evaluation_func.m 的 Python 版本.

返回 (RSE, R_square, AC, h_predict, h_true, h_error, OCC, FS).
输入的 true_list / predict_list 为 list[ np.ndarray(T,) ].
"""

import numpy as np

from .clustering import kmeans_labels


def _column(x):
    return np.asarray(x, dtype=float).ravel()


def calculate_RSE(true_list, predict_list, N):
    out = np.zeros((N, 1))
    for i in range(N):
        t = _column(true_list[i]); p = _column(predict_list[i])
        if np.sum(t ** 2) != 0:
            out[i, 0] = np.sum((t - p) ** 2) / np.sum(t ** 2)
        elif np.sum(p ** 2) != 0:
            out[i, 0] = np.sum((t - p) ** 2) / np.sum(p ** 2)
    return out


def calculate_R_square(true_list, predict_list, N):
    out = np.zeros((N, 1))
    for i in range(N):
        t = _column(true_list[i]); p = _column(predict_list[i])
        if np.sum(t ** 2) != 0:
            avg = np.mean(t)
            out[i, 0] = 1 - np.sum((t - p) ** 2) / np.sum((t - avg) ** 2)
        elif np.sum(p ** 2) != 0:
            avg = np.mean(p)
            out[i, 0] = 1 - np.sum((p - t) ** 2) / np.sum((p - avg) ** 2)
        else:
            out[i, 0] = 1
    return out


def calculate_AC(true_list, predict_list, N):
    out = np.zeros((N, 1))
    for i in range(N):
        t = _column(true_list[i]); p = _column(predict_list[i])
        if np.sum(t) != 0:
            out[i, 0] = 1 - np.sum(np.abs(t - p)) / (2 * np.sum(t))
        elif np.sum(p) != 0:
            out[i, 0] = 1 - np.sum(np.abs(t - p)) / (2 * np.sum(p))
        else:
            out[i, 0] = 1
    return out


def calculate_h(predict_list, N):
    bottom = sum(np.sum(_column(p)) for p in predict_list)
    out = np.zeros((N, 1))
    if bottom != 0:
        for i in range(N):
            out[i, 0] = np.sum(_column(predict_list[i])) / bottom
    return out


def calculate_h_error(h_true, h_predict, N):
    out = np.zeros((N, 1))
    for i in range(N):
        if h_true[i, 0] != 0:
            out[i, 0] = abs(h_true[i, 0] - h_predict[i, 0]) / h_true[i, 0]
        elif h_predict[i, 0] != 0:
            out[i, 0] = abs(h_true[i, 0] - h_predict[i, 0]) / h_predict[i, 0]
    return out


def calculate_OCC(true_list, predict_list):
    true_mat = np.column_stack([_column(t) for t in true_list])
    pred_mat = np.column_stack([_column(p) for p in predict_list])
    if np.sum(true_mat) != 0:
        diff = np.sum(np.abs(true_mat - pred_mat))
        return 1 - diff / (2 * np.sum(true_mat))
    if np.sum(pred_mat) == 0:
        return 1
    diff = np.sum(np.abs(true_mat - pred_mat))
    return 1 - diff / (2 * np.sum(pred_mat))


def calculate_FS(predict_x, predict_list, true_list, S, N, always_on_appliance):
    """FS (分状态误差). predict_x 为 (state_size, T) 的 0-1 状态矩阵."""
    S = np.asarray(S).ravel()
    always_on_appliance = np.asarray(always_on_appliance).ravel()
    predict_x = np.asarray(predict_x).T    # MATLAB: predict_x=predict_x' → (T, state_size)
    T = predict_x.shape[0]
    if predict_x.shape[1] == int(np.sum(S)):
        true_x = np.zeros((T, N))
        for i in range(N):
            labels, _ = kmeans_labels(_column(true_list[i]), S[i])
            # 重编号: 标签即排名
            for j in range(1, S[i] + 1):
                col = (j - 1) if i == 0 else int(np.sum(S[:i])) + (j - 1)
                predict_x[predict_x[:, col] == 1, col] = j
            true_x[:, i] = labels
        ans = np.zeros((N, 1))
        for i in range(N):
            idx = np.flatnonzero(true_x[:, i] > 1)
            t = _column(true_list[i]); p = _column(predict_list[i])
            if np.sum(t[idx]) > 0:
                ans[i, 0] = np.sum(np.abs(t[idx] - p[idx])) / np.sum(t[idx])
        return ans
    else:
        S_temp = S - 1 + always_on_appliance
        true_x = np.zeros((T, N))
        for i in range(N):
            labels, _ = kmeans_labels(_column(true_list[i]), S[i])
            cluster_idx = labels.copy()
            for j in range(1, S[i] + 1):
                if always_on_appliance[i] == 1:
                    cluster_idx[labels == j] = j
                else:
                    cluster_idx[labels == j] = j - 1
            for j in range(1, int(S_temp[i]) + 1):
                col = (j - 1) if i == 0 else int(np.sum(S_temp[:i])) + (j - 1)
                predict_x[predict_x[:, col] == 1, col] = j
            true_x[:, i] = cluster_idx
        ans = np.zeros((N, 1))
        for i in range(N):
            idx = np.flatnonzero(true_x[:, i] > 0)
            t = _column(true_list[i]); p = _column(predict_list[i])
            if np.sum(t[idx]) > 0:
                ans[i, 0] = np.sum(np.abs(t[idx] - p[idx])) / np.sum(t[idx])
        return ans


def evaluation_func(true_list, predict_list, predict_x, S, always_on_appliance):
    """evaluation_func.m 的 Python 版本."""
    N = len(true_list)
    RSE = calculate_RSE(true_list, predict_list, N)
    R_square = calculate_R_square(true_list, predict_list, N)
    AC = calculate_AC(true_list, predict_list, N)
    h_predict = calculate_h(predict_list, N)
    h_true = calculate_h(true_list, N)
    h_error = calculate_h_error(h_true, h_predict, N)
    OCC = calculate_OCC(true_list, predict_list)
    FS = calculate_FS(predict_x, predict_list, true_list, S, N, always_on_appliance)
    return RSE, R_square, AC, h_predict, h_true, h_error, OCC, FS
