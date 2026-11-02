# -*- coding: utf-8 -*-
"""通用工具函数: MATLAB 语义到 NumPy/SciPy 的映射."""

import numpy as np


def as_column(x, dtype=float) -> np.ndarray:
    """等价 MATLAB 列向量."""
    return np.asarray(x, dtype=dtype).reshape(-1, 1)


def as_row(x, dtype=float) -> np.ndarray:
    """等价 MATLAB 行向量."""
    return np.asarray(x, dtype=dtype).reshape(1, -1)


def freshape(vec, rows: int, cols: int) -> np.ndarray:
    """MATLAB reshape(vec, rows, cols) —— 列优先(Fortran 序)."""
    return np.asarray(vec).reshape(cols, rows).T


def fflatten(mat) -> np.ndarray:
    """MATLAB A(:) —— 列优先拉直."""
    return np.asarray(mat).ravel(order="F")


def index_appliance(S) -> np.ndarray:
    """对应 MATLAB: index_appliance=[i*ones(S(i),1)] 拼接."""
    S = np.asarray(S).ravel()
    return np.concatenate([np.full(int(s), i + 1) for i, s in enumerate(S)])


def find(cond) -> np.ndarray:
    """MATLAB find —— 返回 0-based 索引."""
    return np.flatnonzero(np.asarray(cond))


def medfilt1(x, k: int = 3) -> np.ndarray:
    """MATLAB medfilt1(x, k) 的等价实现 (默认 zeropad 端点处理)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0 or k <= 1:
        return x.copy()
    left = (k - 1) // 2
    right = k // 2
    xp = np.concatenate([np.zeros(left), x, np.zeros(right)])
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(xp[i:i + k])
    return out


def diff(x) -> np.ndarray:
    """MATLAB diff (列向量/数组按第一非单维差分)."""
    x = np.asarray(x)
    return np.diff(x, axis=0)


def all_equal_window(window) -> bool:
    """MATLAB all(~diff(window)) —— 窗口内元素是否全部相等."""
    w = np.asarray(window).ravel()
    return bool(np.all(w == w[0])) if w.size > 0 else True


def repmat(v, m: int, n: int) -> np.ndarray:
    """MATLAB repmat(列向量 v, m, n). v 为 (k,1) 或 (k,)."""
    v = np.asarray(v)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    return np.tile(v, (m, n))


def cell2mat_columns(cells) -> np.ndarray:
    """等价 MATLAB cell2mat —— 将列向量 cell 数组拼接为矩阵 (按行拼接 cell 元素)."""
    cols = [np.asarray(c).ravel() for c in cells]
    return np.column_stack(cols)


def matlab_ranges(start, stop):
    """MATLAB start:stop 的 0-based 索引数组 (闭区间)."""
    return np.arange(start - 1, stop)


def blk_rows_to_csr(blocks, n_cols):
    """将若干 (行数, n_cols) 的 csr 块竖向拼接, 空块跳过."""
    from scipy import sparse
    blocks = [b for b in blocks if b is not None and b.shape[0] > 0]
    if not blocks:
        return sparse.csr_matrix((0, n_cols))
    return sparse.vstack(blocks, format="csr")
