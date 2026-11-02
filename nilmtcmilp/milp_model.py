# -*- coding: utf-8 -*-
"""MILP/MIQP model container, mirrors the MATLAB model struct.

MATLAB 中的 model 结构体字段:
    f, Aineq, bineq, Aeq, beq, lb, ub, ctype, H(Q), Qc, l, r,
    variable_location (struct)

本 Python 版本使用 dataclass 保存相同内容:
    - f        : 目标函数一次项 (n,)
    - Q        : 目标函数二次项 (H), scipy sparse, 对应 x'Qx
    - Aineq/bineq, Aeq/beq : 线性约束
    - lb/ub    : 变量界
    - ctype    : 长度为 n 的字符串, 'B' 二进制, 'I' 整数, 'C' 连续
    - Qc/l/r   : 二次约束 x'Qc{i}x + l(:,i)'x <= r(i) (list)
    - variable_location : dict, MATLAB 中 variable_location 结构体
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse


@dataclass
class MilpModel:
    f: Optional[np.ndarray] = None
    Aineq: Optional[sparse.csr_matrix] = None
    bineq: Optional[np.ndarray] = None
    Aeq: Optional[sparse.csr_matrix] = None
    beq: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    ctype: Optional[str] = None            # 长度为 n 的字符数组, 如 'BBBC...'
    Q: Optional[sparse.csr_matrix] = None  # MATLAB model.Q / model.H
    Qc: Optional[List[sparse.csr_matrix]] = None   # MATLAB model.Qc (cell)
    l: Optional[np.ndarray] = None         # MATLAB model.l (n × k)
    r: Optional[np.ndarray] = None         # MATLAB model.r (k,)
    H: Optional[sparse.csr_matrix] = None  # MATLAB model.H (与 Q 同义)
    variable_location: Dict[str, np.ndarray] = field(default_factory=dict)

    def num_vars(self) -> int:
        if self.f is not None:
            return int(self.f.shape[0])
        if self.lb is not None:
            return int(self.lb.shape[0])
        if self.ctype is not None:
            return len(self.ctype)
        return 0

    def model_size(self):
        """对应 MATLAB: [变量数, 约束行数, 非零元个数]"""
        n = self.num_vars()
        rows = 0
        nnz = 0
        for A, b in ((self.Aineq, self.bineq), (self.Aeq, self.beq)):
            if A is not None and A.shape[0] > 0:
                rows += A.shape[0]
                nnz += A.nnz
        return [n, rows, nnz]


class RowAccumulator:
    """高效构造稀疏约束矩阵 (等价于 MATLAB 中逐步 sparse([r],[c],v,m,n) 累加)."""

    def __init__(self):
        self.rows: List[np.ndarray] = []
        self.cols: List[np.ndarray] = []
        self.vals: List[np.ndarray] = []

    def add(self, rows, cols, vals):
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        vals = np.asarray(vals, dtype=np.float64).ravel()
        if rows.size == 0:
            return
        self.rows.append(rows)
        self.cols.append(cols)
        self.vals.append(vals)

    def csr(self, n_rows: int, n_cols: int) -> sparse.csr_matrix:
        if not self.rows:
            return sparse.csr_matrix((n_rows, n_cols))
        r = np.concatenate(self.rows)
        c = np.concatenate(self.cols)
        v = np.concatenate(self.vals)
        return sparse.csr_matrix((v, (r, c)), shape=(n_rows, n_cols))


def vstack_csr(matrices, n_cols: int) -> sparse.csr_matrix:
    """竖向拼接稀疏矩阵, 过滤空矩阵 (对应 MATLAB [A;B;C])."""
    mats = [m for m in matrices if m is not None and m.shape[0] > 0]
    if not mats:
        return sparse.csr_matrix((0, n_cols))
    return sparse.vstack(mats, format="csr")


def hstack_csr(matrices, n_rows: int) -> sparse.csr_matrix:
    """横向拼接稀疏矩阵 (对应 MATLAB [A,B,C])."""
    mats = [m for m in matrices if m is not None and m.shape[1] > 0]
    if not mats:
        return sparse.csr_matrix((n_rows, 0))
    return sparse.hstack(mats, format="csr")
