# -*- coding: utf-8 -*-
"""nilmtcmilp — TMILP-1bin-project (NILM) 的 Python 版本 (COPT 求解器).

当前仅保留研究主线 TMILP-1bin 归一化投影模型及所需基础设施;
其余模型 (EMILP / TMILP 1bin,2bin / TCMILP / CO / SO / OLDA / ALIP / BQP /
CMILP) 已移除, MATLAB 源码仍保留在 _m_src_utf8/ 供对照。
"""

from .milp_model import MilpModel, RowAccumulator, hstack_csr, vstack_csr
from .copt_solver import solve
from .utils import (all_equal_window, as_column, as_row, cell2mat_columns,
                    diff, fflatten, find, freshape, index_appliance, medfilt1,
                    repmat)
from .clustering import get_state_num_by_kmean, kmeans_labels
from .dro_lr import dro_lr, dro_lr_fixB, dro_v1, dro_v2
from .data_read import read_data
from .data_processing import EmilpData, data_processing_emilp
from .evaluation import evaluation_func
from .construct_tmilp_project import construct_tmilp_1bin_project, decompose_power

__all__ = [
    'MilpModel', 'RowAccumulator', 'hstack_csr', 'vstack_csr', 'solve',
    'all_equal_window', 'as_column', 'as_row', 'cell2mat_columns', 'diff',
    'fflatten', 'find', 'freshape', 'index_appliance', 'medfilt1', 'repmat',
    'get_state_num_by_kmean', 'kmeans_labels',
    'dro_lr', 'dro_lr_fixB', 'dro_v1', 'dro_v2',
    'read_data', 'EmilpData', 'data_processing_emilp', 'evaluation_func',
    'construct_tmilp_1bin_project', 'decompose_power',
]
