# -*- coding: utf-8 -*-
"""数据读取, 对应 function_read_data.m.

数据以嵌套 dict 保存, 对应 MATLAB 的 struct:
    train_data['AMPDS']['appliance_P']  <->  train_data.AMPDS.appliance_P

REFIT 数据集需要 appliance_information.mat (scipy.io.loadmat 读取).
"""

import os

import numpy as np


# 数据集元信息 (对应 function_read_data.m 中硬编码的表格)
AMPDS_ALL_ELECTRICITY = ['WHE', 'RSE', 'GRE', 'MHE', 'B1E', 'BME', 'CWE', 'DWE',
                         'EQE', 'FRE', 'HPE', 'OFE', 'UTE', 'WOE', 'B2E', 'CDE',
                         'DNE', 'EBE', 'FGE', 'HTE', 'OUE', 'TVE', 'UNE']
AMPDS_S = [3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 2, 4, 3, 2, 2, 2, 3, 0, 3, 3, 3, 2, 5]
AMPDS_ALWAYS_ON = [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

ALL_APPLIANCE_NAME = ['Television', 'Hi-Fi', 'Fridge-Freezer', 'Fridge', 'Freezer', 'Microwave',
                      'Cooker Hood', 'Kettle', 'Toaster', 'Misc Kitchen', 'Washing Machine',
                      'Washer Dryer', 'Tumble Dryer', 'Dishwasher', 'Computer', 'Router',
                      'Elec Heater', 'Lamp', 'Misc']
REFIT_S = [3, 3, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2]
REFIT_ALWAYS_ON = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# appliance_information.mat 所在的 REFIT 数据目录
REFIT_INFO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'REFIT')


def _read_matrix(path) -> np.ndarray:
    """MATLAB readmatrix 的替代 (跳过表头行, 逗号分隔)."""
    return np.genfromtxt(path, delimiter=',', skip_header=1)


def _load_monitored_appliances() -> np.ndarray:
    from scipy.io import loadmat
    mat_path = os.path.join(REFIT_INFO_DIR, 'appliance_information.mat')
    data = loadmat(mat_path)
    return data['Monitored_appliances']


def read_data(data_folder_path, select_data_set, select_data_type,
              select_appliance_name, train_data_num, test_data_num,
              train_data_start_location, test_data_start_location):
    """function_read_data 的 Python 版本.

    参数与 MATLAB 版本一致 (select_data_type: AMPDS 为 ['P','Q'];
    REFIT 为房屋名 cell, 如 ['CLEAN_House3']).
    返回 (Total_data, train_data, test_data, location, S, always_on_appliance).
    """
    Total_data = {}
    train_data = {}
    test_data = {}

    for ds in select_data_set:
        if ds == 'AMPDS':
            td, trd, ted, location, S, always_on = _read_ampds(
                data_folder_path, select_data_type, select_appliance_name,
                train_data_num, test_data_num, train_data_start_location,
                test_data_start_location)
            Total_data['AMPDS'] = td
            train_data['AMPDS'] = trd
            test_data['AMPDS'] = ted
        elif ds == 'test':
            td, trd, ted, location, S, always_on = _read_test(
                data_folder_path, select_data_type, select_appliance_name,
                train_data_num, test_data_num, train_data_start_location,
                test_data_start_location)
            Total_data['test'] = td
            train_data['test'] = trd
            test_data['test'] = ted
        elif ds == 'REFIT':
            td, trd, ted, location, S, always_on = _read_refit(
                data_folder_path, select_data_type, select_appliance_name,
                train_data_num, test_data_num, train_data_start_location,
                test_data_start_location)
            Total_data['REFIT'] = td
            train_data['REFIT'] = trd
            test_data['REFIT'] = ted
        else:
            print('Unexpected data Set. Please check data Set')
    return Total_data, train_data, test_data, location, S, always_on


def _read_ampds(data_folder_path, select_data_type, select_appliance_name,
                train_data_num, test_data_num, train_data_start_location,
                test_data_start_location):
    td = {}
    for dtype_ in select_data_type:
        if dtype_ == 'P':
            td['P'] = _read_matrix(os.path.join(data_folder_path, 'AMPDS', 'Electricity_P.csv'))
        elif dtype_ == 'Q':
            td['Q'] = _read_matrix(os.path.join(data_folder_path, 'AMPDS', 'Electricity_Q.csv'))
        else:
            print('no exist this data type!')
            break

    location = [AMPDS_ALL_ELECTRICITY.index(a) for a in select_appliance_name]

    s0 = train_data_start_location - 1
    trd, ted = {}, {}
    if 'P' in td:
        trd['appliance_P'] = td['P'][s0:s0 + train_data_num, np.array(location) + 1]
        trd['timeIndex'] = td['P'][s0:s0 + train_data_num, 0]
        ted['appliance_P'] = None
        ted['timeIndex'] = None
        for j in range(len(test_data_start_location)):
            t0 = int(test_data_start_location[j]) - 1
            col = td['P'][t0:t0 + test_data_num, location[j] + 1].reshape(-1, 1)
            col_t = td['P'][t0:t0 + test_data_num, 0].reshape(-1, 1)
            ted['appliance_P'] = col if ted['appliance_P'] is None else np.hstack([ted['appliance_P'], col])
            ted['timeIndex'] = col_t if ted['timeIndex'] is None else np.hstack([ted['timeIndex'], col_t])
    else:
        trd['appliance_P'] = np.array([])
        ted['appliance_P'] = np.array([])

    if 'Q' in td:
        trd['appliance_Q'] = td['Q'][s0:s0 + train_data_num, np.array(location) + 1]
        ted['appliance_Q'] = None
        for j in range(len(test_data_start_location)):
            t0 = int(test_data_start_location[j]) - 1
            col = td['Q'][t0:t0 + test_data_num, location[j] + 1].reshape(-1, 1)
            ted['appliance_Q'] = col if ted['appliance_Q'] is None else np.hstack([ted['appliance_Q'], col])
    else:
        trd['appliance_Q'] = np.array([])
        ted['appliance_Q'] = np.array([])

    S = [AMPDS_S[i] for i in location]
    always_on = [AMPDS_ALWAYS_ON[i] for i in location]
    return td, trd, ted, location, S, always_on


def _read_test(data_folder_path, select_data_type, select_appliance_name,
               train_data_num, test_data_num, train_data_start_location,
               test_data_start_location):
    td = {}
    for dtype_ in select_data_type:
        td[dtype_] = _read_matrix(os.path.join(data_folder_path, 'test', dtype_ + '.csv'))
    S = [4, 4, 2, 2, 5, 2, 2]
    always_on = [0] * 7
    trd, ted = {}, {}
    for dtype_ in select_data_type:
        s0 = train_data_start_location - 1
        trd['appliance_P'] = td[dtype_][s0:s0 + train_data_num, :]
        ted['appliance_P'] = None
        for k in range(len(test_data_start_location)):
            t0 = int(test_data_start_location[k]) - 1
            col = td[dtype_][t0:t0 + test_data_num, :]
            ted['appliance_P'] = col if ted['appliance_P'] is None else np.hstack([ted['appliance_P'], col])
    location = list(range(1, 8))
    return td, trd, ted, location, S, always_on


def _read_refit(data_folder_path, select_data_type, select_appliance_name,
                train_data_num, test_data_num, train_data_start_location,
                test_data_start_location):
    monitored = _load_monitored_appliances()   # (20 × 房屋数)

    S = []
    always_on = []
    index_location = []
    for j in range(len(select_appliance_name)):
        name = select_appliance_name[j][0] if isinstance(select_appliance_name[j], (list, tuple)) else \
            select_appliance_name[j]
        for k in range(len(ALL_APPLIANCE_NAME)):
            if ALL_APPLIANCE_NAME[k] == name:
                index_location.append(k)
                S.append(REFIT_S[k])
                always_on.append(REFIT_ALWAYS_ON[k])

    td = {}
    for dtype_ in select_data_type:
        td[dtype_] = _read_matrix(os.path.join(data_folder_path, 'REFIT', dtype_ + '.csv'))

    # MATLAB: index_appliances=find(Monitored_appliances(:,3)~=0); 第3列对应 House3
    house_col = 2  # 0-based, 对应 (:,3)
    index_appliances = np.flatnonzero(monitored[:, house_col] != 0)
    location = []
    for j in range(len(index_location)):
        loc = np.flatnonzero(index_appliances == index_location[j])
        location.append(int(loc[0]) if loc.size else None)

    trd, ted = {}, {}
    for dtype_ in select_data_type:
        s0 = train_data_start_location - 1
        trd['appliance_P'] = td[dtype_][s0:s0 + train_data_num, np.array(location) + 3]
        trd['timeIndex'] = td[dtype_][s0:s0 + train_data_num, 1]
        ted['appliance_P'] = None
        ted['timeIndex'] = None
        for k in range(len(test_data_start_location)):
            t0 = int(test_data_start_location[k]) - 1
            col = td[dtype_][t0:t0 + test_data_num, location[k] + 3].reshape(-1, 1)
            col_t = td[dtype_][t0:t0 + test_data_num, 0].reshape(-1, 1)
            ted['appliance_P'] = col if ted['appliance_P'] is None else np.hstack([ted['appliance_P'], col])
            ted['timeIndex'] = col_t if ted['timeIndex'] is None else np.hstack([ted['timeIndex'], col_t])
    return td, trd, ted, location, S, always_on
