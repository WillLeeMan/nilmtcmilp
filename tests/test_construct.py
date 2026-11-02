"""模型构造 / 哈希校验 / 功率还原的冒烟测试。

不需要 COPT：只有在装了 coptpy 时才会跑求解测试。
"""

import importlib.util

import numpy as np

from nilmtcmilp import decompose_power
from nilmtcmilp.construct_tmilp_project import construct_tmilp_1bin_project
from nilmtcmilp.construct_tmilp_project_v3 import check_legacy_source

T = 24
PER_DAY = 24
S = np.array([2, 2])


def make_inputs():
    n = int(S.sum())
    P_top = np.array([100.0, 300.0, 50.0, 1500.0])
    P_bottom = np.array([80.0, 250.0, 0.0, 1400.0])
    State_Limit = np.array([5.0, 5.0, 1.0, 1.0])
    P_Sum = np.full(T, 200.0)
    P_up = np.full((n, 1), 50.0)
    P_down = np.full((n, 1), 50.0)
    Transition_model = [np.ones((2, 2)), np.ones((2, 2))]
    C = np.ones((n, 1))
    penlty_s = np.zeros((n, PER_DAY))
    penlty_l = np.ones((n, 1))
    empty = np.array([])
    return dict(
        P_top=P_top, P_bottom=P_bottom, Q_top=empty, Q_bottom=empty,
        State_Limit=State_Limit, P_Sum=P_Sum, Q_Sum=empty,
        P_up=P_up, P_down=P_down, Q_up=empty, Q_down=empty,
        Transition_model=Transition_model, C=C, T=T, S=S, test_day_num=1,
        penlty_s=penlty_s, penlty_l=penlty_l, C_start=0.892, C_run=1.0419,
    )


def test_model_shapes():
    p = make_inputs()
    model = construct_tmilp_1bin_project(
        p["P_top"], p["P_bottom"], p["Q_top"], p["Q_bottom"], p["State_Limit"],
        p["P_Sum"], p["Q_Sum"], p["P_up"], p["P_down"], p["Q_up"], p["Q_down"],
        p["Transition_model"], p["C"], p["T"], p["S"], p["test_day_num"],
        p["penlty_s"], p["penlty_l"], p["C_start"], p["C_run"],
    )
    state_size = int(S.sum())
    expected = 4 * state_size * T + 2 * T
    assert model.num_vars() == expected, (model.num_vars(), expected)
    assert len(model.ctype) == expected
    assert model.ctype.startswith("B" * (state_size * T))
    assert model.Aineq.shape[1] == expected
    assert np.all(np.isfinite(model.f))
    assert model.model_size()[1] > 0
    print("  vars =", model.num_vars(), "rows =", model.model_size()[1],
          "nnz =", model.model_size()[2])


def test_v3_hash_pin():
    check_legacy_source()
    print("  v3 legacy hash OK")


def test_decompose_power():
    state_size = int(S.sum())
    x_vec = np.zeros(state_size * T)
    # 变量排布是列优先: x[s + t*state_size] 表示 t 时刻处于状态 s。
    # 每个设备在每个时刻只激活一个状态 (与模型 one-hot 约束一致)。
    for t in range(T):
        x_vec[0 + t * state_size] = 1.0   # 设备 1 -> 状态 1
        x_vec[2 + t * state_size] = 1.0   # 设备 2 -> 状态 1
    p_value = np.tile(np.array([100.0, 300.0, 50.0, 1500.0]), T)
    parts = decompose_power(x_vec, p_value, S, T)
    assert len(parts) == len(S)
    assert parts[0].shape == (T,)
    np.testing.assert_allclose(parts[0], 100.0)
    np.testing.assert_allclose(parts[1], 50.0)
    print("  decompose OK")


def test_solve_optional():
    if importlib.util.find_spec("coptpy") is None:
        print("  [skip] coptpy 未安装, 跳过求解测试")
        return
    print("  [info] coptpy 已安装, 求解测试未实装 (需要可行数据)")


if __name__ == "__main__":
    for fn in (test_model_shapes, test_v3_hash_pin, test_decompose_power, test_solve_optional):
        print(fn.__name__)
        fn()
    print("OK")
