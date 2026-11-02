# nilmtcmilp

**TMILP-1bin-project**：基于混合整数线性规划的 NILM（非侵入式负荷监测）分解模型，MATLAB 实现的 Python 移植版。

每个电器的功率用若干离散状态表示，在给定总表读数下，求解"各电器各时刻处于哪个状态"的整数规划问题；
模型支持状态驻留时间下限、状态转换约束、功率爬坡约束，以及启动/运行惩罚项。

```text
变量   x : 状态占用 (0/1)，s : 状态切换，P/Q : 功率，kP/kQ : 功率平衡残差
目标   min  C_start·(启动惩罚) + C_run·(运行惩罚) + Σ(kP + kQ)
约束   状态转移、驻留时间、归一化功率投影、爬坡、功率平衡
```

## 安装

需要 Python 3.10+：

```bash
pip install -e .
```

求解 MILP 需要 **COPT**（`coptpy`），这是目前唯一的内置求解后端：

```bash
pip install coptpy
```

COPT 需要许可证；不装也能 import 本包、构造模型，只是在调用 `solve()` 时才会报错。

## 快速开始

```python
import numpy as np
from nilmtcmilp import construct_tmilp_1bin_project, decompose_power, solve

model = construct_tmilp_1bin_project(
    P_top, P_bottom,          # 每个状态的功率上/下界 (state_size,)
    Q_top, Q_bottom,          # 无功（不需要时传空数组）
    State_Limit,              # 每个状态的最短驻留采样点数
    P_Sum,                    # 总表有功功率 (T,)
    Q_Sum,                    # 总表无功（可传 np.array([]) 关闭）
    P_up, P_down,             # 有功爬坡上/下界 (state_size, 1)
    Q_up, Q_down,
    Transition_model,         # 每个电器的状态转移矩阵 (S_i, S_i)
    C,                        # 启动/运行惩罚权重 (state_size, 1)
    T, S, test_day_num,
    penlty_s, penlty_l,       # 状态概率 / 设备使用比例
    C_start, C_run,
)

result = solve(model, tolerance_gap=1e-3, timelimit=60)
x = result.x
```

完整的数据 → 参数流程见 `data_processing.py` 中的 `data_processing_emilp()`。

## 模块

| 模块 | 作用 |
|---|---|
| `milp_model.py` | 与 MATLAB `model` 结构体对应的容器（`f/Aineq/bineq/Aeq/beq/lb/ub/ctype/Q/Qc`） |
| `construct_tmilp_project.py` | 主构造函数（历史版本，带 hash 锁定） |
| `construct_tmilp_project_v3.py` | 状态链接 / 禁止转换边的修复版（会校验 v1 源码 SHA256） |
| `construct_tmilp_project_v4.py` | 在 v3 上再修正左删失边界与下降沿 |
| `data_processing.py` | EMILP 预处理：K-means 状态、DRO 回归求功率界、驻留时间、转移矩阵 |
| `dro_lr.py` | 分布鲁棒线性回归（二阶锥规划，依赖 COPT） |
| `clustering.py` | K-means 状态离散化 |
| `copt_solver.py` | COPT 求解器接口 |
| `evaluation.py` | RSE / R² / AC / 混淆矩阵等指标 |
| `data_read.py` | AMPDS / REFIT 数据读取 |
| `utils.py` | MATLAB 风格的小工具（`medfilt1`、`freshape` 等） |

## 数据

`data_read.read_data()` 面向 AMPDS 与 REFIT 的 CSV 布局，数据本身不随仓库分发；
`REFIT/appliance_information.mat` 需要放在包的上一级 `data/REFIT/` 目录下（见 `data_read.py`）。
如果只使用模型构造函数，可以完全绕开 `data_read`，直接按上面的参数签名传入数组。

## 与 NILMTK 的集成

配套仓库 **[nilmtk-milp](https://github.com/WillLeeMan/nilmtk-milp)** 把本包包装成 NILMTK 的
`Disaggregator`，并提供 HiGHS / CBC / SCIP / CP-SAT 四种求解器后端（求解分解模型时不需要 COPT）。

## 测试

```bash
python tests/test_construct.py
```

测试覆盖：模型构造的维度与变量类型、v3 的源码哈希校验、`decompose_power` 的还原逻辑。
求解相关的测试在检测不到 `coptpy` 时会自动跳过。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。

本包是 MATLAB 版 `function_construction_TMILP_1bin_project.m` 等函数的移植，
MATLAB 源码保留在配套研究仓库的 `_m_src_utf8/` 目录中，未随本仓库分发。

## 引用

如果这个模型对你的研究有帮助，请引用对应论文（待补充）。
