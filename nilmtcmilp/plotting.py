# -*- coding: utf-8 -*-
"""matplotlib 绘图助手, 对应 MATLAB 主脚本中的 figure/subplot/plot 段."""

import numpy as np


def plot_result(predict_list, true_list, p_sum, appliance_names, title,
                show_unknown=True):
    """对应 MATLAB 主脚本的绘图段:
    每台设备一个子图 (预测 vs 真实), 倒数第二个为未知设备偏差, 最后为总功率。
    """
    import matplotlib.pyplot as plt

    predict_list = [np.asarray(p).ravel() for p in predict_list]
    true_list = [np.asarray(t).ravel() for t in true_list]
    p_sum = np.asarray(p_sum, dtype=float).ravel()
    T = p_sum.size
    x_axis = np.arange(1, T + 1)
    n = len(predict_list)
    predict_sum = np.sum(np.vstack(predict_list), axis=0) if n else np.zeros(T)

    extra = 2 if show_unknown else 1
    fig, axes = plt.subplots(n + extra, 1, figsize=(10, 2 * (n + extra)),
                             num=title)
    axes = np.atleast_1d(axes)
    for i in range(n):
        ax = axes[i]
        ax.plot(x_axis, predict_list[i], label='predict')
        ax.plot(x_axis, true_list[i], label='true')
        ax.legend()
        name = appliance_names[i]
        ax.set_title(name[0] if isinstance(name, (list, tuple)) else name)
    if show_unknown:
        axes[n].plot(x_axis, np.zeros(T), label='predict')
        axes[n].plot(x_axis, np.abs(p_sum - predict_sum), label='unknow')
        axes[n].legend()
        axes[n].set_title('unknow device')
    axes[n + extra - 1].plot(x_axis, predict_sum, label='predict')
    axes[n + extra - 1].plot(x_axis, p_sum, label='true')
    axes[n + extra - 1].legend()
    axes[n + extra - 1].set_title('Total consumption')
    fig.tight_layout()
    return fig
