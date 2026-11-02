# -*- coding: utf-8 -*-
"""K-means 聚类, 对应 function_getStateNumByKmean.m.

- index >= 1: 肘部法则, k=1..30 绘制 gap 曲线 (sklearn 版本返回各 k 的 inertia)
- index < 0 : 聚类为 k 类, 并按聚类中心从小到大重新编号 (与 MATLAB 一致)
"""

import numpy as np

_KMEANS_SEED = 0


def kmeans_labels(data, k, replicates=4):
    """MATLAB kmeans(data, k, 'dist','sqEuclidean','Replicates',4) 的标签版本.

    返回按聚类中心升序重编号的标签 (1..k), 以及排序后的中心.
    """
    from sklearn.cluster import KMeans
    data = np.asarray(data, dtype=float).reshape(-1, 1)
    km = KMeans(n_clusters=int(k), n_init=int(replicates),
                random_state=_KMEANS_SEED)
    labels = km.fit_predict(data) + 1
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)          # 中心从小到大
    new_labels = np.empty_like(labels)
    new_centers = centers[order]
    for rank, old in enumerate(order):
        new_labels[labels == old + 1] = rank + 1
    return new_labels, new_centers


def get_state_num_by_kmean(data_mat, name, index, k=None):
    """function_getStateNumByKmean 的 Python 版本.

    index >= 1: 返回 k=1..30 的 inertia 列表 (肘部法则, 绘图由调用方决定)
    index < 0 : 返回重编号后的聚类标签 (k 为空时打印提示并返回 None)
    """
    data_mat = np.asarray(data_mat, dtype=float).ravel()
    if index >= 1:
        from sklearn.cluster import KMeans
        out = []
        for kk in range(1, 31):
            km = KMeans(n_clusters=kk, n_init=4, random_state=_KMEANS_SEED)
            km.fit(data_mat.reshape(-1, 1))
            out.append(km.inertia_)
        try:
            import matplotlib.pyplot as plt
            plt.figure(index)
            plt.plot(range(1, 31), out)
            plt.title(str(name))
            plt.xlabel('k')
            plt.ylabel('gap')
        except Exception:
            pass
        return out
    else:
        if k is None:
            print('cluster num is empty!')
            return None
        labels, _ = kmeans_labels(data_mat, k)
        return labels
