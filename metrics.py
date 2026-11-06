"""
metrics.py —— 评价生成分布与真实分布的距离

二维点云没有 FID 这种指标，直接用两个经典的非参数统计距离：

1) 能量距离 (Energy Distance)：ED = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
   等于 0 当且仅当两个分布相同；无核参数，非常稳定。

2) 切片 Wasserstein 距离 (Sliced W1)：随机取若干方向投影到一维，
   在一维上按分位数积分（等价于排序后求平均绝对差），再对方向求平均。
   它比能量距离更能反映"形状轮廓"的差异。
"""

from __future__ import annotations

import numpy as np


def _pairwise_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    xy = _pairwise_norm(x, y).mean()
    xx = _pairwise_norm(x, x).mean()
    yy = _pairwise_norm(y, y).mean()
    return float(max(2.0 * xy - xx - yy, 0.0))


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 128, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n_proj, x.shape[1]))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    xp = np.sort(x @ theta.T, axis=0)  # [n, n_proj]
    yp = np.sort(y @ theta.T, axis=0)
    return float(np.mean(np.abs(xp - yp)))


def mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    """RBF 核 MMD^2，bandwidth 用中位数启发式。"""
    z = np.concatenate([x, y], axis=0)
    if bandwidth is None:
        d = _pairwise_norm(z[: min(len(z), 500)], z[: min(len(z), 500)])
        bandwidth = float(np.median(d[d > 0]))
    def k(a, b):
        return np.exp(-_pairwise_norm(a, b) ** 2 / (2 * bandwidth**2))
    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


ALL_METRICS = {
    "energy": energy_distance,
    "swd": sliced_wasserstein,
    "mmd": mmd_rbf,
}
