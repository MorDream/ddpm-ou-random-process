"""
data.py —— 二维"可爱形状"点云数据集

扩散模型只关心「数据的分布」，所以最干净的实验床是二维点云：
我们能把它直接画出来、肉眼判断生成质量，同时复杂度足够展示模式恢复。

这里提供三种形状（都可从网上找到同款的 emoji / 图标造型）：

    smiley : 笑脸（脸圈 + 两只眼睛 + 微笑弧线），本实验的主数据集
    heart  : 爱心曲线
    cat    : 猫脸（脸圈 + 两只耳朵 + 两只眼睛 + 鼻子）

所有样本最后会被缩放，使整体标准差约为 1 —— 这是 DDPM 默认超参
（beta 从 1e-4 到 0.02）成立的前提：数据是"单位量级"的。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 三种形状的采样函数
# ---------------------------------------------------------------------------


def _ring(n: int, r: float, w: float, c: tuple[float, float] = (0.0, 0.0),
          a0: float = 0.0, a1: float = 2 * np.pi, rng: np.random.Generator | None = None):
    """圆环（或圆弧）上的点；w 为环的粗细（高斯抖动）。"""
    rng = rng or np.random.default_rng()
    ang = rng.uniform(a0, a1, n)
    rad = r + w * rng.standard_normal(n)
    return np.stack([rad * np.cos(ang) + c[0], rad * np.sin(ang) + c[1]], axis=1)


def _disk(n: int, r: float, c: tuple[float, float], rng: np.random.Generator):
    """实心圆盘内均匀采样（用 sqrt 变换保证面积均匀）。"""
    rad = r * np.sqrt(rng.uniform(0.0, 1.0, n))
    ang = rng.uniform(0.0, 2 * np.pi, n)
    return np.stack([rad * np.cos(ang) + c[0], rad * np.sin(ang) + c[1]], axis=1)


def make_smiley(n: int = 6000, noise: float = 0.012, seed: int = 0) -> np.ndarray:
    """笑脸：脸圈 40%、两只眼睛各 15%、微笑弧 30%。"""
    rng = np.random.default_rng(seed)
    n_head, n_eye, n_mouth = int(0.40 * n), int(0.15 * n), int(0.30 * n)

    head = _ring(n_head, r=1.0, w=0.045, rng=rng)
    eye_l = _disk(n_eye, r=0.115, c=(-0.36, 0.30), rng=rng)
    eye_r = _disk(n_eye, r=0.115, c=(0.36, 0.30), rng=rng)
    mouth = _ring(n_mouth, r=0.55, w=0.035, c=(0.0, -0.02),
                  a0=np.deg2rad(200), a1=np.deg2rad(340), rng=rng)

    pts = np.concatenate([head, eye_l, eye_r, mouth], axis=0)
    pts += noise * rng.standard_normal(pts.shape)
    return pts


def make_heart(n: int = 6000, noise: float = 0.012, seed: int = 0) -> np.ndarray:
    """爱心曲线 x = 16sin^3 t, y = 13cos t - 5cos2t - 2cos3t - cos4t"""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], axis=1) / 12.0
    pts += noise * rng.standard_normal(pts.shape)
    return pts


def make_cat(n: int = 6000, noise: float = 0.012, seed: int = 0) -> np.ndarray:
    """猫脸：脸圈 + 两只三角耳朵 + 两只眼睛 + 鼻子。"""
    rng = np.random.default_rng(seed)
    head = _ring(int(0.34 * n), r=1.0, w=0.04, rng=rng)
    eyes = np.concatenate([
        _disk(int(0.10 * n), r=0.10, c=(-0.33, 0.22), rng=rng),
        _disk(int(0.10 * n), r=0.10, c=(0.33, 0.22), rng=rng),
    ], axis=0)
    nose = _disk(int(0.06 * n), r=0.07, c=(0.0, -0.10), rng=rng)

    # 耳朵：两条边上的点
    def ear(sign: float):
        m = int(0.10 * n)
        u = rng.uniform(0.0, 1.0, m)
        base_x, base_y = sign * 0.72, 0.66
        tip_x, tip_y = sign * 0.52, 1.42
        outer = np.stack([base_x + (tip_x - base_x) * u, base_y + (tip_y - base_y) * u], axis=1)
        base2_x = sign * 0.00
        inner = np.stack([tip_x + (base2_x + sign * 0.30 - tip_x) * u,
                          tip_y + (0.98 - tip_y) * u], axis=1)
        return np.concatenate([outer, inner], axis=0)

    pts = np.concatenate([head, eyes, nose, ear(-1.0), ear(1.0)], axis=0)
    pts += noise * rng.standard_normal(pts.shape)
    return pts


_SHAPES = {"smiley": make_smiley, "heart": make_heart, "cat": make_cat}


def sample_points(shape: str = "smiley", n: int = 6000, seed: int = 0) -> np.ndarray:
    """采一批点，并统一缩放到单位标准差。"""
    if shape not in _SHAPES:
        raise ValueError(f"未知形状 {shape}，可选：{list(_SHAPES)}")
    pts = _SHAPES[shape](n=n, seed=seed)
    pts = pts / pts.std(axis=0).mean()  # 单位量级，匹配 DDPM 默认 beta 范围
    return pts.astype(np.float32)


class ShapeDataset(Dataset):
    """把一次性采样好的点云包成 Dataset，方便按 batch 取。"""

    def __init__(self, shape: str = "smiley", n: int = 200_000, seed: int = 0):
        self.data = torch.from_numpy(sample_points(shape, n=n, seed=seed))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
