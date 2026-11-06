r"""
schedules.py —— 噪声调度 beta(t) 的定义

本作业的核心变量。在连续时间视角下，前向加噪过程是「方差保持型（VP）」
Ornstein-Uhlenbeck 过程：

    dX_t = -1/2 * beta(t) * X_t dt + sqrt(beta(t)) dW_t

对线性 SDE 可用伊藤公式求得解析解，其转移核为

    X_t | X_0 = sqrt(abar_t) * X_0 + sqrt(1 - abar_t) * eps,
    abar_t = exp( - \int_0^t beta(s) ds )

所以 beta(t) 的形状决定了「信息在什么时刻丢失得最快」；
换一种 beta(t)，本质上就是换一种 SDE 系数轮廓，
离散化之后对应不同的时间网格刚度与截断误差。

三种被对比的调度：
    linear    : beta_t 线性增长（Ho et al. 2020, DDPM 原文）
    quadratic : beta_t 二次增长（前期更慢、后期更快）
    cosine    : abar_t 按余弦平方衰减（Nichol & Dhariwal 2021）
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 基础形式
# ---------------------------------------------------------------------------


def linear_beta_schedule(T: int, beta_min: float = 1e-4, beta_max: float = 0.02) -> np.ndarray:
    """原版 DDPM 的线性调度：beta_t = beta_min + (beta_max - beta_min) * t/T"""
    return np.linspace(beta_min, beta_max, T, dtype=np.float64)


def quadratic_beta_schedule(T: int, beta_min: float = 1e-4, beta_max: float = 0.02) -> np.ndarray:
    """二次调度：beta_t 与 (t/T)^2 成正比，相当于让 SDE 系数在早期更"软"。"""
    s = np.linspace(0.0, 1.0, T, dtype=np.float64)
    return beta_min + (beta_max - beta_min) * s**2


def cosine_beta_schedule(T: int, s: float = 0.008, abar_floor: float = 1e-4) -> np.ndarray:
    """余弦调度：直接指定 abar_t = cos^2( ((t/T + s)/(1+s)) * pi/2 ) / cos^2(...)

    再反解出 beta_t = 1 - abar_t / abar_{t-1}。

    注意：原始实现中 t=T 时 abar_T 严格为 0，会让最后一步的 beta 跳到 1，
    这是数值伪影而不是数学性质。我们对 abar 取一个下界 abar_floor=1e-4
    （等价于"噪声已经足够大，之后不再加噪"），使三种调度在相同的总噪声
    预算下才可比。
    """
    steps = T + 1
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    f = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
    f = f / f[0]
    abar = np.clip(f, abar_floor, 1.0)
    betas = 1.0 - abar[1:] / abar[:-1]
    return np.clip(betas, 1e-8, 0.5)


# ---------------------------------------------------------------------------
# 归一化：让三种调度拥有相同的时间积分（同样的"总噪声预算"）
# ---------------------------------------------------------------------------


def total_noise(betas: np.ndarray) -> float:
    """\\int_0^1 beta(t) dt 的离散近似，也等于 -log(abar_T)。"""
    return float(-np.log(np.prod(1.0 - betas)))


def normalize_to_budget(betas: np.ndarray, budget: float) -> np.ndarray:
    """同比例缩放 beta 序列，使其总噪声预算与参照调度一致。

    这样比较的才是「形状」，而不是「加噪总量」——否则 abar_T 不同，
    生成质量差异无法归因到调度形状本身。
    """
    return np.clip(betas * budget / total_noise(betas), 1e-8, 0.999)


def make_schedule(name: str, T: int, budget: float | None = None) -> dict:
    """返回一个调度字典，包含 betas / alphas / abar 等所有后续要用的量。

    Parameters
    ----------
    name : {"linear", "quadratic", "cosine"}
    T    : 离散步数（同时也是 SDE 的离散点数）
    budget : 总噪声预算；None 表示使用 cosine 的自然预算（保证 abar_T ~ 1e-3 量级）
    """
    if name == "linear":
        betas = linear_beta_schedule(T)
    elif name == "quadratic":
        betas = quadratic_beta_schedule(T)
    elif name == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        raise ValueError(f"未知调度: {name}")

    if budget is None:
        budget = total_noise(cosine_beta_schedule(T))
    if name != "cosine" or budget != total_noise(betas):
        betas = normalize_to_budget(betas, budget)

    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return {
        "name": name,
        "betas": betas,
        "alphas": alphas,
        "abar": abar,
        "sqrt_abar": np.sqrt(abar),
        "sqrt_one_minus_abar": np.sqrt(1.0 - abar),
        "budget": float(budget),
    }


def snr(abar: np.ndarray) -> np.ndarray:
    """信噪比 SNR(t) = abar_t / (1 - abar_t)，用 dB 表示时更直观。"""
    return abar / np.maximum(1.0 - abar, 1e-12)


SCHEDULES = ("linear", "quadratic", "cosine")
