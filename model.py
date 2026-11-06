"""
model.py —— 去噪网络

理论上逆时 SDE 需要的是「分数函数」 score(x, t) = ∇_x log p_t(x)。
我们用一个小 MLP 直接回归噪声 eps，(等价于 score = -eps / sqrt(1-abar_t))，
这是 DDPM 的 epsilon-parameterization，数值上比直接回归 score 稳定。

网络结构刻意做小：2 维输入 + 时间嵌入 -> 3 层 256 宽 MLP。
玩具数据不需要 U-Net，这样每一步都能讲清楚。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """把连续时间 t 编码成傅里叶特征，让 MLP 能感知"现在是哪个噪声等级"。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]，取值范围 [0, T-1]，先归一化到 [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    """eps_theta(x_t, t)：给带噪点 x_t 和时间 t，预测当初注入的噪声。"""

    def __init__(self, dim: int = 2, hidden: int = 256, time_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        layers: list[nn.Module] = [nn.Linear(dim + hidden, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.time_mlp(self.time_emb(t))
        return self.net(torch.cat([x, h], dim=-1))
