"""
diffusion.py —— 前向 OU 过程、逆时 SDE 采样、训练损失

三个核心公式（全部由随机过程推导而来，代码里逐行对照）：

1) 前向（离散化的 OU / VP-SDE）
   由 dX = -1/2 beta(t) X dt + sqrt(beta(t)) dW 的解析解，
   q(x_t | x_0) = N( sqrt(abar_t) x_0, (1-abar_t) I )
   一步采样：x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps

2) 逆时 SDE（Anderson 1982 的时间反演）
   dx = [ f(x,t) - g(t)^2 * score(x,t) ] dt + g(t) dw_bar
   把 score 用 -eps / sqrt(1-abar) 替代，就得到 DDPM 的反向转移
   p(x_{t-1} | x_t) = N(mu_theta, Sigma_t)，
   其均值/方差正是后验 q(x_{t-1} | x_t, x_0) 的贝叶斯解。

3) 训练损失
   对 KL 链做化简，逐项都是高斯 KL，最终只剩
   L = E_{t,x0,eps} || eps - eps_theta(x_t, t) ||^2
   —— 形式极简，但它完全来自随机过程的推导。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model import Denoiser
from schedules import make_schedule


class GaussianDiffusion:
    def __init__(self, T: int = 500, schedule: str = "cosine", device: str = "cpu"):
        self.T = T
        self.schedule_name = schedule
        self.device = device
        s = make_schedule(schedule, T)

        def to_torch(a: np.ndarray) -> torch.Tensor:
            return torch.tensor(a, dtype=torch.float32, device=device)

        self.betas = to_torch(s["betas"])
        self.alphas = to_torch(s["alphas"])
        self.abar = to_torch(s["abar"])
        self.sqrt_abar = to_torch(s["sqrt_abar"])
        self.sqrt_one_minus_abar = to_torch(s["sqrt_one_minus_abar"])

        # 后验方差 beta_t * (1 - abar_{t-1}) / (1 - abar_t)，用于祖先采样
        abar_prev = torch.cat([torch.ones(1, device=device), self.abar[:-1]])
        self.posterior_var = self.betas * (1.0 - abar_prev) / (1.0 - self.abar)

    # ------------------------------------------------------------------
    # 1. 前向过程：OU 过程的离散化
    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None):
        """x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps"""
        if eps is None:
            eps = torch.randn_like(x0)
        sa = self.sqrt_abar[t][:, None]
        so = self.sqrt_one_minus_abar[t][:, None]
        return sa * x0 + so * eps, eps

    # ------------------------------------------------------------------
    # 2. 训练损失：噪声预测的 MSE
    # ------------------------------------------------------------------
    def loss(self, model: Denoiser, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        x_t, eps = self.q_sample(x0, t)
        eps_hat = model(x_t, t.float())
        return F.mse_loss(eps_hat, eps)

    # ------------------------------------------------------------------
    # 3. 逆时 SDE 的数值求解（祖先采样，即 Euler-Maruyama）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, model: Denoiser, n: int = 4000, record_every: int | None = None,
               seed: int | None = None):
        """从纯噪声出发，逐步求解逆时 SDE。

        每一步：
            eps_hat = eps_theta(x_t, t)
            mu  = (x_t - beta_t / sqrt(1-abar_t) * eps_hat) / sqrt(alpha_t)
            x_{t-1} = mu + sqrt(posterior_var_t) * z
        """
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        x = torch.randn(n, 2, device=self.device, generator=g)

        traj = []
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps_hat = model(x, tt.float())
            mu = (x - self.betas[t] / self.sqrt_one_minus_abar[t] * eps_hat) / self.alphas[t].sqrt()
            if t > 0:
                z = torch.randn(x.shape, device=self.device, generator=g)
                x = mu + self.posterior_var[t].sqrt() * z
            else:
                x = mu
            if record_every and (t % record_every == 0 or t == self.T - 1):
                traj.append(x.clone().cpu().numpy())
        return x.cpu().numpy(), traj

    # ------------------------------------------------------------------
    # 4. 概率流 ODE 的求解（DDIM, eta=0），用于演示"确定性逆时方程"
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_ddim(self, model: Denoiser, n: int = 4000, steps: int = 50, seed: int | None = None):
        """把 T 步压缩成 steps 步的确定性 Euler 积分，误差来自步长而非随机性。"""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        x = torch.randn(n, 2, device=self.device, generator=g)

        ts = np.linspace(self.T - 1, 0, steps + 1).round().astype(int)
        for i in range(steps):
            t_cur, t_prev = int(ts[i]), int(ts[i + 1])
            tt = torch.full((n,), t_cur, device=self.device, dtype=torch.long)
            eps_hat = model(x, tt.float())
            x0_hat = (x - self.sqrt_one_minus_abar[t_cur] * eps_hat) / self.sqrt_abar[t_cur]
            x = self.sqrt_abar[t_prev] * x0_hat + self.sqrt_one_minus_abar[t_prev] * eps_hat
        return x.cpu().numpy()

    # ------------------------------------------------------------------
    # 辅助：记录前向加噪轨迹（用于"加噪—去噪"对照图）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_trajectory(self, x0: np.ndarray, ts: list[int]) -> list[np.ndarray]:
        x0_t = torch.tensor(x0, dtype=torch.float32, device=self.device)
        out = []
        for t in ts:
            tt = torch.full((x0_t.shape[0],), t, device=self.device, dtype=torch.long)
            x_t, _ = self.q_sample(x0_t, tt)
            out.append(x_t.cpu().numpy())
        return out


def build_model(schedule: str = "cosine", T: int = 500, device: str = "cpu"):
    return GaussianDiffusion(T=T, schedule=schedule, device=device)
