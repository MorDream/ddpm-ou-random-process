"""
experiments.py —— 一键复现全部实验与配图

    python experiments.py --steps 8000 --T 500 --shape smiley

产出：
    figures/fig01_dataset.png       数据集长什么样
    figures/fig02_forward.png       前向加噪（OU 过程离散化）快照
    figures/fig03_schedule.png      beta_t / abar_t / log-SNR 三条曲线
    figures/fig04_reverse.png       逆时 SDE 求解轨迹（三种调度对照）
    figures/fig05_samples.png       生成结果对照
    figures/fig06_metrics.png       定量指标对照
    figures/fig07_ddim_steps.png    采样步数 vs 质量（ODE 离散化误差）
    figures/fig08_euler.png         Euler 格式的前向方差误差（解析计算）
    runs/metrics.json               所有数值结果
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plotting
from data import sample_points
from diffusion import GaussianDiffusion
from metrics import energy_distance, sliced_wasserstein
from model import Denoiser
from schedules import SCHEDULES, make_schedule, snr
from train import train_one

FIG_DIR = "figures"
RUN_DIR = "runs"


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def get_trained(schedule: str, T: int, steps: int, shape: str, device: str = "cpu"):
    path = os.path.join(RUN_DIR, f"ddpm_{shape}_{schedule}_T{T}.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("steps", -1) >= steps:
            model = Denoiser(dim=2, hidden=256).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            return model, GaussianDiffusion(T=T, schedule=schedule, device=device), ckpt["history"]
    model, diff, history, _ = train_one(
        schedule=schedule, steps=steps, T=T, shape=shape, device=device, out_dir=RUN_DIR
    )
    model.eval()
    return model, diff, history


# ---------------------------------------------------------------------------
# 图 1：数据集
# ---------------------------------------------------------------------------
def fig_dataset(shape: str, path: str) -> str:
    pts = sample_points(shape, n=8000, seed=1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    plotting.scatter(ax, pts, plotting.PALETTE["real"], s=2.5, alpha=0.5)
    ax.set_title(f"训练数据：二维「笑脸」点云（n=8000）", pad=10)
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
# 图 2：前向加噪
# ---------------------------------------------------------------------------
def fig_forward(shape: str, T: int, path: str) -> str:
    diff = GaussianDiffusion(T=T, schedule="cosine")
    x0 = sample_points(shape, n=1500, seed=3)
    ts = [0, int(T * 0.15), int(T * 0.35), int(T * 0.6), int(T * 0.85), T - 1]
    snaps = diff.forward_trajectory(x0, ts)

    fig, axes = plt.subplots(1, len(ts), figsize=(13.5, 2.6))
    for ax, t, x in zip(axes, ts, snaps):
        plotting.scatter(ax, x, plotting.PALETTE["real"], s=2.5, alpha=0.45)
        ax.set_title(f"t = {t}", fontsize=10)
    fig.suptitle("前向过程：把数据一步步注入布朗噪声（OU 过程离散化）", y=1.04, fontsize=12)
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
# 图 3：调度曲线
# ---------------------------------------------------------------------------
def fig_schedule(T: int, path: str) -> dict:
    fig = plt.figure(figsize=(13.5, 3.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.28)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    t = np.arange(1, T + 1)
    stats = {}

    for name in SCHEDULES:
        s = make_schedule(name, T)
        c = plotting.PALETTE[name]
        ax1.plot(t, s["betas"], color=c, lw=1.8, label=name)
        ax2.plot(t, s["abar"], color=c, lw=1.8, label=name)
        ax3.plot(t, 10 * np.log10(snr(s["abar"])), color=c, lw=1.8, label=name)
        stats[name] = {
            "beta_min": float(s["betas"].min()),
            "beta_max": float(s["betas"].max()),
            "abar_T": float(s["abar"][-1]),
            "steps_abar_gt_1e-3": int((s["abar"] >= 1e-3).sum()),
        }

    ax1.set_title(r"$\beta_t$：每步注入的噪声强度")
    ax1.set_xlabel("扩散步 t")
    ax2.set_title(r"$\bar{\alpha}_t$：信号残留比例")
    ax2.set_xlabel("扩散步 t")
    ax2.set_yscale("log")
    ax3.set_title("信噪比 SNR(dB)：信息消失的节奏")
    ax3.set_xlabel("扩散步 t")
    for ax in (ax1, ax2):
        ax.legend(frameon=False, fontsize=9)
    return plotting.save(fig, path), stats


# ---------------------------------------------------------------------------
# 图 4：逆时采样轨迹
# ---------------------------------------------------------------------------
def fig_reverse(models: dict, T: int, path: str) -> str:
    n = 1500
    fig, axes = plt.subplots(len(SCHEDULES), 6, figsize=(13.5, 7.0))
    for row, name in enumerate(SCHEDULES):
        diff, model = models[name]
        samples, traj = diff.sample(model, n=n, record_every=T // 5, seed=11)
        for col in range(6):
            ax = axes[row, col]
            plotting.scatter(ax, traj[col], plotting.PALETTE[name], s=2.2, alpha=0.5)
            if row == 0:
                ax.set_title(f"t = {T - 1 - col * (T // 5)}" if col else f"t = {T - 1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(name, color=plotting.PALETTE[name], fontsize=11)
    fig.suptitle("逆时 SDE 的数值求解：从纯噪声回到数据（同行从左到右）", y=0.995, fontsize=12)
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
# 图 5：生成结果对照
# ---------------------------------------------------------------------------
def fig_samples(models: dict, shape: str, path: str) -> str:
    real = sample_points(shape, n=4000, seed=123)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    plotting.scatter(axes[0], real, plotting.PALETTE["real"], s=2.5, alpha=0.5)
    axes[0].set_title("真实数据", fontsize=11)
    for ax, name in zip(axes[1:], SCHEDULES):
        diff, model = models[name]
        gen, _ = diff.sample(model, n=4000, seed=7)
        plotting.scatter(ax, gen, plotting.PALETTE[name], s=2.5, alpha=0.5)
        ax.set_title(f"DDPM · {name}", fontsize=11, color=plotting.PALETTE[name])
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
# 图 6：定量指标
# ---------------------------------------------------------------------------
def compute_metrics(models: dict, shape: str) -> dict:
    real = sample_points(shape, n=4000, seed=123)
    out = {}
    for name in SCHEDULES:
        diff, model = models[name]
        gen, _ = diff.sample(model, n=4000, seed=7)
        out[name] = {
            "energy": energy_distance(gen, real),
            "swd": sliced_wasserstein(gen, real),
        }
    return out


def fig_metrics(metrics: dict, path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    names = list(SCHEDULES)
    colors = [plotting.PALETTE[n] for n in names]
    for ax, key, title in zip(axes, ["energy", "swd"],
                              ["能量距离（越小越好）", "切片 Wasserstein 距离（越小越好）"]):
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(names, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.suptitle("生成分布与真实分布的距离：三种调度的定量对比", y=1.02, fontsize=12)
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
# 图 7：采样步数（ODE 离散化误差）
# ---------------------------------------------------------------------------
def fig_ddim(models: dict, shape: str, path: str) -> dict:
    real = sample_points(shape, n=4000, seed=123)
    step_list = [2, 5, 10, 20, 50, 100, 200, 500]
    res = {}
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for name in SCHEDULES:
        diff, model = models[name]
        vals = []
        for k in step_list:
            gen = diff.sample_ddim(model, n=4000, steps=k, seed=5)
            vals.append(sliced_wasserstein(gen, real))
        res[name] = dict(zip(map(str, step_list), vals))
        ax.plot(step_list, vals, "o-", color=plotting.PALETTE[name], lw=1.8, ms=4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("采样步数（概率流 ODE 的积分步数）")
    ax.set_ylabel("SWD")
    ax.set_title("确定性求解器的离散化误差：步数越多越接近真实分布")
    ax.legend(frameon=False)
    return plotting.save(fig, path), res


# ---------------------------------------------------------------------------
# 图 8：Euler 格式的前向方差误差
# ---------------------------------------------------------------------------
def fig_euler(T: int, path: str) -> dict:
    """对线性 SDE，Euler-Maruyama 的方差递推为
         v_t = (1 - beta_t/2)^2 v_{t-1} + beta_t
       而解析解的方差是 1 - abar_t。两者之差就是离散化误差。"""
    fig = plt.figure(figsize=(13, 3.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    stats = {}

    for name in SCHEDULES:
        s = make_schedule(name, T)
        b, a = s["betas"], s["abar"]
        v, err = 0.0, []
        for i in range(T):
            v = (1 - b[i] / 2) ** 2 * v + b[i]
            err.append(v - (1 - a[i]))
        err = np.abs(np.array(err))
        mask = a >= 1e-3  # 只统计数据尚未被完全湮灭的区间
        ax1.semilogy(np.arange(1, T + 1)[mask], err[mask] + 1e-12,
                     color=plotting.PALETTE[name], lw=1.6, label=name)
        stats[name] = {
            "mean_abs_err": float(err[mask].mean()),
            "max_abs_err": float(err[mask].max()),
        }

    ax1.set_xlabel("扩散步 t")
    ax1.set_ylabel(r"$|v^{Euler}_t - v^{exact}_t|$")
    ax1.set_title("Euler 离散化的方差误差沿时间累积")
    ax1.legend(frameon=False)

    names = list(SCHEDULES)
    vals = [stats[n]["mean_abs_err"] for n in names]
    bars = ax2.bar(names, vals, color=[plotting.PALETTE[n] for n in names], width=0.6)
    for b_, v_ in zip(bars, vals):
        ax2.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.4f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_title(r"平均绝对误差（$\bar{\alpha}_t \geq 10^{-3}$ 区间）")
    ax2.grid(axis="x", visible=False)
    return plotting.save(fig, path), stats


# ---------------------------------------------------------------------------
# 图 9：训练损失
# ---------------------------------------------------------------------------
def fig_loss(histories: dict, path: str) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for name in SCHEDULES:
        h = histories[name]
        ax.plot(h["step"], h["loss"], color=plotting.PALETTE[name], lw=1.5, label=name)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("EMA 损失")
    ax.set_title("训练曲线（三种调度的训练目标完全相同）")
    ax.legend(frameon=False)
    return plotting.save(fig, path)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--shape", default="smiley")
    args = p.parse_args()

    plotting.setup()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    print("== 训练三种调度 ==")
    models, histories = {}, {}
    for name in SCHEDULES:
        model, diff, hist = get_trained(name, args.T, args.steps, args.shape)
        models[name], histories[name] = (diff, model), hist
        print(f"  {name} 就绪")

    print("== 生成配图 ==")
    fig_dataset(args.shape, f"{FIG_DIR}/fig01_dataset.png")
    fig_forward(args.shape, args.T, f"{FIG_DIR}/fig02_forward.png")
    _, sched_stats = fig_schedule(args.T, f"{FIG_DIR}/fig03_schedule.png")
    fig_reverse(models, args.T, f"{FIG_DIR}/fig04_reverse.png")
    fig_samples(models, args.shape, f"{FIG_DIR}/fig05_samples.png")

    metrics = compute_metrics(models, args.shape)
    fig_metrics(metrics, f"{FIG_DIR}/fig06_metrics.png")
    _, ddim = fig_ddim(models, args.shape, f"{FIG_DIR}/fig07_ddim_steps.png")
    _, euler = fig_euler(args.T, f"{FIG_DIR}/fig08_euler.png")
    fig_loss(histories, f"{FIG_DIR}/fig09_loss.png")

    summary = {
        "config": {"T": args.T, "steps": args.steps, "shape": args.shape},
        "schedules": sched_stats,
        "metrics": metrics,
        "ddim_steps": ddim,
        "euler_error": euler,
        "final_loss": {n: histories[n]["loss"][-1] for n in SCHEDULES},
    }
    with open(f"{RUN_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print("完成 ->", FIG_DIR)


if __name__ == "__main__":
    main()
