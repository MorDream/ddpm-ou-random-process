"""
train.py —— 训练一个 DDPM（单一调度）

用法：
    python train.py --schedule cosine --steps 8000 --T 500 --shape smiley

训练目标只有一行：预测注入的噪声。
    t ~ U{0,...,T-1},  x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps,
    loss = || eps - eps_theta(x_t, t) ||^2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ShapeDataset
from diffusion import GaussianDiffusion
from model import Denoiser


def train_one(
    schedule: str = "cosine",
    steps: int = 8000,
    T: int = 500,
    shape: str = "smiley",
    batch: int = 512,
    lr: float = 2e-3,
    seed: int = 0,
    device: str = "cpu",
    out_dir: str = "runs",
    log_every: int = 200,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    diff = GaussianDiffusion(T=T, schedule=schedule, device=device)
    model = Denoiser(dim=2, hidden=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched_lr = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    ds = ShapeDataset(shape=shape, n=200_000, seed=seed)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    it = iter(loader)

    history = {"step": [], "loss": []}
    t0 = time.time()
    ema_loss = None
    for step in range(1, steps + 1):
        try:
            x0 = next(it)
        except StopIteration:
            it = iter(loader)
            x0 = next(it)
        x0 = x0.to(device)

        loss = diff.loss(model, x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched_lr.step()

        ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
        if step % log_every == 0 or step == 1:
            history["step"].append(step)
            history["loss"].append(float(ema_loss))
            if verbose:
                print(f"[{schedule}] step {step:5d}/{steps}  loss {ema_loss:.4f}  "
                      f"({time.time() - t0:5.1f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"ddpm_{shape}_{schedule}_T{T}.pt")
    torch.save({
        "model": model.state_dict(),
        "schedule": schedule,
        "T": T,
        "shape": shape,
        "steps": steps,
        "history": history,
    }, ckpt_path)

    if verbose:
        print(f"[{schedule}] 完成，用时 {time.time() - t0:.1f}s -> {ckpt_path}")
    return model, diff, history, ckpt_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", default="cosine", choices=["linear", "quadratic", "cosine"])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--shape", default="smiley", choices=["smiley", "heart", "cat"])
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one(
        schedule=args.schedule, steps=args.steps, T=args.T, shape=args.shape,
        batch=args.batch, lr=args.lr, seed=args.seed, out_dir=args.out_dir,
    )
