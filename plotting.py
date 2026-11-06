"""
plotting.py —— 全项目统一的绘图风格

配色约定（与 PPT 一致）：
    linear    #4C6EF5（蓝）
    quadratic #F59F00（橙）
    cosine    #E64980（品红）
    真实数据   #212529（近黑）
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

PALETTE = {
    "linear": "#4C6EF5",
    "quadratic": "#F59F00",
    "cosine": "#E64980",
    "real": "#212529",
    "grid": "#DEE2E6",
    "muted": "#868E96",
}

CN_LABEL = {
    "linear": "线性 beta_t",
    "quadratic": "二次 beta_t",
    "cosine": "余弦 abar_t",
}


def setup() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#ADB5BD",
        "axes.labelcolor": "#212529",
        "axes.grid": True,
        "grid.color": "#DEE2E6",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def scatter(ax, pts, color, s=2.0, alpha=0.55, label=None):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, alpha=alpha, linewidths=0, label=label)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CED4DA")
    ax.grid(False)


def save(fig, path: str) -> str:
    fig.savefig(path)
    plt.close(fig)
    return path
