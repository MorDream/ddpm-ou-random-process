# 从布朗运动到扩散模型：OU 过程离散化与逆时 SDE 的一维/二维 DDPM 实现

> 随机过程课程大作业 · 手写 DDPM（无任何扩散模型框架依赖）

Stable Diffusion 一类文生图模型的数学核心并不是"图像处理"，而是**随机过程**：

- **前向加噪**是一个方差保持（VP）型 **Ornstein–Uhlenbeck 过程**的离散化；
- **生成（去噪）**是**逆时 SDE** 的数值求解；
- **噪声调度 β(t) 的选择**，本质上就是 SDE 系数的轮廓设计，决定了离散化误差如何分布在时间轴上。

本项目用一个**二维"笑脸"点云**做实验床，把上述三条逐一写出来、跑出来、画出来。

## 1. 数学主线

**前向（离散化的 OU 过程）**

```
dX_t = -1/2 β(t) X_t dt + sqrt(β(t)) dW_t
  ⇒  X_t | X_0 = sqrt(ᾱ_t) X_0 + sqrt(1-ᾱ_t) ε,   ᾱ_t = exp(-∫_0^t β(s) ds)
```

**逆时（Anderson 时间反演）**

```
dx = [ f(x,t) - g(t)² ∇_x log p_t(x) ] dt + g(t) dw̄
score 用噪声预测替代： ∇ log p_t(x) = -ε_θ(x,t) / sqrt(1-ᾱ_t)
```

**训练目标（KL 链化简的结果）**

```
L = E_{t,x_0,ε} || ε - ε_θ(x_t, t) ||²
```

**离散化误差（本作业的实验落点）**

对线性 SDE，Euler–Maruyama 的方差递推是
`v_t = (1-β_t/2)² v_{t-1} + β_t`，解析解是 `1-ᾱ_t`；两者之差即离散化误差，
其大小与 β 的量级直接相关，因此**不同调度 = 不同的误差分布**。

## 2. 三种被对比的噪声调度

| 调度 | 形式 | 特点 |
|---|---|---|
| linear | β_t 从 1e-4 线性升到 β_max | DDPM 原文，前期加噪过慢 |
| quadratic | β_t ∝ (t/T)² | 前期更"软"，后期更陡 |
| cosine | ᾱ_t = cos²(·) | log-SNR 近似线性下降 |

> 公平性处理：三种调度被**归一化到相同的总噪声预算** `∫β dt`，
> 因此差异只来自调度"形状"，不来自加噪总量。

## 3. 文件结构

```
schedules.py    β(t) 的三种定义 + 归一化 + SNR
data.py         二维可爱形状点云（smiley / heart / cat）
model.py        ε-预测 MLP（正弦时间嵌入）
diffusion.py    前向 OU 采样、训练损失、逆时 SDE（祖先采样）、DDIM/ODE 采样
train.py        单调度训练脚本
experiments.py  一键训练三种调度 + 生成全部配图
metrics.py      能量距离 / 切片 Wasserstein / MMD
plotting.py     统一绘图风格
```

## 4. 运行

```bash
python -m venv .venv && ./.venv/Scripts/activate     # Windows
pip install -r requirements.txt

python experiments.py --steps 8000 --T 500 --shape smiley
```

CPU 即可完成全部实验（约数分钟）。产物在 `figures/` 与 `runs/`。

## 5. 主要结论

（由 `runs/metrics.json` 自动生成，见报告/PPT）

1. 三种调度的前向过程都能把数据"抹平"成 N(0, I)，差异体现在**抹平的节奏**；
2. Euler 离散化的方差误差集中在 β 较大的时段，cosine 调度的平均误差更小；
3. 生成质量（能量距离 / SWD）上 cosine 优于 linear，与 log-SNR 均匀下降的理论预期一致；
4. DDIM 用 50 步即可逼近 500 步祖先采样的效果，说明实际瓶颈是 score 估计误差而非网格误差。

## License

MIT。图表与代码可自由用于课程报告。
