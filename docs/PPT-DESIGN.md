# DESIGN.md —— 《从布朗运动到扩散模型》课程大作业汇报

## 全局视觉约定

- 画布 1280×720，padding 上下 60 / 左右 60，内容区 1160×600
- 风格：学术答辩风（白底 + 学术宝蓝），课程研究报告场景

### 配色

| 槽位 | Hex | 用途 |
|---|---|---|
| 主色 | #1E4FA8 | 顶条、编号块、强调 |
| 辅色 | #E8EFF8 / #3D7BD9 | 横幅底、图表辅助、卡底 |
| 强调色 | #D9534F | 关键结论、最优数值（≤3%） |
| 正文色 | #1A2230 / #4A5568 / #8B97A8 | 主/副/辅助文字 |
| 底色 | #FFFFFF | 主背景 |

### 字体

- 中文：思源黑体（Microsoft YaHei 兜底），层级靠字重
- 英文/数字：Arial
- 字号阶梯：封面主标 54 / 页主标 34 / 卡内小标 22 / 正文 18-20 / 图表标签 14-16 / 脚注 12-14

### 视觉母题

- 顶条：68px 深蓝实色，左白字页主标，右侧章节标识（如「02 · 如何解决」）
- 页脚：y=670，左「随机过程课程大作业」右「NN / 13」
- 结论条：每页底部浅蓝横条（#E8EFF8），一句话 takeaway

### 配图清单（全部来自本地实验输出，material）

| 文件 | 内容 | 使用页 |
|---|---|---|
| assets/dataset.png | 笑脸点云数据集 | P7 |
| assets/forward.png | 前向加噪六快照 | P5 |
| assets/schedules.png | β/ᾱ/SNR 三联曲线 | P8 |
| assets/euler.png | Euler 方差误差 | P8 |
| assets/reverse.png | 逆时采样轨迹 3×6 | P9 |
| assets/samples.png | 生成结果四联对照 | P10 |
| assets/metrics.png | 能量距离/SWD 柱状 | P10 |
| assets/ddim.png | DDIM 步数扫描 | P11 |

## 页面规划（13 页，10 分钟）

| # | type | role | layout | description |
|---|---|---|---|---|
| 1 | cover | hero | 左对齐大标 | 题目 + 副标 + 汇报人/日期 |
| 2 | content | supporting | 三栏卡 + 横幅 | GAN/VAE/Flow 局限 → 共同痛点 |
| 3 | content | supporting | 左右两区 | 扩散模型核心思想；本作业研究问题 |
| 4 | content | supporting | 三段公式链 | 布朗运动→SDE→OU；VP-SDE 定义 |
| 5 | content | supporting | 上图下文 | 前向=OU 离散化，解析解 + forward.png |
| 6 | content | supporting | 上下两区 | 逆时 SDE（Anderson）；KL→ε-MSE |
| 7 | content | supporting | 左图右文 | 实验设计：数据集/网络/调度/指标 |
| 8 | content | supporting | 双图 | SNR 曲线 + Euler 误差：调度=误差分布 |
| 9 | content | hero | 大图 | 成果① 加噪—去噪全流程 reverse.png |
| 10 | content | supporting | 双图 | 成果② 生成质量 samples + metrics |
| 11 | content | supporting | 图文 | 成果③ DDIM 步数=离散化误差 ddim.png |
| 12 | content | supporting | 结论清单 | 结论 3 条 + 局限 + 展望 |
| 13 | ending | — | 列表 | 参考文献 IEEE |

anti_pattern 全局：禁等宽装饰卡、禁 emoji、禁大色块铺满、图表必配结论判断。
