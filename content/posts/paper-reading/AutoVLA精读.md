---
title: "论文精读｜AutoVLA：用自适应推理与 GRPO 微调统一「思考」与「动作」的端到端 VLA"
date: 2026-07-20
draft: false
categories: ["论文精读"]
tags: ["🧠 VLA", "🔮 世界模型", "🚗 自动驾驶", "⚡ 端到端"]
summary: "AutoVLA（NeurIPS 2025, UCLA）把「链式推理 CoT」与「物理动作 token 化」塞进同一个自回归生成模型，用一套代码本把连续轨迹离散成可行动作，再用 SFT 训练出快思考（只出轨迹）与慢思考（带 CoT 推理）双模式，最后用 GRPO 强化微调在简单场景关掉多余推理。它是 NAVSIM v1 上 VLA 路线的代表基线（PDMS 89.1 / 92.1+锚点），也是 DriveVLA-W0 的主要对比对象。"
weight: 4
---

## 📄 论文信息

- **标题**：*AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning*
- **团队**：UCLA（Zewei Zhou, Tianhui Cai, Bolei Zhou, Jiaqi Ma 等）
- **发表**：NeurIPS 2025，arXiv:2506.13757
- **代码**：`github.com/ucla-mobility/AutoVLA`
- **一句话总结**：把「推理」和「动作」统一进一个自回归 VLM——轨迹被离散化成可行动作 token，简单场景快思考直出轨迹，复杂场景慢思考先 CoT 再出轨迹，最后用 GRPO 把简单场景里多余的推理「省掉」。

## 整体框架

![AutoVLA 框架](/images/autovla/AutoVLA_framework.png)

*图：AutoVLA 把视觉、语言、动作三个空间统一到一套自回归生成里。轨迹经 codebook 离散成动作 token，与文本 token、视觉 token 一起被 LLM 生成；推理 token 可选出现（快慢思考切换）。*

---

## 🤔 要解决什么问题？

VLA 模型用于自动驾驶有三个通病，AutoVLA 逐一对症下药：

| 痛点 | 表现 | AutoVLA 的解法 |
|------|------|----------------|
| **动作不可行** | VLM 直接生成「左转 30 度」这种自然语言动作，常物理不可执行 | 把连续轨迹 token 化成**离散、可行的动作 token**（codebook 约束） |
| **结构复杂** | 推理模块、规划模块各管各的，流水线长 | 推理与动作**统一进同一个自回归模型**，一次生成 |
| **推理太长** | 每个场景都走一遍长 CoT，实时性差 | **双思考模式 + GRPO 微调**，简单场景跳过推理 |

核心洞察是：**不是每个场景都需要深思熟虑**。直行、跟车这种简单场景，直接出轨迹就好；只有施工区、非常规博弈才需要 CoT 推理。AutoVLA 想让模型「自适应」决定想不想。

---

## 💡 核心创新一：物理动作 token 化（Action Tokenization）

这是 AutoVLA 能「直接生成轨迹」的基石。它借鉴 VQ-VAE / 轨迹聚类思路，预先用一个 **codebook（动作码本）** 把训练集里的真实轨迹聚成 K 个「可行动作原型」。

- 连续轨迹（比如未来 8 个时刻的 x/y/θ）被**量化**成 codebook 里最近的那个离散 token id；
- 推理时模型自回归生成的就是这些动作 token，再经 codebook 查表**反量化**回连续轨迹；
- 因为 codebook 里的每个原型都来自真实人类驾驶，生成的轨迹天然**物理可行**，避免了「语言式动作」不可执行的问题。

> 关键判断：**把动作变成词表里的 token，VLA 就既能「说话」也能「开车」**。这和 OpenVLA / FAST tokenizer 的离散化思路一脉相承，但 AutoVLA 强调 codebook 必须保证「可行」（feasible），不是任意分箱。

## 💡 核心创新二：快慢思考双模式（Dual Thinking）

AutoVLA 用**有监督微调（SFT）**让同一个模型学会两种「性格」：

- **快思考（Fast Thinking）**：输入图像 + 指令，直接自回归生成动作 token 序列，不出任何推理文字。适合大多数常规场景，延迟低。
- **慢思考（Slow Thinking）**：先生成一段**链式推理（CoT）** token（描述场景、关键物体、意图、最佳动作），再生成动作 token。适合长尾、歧义、需要反事实推理的复杂场景。

训练数据里同时有「轨迹-only」样本和「CoT + 轨迹」样本，模型在 SFT 阶段就把两种模式都学到了。推理时由一个简单的**路由策略**决定走哪条：复杂场景触发慢思考，简单场景走快思考。

## 💡 核心创新三：GRPO 强化微调（Reinforcement Fine-Tuning）

光有双模式还不够——模型可能「不管简不简单都慢慢想」，推理又长又慢。AutoVLA 引入基于 **GRPO（Group Relative Policy Optimization）** 的强化微调来「逼」模型在简单场景少想：

- 对每个 prompt 采样一组轨迹（含/不含 CoT），用**可验证的奖励**（如轨迹是否安全、PDMS 高低、是否多余推理）打分；
- GRPO 不依赖一个独立 critic，而是用**同一组样本内的相对优劣**做基线，稳定且省显存；
- 奖励函数同时鼓励「规划准」和「推理省」：简单场景里若模型硬走慢思考，会被扣分；复杂场景走快思考导致出错，也会被扣分。

这样训练出的模型学会了**自适应推理**——只在值得的场景才展开 CoT。

> 这和同在博客里的 **AlphaDrive / Flow-GRPO** 是同一类技术（GRPO 用于驾驶策略对齐），说明 GRPO 已成为自动驾驶 VLA 后训练的事实标准之一。

---

## 🧠 与 DriveVLA-W0 / DriveVLM 的对比

| 维度 | **AutoVLA**（UCLA） | **DriveVLM**（SJTU×NIO） | **DriveVLA-W0**（CASIA×蔚来） |
|------|---------------------|--------------------------|-------------------------------|
| 推理形式 | 自适应快慢思考（CoT 可选） | 显式语言 CoT（描述/分析/提取/推理） | 隐式预测式（世界模型预测未来） |
| 动作生成 | 离散动作 token（codebook） | 高层决策指导快系统出轨迹 | 连续（Query/AR/Flow Matching） |
| 监督密度 | 稀疏（动作 + 可选 CoT） | 稀疏（决策/动作） | **稠密**（未来图像） |
| 关键训练 | SFT 双模式 + GRPO 微调 | 常规 SFT | 两阶段 + 世界模型稠密监督 |
| 传感器 | 多相机（3 cam） | 多相机 | **单目前视** |
| NAVSIM v1 PDMS | 89.1 / 92.1（+锚点） | — | 93.0（AR） |

三者代表 VLA 自动驾驶的三种「如何理解世界」的哲学：AutoVLA 用**语言推理的可解释性** + GRPO 效率；DriveVLM 用**显式 CoT**；DriveVLA-W0 用**稠密世界模型**把监督密度拉满。AutoVLA 是 NAVSIM 上 VLA 路线的代表基线，DriveVLA-W0 论文的主要对比对象就是它。

---

## 🧪 实验与结果

AutoVLA 在 **nuPlan、nuScenes、Waymo、CARLA** 四个真实/仿真基准上做了开环与闭环评测，跨数据集证明泛化。

NAVSIM v1 榜单（VLA 路线代表）：

| 方法 | 传感器 | PDMS ↑ |
|------|--------|--------|
| UniAD | 6 cam | 83.4 |
| DiffusionDrive | 3 cam + L | 88.1 |
| AutoVLA | 3 cam | 89.1 |
| AutoVLA + 锚点 | 3 cam | 92.1 |
| **DriveVLA-W0（AR）** | **1 cam** | **93.0** |

几个关键发现：

1. **动作 token 化有效**：相比直接生成语言动作，codebook 量化后的轨迹在可行性和安全性上明显更好（碰撞率下降）。
2. **慢思考在长尾场景立功**：在需要博弈、施工的困难场景下，带 CoT 的慢思考 PDMS 显著高于快思考；常规场景两者持平。
3. **GRPO 微调既提分又提速**：强化微调后，模型在简单场景自动走快思考，平均推理 token 数下降，端到端延迟降低，同时规划指标不降反升。
4. **闭环也 work**：在 CARLA 闭环里，AutoVLA 能处理动态交互，说明「自适应推理」不只服务于开环指标。

---

## 📝 个人思考

AutoVLA 最值得借鉴的是**「推理不该是免费的」**这一工程直觉。很多 VLA 工作默认「越多推理越好」，AutoVLA 用 GRPO 把推理变成一种**可开关、有成本**的资源——这其实和人类驾驶一致：大部分时间靠直觉（快思考），只有陌生场景才认真分析（慢思考）。对车端部署，这种自适应比「永远深思」现实得多。

不过它和 DriveVLA-W0 暴露了 VLA 路线的同一个软肋：**监督仍然偏稀疏**。AutoVLA 的监督是「动作 token + 可选 CoT」，没有世界模型那种每像素稠密信号；所以它 PDMS 上限（92.1+锚点）略低于 DriveVLA-W0 的 93.0，且依赖多相机。我个人判断：**AutoVLA 的「自适应推理」与 DriveVLA-W0 的「稠密世界模型」并非互斥**——前者管「想不想想、怎么想」，后者管「监督够不够密」。未来若把 GRPO 自适应推理接到世界模型稠密监督上，可能同时拿到可解释性、效率与上限。这恰好是博客里三条 VLA 主线的交汇点。

另外，codebook 动作 token 化带来的**量化误差**和 DriveVLA-W0 里 AR 解码器小数据掉点如出一辙——说明「离散动作」与「连续动作」的优劣之争，本质是数据规模与表达力的权衡，和 DriveVLA-W0 发现的「解码器随数据规模反转」是同一个规律。

---

## 🔗 延伸阅读

| 工作 | 团队 | 与 AutoVLA 的关系 |
|------|------|-------------------|
| **DriveVLM** | SJTU × NIO | 显式语言 CoT 路线，AutoVLA 慢思考的思想来源之一 |
| **DriveVLA-W0** | CASIA × 蔚来 | 稠密世界模型路线，NAVSIM 主要对比对象，PDMS 更高 |
| **AlphaDrive** | — | GRPO 用于驾驶策略对齐，同属强化微调范式 |
| **Flow-GRPO** | — | 流匹配 + GRPO 驾驶策略，强化微调的另一实现 |
| **OpenDriveVLA** | — | 结构化视觉-语言 token 自回归出轨迹，动作 token 化同类 |
| **UniAD** | Shanghai AI Lab | BEV 端到端范式，AutoVLA 的超越对象 |

---

*📖 这是 VLA 自动驾驶系列的又一篇。AutoVLA（自适应推理）与 DriveVLA-W0（稠密世界模型）看似两条路，实则都在回答「怎么让大模型既懂世界、又开得动」——欢迎留言讨论哪条更适合车端。*
