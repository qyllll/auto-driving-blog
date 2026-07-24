---
title: "Cosmos-3 + MoE + Flow-GRPO 架构概览"
date: 2026-07-23
draft: false
categories: ["个人思考"]
summary: "本文提出一种将 Cosmos-3 双塔 MoT、DeepSeek-style Sparse MoE、Flow Matching 与 Flow-GRPO 结合的架构设计方案，系统描述各模块的角色、接口与训练推理流程。"
tags: ["架构", "Cosmos-3", "Sparse MoE", "Flow-GRPO", "VLA"]
math: true
---

## 1. 一句话介绍

该架构以 **Cosmos-3 双塔 MoT** 为基础，在 Reasoner 和 Generator 的前馈层中加入 **DeepSeek-style Sparse MoE**，再通过 **Flow Matching** 生成连续轨迹，最后使用 **Flow-GRPO** 根据环境奖励优化轨迹策略。

可以概括为：

> **Cosmos-3 负责理解和生成表征，MoE 负责稀疏专家计算，Flow 负责生成连续轨迹，Flow-GRPO 负责强化学习优化。**

---

## 2. 总体流程

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/overall-flow.svg" alt="总体流程图" style="max-width:100%;">
</div>

整体包含四个核心阶段：

1. **多模态理解**：编码视觉、语言、智能体状态和任务目标；
2. **稀疏专家计算**：为不同 token 选择合适的专家；
3. **连续轨迹生成**：从噪声逐步生成完整轨迹块；
4. **强化学习优化**：利用环境奖励改进生成策略和专家路由。

---

## 3. 输入与输出

### 3.1 输入

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/input-processing.svg" alt="输入处理" style="max-width:100%;">
</div>

| 输入 | 作用 | 编码结果 |
|---|---|---|
| 视觉观测 | 提供场景、对象与空间关系 | Visual tokens |
| 语言指令 | 描述任务需求和行为目标 | Language tokens |
| 智能体状态 | 提供当前可观测状态 | State tokens |
| 场景与目标 | 提供结构化条件和目标位置 | Context tokens |

### 3.2 输出

模型输出未来一段时间的连续轨迹：

$$
A_t=\{a_{t+1},a_{t+2},\ldots,a_{t+T}\}.
$$

轨迹点可以表示位置、朝向和速度，也可以根据任务接口定义为连续控制量。部署时只执行轨迹块的前若干步，然后根据新观测重新生成下一段轨迹。

---

## 4. Cosmos-3 双塔 MoT

Cosmos-3 将 token 分为两类，并分别送入两个专用路径：

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/cosmos-mot.svg" alt="Cosmos-3 双塔 MoT" style="max-width:100%;">
</div>

### Reasoner Tower

- 处理视觉理解、语言和任务上下文；
- 使用因果注意力；
- 产生场景语义、对象关系和任务意图；
- 不读取后方的带噪生成 token。

### Generator Tower

- 处理连续条件 token 和带噪轨迹 token；
- 使用全双向注意力；
- 可以读取 Reasoner 提供的上下文；
- 为后续 Flow Head 生成条件特征。

两塔的关系是单向条件化：

$$
\text{Reasoner}\rightarrow\text{Generator}.
$$

Generator 可以利用 Reasoner 的结果，但生成噪声不会反向污染 Reasoner 的状态。

---

## 5. 双塔内部的 Sparse MoE

MoT 和 MoE 分别负责不同层级的路由：

- **MoT**：决定 token 进入 Reasoner 还是 Generator；
- **MoE**：决定进入某个塔之后，由哪些专家处理该 token。

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/sparse-moe.svg" alt="Sparse MoE 结构" style="max-width:100%;">
</div>

每个 Sparse MoE 包含：

1. **Router**：计算 token 与各专家的匹配分数；
2. **Shared Experts**：始终激活，学习公共能力；
3. **Routed Experts**：数量较多，每个 token 只激活 Top-K 个；
4. **Weighted Aggregation**：合并共享专家和被选专家的输出。

Reasoner 和 Generator 使用不同的专家池：

| 专家池 | 主要处理内容 |
|---|---|
| Reasoner MoE | 空间关系、时间关系、语言指令、对象交互和任务推理 |
| Generator MoE | 轨迹结构、时间一致性、目标条件和连续运动生成 |

这种设计能够提高总参数容量，同时避免每个 token 都激活全部专家。

---

## 6. Flow 连续轨迹生成

Flow Head 接收以下条件：

- Reasoner 上下文 $h_t$；
- Generator MoE 特征 $z_t$；
- 当前智能体状态 $s_t$；
- 随机噪声轨迹 $A^0$。

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/flow-generation.svg" alt="Flow 轨迹生成" style="max-width:100%;">
</div>

生成过程从高斯噪声开始：

$$
A^0\sim\mathcal N(0,I),
$$

Flow Head 在多个积分步骤中预测轨迹的变化方向，最终将无结构噪声变换为完整、连续的轨迹块。

它与自回归动作生成的主要区别是：

- 不逐个产生离散动作 token；
- 一次联合建模完整未来轨迹；
- 更容易保持轨迹内部的连续性；
- 可以从不同初始噪声生成多条候选轨迹。

---

## 7. Flow-GRPO 强化学习

监督式 Flow Matching 让模型模仿训练数据，Flow-GRPO 则进一步根据实际任务奖励优化轨迹质量。

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/flow-grpo.svg" alt="Flow-GRPO 强化学习" style="max-width:100%;">
</div>

训练过程为：

1. 在相同条件下采样 $G$ 条候选轨迹；
2. 在环境中分别执行或评估这些轨迹；
3. 根据任务成功、目标进度、安全性、约束满足、平滑性和效率计算奖励；
4. 在组内比较奖励并计算相对优势；
5. 更新 Flow Head、Generator Router 和被激活的 Generator experts；
6. 使用 clipping 和参考策略约束更新幅度。

Reasoner 通常保持冻结或只进行保守微调，以减少强化学习对原有多模态理解能力的影响。

---

## 8. 训练流程

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/training-pipeline.svg" alt="训练流程" style="max-width:100%;">
</div>

### Stage 1：Cosmos-3 初始化

继承 Cosmos-3 的多模态理解、双塔交互和连续生成能力。

### Stage 2：MoE 转换与预热

将两塔的 dense FFN 替换为 shared experts 与 routed experts，并使 Router 的负载逐渐稳定。

### Stage 3：监督式 Flow 训练

使用示范轨迹训练 Flow Head，使模型先获得可靠的连续轨迹生成能力。

### Stage 4：Flow-GRPO

使用组内候选轨迹和环境奖励优化 Flow Head、Generator Router 与被激活专家。

---

## 9. 推理流程

<div align="center">
  <img src="/images/cosmos-moe-flowgrpo/inference-flow.svg" alt="推理流程" style="max-width:100%;">
</div>

推理采用滚动闭环方式：

1. 编码当前观测；
2. Reasoner 生成任务与场景上下文；
3. Generator Router 为轨迹 token 选择专家；
4. Flow Head 生成未来轨迹块；
5. 只执行前 $K$ 步；
6. 获取新观测并重新生成轨迹。

---

## 10. 各模块作用总结

| 模块 | 核心作用 | 主要输出 |
|---|---|---|
| Multimodal Encoders | 将不同输入映射到统一 token 空间 | AR/DM tokens |
| Reasoner Tower | 理解场景、语言与任务关系 | Reasoning context $h_t$ |
| Generator Tower | 处理连续生成 token | Generator features $z_t$ |
| Sparse MoE | 为不同 token 选择共享或专用专家 | Expert-enhanced features |
| Flow Head | 从噪声生成完整连续轨迹 | Trajectory chunk $A_t$ |
| Environment | 执行并评价候选轨迹 | Rewards $r_i$ |
| Flow-GRPO | 根据组相对优势优化策略 | Updated Flow policy |

---

## 11. 概念边界

下面区分 Cosmos-3 原生机制与本文方案扩展：

| 内容 | 属性 |
|---|---|
| Cosmos-3 双塔 MoT | Cosmos-3 原生机制 |
| AR/DM token arrangement | Cosmos-3 原生机制 |
| Dual-stream joint attention | Cosmos-3 原生机制 |
| Flow-matching Generator 与 action token | Cosmos-3 原生支持 |
| DeepSeek-style Sparse MoE | 本文架构扩展 |
| 连续轨迹 Flow Head | 本文任务扩展 |
| 面向轨迹策略的 Flow-GRPO | 本文训练扩展 |

因此，该方法最简洁的描述是：

> 在 Cosmos-3 双塔 MoT 中加入 DeepSeek-style Sparse MoE，将 Generator 专门化为连续轨迹 Flow Policy，并使用 Flow-GRPO 根据环境奖励优化 Flow Head、Generator Router 和被激活专家。

---

## 参考资料

1. NVIDIA et al. **Cosmos 3: Omnimodal World Models for Physical AI.** arXiv:2606.02800, 2026. <https://arxiv.org/abs/2606.02800>
2. Dai, D. et al. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.** arXiv:2401.06066, 2024. <https://arxiv.org/abs/2401.06066>
3. Liu, J. et al. **Flow-GRPO: Training Flow Matching Models via Online RL.** arXiv:2505.05470, 2025. <https://arxiv.org/abs/2505.05470>
