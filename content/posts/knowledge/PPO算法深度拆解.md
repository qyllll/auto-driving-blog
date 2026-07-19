---
title: "知识点拆解｜PPO 算法深度拆解：从 Policy Gradient 到 GRPO 的进化之路"
date: 2026-07-19
draft: false
categories: ["知识点拆解"]
tags: ["🎮 强化学习", "⚡ PPO", "📐 策略梯度", "⚙️ GAE"]
summary: "PPO 通过 clipped surrogate objective 和 GAE 实现稳定策略更新，是 RLHF 时代的核心算法。本文从 REINFORCE 出发，拆解 PPO 的每个关键组件，并对比 GRPO 的革新——去掉 critic，用组内相对优势替代。最后梳理 PPO/GRPO 在 VLA 驾驶模型（AlphaDrive、ReCogDrive、DriveVLA-W0）中的应用。"
weight: 11
---

## 🎯 一句话理解 PPO

> **PPO = 用 clip 把每次策略更新的步子限制在'安全范围'内，既让策略往好的方向改，又不让改的幅度太大导致崩盘。**

它是 OpenAI 在 2017 年提出的算法，至今仍是 RLHF、自动驾驶等领域的核心训练引擎。

---

## 📜 从 Policy Gradient 说起

### REINFORCE：最朴素的策略梯度

强化学习的目标是找到一个策略 $\pi_\theta(a|s)$ 最大化累积奖励的期望。最直接的方法就是 **REINFORCE**（也叫 Monte Carlo Policy Gradient）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t\right]$$

其中 $R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从 $t$ 时刻开始的累积折扣奖励。核心思想很直观：**如果一条轨迹的总奖励高，就增大这条轨迹中每个动作的概率；总奖励低，就降低它们**。

这里的 $\log \pi_\theta(a_t|s_t)$ 是关键——它来自 **log-probability trick**：

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

这个恒等式把"对概率的梯度"转化成"对 log 概率的梯度"，让我们能直接对概率分布做梯度上升，而不需要处理概率本身的约束。

### REINFORCE 的问题：高方差

REINFORCE 的梯度估计是无偏的，但**方差极大**。想象一下：同一个状态下采 10 条轨迹，奖励可能从 -100 到 +100 波动。直接用 $R_t$ 做权重，梯度噪声大得惊人，收敛极其缓慢。

方差来源有两个：
1. **奖励本身的随机性**：环境动态、对手行为都会导致相同动作得到不同奖励
2. **轨迹长度的累积**：$R_t$ 是未来所有奖励的和，越往后噪声越大

降低方差是策略梯度方法的核心课题，几乎所有后续改进都在围绕这个做文章。

| 方法 | 降方差手段 | 是否引入偏置 |
|-----|-----------|------------|
| REINFORCE | 无 | 无偏（高方差） |
| REINFORCE + baseline | 减基线 $b(s)$ | 无偏 |
| Actor-Critic | 用 $Q$ 替代 $R_t$ | 有偏（估计误差） |
| GAE | $\lambda$ 加权折中 | 可控偏置-方差权衡 |

---

## 🏗️ PPO 的四个核心组件

PPO 之所以成为经典，是因为它把策略梯度方法的每个环节都做了工程友好的设计。

### 组件一：重要性采样

PPO 是 **on-policy** 算法，但为了样本效率，它允许用旧策略 $\pi_{\text{old}}$ 采样的数据来更新新策略 $\pi_\theta$。这需要**重要性采样（Importance Sampling）** 来修正分布偏移：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

当新旧策略差异很小时，$r_t(\theta) \approx 1$，重要性采样引入的偏置可以忽略。但一旦分布偏移过大，重要性权重的方差会爆炸——这引出了 clip 的必要性。

### 组件二：Clipped Surrogate Objective

PPO 的目标函数是带 clip 的 surrogate objective：

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

这个设计妙在哪里？

| $A_t$ 符号 | 意思 | $r_t$ 行为 | clip 效果 |
|-----------|------|-----------|----------|
| $A_t > 0$ | 这个动作好 | 希望 $r_t$ 增大 | 但 $r_t > 1+\epsilon$ 时被 clip 住 |
| $A_t < 0$ | 这个动作差 | 希望 $r_t$ 减小 | 但 $r_t < 1-\epsilon$ 时被 clip 住 |

**clip 的本质是'悲观'约束**：对于好动作，你最多能增加到 $(1+\epsilon)$ 倍概率；对于差动作，你最多能减少到 $(1-\epsilon)$ 倍。这确保了单次更新不会让策略发生剧烈变化。

$\epsilon$ 是 PPO 最重要的超参数，典型值 0.1~0.3。太小更新太慢、太大失去约束意义。

### 组件三：GAE（Generalized Advantage Estimation）

GAE 是 PPO 降方差的关键武器。它平衡了 bias 和 variance，用一个参数 $\lambda$ 在两者间光滑插值：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 TD-error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

| $\lambda$ | 含义 | bias | variance |
|-----------|------|------|----------|
| $\lambda = 0$ | 单步 TD | 高偏置 | 低方差 |
| $\lambda = 1$ | Monte Carlo | 无偏 | 高方差 |
| $\lambda \in (0,1)$ | 折中 | 可控 | 可控 |

典型值 $\lambda = 0.95$，在自动驾驶场景中常调高到 0.98 以利用更长期的奖励信号（如安全到达终点这种稀疏奖励）。

GAE 需要 **value network（critic）** 提供 $V(s)$ 估计，这也正是 GRPO 要砍掉的目标。

### 组件四：Value Network（Critic）

价值网络 $V_\phi(s)$ 估计状态 $s$ 的平均期望回报，它的作用是充当**基线（baseline）**：

$$A_t = Q(s_t, a_t) - V(s_t)$$

Critic 通过最小化 MSE 损失训练：

$$\mathcal{L}^{\text{VF}}(\phi) = \mathbb{E}_t\left[(V_\phi(s_t) - R_t)^2\right]$$

| Critic 的问题 | 具体表现 |
|-------------|---------|
| **同 policy 一样大** | LLM 场景下显存翻倍 |
| **收敛困难** | 价值估计需要大量数据 |
| **带偏 advantage** | critic 估计不准时，优势信号被污染 |
| **双网络调参** | policy 和 value 的 learning rate 互相影响 |

PPO 的完整损失是 policy loss、value loss 和 entropy bonus 的加权和：

$$\mathcal{L}^{\text{PPO}}(\theta, \phi) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\phi) + c_2 \mathcal{H}(\pi_\theta)$$

---

## 🔄 PPO 算法步骤

PPO 的完整训练循环可以分解为五个步骤：

### Step 1：收集 Rollouts

用当前策略 $\pi_{\theta_{\text{old}}}$ 与环境交互，收集 $N$ 条完整轨迹（或 $T$ 个 timestep）。每条轨迹存储 $(s_t, a_t, r_t, s_{t+1})$ 四元组。这一步最耗时——**PPO 的样本效率瓶颈就在这里**。

### Step 2：计算 GAE

利用收集到的轨迹和 critic 网络 $V_\phi(s)$，计算每条轨迹每个时间步的 GAE $\hat{A}_t$ 和 discounted return $R_t$。这是 off-line 计算，不涉及网络前向。

### Step 3：优化 Clipped Loss

将数据组织成 mini-batch，对 PPO 损失做多 epoch（通常 3~10 epoch）优化。这里每个 epoch 都会重新计算 $\pi_\theta(a_t|s_t)$ 以更新重要性比 $r_t$，但**始终用 Step 1 收集的旧策略数据**——这就是 importance sampling 的关键。

### Step 4：更新 Policy

用优化后的 $\pi_\theta$ 替换 $\pi_{\theta_{\text{old}}}$，进入下一轮迭代。

### Step 5：重复

回到 Step 1，用新策略重新收集数据。

这套流程的工程密码是 **'收集一次、复用多步'**——在旧数据上做多步梯度更新，大幅提升样本效率。但如果更新步数太多，新旧策略差异过大会导致 importance sampling 失效，所以 clip 和 early stopping 是必要的安全阀。

| 步骤 | 计算量占比 | 关键瓶颈 |
|-----|-----------|---------|
| 收集 rollouts | ~80% | 环境交互速度 |
| GAE 计算 | ~5% | 矩阵运算 |
| 多 epoch 优化 | ~15% | 网络前向/反向 |

---

## ⚡ GRPO vs PPO：革命性的简化

GRPO（Group Relative Policy Optimization）由 DeepSeek 提出，核心洞察是：**能否去掉 critic 网络？**

### 核心差异

| 维度 | PPO | GRPO |
|------|-----|------|
| **基线来源** | Critic 网络 $V(s)$ | 组内 reward 均值 |
| **优势函数** | $\hat{A}^{\text{GAE}}_t$ | $(r_i - \bar r) / \sigma$ |
| **额外网络** | Policy + Critic | 仅 Policy |
| **显存开销** | ~2x | ~1x |
| **样本使用** | 跨时间步串联 | 同 prompt 并行采样 |
| **适用场景** | 连续控制、游戏 | LLM、可验证 reward |

### 为什么 GRPO 在某些场景更优？

**1. 显存减半**：对百亿参数模型，去掉 critic 直接省一半显存，这是 GRPO 在大模型时代胜出的直接原因。

**2. 规避 critic 估计误差**：critic 本身需要训练、可能不准，不准的 baseline 会污染 advantage 信号。GRPO 用 real reward 的统计量替代学习到的估计。

**3. 天然适配可验证 reward**：当 reward 来自客观规则（碰撞检测、数学答案对错），组内比较比 critic 估计更直接可靠。

**4. 训练更稳定**：单网络减少了调参复杂度，group size 是唯一新引入的超参数。

### GRPO 的代价

GRPO 并非免费午餐。它用 **样本量的增加** 换 critic 的去除：

- 每个 prompt 需要采样 $G$ 个回答（$G$ 通常 8~64）
- 当 $G$ 太小，组内统计的方差大、信号弱
- 当 reward 区分度不高时，组内比较的优势不明显

PPO 和 GRPO 的关系不是替代而是互补：

> **PPO 适合'连续交互、单步反馈'的场景（游戏、机器人控制），GRPO 适合'批量评估、可验证 reward'的场景（LLM、驾驶轨迹优化）。**

---

## 🧠 为什么 PPO / GRPO 在工作？

### 平衡探索与利用

PPO 的 entropy bonus $c_2 \mathcal{H}(\pi_\theta)$ 直接鼓励策略保持随机性——entropy 越高，动作分布越均匀，探索越充分。随着训练推进，entropy 自然衰减，策略逐渐从探索转向利用。

GRPO 虽然没有显式 entropy term，但**组内采样天然是一种探索**：每次对同一 prompt 采样多个候选，候选间的多样性就是探索的体现。

### 约束更新大小

策略梯度最大的敌人是 **destructive updates**——某次更新让策略"翻车"，后续再多的训练也救不回来。PPO 的三道防线：

1. **Clip**：限制单步重要性比的范围
2. **KL 约束（可选）**：强制新旧策略的 KL 散度在阈值内
3. **Early stopping**：当 KL 超限时提前终止本轮更新

GRPO 的约束更轻量，但通过组内归一化同样限制了梯度信号的幅度，避免了策略"跳崖"。

### Why Works 的核心直觉

PPO 成功的关键在于 **'可信区域内的逐步改进'**：每次更新都确保新策略不会离旧策略太远，在每个小半径内寻找改进方向。这好比登山时的"之"字形路线——每一步都很短，但累计起来能爬很高的山。

---

## 🚗 PPO / GRPO 在 VLA 中的应用

随着 VLA（Vision-Language-Action）模型成为端到端驾驶的主流范式，PPO 和 GRPO 作为策略优化工具开始密集出现。

### AlphaDrive：GRPO 驾驶策略优化

AlphaDrive 是首个将 GRPO 系统应用于自动驾驶策略的代表性工作。它将每个驾驶场景视为一个"prompt"，采样多条候选轨迹，用可验证的驾驶规则（碰撞检测、车道保持、限速合规）作为 reward，通过 GRPO 优化生成策略。关键贡献在于证明了 GRPO 在**连续动作空间**下的可行性——通过将轨迹参数化为 action token 序列，与 LLM 式的 autoregressive 生成兼容。

### ReCogDrive：PPO 用于驾驶认知对齐

ReCogDrive 使用 PPO 对驾驶 VLM 进行 **推理-行动对齐（reasoning-action alignment）**。它首先让模型输出推理链（Chain-of-Thought reasoning about driving situation），然后基于推理生成驾驶动作。PPO 的 critic 网络在训练中学会了评估"当前状态下好的推理应该长什么样"，从而引导模型生成既安全又有解释性的驾驶决策。

### DriveVLA-W0：PPO 微调 VLA 基础模型

DriveVLA-W0 采用两阶段训练：第一阶段用大规模驾驶数据做行为克隆（BC）预训练，第二阶段用 PPO 做 RL 微调。PPO 的优势在此得到充分发挥——critic 网络在预训练阶段已经学到不错的状态价值估计，RL 阶段只需精调策略头，收敛快速且稳定。

### 方法对比

| 工作 | RL 算法 | 优化对象 | Reward 来源 | 特色 |
|------|--------|---------|------------|------|
| AlphaDrive | **GRPO** | 轨迹 token 序列 | 可验证规则 | 首个 GRPO 驾驶应用 |
| ReCogDrive | **PPO** | 推理-行动联合 | VLM 评分 | 推理链对齐 |
| DriveVLA-W0 | **PPO** | 动作头参数 | 驾驶模拟器 | 两阶段 BC + RL |
| Flow-GRPO | **GRPO** | 扩散去噪过程 | 图像/轨迹质量 | 生成式策略 RL |

### 趋势观察

VLA 的 RL 微调正在发生两个范式转变：

1. **从 PPO 走向 GRPO**：随着模型规模增大，critic 的显存成本难以承受，GRPO 的趋势性优势越来越明显
2. **从模拟器 reward 走向可验证 reward**：用规则（碰撞、舒适度）替代模拟器评分，使 reward 更透明、更可控

---

## 📝 个人思考

PPO 之所以能在 2017 年提出后统治 RL 领域近十年，归根结底是它**在'算法效果'和'工程可用'之间找到了黄金平衡点**。clip 机制不漂亮——它没有理论上的单调改进保证，没有 TRPO 那种 elegant 的 KL 约束推导——但它就是好用、好调、好收敛。在工程世界里，**'work'比'prove'重要得多**。

GRPO 对 PPO 的革新同样遵循这个逻辑。从理论上看，用组内均值替代 critic 应该会引入更高的方差（因为只用了 $G$ 个样本而非全局价值函数），但实践中它在 LLM 场景里表现更好。**这说明在超大模型场景下，算法的 bottleneck 已经从'统计效率'转向了'计算效率'**——显存和算力的约束比方差更致命。

对自动驾驶从业者来说，我认为最重要的是理解 **PPO/GRPO 是一套可插拔的训练方法论，而不是固定的算法实现**。你完全可以做 GRPO + GAE 的混合、PPO + 组内 baseline 的杂交，关键在于根据场景的 reward 结构、模型规模和算力预算选择最合适的配置。**RL 算法选的不是'最好的'，而是'最不坏的'**。

---

*📖 这是知识点拆解系列的第 11 篇。从 REINFORCE 到 PPO 再到 GRPO，策略梯度这条路走了近三十年，远没有到终点。*
