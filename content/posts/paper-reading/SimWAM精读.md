---
title: "论文精读｜SimWAM：把世界模型当'训练老师'，用 Flow-GRPO 让 Action Expert 学会'自己开车'"
date: 2026-08-17
draft: false
categories: ["论文精读"]
tags: ["🌊 Flow Matching", "🎮 强化学习", "⚡ GRPO", "🌍 世界模型", "🚗 自动驾驶"]
summary: "SimWAM 用 joint flow matching 把预训练视频专家的交通动态先验'蒸馏'进轻量级 Action Expert，再用隔离注意力掩码让动作分支彻底摆脱未来帧依赖；随后把确定性 Flow ODE 转成可探索的 SDE，用 GRPO 直接优化 NAVSIM 组合驾驶奖励，突破模仿学习的上限。91.5 PDMS 新 SOTA，推理时延大幅低于 imagine-then-act 的 WAM，并零样本迁移到 nuScenes。"
weight: 8
---

## 📄 论文信息

- **标题**：*SimWAM: A Simple World Action Model for End-to-End Autonomous Driving*
- **作者机构**：华中科技大学（白翔团队）× **东风汽车研发总院**（Zongchuang Zhao, Xin Zhou, Tianyang Xu, Zhengyang Sun, Kaixuan Zhou, Honglin Li, Dingkang Liang, Xiang Bai）
- **arXiv**：[2608.07468](https://arxiv.org/abs/2608.07468)
- **代码**：[github.com/H-EmbodVis/SimWAM](https://github.com/H-EmbodVis/SimWAM)
- **模型权重**：[huggingface.co/H-EmbodVis/SimWAM](https://huggingface.co/H-EmbodVis/SimWAM)
- **一句话总结**：**未来视频预测只当"训练期的老师"，不参与推理**。SimWAM 用 joint flow matching 把预训练视频生成模型的交通动态先验"迁移"进一个轻量级 Action Expert，再用隔离注意力掩码让动作分支完全独立于未来帧；训练完之后把整个视频分支扔掉，留下一个自包含的 planner 直接出轨迹。最后，它把确定性的 Flow ODE 改写成可探索的 SDE，用 **GRPO** 直接优化 NAVSIM 组合驾驶奖励，**在 NAVSIM navtest 上刷到 91.5 PDMS 新 SOTA**，推理时延却远低于同级的 world-model-based planner。

> 🎯 一句话记忆点：**SimWAM = "视频生成是教练，Action Expert 是运动员"——训练时教练带着练（动态先验），比赛时只有运动员上场（无未来帧推理），再用 GRPO 给运动员加练（RL 优化奖励）。**

---

## 🤔 要解决什么问题？

### 背景：端到端规划的两种"补课"方式

端到端自动驾驶（E2E AD）直接用原始传感器输入预测轨迹。传统 E2E planner（UniAD、VAD 这类）本质都是**模仿策略（imitation policy）**：从日志数据里复现专家行为。问题在于：交通语义、用户意图、场景动态这些信息，专家轨迹里只有**隐式**的体现，模型学到的只是"照着数据开车"。

于是两条"补课"路线兴起：

| 路线 | 代表 | 补什么课 | 问题 |
|------|------|---------|------|
| **VLA（视觉-语言-动作）** | AutoVLA、ReCogDrive | 补**语义理解与推理** | 推理链与动作松散耦合，时序动态仍是间接建模 |
| **WAM（World-Action Model）** | DriveLaW、DriveWAM、Epona | 补**交通动态先验**（显式建模未来场景演化） | imagine-then-act，推理时还得生成未来帧，时延爆炸 |

### 核心痛点：imagine-then-act 让视频生成进了实时链路

现有驾驶 WAM 几乎都走 **imagine-then-act** 的因子分解：

$$p_\theta(a_{t+1:t+H} \mid o_t, s_t, l) = \int p_\theta(z_{t+1:t+N} \mid o_t, s_t, l)\, p_\theta(a_{t+1:t+H} \mid o_t, s_t, l, z_{t+1:t+N})\, dz$$

**先合成未来场景的 latent `z`，再以它作为条件生成轨迹。** 这意味着：

1. **昂贵的视频生成被塞进了实时规划环路**——每一步决策前都要先"想象未来"，推理时延爆炸（见下方图 1，DriveLaW/DriveWAM 时延高达 1-3 秒级）；
2. **动作预测与未来场景预测紧密耦合**——视频 token 的高维视觉表示会干扰低维动作空间，代表性错配导致泛化下降。

![图 1：SimWAM 与现有 WAM planner 的时延-性能对比（源图 arXiv:2608.07468 Figure 1）。横轴是推理时延（ms），纵轴是 NAVSIM PDMS。世界模型系方法（DriveLaW、DriveWAM、Epona、DriveVLA-W0）分布在右侧高时延区，SimWAM 以接近纯 planner 的时延拿到最高 PDMS。](/images/simwam/fig1_latency_pdms.png)

#### 🔍 图 1 怎么读？

一张图说明"为什么要避免 imagine-then-act"：

- **横轴 = 推理时延（毫秒，对数/线性刻度）**，纵轴 = NAVSIM PDMS（越高越好）。任何 planner 都想往**左上角**跑（又快又好）。
- **右半区是"想象派"**：DriveLaW、DriveWAM、Epona 这类 imagine-then-act 方法，每一步决策前都要先生成未来帧——视频生成是逐帧串行的，所以时延普遍跑到**数百毫秒到数秒级**（图上显著偏右）。
- **左半区是"纯 planner"**：直接出轨迹的方法时延很低（几十毫秒级），但此前 PDMS 明显不如"想象派"——性能和效率不可兼得。
- **SimWAM 的位置**：纵轴拿到最高的 PDMS（91.5），横轴却停在**纯 planner 的量级**——它靠"训练期视频监督 + 推理期删视频分支"同时吃到了两边的红利：世界模型的精度 + 纯规划器的速度。

> 🎯 这张图的潜台词是：**"想象未来"真正值钱的地方在训练期（学动态先验），而不是推理期（实时生成）**。谁能把视频生成从实时链路里挪出去，谁就同时赢得精度和速度。

### 第二个问题：Action Expert 只吃模仿饭，上限锁死

即便摆脱了未来帧依赖，WAM 里 Action Expert 的训练目标仍是**专家轨迹模仿（BC）**。模仿学习的三个老毛病一个不少：

- **分布偏移**：只在专家轨迹附近学得好，上线跑偏没人拉回来；
- **无法超越专家**：上限就是专家数据的质量，而人类数据本身可能保守、犹豫；
- **真正目标不可微**：碰撞率、舒适 jerk、规则遵守这些驾驶质量指标，BC 根本优化不了。

> **SimWAM 的立场**：未来视频预测能提供更丰富的交通动态信息，但它**没有改变轨迹学习本身"以模仿为主"的优化目标**。要在保留动态先验的同时突破模仿限制，就必须引入**面向驾驶奖励的策略优化**——这就是 RL 出场的位置。

---

## 💡 核心思路：三层解耦

SimWAM 的核心设计可以拆成三句话：

1. **训练/推理解耦**：视频生成**只在训练期当监督**，推理期整个视频分支删掉，只留 Action Expert 直接出轨迹（图 2 右侧 "Removed"）；
2. **信息流解耦**：用一个**隔离注意力掩码（Isolated Attention Mask）**让 Action Expert 永远看不见未来帧 token，从结构上保证"动作不依赖未来"；
3. **学习目标解耦**：先 joint flow matching 做"动态先验迁移"（补时序建模），再 Flow ODE→SDE + GRPO 做"奖励优化"（补驾驶质量），两阶段互补。

![图 2：SimWAM 整体架构（源图 arXiv:2608.07468 Figure 2）。训练阶段：Video DiT（继承 Wan2.2-5B 视频先验）与轻量 Action DiT（hidden=1024）通过共享注意力流联合训练，隔离注意力掩码限制 action token 只能看当前观测；推理与 RL 阶段：Video 分支被移除，只保留 Action DiT 直接出轨迹。](/images/simwam/fig2_architecture.png)

#### 🔍 图 2 逐块拆解（从上到下、从左到右）

这张图是整个论文的"总装图"，分三块读：

**① 左上——输入侧（三种模态进同一编码器）：**

| 输入 | 编码器 | 变成什么 |
|------|--------|---------|
| 当前帧 `o_t`（单前视相机） | **Video VAE**（继承自 Wan2.2-5B，图像→latent token） | `z(o_t)`：当前观测的 latent 表示 |
| 导航指令 `l` | **T5 文本编码器** | 文本 embedding（通过 cross-attention 注入） |
| 自车状态 `s_t`（速度/加速度/横摆角速度） | **Ego Encoder（MLP）** | 一个低维向量 |

三个来源的表示**拼进同一个注意力流**——这就是"共享注意力"的入口。注意图里 Video VAE 只编码**当前帧**，未来帧是"目标"（要生成的对象），不是输入。

**② 中部——共享注意力流 + 隔离掩码（图里最关键的画法）：**

图中会画出三类 token：`z(o_t)`（当前观测）、未来帧 latent `z_{t+1:t+N}`、action token。箭头/连线表示注意力允许看谁：

- 未来帧 token → 能看 `z(o_t)`（生成未来帧需要知道当前长什么样）；
- action token → 也能看 `z(o_t)`（规划需要知道当前观测）；
- **未来帧 ↔ action：互不可见**（这就是"隔离"——图上会用一条隔断或不同的箭头颜色标出来）。

读图要点：**action 的输入箭头永远不指向未来帧那一侧**。图上的掩码块（Isolated Attention Mask）不是独立的网络，而是一张注意力"权限表"，在注意力计算时把"action 看未来帧"的条目直接遮成 `-∞`。

**③ 右侧——两阶段的"同一张图、两种读法"：**

架构图通常左右分成两个面板：左半是**训练期**（Video DiT 和 Action DiT 都在），右半是**推理/RL 期**（Video DiT 被画成删除线/虚线框，标注 "Removed"）。也就是说图 2 同时画了"训练时的全貌"和"部署时只剩什么"——读图时务必看清自己看的是哪一半，这是理解"训练-推理解耦"的关键。

**先看宏观数据流（训练期）：**

1. **编码**：把三种输入映射成条件。
   - 当前帧 → Video VAE → `z(o_t)`（当前观测 latent）；
   - 导航指令 `l` → T5 Text Encoder；
   - 自车状态 `s_t` → Ego Encoder（MLP）。
   - 三者一起进入**共享注意力流**，但受隔离注意力掩码约束（未来视频与动作互不可见）。
2. **双专家并行**：
   - Video DiT：生成未来帧 `z_{t+1:t+N}`（flow matching 目标，提供动态监督）；
   - Action DiT：预测轨迹速度场 `v_θa`（flow matching 目标，生成轨迹）。
3. **训练完成后**：Video DiT 整个删除，只剩 Action DiT 部署。

**推理期数据流（极简）：**

`当前帧 → Video VAE → z(o_t) → Action DiT → 轨迹 a_{t+1:t+H}`（指令与 Ego 状态同样作为条件注入 Action DiT；**全程无未来帧生成**）。

---

## 🧩 架构讲解：输入 → 模型 → 输出

> 📍 **架构图在这**：SimWAM 的完整架构图是 **图 2（Figure 2，见上方"核心思路"章节的图 2）**，它同时画了训练期和推理期两套结构。下面先用一张我自己画的流程图把**整体数据流**讲清楚——特别注意"训练期"和"推理期"是**两套不同的结构**，这是 SimWAM 最核心的 design choice。

![SimWAM 数据流总览（自绘 SVG 流程图）：训练期 Video Expert + Action Expert 经隔离掩码联合训练（视频当"老师"）；推理/RL 期视频分支删除，只剩 Action Expert 直接出轨迹。](/images/simwam/flow_architecture.svg)

#### ① 输入是什么？（3 类）

| 输入 | 类型 | 编码器 | 变成什么 |
|------|------|--------|---------|
| **当前帧 `o_t`**（单前视相机） | 图像 | **Video VAE**（继承 Wan2.2-5B） | `z(o_t)` 当前观测 latent |
| **导航指令 `l`** | 文本 | **T5 文本编码器** | 文本 embedding（cross-attention 注入） |
| **自车状态 `s_t`** | 向量 | **Ego Encoder（MLP）** | 低维状态向量 |

#### ② 中间经历了什么？（分训练/推理两种）

- **训练期**：三个输入编码后进入**共享注意力流**（受隔离掩码约束）→ 两条专家并行：Video DiT 生成未来帧（提供动态先验）、Action DiT 预测轨迹 → 联合 flow matching 训练；
- **推理/RL 期**：Video DiT **整个删除**，只剩 Action DiT（10 步 ODE）直接出轨迹；RL 阶段再叠加 Flow-GRPO（ODE→SDE + GRPO）优化驾驶奖励。

#### ③ 输出是什么？（1 类）

**一条未来 4 秒的轨迹** `a_{t+1:t+H} = (x, y, θ)`（8 个 waypoint @ 2Hz）。**全程无未来帧生成**——这就是"训练/推理解耦"的含义：视频只在训练期当老师。

---

## ⚙️ 模块一：Video-Action Co-training（动态先验迁移）

### 3.1 问题形式化

SimWAM 的定义非常简单直接。给定前视相机观测 `o_t`、自车状态 `s_t`（速度、加速度、横摆角速度）和导航指令 `l`，planner 预测自车坐标系下的轨迹：

$$a_{t+1:t+H} = (a_{t+1}, \dots, a_{t+H}), \quad a_i = (x_i, y_i, \theta_i)$$

与 imagine-then-act 不同，SimWAM 的策略接口是**直接的**：

$$p_\theta(a_{t+1:t+H} \mid o_t, s_t, l) = p_\theta\!\left(a_{t+1:t+H} \mid z(o_t), s_t, l\right)$$

其中 `z(o_t)` 是当前观测的表示。**交通动态知识全部在训练期获得**，推理既不需要未来场景 latent，也不需要辅助运动模块——和直接轨迹预测一样高效。

### 3.2 两个专家：Video Expert + Action Expert

| 专家 | 实现 | 参数 | 职责 |
|------|------|------|------|
| **Video Expert** | Video Diffusion Transformer，从 **Wan2.2-5B** 初始化（含 video VAE + T5 文本编码器） | ~5B | 生成未来帧，提供交通动态先验 |
| **Action Expert** | 轻量 Diffusion Transformer | hidden=1024（约 1.02B，见消融） | 预测轨迹，唯一部署的模型 |

- **Video Expert**：VAE 把每帧图像映射成 latent token；导航指令通过 T5 cross-attention 进入；当前帧作为干净条件（clean condition），`N` 个未来帧被加噪后重建。这就是标准的视频生成目标，**不引入任何驾驶专用预测模块**——纯粹用视频生成的"副产物"给 Action Expert 供应交通动态先验。
- **Action Expert**：一个小 MLP 嵌入 ego 状态，条件为 `c = {z(o_t), s_t, l}`，用 flow matching 预测轨迹速度场 `v_{θa}(a^τ_{t+1:t+H}, τ, c)`。积分 ODE 就把噪声映射成规划轨迹。

### 3.3 Isolated Attention Mask（隔离注意力掩码）

这是**两个专家解耦的唯一结构改动**，也是最精妙的设计：

> 共享注意力流里包含三类 token：**当前观测 latent `z(o_t)`**、**未来帧 latent `z_{t+1:t+N}`**、**action token**。掩码规则：
> - 未来帧 token 和 action token **都能看当前观测 `z(o_t)`**；
> - 但未来帧与 action **彼此不可见（mutually invisible）**。

换句话说：Action Expert 从头到尾**接触不到未来帧 token**，只能从"当前观测的表示"里学东西。视频生成在此纯粹是**训练信号**——它通过共享注意力把交通动态"注入"到观测表示里，从而间接塑造 Action Expert。

**为什么这比让 action 看未来帧更好？** 看论文的掩码消融（Tab. 3）：

| Mask | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|------|------|------|------|------|-------|
| Bidirectional（双向） | 98.4 | 98.0 | 84.7 | 95.1 | 90.2 |
| Action→video（动作看未来帧） | 98.5 | 97.8 | 84.3 | 95.5 | 90.1 |
| **Isolated（隔离）** | **98.7** | **98.0** | 83.9 | **95.9** | **90.3** |

有意思的是：**让 action 分支看到视频 token 并没有带来可衡量的收益**，而隔离设计反而拿到最高 PDMS（90.3）以及最强的 NC 和 TTC。原因在于——暴露未来帧只会让动作预测耦合进"可能生成错的"未来内容，而共享当前观测表示 + 联合训练已经足够把动态先验传过去。

> 🔑 这个掩码同时带来三个连锁好处：
> 1. **训练-推理一致**：训练时动作就不看未来帧，推理时把视频分支删掉，两者状态完全对齐，没有 train/inference mismatch；
> 2. **视频专家可替换**：Action Expert 只通过共享观测表示与视频专家交互，换掉视频 backbone（Wan2.1-1.3B / Wan2.2-5B / Cosmos-Predict2.5）不影响动作分支和推理管线；
> 3. **独立缩放**：视频专家和动作专家的容量是两个互相独立的控制旋钮——大视频模型只影响训练监督质量，不影响部署成本；Action DiT 可单独改宽改深来满足时延预算。

### 3.4 Joint Flow Matching 训练目标

SimWAM 用 **rectified flow** 对轨迹和未来帧统一建模。给定干净目标 `x` 和高斯噪声 `ϵ ~ N(0, I)`，线性插值：

$$x_\tau = (1-\tau)\,x + \tau\,\epsilon, \quad \tau \in [0,1]$$

中间状态的速度恒为 `ϵ - x`，网络 `v_θ` 在条件 `c` 下预测这个速度：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x,\epsilon,\tau} \left\| v_\theta(x_\tau, \tau, c) - (\epsilon - x) \right\|_2^2$$

采样时沿**概率流 ODE** 积分：`dx_τ = v_θ(x_τ, τ, c) dτ`，从噪声（τ=1）走到数据（τ=0）。这个"直线路径 + 一步到位"的特性是 flow matching 相比扩散模型的最大优势（少步采样、训练目标简单稳定）。

SimWAM 的联合训练目标就是把两个专家的 FM 损失加起来：

$$\mathcal{L} = \mathcal{L}^{\text{act}}_{\text{FM}} + \lambda\, \mathcal{L}^{\text{vid}}_{\text{FM}}, \quad \lambda = 1$$

- `L_act_FM`：在动作轨迹 `a_{t+1:t+H}` 上实例化 Eq.1；
- `L_vid_FM`：在 future-frame latent `z_{t+1:t+N}` 上实例化 Eq.1。

两个专家**不共享参数**，只在注意力流里交换信息。联合训练后，Video Expert 学到的"世界怎么演化"就通过共享观测表示"渗透"进了 Action Expert——这就是**动态先验迁移**的机制。

> 💡 这里值得停下来想一层：为什么视频监督能帮动作预测？因为**两者在联合训练中共享了同一个当前观测表示**。视频生成迫使这个表示必须"足以预测未来"，于是它就被迫编码了交通对象的速度、方向、交互关系——Action Expert 从这样一个"面向未来"的表示上做规划，自然比从纯感知表示上做规划强。这其实和 DriveVLA-W0 的"世界模型放大数据"逻辑同源：**未来预测是一种隐式的表征学习正则**。

---

## 🎮 模块二：Flow ODE → SDE + GRPO（强化学习优化驾驶策略）

> 这是本文要**重点拆解**的部分，也是用户最关心的地方。SimWAM 在这里几乎是把我们博客里的 **Flow-GRPO**（arXiv:2505.05470，腾讯 ARC Lab）整套"搬"进了自动驾驶——**论文自己也在 Eq.2 里明确写了 "Following Flow-GRPO [30]"**。但它不是简单的照抄，而是做了几个关键的适配。下面我们一层层拆。

### 4.1 为什么纯 Flow ODE 没法做策略优化？

co-training 结束后的 Action Expert 是个 flow matching 模型，本质仍是**模仿策略**。要让它突破模仿、直接优化驾驶质量，就得用 RL。但 RL 用不了原封不动的 flow ODE，有两个硬伤：

| 硬伤 | 解释 | 后果 |
|------|------|------|
| **① 确定性，无探索** | ODE 给定初始噪声就确定性地生成唯一一条轨迹 | 无法在同一场景生成"多样性候选轨迹"给 RL 挑选/比较，探索不出新的驾驶行为 |
| **② 无解析过渡密度** | ODE 是确定性映射，没有概率密度函数（PDF） | 策略梯度 / importance ratio $r = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})$ 根本算不出来——GRPO 的损失函数建在 log-prob 上 |

### 4.2 ODE → SDE：Flow-GRPO 的核心改造，SimWAM 直接引用

**Flow-GRPO 的思路**（也是 SimWAM 照做的）：把 flow step 从确定性的 ODE 改造成一个**保持边缘分布的 SDE（marginal-preserving SDE）**。关键在于：SDE 的每一步转移都从一个**解析的高斯分布**里采样，于是 log-prob 可以显式写出来——PPO/GRPO 就有了"可计算的策略密度"。

SimWAM 论文 Eq.2 给出了精确的 SDE 形式：

$$dx_\tau = \left[ v_\theta(x_\tau, \tau) + \frac{\sigma_\tau^2}{2\tau}\Big(x_\tau + (1-\tau)\, v_\theta(x_\tau, \tau)\Big) \right] d\tau + \sigma_\tau \, dw, \quad \sigma_\tau = a\sqrt{\frac{\tau}{1-\tau}}$$

其中 `dw` 是 Wiener 增量，`a` 控制噪声尺度。这个公式的**关键性质**是：它和原 ODE **共享相同的边缘分布 `p_τ(x_τ)`**——也就是说，SDE 采样出来的轨迹"长得还是原来的轨迹该长的样子"（保持可行性），只是多了随机探索的分量。

每次 Euler-Maruyama 步得到各向同性高斯转移：

$$\pi_\theta(x_{\tau-\Delta\tau} \mid x_\tau) = \mathcal{N}\!\left(\mu_\theta(x_\tau, \tau),\; \sigma_\tau^2 \Delta\tau\, I\right)$$

这个**可解析的转移密度**就是整个 RL 改造的支点——它让 importance sampling（旧策略下采样的数据用新策略的 log-prob 来重加权）变成可能。

> 🧠 **用大白话讲 ODE→SDE 是什么**：想象你从山上往下滑。ODE 是你只走一条确定的滑道，闭着眼从同一位置开始永远到同一个终点；SDE 是滑道上额外撒了一层"随机颠簸"，每次滑的路径略有不同，但**滑到的区域分布和原来一致**。这些"不同的路径"就是 RL 要的候选轨迹——有的更激进、有的更保守，让模型有机会比较哪条"驾驶质量更高"。

### 4.3 GRPO：组内相对优势，丢掉 Critic

有了可计算的 log-prob，接下来就是 GRPO 登场。GRPO（源自 DeepSeek）的核心洞察：**用"组内相对比较"替代 PPO 的 critic 网络**。

对同一个场景采样一组 `G` 条候选轨迹，各自拿到奖励 `R(τ_i)`，组内优势为：

$$A_i = \frac{R(\tau_i) - \bar{R}}{\sigma_R + \epsilon}$$

- `\bar{R}`：同场景 `G` 条候选的奖励均值；
- `\sigma_R`：组内标准差。

**为什么"组内相对"比"绝对奖励"合理？** 不同场景的驾驶难度天差地别：直路场景奖励天然高，无保护左转奖励天然低。如果全局比较，模型会一味偏向"简单场景的路径"，复杂场景能力退化。按场景分组归一化，把"绝对好不好"换成"在同场景候选里排第几"，自动抵消场景难度差异——**这在驾驶上比在图像生成上更关键**，因为场景间的差异比 prompt 间的差异大得多。

**GRPO 的 clipped 策略更新**（沿用 DeepSeekMath / PPO-clip 的写法）：

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\left[\min\!\left(\frac{\pi_\theta(a \mid o)}{\pi_{\theta_{\text{old}}}(a \mid o)}\, A_i,\; \operatorname{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) A_i\right)\right]$$

| 组件 | PPO | GRPO |
|------|-----|------|
| baseline 来源 | Critic 网络估计 | 组内奖励均值 `\bar{R}` |
| 是否需要 critic | ✅ 必须 | ❌ 不需要 |
| 显存/调参成本 | 高（双模型） | 低 |
| advantage | Q − V | (r_i − \bar r)/σ |

**"终点打分、整条链领奖"**：轨迹是 `10` 步 flow/SDE 采样出来的，中间 latent 没有语义，reward 只能对**最终 decode 后的完整轨迹**打分。GRPO 的语义是：完整采样完成后统一算 advantage，然后把这个 advantage 广播到该轨迹每个去噪步的 log-prob ratio 上。**不是每一步重新打分，而是终点一次性打分，整条去噪链一起承担。**

### 4.4 SimWAM 的 RL 具体配置（细节拉满）

| 配置项 | SimWAM 设定 | 说明 |
|--------|-------------|------|
| **更新对象** | 只更新 Action Expert 的 **rank-32 LoRA**（scale α=16）适配器，加在 attention projections 上 | 保留蒸馏好的运动先验，且保持 planner 结构简单；Video Expert 在 RL 阶段完全不在场 |
| **组大小 G** | 每场景采样 **G=8** 条候选轨迹 | Flow-GRPO 图像版通常用 16-24，驾驶场景少一些（奖励评估贵） |
| **奖励函数** | **NAVSIM PDM reward**（组合驾驶奖励） | 见下方公式 |
| **RL 数据选择** | 只挑 imitation 后 **PDMS < 90 的困难 navtrain 场景** | 困难场景的候选轨迹间差异大，reward 信号信息量足；简单场景模仿已够好，RL 信号稀释 |
| **学习率** | 5×10⁻⁵ | 比联合训练（1×10⁻⁴）更小，微调性质 |
| **推理时** | 10 步 denoising，CFG=1.0 | RL 阶段用 SDE 采样，推理用确定性 ODE 也无妨 |

**组合驾驶奖励（PDM Reward）**：SimWAM 直接用了 NAVSIM 的 PDM 分数作为 RL 奖励，这个奖励本身就是"安全 × 合规 × 质量"的组合：

$$\text{PDMS} = \underbrace{\prod_{m \in \{NC, DAC\}} r_m}_{\text{惩罚因子：无碰撞 × 可行驶}} \times \underbrace{\frac{\sum_{m \in \{EP, TTC, C\}} w_m\, r_m}{\sum_{m \in \{EP, TTC, C\}} w_m}}_{\text{加权质量：进度×时间到碰撞×舒适}}$$

| 子指标 | 全称 | 类别 |
|--------|------|------|
| **NC** | No-at-fault Collision | 安全（惩罚因子） |
| **DAC** | Drivable Area Compliance | 合规（惩罚因子） |
| **EP** | Ego Progress | 效率（质量项） |
| **TTC** | Time-to-Collision | 安全（质量项） |
| **C** | Comfort | 舒适（质量项） |

> ⚠️ **注意一个工程细节**：NAVSIM 是**非反应式（non-reactive）**的闭环仿真器——场景里的其他 agent 不会响应本车的轨迹而改变行为。这意味着 reward 可以在离线/开环状态下直接评估每条候选轨迹，不用真正闭环 rollout，极大降低了 RL 的采样成本。这是 SimWAM 能在自动驾驶上把 Flow-GRPO 落地的重要原因。

### 4.5 训练流程全景（对应图 2 右侧 "Inference & RL"）

RL 阶段（只在困难 navtrain 场景上做，即 imitation PDMS < 90 的场景）：

1. **采样**：用 SDE 采样 G=8 条候选轨迹；
2. **打分**：每条轨迹送到 NAVSIM 的 PDM reward 打分，得 R(τ_i)；
3. **归一化**：组内归一化得优势 A_i = (R_i − R̄) / σ；
4. **策略更新（clipped policy update）**：
   - 用当前模型重算每步 SDE 转移的 log-prob；
   - 算 ratio = exp(logπ_new − logπ_old)；
   - 损失 = min(ratio·A, clip(ratio)·A) + KL 约束；
   - **只更新 Action Expert 的 rank-32 LoRA**。

推理阶段则极简：`当前帧 → VAE → Action DiT（10 步 ODE）→ 轨迹`，全程无未来帧生成。

**RL 训练动态**（论文图 3，也是我们在图 3 里看到的曲线）：在困难子集上训练，PDMS 稳步爬到 **15k steps 时 91.5 的峰值**；而在全部 navtrain 上训练反而更差——因为大量场景模仿已经处理得很好，RL 信号被稀释。两条曲线在 15k 步后都略有回落，说明长时间优化收益递减。

![图 3：RL 训练动态（源图 arXiv:2608.07468 Figure 3）。星号为模仿学习 checkpoint。实线是只在困难子集（imitation PDMS<90）上训练，虚线是在全部 navtrain 场景训练；困难子集在 15k 步达到 91.5 PDMS 峰值且全程优于全量训练。](/images/simwam/fig3_rl_dynamics.png)

#### 🔍 图 3 怎么读？

这是一张 RL 训练曲线图，重点看三条信息：

- **横轴 = RL 训练步数（steps）**，纵轴 = navtest PDMS（评估分数）。曲线描述"用 GRPO 再训练过程中，模型在 NAVSIM navtest 上的分数怎么涨"。
- **星号（★）起点 = 模仿学习 checkpoint**：曲线从"联合训练完、还没做 RL"的那个模型开始。它本身已经有 ~90.3 的 PDMS（就是表 2 里 "+Video" 的成绩），RL 的目标是从这里**再往上**。
- **两条曲线的区别是 RL 数据范围**：
  - **实线 = 只在困难子集上做 RL**（imitation PDMS<90 的 navtrain 场景）。分数稳步爬升，约 **15k 步时达到 91.5 峰值**；
  - **虚线 = 在全部 navtrain 上做 RL**。起步更早、曲线更平缓，但**全程低于困难子集**——因为大部分场景模仿已经学得很好，RL 的 reward 信号被"简单场景的高分"稀释了。
- **15k 步之后两条线都有点回落**：说明长时间对着同一批场景做 GRPO，策略开始过拟合那批场景的 PDM 分数分布，收益递减甚至回吐——这也是为什么论文把训练步数控制在 15k。

> 🎯 图 3 想证明：**RL 不是数据越多越好，而是"难题越集中越好"**。把奖励资源集中在模仿学不会的场景上，信号的信噪比最高。这条经验可以直接移植到任何"模仿预训练 + RL 微调"的流水线。

### 4.6 RL 探索方式消融：为什么 SDE 优于"随机噪声扰动"

一个特别值得注意的消融（Tab. 7）：

| Sampler | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|---------|------|------|------|------|-------|
| **Random noise（随机噪声扰动）** | 97.7 | 98.4 | **88.0** | 94.1 | 91.3 |
| **SDE（边际保持的随机微分方程）** | **98.4** | **98.7** | 86.4 | **95.5** | **91.5** |

- **随机噪声扰动**：简单粗暴给轨迹加随机扰动。能增加多样性、**甚至把 EP（前进进度）刷到 88.0**——因为它把轨迹"戳"得更激进、走得更远——但同时**把 NC（无碰撞）和 TTC（碰撞时间）打崩**，轨迹缺乏结构化可行性，安全侧崩盘。
- **SDE**：在**保持原始 flow 边缘分布**的前提下引入随机探索，候选轨迹既有多样性又保持"长得像可行轨迹"。整体 PDMS 更高（91.5），安全与效率平衡得更好。

> 🔑 **关键洞察**：RL 探索的重点**不是简单地增加随机性，而是构造一个"与生成过程一致、且具有可计算 likelihood 的结构化探索机制"**——这正是 ODE→SDE 转换的意义。随机扰动虽然也能探索，但它破坏了轨迹分布的可行性结构（不保持边缘分布），也不提供可计算的 log-prob（GRPO 需要），两头都不占。

---

## ⚖️ 与博客 Flow-GRPO 的深度对比

SimWAM 的 RL 部分明确继承了 Flow-GRPO（[博客知识篇完整拆解](/posts/knowledge/flow-grpo详解/)），但二者领域不同、落地细节也不同。这张表值得反复对照：

| 维度 | **Flow-GRPO**（图像生成，arXiv:2505.05470） | **SimWAM RL**（自动驾驶，arXiv:2608.07468） |
|------|---------------------------------------------|---------------------------------------------|
| **应用领域** | 文生图（SD3/FLUX/Qwen-Image/Wan2.1） | 端到端自动驾驶轨迹规划 |
| **policy 对象** | 图像 latent 的去噪网络 | Action Expert 的轨迹速度场 v_θa |
| **"状态"** | 当前 latent x_t + prompt + 时间 t | 当前观测 z(o_t) + ego 状态 + 指令 + flow 时间 τ |
| **"动作"** | x_t → x_{t-1} 的 latent 转移 | 轨迹在 flow 时间上的转移 |
| **ODE→SDE** | ✅ 核心改造（sde_type 支持 sde/cps 两种） | ✅ 直接沿用 Flow-GRPO 的 marginal-preserving SDE（Eq.2 明确引用 [30]） |
| **reward** | 图文一致性/偏好/质量（GenEval、OCR、PickScore、CLIPScore、Aesthetic…） | **NAVSIM PDM 组合驾驶奖励**（NC×DAC×加权 EP/TTC/C） |
| **reward 评估器** | 预训练 reward model 打分最终图像 | **非反应式闭环仿真器**给轨迹打分（离线可评） |
| **组大小 G** | num_image_per_prompt=24（图像） | **G=8 候选轨迹/场景** |
| **更新对象** | LoRA policy（KL 约束到 base） | **只更新 Action Expert 的 rank-32 LoRA**（保留蒸馏先验） |
| **数据选择** | 全量 prompt 训练 | **只挑 imitation PDMS<90 的困难场景** |
| **强化变体** | 还有 Flow-GRPO-Fast（窗口训练）、GRPO-Guard（RatioNorm+重加权） | 暂无，用最朴素的标准 GRPO |
| **训练-推理一致性** | 无视频分支概念 | **视频分支训练完删除**，动作独立推理 |

### 三个"不是照抄"的关键差异

1. **奖励的"免费午餐"不同**：图像版 Flow-GRPO 需要一个独立的 reward model（图片本身没有真值标签）；SimWAM 直接复用 NAVSIM 的 **PDM 分数**——它本身就是为评价驾驶行为设计的**可微目标的对齐物**，而且是**非反应式闭环可离线评估**的。这让"在线 RL"在自动驾驶里第一次变得可行（不需要真车试错，不需要昂贵的强化世界模型 rollout）。

2. **探索的尺度不同**：图像生成里 Flow-GRPO 探索的是"像素怎么排布"这种高维语义空间；SimWAM 探索的是**低维驾驶轨迹**（每步 8 waypoint × (x,y,θ)），而且候选轨迹天然被"保持边缘分布的 SDE"约束在可行驶、合理的形状内——探索空间小但更有意义，这也解释了为什么 G=8 就够。

3. **"先训练后删"的结构红利**：Flow-GRPO 把 RL 作用在整个生成模型上；SimWAM 因为**隔离注意力掩码**把动作分支独立出来了，RL 阶段可以**完全不理视频分支**，只优化那个最终要部署的轻量 Action Expert——优化目标和部署目标严格一致，没有"训练一个大家伙、部署一个蒸馏小模型"的落差。

---

## 🗺️ 与博客系列四篇的架构对比 + 做法对比

要真正看懂 SimWAM 在"世界模型 + 生成式规划 + 强化学习"这条知识链上的位置，最有效的方式是把博客里同一条脉络上的四篇文章和它放到一起横向比。这四个参照系分别补上了 SimWAM 的不同"前件"：

| 参照 | 在知识链中的角色 | 补上了什么 |
|------|----------------|-----------|
| [DriveVLA-W0 精读](/posts/paper-reading/drivevla-w0精读/) | 世界模型**放大数据** | 证明"未来图像预测当稠密监督"能喂饱大模型 |
| [GoalFlow 精读](/posts/paper-reading/goalflow精读/) | **Flow Matching** 高效生成 | 证明"直线路径 + 少步采样"让生成式规划上车可行 |
| [Flow-GRPO 详解](/posts/knowledge/flow-grpo详解/) | **RL 底座** | 给出"Flow Matching × GRPO"的完整算法与源码 |
| [ReCogDrive 精读](/posts/paper-reading/recogdrive精读/) | **DiffGRPO** 强化认知 | 首次把 GRPO 用到驾驶扩散规划器上 |
| **SimWAM（本文）** | 以上三者的**合流** | 世界模型监督 + Flow 生成 + **Flow-GRPO** 强化 |

### 架构对比：四条路线的"世界模型"长什么样

| 维度 | **DriveVLA-W0** | **GoalFlow** | **ReCogDrive** | **SimWAM** |
|------|----------------|--------------|----------------|------------|
| 骨干 | VLM（Emu3-8B / Qwen2.5-VL-7B）+ MoE Action Expert | Transfuser 感知 → BEV + 目标点词表 | Qwen2.5-VL + 扩散规划器 | Video Expert（Wan2.2-5B）+ Action Expert（1.02B） |
| 动作生成器 | AR / Flow Matching / Query 三种解码器对比 | **Flow Matching 单步生成** | **扩散规划器**（DDPM） | **Flow Matching DiT**（10 步） |
| 世界模型形态 | 显式预测**未来图像**（AR 或 Diffusion） | 无显式世界模型 | 无显式世界模型 | 显式预测**未来视频**（联合训练） |
| 世界模型监督 | 训练期稠密监督，推理期旁路 | — | — | **训练期注入，推理期删除视频分支** |
| 视频-动作耦合 | 同一 VLA 骨干联合预测 | — | VLM 认知表征 → 扩散规划器 | **隔离注意力掩码**（训练期解耦） |
| RL 环节 | ❌ 无 | ❌ 无 | ✅ **DiffGRPO**（扩散 × GRPO） | ✅ **Flow-GRPO**（流匹配 × GRPO） |
| RL 作用对象 | — | — | 扩散规划器去噪网络 | **Action Expert 的 rank-32 LoRA** |

### 做法对比：四篇的关键"动作"

| 做法 | **DriveVLA-W0** | **GoalFlow** | **ReCogDrive** | **SimWAM** |
|------|----------------|--------------|----------------|------------|
| 数据策略 | 7000 万帧内部数据 + 世界模型放大缩放律 | NAVSIM + 目标点评分 | VQA 认知数据流水线 + 专家轨迹 | NAVSIM（navtrain 困难子集做 RL） |
| 两阶段训练 | 先世界预训练（6VA）→ 动作专精（2VA） | 感知/选点/生成/评分联合 | 认知预训练 → 规划器联合 → DiffGRPO | 先视频-动作联合模仿 → 再 Flow-GRPO |
| 推理时延控制 | Action Expert 瘦身 74ms、旁路世界模型 | **单步生成** | 隐状态注入 7.8× 加速 | 推理期删视频分支，纯动作 DiT |
| 多模态轨迹 | 三种解码器（大数据下 AR 最优） | 目标点分隔模态 | 扩散天然多模态 | Flow DiT 多模态 + 候选轨迹 |
| 闭环策略 | 开环 NAVSIM 为主 | 开环 NAVSIM | 开环 + Bench2Drive 闭环 | 开环 NAVSIM + nuScenes 零样本 |

### 关键论断：为什么说 SimWAM 是首个把 Flow-GRPO 引入自动驾驶的工作

这是这篇精读最想强调的一点，需要拆成两层看：

**第一层：GRPO 进入驾驶，ReCogDrive 是先行者，但它用的是"扩散版"。** [ReCogDrive](/posts/paper-reading/recogdrive精读/)（arXiv:2506.08052，2025 年 6 月）首次把 GRPO 用到了自动驾驶的**扩散规划器**上——即 **DiffGRPO**（Diffusion × GRPO）。它验证了"组内相对优势 + 扩散多模态生成"在驾驶里的可行性。但它的生成底座是 **DDPM 式扩散**：弯曲的去噪路径、需要较多采样步数、log-prob 通过 ELBO 近似计算。

**第二层：Flow-GRPO 进入驾驶，SimWAM 是第一个。** Flow-GRPO 本身（arXiv:2505.05470，字节跳动，NeurIPS 2025）最初是给**文生图**设计的（SD3/FLUX/Qwen-Image/Wan2.1）。在自动驾驶领域，把"**Flow Matching（rectified flow）** × **GRPO**"这套组合完整落地——包含关键的 **ODE→SDE 转换**（让确定性 ODE 采样变成有可解析 log-prob 的随机探索）——SimWAM（arXiv:2608.07468）是目前首个明确继承该范式的工作，论文 Eq.2 直接标注 "Following Flow-GRPO [30]"。

为什么"扩散版 GRPO"不能算"Flow-GRPO"？三点本质差异：

| 维度 | **DiffGRPO**（ReCogDrive） | **Flow-GRPO**（SimWAM 所用） |
|------|---------------------------|------------------------------|
| 生成路径 | 弯曲的 DDPM 反向去噪链 | **直线 rectified flow**（最优传输） |
| 采样步数 | 较多（20+ 步） | **少（10 步即可，可降到 2 步）** |
| log-prob 来源 | ELBO / 变分下界近似 | **SDE 转移的高斯密度解析式** |
| 探索机制 | 扩散噪声本身 | **ODE→SDE 保持边缘分布的随机化** |
| 轨迹形状约束 | 无显式约束，易发散 | 直线路径天然稳定、少步收敛 |

> 换句话说：**ReCogDrive 证明了"GRPO 这种 RL 能进驾驶"；SimWAM 证明了"Flow-GRPO 这种'直线路径 + 少步 + 解析密度'的特定配方能进驾驶"，并拿到了比扩散版更高的上限（91.5 vs ReCogDrive 的 90.8 PDMS）。** 这两者不是同一件事——前者是扩散底座，后者是流匹配底座。

还有一个容易被忽略的先后关系佐证：SimWAM 依赖的三大件——世界模型训练期监督（DriveVLA-W0 已验证）、Flow Matching 高效生成（GoalFlow 已验证）、GRPO 强化（ReCogDrive/Flow-GRPO 已验证）——**在 SimWAM 之前没有一篇同时具备**。ReCogDrive 有 GRPO 但没有显式世界模型；DriveVLA-W0 有世界模型但没有 RL；GoalFlow 有 Flow Matching 但没有世界模型和 RL。**SimWAM 是第一个把这三块拼成一个闭环的工作，而其中"Flow-GRPO 上驾驶"这一步，在公开文献里尚无更早的先例。**

---

## 📊 实验结果：NAVSIM 新 SOTA

### 5.1 主结果（NAVSIM navtest，单前视相机）

| 类别 | 方法 | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|------|------|------|------|------|------|-------|
| **Human Agent** | — | 100.0 | 100.0 | 87.5 | 100.0 | 94.8 |
| 传统 E2E | UniAD (6C) | 97.8 | 91.9 | 78.8 | 92.9 | 83.4 |
| 传统 E2E | DiffusionDrive (3C+L) | 98.2 | 96.2 | 82.2 | 94.7 | 88.1 |
| 传统 E2E | SeerDrive (3C+L) | 98.4 | 97.0 | 83.2 | 94.9 | 88.9 |
| VLM 系 | ReCogDrive (1C) | 97.9 | 97.3 | 87.3 | 94.9 | 90.8 |
| VLM 系 | DriveVLA-W0 (1C) | 98.7 | 99.1 | 83.3 | 95.3 | 90.2 |
| VLM 系 | SGDrive (1C) | 98.6 | 97.8 | 85.8 | 96.2 | **91.1** |
| WAM 系 | Epona (1C) | 97.9 | 95.1 | 80.4 | 93.8 | 86.2 |
| WAM 系 | DriveLaW (1C) | 99.0 | 97.1 | 81.3 | 96.7 | 89.1 |
| WAM 系 | DriveWAM (1C) | 98.3 | 98.1 | 84.3 | 95.2 | 90.1 |
| **WAM 系** | **SimWAM (1C)** | **98.4** | **98.7** | **86.4** | 95.5 | **91.5** |

**结论拆解：**

- **91.5 PDMS，端到端规划新 SOTA**，超过最强 VLM planner SGDrive **0.4 分**；
- 超过显式做未来图像预测的 ExploreVLA **1.1 分**（ExploreVLA 90.4）——**"训练期想象"打赢了"测试期想象"**，这是 SimWAM 最想证明的点；
- 相同单相机设置下，超过 imagine-then-act 的 **DriveLaW 2.4 分**、**DriveWAM 1.4 分**——在**不生成未来帧的情况下反而规划得更好**；
- 在 world-model 系里拿到**最高 DAC（98.7）和最高 EP（86.4）**，NC 与 TTC 保持竞争力。

> 结合图 1 的时延-性能散点，SimWAM 的卖点非常清晰：**横轴（时延）靠近纯 planner，纵轴（PDMS）登顶**——它证明"训练期世界建模"可以同时带来规划质量提升和推理效率，不用像 imagine-then-act 那样在二者间做取舍。

### 5.2 组件消融：video co-training 和 RL 的互补性

| 配置 | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|------|------|------|------|------|-------|
| Action-only（纯动作 DiT，无视频） | 97.6 | 95.7 | 81.7 | 92.6 | 86.6 |
| **+ Video**（联合训练） | 98.7 | 98.0 | 83.9 | 95.9 | **90.3** |
| **+ RL**（GRPO 优化） | 98.4 | 98.7 | 86.4 | 95.5 | **91.5** |

- **+Video 提升 +3.7**：NC、DAC、EP、TTC 全线提升，证明未来视频监督确实把交通动态先验"装进"了共享观测表示，收益不局限在单一指标；
- **+RL 提升 +1.2**：DAC 到 98.7、EP 到 86.4（质量项明显改善），NC/TTC 有小幅回落——RL 在安全、合规、进度之间重新做了权衡；
- **两阶段累计 +4.9**，且始终不需要未来帧推理。

### 5.3 Video Expert 可替换性

| Video model | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|-------------|------|------|------|------|-------|
| LTX-Video（轻量） | 98.1 | 97.2 | 83.1 | 94.3 | 88.7 |
| Wan2.1-1.3B | 98.6 | 98.1 | 84.0 | 95.9 | 90.2 |
| Wan2.2-5B | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |
| **Cosmos-Predict2.5**（驾驶视频预训练） | 98.7 | 98.0 | **84.2** | **96.0** | **90.4** |

**结论**：SimWAM 不绑定特定视频模型——**换 Video Expert 不需要改动作分支、训练目标或推理管线**。视频先验质量直接决定规划性能：在驾驶视频上预训练的 Cosmos-Predict2.5 给出最强的 EP/TTC；这也预示着一个滚雪球式的升级路径——视频生成模型每进步一代，SimWAM 的规划免费受益一次。

### 5.4 Action Expert 独立缩放

| Action DiT | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|------------|------|------|------|------|-------|
| 0.21B | 98.6 | 97.8 | 84.0 | 95.4 | 89.9 |
| 0.45B | 98.6 | 97.9 | 83.8 | 95.9 | 90.1 |
| 1.02B | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

动作分支从 0.21B 涨到 1.02B，PDMS 从 89.9 稳步升到 90.3——**两个专家互不依赖参数，容量各自独立调节**。大视频专家增强训练监督但不增加部署成本，动作专家按部署预算单独选尺寸。

### 5.5 Zero-shot 跨数据集泛化（nuScenes 开环）

NAVSIM 训练的 SimWAM 不做任何微调直接测 nuScenes 开环规划：

| 方法 | 微调 | 输入 | L2 avg (m)↓ | Collision avg (%)↓ |
|------|------|------|-------------|---------------------|
| UniAD | ✅ | Camera | 1.03 | 0.31 |
| OccWorld | ✅ | Camera | 1.40 | 0.87 |
| GenAD | ✅ | Camera | 0.91 | 0.43 |
| Epona | ✅ | Camera* | 1.25 | 0.36 |
| DriveVA | ❌ | Camera* | 0.84 | 0.06 |
| DriveWAM | ❌ | Camera* | 0.96 | 0.06 |
| **SimWAM** | ❌ | Camera* | **0.96** | **0.04** |

- 在零样本方法里拿到**最低碰撞率 0.04%**，平均 L2 0.96 m 与 DriveWAM 持平；
- **L2 强调与目标数据集专家轨迹的几何一致性，碰撞率更直接反映对交通交互和安全边界的建模能力**——在数据分布明显漂移（NAVSIM→nuScenes）下碰撞率仍最低，说明 SimWAM 学到的动态先验有一定跨数据集可迁移性，而不只是拟合了 NAVSIM 轨迹分布。

### 5.6 定性可视化（图 4）

![图 4：Ours-IL（模仿）与 Ours-RL（强化学习后）定性对比（源图 arXiv:2608.07468 Figure 4）。红圈标出 RL 后轨迹在保持可行驶区域内更充分地沿目标方向推进（路口与窄街场景）。](/images/simwam/fig4_qualitative.png)

#### 🔍 图 4 怎么读？

这是"RL 到底改变了什么"的直观证据，一行两个场景、每个场景三栏：

- **每行一个 navtest 场景**（上排是交叉路口，下排是狭窄街道），每栏画的是相机视角上叠加的预测轨迹（Concat View）。
- **左栏 = 原始视角**：只展示场景原貌（车、路、红绿灯），供你对照后面两栏轨迹在空间上落在哪。
- **中栏 = Ours-IL（模仿学习版）**：轨迹往往**保守**——到路口就减速、靠边停住，推进距离短，像"照着专家录像学了个胆小的司机"。
- **右栏 = Ours-RL（GRPO 强化后）**：**红圈标出的地方**，轨迹在保持可行驶区域内更充分地**沿目标方向推进**——路口敢转了、窄街敢走了，机动动作更完整。

> 🎯 图 4 想说明：RL 后模型学会了**在"安全底线之内"更有效率地驾驶**。模仿版"会开但胆小"（停在安全里却走不远），RL 版"会开也会权衡"（在不碰撞、不出可行驶区的前提下把路走完）——这正是直接优化组合驾驶奖励（PDMS）带来的行为差异。

### 5.7 其余工程消融速览

| 消融 | 结论 |
|------|------|
| **推理分辨率**（Tab.9） | 192×352→384×672：PDMS +1.4，时延仅 +9ms；再涨到 768×1344 只 +0.3，算力暴涨。**384×672 最优平衡** |
| **采样步数**（Tab.10） | 1 步严重不足（68.9）；5 步接近最优（90.1）；10 步封顶（90.3）；20 步无增益且时延翻倍。**10 步收敛** |
| **未来帧配置**（Tab.8） | 4 帧/2s/2Hz：89.9；4帧/4s/1Hz：90.2；**8 帧/4s/2Hz：90.3**——时间覆盖广度比帧密度更重要 |

---

## 🔧 实现细节汇总

| 设置 | 值 |
|------|-----|
| 视频专家 | Wan2.2-5B + video VAE + T5 编码器 |
| 动作专家 | Diffusion Transformer，hidden=1024（1.02B） |
| 输入 | 单前视相机 384×672 |
| 轨迹表示 | 8 waypoints × 4s @ 2Hz（x,y,θ）；视频预测对应 8 帧 |
| 联合训练 | AdamW，cosine LR，初始 1e-4，100 epochs，batch 64，λ=1 |
| RL | rank-32 LoRA（α=16），仅更新 attention projections |
| RL 采样 | G=8 候选轨迹/场景，SDE 采样 |
| RL 数据 | navtrain 中 imitation PDMS<90 的困难场景 |
| RL 优化 | 学习率 5e-5，15k 步达峰 |
| 推理 | 10 步 denoising，CFG=1.0 |

---

## ⚠️ 优势与局限

### ✅ 优势

- **训练/推理彻底解耦**：视频先验训练期注入，推理期删除视频分支，planner 自包含、低时延；
- **可替换、可缩放**：Video Expert 换新的、Action Expert 单独调参，都不动对方和训练目标；
- **RL 直接优化驾驶质量**：Flow ODE→SDE 保留边缘分布的探索 + GRPO 组内相对优势，突破模仿上限，+4.9 PDMS；
- **明确继承 Flow-GRPO 范式**：把已经验证的图像生成 RL 方法无缝迁移到驾驶轨迹生成，工程上可复现性强；
- **零样本跨域迁移**：nuScenes 上最低碰撞率，动态先验不局限于训练数据集。

### ❌ 局限

- **非反应式仿真依赖**：NAVSIM 的 PDM 奖励基于非反应式环境——其他 agent 不响应本车，reward 与真实闭环驾驶仍有 gap，需要 Bench2Drive 这类反应式闭环进一步验证；
- **EP 与 TTC 的此消彼长**：RL 后 EP 大涨但 TTC 微降，组合奖励的权重选择仍有主观性，reward hacking 风险需要 KL 约束兜底；
- **开环指标≠闭环安全**：PDMS 强调"模仿得像"的程度，零样本碰撞率虽低但真车闭环尚未完全验证（论文提到 real-robot 部署在补充材料）；
- **视频先验依赖预训练模型**：视频专家替换表显示先验质量直接决定性能，说明它本质上仍在"吃视频生成模型的进步红利"。

---

## 📝 个人思考

**这是 Flow-GRPO 在自动驾驶上的"标准答案式"落地**。SimWAM 的 RL 部分和我们博客里 Flow-GRPO 文章讲的方法一脉相承：确定性 ODE 无法探索、没有可解析密度 → 转成边际保持的 SDE → 组内相对优势 → clipped policy update + KL 约束。它证明了我之前一直强调的判断——**Flow Matching 是 GRPO 在生成式规划里的天然基底**：直线路径 + 少步采样 + 可解析转移密度，三样都是 RL 友好特性。GoalFlow 证明了 Flow Matching 适合做单步高效生成，SimWAM 则把这条线推进到了"Flow Matching + RL"。

**三个让我眼前一亮的工程智慧**：

1. **非反应式仿真 = 自动驾驶的"免费 RL 环境"**。NAVSIM 不开真车、不需要 agent 响应，就能离线给每条候选轨迹打分——这让 Flow-GRPO 的"在线 RL"在驾驶里落地成了"离线采样 + 离线打分 + 策略更新"的低成本版本。这是很多人没意识到的关键前提。

2. **困难场景筛选是数据效率的胜负手**（图 3）。只在 imitation PDMS<90 的场景上做 RL，比全量训练更好——这个洞察可以迁移到任何 RL 微调场景：**不要在全量数据上平均用力，要集中火力在"模仿学不会"的硬骨头上**。

3. **RL 阶段只训 LoRA、连视频分支都不带**。因为隔离掩码把动作分支独立出来了，RL 的优化目标和部署目标严格一致。这比"RL 一个完整大模型、再蒸馏"的路线省事一个数量级，也避开了"训练部署不一致"的隐性坑。

**下一步最值得关注**：①在 Bench2Drive 这类**反应式闭环**基准上验证（这是所有 NAVSIM SOTA 的共同考卷）；②把 GRPO-Guard（RatioNorm + 梯度重加权）这类防过优化机制搬进来，防止 RL 训久了对 PDM 奖励钻空子；③Video Expert 换成更强的驾驶域视频模型（Cosmos-Predict2.5 已经显示出趋势），动作先验还能再涨。

**关联阅读**：这篇和 [DriveVLA-W0 精读](/posts/paper-reading/drivevla-w0精读/)（世界模型放大数据）、[ReCogDrive 精读](/posts/paper-reading/recogdrive精读/)（DiffGRPO 强化认知）、[Flow-GRPO 详解](/posts/knowledge/flow-grpo详解/)（RL 底座）、[GoalFlow 精读](/posts/paper-reading/goalflow精读/)（Flow Matching 高效生成）串起来读，正好构成"世界模型 → Flow 生成 → GRPO 强化"的完整知识链——而 SimWAM 正是这条链上**第一根全部串起来的闭环**：世界模型只在训练期当老师（DriveVLA-W0 的教训）、Flow Matching 直线路径做高效生成（GoalFlow 的教训）、把 Flow-GRPO 的 ODE→SDE 配方首次搬上驾驶（见上文横向对比）——这就是它配得上"首个把 Flow-GRPO 引入自动驾驶"论断的三块基石。再与 [Metis 精读](/posts/paper-reading/metis精读/)（MoT 双专家 + 非对称掩码）对照，就能看清同期"解耦视频-动作"的两种取向。

---

*📖 论文精读系列。SimWAM 是"世界模型只在训练期当老师"这一范式在自动驾驶的标杆性工作，强烈建议与 Metis（arXiv:2606.15869，MoT 双专家+非对称掩码）对照阅读——二者同期探索了"解耦视频生成与动作预测"的 WAM 设计，SimWAM 多了 GRPO 强化学习这一步。*
