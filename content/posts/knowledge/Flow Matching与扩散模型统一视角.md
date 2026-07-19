---
title: "知识点拆解｜Flow Matching 与扩散模型统一视角：从 SDE 到 ODE 的生成范式"
date: 2026-07-19
draft: false
categories: ["知识点拆解"]
tags: ["🌊 Flow Matching", "🎨 扩散模型", "📐 数学基础", "🔗 统一视角", "🚗 自动驾驶"]
summary: "扩散模型与 Flow Matching 是当前生成式 AI 的两大核心范式。本文从 SDE 与 ODE 的统一视角出发，深入讲解扩散模型的随机微分方程解释、Flow Matching 的向量场学习方法、Rectified Flow 的路径整流机制，揭示扩散是 Flow Matching 的特殊情况。同时总结两者在自动驾驶场景生成与轨迹规划中的应用对比。"
weight: 20
---
## 引言：两个框架，一个本质
扩散模型（Diffusion Models）和 Flow Matching（流匹配）看起来是两条不同的技术路线——前者通过加噪去噪生成数据，后者通过学习速度场"搬运"数据。但它们在数学上分享同一个深层结构：**学习一个将噪声分布变换到数据分布的确定性或随机性过程**。
本文的目标是提供一个**统一的视角**，展示 SDE、ODE、扩散模型、Flow Matching、Rectified Flow 之间的内在联系。理解了这个统一框架，就能理解为什么扩散可以做、Flow Matching 也可以做，以及两者各自适合什么场景。
---
## 🌀 扩散模型的 SDE 视角
### 从离散马尔可夫链到连续 SDE
标准 DDPM 的离散马尔可夫加噪链可以推广到**连续时间的随机微分方程**（SDE）：
$$dx = f(x, t)dt + g(t)dw$$
其中：
- $f(x, t)$ 是**漂移系数**（drift），控制信号的确定性衰减
- $g(t)$ 是**扩散系数**（diffusion），控制噪声的注入强度
- $dw$ 是维纳过程（Wiener process，即布朗运动）
**VP-SDE（Variance Preserving）** 对应 DDPM 的连续版本：
$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dw$$
其中 $\beta(t)$ 是噪声调度的连续版本。VP-SDE 保证了过程在任意时刻的方差保持为 1。
### 反向 SDE：去噪的连续形式
安德森在 1982 年证明：任何正向 SDE 都存在对应的**反向 SDE**，从 $T$ 时刻逆向回到 $0$ 时刻：
$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$
其中 $\nabla_x \log p_t(x)$ 是 **score 函数**——扩散模型实际上在估计的就是这个 score 函数。
### 概率流 ODE（Probability Flow ODE）
SDE 的另一个重要性质：**每个 SDE 都对应一个确定性 ODE**，其边际分布 $p_t(x)$ 完全相同：
$$dx = [f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)]dt$$
这个 **Probability Flow ODE** 是连接扩散模型和 Flow Matching 的关键桥梁——它表示我们完全可以用一个确定性过程（ODE）来实现与随机过程（SDE）相同的分布变换。
### SDE vs ODE 采样对比
| 维度 | SDE 采样 | ODE 采样（概率流） |
|------|----------|-------------------|
| 噪声 | ✅ 有随机噪声项 | ❌ 确定性 |
| 采样质量 | 高（随机性弥补模型误差） | 略低（无随机修正） |
| 采样步数 | 通常需要多步 | 可用大步长求解 |
| 可逆性 | ❌ 不可逆（带噪声） | ✅ 可逆（单向确定性） |
| 似然计算 | ❌ 不支持 | ✅ 可计算数据似然（瞬时换元法） |
---
## 🌊 Flow Matching：从 SDE 到 ODE 的跳跃
### 核心直觉
扩散模型走得有些绕——先加噪再减噪，路径弯曲导致需要很多步。Flow Matching 问了一个更直接的问题：**能不能直接学一条从噪声分布到数据分布的直线路径？**
### 概率路径（Probability Path）
定义**时间相关概率路径** $p_t(x)$，其中 $t \in [0, 1]$：
- $t=0$：$p_0(x)$ 是简单分布（标准高斯噪声）
- $t=1$：$p_1(x)$ 是目标数据分布
Flow Matching 学习一个**时间相关向量场** $v_\theta(x, t)$，使得：
- 沿着 $v_\theta$ 的积分曲线（流），$p_0$ 会被"推"到 $p_1$
- $v_\theta$ 的方向指向从 $p_0$ 到 $p_1$ 的最有效路径
数学上，ODE 形式为：
$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim p_0$$
通过求解这个 ODE 从 $t=0$ 到 $t=1$，我们得到 $x(1) \sim p_1$。
### 条件流匹配（Conditional Flow Matching）
直接学习 $v_\theta$ 来匹配 $p_t$ 是困难的（因为我们不知道 $p_t$ 的解析形式）。**条件流匹配**（CFM）的技巧是引入**单个数据点** $x_1$ 的条件：
对于每个数据点 $x_1$，定义条件概率路径 $p_t(x|x_1)$，使得：
- $p_0(x|x_1)$ 是以 $x_1$ 为中心的高斯分布
- $p_1(x|x_1)$ 是 $\delta(x - x_1)$（狄拉克分布，即确定在 $x_1$）
最常用的条件路径是**最优传输（OT）路径**：
$$p_t(x|x_1) = \mathcal{N}(x | tx_1, (1 - t)^2 I)$$
对应的条件向量场为：
$$u_t(x|x_1) = \frac{x_1 - x}{1 - t}$$
CFM 的训练目标就是让网络匹配这个条件向量场：
$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_1, x_t \sim p_t(x_t|x_1)} \left[ \| v_\theta(x_t, t) - u_t(x_t|x_1) \|^2 \right]$$
这个损失函数与扩散模型的 MSE loss 在形式上非常相似——**两者都在训练一个网络去匹配某个目标向量/噪声**。
### 训练与采样对比
| 步骤 | 扩散模型 | Flow Matching |
|------|----------|---------------|
| 前向采样 | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | $x_t = t x_1 + (1-t) \epsilon$ |
| 目标 | 预测噪声 $\epsilon$ | 预测速度 $v(x_t, t)$ |
| 损失 | $\|\epsilon_\theta(x_t, t) - \epsilon\|^2$ | $\|v_\theta(x_t, t) - (x_1 - \epsilon)\|^2$ |
| 采样 ODE | 概率流 ODE（较弯曲） | 直线 ODE |
| 步数需求 | 50~1000 步 | 4~20 步 |
---
## ➡️ Rectified Flow：把弯曲的路径"拉直"
### 问题：扩散路径为什么弯曲
扩散模型的概率流 ODE 路径弯曲的原因：**噪声到数据的映射不是线性的**。在加噪过程中，不同数据点的噪声版本混杂在一起，导致去噪路径需要绕路才能将它们分开。
### 整流思想
Rectified Flow 的核心思想非常优雅：**你走的路径太弯了，那就走一次，记住走过的路线，再从起点沿着这条路线走直**。
**Reflow 算法**（一步整流）：
1. **采样配对**：从 $p_0$ 采样 $x_0$，从 $p_1$ 采样 $x_1$，用当前 ODE 生成 $(x_0, x_1)$ 的路径轨迹
2. **重新学习**：训练新的向量场 $v_\theta^{new}$ 直接拟合从 $x_0$ 到 $x_1$ 的**直线映射**：
   $$\mathcal{L}_{reflow} = \mathbb{E}_{(x_0, x_1), t} \left[ \| v_\theta^{new}(x_t, t) - (x_1 - x_0) \|^2 \right]$$
这个"配对-重学"的过程可以递归执行（2-reflow、3-reflow...），每一步都将路径拉得更直。
### 整流效果的直观理解
```
ReFlow 的核心思想是：对初始 ODE 路径执行"先正向采样、再反向配对"的操作，使路径逐步拉直。
- 初始 ODE：噪声路径与数据路径弯曲交叉
- 1-reflow 后：路径大致对齐
- 2-reflow 后：路径近似直线
```
### Rectified Flow 的优势
| 属性 | 原始 ODE | 1-reflow | 2-reflow |
|------|----------|----------|----------|
| 路径弯曲度 | 高 | 中 | 低 |
| ODE 求解步数 | 50+ | 10-20 | 4-10 |
| 质量损失 | — | 轻微 | 可接受 |
| 训练成本 | 一次 | 再次 | 额外 |
Rectified Flow 的核心价值：**用一次额外的训练，换取推理时 5~10 倍的加速**。
---
## 🔗 统一视角：扩散是 Flow Matching 的特殊情况
### 扩散模型 = 特定路径的 Flow Matching
扩散模型的概率流 ODE 本质上是 Flow Matching 的一种——选择了**VP/VE 特定的路径**。如果我们将扩散的加噪过程视为定义了一个特定的概率路径 $p_t(x)$，那么：
- 扩散模型估计的是 score $\nabla_x \log p_t(x)$
- Flow Matching 估计的是向量场 $v_\theta(x, t)$
- 两者通过**概率流 ODE** 等价：$v_\theta(x, t) = f(x, t) - \frac{1}{2}g(t)^2 s_\theta(x, t)$
更直接地说：**扩散模型是 Flow Matching 在特定噪声调度下的实例化**。
### 统一公式表
| 概念 | 扩散模型术语 | Flow Matching 术语 |
|------|--------------|-------------------|
| 前向过程 | 加噪（Noising） | 概率路径（Probability Path） |
| 训练目标 | 噪声预测 $\epsilon_\theta$ | 速度预测 $v_\theta$ |
| 采样过程 | 去噪 | ODE 求解 |
| 时间范围 | $t \in [0, T]$ | $t \in [0, 1]$ |
| 初始分布 | 近似高斯（$x_T \approx \mathcal{N}(0, I)$） | **精确高斯**（$x_0 \sim \mathcal{N}(0, I)$） |
| 路径设计 | 固定的 $\beta_t$ 调度 | **可任意设计**（OT, VP, VE, etc.） |
### 关键区别对比
**区别 1：路径自由度**
- 扩散：路径由 $\beta_t$ 固定，不能改变
- FM：路径可任意设计（直线、曲线、混合），自由度更高
**区别 2：初始分布**
- 扩散：$x_T$ 只是接近高斯（$\bar{\alpha}_T \approx 0$ 但不等于 0）
- FM：$x_0$ **精确**为高斯分布（标准正态），没有近似误差
**区别 3：采样效率**
- 扩散：弯曲路径需要小步长 ODE 求解
- FM：直线路径允许大步长，显著减少步数
**这个统一视角的启示**：扩散模型在过去几年积累的所有技术（CFG、LDM、DiT、AdaLN 等）几乎都可以直接迁移到 Flow Matching 上。反过来，Flow Matching 的路径设计自由度也启发了对扩散噪声调度的改进。
---
## 🏗️ 统一框架下的技术迁移
### CFG 的迁移
CFG 在 Flow Matching 中的形式与扩散完全一致：
$$\tilde{v}_\theta(x_t, t, y) = v_\theta(x_t, t, \varnothing) + w \cdot (v_\theta(x_t, t, y) - v_\theta(x_t, t, \varnothing))$$
### DiT 的迁移
DiT 架构无需修改即可用于 Flow Matching——只需将输出从"预测噪声 $\epsilon$"改为"预测速度场 $v$"，网络结构、AdaLN-Zero、patchify 全部保持不变。
### LDM 的迁移
潜在扩散（LDM）也可直接迁移为**潜在流匹配（Latent Flow Matching）**——在 VAE 潜空间中做 Flow Matching。这已成为许多最新方法的选择，因为 FM 在低维潜空间中的收敛更稳定。
### 自适应训练策略
| 技术 | 扩散 | Flow Matching | 迁移难度 |
|------|------|---------------|----------|
| CFG | ✅ | ✅ | 直接使用 |
| DiT | ✅ | ✅ | 只需改输出头 |
| LDM | ✅ | ✅ | 直接使用 |
| DDIM 采样 | ✅ | 类似方法（DPM-Solver） | 需适配 |
| Consistency Model | ✅ | 同样适用 | 需适配 |
| 蒸馏 | ✅ | 同样适用 | 需适配 |
---
## ⚖️ 应用选择：扩散还是 Flow Matching？
### 扩散模型的优势场景
| 场景 | 原因 |
|------|------|
| **文本到图像生成** | Stable Diffusion 生态完善，社区基础设施丰富 |
| **高质量图像/视频生成** | 扩散模型在高质量生成上有最多经验积累 |
| **需要精细光照/纹理** | 扩散的加噪过程天然擅长纹理合成 |
### Flow Matching 的优势场景
| 场景 | 原因 |
|------|------|
| **实时轨迹规划** | 少步采样（2~10 步）满足车端实时需求 |
| **低维生成任务** | 路径直线化在小维度空间效果更明显 |
| **需要精确似然计算** | Flow 的 ODE 可逆性支持 likelihood 评估 |
| **多步/链式生成** | 确定性 ODE 在链式预测中更稳定 |
### 自动驾驶中的具体选择
| 任务 | 推荐方法 | 原因 |
|------|----------|------|
| 场景视频生成 | 扩散模型（LDM + DiT） | 需要高质量的视觉细节 |
| 轨迹规划 | **Flow Matching** | 少步数、低延迟、多模态 |
| 世界模型预测 | **Flow Matching** | 链式预测中 ODE 的确定性优势 |
| 数据增强 | 扩散模型 | 多样性更重要 |
| 闭环仿真 | 扩散（或两者结合） | 视觉质量 + 预测速度需平衡 |
---
## 🚗 自动驾驶中的代表工作
### Flow-GRPO
**核心思想**：将 Flow Matching 与 GRPO（Group Relative Policy Optimization）结合，实现**轨迹生成的策略优化**。
**技术架构**：
1. 用 Flow Matching 学习从噪声到轨迹的映射
2. 引入 GRPO 强化学习，根据 reward（安全性、舒适性）优化轨迹分布
3. 在 Flow 的路径上做策略梯度更新
**关键公式**——Flow ODE + GRPO 更新：
$$v_\theta^{new} \leftarrow v_\theta^{old} + \eta \cdot \mathbb{E}\left[ \frac{\partial \log p_\theta(\tau)}{\partial \theta} \cdot R(\tau) \right]$$
$\tau$ 是轨迹，$R(\tau)$ 是 reward。Flow 的连续性质使得梯度计算比离散扩散更容易。
**优势**：
- Flow Matching 的少步采样使得 RL rollout 非常快
- 连续路径允许更平滑的策略优化
- 在 NAVSIM 等驾驶规划基准上优于扩散规划器
### GoalFlow
**核心思想**：以目标为条件的 Flow Matching 规划器。
**设计思路**：
1. 不再从噪声中随机生成轨迹，而是以**目标状态**（如"10秒后到达路口的哪个位置"）为条件
2. 学习从当前状态 + 目标状态到中间轨迹的速度场
3. 使用 Rectified Flow 技术进一步减少采样步数
**应用场景**：
- 高速公路变道规划（给定目标车道位置）
- 路口转向规划（给定转向后的目标位置）
- 泊车规划（给定泊车终点）
### ReWorld
**核心思想**：在世界模型的潜在空间中使用 Flow Matching 预测未来状态。
**关键技术**：
1. 将观测编码到潜在空间 $z_t = E(x_t)$
2. 在潜在空间中学习 Flow Matching 速度场 $v_\theta(z, t)$
3. 从当前 $z_t$ 开始，求解 ODE 获得未来潜在状态 $z_{t+1}, ..., z_{t+K}$
4. 将潜在状态解码回像素或送入规划器
**优势**：
- Flow Matching 的确定性 ODE 在潜在空间链式预测中更稳定（无累积随机噪声）
- 少步采样使实时推理成为可能
### DriveDreamer v2
**核心变化**：从扩散模型升级到 Flow Matching。
**v1 vs v2 对比**：
| 维度 | DriveDreamer v1 | DriveDreamer v2 |
|------|----------------|-----------------|
| 生成框架 | LDM（潜在扩散） | Latent Flow Matching |
| 采样步数 | 50 步 DDIM | 4~8 步 ODE |
| 路径 | VP 噪声调度 | Rectified Flow |
| 推理速度 | ~800ms/帧 | ~80ms/帧（10×加速） |
| 条件控制 | HDMap + 3D bbox | HDMap + 3D bbox + 轨迹 |
v2 的升级验证了一个重要趋势：**从扩散到 Flow Matching 的迁移可以带来实际的数量级推理速度提升**。
---
## 💭 个人思考
站在统一视角下看待扩散和 Flow Matching，我认为有两点值得深入思考：
1. **扩散 vs Flow Matching 不是竞争关系，而是不同层级的抽象**。扩散是 Flow Matching 在噪声调度上的一个具体实例。理解了这个统一框架后，很多"新技术"无非是在这个框架的某个维度上做调整——改变路径、改变目标、改变求解器。**掌握统一视角，就能一眼看穿新方法的本质"新"在哪里**。
2. **Flow Matching 在自动驾驶中的应用才刚刚开始**。扩散模型已经在图像/视频生成中建立了强大的技术栈（CFG、LDM、DiT），Flow Matching 可以直接继承这些成熟技术，同时提供更快的采样和更简单的训练。我认为未来 1-2 年，**自动驾驶领域的生成式模块（轨迹规划、场景预测、数据增强）将加速从扩散向 Flow Matching 迁移**。
3. **Rectified Flow 的"一次训练换十倍加速"是目前性价比最高的升级路径**。它不需要改变网络架构、不需要蒸馏、不需要额外的 loss 项，只是换一种训练方式。对已经在用扩散的团队，**迁移到 Rectified Flow 是管线改造最小的加速方案**。
