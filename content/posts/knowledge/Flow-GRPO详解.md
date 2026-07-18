---
title: "Flow-GRPO 详解：流匹配策略的强化学习优化"
date: 2026-07-19
draft: false
categories: ["知识点拆解"]
tags: ["🌊 Flow Matching", "🎮 强化学习", "⚡ GRPO", "🎨 扩散模型", "🚗 自动驾驶"]
summary: "Flow-GRPO 将 Group Relative Policy Optimization 引入流匹配策略训练，让扩散/流匹配模型不再只靠模仿学习，而是能通过奖励信号自主探索更优的驾驶策略。"
weight: 7
---

![Flow-GRPO 整体概览：ODE-to-SDE 转换 + Denoising Reduction + 组内奖励归一化（来源：NeurIPS 2025）](/images/flow-grpo/overview.png)

![Flow-GRPO 训练流程：采样 → 打分 → 组内优势计算 → 策略更新（来源：NeurIPS 2025）](/images/flow-grpo/pipeline.png)

## 一句话理解 Flow-GRPO

> **Flow-GRPO = 把"扩散/流匹配的多步去噪过程"当成一条可被强化学习优化的轨迹，用最终结果的奖励信号去反向更新每一步的去噪策略。**

传统的扩散/Flow Matching 模型只会"模仿数据分布"，而 Flow-GRPO 让模型能根据**任务奖励**（如图文一致性、人类偏好，或自动驾驶中的安全舒适性）**自主探索更优的生成路径**。它由字节跳动提出（arXiv:2505.05470），最初在 SD3 上验证，现已扩展到 FLUX、Qwen-Image、Wan2.1、Bagel 等模型。本文结合源码笔记，把这个算法彻底拆开。

---

## 🌊 Flow Matching 策略：用 ODE 生成轨迹

要理解 Flow-GRPO，先要理解 **Flow Matching 策略**到底是什么。

普通的回归式模型（如直接回归轨迹点 $(x,y,yaw)$ 的网络）只能学到一个**确定性映射**：输入场景 → 输出唯一答案。而 Flow Matching 策略把生成过程建模成一条从噪声到数据的**连续轨迹**：

$$\frac{dx_t}{dt} = v_\theta(x_t, t, c)$$

其中 $v_\theta$ 是网络预测的**速度场**，$c$ 是条件（图像任务里是 prompt，驾驶任务里是场景上下文）。采样时从纯噪声 $x_0 \sim \mathcal{N}(0, I)$ 出发，沿速度场一步步积分到 $x_1$（数据样本）。这就是 **Rectified Flow / Flow Matching** 的核心，Stable Diffusion 3 用的就是它。

**对比确定性回归，Flow Matching 策略有三个关键优势：**

| 维度 | 确定性回归 | Flow Matching 策略 |
|------|-----------|-------------------|
| **输出形式** | 唯一答案 | 一个**分布**（多模态） |
| **多解支持** | ❌ 无法表达"多种合理轨迹" | ✅ 不同噪声 → 不同样本 |
| **可微分性** | 直接对输出求梯度 | 轨迹中间状态可插值、可控 |
| **生成步数** | 1 步前向 | 通常 5–28 步去噪 |

在自动驾驶里，这意味着：同一个拥堵路口，模型可以生成**多条不同但都合理**的未来轨迹（让行/变道/缓行），这正是规划需要的多模态性。

**采样步数的工程含义**需要特别澄清：配置里的 `config.sample.num_steps = 10` **不是训练 10 个 epoch**，而是每张样本从噪声生成到结果 latent 时一共走的 flow/SDE 步数。训练采样少（10 步）是为了降低在线采样成本——因为 RL 每轮都要生成大量样本并打 reward；而评估时可以走更多步（如 40 步）以拿到更高质量的结果。这个"训练少步、评估多步"的非对称设计，是 Flow-GRPO 工程上的一个关键取舍。

---

## 🤔 为什么要在 Flow Matching 上做强化学习

Flow Matching 的预训练目标是**分布拟合**（让 $v_\theta$ 逼近数据构造的目标速度），本质上是一种**模仿学习（Behavior Cloning）**。这带来几个根本性局限：

### 1. 分布偏移（Covariate Shift）

模仿学习只在"专家轨迹附近"学得好。一旦上线时模型自己的输出偏离了训练数据分布，就没有任何信号把它拉回来，错误会**指数级累积**。驾驶场景的开放性让这个问题尤其严重。

### 2. 无法超越专家

BC 的损失是"像不像数据"，它的上限就是**专家水平**。而人类驾驶数据本身可能保守、犹豫、舒适性不稳定——模型会把这些不足也学进去。

### 3. 真正的目标不可微

我们关心的指标——**碰撞率、舒适性 jerk、规则违反、路线完成度**——很难写成可微的监督标签。同一个场景往往有多条可行轨迹，监督学习根本说不出"哪一条是标准答案"。

**三者的对比：**

| 范式 | 优化目标 | 能否超越数据 | 对稀疏指标的处理 |
|------|---------|------------|----------------|
| **监督/Flow Matching** | 似然/MSE 拟合数据 | ❌ 上限是数据质量 | 难（需硬编码标签） |
| **RL（Flow-GRPO）** | 任务奖励 | ✅ 可探索更优解 | 直接转化为 reward |

RL 的核心价值在于：**它优化的是"结果好不好"，而不是"像不像数据"**——这两个目标本就不是一回事。Flow-GRPO 就是把这种 RL 思想注入到 Flow Matching 的去噪过程中。

还有一个常被忽略的点：**预训练目标（似然）和后训练目标（奖励）本质上是解耦的**。基础模型先用海量数据学会"会生成"，把数据分布的先验压进权重；再用少量计算量的在线 RL，把这个先验**对齐**到具体任务偏好上。这种"先预训练、后 RL 对齐"的两阶段范式，和语言模型里 SFT → RLHF 的思路一脉相承，只是 Flow-GRPO 把它搬到了连续生成的去噪轨迹上。

---

## 🎯 GRPO 方法详解：组内相对优势

Flow-GRPO 用的强化学习算法是 **GRPO（Group Relative Policy Optimization）**，它源自 DeepSeek 的语言模型工作。

### 从 PPO 说起

PPO 用旧策略采样数据，再用当前策略做 **clipped objective**：

$$\text{ratio} = \exp(\log \pi_{\text{new}}(a|s) - \log \pi_{\text{old}}(a|s))$$

$$\mathcal{L} = -\mathbb{E}\big[\min\big(r \cdot A,\ \text{clip}(r, 1-\epsilon, 1+\epsilon)\cdot A\big)\big]$$

PPO 通常需要一个独立的 **critic（value network）** 来估计 baseline，从而算出 advantage $A$。这个 critic 体积和 policy 一样大，训练成本高、调参难。

### GRPO 的关键差异

GRPO 的核心洞察是：**critic 可以被"组内均值"替代**。对同一个 prompt 采样一组 $G$ 个结果，用组内 reward 的均值和标准差做 baseline：

$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G}) + \epsilon}$$

**PPO 与 GRPO 的对比：**

| 维度 | PPO | GRPO |
|------|-----|------|
| **baseline 来源** | Critic 网络估计 | 组内 reward 均值 |
| **是否需要 critic** | ✅ 必须 | ❌ 不需要 |
| **显存/调参成本** | 高（双模型） | 低 |
| **advantage 计算** | $Q - V$ | $(r_i - \bar r)/\sigma$ |
| **group size** | 无 | 关键超参（如 24） |

源码里这个逻辑位于 `flow_grpo/stat_tracking.py` 的 `PerPromptStatTracker.update()`：

```python
for prompt in unique:
    prompt_rewards = rewards[prompts == prompt]
    mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
    std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
    advantages[prompts == prompt] = (prompt_rewards - mean) / std
```

**一个直观例子：** 同一个 prompt 生成 4 张图，reward 是 $[0.2, 0.4, 0.8, 0.6]$，均值 $0.5$。那么 $0.8$ 的那张得到**正 advantage**（提高其生成概率），$0.2$ 的得到**负 advantage**（降低其生成概率）。这就是"**最终结果打分，整条生成路径领奖或背锅**"。

### 为什么"组内相对"比"绝对 reward"更合理

直接用绝对 reward 训练会有严重的**难度偏置**问题。简单 prompt（如"a red apple"）平均 reward 天然高，复杂 prompt（如"three dogs wearing hats under a bridge"）天然低；如果全局比较，模型会一味偏向简单 prompt 的生成路径，复杂场景的能力反而退化。组内相对把"绝对好不好"换成"在同题候选里排第几"，自动抵消了不同 prompt 之间的难度差异。这个洞察对驾驶同样成立——直路场景的 reward 普遍高于无保护左转，必须按**场景分组**比较才有意义。

> 💡 当 reward 稀疏导致同组方差为 0 时，可设置 `config.sample.global_std=True`，用全局 std 替代组内 std，避免训练信号消失。

---

![Flow-GRPO 训练循环细节：Denoising Reduction 和 ODE-to-SDE 转换示意（来源：NeurIPS 2025）](/images/flow-grpo/training_loop.png)

![Flow-GRPO 实验效果：GenEval 从 63% 提升至 95%，Text Rendering 从 59% 提升至 92%（来源：NeurIPS 2025）](/images/flow-grpo/results.png)

## 🔧 Flow-GRPO 核心：从离散 token 迁移到连续去噪

这是整个算法最精妙的地方。GRPO 最初是为**离散 token**设计的（语言模型），而 Flow Matching 是**连续 latent 空间**的迭代去噪。Flow-GRPO 怎么把它们对接起来？

### 三个关键改造

**改造 1：把"去噪转移"定义为 action。** 在语言模型里 action 是下一个 token；在 Flow-GRPO 里，action 是从 $x_t$ 到 $x_{t-1}$ 的一步 latent 转移：

| RL 概念 | 语言模型 GRPO | Flow-GRPO |
|--------|--------------|-----------|
| **state** | 已生成 token | 当前 latent $x_t$ + 条件 + 时间 $t$ |
| **action** | 下一个 token | $x_t \to x_{t-1}$ 的转移 |
| **policy** | $\pi(\text{token})$ | 去噪网络预测的速度 |
| **reward** | 答案正确性 | **最终图像/轨迹**的任务 reward |
| **轨迹长度** | 序列长度 | 采样步数（如 10 步） |

**改造 2：必须引入 SDE 才有 log-prob。** 纯 ODE 采样是确定性的——给定初始噪声和 prompt，每一步没有概率密度可言，PPO/GRPO 的 importance ratio $r = \exp(\log\pi_{\text{new}} - \log\pi_{\text{old}})$ 根本算不出来。所以 Flow-GRPO 把 flow step 改造成 **SDE**：

$$x_{t-1} = \underbrace{\mu_\theta(x_t, t)}_{\text{模型预测的均值}} + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这样 $x_{t-1}$ 就是从一个高斯分布里采样出来的，可以解析地写出 log-prob。源码在 `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`，log-prob 就是这个高斯的密度取对数，再对所有 latent 维度取均值。

**改造 3：最终 reward 广播到每个去噪步。** 中间 latent 没有直接的图像/轨迹语义，reward model 只能对**最终 decode 后的结果**打分。于是代码把每个 prompt 的一个最终 reward 复制成 10 份，分摊给这条轨迹的每个可训练 step：

```python
samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1) \
    .repeat(1, num_train_timesteps)
```

> ⚠️ 这是最容易误解的点：**不是每一步都重新打分**，而是"完整 10 步生成完成后统一算 reward 和 advantage；训练时把同一个 advantage 用到该轨迹每个去噪步的 log-prob ratio 上"。

准确地说，Flow-GRPO 的语义是：**GRPO 的 group 是"同 prompt 的多张最终结果/完整轨迹"，PPO ratio 的 action 粒度是"轨迹里的每个 latent transition"，最终 reward 通过 advantage 被广播到轨迹中的多个 action**。这种"终点打分、整链更新"的设计，本质上是把一条马尔可夫去噪链当成一个**多步 action 序列**来优化，reward 只在终点出现，但梯度通过每一步的 transition log-prob 反传到去噪网络。这也是为什么中间 latent 不需要单独的 reward model——它们的价值由最终结果的好坏间接体现。

---

## 🔄 训练流程全解

一轮完整 epoch 的训练流程可以拆成五个阶段。

### 阶段 1：数据收集（采样 + 打分）

对每个 prompt，用 `DistributedKRepeatSampler` 把它重复 $K$ 次（即 group size，如 24），多卡 gather 后同一 prompt 会产生 24 张候选图。每张图走完整 `num_steps`（如 10 步）SDE 采样，记录**中间 latent、每步 log-prob、最终图像**。采样完后异步把最终图像送给 reward model 打分。

### 阶段 2：计算组内优势

等待所有 reward future 完成后，调用 `PerPromptStatTracker.update()`，按 prompt 分组做 $(r_i - \bar r)/\sigma$ 归一化，得到每个样本的 advantage。

### 阶段 3：更新去噪网络

进入两层循环——外层遍历 minibatch，内层遍历每个去噪步 $j$：

```python
for j in range(num_train_timesteps):
    # 用当前模型重新算这一步 transition 的 log-prob
    prev_sample, log_prob, _, _ = compute_log_prob(model, latents[:, j],
                                                    next_latents[:, j], t=j)
    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
    unclipped = -advantages[:, j] * ratio
    clipped = -advantages[:, j] * torch.clamp(ratio, 1-clip_range, 1+clip_range)
    policy_loss = torch.mean(torch.maximum(unclipped, clipped))
    loss = policy_loss + beta * kl_loss
    loss.backward(); optimizer.step()
```

### 关键张量形状（以 SD3、`num_steps=10` 为例）

| 张量 | 形状 | 含义 |
|------|------|------|
| `latents` | $(B, T+1, C, h, w)$ | 完整轨迹的中间 latent |
| `log_probs` | $(B, T)$ | 每步旧模型 log-prob |
| `rewards["avg"]` | $(B,) \to (B, T)$ | 最终 reward 复制到时间维 |
| `advantages` | $(B, T)$ | 广播后的优势 |

> 💡 **KL 约束**：当 `beta > 0` 时，额外约束 LoRA policy 的 `prev_sample_mean` 不要偏离 base model 太远（MSE 形式正则）。这对驾驶安全至关重要——防止 RL 把策略推到危险区域。

### 一个细节：on-policy 一致性校验

Flow-GRPO 是 on-policy 算法——采样用的模型和被训练的模型应该是同一个。README 给出一个自检技巧：设置 `num_batches_per_epoch=1` 且 `gradient_accumulation_steps=1`，此时 ratio 应当**严格等于 1**。如果发现 ratio 偏离，多半是采样路径和训练路径不一致导致的（如 `torch.compile` 包装、batch size 不同引起 SD3 的数值差异等）。这种数值一致性是 RL 稳定训练的前提，源码里特意用 `fp16` 而非 `bf16` 来减小 log-prob 的收集/训练误差（误差在低噪声步更大，所以建议优先训练高噪声步）。

---

## 💰 奖励函数设计

Flow-GRPO 把奖励做成**统一接口**：`reward_fn(images, prompts, metadata) -> scores`，入口在 `flow_grpo/rewards.py` 的 `multi_score()`。它支持多个 reward 加权相加，最终只认汇总的 `avg`：

```python
score_details['avg'] = total_scores  # 各子 reward 加权和
```

**支持的奖励（及自动驾驶对应物）：**

| 图像 reward | 含义 | 驾驶 reward 对应 |
|------------|------|-----------------|
| **GenEval** | 物体数量/颜色/关系是否正确 | 规则 reward（红停绿行、让行） |
| **OCR** | 文字渲染正确率 | 关键约束满足度 |
| **PickScore** | 人类偏好模型评分 | 偏好/风格 reward |
| **CLIPScore** | 图文匹配度 | 与导航指令一致性 |
| **ImageReward** | 综合质量+安全 | 安全 reward（碰撞/TTC） |
| **Aesthetic** | 审美评分 | 舒适性（jerk/加速度） |
| **JPEG compressibility** | 可压缩性（玩具 reward） | 效率/路线进展 |

**多 reward 加权示例**：`{"pickscore": 0.5, "ocr": 0.2, "aesthetic": 0.3}`。驾驶里可以类似设计 $\mathcal{R} = w_1 \cdot \text{safety} + w_2 \cdot \text{rule} + w_3 \cdot \text{comfort} + w_4 \cdot \text{progress} + w_5 \cdot \text{imitation}$，其中 imitation 项相当于图像里的 KL 约束，防止完全丢掉专家数据。

> ⚠️ **Reward hacking 风险**：模型可能"钻空子"（如为安全 reward 原地不动、为舒适 reward 永远慢开）。必须用多目标加权 + KL 约束 + 闭环仿真评估三重防护。

---

## 🚀 CPS 采样与 Flow-GRPO-Fast

### CPS：系数保持采样

源码里 SDE step 有两种模式：标准 `sde` 和 `cps`（**Coefficients-Preserving Sampling**，arXiv:2509.05952）。CPS 重新设计了均值和方差：

```python
std_dev_t = sigma_prev * sin(noise_level * pi / 2)
pred_original_sample = sample - sigma * model_output
prev_sample_mean = pred_original_sample * (1 - sigma_prev) + ...
log_prob = -((prev_sample - prev_sample_mean) ** 2)  # 去掉常数项
```

README 指出 CPS 在 GenEval 上有显著提升，采样质量更高，典型设置 `noise_level = 0.8`，跨模型、跨步数都好用，几乎免调。Fast/no-CFG 配置默认用 `sde_type="cps"`。

### Flow-GRPO-Fast：只训一两个窗口步

标准版每条轨迹的 10 个 step 都要算 log-prob 和反传，成本高。Flow-GRPO-Fast 的思路是：**完整生成照常走完，但只在一个随机窗口内注入随机性并训练**：

```python
start = random.randint(0, num_steps//2 - window_size)
end = start + window_size
# 窗口外：noise_level=0，确定性 ODE 传播
# 窗口内：注入 SDE 随机性，记录 log-prob 参与训练
```

**三个加速变体的对比：**

| 方法 | 每条轨迹训练步数 | 特点 |
|------|----------------|------|
| **Flow-GRPO** | 10 步全训 | 基线，最稳 |
| **Flow-GRPO-Fast** | 1–2 步（窗口） | 训练快数倍，PickScore 上 2 步即追平 |
| **GRPO-Guard** | 10 步 | 加 RatioNorm + Gradient Reweight，抑制过优化 |

实验显示 Flow-GRPO-Fast 在 PickScore 上**只用 2 步训练**就能达到甚至超过 10 步的标准版；同时训练和评估都不用 CFG，RL 过程相当于顺带做了 **CFG distillation**。

### GRPO-Guard：对抗过优化

Flow-GRPO 训练久了会出现一个隐患——**reward 越涨、真实质量越差**（即 over-optimization，模型学会了钻 reward 漏洞）。GRPO-Guard 发现 importance ratio 存在**固有偏置**：其均值持续小于 1，且在低噪声步（如 SD3.5-M 的第 8 步）特别明显。这导致 PPO 的 clipping 机制失衡——本应被裁掉的过自信正样本梯度逃过了约束，把策略推向 reward hacking。GRPO-Guard 用两个机制缓解：**RatioNorm**（矫正 ratio 分布偏置并统一各步方差）+ **Gradient Reweight**（基于 RatioNorm 对不同去噪步梯度重新加权）。对自动驾驶这类安全敏感任务，这种防护尤其值得借鉴。

---

## ⚔️ 与 DiffGRPO 的对比

ReCogDrive 等工作把 GRPO 用在了**扩散模型（DDPM-style）**的动作头上（即 DiffGRPO），思路相近但底座不同。Flow-GRPO 与之的关键差异：

| 维度 | DiffGRPO（扩散底座） | Flow-GRPO（流匹配底座） |
|------|--------------------|-----------------------|
| **生成路径** | 弯曲的 SDE 反向链 | 直线 ODE/Rectified Flow |
| **采样步数** | 较多（20+ 步） | 少（10 步甚至 1–2 步） |
| **sde_type** | 传统 DDPM 噪声调度 | 支持 `sde` 和 `cps` 两种 |
| **工程加速** | 一般 | Flow-GRPO-Fast 窗口 + No-CFG 蒸馏 |
| **过优化防护** | 标准 clipping | GRPO-Guard（RatioNorm + 重加权） |
| **应用领域** | 自动驾驶轨迹 | 图像/视频生成（思路可迁移） |

**核心结论**：两者都是"GRPO × 生成模型"，但 Flow-GRPO 借助 Flow Matching 的**直线路径**天然支持少步采样，配合 Fast 窗口机制和 CPS，训练效率显著更高；而 DiffGRPO 的优势在于与自动驾驶扩散头（如一些 planner 用 DDPM）天然兼容。**对使用 Flow Matching 动作头的 VLA**（这正是当前趋势），Flow-GRPO 的迁移更直接。

---

## 🚗 自动驾驶迁移启示

如果你的 VLA 用 Flow Matching 生成未来轨迹，Flow-GRPO 几乎可以"换皮"迁移。对应关系非常清晰：

| Flow-GRPO（图像） | 自动驾驶 Flow-GRPO |
|------------------|-------------------|
| prompt | 场景上下文（BEV/相机/地图/指令） |
| image latent | noisy future trajectory |
| 最终 image | 最终规划轨迹 |
| `num_image_per_prompt=24` | `num_traj_per_scene=16` |
| image reward | safety + comfort + rule + progress |
| SD3 transformer | VLA action flow 模型 |
| KL to base model | imitation / 专家约束 |

**最小落地路线**：①先训好 BC + Flow Matching 基础策略；②实现 `sample_with_logprob()`（参考 `sd3_pipeline_with_logprob.py`）让采样返回中间轨迹和 log-prob；③同场景生成 $K$ 条候选轨迹并打 reward；④算组内 advantage；⑤clipped policy loss + KL 约束更新；⑥**务必闭环仿真评估**，不能只看训练 reward——开环 reward 好不等于闭环安全。

> ⚠️ **不要直接在线真车探索**。建议顺序：离线采样候选 → 离线 reward 后训练 → 仿真闭环 → 安全过滤 → 小规模 shadow mode。

---

## 📌 总结：记住这五点

1. **Flow Matching 策略**用 ODE 积分生成多模态样本，比确定性回归更适合规划。
2. **要上 RL 是因为**模仿学习有分布偏移、无法超越专家、目标不可微。
3. **GRPO** 用组内均值替代 critic，省显存好调参，advantage $= (r_i - \bar r)/\sigma$。
4. **核心改造**三件事：去噪转移当 action、引入 SDE 才能算 log-prob、最终 reward 广播到每一步。
5. **工程利器**：CPS 采样提升质量、Flow-GRPO-Fast 窗口机制大幅加速、GRPO-Guard 防止过优化。

一句话再强调那条最容易踩坑的：**奖励/优势在完整生成后统一算，参数更新沿轨迹每一步用 log-prob ratio 分摊——最终结果打分，整条去噪路径领奖或背锅。** 这就是 Flow-GRPO 把 RL 接进 Flow Matching 的全部精髓。
