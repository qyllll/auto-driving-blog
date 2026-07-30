---
title: "Flow-GRPO 完全讲解：与扩散模型的在线强化学习训练"
date: 2026-07-30
draft: false
categories: ["个人思考"]
summary: "深入理解 Flow-GRPO——用 GRPO（Group Relative Policy Optimization）训练 Flux 扩散模型的完整框架。本文从原理到代码，逐层展开，解释每个模块的角色、数据流和设计动机。"
tags: ["Flow Matching", "GRPO", "强化学习", "扩散模型", "Flux", "SDE"]
math: true
weight: 99
---

## 本文导览

Flow-GRPO 是将 **GRPO（Group Relative Policy Optimization）** 应用于 **Flow Matching（流匹配）** 模型的在线强化学习训练框架。它挑战了"扩散模型只能做模仿学习 / 监督微调"的固有认知——用 RL 来优化扩散策略，**直接在推理阶段根据下游任务的 reward（奖励）信号更新模型参数**。

**逻辑树总图：**

![Flow-GRPO 训练循环总图](/images/flowgrpo/flowgrpo_overview.svg)

本文的讲解路径：

1. **GRPO 基础**：为什么需要 GRPO？它跟 PPO 的区别是什么？
2. **Flow Matching 基础**：扩散模型怎么变成了"速度场"？
3. **Flow-GRPO 训练流程**：从 train.txt 里的 prompt 到模型参数更新的完整逻辑树
4. **Flow-GRPO-Fast 加速**：SDE 滑动窗口技术
5. **Reward 系统**：各种 reward 函数的设计
6. **关键超参数**：config 里每个参数的意义
7. **实验结果**：训练出了什么效果？

---

## 1 | 先看 GRPO：不需要 Critic 的 PPO

### 1.1 标准 PPO 的痛点

PPO（Proximal Policy Optimization）是目前最主流的在线 RL 算法。它的目标函数是：

$$ L^{PPO} = \mathbb{E}\left[ \min\left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot A^{\pi}(s,a), \;\; \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right) \cdot A^{\pi}(s,a) \right) \right] $$

其中 advantage $A^{\pi}(s,a)$ 一般通过 GAE（Generalized Advantage Estimation）计算：

$$ A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s) $$

这意味着你需要一个 **Critic（评价者）网络**来估计状态价值函数 $V^{\pi}(s)$。Critic 网络通常和 Policy 网络结构类似（比如都是 Transformer），这导致：

- 模型参数量翻倍 → 显存翻倍
- 需要同时维护 Policy 和 Critic 两个优化器
- 训练不稳定：Critic 的估计误差会引入 bias

### 1.2 GRPO 的核心洞察

GRPO（Group Relative Policy Optimization）来自 DeepSeek 团队的 DeepSeekMath 论文。它提出了一个极简的想法：

> **不需要 Critic！用组内统计量代替价值函数。**

具体来说：对于同一个 prompt（或 state），采样 G 个不同的输出（action），计算这 G 个输出的 reward，然后**在组内做归一化**：

$$ A_i = \frac{r_i - \mu_{group}}{\sigma_{group}} $$

其中 $\mu_{group} = \frac{1}{G}\sum_{i=1}^{G}r_i$, $\sigma_{group} = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i - \mu_{group})^2}$.

**这个公式为什么 work？**

- 若当前输出的 reward 比组内平均好 → $A_i > 0$ → 梯度更新会**提高**这个输出路径的概率
- 若当前输出的 reward 比组内平均差 → $A_i < 0$ → 梯度更新会**降低**这个输出路径的概率
- 关键在于：**不需要绝对准确的 reward 值，只需要相对比较！**

### 1.3 GRPO vs PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 需要 Critic | ✅ 需要额外 Value Network | ❌ 不需要 |
| 参数量 | Policy + Critic ≈ 2×Policy | Policy ≈ 1×Policy |
| Advantage 来源 | GAE (需要多步采样或 TD) | 组内归一化 |
| 代价 | 增加一次前向传播（Critic） | 增加采样量（group_size 更大） |
| 适合场景 | 单步 reward 可得的场景 | 每步 reward 可得 + 可批量采样 |

GRPO 的缺点也很明显：**组内必须有足够多样本**才能让 $\mu, \sigma$ 的估计可靠。Flow-GRPO 的典型 group_size = 24（分布在 32 张 GPU 上）。

---

## 2 | 再看 Flow Matching：扩散模型的"速度"视角

### 2.1 从 Diffusion 到 Flow

传统扩散模型（DDPM）定义了一个**噪声预测**问题：

$$ \mathcal{L}_{DDPM} = \mathbb{E}_{t,x_0,\epsilon}\left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right] $$

其中 $\epsilon_\theta$ 预测当前时刻的噪声，推理时从纯噪声 $x_T$ 逐步去噪到 $x_0$。

**Flow Matching**（也叫 Rectified Flow / Stochastic Interpolation）换了一个角度：不再预测噪声，而是预测**速度场（vector field）**。

给定数据分布 $p_1$ 和高斯分布 $p_0$（$x_0 \sim \mathcal{N}(0,I)$），定义线性插值路径：

$$ x_t = (1-t) \cdot x_0 + t \cdot x_1 \quad \text{for } t \in [0,1] $$

其中 $x_1$ 是真实数据。对时间求导：

$$ \frac{dx_t}{dt} = x_1 - x_0 $$

所以目标函数变成：

$$ \mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1}\left[ \| (x_1 - x_0) - v_\theta(x_t, t) \|^2 \right] $$

即：**学习一个向量场 $v_\theta$，让它匹配从噪声到数据的直线速度**。

推理时（采样），从 $x_0 \sim \mathcal{N}(0,I)$ 出发，沿预测的速度场积分：

$$ \frac{dx_t}{dt} = v_\theta(x_t, t) \quad \rightarrow \quad x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t $$

### 2.2 Flow vs DDPM 对比图

![Flow Matching 去噪过程](/images/flowgrpo/flowgrpo_flow_matching.svg)

**关键区别总结：**

| 维度 | DDPM | Flow Matching |
|------|------|---------------|
| 预测目标 | 噪声 $\epsilon$ | 速度 $v$ |
| 训练损失 | $\|\epsilon - \epsilon_\theta\|^2$ | $\|(x_1 - x_0) - v_\theta\|^2$ |
| 采样方式 | 逐步去噪（100-1000步） | ODE/SDE 积分（10-50步） |
| 轨迹 | 离散 Markov 链 | 连续时间路径 |
| 步数 | 多（100-1000） | 少（4-50） |

Flow Matching 最大的优势是**采样步数少**——这也是 Flow-GRPO 能够用 RL 训练的关键前提。如果采样需要 1000 步，计算图会爆炸。

---

## 3 | Flow-GRPO 训练流程（逐层展开）

现在进入核心。训练流程可以用下面的逻辑树完全展开：

### 逻辑树 Level 1：主循环

```
train_flux_fast.py
│
├── 1. 初始化
│   ├── 加载 config（grpo.py × base.py）
│   ├── 初始化模型（Flux.1-dev + LoRA）
│   ├── 初始化 optimizer + dataset
│   └── 准备 reward 函数
│
├── 2. 每个 epoch 循环
│   ├── 2.1 采样阶段
│   │   ├── 读 prompt 列表（batch 个 prompt）
│   │   ├── T5 编码 prompt_embeds
│   │   ├── SDE 采样（window 模式）
│   │   └── 记录 log_prob_old + 保存图片
│   │
│   ├── 2.2 Reward 阶段
│   │   ├── 对每张图片计算 reward
│   │   └── 去中心化计算
│   │
│   ├── 2.3 同步阶段
│   │   ├── all_gather 所有 GPU 的 reward
│   │   └── PerPromptStatTracker → advantage
│   │
│   ├── 2.4 训练阶段
│   │   ├── 再次推理（梯度模式）→ log_prob_new
│   │   ├── 计算 PPO loss
│   │   └── backward() + optimizer.step()
│   │
│   └── 3. 日志 & 保存
│       ├── wandb 记录 reward / loss / 图片
│       └── 定期保存 checkpoint
```

### 逻辑树 Level 2：采样阶段的细节

采样是 Flow-GRPO 最复杂的部分，因为它需要同时满足两个要求：
1. **生成图片**（用于计算 reward）
2. **记录生成的 log_prob**（作为 $\pi_{\theta_{old}}$ 的行为）

#### 采样过程的代码结构（伪代码）：

```python
# ===== 采样阶段 =====
with torch.no_grad():
    noise = torch.randn(B, C, H, W)  # 初始噪声 x_0
    
    # 前 N 步：no_grad，不记录计算图
    for i in range(num_inference_steps - window_size):
        x_t = model(x_t, t_i, prompt_embeds)
        log_prob += compute_sde_log_prob(x_t, x_t_plus_1, ...)
    
    # 后 window_size 步：no_grad，但记录 log_prob
    for i in range(window_size):
        x_t = model(x_t, t_i, prompt_embeds)
        log_prob += compute_sde_log_prob(x_t, x_t_plus_1, ...)
    
    images = vae.decode(x_t)  # latent → pixel

# ===== Reward 阶段 =====
rewards = reward_fn(images, prompts)

# ===== 训练阶段 =====
# 再次做 SDE 推理，但这次开启梯度
x_t = noise
for i in range(num_inference_steps - window_size):
    x_t = model(x_t, t_i, prompt_embeds)  # no_grad
for i in range(window_size):
    x_t = model(x_t, t_i, prompt_embeds)  # grad ON
    log_prob_new += compute_sde_log_prob(x_t, x_t_plus_1, ...)

# PPO loss
ratio = torch.exp(log_prob_new - log_prob_old)
loss = -torch.min(
    ratio * advantage,
    torch.clamp(ratio, 1-eps, 1+eps) * advantage
)
loss.backward()
optimizer.step()
```

### 逻辑树 Level 3：log_prob 的计算

这是 Flow-GRPO 最精妙的地方。在 SDE 采样过程中，每一步的转移概率可以解析计算。

对于 SDE：

$$ dx = v_\theta(x, t) dt + g(t) dW $$

每一步的转移概率是高斯分布：

$$ p(x_{t+1}|x_t) = \mathcal{N}\left(x_t + v_\theta(x_t, t)\Delta t, \; g(t)^2\Delta t \cdot I\right) $$

所以 log_prob 是：

```python
def compute_sde_log_prob(x_t_plus_1, x_t, v_pred, noise_std, dt):
    mean = x_t + v_pred * dt
    var = noise_std ** 2 * dt
    diff = x_t_plus_1 - mean
    log_prob = -0.5 * (diff ** 2 / var + torch.log(2 * torch.pi * var))
    return log_prob.sum(dim=(1, 2, 3))
```

**注意一个重要的细节**：Flow-GRPO 的 `log_prob_old` 是在**采样阶段**（`torch.no_grad()`）记录的，而 `log_prob_new` 是在**训练阶段**（开启梯度）计算的。由于采样阶段和训练阶段用的是**同一组模型参数**（在线强化学习的特性），ratio 的期望值理论上是 1.0——这意味着 GRPO 在每一步开始时的 KL 散度接近 0，保证了训练稳定性。

---

## 4 | PerPromptStatTracker：Advantage 计算的完整实现

这是 `flow_grpo/stat_tracking.py` 中最重要的类。它的设计值得仔细分析。

### 4.1 数据结构

```python
class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size      # 统计缓冲区大小
        self.min_count = min_count           # 最小计数
        self.stats = {}                      # {prompt: [reward1, reward2, ...]}
```

`stats` 字典以 prompt 为 key，存储该 prompt 的**历史 reward 列表**。这允许计算 running mean/std 来替代纯 group 内的归一化。

### 4.2 Update 方法

```python
def update(self, prompts, rewards):
    # 将新 reward 追加到对应 prompt 的历史队列
    for prompt, reward in zip(prompts, rewards):
        if prompt not in self.stats:
            self.stats[prompt] = []
        self.stats[prompt].append(reward)
        # 保持 buffer 大小
        if len(self.stats[prompt]) > self.buffer_size:
            self.stats[prompt].pop(0)
    
    # 对每个 prompt 计算 advantage
    advantages = []
    for prompt, reward in zip(prompts, rewards):
        prompt_stats = self.stats[prompt]
        if len(prompt_stats) < self.min_count:
            advantages.append(0.0)  # 数据不足时优势为 0
        else:
            mean = np.mean(prompt_stats)
            std = np.std(prompt_stats) + 1e-8
            advantages.append((reward - mean) / std)
    
    return advantages
```

### 4.3 核心公式

$$ A_i = \frac{r_i - \mu_{history}}{\sigma_{history}} $$

**为什么用历史统计而非 group 内统计？**

理论上 GRPO 用同一组（同一步的多个采样）做归一化。但在分布式训练中，一个 prompt 可能只在一张 GPU 上出现一次。PerPromptStatTracker 的 buffer 机制实现了**跨时间步的组内归一化**——用历史 reward 的均值和标准差来近似当前 group 的均值和标准差。

### 4.4 Advantage 计算示意图

![GRPO Advantage 计算](/images/flowgrpo/flowgrpo_grpo_advantage.svg)

---

## 5 | Flow-GRPO-Fast：SDE 滑动窗口技术

这是 Flow-GRPO 训练加速的关键创新。

### 5.1 问题：计算图太大

标准的 SDE 采样需要依次执行 10 步推理：

```
x_0 → x_1 → x_2 → ... → x_10 (final image)
```

如果每一步都用 `torch.no_grad()`，那就无法计算 loss（因为没有计算图）。如果每一步都开启 `requires_grad`，计算图会包含所有 10 步 DiT（Diffusion Transformer），Flux.1-dev 约 12B 参数，10 步的激活值 → 100B+ 激活 → GPU 显存爆炸。

### 5.2 解决方案：Window

观察：SDE 的马尔可夫性质意味着**后几步的梯度足以更新所有参数**。因此：

1. **前 N 步**：`torch.no_grad()`，不记录计算图，只存储 latent 状态
2. **后 window_size 步**：开启 `requires_grad`，构造计算图

这样计算图只包含 `window_size` 步，显存占用减少约 `(10-window_size)/10`。

### 5.3 可视化

![Flow-GRPO-Fast SDE 窗口](/images/flowgrpo/flowgrpo_fast_sde.svg)

### 5.4 代码实现

```python
# !!!!!!!!!!!! 关键函数：训练阶段的 SDE 采样（含梯度窗口）!!!!!!!!!!!!
def compute_policy_loss(
    pipe,                        # Flux pipeline
    noise,                       # 初始噪声
    prompt_embeds,               # T5 编码后的 prompt
    num_inference_steps,         # 总步数（默认 10）
    window_size,                 # 梯度窗口大小（默认 5）
    log_prob_old,                # 采样阶段记录的旧 log_prob
    advantage,                   # PerPromptStatTracker 计算的 advantage
    clip_range=0.2,              # PPO clip 范围
):
    x = noise
    
    # 阶段 1：no_grad 窗口
    for i in range(num_inference_steps - window_size):
        with torch.no_grad():
            v_pred = pipe.transformer(x, t, prompt_embeds).sample
            x = euler_step(x, v_pred, dt)  # x_t+1 = x_t + v * dt
    
    # 阶段 2：梯度窗口
    log_prob_new = 0
    for i in range(window_size):
        v_pred = pipe.transformer(x, t, prompt_embeds).sample
        x = euler_step(x, v_pred, dt)
        log_prob_new += compute_step_log_prob(x_t_plus_1, x_t, v_pred, noise_std, dt)
    
    # PPO loss
    ratio = torch.exp(log_prob_new - log_prob_old)
    pg_loss = -torch.min(
        ratio * advantage,
        torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage,
    ).mean()
    
    return pg_loss
```

### 5.5 加速效果

| 配置 | 显存占用 | 每步时间 | 梯度精度 |
|------|---------|---------|---------|
| window_size = 10（全梯度） | OOM | - | 完美 |
| window_size = 5 | ~36GB | 1.0× | 良好 |
| window_size = 3 | ~25GB | 0.7× | 可接受 |
| window_size = 1 | ~18GB | 0.5× | 较差（仅最后 1 步） |

默认 `window_size=5` 做到了显存和梯度精度的最佳平衡。

---

## 6 | Reward 系统

Flow-GRPO 实现了丰富的 reward 函数，所有函数都在 `flow_grpo/rewards.py` 中。每个 reward 函数接收 `images (PIL list)` 和 `prompts (str list)`，返回 `rewards (float list)`。

### 6.1 内置 Reward 函数

| 名称 | 类型 | 功能 | 模型 |
|------|------|------|------|
| `ocr_reward_fn` | OCR | 检测图片里文字是否匹配 prompt | EasyOCR / PPOCR |
| `gen_eval_reward_fn` | 视觉语义 | CLIP 图文对齐 + 生成质量 | CLIP + 评估器 |
| `pick_score_reward_fn` | 美学 | 图片美观度评分 | PickScore |
| `aesthetic_score_reward_fn` | 美学 | 美学评分 | LAION Aesthetic |
| `gpt_fine_grained_reward_fn` | 视觉语义 | GPT-4o 做细粒度评分 | GPT-4o API |
| `grounding_reward_fn` | 布局 | 检测物体是否出现在指定位置 | Grounding DINO |
| `color_reward_fn` | 颜色 | 主色调是否匹配 prompt | 颜色直方图 |

### 6.2 Reward 的组合使用

在 `config/grpo.py` 中，reward 可以组合使用：

```python
# OCR 实验：只用 OCR reward
experiment = dict(
    model="black-forest-labs/FLUX.1-dev",
    grpo=dict(
        reward_fn="ocr",
        ...
    )
)

# GenEval 实验：多维度 reward
experiment = dict(
    grpo=dict(
        reward_fn="gen_eval",
        ...
    )
)

# 自定义组合
experiment = dict(
    grpo=dict(
        reward_fn="custom",
        reward_weights={            # 多个 reward 加权
            "ocr": 0.5,
            "pick_score": 0.3,
            "aesthetic": 0.2,
        },
        ...
    )
)
```

### 6.3 Reward 的去中心化计算

在多 GPU 训练中，reward 计算是**去中心化**的。每张 GPU 独立计算自己生成的图片的 reward，然后通过 `all_gather` 同步所有 GPU 上的 reward 值。

```python
# 每张 GPU 计算自己的 reward
local_rewards = reward_fn(local_images, local_prompts)

# all_gather 同步所有 GPU 的 reward
all_rewards = all_gather(local_rewards)  # [global_batch_size]
all_prompts = all_gather(local_prompts)

# PerPromptStatTracker 的 update 接受全部 prompt 和 reward
advantages = stat_tracker.update(all_prompts, all_rewards)

# 每张 GPU 取自己那部分的 advantage
local_advantages = advantages[rank * local_batch_size : (rank+1) * local_batch_size]
```

这种设计的优势是**没有单点瓶颈**——计算 reward 不涉及跨 GPU 通信，只有最终的 all_gather 需要一次同步。

---

## 7 | 关键超参数详解

来自 `config/base.py` 和 `config/grpo.py` 的完整超参数解读：

### 7.1 训练超参数

```python
class GRPOConfig:
    # === 数据 ===
    dataset_path: str = "train.txt"           # 每行一个 prompt
    batch_size: int = 1                       # 每张 GPU 的 prompt 数
    # 实际：batch_size × num_processes × group_size 张图
    # 例如：1 × 32 × 24 = 768 张图/步
    
    # === 采样 ===
    num_inference_steps: int = 10             # SDE 总步数
    window_size: int = 5                      # 梯度窗口大小
    guidance_scale: float = 3.5              # CFG 强度
    
    # === GRPO ===
    group_size: int = 24                      # 每个 prompt 的采样数
    clip_range: float = 0.2                   # PPO clip 范围
    
    # === 优化 ===
    learning_rate: float = 1e-4               # LoRA 学习率
    lora_rank: int = 64                       # LoRA 秩
    lora_alpha: float = 128                   # LoRA alpha
    
    # === Reward ===
    reward_fn: str = "ocr"                    # 奖励函数类型
    reward_weights: dict = {"ocr": 1.0}       # 多 reward 权重
    
    # === 统计 ===
    buffer_size: int = 30                     # PerPromptStatTracker buffer
    min_count: int = 5                        # 最小统计数
```

### 7.2 关键参数的设计动机

**为什么 group_size = 24？**
- GRPO 的核心是组内比较，group_size 越大，$\mu, \sigma$ 估计越准确
- 但 group_size 受 GPU 显存限制：每张图需要 VAE decode + reward 模型
- 24 是一个平衡值：32 GPU × 1 prompt/GPU = 32 个 prompt，每个 group 24 张图

**为什么 num_inference_steps = 10？**
- Flow Matching 的优势就是步数少
- 10 步已经能生成高质量图片（DDPM 需要 50-1000 步）
- 步数越少，计算图越小，显存越可控

**为什么 window_size = 5？**
- 全梯度（10 步）显存 OOM
- 窗口越小，梯度信息越少（后几步的噪声影响变大）
- 5 步是经验上精度-显存的平衡点

**为什么 clip_range = 0.2？**
- 标准的 PPO 超参数
- 限制 ratio 范围在 [0.8, 1.2]，防止更新过大导致 collapse

### 7.3 FSDP 配置

Flow-GRPO 使用 PyTorch FSDP（Fully Sharded Data Parallel）来训练 12B 的 Flux：

```python
# fsdp_utils.py 中的 FSDP 配置
fsdp_config = dict(
    sharding_strategy="FULL_SHARD",           # 分片全部参数
    cpu_offload=False,                         # 不 CPU offload（加速）
    mixed_precision="bf16",                    # BF16 混合精度
    limit_all_gathers=True,                    # 限制 all_gather 内存
)
```

FSDP 的关键作用：Flux.1-dev 约 12B 参数（≈24GB BF16），如果没有 FSDP 分片，单张 GPU 放不下。

---

## 8 | 完整训练实验（OCR 场景为例）

### 8.1 实验目标

提升 Flux.1-dev 生成带有正确文字的图片的能力。Prompt 样例：

```
"A sign that says 'Hello World'"
"A book cover with title 'Deep Learning'"
"A storefront with 'OPEN' sign"
"A birthday cake with 'Happy Birthday' written on it"
```

### 8.2 训练过程

```
Epoch 0: 文字几乎不可读，OCR reward ≈ 0.05
Epoch 10: 文字开始出现但错误较多，reward ≈ 0.20
Epoch 50: 文字基本清晰，少量拼写错误，reward ≈ 0.55
Epoch 100: 文字正确率 > 90%，reward ≈ 0.85
Epoch 200: 文字高度准确，reward ≈ 0.95
```

### 8.3 可视化结果

（训练前后的图片对比：左边是初始 Flux 生成的文字（无法辨识的乱码），右边是 200 步 GRPO 训练后的文字（清晰可读））

### 8.4 训练的 KL 变化

在整个训练过程中，`ratio = exp(log_prob_new - log_prob_old)` 的均值始终保持在 1.0 附近（因为采样和训练用的是同一组模型参数），标准差从 0.05 逐步上升到 0.2 左右——说明模型在不断适应 reward 信号，但更新幅度被 clip_range 控制了。

---

## 9 | 代码架构全图

![Flow-GRPO 代码架构](/images/flowgrpo/flowgrpo_code_arch.svg)

### 9.1 文件映射

| 文件 | 职责 |
|------|------|
| `scripts/train_flux_fast.py` | 主训练脚本：采样 → reward → loss → backward |
| `config/grpo.py` | 实验配方：OCR / GenEval / PickScore 等预设 |
| `config/base.py` | 默认超参数：learning_rate, group_size, window_size 等 |
| `flow_grpo/rewards.py` | 所有 reward 函数的实现 |
| `flow_grpo/stat_tracking.py` | PerPromptStatTracker：reward → advantage |
| `flow_grpo/ema.py` | EMA（指数移动平均）模型参数 |
| `flow_grpo/fsdp_utils.py` | FSDP 分布式训练工具 |
| `flow_grpo/diffusers_patch/*` | 改造后的 Flux pipeline（SDE 采样 + log_prob） |

### 9.2 训练流程的完整代码（精简标注版）

```python
# ==== train_flux_fast.py ====

# 1. 加载配置
from config.grpo import ocr_experiment, gen_eval_experiment
config = ocr_experiment["grpo"]

# 2. 初始化模型
model = FluxTransformer.from_pretrained("black-forest-labs/FLUX.1-dev")
model = wrap_fsdp(model)  # FSDP 分片
optimizer = AdamW(model.parameters(), lr=config.learning_rate)

# 3. 加载 prompt 数据
prompts = load_prompts(config.dataset_path)

# 4. 主训练循环
for epoch in range(config.num_epochs):
    for batch_prompts in dataloader(prompts, batch_size=config.batch_size):
        
        # --- 4.1 采样阶段（no_grad）---
        # 每个 prompt → group_size 张不同噪声的图片
        batch_prompts_repeated = repeat(batch_prompts, config.group_size)
        noises = torch.randn(len(batch_prompts_repeated), ...)
        
        with torch.no_grad():
            images, log_probs_old = sde_sampling(
                model, noises, batch_prompts_repeated,
                num_steps=config.num_inference_steps,
                window_size=config.window_size,
            )
        
        # --- 4.2 Reward 阶段 ---
        rewards = reward_fn(images, batch_prompts_repeated)
        
        # --- 4.3 同步 + Advantage 计算 ---
        all_rewards = all_gather(rewards)
        all_prompts = all_gather(batch_prompts_repeated)
        advantages = stat_tracker.update(all_prompts, all_rewards)
        
        # 取本 GPU 对应的 advantage
        local_advantages = get_local_advantages(advantages)
        
        # --- 4.4 训练阶段（梯度开启）---
        _, log_probs_new = sde_sampling(  # 再次采样，但保持输入一致
            model, noises, batch_prompts_repeated,
            num_steps=config.num_inference_steps,
            window_size=config.window_size,
            compute_grad=True,
        )
        
        # --- 4.5 PPO Loss ---
        ratio = torch.exp(log_probs_new - log_probs_old)
        pg_loss = -torch.min(
            ratio * local_advantages,
            torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * local_advantages,
        ).mean()
        
        # KL 惩罚（可选）
        kl_loss = (ratio - 1 - torch.log(ratio)).mean()
        loss = pg_loss + config.kl_coef * kl_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # --- 4.6 日志 ---
        wandb.log({
            "reward/mean": rewards.mean(),
            "reward/std": rewards.std(),
            "loss/pg": pg_loss.item(),
            "loss/kl": kl_loss.item(),
            "ratio/mean": ratio.mean().item(),
        })
```

---

## 10 | 总结与思考

### 10.1 Flow-GRPO 的核心贡献

1. **首次将 GRPO 引入扩散模型训练**：证明了在线 RL 可以显著提升扩散模型的下游任务性能
2. **SDE 窗口技术**：解决了 SDE 采样计算图过大的工程难题
3. **丰富的 reward 系统**：OCR / GenEval / PickScore / 美学 / 布局等，覆盖多种任务
4. **去中心化 reward 计算**：避免了单点瓶颈，支持大规模分布式训练

### 10.2 局限与挑战

1. **显存仍然是瓶颈**：即使有 window_size=5，Flux.1-dev 的训练仍需要 8×80GB GPU
2. **reward 设计依赖人工**：OCR 一类的 reward 很好定义，但"生成质量"很难用单个数值衡量
3. **多样性可能下降**：RL 训练倾向于收敛到少数高 reward 模式，需要 careful prompt 设计
4. **ON-POLICY 采样开销大**：每一步都需要重新采样，训练成本是离线方法的数倍

### 10.3 可能的应用方向

- **广告图文生成**：通过 OCR reward 确保文字准确
- **产品图生成**：通过 grounding reward 确保物体在指定位置
- **艺术风格生成**：通过 aesthetic reward 提升美学质量
- **医学影像生成**：通过 domain-specific reward 确保解剖正确性
- **与 VLM 结合**：用 GPT-4o/VLM 做 reward 模型，实现开放式质量的自动评估

---

## 参考

- Flow-GRPO 代码库：[https://github.com/your-org/Flow-GRPO](https://github.com)
- GRPO 论文：DeepSeekMath (arXiv:2402.03300)
- Flow Matching 论文：Flow Matching for Generative Modeling (arXiv:2210.02747)
- Flux 模型：black-forest-labs/FLUX.1-dev
- PPO 论文：Proximal Policy Optimization Algorithms (arXiv:1707.06347)
