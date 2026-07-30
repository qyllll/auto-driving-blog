---
title: "Flow-GRPO 完全讲解：训练/推理/梯度流/Loss 设计的逐行拆解"
date: 2026-07-30
draft: false
categories: ["个人思考"]
summary: "从 train_flux_fast.py 第 1 行开始，逐层追踪 Flow-GRPO 的完整逻辑链：采样阶段做了什么？reward 怎么变成 advantage？训练阶段的计算图是怎么构造的？loss 为什么那样设计？梯度如何从最后一个 log_prob 传到 LoRA 参数？每段代码都标注了源文件行号。"
tags: ["Flow Matching", "GRPO", "强化学习", "扩散模型", "Flux", "SDE", "梯度流"]
math: true
weight: 99
---

## 本文的讲解方式

**不从概念讲起，而是从代码的入口出发，沿着执行路径一步一步深入。**

我们追踪一辆"数据快车"：

```
train_flux_fast.py:1 → 初始化 → 采样阶段 → reward → advantage → 训练阶段 → loss → backward → optimizer
```

每到一个关键节点，我都会：
1. 标注代码位置（文件名 + 行号）
2. 说明"这个函数接收什么、输出什么"
3. 解释背后的数学/算法动机
4. 配逻辑分支图

---

## 第 0 步：先理解总图

![Flow-GRPO 训练循环总图](/images/flowgrpo/flowgrpo_overview.svg)

**整个框架只有两个大阶段，反复交替：**

| 阶段 | 发生位置 | 是否记录梯度 | 产出 |
|------|---------|------------|------|
| **采样阶段** (SAMPLE) | `train_flux_fast.py:610-712` | `torch.no_grad()` ❌ | 图片 + log_prob_old + latent 轨迹 |
| **训练阶段** (TRAIN) | `train_flux_fast.py:803-917` | `requires_grad=True` ✅ | loss → backward → optimizer |

这是**在线（on-policy）强化学习的标志**：采样和训练用同一组模型参数。采样时你的行为（policy）是什么，训练时就优化那个行为。

---

## 第 1 步：入口 & 初始化 (`train_flux_fast.py:329-565`)

### 1.1 入口

```python
# train_flux_fast.py:329
def main(_):
    config = FLAGS.config          # 加载 config/base.py 中的默认参数
```

### 1.2 关键初始化

**FSDP 配置** (`train_flux_fast.py:350-358`):
```python
accelerator = Accelerator(
    gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
)
```
注意 `num_train_timesteps` = `config.sample.sde_window_size`（默认 5）——梯度累积步数乘以 window_size，因为每一步 DiT 前向都算一次梯度累积。

**LoRA 注入** (`train_flux_fast.py:414-441`):
```python
transformer_lora_config = LoraConfig(
    r=64, lora_alpha=128,           # LoRA 秩和缩放
    target_modules=["attn.to_k", "attn.to_q", ...]  # 所有 attention + FFN 层
)
pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
```
只有 LoRA 参数可训练，Flux 的原始权重全部冻结。

**Reward 函数工厂** (`train_flux_fast.py:554-558`):
```python
reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
```
根据 `config.reward_fn`（如 `"ocr"`）动态加载对应的 reward 计算器。

**EMA 包装器** (`train_flux_fast.py:446`):
```python
ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8)
```

---

## 第 2 步：采样阶段——生成图片 + 记录 log_prob_old

### 2.1 总体结构

```python
# train_flux_fast.py:610-712
#################### SAMPLING ####################
pipeline.transformer.eval()        # 切到 eval 模式（BN/ dropout 行为不同）
for i in range(config.sample.num_batches_per_epoch):
    prompts, prompt_metadata = next(train_iter)
    
    # 2.1.1 文本编码 (line 623-629)
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(...)
    
    # 2.1.2 采样 (line 643-659)
    with torch.no_grad():           # ← 整个采样不记录任何梯度！
        images, latents, image_ids, text_ids, log_probs, timesteps = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            num_inference_steps=config.sample.num_steps,     # 默认 10
            guidance_scale=config.sample.guidance_scale,
            sde_window_size=config.sample.sde_window_size,   # 默认 5
            sde_window_range=config.sample.sde_window_range,
            sde_type=config.sample.sde_type,
        )
    
    # 2.1.3 整理采样结果 (line 661-664)
    latents = torch.stack(latents, dim=1)       # (B, num_steps+1, 16, 96, 96)
    log_probs = torch.stack(log_probs, dim=1)   # (B, window_size)
    
    # 2.1.4 异步提交 reward 计算 (line 667)
    rewards = executor.submit(reward_fn, images, prompts, ...)
```

**关键理解：** `pipeline_with_logprob` 返回了 `log_probs`——这是**旧模型**（当前参数）下，生成这张图片的 SDE 轨迹中每一步的 log_prob。它们将作为 `π_θ_old` 的行为，用于后续的 PPO ratio 计算。

### 2.2 pipeline_with_logprob 内部 (`flux_pipeline_with_logprob_fast.py:23-213`)

```
输入: prompt_embeds, num_inference_steps=10, sde_window_size=5, noise_level=0.7
输出: images, all_latents, image_ids, text_ids, all_log_probs, all_timesteps
```

**SDE 窗口机制** (`flux_pipeline_with_logprob_fast.py:136-142`):

```python
if sde_window_size > 0:
    start = randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
    end = start + sde_window_size
    sde_window = (start, end)              # 例如 (2, 7)，表示只记录第 3-7 步的 log_prob
else:
    sde_window = (0, len(timesteps) - 1)   # 全部记录
```

**去噪循环** (`flux_pipeline_with_logprob_fast.py:158-202`):

```python
for i, t in enumerate(timesteps):                    # timesteps 共 10 个
    # 决定当前步的噪声水平
    if i < sde_window[0]:     cur_noise_level = 0     # 窗口前：确定性 ODE
    elif i == sde_window[0]:  cur_noise_level = noise_level  # 窗口起点：开始加噪声
    elif i < sde_window[1]:   cur_noise_level = noise_level  # 窗口内：SDE
    else:                     cur_noise_level = 0     # 窗口后：ODE
    
    # DiT 前向 (line 173-183)
    noise_pred = self.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        ...
    )[0]
    
    # SDE 步进 (line 185-192)
    latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
        self.scheduler, noise_pred.float(), t, latents.float(),
        noise_level=cur_noise_level,
        sde_type=sde_type,
    )
    
    # 仅在窗口内记录 (line 196-199)
    if i >= sde_window[0] and i < sde_window[1]:
        all_latents.append(latents)     # 存储 latent 轨迹（供训练阶段复用）
        all_log_probs.append(log_prob)  # 存储 log_prob_old
        all_timesteps.append(t)
```

### 2.3 SDE 一步的数学

`sde_step_with_logprob` (`sd3_sde_with_logprob.py:39-171`) 是理解整个框架的核心函数。它做了两件事：

**第 1 件事**：计算下一步的均值（模型预测方向）
```python
# sd3_sde_with_logprob.py:109 (sde_type='sde')
prev_sample_mean = sample * (1 + std_dev_t²/(2*sigma)*dt) 
                 + model_output * (1 + std_dev_t²*(1-sigma)/(2*sigma)) * dt
```

**第 2 件事**：从均值 + 噪声得到下一步 latent，并计算它的 log_prob
```python
# sd3_sde_with_logprob.py:111-119
if prev_sample is None:     # 采样阶段
    variance_noise = randn_tensor(...)
    prev_sample = prev_sample_mean + std_dev_t * √(-dt) * variance_noise

# sd3_sde_with_logprob.py:121-133
# 高斯 log_prob 公式
log_prob = -((prev_sample.detach() - prev_sample_mean)²) / (2 * (std_dev_t * √(-dt))²)
           - log(std_dev_t * √(-dt))
           - log(√(2π))

# sd3_sde_with_logprob.py:167: 除 batch 维外全部 mean 掉
log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # (B,) 每个样本一个标量
```

**注意这里**：`prev_sample.detach()`——在采样阶段 prev_sample 是刚随机采出来的，detach() 切断梯度。但如果你看训练阶段（稍后），情况会不同！

---

## 第 3 步：Reward 阶段 (`train_flux_fast.py:689-701`)

异步 reward 计算完成后，取出结果：

```python
# train_flux_fast.py:696-701
rewards, reward_metadata = sample["rewards"].result()
sample["rewards"] = {
    key: torch.as_tensor(value, device=accelerator.device).float()
    for key, value in rewards.items()
}
```

Reward 函数（`flow_grpo/rewards.py`）返回一个 dict：
```python
{
    "avg": [0.45, 0.78, ...],       # 平均 reward
    "strict_accuracy": [0, 1, ...],  # 严格准确率
    ...
}
```

每条图片对应一个 reward 标量。

---

## 第 4 步：跨 GPU 同步 + Advantage 计算 (`train_flux_fast.py:747-796`)

### 4.1 all_gather

```python
# train_flux_fast.py:747
gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
```
所有 GPU 的 reward 被收集到一起，形成 `[total_GPU × batch_per_GPU]` 长度的数组。

### 4.2 PerPromptStatTracker (`stat_tracking.py:50-116`)

```python
# train_flux_fast.py:766
advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
```

**核心逻辑** (`stat_tracking.py:64-92`):

```python
prompts = np.array(prompts)       # [样本1所属prompt, 样本2所属prompt, ...]
rewards = np.array(rewards)       # [样本1的reward, 样本2的reward, ...]

for prompt in unique:
    # 收集该 prompt 的所有历史 reward
    self.stats[prompt].extend(prompt_rewards)    # 追加到历史 buffer
```

对于 GRPO 模式（`type='grpo'`，`stat_tracking.py:82-92`）：

```python
for prompt in unique:
    mean = np.mean(self.stats[prompt], axis=0)      # μ: 该 prompt 的历史均值
    std = np.std(self.stats[prompt], axis=0) + 1e-4  # σ: 该 prompt 的历史标准差
    advantages[prompts == prompt] = (prompt_rewards - mean) / std
```

**公式：** $A_i = \frac{r_i - \mu_{history}}{\sigma_{history}}$

**设计动机：**
- 同一个 prompt 的不同图片之间做比较（而不是跨 prompt）
- 等于问：**针对这个 prompt，这张图比平均水平好多少？**
- 好（A>0）→ 提高这条路径概率；差（A<0）→ 降低

### 4.3 为什么不直接用 group 内统计？

理论上 GRPO 是在一个 group（同一组采样）内做归一化。但在分布式训练中，同一 prompt 的不同采样可能分布在多张 GPU 上。PerPromptStatTracker 用**跨越多个训练步的历史 reward**来近似 group 统计，既解决了分布式同步问题，又让估计更稳定。

### 4.4 Advantage 裁剪

```python
# train_flux_fast.py:842-846
advantages = torch.clamp(
    sample["advantages"][:, j],
    -config.train.adv_clip_max,     # 默认 2.0
    config.train.adv_clip_max,
)
```
防止个别 outlier advantage 主导训练。

---

## 第 5 步：训练阶段——计算图如何构造？梯度如何流动？(核心!)

这是整个文章最重要的部分。训练阶段的代码在 `train_flux_fast.py:803-917`。

### 5.1 总体结构

```python
# train_flux_fast.py:803-917
#################### TRAINING ####################
for inner_epoch in range(config.train.num_inner_epochs):
    pipeline.transformer.train()        # 切到 train 模式
    for i, sample in enumerate(samples_batched):
        for j in train_timesteps:       # j = 0,1,2,3,4  (window_size 步)
            with accelerator.accumulate(transformer):
                # 5.2 计算 log_prob_new (关键!)
                prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                    transformer, pipeline, sample, j, config
                )
                # 5.3 计算 loss
                ...
                # 5.4 反向传播
                accelerator.backward(loss)
                optimizer.step()
```

### 5.2 compute_log_prob：计算图在这里构造

```python
# train_flux_fast.py:186-218
def compute_log_prob(transformer, pipeline, sample, j, config):
    # 取出第 j 步的 latent（采样阶段存储的）
    packed_noisy_model_input = sample["latents"][:, j]        # (B, 16, 96, 96)
    
    # ===== DiT 前向传播（这是计算图的根节点）=====
    model_pred = transformer(                                  # ← 这是 requires_grad=True 的
        hidden_states=packed_noisy_model_input,
        timestep=sample["timesteps"][:, j] / 1000,
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        ...
    )[0]                                                       # shape (B, 16*96*96, 1)
    
    # ===== SDE 步进 + log_prob 计算 =====
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),      # ← 传入旧轨迹的 next_latent!
        noise_level=config.sample.noise_level,
        sde_type=config.sample.sde_type,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

**和采样阶段的关键区别：**

| | 采样阶段 | 训练阶段 |
|--|---------|---------|
| `requires_grad` | ❌ 全部 no_grad | ✅ 开启 |
| `prev_sample` 参数 | `None`（自己随机采） | 传入旧轨迹的 `next_latents` |
| `prev_sample.detach()` | 有效（反正不记梯度） | 有效（防止梯度流到 prev_sample） |

### 5.3 训练阶段的计算图——用图形理解

训练阶段执行 `compute_log_prob` 时，PyTorch 自动构造了如下计算图：

```
sample["latents"][:, j] (detached, from sampling phase)
                         │
                         ▼
              ┌──────────────────┐
              │   transformer    │  ← LoRA 参数是 leaf node（requires_grad=True）
              │  (DiT forward)   │
              └────────┬─────────┘
                       │ model_pred (shape: B×C×H×W)
                       ▼
              ┌──────────────────┐
              │ sde_step_with    │  ← 数学公式：mean = f(sample, model_pred, dt)
              │ _logprob         │     log_prob = -||prev_sample - mean||² / (2σ²) - log(√(2πσ²))
              └────────┬─────────┘
                       │ log_prob (shape: B,) —— 标量 per sample
                       ▼
              ┌──────────────────┐
              │ PPO Loss         │  ← loss = -min(ratio×A, clip(ratio)×A)
              │                   │     ratio = exp(log_prob - log_prob_old)
              └────────┬─────────┘
                       │ loss (shape: scalar)
                       ▼
              ┌──────────────────┐
              │ loss.backward()  │  ← 链式法则：∂loss/∂θ = ∂loss/∂log_prob × ∂log_prob/∂model_pred × ∂model_pred/∂θ
              └──────────────────┘
```

**梯度流动路径（链式法则）：**

```
∂loss/∂θ = 
    ∂loss/∂log_prob          (从 PPO loss 到 log_prob)
  × ∂log_prob/∂model_pred   (从 log_prob 公式到 model_pred——高斯 log_prob 对 mean 求导)
  × ∂model_pred/∂θ           (从 DiT 前向到 LoRA 参数——autograd 自动完成)
```

展开第二项：

```python
# sd3_sde_with_logprob.py:109,121-133
log_prob = -((prev_sample.detach() - prev_sample_mean)²) / (2 * var) - log(√(var * 2π))

# 其中 prev_sample_mean = g(model_pred, sample, dt, noise_level)
#     var = (std_dev_t * √(-dt))²

# ∂log_prob/∂model_pred = -(prev_sample - prev_sample_mean) / var * ∂prev_sample_mean/∂model_pred
```

因为 `prev_sample.detach()` 了，梯度不会流到 `prev_sample`，只会通过 `prev_sample_mean` 流到 `model_pred`。

**重要：计算图只包含当前第 j 步！**
- `sample["latents"][:, j]` 是 detached 的（采样阶段产生，不参与计算图）
- 所以梯度只从第 j 步的 model_pred 反向传播到 LoRA 参数
- **为什么这可行？** SDE 的马尔可夫性质：第 j 步的 latent 只由第 j-1 步决定。虽然每一步的梯度只包含一步的信息，但经过 window_size=5 步的累积（循环 5 次 `compute_log_prob` + `backward`），梯度实际上包含了从窗口起点到终点的完整信息。

### 5.4 Loss 在训练循环中的位置

```python
# train_flux_fast.py:825-901
train_timesteps = [step_index for step_index in range(num_train_timesteps)]  # [0,1,2,3,4]

for j in train_timesteps:                # 遍历窗口内的每一步
    with accelerator.accumulate(transformer):
        # 5.4.1 前向 → 得到 log_prob_new (line 835)
        prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(...)
        
        # 5.4.2 PPO ratio (line 847)
        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
        
        # 5.4.3 无裁剪和有裁剪的 loss (line 849-854)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
        )
        
        # 5.4.4 PPO loss: 取最大值（最悲观）（line 855）
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        # 5.4.5 KL 惩罚（可选）（line 856-861）
        if config.train.beta > 0:
            # 计算 reference model（禁用 LoRA adapter）的预测均值
            with torch.no_grad():
                _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref)²).mean(dim=(1,2)) / (2 * std_dev_t²)
            loss = policy_loss + config.train.beta * kl_loss
        else:
            loss = policy_loss
        
        # 5.4.6 backward (line 895)
        accelerator.backward(loss)      # ← 梯度累积 + 反向传播
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 第 6 步：Loss 设计详解

### 6.1 PPO Loss 数学形式

$$ L^{PPO} = \mathbb{E}\left[ \max\left( -\frac{\pi_\theta}{\pi_{\theta_{old}}} \cdot A,\; -\text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right) \cdot A \right) \right] $$

在代码中：

| 符号 | 代码变量 | 位置 |
|------|---------|------|
| $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ | `ratio = torch.exp(log_prob - sample['log_probs'][:, j])` | line 847 |
| $A$ | `advantages`（已裁剪到 [-adv_clip_max, adv_clip_max]） | line 842-846 |
| $\epsilon$ | `config.train.clip_range`（默认 0.2） | line 852 |
| $-\text{ratio} \cdot A$ | `unclipped_loss = -advantages * ratio` | line 849 |
| $-\text{clip}(\text{ratio}) \cdot A$ | `clipped_loss = -advantages * torch.clamp(...)` | line 850-854 |
| $\max$ | `policy_loss = torch.mean(torch.maximum(...))` | line 855 |

**为什么这样设计？**

- 当 `A > 0`（这张图比平均好）：我们希望提高它的概率，但不想提高太猛
  - ratio > 1.2 → clip 到 1.2 → clipped_loss 更大 → loss 取 max → 惩罚过度更新
- 当 `A < 0`（这张图比平均差）：我们希望降低它的概率，但同样不想降太猛
  - ratio < 0.8 → clip 到 0.8 → clipped_loss 更大 → loss 取 max → 惩罚过度更新

### 6.2 KL 惩罚项

当 `config.train.beta > 0` 时，额外加一个 KL 惩罚：

```python
# train_flux_fast.py:837-838
with torch.no_grad():
    with transformer.module.disable_adapter():   # 禁用 LoRA → reference model
        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)

# train_flux_fast.py:857-858
kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
kl_loss = torch.mean(kl_loss)
loss = policy_loss + config.train.beta * kl_loss
```

**设计动机：**
- 通过 `disable_adapter()` 暂时禁用 LoRA 权重，得到 base model（reference）的预测均值
- 当前模型的预测均值 `prev_sample_mean` 不应偏离 reference 太远
- KL 惩罚项防止 PPO 过度优化导致模型遗忘原始能力（catastrophic forgetting）

### 6.3 日志指标

```python
# train_flux_fast.py:863-888
info["approx_kl"].append(0.5 * ((log_prob - sample["log_probs"][:, j]) ** 2).mean())
    # 近似 KL 散度：log_prob_new 与 log_prob_old 的差异平方均值
info["clipfrac"].append((|ratio - 1.0| > clip_range).float().mean())
    # 被 clip 的样本比例
```

---

## 第 7 步：一次完整的训练迭代——按时间线串联

现在我们把所有步骤串联起来，看一次完整的迭代：

```
Epoch 42, 第 3 个 batch:
│
├── [采样阶段] pipeline.transformer.eval()
│   ├── 读 prompts = ["a red cat", "blue dog", ...]  (每 GPU batch_size=1)
│   ├── T5 编码 → prompt_embeds
│   ├── pipeline_with_logprob (10 步 SDE, window_size=5)
│   │   ├── 第 0-4 步: no_grad, cur_noise_level=0      (ODE, 不记录)
│   │   ├── 第 5 步: cur_noise_level=0.7, 记录 latent  (SDE start)
│   │   ├── 第 6-8 步: cur_noise_level=0.7, 记录 latent (SDE within window)
│   │   ├── 第 9 步: cur_noise_level=0                 (ODE, 不记录)
│   │   └── 返回 images (B,3,512,512), log_probs (B,5)
│   ├── executor.submit(reward_fn, images, prompts)    (异步)
│   └── 存储 latents, timesteps, prompt_embeds 供训练阶段复用
│
├── [等待 reward] 从 Future 中取出 reward 值
│   └── rewards = {"avg": [0.12, 0.45, ...], "strict_accuracy": [0, 1, ...]}
│
├── [跨 GPU 同步] all_gather 所有 GPU 的 reward 和 prompt
│   └── 32 GPU × 1 batch = 32 个样本
│
├── [Advantage 计算] PerPromptStatTracker.update()
│   └── 对每个 prompt: A_i = (r_i - μ_history) / σ_history
│       └── A = [-1.37, -0.24, 1.16, ...]
│
├── [训练阶段] pipeline.transformer.train()
│   ├── inner_epoch=0:
│   │   ├── j=0 (SDE 窗口第 1 步)
│   │   │   ├── transformer(latents[:,0], ...) → model_pred    ← 前向（梯度开启）
│   │   │   ├── sde_step_with_logprob(..., prev_sample=next_latents[:,0]) → log_prob
│   │   │   ├── ratio = exp(log_prob - log_probs_old[:,0])
│   │   │   ├── policy_loss = max(-A·ratio, -A·clip(ratio))
│   │   │   ├── (可选) KL_loss = ||mean_new - mean_ref||² / (2·var)
│   │   │   ├── loss = policy_loss + β·KL_loss
│   │   │   ├── accelerator.backward(loss)                     ← 梯度累加到 LoRA 参数
│   │   │   └── optimizer.step()                               ← LoRA 参数更新
│   │   │
│   │   ├── j=1 (SDE 窗口第 2 步) → 重复...
│   │   ├── j=2 (SDE 窗口第 3 步)
│   │   ├── j=3 (SDE 窗口第 4 步)
│   │   └── j=4 (SDE 窗口第 5 步)
│   │
│   └── inner_epoch=1:
│       └── 用同一批样本再训练一次 (num_inner_epochs=1 时跳过)
│
└── epoch + 1
```

---

## 第 8 步：推理阶段（和训练完全不同！）

推理 (`train_flux_fast.py:220-311`, `eval` 函数) 和训练有本质区别：

```python
# train_flux_fast.py:240-254 (eval 函数)
with torch.no_grad():                               # 不记录任何梯度
    images, _, _, _, _, _ = pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds,
        num_inference_steps=config.sample.eval_num_steps,     # 一般和训练一致
        guidance_scale=config.sample.eval_guidance_scale,
        sde_window_size=0,                                     # ← 推理时 window=0！
        noise_level=0,                                          # ← 推理时 noise_level=0！
        sde_type=config.sample.sde_type,
    )
```

**推理 vs 训练的关键差异：**

| | 训练采样阶段 | 推理阶段 |
|--|------------|---------|
| `sde_window_size` | 5 | **0**（全 ODE） |
| `noise_level` | 0.7 | **0**（不加噪声） |
| `requires_grad` | ❌ | ❌ |
| 返回 `log_probs` | ✅ 需要 | **不返回**（`_` 丢弃） |
| 返回 `latents` | ✅ 供训练复用 | ❌ |

**为什么推理不用 SDE？**
- 推理只关心图片质量，不需要计算 log_prob
- ODE（noise_level=0）生成的图片更清晰、质量更高
- SDE 在训练时是为了引入随机性 → 多样性 → group 内 reward 出现差异 → advantage 信号

---

## 第 9 步：完整代码架构

![Flow-GRPO 代码架构](/images/flowgrpo/flowgrpo_code_arch.svg)

### 文件命中表（每个文件的精确业务）

| 文件 | 行数 | 核心函数/类 | 精确职责 |
|------|------|------------|---------|
| `scripts/train_flux_fast.py` | 925 | `main()`, `compute_log_prob()`, `eval()` | 主循环编排、采样/训练两阶段切换、loss 计算与 backward |
| `config/base.py` | - | `GRPOConfig` | 所有超参数的默认值（lr, group_size, window_size...） |
| `config/grpo.py` | - | `ocr_experiment`, `gen_eval_experiment`, ... | 实验预设配方 |
| `flow_grpo/diffusers_patch/flux_pipeline_with_logprob_fast.py` | 213 | `pipeline_with_logprob()` | SDE 采样循环、窗口控制、latent 轨迹记录 |
| `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` | 171 | `sde_step_with_logprob()` | **单步 SDE 前向 + log_prob 解析计算**（梯度流的数学核心） |
| `flow_grpo/stat_tracking.py` | 139 | `PerPromptStatTracker.update()` | reward → advantage 的组内归一化（GRPO 核心） |
| `flow_grpo/rewards.py` | 567 | `multi_score()`, `ocr_reward_fn`, ... | 各种 reward 函数的实现 |
| `flow_grpo/ema.py` | - | `EMAModuleWrapper` | 指数移动平均（保存最优参数快照） |
| `flow_grpo/fsdp_utils.py` | - | FSDP 配置 | 12B 模型的分片策略 |

---

## 第 10 步：核心公式一览

### 10.1 GRPO Advantage

$$ A_i = \frac{r_i - \mu_{prompt}}{\sigma_{prompt}} $$

`stat_tracking.py:76-92`

### 10.2 SDE Step (sde_type='sde')

$$ \text{mean} = x_t \cdot \left(1 + \frac{\sigma_{noise}^2}{2\sigma}\Delta t\right) + v_\theta \cdot \left(1 + \frac{\sigma_{noise}^2(1-\sigma)}{2\sigma}\right)\Delta t $$

$$ x_{t+1} \sim \mathcal{N}(\text{mean}, \sigma_{noise}^2 \cdot (-\Delta t) \cdot I) $$

`sd3_sde_with_logprob.py:106-109`

### 10.3 Log Prob

$$ \log p(x_{t+1}|x_t) = -\frac{\|x_{t+1} - \text{mean}\|^2}{2\sigma_{noise}^2(-\Delta t)} - \log(\sigma_{noise}\sqrt{-\Delta t}) - \log\sqrt{2\pi} $$

`sd3_sde_with_logprob.py:121-133`

### 10.4 PPO Ratio

$$ r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{old}}) $$

`train_flux_fast.py:847`

### 10.5 PPO Policy Loss

$$ L^{PG} = \mathbb{E}\left[\max\left(-r_t(\theta)A_t,\; -\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right] $$

`train_flux_fast.py:849-855`

### 10.6 KL Penalty

$$ L^{KL} = \frac{\|\mu_\theta - \mu_{ref}\|^2}{2\sigma_t^2} $$

`train_flux_fast.py:857-858`

### 10.7 Total Loss

$$ L = L^{PG} + \beta L^{KL} $$

`train_flux_fast.py:859`

---

## 附：关键流程图

![GRPO Advantage 计算](/images/flowgrpo/flowgrpo_grpo_advantage.svg)

![Flow Matching 去噪过程](/images/flowgrpo/flowgrpo_flow_matching.svg)

![Flow-GRPO-Fast SDE 窗口](/images/flowgrpo/flowgrpo_fast_sde.svg)

---

## 参考

- 代码库：`/root/workspace_qyl/Flow-GRPO/flow_grpo/`
- GRPO 论文：DeepSeekMath (arXiv:2402.03300)
- Flow Matching：Flow Matching for Generative Modeling (arXiv:2210.02747)
- PPO：Proximal Policy Optimization Algorithms (arXiv:1707.06347)
- DDPO：Denoising Diffusion Policy Optimization (arXiv:2307.09439)
