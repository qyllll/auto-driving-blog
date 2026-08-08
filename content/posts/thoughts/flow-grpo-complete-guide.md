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

## 入门篇：Flow-GRPO 到底是什么？（小白版）

> 如果你第一次听说「扩散模型」「Flow Matching」「GRPO」这些词，或者听到就头大——**先读这一篇**。
> 这一篇不讲任何代码，只帮你建立**直觉**。等脑子里有画面了，再进入下面的「先回答」和代码逐行拆解。

### 0.1 一句话版

> **Flow-GRPO 是一种「训练」图像生成模型的算法：让一个已经会画图的模型（如 FLUX、SD3.5-M），通过「自己画图 → 自己打分 → 自己改进」的循环，越画越好——「好」的标准由你定（比如文字渲染得对不对、人喜不喜欢、组合合不合理）。**

论文：*Flow-GRPO: Training Flow Matching Models via Online RL*（arXiv:2505.05470，腾讯 ARC Lab）。

它和传统做法的本质区别：

| 传统做法（SFT 模仿学习） | Flow-GRPO（在线强化学习） |
|---|---|
| 照着「标准答案」模仿 | 没有标准答案，只有「好不好」的打分 |
| 训练集给什么就学什么 | 自己生成、自己试错、自己改进 |
| 上限 = 训练数据的质量 | 上限 = 你定义的奖励函数 |
| 一次性训完就结束 | **边采样边训练**（在线 on-policy） |

一句话概括动机：**只让模型「会画」还不够，还要让它「画得好」——用打分信号告诉它该往哪个方向改进。**

### 0.2 先看思维导图

![Flow-GRPO 思维导图](/images/flowgrpo/flowgrpo_mindmap.svg)

图里有 5 个分支：**是什么、为什么、怎么生成（Flow Matching）、怎么改进（GRPO）、宏观流程**。先让它们在脑子里占个位置，下面逐个展开。

### 0.3 两个必须搞懂的核心概念

把名字拆开就是它的两个引擎：

| 名字的一部分 | 对应概念 | 负责的事 |
|---|---|---|
| **Flow** | Flow Matching | 「怎么生成图片」 |
| **GRPO** | 一种强化学习算法 | 「怎么改进模型」 |

#### 概念一：从「噪声」到「图片」——扩散模型 & Flow Matching

今天几乎所有图像生成模型（Stable Diffusion、FLUX、Midjourney 背后的模型……）都长这样：

> 生成图片 = 从一个**纯噪声**出发，一步步把它「整形」成一张清晰的图片。

**扩散模型（Diffusion）**的做法是"一点点擦马赛克"：每一步去掉一点噪声，走 20~50 步。每一步用一个「噪声预测器」决定去掉多少。

**Flow Matching**是扩散的"直连版"：不弯弯绕绕，而是在「纯噪声」和「图片」之间**拉一条直线**，训练模型学习这条线上的**速度场（velocity field）**——告诉它"在任意一个中间点，下一步该往图片方向走多少"。步数可多可少（大步走也行），所以更快、更适合反复采样的强化学习。

![扩散模型 vs Flow Matching](/images/flowgrpo/flowgrpo_fm_concept.svg)

> 小知识：FLUX 用的就是 Flow Matching 家族（rectified-flow）。本文训练的就是它。

#### 概念二：GRPO——「组内竞争」的强化学习算法

GRPO 全称 *Group Relative Policy Optimization*（组相对策略优化）。它解决一个更朴素的问题：

> 图片没有"标准答案"，只有"打分数"。分数是**相对**才有意义的——**同一个 prompt 生成的一组图里，谁比平均好、谁比平均差？**

过程长这样：

1. 同一个 prompt（如「一只戴红色帽子的狗」）让模型生成 **一组** 图片（默认 24 张）；
2. 用奖励模型逐张打分（OCR 看字写没写对、PickScore 看人喜不喜欢、GenEval 看组合逻辑）；
3. **组内归一化**：算出这组的均值 μ、标准差 σ，然后

$$advantage_i = \frac{r_i - \mu}{\sigma}$$

比平均好 → advantage > 0 → **提高**这类走法的概率；比平均差 → advantage < 0 → **降低**。

4. 用 PPO 风格的损失做参数更新，加 clip 防止一步更新太猛。

![GRPO 组内竞争](/images/flowgrpo/flowgrpo_grpo_concept.svg)

> 为什么"相对比较"这么重要？因为奖励模型的分数尺度、分布会漂移（今天 OCR 打 0.8，明天可能整体打 0.9）。只看组内相对好坏，就能**甩开这些波动**，只关注"这组里哪个更好"这个稳定信号。

### 0.4 概念关系图：五个名词怎么拼起来

把上面串起来：

- **蓝色（生成器）**：扩散模型 → Flow Matching → 具体模型 FLUX。负责"怎么生成"；
- **橙色（裁判）**：奖励模型。负责"什么算好"；
- **紫色（学习算法）**：GRPO → advantage → PPO Loss。负责"怎么改"。

三者合起来就是一个完整的 **Flow-GRPO** 训练循环。

![概念关系图](/images/flowgrpo/flowgrpo_concept_rel.svg)

### 0.5 和自动驾驶 / 世界模型有什么关系？

你在这个博客里读过的很多 VLA（视觉语言行动模型）工作，底层就是这套组合拳的变体：

- **DiffusionDrive / 扩散策略（Diffusion Policy）**：用扩散/Flow Matching 做动作生成——把"动作轨迹"当作"要生成的图片"，把噪声一步步整形成一条可执行的轨迹；
- **AlphaDrive-GRPO 等**：把 GRPO 直接用在驾驶策略上——同一段场景生成多条动作轨迹，让**奖励模型（安全、舒适、任务达成）**打分，组内比较后改进策略。

所以这篇讲的 Flow-GRPO 不是"只跟画图有关"的孤立技巧：**「生成模型 + 奖励信号 + 组内竞争」这套配方，就是 2025 年驾驶/具身领域最主流的强化学习范式之一。** 读懂了它，等于读懂了那一批工作的共同骨架。

### 0.6 接下来怎么读？

后面内容沿着**一条执行主线**走：

```
初始化 → 采样（生成图片 + 记 log_prob）→ 打 reward → 组内算 advantage → 训练（构造计算图 + PPO loss）→ backward → 回到采样
```

建议带着入门篇建立的三个"心智锚点"去读：

1. **采样阶段 = Flow Matching 的 SDE 生成**（对应入门篇概念一）；
2. **advantage = 组内归一化**（对应概念二第 3 步）；
3. **训练阶段 = 让高分走法概率上升**（对应概念二第 4 步）。

---

## 先回答：Flow-GRPO 是什么？train_flux_fast.py 是什么？

### 这是一份可运行的代码仓库

Flow-GRPO **不是一篇只能读的论文，而是一份完整的开源代码仓库**，代码就在 `flow_grpo/` 目录下：

![flow_grpo 代码仓库结构](/images/flowgrpo/flowgrpo_repo_structure.svg)

### train_flux_fast.py 和 train_flux.py 有什么区别？先厘清三个"步数"

读代码前先分清三个容易混淆的步数概念，否则后面全是坑：

| 概念 | 含义 | 取值 |
|------|------|------|
| `num_steps` | **训练时完整去噪的步数**（每次采样走几步生成一张图） | SD3 系默认 **10**、FLUX 系默认 **6** |
| `eval_num_steps` | **推理/评测时的步数**（只用于 `eval` 函数） | 通常 40 或 50 |
| `sde_window_size` | **fast 版参与梯度计算的窗口宽度**（分位数） | 各配置不同：3、2、4 都有，**没有统一默认值** |

> ⚠️ **"40 步"是 `eval_num_steps`（评测步数），不是训练步数。** 下面的 fast vs 基础版对比里，基础版"参与梯度计算的步数 = 40"指的是**评测/长时间步**场景，不要和训练采样（10 步）搞混。

很多文件名带 `_fast` 后缀（`train_flux_fast.py`、`flux_pipeline_with_logprob_fast.py`）。它们和基础版**训练逻辑完全一样，只有一处关键区别——采样时记录梯度的 timestep 数量**：

| | 基础版 `train_flux.py` | fast 版 `train_flux_fast.py` |
|--|----------------------|----------------------|
| 导入的采样器 | `flux_pipeline_with_logprob.py` | `flux_pipeline_with_logprob_fast.py` |
| 参与梯度计算的步数 | **全部** `num_steps`（训练采样步） | **只有** `sde_window_size` 步（随机窗口） |
| 训练步数 `num_train_timesteps` | `int(num_steps × timestep_fraction)` | `config.sample.sde_window_size`（`main()` 里第 340 行） |
| 窗口外如何采样 | 每一步都是 SDE | 窗口外用**确定性 ODE**（`noise_level=0`，不记梯度） |
| 计算图大小 | 全部步展开 → **显存爆炸** | 只展开 window_size 步 → **显存小、速度快** |
| log_prob 记录 | 每一步都记录 | 只在窗口内记录 |
| 采样返回 | `image, latents, image_ids, text_ids, log_probs` | 额外多返回 `timesteps` |

一个具体的数值例子（`geneval_flux_fast` 配置）：`num_steps=6`、`sde_window_size=3`、`sde_window_range=(0, 3)`。每次采样走 **6 步**去噪生成图，但只有随机选的 **3 步**（例如第 2、3、4 步）进计算图参与梯度。

核心代码对比（`diffusers_patch/` 下的两个文件）：

```python
# flux_pipeline_with_logprob.py（基础版）：每个 timestep 都记录
for i, t in enumerate(timesteps):               # num_steps 步全走 SDE
    latents, log_prob, _, _ = sde_step_with_logprob(
        scheduler, noise, t, latents, noise_level=noise_level, ...)
    all_latents.append(latents)                 # 每一步都进计算图
    all_log_probs.append(log_prob)
```

```python
# flux_pipeline_with_logprob_fast.py（fast 版）：只在窗口内记录
sde_window = (start, start + sde_window_size)   # 随机选 window_size 步窗口
for i, t in enumerate(timesteps):
    cur_noise_level = noise_level if (窗口起点 <= i < 窗口终点) else 0
    latents, log_prob, _, _ = sde_step_with_logprob(..., noise_level=cur_noise_level)
    if 窗口起点 <= i < 窗口终点:                  # 只有窗口内的步
        all_latents.append(latents)             # 进计算图
        all_log_probs.append(log_prob)
```

**为什么这样省显存？** 因为训练时 `loss.backward()` 要沿着计算图回传，计算图里每多一个 timestep 就要多存一份 DiT 的中间激活值（Flux 有 12B 参数，一份激活就有几十 GB）。基础版把全部步展开，直接 OOM；fast 版只展开 window_size 步，显存可控。

**为什么只用几步也够用？** 因为 SDE 是马尔可夫的：第 j 步的 latent 只由第 j-1 步决定。虽然每次 backward 只回传窗口内的梯度，但模型参数是共享的——只要每个 epoch 随机窗口滑过不同位置（`sde_window_range` 控制起点范围），长期看每个 timestep 都会被训练到，是一种"逐步轮流覆盖"的均匀采样。

> 一句话总结：**fast 版 = 基础版 + SDE 窗口，牺牲一点点梯度完整性，换来 10~20 倍显存节省。** 代码里 `_fast` 的后缀就是这个意思。

### 为什么从 train_flux_fast.py 开始？

**`train_flux_fast.py` 是整个框架的入口（main 函数所在），相当于一辆车的发动机舱。** 其它文件都是"零件"：

- `config/*.py` 是"仪表盘"——设定怎么跑（超参数）
- `flow_grpo/rewards.py` 是"油门刹车"——告诉模型好不好
- `flow_grpo/stat_tracking.py` 是"大脑"——把 reward 变成学习信号 advantage
- `diffusers_patch/*.py` 是"变速箱"——模型内部怎么一步步生成图片

而 `train_flux_fast.py` 负责把所有这些零件组装起来，按正确的顺序调用。**读懂了它，就抓住了整个框架的主线**；其它文件都是被它调用、服务于它的。

具体来说，这份 Python 文件（925 行）只干一件事：**循环执行「采样 → 打 reward → 算 advantage → 算 loss → 反向传播」这五个动作，直到模型变好。** 本文接下来就顺着这个循环，一行一行拆开看。

## 本文的讲解方式

**不从概念讲起，而是从代码的入口出发，沿着执行路径一步一步深入。**

我们追踪一辆"数据快车"：

```
train_flux_fast.py → 初始化 → 采样阶段 → reward → advantage → 训练阶段 → loss → backward → optimizer → 回到采样
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
注意 `num_train_timesteps` = `config.sample.sde_window_size`（`train_flux_fast.py:339-342`，值因配置而异，如 3 / 2 / 4）——梯度累积步数乘以 window_size，因为**窗口内的每一步 DiT 前向都算一次梯度累积**，一个训练微批次（iter）内部每个 timestep 都要累积一次梯度、最后合起来做一次 optimizer 更新。

**LoRA 注入** (`train_flux_fast.py:414-441`):
```python
transformer_lora_config = LoraConfig(
    r=64, lora_alpha=128,           # LoRA 秩和缩放
    target_modules=["attn.to_k", "attn.to_q", ...]  # 所有 attention + FFN 层
)
pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
```
只有 LoRA 参数可训练，Flux 的原始权重全部冻结。

**那到底在更新什么？（重要）**
`train_flux_fast.py:383` 有一个前提：
```python
pipeline.transformer.requires_grad_(not config.use_lora)   # use_lora=True → 全冻结
```
`use_lora=True` 时整个 transformer 的原始权重 `requires_grad=False`，optimizer（第 466 行）只接收 `transformer_trainable_parameters`（第 444 行过滤出的 `requires_grad` 参数）＝**只有 LoRA 的 A/B 矩阵**。所以 `backward()` 的梯度一路沿速度场网络回传，最终只落在 LoRA 参数上——**Flux 的 12B 原始权重整个训练过程一个数都不变**。

那"生成能力"是怎么被改的？`W = W0 + (α/r)·B·A`：LoRA 的 A/B 微调 → 每层 attention/FFN 输出跟着变 → 同一个 DiT 网络输出的**速度场 v_θ 方向被悄悄改了** → 采样出的图片/轨迹就更符合 reward。**你可以把整个 Flow-GRPO 理解成：通过只动 LoRA，把"去噪轨迹上的每一步速度"往 reward 期望的方向推。**

**EMA 包装器**（`flow_grpo/ema.py`，`train_flux_fast.py:446-446`）：
```python
ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, ...)
```
EMA = 指数滑动平均（Exponential Moving Average）：给 LoRA 参数额外维护一套"影子参数"，按 `shadow += (1-decay)·(param - shadow)` 缓慢追随实时参数。它的角色：
- **训练中**：optimizer 更新的是实时参数，EMA 影子每 8 步平均一次，decay=0.9 → 大约把最近 20×8=160 步平均掉（`ema.step` `train_flux_fast.py:917`）；
- **评测/存盘时**：`copy_ema_to`（eval 第 222 行 / save 第 324 行）把平滑后的影子参数**临时拷回模型**再跑评估/存权重，因为训练中的参数抖，平均后更稳、reward 曲线更光滑；跑完 `copy_temp_to`（第 311 行）还原回实时参数。

> 一句话：**实时参数负责"冲"，EMA 影子负责"记"——评测和存盘用的是记下来的稳定版本。**

**对应公式（LoRA 注入）：**

$$ h = W_0 x + \frac{\alpha}{r}\,B A\, x $$

其中 $W_0\in\mathbb{R}^{d\times d}$ 是冻结的原权重，$A\in\mathbb{R}^{r\times d}$、$B\in\mathbb{R}^{d\times r}$ 是两个可训练的低秩矩阵（$r=64\ll d$），缩放系数 $\frac{\alpha}{r}=\frac{128}{64}=2$。所以"全量更新 $d^2$ 个参数"被压缩成"只更新 $2dr$ 个参数"，这就是 `r`、`alpha` 这两个超参数对应的数学。

**Reward 函数工厂** (`train_flux_fast.py:554-558`):
```python
reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
```
根据 `config.reward_fn`（如 `"ocr"`）动态加载对应的 reward 计算器。

**EMA 包装器** (`train_flux_fast.py:446`):
```python
ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8)
```

**对应公式（EMA 指数移动平均）：**

$$ \theta_{\mathrm{ema}} \leftarrow \text{decay}\cdot\theta_{\mathrm{ema}} + (1-\text{decay})\cdot\theta $$

即每隔 8 步把当前参数 $\theta$ 以 $0.1$ 的比例"混入"历史平均 $\theta_{\mathrm{ema}}$（`decay=0.9`），用滑动平均保留一个更稳的参数快照。

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
            num_inference_steps=config.sample.num_steps,     # 训练采样步数（SD3 默认 10 / FLUX 默认 6）
            guidance_scale=config.sample.guidance_scale,
            sde_window_size=config.sample.sde_window_size,   # 窗口宽，各配置不同（如 3/2/4）
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

**逐步 log_prob vs 整条轨迹的总 log_prob**（这里先厘清，后文 training 会用到）：

假设 `num_steps=10`（然后 fast 版窗口挑了其中 `window_size` 步记录）。每步返回一个 log_prob，对应去噪轨迹上那一步的转移概率：
```
x_0 --log p(x_1|x_0)--> x_1 --log p(x_2|x_1)--> ... --log p(x_10|x_9)--> x_10(图)
```
整条轨迹的总对数概率（马尔可夫连乘取 log）是**各步之和**：
$$\log\pi_\theta(\tau) = \sum_k \log p_\theta(x_{k+1}\mid x_k)$$

但 **Flow-GRPO 代码里并不显式打这个总和**——它把 log_prob 保持成**逐 timestep 的向量**（shape `(B, window_size)`），因为 PPO 的 ratio 也是**逐 step 算、逐 step 用同一个 advantage 加权**（见 5.4）。数学上"先求和得总 log，再取 exp 得总 ratio（连乘）"与"每一步单独算 ratio、分别加权"在梯度上等价，所以代码偷懒（也省显存）只保留逐步版本。

### 2.2 pipeline_with_logprob 内部 (`flux_pipeline_with_logprob_fast.py:23-213`)

```
输入: prompt_embeds, num_inference_steps=10, sde_window_size=3, noise_level=0.7
输出: images, all_latents, image_ids, text_ids, all_log_probs, all_timesteps
```

**SDE 窗口机制** (`flux_pipeline_with_logprob_fast.py:136-142`):

```python
if sde_window_size > 0:
    start = randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
    end = start + sde_window_size
    sde_window = (start, end)              # 例如 (2, 5)，表示只记录第 3-5 步的 log_prob
else:
    sde_window = (0, len(timesteps) - 1)   # 全部记录（最后一步接近图片，分布很尖，易精度溢出，故 -1）
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

**对应公式：** 上面三行代码就是下面的均值公式（也就是附录/10.2 那个式子的代码形态）：

$$ \text{mean} = x_t\Big(1+\frac{\sigma_{\mathrm{noise}}^2}{2\sigma}\Delta t\Big) + v_\theta\Big(1+\frac{\sigma_{\mathrm{noise}}^2(1-\sigma)}{2\sigma}\Big)\Delta t $$

其中 `std_dev_t` 就是 $\sigma_{\mathrm{noise}}=\sqrt{\frac{\sigma}{1-\sigma}}\cdot\text{noise\_level}$，`model_output` 是 DiT 预测的速度 $v_\theta$，`dt` 是相邻两个噪声水平的差 $\Delta t=\sigma_{prev}-\sigma$。

#### 均值公式是怎么来的（推到 4 步）

这条均值公式不是拍脑袋定的，而是 Flow-GRPO（arXiv 2505.05470，附录 A）**把确定性的概率流 ODE 转成"边际分布不变"的逆时 SDE 后，再做 Euler–Maruyama 离散化**的结果。四个推导步骤：

**第 1 步｜为什么要引入 SDE。** GRPO 的 PPO ratio 需要采样轨迹的逐转移概率 $p_\theta(x_{t+1}\mid x_t)$：
- 纯 **ODE** $\mathrm{d}x_t = v_t\,\mathrm{d}t$ 是确定性的，算它的 log-prob 要估计 **divergence**（Jacobian 迹），昂贵且不稳；
- **RL 需要探索随机性**——确定性采样除初始 noise 外没有任何随机源，探索效率低（论文实验：噪声太小训练效率骤降）。

所以要把 ODE 变成边际密度 $p_t$ 在**所有时刻都不变**的 SDE，这样转移核是高斯、log-prob 直接可算。

**第 2 步｜用 Fokker–Planck 反解漂移。** 令"一般 SDE 的边际演化"与"ODE 的边际演化"相等，即保证边际不变：
- ODE 的边际（连续性/Liouville 方程）：$\partial_t p_t = -\nabla\cdot[v_t\,p_t]$
- 一般 SDE $\mathrm{d}x_t = f_{\mathrm{SDE}}\,\mathrm{d}t + \sigma_{\mathrm{noise}}\,\mathrm{d}w$ 的边际（Fokker–Planck）：$\partial_t p_t = -\nabla\cdot[f_{\mathrm{SDE}}p_t] + \tfrac12\nabla^2[\sigma_{\mathrm{noise}}^2\,p_t]$

两者相等，用 $\nabla^2(\sigma^2 p)=\sigma^2\nabla\cdot(p\,\nabla\log p)$ 整理，得到漂移：

$$ f_{\mathrm{SDE}} = v_t - \frac{\sigma_{\mathrm{noise}}^2}{2}\nabla\log p_t(x) $$

（这正是经典概率流 ODE ⇄ score-SDE 的关系，Song et al. 2021。）

**第 3 步｜整流流下用速度场替换 score（关键一步）。** 对整流流插值 $x_t=(1-t)x_0 + t x_1$，score 可用速度场**闭式表示**：

$$ \nabla\log p_t(x) = -\frac{x}{t} - \frac{1-t}{t}\,v_t(x) $$

> 直觉来源：该插值的条件密度 $p_{t|0}=\mathcal{N}((1-t)x_0,\,t^2I)$，其 score 正比于 $-\mathbb{E}[x_1\mid x_t]/t$；而速度场 $v_t(x)=\mathbb{E}[x_1-x_0\mid x_t]$。两条式子消去 $\mathbb{E}[x_1\mid x_t]$ 即得该恒等式。

**第 4 步｜代回 + 欧拉离散化。** 把上式代入 $f_{\mathrm{SDE}} = v_t - \frac{\sigma_{\mathrm{noise}}^2}{2}\nabla\log p_t$，得到逆时 SDE：

$$ \mathrm{d}x_t = \Big[\underbrace{v_t(x_t) + \frac{\sigma_{\mathrm{noise}}^2}{2t}\big(x_t + (1-t)\,v_t(x_t)\big)}_{\text{漂移（均值方向）}}\Big]\mathrm{d}t + \sigma_{\mathrm{noise}}\,\mathrm{d}w $$

再按 **Euler–Maruyama** 离散化（式 12）：

$$ x_{t+\Delta t} = x_t + \left[v_\theta + \frac{\sigma_{\mathrm{noise}}^2}{2t}\big(x_t + (1-t)v_\theta\big)\right]\Delta t + \sigma_{\mathrm{noise}}\sqrt{-\Delta t}\,\epsilon $$

把方括号按 $x_t$ 与 $v_\theta$ 的系数拆开（$t=\sigma$），就是第 1 步那三行均值公式；高斯转移核方差为 $\sigma_{\mathrm{noise}}^2(-\Delta t)$，即第 2 步的 log_prob 公式。

**各项含义一览**：

| 项 | 角色 |
|---|---|
| $x_t$ 上的 $\frac{\sigma_{\mathrm{noise}}^2}{2\sigma}\Delta t$ | 补偿项：抵消 $\frac{\sigma^2}{2}\nabla\log p$ 对 $x_t$ 的拉拽（式 20 中 $-\frac{\sigma^2}{2}(-\frac{x}{t})$），保边际 |
| $v_\theta$ 上的 $\frac{\sigma_{\mathrm{noise}}^2(1-\sigma)}{2\sigma}\Delta t$ | 抵消 score 恒等式中 $-\frac{1-t}{t}v_t$ 的贡献 |
| $\sigma_{\mathrm{noise}}\sqrt{-\Delta t}\,\epsilon$ | 扩散项；去噪时 $\Delta t<0$，取 $\sqrt{-\Delta t}$ 保证方差为正 |
| $\sigma_{\mathrm{noise}}=\sqrt{\frac{\sigma}{1-\sigma}}\cdot\text{noise\_level}$ | 工程选择：注噪与当前"噪声/信号比"成正比（$t\to0$ 接近图片时噪声消失，$t\to1$ 接近纯噪时噪声最大） |

> 一句话：**均值公式 = "带 score 修正的逆时扩散 SDE" + "整流流下用速度场替换 score 的闭式恒等式" + "欧拉离散化"**。它让确定性 Flow-ODE 变成一个能逐高斯核算 log_prob、且带探索随机性的 SDE，从而支撑 GRPO 的在线策略梯度。

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

**对应公式：** 上面代码分别是"采样一步"和"算这一步的对数概率"：

$$ x_{t+1} = \text{mean} + \sigma_{\mathrm{noise}}\sqrt{-\Delta t}\;\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I) $$

$$ \log p(x_{t+1}\mid x_t) = -\frac{\|x_{t+1}-\text{mean}\|^2}{2\sigma_{\mathrm{noise}}^2(-\Delta t)} - \log\big(\sigma_{\mathrm{noise}}\sqrt{-\Delta t}\big) - \log\sqrt{2\pi} $$

最后一行 `.mean(dim=...)` 是把 latent 的每个维度（`C×H×W`）的 log_prob 平均成一个标量，对应公式里 $\|\cdot\|^2$ 求和后再除以维度数。

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

**对应公式：**

$$ A_i^{\text{clip}} = \operatorname{clamp}\big(A_i,\; -c,\; c\big) = \begin{cases} -c & A_i < -c\\ A_i & -c \le A_i \le c\\ c & A_i > c\end{cases}, \qquad c=\texttt{adv\_clip\_max}=2.0 $$

即把 A 超过 $[-2,2]$ 的部分"掐掉"，避免某一张图的极端高分/低分带偏整个梯度。

---

## 第 5 步：训练阶段——计算图如何构造？梯度如何流动？(核心!)

这是整个文章最重要的部分。训练阶段的代码在 `train_flux_fast.py:803-917`。

> ⚠️ **先澄清一个最常见的误解**（你可能会一直嘀咕"不是要学速度场吗？"）：
> **训练阶段不是重新生成一条新轨迹，而是把采样阶段已经走好的"旧轨迹"放回新模型面前，重新给每个已发生步骤打分。**
> 这里的模型**加载时已经会画画了**（速度场 v_θ 在 SFT/Flow Matching 预训练里就学好了）。Flow-GRPO 的"训练"**不是学 v_θ 本身，而是用 LoRA 把 v_θ 的方向往 reward 期望的方向推一点**。所以你在训练循环里看不到 `v_target` 这种监督——你看到的一堆 latent / log_prob / ratio，正是"旧轨迹在新模型下"的评分。整个训练代码里看不见"学习速度场"的固定 $$\frac{\partial v}{\partial t}$$ 目标，因为**速度场是现成的，改的是它上面插的 LoRA**。

### 5.1 总体结构

```python
# train_flux_fast.py:803-917
#################### TRAINING ####################
for inner_epoch in range(config.train.num_inner_epochs):
    pipeline.transformer.train()        # 切到 train 模式
    for i, sample in enumerate(samples_batched):
        for j in train_timesteps:       # j 遍历窗口内 num_train_timesteps 步
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

**对应公式：** 这一段代码做的事 = 用当前参数 $\theta$ 跑一次 SDE 步，并算这一步的对数概率：

$$ \text{mean}_\theta = x_j\Big(1+\frac{\sigma_{\mathrm{noise}}^2}{2\sigma}\Delta t\Big) + v_\theta(x_j,t_j,c)\Big(1+\frac{\sigma_{\mathrm{noise}}^2(1-\sigma)}{2\sigma}\Big)\Delta t $$

$$ \log p_\theta(x_{j+1}\mid x_j) = -\frac{\|x_{j+1}-\text{mean}_\theta\|^2}{2\sigma_{\mathrm{noise}}^2(-\Delta t)} - \log\big(\sigma_{\mathrm{noise}}\sqrt{-\Delta t}\big) - \log\sqrt{2\pi} $$

注意这里 `prev_sample` 传入的是旧轨迹的 `next_latents[:, j]`（不是重新随机采），所以 $\log p_\theta(x_{j+1}\mid x_j)$ 表示"**旧轨迹那一步，在新参数 $\theta$ 看来有多可能**"——这正是 PPO ratio 需要的 $\log\pi_{\theta_{\text{new}}}$。

**和采样阶段的关键区别：**

| | 采样阶段 | 训练阶段 |
|--|---------|---------|
| `requires_grad` | ❌ 全部 no_grad | ✅ 开启 |
| `prev_sample` 参数 | `None`（自己随机采） | 传入旧轨迹的 `next_latents` |
| `prev_sample.detach()` | 有效（反正不记梯度） | 有效（防止梯度流到 prev_sample） |

### 5.3 训练阶段的计算图——用图形理解

训练阶段执行 `compute_log_prob` 时，PyTorch 自动构造了如下计算图（蓝色 = 前向构造计算图，红色 = 反向传播）：

![训练阶段计算图与梯度流](/images/flowgrpo/flowgrpo_compute_graph.svg)

**梯度流动路径（链式法则）：**

```
∂loss/∂θ =
    ∂loss/∂log_prob          (从 PPO loss 到 log_prob)
  × ∂log_prob/∂model_pred   (从 log_prob 公式到 model_pred——高斯 log_prob 对 mean 求导)
  × ∂model_pred/∂θ           (从 DiT 前向到 LoRA 参数——autograd 自动完成)
```

**对应公式（链式法则）：**

$$ \frac{\partial \mathcal{L}}{\partial \theta} = \underbrace{\frac{\partial \mathcal{L}}{\partial \log p}}_{\text{PPO loss 对 log\_prob}} \cdot \underbrace{\frac{\partial \log p}{\partial v_\theta}}_{\text{解析可得，见下}} \cdot \underbrace{\frac{\partial v_\theta}{\partial \theta}}_{\text{autograd 自动}} $$

$\log p$ 只通过 $\text{mean}_\theta$（进而 $v_\theta$）依赖参数 $\theta$，且 $\text{mean}_\theta$ 是 $v_\theta$ 的线性函数，所以中间那项能直接手推出来，剩下两层交给 PyTorch。

展开第二项：

```python
# sd3_sde_with_logprob.py:109,121-133
log_prob = -((prev_sample.detach() - prev_sample_mean)²) / (2 * var) - log(√(var * 2π))

# 其中 prev_sample_mean = g(model_pred, sample, dt, noise_level)
#     var = (std_dev_t * √(-dt))²

# ∂log_prob/∂model_pred = -(prev_sample - prev_sample_mean) / var * ∂prev_sample_mean/∂model_pred
```

**对应公式（高斯 log_prob 对速度求导）：** 中间那一项可以手推闭式解，关键是 `prev_sample` 被 `.detach()` 了、只有 `prev_sample_mean` 带梯度：

$$ \frac{\partial \log p}{\partial v_\theta} = \frac{x_{j+1}-\text{mean}_\theta}{\sigma_{\mathrm{noise}}^2(-\Delta t)} \cdot \Big(1+\frac{\sigma_{\mathrm{noise}}^2(1-\sigma)}{2\sigma}\Big)\Delta t $$

推导：$\frac{\partial \log p}{\partial \text{mean}} = \frac{x_{j+1}-\text{mean}}{\sigma_{\mathrm{noise}}^2(-\Delta t)}$（对 log_prob 的第一项求导），再乘 $\frac{\partial \text{mean}}{\partial v_\theta} = \big(1+\frac{\sigma_{\mathrm{noise}}^2(1-\sigma)}{2\sigma}\big)\Delta t$（均值公式里 $v_\theta$ 的系数）。梯度只会顺着这条链流到 `model_pred`。

因为 `prev_sample.detach()` 了，梯度不会流到 `prev_sample`，只会通过 `prev_sample_mean` 流到 `model_pred`。

**重要：计算图只包含当前第 j 步！**
- `sample["latents"][:, j]` 是 detached 的（采样阶段产生，不参与计算图）
- 所以梯度只从第 j 步的 model_pred 反向传播到 LoRA 参数
- **为什么这可行？** SDE 的马尔可夫性质：第 j 步的 latent 只由第 j-1 步决定。虽然每一步的梯度只包含一步的信息，但经过 window_size 步的累积（循环 `num_train_timesteps` 次 `compute_log_prob` + `backward`），梯度实际上包含了从窗口起点到终点的完整信息。

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

**对应公式（窗口内第 $j$ 步的完整损失）：**

$$ r_j(\theta) = \exp\!\big(\underbrace{\log p_\theta(x_{j+1}\mid x_j)}_{\texttt{log\_prob}} - \underbrace{\log p_{\theta_{old}}(x_{j+1}\mid x_j)}_{\texttt{sample['log\_probs'][:, j]}}\big) $$

$$ L_j = \mathbb{E}\Big[\max\big(-A_j\,r_j(\theta),\; -A_j\,\operatorname{clip}\big(r_j(\theta),\,1-\epsilon,\,1+\epsilon\big)\big)\Big] $$

若开启 KL 惩罚（见 6.3），最终 `loss = policy_loss + beta * kl_loss`（即 $L_j + \beta L^{KL}$）。

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

### 6.2 为什么 GRPO 用的是 PPO Loss？(GRPO vs PPO 的关系)

很多人会疑惑：GRPO 和 PPO 到底啥关系？为什么 GRPO 的损失还是长 PPO 的样子？

> **一句话结论：GRPO 的"组相对"只改了 advantage 怎么算；策略更新的损失函数确实延续了 PPO 的 clipped surrogate（带 clip 的 ratio loss）。**

把 GRPO 拆成两件事看就明白了：

```text
GRPO =  PPO 的损失函数（clip ratio）  +  GRPO 特有的 advantage（组内归一化，不用 critic）
        └──── 为什么 loss 是 PPO 样子的 ────┘    └────────── "组相对"在这 ──────────┘
```

**① advantage 怎么算 —— 这才是 GRPO 的创新点**

- PPO 要训练一个 **critic 价值网络**去估计"这个状态有多好"，作为 baseline（基线）；
- GRPO 把 critic **整个砍掉**，改用同一个 prompt 生成的 `G` 条轨迹的 reward **组内归一化**当 baseline：

$$A_i = \frac{r_i - \mu}{\sigma}$$

- "组相对"就在这里：比组内平均好→正，差→负。**这一步只改变 `A`，不改 loss 的"外壳"。**

**② 策略更新损失 —— 沿用 PPO**

不管 `A` 来自 critic 还是组内归一化，最终要更新的目标都是同一个 PPO 式子（见 6.1）：

$$\text{ratio} = \exp\!\big(\log\pi_\theta - \log\pi_{\theta_{old}}\big), \qquad L = \mathbb{E}\Big[\min\big(\text{ratio}\cdot A,\; \operatorname{clip}(\text{ratio},1\pm\epsilon)\cdot A\big)\Big]$$

GRPO 在损失这里**没有发明新东西**，直接继承 PPO 的 clip 技巧。所以你会看到"**advantage 换成组内算，但 loss 还是 PPO 那个**"。

**为什么 loss 必须参考 PPO / 必须带 clip？**

因为 GRPO 砍掉 critic 之后，如果连 clip 也拿掉，就退化成最朴素的 **REINFORCE**：

- REINFORCE 方差大、一步更新就没谱，很容易"一步更新过猛导致崩溃"；
- PPO 的 `clip(ratio, 1±ε)` 限制"新策略和旧策略偏离别超过 ε"，保证每步更新温和，**还能安全地同一批样本复用多轮**（提升样本效率）；
- 文生图/驾驶这类 reward 稀疏又易 hack 的场景，还叠加了 **over-optimization**（reward 越编越高、图却越来越差）风险，`clip` + KL 惩罚正是压住它的。

> **一句话记住：GRPO = 没有 critic（不用价值网络）的 PPO**——把"状态价值基线"换成"组内 reward 的均值/标准差"，其余损失设计（clip ratio + KL 惩罚）照搬 PPO。
>
> 补充：DeepSeek-R1 的 GRPO 有时也用**不带 clip 的 REINFORCE + KL**版本（加了 `beta·KL` 后 KL 本身就能"别偏太远"，所以不需要 clip）。但**驾控/文生图这种 reward 易 hack 的场景，clip 版更稳**——本仓库用的就是 clip 版。

### 6.3 KL 惩罚项

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

**对应公式：**

$$ L^{KL} = \frac{1}{2\sigma_{\mathrm{noise}}^2}\,\big\|\text{mean}_\theta(x_j) - \text{mean}_{\theta_{\text{ref}}}(x_j)\big\|^2, \qquad L = L^{PG} + \beta\,L^{KL} $$

即"当前模型（开 LoRA）与 reference（禁用 LoRA）在同一个 $x_j$ 上预测的下一步均值不能差太远"，用 $v_\theta$ 一步的方差 $\sigma_{\mathrm{noise}}^2$ 做归一化。

**设计动机：**
- 通过 `disable_adapter()` 暂时禁用 LoRA 权重，得到 base model（reference）的预测均值
- 当前模型的预测均值 `prev_sample_mean` 不应偏离 reference 太远
- KL 惩罚项防止 PPO 过度优化导致模型遗忘原始能力（catastrophic forgetting）

### 6.4 日志指标

```python
# train_flux_fast.py:863-888
info["approx_kl"].append(0.5 * ((log_prob - sample["log_probs"][:, j]) ** 2).mean())
    # 近似 KL 散度：log_prob_new 与 log_prob_old 的差异平方均值
info["clipfrac"].append((|ratio - 1.0| > clip_range).float().mean())
    # 被 clip 的样本比例
```

**对应公式：**

$$ \text{approx\_kl} = \mathbb{E}\Big[ \tfrac{1}{2}\big(\log p_{\theta_{\text{new}}} - \log p_{\theta_{\text{old}}}\big)^2 \Big] $$

$$ \text{clipfrac} = \mathbb{E}\big[ \mathbb{1}\{\,|r_j(\theta)-1|>\epsilon\,\} \big] $$

两个都是监控指标：`approx_kl` 用"新旧 log_prob 之差的平方均值"近似 KL 散度，衡量策略每次更新偏离多远；`clipfrac` 统计有多少样本的 ratio 超出 $1\pm\epsilon$ 被裁剪（值偏高说明每次步子太大）。

### 6.5 ratio 是"每步"还是"总"？——逐步各算，等价于总体连乘

你可能会想：PPO 的 ratio 不是应该用**整条轨迹** $\pi_\theta(\tau)/\pi_{\theta_{old}}(\tau)$ 吗？为什么代码里 `ratio = torch.exp(log_prob - sample["log_probs"][:, j])` 只有第 j 步？

- **每步独立算**：`ratio_j = π_θ(x_{j+1}|x_j) / π_θ_old(x_{j+1}|x_j)`，只用到第 j 步的转移概率；
- **整条轨迹**：$\text{ratio}_\tau = \frac{\pi_\theta(\tau)}{\pi_{\theta_{old}}(\tau)} = \prod_j \frac{\pi_\theta(x_{j+1}|x_j)}{\pi_{\theta_{old}}(x_{j+1}|x_j)} = \prod_j \text{ratio}_j$（马尔可夫连乘，取 log 就是各步 log_prob 之差的和）；
- **为什么逐步算没问题**：PPO 损失对每步取 `min(wA, clip(w)A)`，advantage 对每个 timestep 用的是**同一个** $A_j=A$（`train_flux_fast.py:745` 把 `adv` 沿 timestep 维复制了 `num_train_timesteps` 次）。于是"一条轨迹的梯度" = 各步梯度之和，与"先乘出总 ratio 再算"在数学上一致，只差 clip 边界的实现细节——这也是 PPO 的标准做法。

**一句话：你不用在代码里找"总 log_prob 求和"，它被隐式地分散在每一步的梯度里。**

---

### 6.6 "旧模型" vs "新模型"：其实是同一个模型的两个瞬间

新手最容易在这里想岔：以为训练时"采样用一个旧模型，打分再换一个新模型"。不——**全仓库只有一个 FLUX+LoRA 模型**，"旧/新"只是 `optimizer.step()` 前后那一个瞬间的参数快照：

![旧模型和新模型是同一个模型的两个瞬间](/images/flowgrpo/flowgrpo_old_new_model.svg)

按时间线看一遍（对应上图）：

1. **采样**（用的就是当时的那套参数，记作 $\theta$）：走完去噪轨迹，缓存 `next_latent`，同时用 $\theta$ 记下 `log_prob_old`。此刻的模型就是"即将被更新"的那个。
2. **打分**：`reward` → `A`（正奖励抬这条轨迹，负奖励压它）。这是方向信号。
3. **训练开始**：`compute_log_prob` 算出 `log_prob_new`。注意——**此刻参数还是 $\theta$，没动过**，所以第一轮 $ratio=\exp(\log p_{new}-\log p_{old})\approx 1$，`loss≈0`。
4. **`optimizer.step()`** ——就在这一行喝完咖啡之后，$\theta$ 第一次发生了变化，变成"新模型 $\theta_{new}$"。
5. **下一轮采样**：用的已经是更新后的 $\theta_{new}$。从这一轮起，你先前存的 $\theta_{new}$ 自动"降级"成新的"旧模型"，新的 `log_prob_old` 由它给出，于是 $ratio \neq 1$，训练真正开始推动。

**第一轮是"空转热身"吗？不是白干。** 虽然 step 前 `ratio≈1`、`loss≈0`，但 `backward()` 已经算出了**非零梯度**——因为每条轨迹的 reward/Advantage 不同，梯度方向早已指向"该抬的抬，该压的压"。`step()` 就是按这个方向挪第一小步（步长由 `lr` 控制）。所以第一轮就开始朝正确方向挪了，只是幅度很小；第二轮采样的轨迹已是挪过后的模型生成的。

**监督信号到底是什么？** 整条链里没有一个"标准答案轨迹"。模型是被**打分（Advantage 的正负）**推着走的：A 决定方向（往哪个区域挪），`log_prob_new` 只是把"该往那挪"量化成梯度大小。加速的是"速度方向分布"（LoRA 参数），FLUX 原始权重全程冻结。

一句话：**"新模型"从来不是凭空变出来的，它只是"旧模型"在打分和梯度推完、被 step() 哐当一撞之后的那一个瞬间。**

---

## 第 7 步：一次完整的训练迭代——按时间线串联

现在我们把所有步骤串联起来，看一次完整的迭代：

![一次完整的训练迭代](/images/flowgrpo/flowgrpo_training_iteration.svg)

上面这幅图对应下面的完整时间线：

**① 采样阶段** (`pipeline.transformer.eval()`，no_grad)
1. 读 prompts = `["a red cat", "blue dog", ...]`（每 GPU 1 个）
2. T5 编码 → prompt_embeds
3. `pipeline_with_logprob`（10 步去噪，其中窗口 3 步随机 SDE）
   - 随机选一个窗口，例如第 2-5 步：SDE，noise=0.7，记录 latent + log_prob_old
   - 窗口外（第 0-1、6-9 步）：ODE，noise=0，不记录
   - 返回 images (B,3,512,512) 和 log_probs_old (B,3)
4. `executor.submit(reward_fn, ...)` 异步提交 reward 计算
5. 存储 latents / timesteps / prompt_embeds 供训练阶段复用

**② 异步 Reward 计算**（等 Future 结果）
- rewards = `{"avg": [0.12, 0.45, ...], "strict_accuracy": [0, 1, ...]}`

**③ 跨 GPU 同步 + Advantage**
1. `all_gather`：32 GPU × 1 batch = 32 个样本汇总
2. `PerPromptStatTracker.update()`：对每个 prompt 算 `A = (r - μ_history) / σ_history`
3. A 裁剪到 [-2.0, 2.0]

**④ 训练阶段** (`pipeline.transformer.train()`，梯度开启)
- inner_epoch=0 里，j 遍历窗口内的 `num_train_timesteps` 步（例 window_size=3），每步：
  1. `transformer(latents[:,j], ...) → model_pred`（前向，梯度开启）
  2. `sde_step_with_logprob(..., prev_sample=next_latents[:,j]) → log_prob_new`
  3. `ratio = exp(log_prob_new - log_probs_old[:,j])`
  4. `policy_loss = max(-A·ratio, -A·clip(ratio))`
  5. （可选）KL_loss = `||mean_new - mean_ref||² / (2·var)`
  6. `loss = policy_loss + β·KL_loss`
  7. `accelerator.backward(loss)` → 梯度累加到 LoRA 参数
  8. `optimizer.step()` → LoRA 参数更新
- 窗口内所有步完成后：epoch + 1，回到采样阶段

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
| `sde_window_size` | `sde_window_size`（如 3） | **0**（全 ODE） |
| `noise_level` | 0.7 | **0**（不加噪声） |
| `requires_grad` | ❌ | ❌ |
| 返回 `log_probs` | ✅ 需要 | **不返回**（`_` 丢弃） |
| 返回 `latents` | ✅ 供训练复用 | ❌ |

**为什么推理不用 SDE？**
- 推理只关心图片质量，不需要计算 log_prob
- ODE（noise_level=0）生成的图片更清晰、质量更高
- SDE 在训练时是为了引入随机性 → 多样性 → group 内 reward 出现差异 → advantage 信号

### 8.1 采样 / 训练 / 推理三阶段到底各干什么？（一张表收尾）

到这里所有函数都出现了，最后把三阶段的职责和它们共用的函数一次性理清。**你会发现同一个 `sde_step_with_logprob` 被三处调用，只是 `prev_sample` 传的东西不同**：

| | 采样阶段 (SAMPLE) | 训练阶段 (TRAIN) | 推理阶段 (EVAL) |
|--|------------------|------------------|----------------|
| **代码位置** | `train_flux_fast.py:610-712` | `train_flux_fast.py:803-917` | `train_flux_fast.py:220-311` |
| **目标** | 用**当前(旧)模型**生成图片，并**记录旧概率** | 让新模型给旧轨迹**重新打分**，算 PPO loss | 用（EMA 平滑的）模型纯生成，测 reward |
| **有梯度吗** | ❌ `no_grad` | ✅ `requires_grad` | ❌ |
| **`prev_sample` 参数** | `None` → 函数**掷骰子随机生成**下一步 | `sample["next_latents"][:, j]` → **不采样**，只算已发生步骤的概率 | `None`，但 `noise_level=0` → **纯 ODE 不掷骰子** |
| **返回的 log_prob 是** | $\log p_{\theta_{old}}$（旧概率） | $\log p_{\theta}$（新概率） | 不需要（丢弃） |
| **产出** | 图片、轨迹 $x_0..x_N$、`log_probs_old` | `loss` → `backward` → 更新 LoRA | 图片 + reward |
| **调用链** | `pipeline_with_logprob` → `sde_step_with_logprob(prev=None)` | `compute_log_prob` → `sde_step_with_logprob(prev=next_latents)` | `pipeline_with_logprob(window=0, noise=0)` |

**为什么同一个函数三处都能用？** 看 `sd3_sde_with_logprob.py:111-119` 的分支：
```python
if prev_sample is None:          # 采样/推理：自己掷骰子生成下一步
    prev_sample = prev_sample_mean + std_dev_t * sqrt(-dt) * noise
# 不管 prev_sample 从哪来，log_prob 永远是：
#   拿已知的 prev_sample 对比当前模型的 prev_sample_mean（高斯分布中心）
```
log_prob 的定义只依赖"**已知点 vs 模型分布中心**"，所以：
- 采样时 `prev_sample` 是刚掷出的点 → 记下的是旧模型掷出它的概率；
- 训练时 `prev_sample` 换成已固定的旧轨迹点 → 算的是新模型眼里旧轨迹上这个点的概率 = `log_prob_new`；
- 推理时 `noise_level=0` 使 `std_dev_t=0` → SDE 退化为 ODE → 不再随机，也就不需要概率。

> **一个图**：三阶段的循环关系就是上面 `flowgrpo_overview.svg` 画的：**采样 → reward → advantage → 训练 → 回到采样**。推理只在 eval 时独立跑，不参与闭环。

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

## 附：关键流程图 + 完整数学原理

下面三张流程图的目的是把正文里的公式**图形化**。结合图，我把每张图背后的数学公式完整写出来——尤其是**第二张 Flow Matching 图**，它是整条 `ODE → SDE → log_prob → GRPO` 链条的起点，也是最容易"看着网页代码却不知道在算什么的"地方。

---

### 附 A｜Flow Matching 图：原生 FM + ODE→SDE 的完整数学

![Flow Matching 去噪过程](/images/flowgrpo/flowgrpo_flow_matching.svg)

上图是"三步动画"：从纯噪声出发，靠**速度**一步一步挪到最终图片。整张图就靠"速度"一个概念，所以我们从最基础的开始讲。

**第 1 步：先认识主角：噪声 $x_0$、图片 $x_1$，以及"在它们中间"的 $x_t$**

- $x_0$：`t=0` 那一端，一张全是随机雪花点的图（像素值全是随机数）。
- $x_1$：`t=1` 那一端，我们最终想要的那张图（比如 "a red cat"）。
- 中间的 $x_t$：噪声朝图片走了一半时的状态。可以把 $t$ 理解成**进度条**（0=还没走，1=走完了）。

这两个主角之间的"路"怎么定义？——在两点之间拉一条直线。为什么是直线、为什么系数长这样？下面**一步一步推**给你看。

**公式 ① 从哪来：中间点 $x_t=(1-t)x_0+t x_1$ 是"推"出来的，不是拍脑袋定的**

我们要求中间状态 $x_t$ 满足三个朴素条件：
1. $t=0$（还没走）时必须正好是噪声：$x_0$；
2. $t=1$（走完了）时必须正好是图片：$x_1$；
3. 中间平滑过渡——最简单的假设就是**线性**（直线）。

设直线 $x_t = a + b\,t$（$a,b$ 是待定系数），代条件 1 和 2：

- $t=0$：$x_0 = a + b\cdot0 = a$，所以 $\boxed{a = x_0}$；
- $t=1$：$x_1 = a + b\cdot1$，把 $a=x_0$ 代入得 $\boxed{b = x_1 - x_0}$。

把 $a,b$ 代回直线方程：

$$
x_t = x_0 + t\,(x_1 - x_0) = (1-t)\,x_0 + t\,x_1
$$

逐项翻译：噪声 $x_0$ 的系数是 $(1-t)$，图片 $x_1$ 的系数是 $t$，**两个系数加起来恒等于 1**（$(1-t)+t=1$）。所以它本质是"噪声和图片的**加权平均**"，权重随进度条 $t$ 变化：$t=0$ 全是噪声，$t=1$ 全是图片，$t=0.5$ 各占一半。

**公式 ② 从哪来：速度为什么等于 $x_1-x_0$？——求一次导就出来**

速度 = "位置随时间的变化率"，也就是对 $x_t$ 关于 $t$ 求导。逐项来：

- $(1-t)x_0$ 对 $t$ 求导：$(1-t)$ 的导数是 $-1$，乘 $x_0$ 得 $-x_0$；
- $t\,x_1$ 对 $t$ 求导：$t$ 的导数是 $1$，乘 $x_1$ 得 $x_1$。

相加：

$$
u_t = \frac{d x_t}{dt} = \frac{d}{dt}\big((1-t)x_0 + t x_1\big) = -x_0 + x_1 = x_1 - x_0
$$

结论：**直线轨迹上每个点的瞬时速度都是同一个常数 $x_1-x_0$**（从噪声指向图片的那个方向）。这就是为什么图里每个小方框都写着 `v·Δt`——整条路速度不变，每步都朝同一个方向、用同一个速度。

**第 2 步：训练：让网络学会"猜速度"（CFM 损失）**

网络 $v_\theta(x_t,t,c)$ 的输入是"我在哪（$x_t$）、几点（$t$）、目标长啥样（$c$=prompt）"，输出是"我认为该往哪走（速度）"。

训练时我们有成对的"噪声 + 它对应的真图片"，所以**正确答案是现成的**：就是 $x_1-x_0$。于是损失（衡量网络猜得有多离谱的"扣分器"）就是：

**公式 ③ 从哪来：训练目标 CFM 损失**

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}\Big[\big\| v_\theta(x_t,t,c) - (x_1-x_0) \big\|^2\Big]
$$

逐项翻译：
- $v_\theta(x_t,t,c)$：网络猜的速度；$(x_1-x_0)$：正确的速度；两者相减再平方 = "**猜偏了多少**"。
- $\mathbb{E}[\cdot]$：期望 = 拿很多很多对"噪声+图片"反复问，取平均。$\mathbb{E}_{t\sim U[0,1]}$ 表示还随机抽进度条位置 $t$（每个中间位置都练到）。
- 这就是最普通的**回归**训练：目标是一个连续数值（速度向量），不是选择题。

**顺便回答"为什么是平方"**：平方误差不是随便选的——如果假设"网络预测速度时带高斯噪声"，预测越准概率越大，取负对数似然恰好就得到平方误差（这是标准回归的推导，一个高斯 PDF 的负对数就是一个平方项）。所以 CFM 损失本质就是："在每个中间点上，让网络猜的速度尽量贴近真速度"。

**为什么这么简单就够用？** 直觉类比：一群同学乱糟糟站在操场上（一堆噪声点），老师喊"按你们该去的位置走"。只要**每个点都朝正确方向、按正确速度走**，整群人就会从乱糟糟的站位，整齐地"搬运"到各自的正确位置（= 图片的分布）。Flow Matching 的定理说的就是这个：让每个点朝正确速度走 ⟺ 把噪声分布整体搬成图片分布。不需要 GAN 那种"判别器"、也不需要额外技巧。

**第 3 步：推理：生成图片 = 解方程，这里必须讲清楚"积分"和"Euler"**

学完速度后怎么生成一张图？——从噪声出发，沿学到的速度一路走到底。这条路在数学上叫**解 ODE（常微分方程）**：

**公式 ④ 从哪来：ODE 就是"速度配方"**

$$
\frac{dx}{dt} = v_\theta(x,\;t,\;c), \qquad x_0\sim \mathcal{N}(0,I),\;\; t:0\to 1
$$

翻译：这是一条"**速度配方**"——它说"无论你现在在哪（$x$）、现在几点（$t$），你的速度应该是 $v_\theta(x,t,c)$"。$x_0\sim\mathcal{N}(0,I)$ 表示起点是随机噪声。

**那"积分"是什么？** 我们已经知道任意时刻的速度，但想知道"最终到哪"，得把一路上每个瞬间的速度**累加起来**：

**公式 ⑤ 从哪来：积分就是"把所有瞬间的速度 × 时间 加起来"**

- 匀速时很简单：**路程 = 速度 × 时间**。
- 但我们的速度 $v_\theta$ 一直在变（你换了个位置，配方给的速度就变了），没法一次算完。
- "把无数小段 '速度 × 时间' 加在一起"——这件事就是**积分**。就像开车：时速表在变，行车记录仪把每秒钟的时速累计起来就是总里程。数学上写作

$$
x(1) = x(0) + \int_0^1 v(x(t),t,c)\,dt
$$

那个拉长的 $\int$ 是积分号，意思就是"把 0 到 1 之间每个瞬间的速度都加一遍"。

**那 Euler 又是什么？** 积分不能一笔算完（因为速度本身依赖位置，循环依赖），所以只能**近似**。**Euler 步的公式其实是"泰勒展开的第一项"推出来的**：

已知当前位置 $x(t)$ 和这一瞬间的变化速度 $\frac{dx}{dt}$，往前挪一小段 $\Delta t$，位置可以展开成：

$$
x(t+\Delta t) = x(t) + \underbrace{\frac{dx}{dt}\Big|_{t}\cdot \Delta t}_{\text{第一项：速度 × 时间}} + \underbrace{O(\Delta t^2)}_{\text{高阶小量：丢掉}}
$$

把 $\frac{dx}{dt}=v_\theta(x_t,t,c)$ 代入，并丢掉高阶小量 $O(\Delta t^2)$，就得到图里那个公式：

**公式 ⑥ 从哪来：Euler 步**

$$
x_{t+\Delta t} = x_t + v_\theta(x_t,t,c)\,\Delta t
$$

- 翻译：**新位置 = 旧位置 + 当前速度 × 一小段时间**。这就是"一步"。
- 误差来源：被丢掉的高阶小量 $O(\Delta t^2)$——所以 $\Delta t$ 越小、每一步越精确。
- 类比开车：你只看了一眼时速表（50 km/h），就假设接下来 1 分钟一直开 50，于是估算"这 1 分钟走了大约 50/60 公里"，然后重新看表再算。
- 图里把 $t:0\to1$ 切成 **10 段**（`num_inference_steps=10`），每段 $\Delta t=0.1$，10 个小方框就是 10 次 Euler 步进。

一个具体的数字例子（假设网络恒输出 $v=0.5$，起点 $x_0=0$，$\Delta t=0.1$，走 10 步）：

$$
0 \xrightarrow{+0.05} 0.05 \xrightarrow{+0.05} 0.10 \to \cdots \to 0.50
$$

速度恒定时 Euler 恰好精确（没有"变化"需要近似）；速度会变时，每步有一点点小误差，但 $\Delta t$ 越小误差越小。**总结：Euler 是一种"数值积分"——用"小步走 + 每一步都按当前速度走"来近似真正的积分。**

**第 4 步：为什么接 GRPO 必须把 ODE 转成 SDE？**

先说结论：GRPO 要"算出一条轨迹的概率，再用奖励去推高/压低它"。这句话里最关键的是那条**策略梯度公式**——它到底怎么来的？这是本附录最重要的推导，我们花六小步把它彻底拆开。

**公式 ⑦ 从哪来：策略梯度 $\nabla_\theta \mathcal{L} = -\,\mathbb{E}_\tau[A(\tau)\nabla_\theta\log\pi_\theta(\tau)]$ 的完整推导**

*第 1 步：写清楚目标。* 设参数为 $\theta$ 时，策略以概率 $\pi_\theta(\tau)$ 走出轨迹 $\tau$，每条轨迹有一个奖励 $R(\tau)$（来自 OCR/奖励函数，跟 $\theta$ 无关）。我们要最大化"期望奖励"：

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\big[\,R(\tau)\,\big] = \int \pi_\theta(\tau)\,R(\tau)\,d\tau
$$

*第 2 步：难在哪里？* 奖励 $R(\tau)$ 是**外部打分，里面没有 $\theta$**。直接对 $J$ 求导会卡住：梯度想钻进 $\pi_\theta$ 里去改变概率，但 $R$ 那边没东西可导。必须用一个数学技巧，把"对 $R$ 求导"换成"对 $\pi_\theta$ 求导"。

*第 3 步：log 求导技巧（score function trick）。* 利用"对数函数的导数 = 自身的导数 ÷ 自身"：

$$
\frac{d}{d\theta}\log \pi_\theta(\tau) = \frac{1}{\pi_\theta(\tau)}\cdot\frac{d\pi_\theta(\tau)}{d\theta}
\;\;\Longrightarrow\;\;
\boxed{\;\frac{d\pi_\theta(\tau)}{d\theta} = \pi_\theta(\tau)\cdot\frac{d}{d\theta}\log\pi_\theta(\tau)\;}
$$

翻译：**"概率的梯度 = 概率自己 × log概率的梯度"**。这一步把求导对象从 $R$ 成功换到了 $\log\pi_\theta$ 上。

*第 4 步：代入目标，把积分还原成期望。*

$$
\nabla_\theta J(\theta) = \int \frac{d\pi_\theta(\tau)}{d\theta}\,R(\tau)\,d\tau
= \int \pi_\theta(\tau)\;\nabla_\theta\log\pi_\theta(\tau)\;R(\tau)\,d\tau
= \mathbb{E}_\tau\Big[\,\underbrace{R(\tau)}_{\text{奖励}}\;\underbrace{\nabla_\theta\log\pi_\theta(\tau)}_{\text{log概率的梯度}}\,\Big]
$$

注意积分里又出现了 $\pi_\theta$，于是积分变回期望——这就是"期望奖励的梯度 = 奖励 × log概率梯度 的期望"。

*第 5 步：蒙特卡洛近似 + 用 advantage 替代奖励。* 期望用采样近似（多生成几条轨迹取平均）；再把纯奖励 $R$ 换成 advantage $A$（减掉一个基线不会改变梯度方向，但能大幅降方差——GRPO 的组内归一化就是干这个的）：

$$
\nabla_\theta J(\theta) \approx \frac{1}{M}\sum_{i=1}^{M} A(\tau_i)\,\nabla_\theta\log\pi_\theta(\tau_i)
$$

*第 6 步：转成"梯度下降"用的 loss。* PyTorch 优化器做的是梯度下降（$L$ 越小越好），所以加个负号，得到附录开头那条公式：

$$
\boxed{\;\nabla_\theta \mathcal{L} = -\,\mathbb{E}_\tau\Big[\, A(\tau)\; \nabla_\theta \log\pi_\theta(\tau) \Big]\;}
$$

直觉收尾：$A>0$（好事）的轨迹，梯度方向让 $\log\pi_\theta(\tau)$ **增大**（更可能再走出这条路）；$A<0$（坏事）的轨迹，让它**减小**。

现在回头看 ODE 为什么不行就非常清楚了——这个公式**必须要有"可微的概率"**，而纯 ODE 恰恰给不了：

1. **没有随机性 = 没有可调的概率。** ODE 是"复印机"：同样的噪声进去，永远出来同一张图。$\pi_\theta(\tau)$ 要么是 1、要么是 0，$\nabla_\theta\log\pi_\theta$ 处处退化、梯度算不出来。就像一位从不失误、每次都做同一个动作的运动员——没有"发挥好坏"的变化，教练无从指导。
2. **没有随机性 = 无法探索。** 让图片变好，必须靠"多试试不同的中间路线，看哪条 reward 高"。纯 ODE 每次路线都一样，采样 $M$ 次也是同一条路，期望里没有多样性可用。

所以做法是：**给同一个速度场加一点"随机抖动"，把确定性 ODE 变成随机 SDE**：

**公式 ⑧ 从哪来：ODE→SDE，在速度上叠一层随机噪声**

$$
dx_t = v_\theta(x_t,t,c)\,dt + g(t)\,dW_t
$$

- 第一项 $v_\theta\,dt$：还是学好的速度，负责**带路**。
- 第二项 $g(t)\,dW_t$：随机扰动（$W_t$ 是布朗运动，$dW_t$ 可以理解成"一撮随机噪声"），负责**制造变化**。
- 类比：**一手扶着画轨（速度），一手轻轻颤抖（噪声）**。轨迹大体还是沿学好的方向走，但每次会有点随机偏差——这正是"随机策略"：同样的 prompt，每次生成的图不完全一样。
- 抖动大小由 `noise_level` 控制：0 = 不抖 = 退回 ODE；越大 = 越抖 = 探索越强。

**为什么这不算"扔掉学好的模型"？** 漂移项仍是训练好的 $v_\theta$，轨迹主体还是沿着图片流形走，噪声只是叠加的"探索开关"。这也解释了图里下方对比框：**Flow 学的是速度 $v$，DDPM 学的是噪声 $\varepsilon$**——所以两者的 SDE 长得很像，但离散公式的系数完全不同。

**第 5 步：离散 SDE 一步（Euler–Maruyama）+ log_prob**

"Euler–Maruyama"就是 **Euler + 噪声**：每走一步，除了按 Euler 的规则前进，还额外叠一个高斯随机数。分四小步看。

**公式 ⑨ 从哪来：先算这一步该抖多大**

$$
\sigma_{\mathrm{noise}} = \sqrt{\frac{\sigma}{1-\sigma}}\cdot \text{noise\_level}
$$

（$\sigma$ 是当前噪声水平，`noise_level` 是预设强度。具体数值不用背，理解成"这步抖多大"。注意：当 `noise_level=0` 时它等于 0——回到不抖，这个性质下面马上要用。）

**公式 ⑩ 从哪来：下一步"最可能到哪"（均值公式）**

$$
\text{mean} = x_t\Big(1+\frac{\sigma_{\mathrm{noise}}^2}{2\sigma}\Delta t\Big) + v_\theta\Big(1+\frac{\sigma_{\mathrm{noise}}^2(1-\sigma)}{2\sigma}\Big)\Delta t
$$

别被长式子吓到，用一个**"退火检查"**（sanity check）瞬间看穿它：把噪声关掉，即 $\sigma_{\mathrm{noise}}=0$，则两个 $(1+\cdots)$ 系数都变成 $1+0=1$，公式退化为：

$$
\text{mean} = x_t + v_\theta\,\Delta t
$$

——这正好是前面 Euler 步的公式！所以这条式子的本质就是"**Euler 步 + 噪声修正系数**"。那些多出来的 $(1+\frac{\sigma_{\mathrm{noise}}^2\cdots}{2\sigma})$ 是 SDE 框架里的"漂移修正"：直接往 Euler 步上加噪声会把轨迹的整体走向微微带偏，作者加了这些系数，让"加噪之后边缘分布仍然对齐"。系数具体长成什么样取决于噪声注入方式，我们记住结构即可：**mean = 走一点 + 加速一点，噪声只负责让落点有随机性。**

**采样（随机策略真正落地的地方）**

$$
x_{t+1} \sim \mathcal{N}\big(\text{mean},\; \sigma_{\mathrm{noise}}^2(-\Delta t)\,\mathbf{I}\big)
$$

翻译：以 mean 为中心、$\sigma_{\mathrm{noise}}^2(-\Delta t)$ 为散布，按**高斯分布（钟形曲线）**抽一个点——**大概率落在 mean 附近，偶尔偏出去一点**。每次采样都得到一条略有不同的轨迹，这就是策略的"随机性"来源。

**公式 ⑪ 从哪来：log_prob 就是"高斯 PDF 取对数"，不需要记公式**

这一步只需要认识一维高斯分布的概率密度函数（钟形曲线的公式）：

$$
p(z) = \frac{1}{\sigma\sqrt{2\pi}}\;\exp\Big(-\frac{(z-\mu)^2}{2\sigma^2}\Big)
$$

现在把我们的量代入：$z = x_{t+1}$，$\mu = \text{mean}$，标准差 $\sigma = \sigma_{\mathrm{noise}}\sqrt{-\Delta t}$（因为方差是 $\sigma_{\mathrm{noise}}^2(-\Delta t)$）。代入后对两边取 $\log$——$\log$ 会把"乘"变成"加"、把"指数"拉下来：

$$
\log p(x_{t+1}\mid x_t)
= \underbrace{-\frac{\|x_{t+1}-\text{mean}\|^2}{2\,\sigma_{\mathrm{noise}}^2(-\Delta t)}}_{\text{指数部分的 log}} - \underbrace{\log\big(\sigma_{\mathrm{noise}}\sqrt{-\Delta t}\big)}_{\text{前面的 } \frac{1}{\sigma}} - \underbrace{\log\sqrt{2\pi}}_{\text{常数}}
$$

逐项读：

- 第一项：$-\frac{(z-\mu)^2}{2\sigma^2}$ 展开 = "**离 mean 越远，越不像是随机抽出来的**"（偏离越离谱，概率越小）；
- 第二项：$-\log\sigma$ = "抖动越大，单点概率越小"（分布被摊开了）；
- 第三项：$-\log\sqrt{2\pi}$ = 高斯分布的归一化常数，保证所有概率加起来是 1，跟数据无关。

连起来 = "这个高斯落在 $x_{t+1}$ 处的对数概率"，正是训练阶段 PPO ratio $\exp(\log\pi_{\text{new}}-\log\pi_{\text{old}})$ 里用的 $\log\pi_\theta(x_{t+1}\mid x_t)$。

> **一句话串联（附 A）：** 纯 FM 教会 $v_\theta$ 速度（CFM 回归）→ 确定性 ODE 只能采样无梯度 → 给同一 $v_\theta$ 注入 $g(t)dW$ 变成 SDE → 单步是高斯的，$\log p$ 闭合可微 → 于是能算 PPO ratio 和 advantage，GRPO 才"有戏唱"。

---

### 附 B｜GRPO Advantage 图：组内归一化的数学

![GRPO Advantage 计算](/images/flowgrpo/flowgrpo_grpo_advantage.svg)

这张图回答一个问题：**同一个 prompt 生成了 $G$ 张图、各有各的分数，怎么把"分数"翻译成"谁该被表扬、谁该被批评"？** 答案就两步：先算平均和分散程度，再看每张图偏离平均几个"标准差"。

**① 平均分 $\mu$ 和标准差 $\sigma$（先认识这两个词）**

$$
\mu = \frac{1}{G}\sum_{i=1}^{G} r_i, \qquad
\sigma = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu)^2 + 10^{-4}}
$$

- $\mu$ = **平均分**：这组图的"普通水平"。公式就是"分数全加起来除以个数"（图里 `μ = mean(r₁⋯r_G)`）。
- $\sigma$ = **标准差**：分数有多"散"。先算每个分数离平均多远（$(r_i-\mu)^2$），把这些距离取平均，再开根号。$\sigma$ 大 = 分数高低落差大；$\sigma$ 小 = 大家分数都差不多（图里 `σ = std(r₁⋯r_G)`）。
- 那个 $+10^{-4}$：防止大家分数一模一样时 $\sigma=0$，除以 0 会炸，所以垫一个极小值。

**② 优势 $A_i$——"比平均好（差）几个标准差"**

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

- 分子 $r_i-\mu$：这张图比平均水平**高出多少分**。
- 再除以 $\sigma$：把这个分差换成"**几个标准差**"为单位——这叫 z-score（标准化），把不同打分量纲统一成"相对组内好坏"。
- 结论：$A_i>0$ = 比组内平均好；$A_i<0$ = 比组内平均差。
- 图里例子：$r=[0,0.12,0.45,0.78,0.91,1.00]$，$\mu=0.543$，$\sigma=0.395$，于是 $A=[-1.37,-1.07,-0.24,0.60,0.93,1.16]$——最后一张比平均好 1.16 个标准差。

**③ 它怎么驱动训练（把"表扬/批评"变成梯度）**

这条公式不需要重新记——它正是**附 A 公式⑦（策略梯度）**，只是把"奖励 $R$"换成"advantage $A_i$"：

$$
\nabla_\theta \mathcal{L} \;\supseteq\; -A_i\,\nabla_\theta \log\pi_\theta(\tau_i)
$$

（推导过程见附 A 第 4 步：期望奖励对 $\theta$ 求导 → log 求导技巧 → 蒙特卡洛近似。这里只需理解方向——）每条轨迹 $\tau_i$ 把自己的"概率的对数导数"$\nabla_\theta\log\pi_\theta(\tau_i)$ 按它的优势 $A_i$ 加权：

- $A_i>0$（这图比同组平均好）→ 梯度方向**抬高**这条轨迹的生成概率（表扬）；
- $A_i<0$（这图比平均差）→ 梯度方向**压低**这条轨迹的生成概率（批评）。

**④ 为什么 GRPO 不需要 Critic（价值网络）？**

- PPO 的 advantage 是 $A(s,a)=Q(s,a)-V(s)$，需要额外训练一个价值网络 $V$ 当"基线"，参数量≈策略网络，显存翻倍。
- GRPO 的妙招：**拿"同组平均 $\mu$"当基线、"同组标准差 $\sigma$"当尺度**——就像全班互相评分、按曲线给分（grading on a curve），不需要老师预先定标准，也就省掉了整套 critic。
- 代价：每组样本 $G$ 必须足够多，平均分 $\mu$ 和 $\sigma$ 才估得准（图里底部注释也提到采样量增大）。

---

### 附 C｜SDE 窗口图：滑动窗口 + 马尔可夫链的数学

![Flow-GRPO-Fast SDE 窗口](/images/flowgrpo/flowgrpo_fast_sde.svg)

这张图解决一个问题：**训练时"10 步全要存进计算图"，12B 的模型显存直接炸，怎么办？** 答案：只让其中几步"随机"，其余几步当普通 ODE 走完。

**① 先认识马尔可夫链（Markov chain）：每一步只记得上一步**

一条去噪轨迹 $x_0,x_1,\dots,x_N$（$N$=10 步）是一串前后相连的状态。**马尔可夫性**就是"记忆只有一步"：这一步只跟紧挨着的上一步有关，更早的不记得。类比接力赛：每位选手只从前面那位手里接棒，不关心第一个人是谁。

于是整条轨迹的概率 = 每步转移概率连乘。**这个连乘不是假设，是概率链式法则**：联合概率总可以先拆成"第一步的概率 × 已知第一步时第二步的条件概率 × …"（条件概率的定义 $P(A\cap B)=P(A)\,P(B\mid A)$ 反复套用）：

$$
p(x_0,\dots,x_N\mid c) = p(x_0)\prod_{k=1}^{N}p(x_k\mid x_{k-1},\;c)
$$

- $p(x_0)$：起点是噪声的概率（基本是常数）。
- $p(x_k\mid x_{k-1},c)$：已知上一步，这一步会走到哪的概率——正是附 A 那个**高斯核**。而"每步只依赖上一步"（马尔可夫性）则让这个式子从"每一步依赖前面所有步"简化成"只乘相邻两步的条件概率"。

理论上要对"整条轨迹"求梯度，这 $N$ 个高斯核的每一步都得存进计算图；但每步都过 12B 的 DiT，全存 = OOM。

**② 窗口化 = 只让部分步骤"随机"**

既然"概率"只来自**随机**的步骤，那就只保留一小段随机，其余走确定性 ODE：

- **窗口前（前几步）**：`noise_level=0`，退化为确定性 ODE——一步一个确定结果，不需要存中间状态（`torch.no_grad()`）。
- **窗口内（后 `window_size` 步）**：恢复噪声，走 SDE——随机、有 $\log p$，才需要建计算图记录梯度。
- **窗口后**：再回到 ODE 收尾。

于是损失只对窗口内（随机段）取期望，其余步骤对梯度没贡献：

$$
\mathcal{L} \approx -\sum_{k\in\,\text{window}} A\,\log p_\theta(x_k\mid x_{k-1},c)
$$

**③ 为什么梯度仍能回传（不会断）？**

- `backward()` 只展开窗口内那几步的计算图，所以**显存只存窗口内的 DiT 激活**（`window_size` 步，如 `geneval_flux_fast` 的 3 步 → 约 36B 激活 → ~22GB），窗口外步骤在 `no_grad` 下算完，只把最终 latent 递给窗口。
- 因为窗口内每步都**复用同一套权重 $v_\theta$**，由链式法则，窗口路径的梯度会正常累积到所有参数上——不会因为窗口小就"传不回去"。
- 真正的取舍：窗口外的步骤**不贡献梯度**（它们只负责把 $x_0$ 推进到窗口起点）。这是 Flow-GRPO-Fast 用"梯度近似"换"显存可行"的妥协。

---

## 🧭 总结：一图串起整个 Flow-GRPO

到这里，正文的"十步"、代码里的每个函数、附录里的每条公式都散落在各处。这一节把它们**串成一条完整的因果链**——从上到下读，每一步都是"上一步为什么必须存在"的答案：

![Flow-GRPO 完整逻辑链总结](/images/flowgrpo/flowgrpo_summary.svg)

### 沿着因果链走一遍（8 个节点）

| 节点 | 一句话 | 核心公式 | 为什么必须这么做 |
|------|--------|---------|-----------------|
| **① 学速度** | Flow Matching 训练让网络学会"往哪走" | $L=E\|v_\theta-(x_1-x_0)\|^2$ | 没有速度场就没有"生成"这回事 |
| **② ODE→SDE** | 为了算概率，给速度场加噪声 | $dx=v_\theta dt+g\,dW$ | 纯 ODE 无随机性 → 无 $\log\pi$ → 无梯度 |
| **③ 高斯 log p** | SDE 单步是高斯核，概率有闭式解 | $\log p=-\frac{\|x_{k+1}-\text{mean}\|^2}{2\sigma^2(-\Delta t)}-\log(\sigma\sqrt{-\Delta t})-\log\sqrt{2\pi}$ | PPO ratio 需要可微的逐跳概率 |
| **④ 轨迹概率** | 整条路概率 = 各步连乘 | $\pi_\theta(\tau)=\prod_k p(x_{k+1}\mid x_k)$ | 采样记 $\log\pi_{old}$，训练算 $\log\pi_{new}$ |
| **⑤ Advantage** | 同组归一化定好坏 | $A_i=(r_i-\mu)/\sigma$，clip 到 ±2 | 省掉 critic，把奖励翻译成梯度方向 |
| **⑥ 策略梯度** | 好事抬、坏事压 | $\nabla L=-E[A\,\nabla\log\pi_\theta]$ | 这是 GRPO 优化策略的本质动作 |
| **⑦ PPO + KL** | 稳住更新、防遗忘 | $\max(-Ar,\ -A\cdot\text{clip}(r))\ +\ \beta L^{KL}$ | reward 易 hack，必须限制每步步子大小 |
| **⑧ 闭环** | 回到采样，再来一个 epoch | 循环 ①~⑦ | on-policy：采样与训练始终用同一组参数 |

### 从三条主线理解

- **FM 生成（①②）**：先让模型学会"生成"（速度场），再为了强化学习"改造生成过程"（注入噪声得概率）。这一段的全部数学细节在**附录 A**。
- **对齐打分（③④⑤）**：把"生成的好坏"变成"可优化的信号"——高斯给出概率（③④），组内归一化给出 Advantage（⑤）。细节在**附录 A ⑤** 和**附录 B**。
- **策略优化（⑥⑦）**：用策略梯度抬好压坏（⑥），用 PPO clip + KL 保证稳定更新 LoRA（⑦）。细节在**正文第 5、6 步**。

### 三句话总串

> 1. **Flow Matching 先教会模型"从噪声走到图片的速度"**——这是生成能力的来源；
> 2. **为了强化学习，把确定性 ODE 变成随机 SDE**——得到可微的 $\log p$，配合组内 Advantage，就能算出"该抬哪条轨迹、压哪条轨迹"的策略梯度；
> 3. **用 PPO 的 clip 技巧和 KL 惩罚稳住每一步更新**，只更新 LoRA 参数，然后回到采样继续下一轮——这就是 Flow-GRPO 的全部逻辑闭环。

---

## 参考

- 代码库：`/root/workspace_qyl/Flow-GRPO/flow_grpo/`
- GRPO 论文：DeepSeekMath (arXiv:2402.03300)
- Flow Matching：Flow Matching for Generative Modeling (arXiv:2210.02747)
- PPO：Proximal Policy Optimization Algorithms (arXiv:1707.06347)
- DDPO：Denoising Diffusion Policy Optimization (arXiv:2307.09439)
