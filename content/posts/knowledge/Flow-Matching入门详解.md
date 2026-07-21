---
title: "Flow Matching 入门详解：从扩散模型到连续动作生成（含代码讲解）"
date: 2026-07-18
draft: false
categories: ["知识点拆解"]
tags: ["🌊 Flow Matching", "🎨 扩散模型", "📐 数学基础", "🤖 动作生成", "💻 代码讲解"]
summary: "Flow Matching 通过直接学习从噪声到数据的直线 ODE 路径，解决了扩散模型弯曲路径导致采样步数过多的问题。本文从数学直觉出发讲解向量场、ODE 与流的核心概念，并对比 CFM、Rectified Flow、OT路径等关键变体。与普通教程不同的是，本文所有代码均来自 ByteDance Bagel / Flow-GRPO 的真实项目实现，包含 CFG 推理、SDE 采样、GRPO 策略优化等工业级代码讲解。"
weight: 4
---

## 一句话理解 Flow Matching

![Flow Matching 核心概念：从噪声到数据的直线路径 —— ODE 速度场学习（来源：arXiv:2210.02747）](/images/flow-matching/fm_concept.png)

![Flow Matching vs Diffusion 路径对比：OT 路径直（左）、扩散路径弯（右）（来源：arXiv:2210.02747）](/images/flow-matching/ot_vs_diffusion.png)

> 上图对比了 Flow Matching（OT 路径）和 Diffusion（VP/VE 路径）的生成轨迹。Flow Matching 的直线路径意味着更少的采样步数和更稳定的训练。

> **Flow Matching = 学一个"风场"，让随机噪声顺着风飘到数据样本的位置。**

扩散模型（Diffusion）是先加噪再去噪，路径弯弯绕绕；**Flow Matching（流匹配）**则是直接学一条把噪声"搬"到数据的**直线路径**。路径越直，跑完这条路需要的步数就越少，训练也就越稳。这就是它近两年在图像生成、动作生成、轨迹规划里快速取代扩散头的根本原因。

形式化地说，Flow Matching 学习的是一个**速度场** $v_\theta(x,t)$，它告诉你在时刻 $t$、位置 $x$ 处，应该沿哪个方向、以多快的速度移动，才能从噪声分布 $p_0$（通常是高斯）"流"到目标数据分布 $p_1$。

---

## 🔄 从 Diffusion 说起：为什么需要 Flow Matching

要理解 Flow Matching，先得明白扩散模型"慢"在哪里。

**扩散模型的两条路径：**

- **前向过程**：把一张干净图像逐步加高斯噪声，最后变成纯噪声。这是一个**固定的、不学习的**马尔可夫链。
- **逆向过程**：训练一个网络 $v_\theta$（或预测噪声 $\epsilon_\theta$）去**反转**这个过程，从纯噪声一步步去噪回图像。

![Flow Matching 生成轨迹示例：噪声粒子沿直线路径逐步走向数据分布（来源：arXiv:2210.02747）](/images/flow-matching/trajectory.png)

**问题出在哪？** 扩散前向过程的轨迹是**弯曲的**——它对应一个非线性时变的随机微分方程（SDE）。逆向去噪必须用很小的步长才能跟上这条弯路，所以经典 DDPM 要采样 **1000 步**，DDIM 也要 20–50 步。即便后续有 DPM-Solver、一致性模型（Consistency Model）等各种加速技巧，**弯曲路径**这个根因始终没消除。

**Flow Matching 的洞察：** 既然路径弯曲是病根，那为什么不**直接构造一条直的路径**？一条直线只需要学一个**平滑的速度场**，推理时用大步长的 ODE solver，几步甚至一步就能走完。这就是 Flow Matching 的核心动机——它不是对扩散的修补，而是**从更一般的视角重新定义生成过程**，扩散只是它的一个特例。

---

## 📐 数学直觉：向量场、ODE 与"流"

这一节是 Flow Matching 的数学核心，但我们会用**最直白的方式**讲。

### 什么是向量场？

想象一张地图，每个城市上空都插着一个**小箭头**，告诉你"如果从这里出发，该往哪个方向、以多大速度走"。这些布满空间的箭头集合，就叫**向量场**。

在 Flow Matching 里，向量场 $v(x,t)$ 是**依赖于时间 $t$** 的：同一个位置，在 $t=0$ 和 $t=0.5$ 的箭头方向可能不同。它的物理含义是——"流"在演化过程中，风向是会变的。

### 什么是 ODE 与"流"？

有了向量场，我们就能写出一个**常微分方程（ODE）**，描述一个粒子如何随时间运动：

$$\frac{dx}{dt} = v_\theta(x,t), \quad x(0) \sim \mathcal{N}(0,I)$$

这条公式的意思是：**"粒子在每一瞬间的速度，等于向量场在当前位置和当前时刻的取值"**。从 $t=0$ 的噪声点出发，顺着这个 ODE 一路积分到 $t=1$，粒子到达的位置就构成一个**样本**。这个由向量场定义的、把分布从 $p_0$ 推到 $p_1$ 的演化过程，就叫**流（Flow）**。

**用最通俗的话说：** 训练阶段，网络在学"风怎么吹"；推理阶段，随机撒一把噪声粒子，让它们顺着风飘，飘到 $t=1$ 时聚集成的形状就是数据分布。

### Flow Matching 的训练目标

核心问题是：**怎么训练这个向量场 $v_\theta$？** Flow Matching 的答案是——**匹配**。构造一个"理想的风场" $u_t(x)$（它把噪声沿直线路径推到数据），然后让网络去**逼近**它：

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t,\,x_t}\left[\,\big\|\,v_\theta(x_t,t) - u_t(x_t)\,\big\|^2\,\right]$$

其中：

- $t \sim \mathcal{U}(0,1)$ 是随机采样的时刻
- $x_t = (1-t)\,x_0 + t\,x_1$ 是噪声 $x_0$ 和数据 $x_1$ 之间的**线性插值**（这就是"直线路径"的来源）
- $u_t(x_t) = x_1 - x_0$ 是沿这条直线的**恒定速度**

![Flow Matching 的线性插值路径：噪声 x₀ 到数据 x₁ 的直线概率路径（来源：arXiv:2210.02747）](/images/flow-matching/interpolation.png)

换句话说，**理想的"风向"就是"从当前插值点指向数据点"的方向**，网络只要学会预测这个方向即可。这个形式简单到令人惊讶——没有马尔可夫链，没有 SDE，没有繁琐的噪声调度，就是一个回归。

---

## 🆚 Flow Matching vs Diffusion

两者都属于**基于分数/速度场的生成模型**，但建模哲学截然不同。

| 维度 | 扩散模型（Diffusion） | Flow Matching |
|------|----------------------|---------------|
| **前向路径形状** | 弯曲（非线性噪声调度） | **直线**（线性插值） |
| **底层方程** | SDE（含随机项） | **ODE（确定性）** |
| **采样步数** | DDPM 1000 步，DDIM 20–50 步 | **几步到十几步**即可 |
| **训练目标** | 预测噪声 $\epsilon$ 或分数 $\nabla\log p$ | 预测速度场 $v$ |
| **路径可设计性** | 固定的加噪过程 | **任意可设计**（条件流） |
| **理论统一性** | Flow Matching 的一个特例 | **更一般的框架** |
| **训练稳定性** | 一般 | 通常**更稳**（目标更平滑） |
| **多模态支持** | ✅ | ✅（天然支持） |

**三个关键差异的深入解读：**

- **路径直 vs 弯** 是最本质的差别。直线路径意味着 ODE 的曲率小，数值积分可以用大步长——这就是 Flow Matching 能用 5–10 步采样的根本原因，而 Diffusion 想做到同样质量往往要更多步。
- **确定性 vs 随机性** 也带来工程影响。Flow Matching 的推理是一条确定性 ODE，容易缓存、容易蒸馏（把多步蒸馏成一步）；扩散的 SDE 有随机项，控制起来更复杂。
- **统一视角** 在理论上很优雅：当 Flow Matching 的路径取特定的非线性形式时，它就退化为扩散模型。所以学界常说"**Diffusion 是 Flow Matching 的特例**"。

> 💡 一句话记忆：**Diffusion 是"弯路慢走"，Flow Matching 是"直路快跑"。**

---

## 🔀 主要变体：OT-CFM 等

Flow Matching 的一个强大之处是**路径和边际分布可以自由设计**，由此衍生出多个变体。

| 变体 | 全称 | 核心改进 | 作用 |
|------|------|---------|------|
| **FM** | Flow Matching | 最早的原始形式，固定边缘分布 | 奠基 |
| **CFM** | Conditional Flow Matching | 以单个数据点为条件构造直线路径 | 让训练**可计算**，无需知道真实边缘分布 |
| **OT-CFM** | Optimal Transport CFM | 用**最优传输**对 $x_0$ 和 $x_1$ 配对 | 路径更直、训练更快更稳 |
| **Rectified Flow** | 整流流 | 迭代地"拉直"路径（多次 rectify） | 路径接近理想直线，支持一步生成 |
| **Stochastic FM** | 随机流匹配 | 在 ODE 里加回少量随机项 | 兼顾稳定性和样本多样性 |

**最常用的是 OT-CFM。** 它的关键洞察是：如果噪声 $x_0$ 和数据 $x_1$ 是**随机配对**的，路径不一定是最优的；但如果用**最优传输（Optimal Transport）**来配对（本质是找一个最小代价的一一映射），路径会尽可能短而直。形象比喻：**随机配对像是让每个人去随机一辆出租车，OT 配对像是全局调度让每个人上最近的车**。后者路径短、不绕路，训练时梯度方向也更一致。

**Rectified Flow**（Liu et al.）走了另一条路：先用 CFM 训一个模型，把学到的轨迹"重新拉直"成更直的直线，再训一遍——反复整流几次，路径越来越直，最终可以做到**一步生成**。Stable Diffusion 3 就采用了 Rectified Flow 的思路。

---

## 💻 真实项目代码讲解：Bagel / Flow-GRPO

> 下面所有代码节选自 [ByteDance-Seed/BAGEL](https://github.com/ByteDance-Seed/BAGEL)（一个全模态理解+生成的 VLM）以及 Flow-GRPO 扩展。这是目前最完整的**将 Flow Matching 融入 LLM 做生成的工业级实现**，比伪代码有营养得多。

### 模型架构概览

Bagel 模型是一个"通才"——它用一个 Qwen2 LLM 做骨干，同时支持**视觉理解**（SIGLIP ViT 编码图像）和**视觉生成**（Flow Matching 解码图像）。生成部分的核心组件：

```
输入: tokenized text + latent patches (来自 VAE)
        │
        ├─ text → embedding (Qwen2 embed_tokens)
        ├─ latent patches → vae2llm Linear + timestep_embed + pos_embed
        │
        └─ packed_sequence → Qwen2 LLM (融合理解与生成的 token 序列)
                │
                └─ llm2vae Linear → 预测速度场 v_t
```

关键配置（`BagelConfig`）：

```python
class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,        # 开启 Flow Matching 生成
        visual_und=True,        # 开启视觉理解
        llm_config=None,        # Qwen2 config
        vit_config=None,        # SIGLIP config
        vae_config=None,        # VAE config（输出 latent）
        latent_patch_size=2,    # 每个 latent patch 的尺寸
        max_latent_size=32,     # 最大 latent 网格
        timestep_shift=1.0,     # timestep 偏移（推理时常用 3.0）
        ...
    )
```

### 训练阶段：真正的 Flow Matching 前向

训练时，模型接收 VAE 编码后的 latent、文本 token、以及随机 timestep，经过 LLM 后预测速度场。代码如下（`Bagel.forward` 视觉生成部分）：

```python
# -----------------------------------------------------------
# Step 1: 从 VAE latent 构造干净的 packed_latent
# -----------------------------------------------------------
p = self.latent_patch_size          # 通常为 2
packed_latent = []
for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
    # VAE latent shape: [C, H, W] → reshape to patches
    # 把 latent 切成 p×p 的 patch，每个 patch 展平成向量
    latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
    packed_latent.append(latent)
packed_latent_clean = torch.cat(packed_latent, dim=0)

# -----------------------------------------------------------
# Step 2: 采样噪声 + 线性插值（CFM 核心）
# -----------------------------------------------------------
noise = torch.randn_like(packed_latent_clean)          # x₀: 高斯噪声
packed_timesteps = torch.sigmoid(packed_timesteps)     # t: 0→1 的 timestep
# timestep shift: 让模型在 t 接近 0 时"看得更细"
packed_timesteps = self.timestep_shift * packed_timesteps / \
                   (1 + (self.timestep_shift - 1) * packed_timesteps)
# x_t = (1-t) * x₀ + t * x₁  直线插值
packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean \
                + packed_timesteps[:, None] * noise

# -----------------------------------------------------------
# Step 3: 把 latent patch 映射到 LLM 隐空间 + 加 timestep/位置编码
# -----------------------------------------------------------
packed_timestep_embeds = self.time_embedder(packed_timesteps)
latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
# vae2llm: Linear 把 patch 投影到 hidden_size
packed_latent = self.vae2llm(packed_latent) \
                + packed_timestep_embeds + latent_token_pos_emb
# 插入到 packed_sequence 的对应位置（与文本 token 拼接）
packed_sequence[packed_vae_token_indexes] = packed_latent

# -----------------------------------------------------------
# Step 4: 过 LLM backbone，得到隐状态
# -----------------------------------------------------------
last_hidden_state = self.language_model(packed_sequence=packed_sequence, ...)

# -----------------------------------------------------------
# Step 5: 预测速度场 + MSE loss
# -----------------------------------------------------------
packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
# 注意这里的 target 方向：x₁ - x₀  即"从数据指向噪声"
# （与论文中 u_t = x₁ - x₀ 一致，符号取决于路径定义方向）
target = noise - packed_latent_clean
has_mse = packed_timesteps > 0
mse = (packed_mse_preds - target[has_mse]) ** 2
```

**和伪代码的区别：** 真实代码里多了**timestep shift**（把 timestep 重新分布）和**patchify**（把 dense latent 切成 patch 序列以适配 LLM 的 token 输入格式）。核心逻辑 `(1-t)*x₀ + t*x₁` 和 MSE loss 是完全一致的。

### Timestep 编码：正弦位置嵌入

Flow Matching 的 $t$ 是连续浮点数，需要编码成向量才能送入网络。真实实现使用了 DiT 式正弦编码：

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)          # 256 → hidden_size
        return t_emb
```

`timestep_shift` 是一个容易被忽视但很重要的技巧。原始的 timestep $t$ 均匀分布在 $[0,1]$，但实际生成时 $t$ 靠近 0 的步（细节塑造）比靠近 1 的步（轮廓布局）需要更多分辨率。shift 公式 `shift * t / (1 + (shift-1) * t)` 会**把更多步压缩到细节区域**，`shift=3.0` 是常见的经验值。

### 推理阶段（1）：ODE 求解 + SDE 噪声注入

Flow Matching 推理的金标准不是纯 ODE 而是**带噪声注入的 SDE**——每一步在预测方向上加适量随机性，兼顾确定性和多样性。核心 step 实现：

```python
def _sde_step_with_logprob(self, model_output, timestep, prev_timestep,
                           d_timestep, sample, prev_sample=None,
                           noise_level=0.8):
    model_output = model_output.float()
    sample = sample.float()

    # 计算当前步的噪声标准差
    # std_dev_t = sqrt(t / (1 - t')) * noise_level
    # 其中 t' = max(sigma_max, t) 防止除零
    std_dev_t = torch.sqrt(
        timestep / (1 - torch.where(timestep == 1, sigma_max, timestep))
    ) * noise_level

    # 预测下一步均值（Euler 步 + 噪声修正项）
    #   x_{t-1} = x_t*(1 + σ²/(2t)*dt) + v_t*(1 + σ²*(1-t)/(2t))*dt
    # 当 noise_level=0 时退化为纯 Euler: x_{t-1} = x_t + v_t * dt
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * timestep) * d_timestep)
        + model_output * (1 + std_dev_t**2 * (1 - timestep) / (2 * timestep)) * d_timestep
    )

    if prev_sample is None:
        variance_noise = randn_tensor(model_output.shape, ...)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-d_timestep) * variance_noise

    # log_prob 用于 RL 策略梯度（Flow-GRPO 需要）
    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) \
               / (2 * (std_dev_t * torch.sqrt(-d_timestep))**2)
    log_prob = log_prob.mean()

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

理解这个代码的三个层次：
- `noise_level=0` → 纯 **Euler ODE**，$x_{t-1} = x_t + v_t \cdot dt$
- `noise_level>0` → 加噪声的 **SDE**，$x_{t-1} = \text{mean} + \sigma \cdot \epsilon$
- `log_prob` 计算了每一步 transition 的高斯 log-probability，**这是 Flow-GRPO 能做 RL 的关键**——没有它就没法算策略梯度

### 推理阶段（2）：完整采样循环 + CFG

完整的采样循环从 $t=1$ 的纯噪声积分到 $t=0$ 的生成结果，中间可以选择性使用 **Classifier-Free Guidance (CFG)** 提升条件控制强度。

```python
def generate_image(self, ...):
    x_t = packed_init_noises                      # 起点：纯噪声

    # 准备离散 timestep 序列（从 1 → 0）
    timesteps = torch.linspace(1, 0, num_timesteps)  # 例如 50 步
    timesteps = timestep_shift * timesteps / \
                (1 + (timestep_shift - 1) * timesteps)
    dts = timesteps[1:] - timesteps[:-1]

    for i, t in enumerate(timesteps[:-1]):
        # -----------------------------------------------------------
        # 预测速度场（含可选的 CFG）
        # -----------------------------------------------------------
        v_t = self._forward_flow(
            x_t=x_t, timestep=torch.tensor([t]),
            cfg_text_scale=cfg_text_scale,   # 文本 CFG 强度
            cfg_img_scale=cfg_img_scale,     # 图像 CFG 强度
            ...
        )
        # -----------------------------------------------------------
        # SDE step（含可选的噪声注入）
        # -----------------------------------------------------------
        x_t, log_prob, _, _ = self._sde_step_with_logprob(
            v_t, timesteps[i], timesteps[i+1], dts[i], x_t,
            noise_level=cur_noise_level
        )

    return x_t  # 生成结果
```

CFG 在 `_forward_flow` 中实现（简化版）：

```python
def _forward_flow(self, x_t, timestep, ..., cfg_text_scale, cfg_img_scale):
    # 1) 条件前向（有 text/image 条件）
    v_t = self.llm2vae(llm_output)

    # 2) 无条件前向（text CFG：用空文本做条件）
    if cfg_text_scale > 1.0:
        cfg_text_v_t = self.llm2vae(cfg_text_llm_output)

    # 3) 图像无条件前向（image CFG）
    if cfg_img_scale > 1.0:
        cfg_img_v_t = self.llm2vae(cfg_img_llm_output)

    # 4) CFG 插值 + renormalization（防止 CFG 导致向量范数爆炸）
    #    v = v_uncond + scale * (v_cond - v_uncond)
    v_t = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
    v_t = v_t * (norm(v_t) / norm(v_t_))  # CFG-Renorm 保持速度尺度

    return v_t
```

**CFG 的核心：** 同时推理有条件和无条件两个分支，然后外推 `v = v_uncond + s * (v_cond - v_uncond)`。$s>1$ 时模型会更"卖力"地满足条件，但同时速度场范数会膨胀，所以需要 renormalization 把速度尺度拉回合理范围。

### 推理超参数速查

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `num_timesteps` | 采样步数 | 10–50（步数越少越快，但质量下降） |
| `cfg_text_scale` | 文本条件强度 | 4.0–8.0（1.0 = 关闭） |
| `cfg_img_scale` | 图像条件强度（编辑/图生图） | 1.0–2.0 |
| `cfg_interval` | CFG 生效的 t 范围 | [0, 1] 全程或 [0.4, 1] 后半段 |
| `timestep_shift` | 步数分布偏移 | 3.0（t→0 细节区域分配更多步） |
| `noise_level` | SDE 噪声强度 | 0.7–0.8（0 = 纯 ODE） |

---

## 🚗 自动驾驶中的应用：多模态轨迹生成

### 为什么轨迹规划需要多模态？

这是理解 Flow Matching 在自动驾驶价值的起点。考虑一个场景：**前方有静止障碍物，左车道空、右车道也空。** 此时合理的选择有至少两种——向左变道、向右变道。如果用普通的回归头输出一条"平均轨迹"，结果往往是**压着障碍物正中间走**（mode averaging），这比任何一个合理选择都更危险。

多模态意味着输出分布有**多个峰**（多个合理轨迹），模型需要能表达"这两种选择都合理"，而不是强行取平均。这正是生成式模型（扩散、Flow Matching）的强项——它们输出的是一个**分布**，采样多次能得到不同的合理轨迹。

### 代表工作：DiffusionDrive 与 Flow 头规划

| 工作 | 团队 | 动作头 | 核心特点 |
|------|------|--------|---------|
| **DiffusionDrive** | 华中科大等 | 扩散头 | 把规划建模成条件去噪，输出多模态轨迹分布 |
| **CTG++** | CMU | 条件扩散 | 用代价函数引导扩散，融合规则约束 |
| **GameFormer** | 上交等 | 迭代预测 | 多智能体博弈下的层次化轨迹预测 |
| **Flow-based Planner** | 多家 | **Flow Matching 头** | 用直线流路径替代扩散，**采样更快** |

**为什么 Flow Matching 在规划里比 Diffusion 更有吸引力？** 答案还是**速度**。规划在车端是高频任务（10 Hz 以上），扩散头的多步去噪常常是延迟瓶颈；Flow Matching 的几步 ODE 采样能让规划耗时压到一个量级的提升。此外，规划对**多模态**和**可控性**都有强需求——前者通过采样多个初始噪声得到多条轨迹，后者通过条件注入（自车状态、地图、障碍）控制生成。

### 一个具体的轨迹生成例子

假设要生成未来 3 秒的轨迹（30 个时间点 × 2 维坐标，共 60 维向量）：

1. **噪声 $x_0$**：从 60 维高斯分布采样
2. **条件 $c$**：当前自车状态、HDMap、周围障碍物编码成特征
3. **训练**：让模型学速度场 $v_\theta(x_t, t, c)$，目标是把噪声沿直线推到真实轨迹 $x_1$
4. **推理**：用 5 步 Euler 法积分，得到一条候选轨迹；采样多个 $x_0$ 得到**多条候选轨迹**，再用代价函数（碰撞、舒适度、合规）挑最优的一条下发

整个流程清晰、高效、天然多模态。

---

## 🤖 在 VLA 中的应用：作为 Action Head

在 VLA（Vision-Language-Action）模型里，Action Head 是把 LLM 的推理结果变成**连续动作**的关键。Flow Matching 正在成为**最强 Action Head** 的代名词，代表就是 **π0**。

### π0：Flow Matching 动作头的标杆

Physical Intelligence 的 **π0** 把 Flow Matching 用作通用机器人的动作生成头，是目前 VLA 能力的天花板之一。它的设计要点：

- **骨干**：PaLI/Gemma 风格的 VLM 做视觉-语言理解
- **动作头**：一个**Flow Matching** 网络，以 VLM 的隐状态为条件，从噪声生成一段**动作序列（action chunk）**
- **训练**：用大规模示教数据（含跨本体、跨任务）训练，损失就是上文那个简单的 MSE 流匹配损失
- **推理**：用 ODE solver 生成一段未来 N 步动作，按需执行（action chunking）

**为什么 π0 选 Flow Matching 而不是 Diffusion 或回归？**

- 比回归强：机器人动作**高度多模态**（同一个"把杯子放到架子上"有多种抓取和放置路径），回归会 mode averaging
- 比扩散快：机器人控制需要 10–50 Hz，Flow Matching 的少步采样对实时性更友好
- 比扩散稳：Flow Matching 的训练目标更平滑，大规模数据上收敛更可靠

### VLA Action Head 三大流派回顾

| 方案 | 精度 | 速度 | 多模态 | 代表 |
|------|------|------|--------|------|
| 离散化 Token | 中 | 慢 | 弱 | RT-2、OpenVLA |
| 连续回归 | 中 | **快** | 弱 | 早期 SFT 头 |
| **Flow Matching** | **高** | 较快（可优化） | **强** | **π0、π0.5** |

Flow Matching 正在成为高精度 VLA 动作头的事实标准。

---

## 🔗 与 GRPO 的结合：Flow-GRPO 的真实训练代码

Flow-GRPO 是 ByteDance Bagel 项目的一个扩展，把 Flow Matching 的图像生成和 GRPO 策略优化结合起来。我们直接从实际训练代码中拆解其核心逻辑。

### 核心挑战：Flow Matching 策略的 log-prob 从哪来？

GRPO 需要 $\log p_\theta(a)$ 计算 importance ratio。Flow Matching 的策略分布通过 ODE 定义，没有显式密度。但**SDE 的每一步 transition 是高斯分布**，因此整个轨迹的 log-prob 就是各步 log-prob 之和。

在 `_sde_step_with_logprob` 中，我们早就埋好了 log-prob 计算：

```python
# transition: x_{t+1} ~ N(mean_t, σ_t²)
# log p(x_{t+1} | x_t, θ) = -(x_{t+1} - mean_t)² / (2σ_t²)
log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) \
           / (2 * (std_dev_t * torch.sqrt(-d_timestep))**2)
log_prob = log_prob.mean()
```

**关键洞察**：Flow Matching + SDE 的每一步 transition 恰好是高斯分布，所以 log-prob 有**闭式解**，不需要复杂的瞬时变量替换公式。每一步的 log_prob 累加就是整条轨迹的 $\log p_\theta(\text{轨迹})$。

### Flow-GRPO 训练循环

训练时，模型先生成一组候选图像（多条轨迹），用奖励模型打分，然后做 GRPO 策略更新。

#### Step 1: 采样生成 + 保存中间 latent

```python
# generate_image() 中保存每一步的 latent 和 log_prob
all_latents = []     # 每一步的 latent
all_log_probs = []   # 每一步的对数概率
all_timesteps = []   # 对应的时间步

for i, t in enumerate(timesteps):
    v_t = self._forward_flow(x_t, ...)
    x_t, log_prob, _, _ = self._sde_step_with_logprob(v_t, ...)
    # 在指定窗口内保存中间结果用于 RL 训练
    if i >= sde_timestep_begin and i < sde_timestep_begin + window_size:
        all_latents.append(x_t)
        all_log_probs.append(log_prob)
        all_timesteps.append(t)
```

生成完成后，把图像送入奖励模型得到标量奖励 $r$，同 prompt 的 $G$ 条轨迹做组内归一化得到 advantage $A^{(i)} = \frac{r^{(i)} - \bar{r}}{\sigma_r}$。

#### Step 2: 逐 timestep 做 PPO-style 策略更新

```python
def generate_image_learn(self, sample, grpo_config, accelerator, optimizer, ...):
    latents = sample["latents"]           # 采样时保存的 latent 序列
    prev_latents = sample["prev_latents"] # 上一步的 latent
    timesteps = sample["timesteps"]
    advantages = torch.clamp(
        sample["advantages"],
        -grpo_config.train.adv_clip_max,
        grpo_config.train.adv_clip_max,
    )

    for i, t in enumerate(timesteps):
        with accelerator.accumulate(transformer):
            # 用当前策略预测速度场（online）
            v_t = self._forward_flow(x_t=latents[i], timestep=t, ...)

            # 计算 log_prob（含重参数化）
            _, log_prob, prev_sample_mean, std_dev_t = self._sde_step_with_logprob(
                v_t, timesteps[i], timesteps[i+1], dts[i],
                latents[i], prev_sample=prev_latents[i], ...
            )

            # KL 正则：对 reference model 也做一步推理
            if grpo_config.train.beta > 0:
                v_t_ref = self._forward_flow(..., ref_model=True)
                _, _, prev_sample_mean_ref, _ = self._sde_step_with_logprob(
                    v_t_ref, ...
                )

            # -----------------------------------------------------------
            # GRPO 策略梯度（核心 4 行）
            # -----------------------------------------------------------
            ratio = torch.exp(log_prob - sample["log_probs"][i])
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - grpo_config.train.clip_range_lt,
                1.0 + grpo_config.train.clip_range_gt,
            )
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

            # KL 散度（高斯分布的 KL 有闭式解）
            if grpo_config.train.beta > 0:
                kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() \
                          / (2 * std_dev_t ** 2)
                loss = policy_loss + grpo_config.train.beta * kl_loss
            else:
                loss = policy_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
```

### 代码对应的 GRPO 公式

上面的核心 4 行代码对应 GRPO 的**裁剪 surrogate 目标**：

$$L(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = e^{\log p_\theta - \log p_{\theta_{\text{old}}}}$，$A_t$ 是组内归一化优势。

**和 PPO 的区别：** GRPO 不用 critic network（价值网络），优势直接从组内奖励归一化得到 $$\hat{A}_i = \frac{r_i - \mu_\text{group}}{\sigma_\text{group}}$$。这个简化对 Flow Matching 尤其友好——不用额外训一个价值网络去逼近连续动作空间的价值函数。

### 为什么 Flow + RL 特别契合自动驾驶？

| 契合点 | 说明 |
|--------|------|
| **多模态探索** | Flow 采样天然给出多条不同轨迹，正是 RL 探索所需的样本多样性 |
| **连续动作平滑** | ODE 生成的动作平滑连续，比离散 token 更适合车辆控制 |
| **奖励稀疏可处理** | GRPO 的组内比较把"绝对奖励"变"相对优势"，缓解驾驶奖励极度稀疏的问题 |
| **世界模型协同** | 可用世界模型做 rollout 评估，无需真实路测，安全且低成本 |

> 💡 一句话理解 Flow-GRPO：**用 Flow Matching 的 SDE 采样提供多模态候选轨迹 + 高斯 log-prob 闭式解，用 GRPO 的组内归一化做免价值网络策略优化。**

---

## ⚖️ 什么时候该用 Flow Matching？

| 场景 | 推荐度 | 理由 |
|------|--------|------|
| 多模态动作/轨迹生成 | ⭐⭐⭐⭐⭐ | 天然支持多峰分布 |
| 实时性要求高的生成 | ⭐⭐⭐⭐⭐ | 少步采样，速度远胜扩散 |
| 单模态精确回归 | ⭐⭐ | 杀鸡用牛刀，普通回归头更快 |
| 离散决策（如左转/右转） | ⭐ | 适合分类，不必用生成模型 |
| 大规模图像/视频生成 | ⭐⭐⭐⭐ | SD3、Meta MovieGen 都在用 |
| 高频控制（机器人/车端） | ⭐⭐⭐⭐⭐ | 少步采样 + action chunking 是当前最优解 |

---

## ✅ 小结

记住这三个要点，就能抓住 Flow Matching 的精髓：

1. **本质** = 学习一个**速度场** $v_\theta(x,t)$，通过 ODE 把噪声分布"流"到数据分布，**路径是直线**。
2. **优势** = 路径直 → **采样步数少**；目标平滑 → **训练稳定**；理论上是扩散的更一般框架。
3. **落地** = 在自动驾驶做**多模态轨迹生成**（DiffusionDrive 等）、在 VLA 做**连续动作头**（π0）、与 **GRPO** 结合做偏好对齐（Flow-GRPO）。

一句话总结：**Flow Matching 把生成式建模从"弯路慢走"升级为"直路快跑"**，正在成为连续动作生成的事实标准，也是连接"模仿学习"和"强化学习"的关键桥梁。

---

## 📚 延伸阅读

**奠基论文：**

- **Flow Matching for Generative Modeling**（Lipman et al., ICLR 2023）—— Flow Matching 原始论文
- **Stochastic Interpolants**（Albergo & Vanden-Eijnden, 2023）—— 同期独立工作，与 FM 等价
- **Flow Straight and Fast: Rectified Flow**（Liu et al., ICLR 2023）—— 整流流，路径拉直
- **Optimal Transport CFM**（Tong et al., 2023）—— OT-CFM 变体

**应用论文：**

- **π0 / π0.5**（Physical Intelligence, 2024）—— Flow Matching VLA 标杆
- **Diffusion Policy**（Chi et al., 2023）—— 扩散动作头奠基，理解 FM 动作头的基础
- **DiffusionDrive**（华中科大等）—— 扩散头轨迹规划
- **Stable Diffusion 3**（Stability AI, 2024）—— Rectified Flow 用于图像生成

**代码仓库（本文的代码来源）：**

- **ByteDance-Seed/BAGEL**（https://github.com/ByteDance-Seed/BAGEL）—— 全模态 VLM，含 Flow Matching 图像生成的完整工业级实现
- **Flow-GRPO**（https://github.com/anomalyco/Flow-GRPO）—— 在 Bagel 基础上扩展 GRPO 策略优化，支持 RL 训练

**博客与教程：**

- Lily Yang 的 *Flow Matching for Generative Modeling* 教程（直观图解）
- **torchcfm**（https://github.com/atong01/conditional-flow-matching）—— 官方 CFM 最小实现
- HuggingFace *Diffusers* 库已原生支持 Flow Matching / Rectified Flow

> 💡 新手建议：先读 torchcfm 的最小示例（100 行就能跑通），然后在 Bagel 仓库里看实际的 bagel.py forward 函数，再回来看本文的代码讲解，会非常通透。

---

*💡 觉得有用？这是「知识点拆解」系列的第 4 篇，后续会继续讲强化学习（GRPO）和世界模型如何与这些生成模型协同。点个关注不迷路。*
