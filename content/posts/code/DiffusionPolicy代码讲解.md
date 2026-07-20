---
title: "Diffusion Policy 代码讲解：用 DDPM 给机器人「生成」一连串动作"
date: 2026-07-20
description: "拆解 real-stanford/diffusion_policy：ConditionalUnet1D 去噪网络、观察/动作视野、LinearNormalizer 归一化、Workspace/Dataset/Policy 解耦，看通用扩散动作策略如何落地"
tags: ["扩散模型", "DDPM", "机器人", "代码讲解"]
categories: ["代码讲解"]
summary: "「Diffusion Policy 把『给一段观察、出一段动作序列』做成条件扩散生成：噪声动作被 ConditionalUnet1D 逐步去噪，训练是 DDPM 标准损失，推理是 DDIM 多步采样，本文顺着 real-stanford 官方仓库讲清它的工程解耦」"
---

> 本文是「代码讲解」路线的第 7 篇，也是「扩散用于动作」的通用基线（RSS 2023，real-stanford）。它不特指自动驾驶，但是 DriveVLA-W0 / DiffusionDrive / Diffusion Planner 的方法论源头。代码在 **real-stanford/diffusion_policy**。

## 0. 名词速查

- **DDPM / DDIM**：去噪扩散概率模型。训练时往干净数据加噪，学「预测噪声」；推理时从纯噪声逐步去噪采样。DDIM 是确定性快速采样变体。
- **观察视野 To / 动作视野 Ta**：模型看过去 To 步观测，输出未来 Ta 步动作（典型 To=2, Ta=16），多步动作比单步更稳。
- **LinearNormalizer**：对观测和动作做线性（均值/标准差或 min/max）归一化，扩散对量纲极敏感，归一化是必做项。
- **recurrent / CNN / Transformer backbone**：Diffusion Policy 支持三种观测编码器，本文以 CNN 版（图像）为例。

## 1. 仓库解耦设计

Diffusion Policy 最值得学的是**模块拆分**：

- `diffusion_policy/`
  - `policy/`
    - `diffusion_policy.py`：总装，把 obs 编码 + 去噪网络 + 采样器串起来
    - `diffusion_unet1d_policy.py`：策略入口（对外 API）
    - `model/`
      - `noise_unet.py`：ConditionalUnet1D 定义
      - `diffusion.py`：DDPM / DDIM 调度
  - `model/`
    - `cfg_scheduler.py`：噪声调度（cosine / linear）
    - `obs_encoder.py`：图像/状态编码器
  - `dataset/`
    - `base_dataset.py`：拼 (obs, action) 对，做归一化
  - `workspace/`
    - `train_diffusion_policy.py`：训练循环 + 日志 + checkpoint
  - `configs/`
    - `train_diffusion_unet_...yaml`：所有超参在 yaml

要点：**模型、数据、训练循环三者完全解耦**，换任务只改 yaml 和 dataset，不动网络。

## 2. 核心网络：ConditionalUnet1D

动作是一维序列（Ta 步），所以用 **1D U-Net**（不是图像用的 2D U-Net）。它把噪声动作 + 时间步 + 观察特征 当条件输入。

```python
# model/noise_unet.py（结构简化）
class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=(256,512,1024)):
        # 下采样 1D conv 块
        self.down_blocks = nn.ModuleList([...])
        self.mid_block = ...
        self.up_blocks = nn.ModuleList([...])
        # 时间步与全局条件各自编码后叠加
        self.time_emb = nn.Sequential(nn.Linear(256,256), nn.SiLU())
        self.cond_obs = nn.Linear(global_cond_dim, 256)

    def forward(self, sample, timestep, global_cond):
        # sample: (B, Ta, action_dim) 噪声动作
        t = self.time_emb(timestep_embed(timestep))
        c = self.cond_obs(global_cond)        # 观察编码
        h = sample.permute(0,2,1)             # (B, D, Ta)
        for down in self.down_blocks:
            h = down(h) + c[:, :, None]
        ...
        return h.permute(0,2,1)               # 预测的噪声 (B, Ta, action_dim)
```

关键：观察 `global_cond` 通过 FiLM 式加偏置注入每一层，实现「条件扩散」。

## 3. 训练：标准 DDPM 损失

```python
# diffusion.py 中训练一步（简化）
def compute_loss(self, model, obs, action):
    noise = torch.randn_like(action)               # 纯噪声
    timestep = torch.randint(0, T, (B,))
    noisy = self.q_sample(action, noise, timestep) # 加噪到 t 步
    pred_noise = model(noisy, timestep, cond=obs)  # 网络预测噪声
    loss = F.mse_loss(pred_noise, noise)           # 噪声回归
    return loss
```

注意是**预测噪声**（ε-prediction），和 DDPM 原论文一致。训练时随机采样 timestep，对所有动作步联合回归。

## 4. 推理：DDIM 多步采样

```python
# policy/diffusion_unet1d_policy.py
def predict_action(self, obs):
    obs_cond = self.obs_encoder(obs)               # 编码观察
    sample = torch.randn((B, Ta, action_dim))      # 从纯噪声起步
    for t in self.scheduler.step_timesteps:        # DDIM 反向
        noise_pred = self.model(sample, t, obs_cond)
        sample = self.scheduler.step(noise_pred, t, sample)
    return sample                                  # 去噪后的动作序列
```

默认用 DDIM 20~100 步，比 DDPM 快很多。输出整段 Ta 步动作，执行时通常只取前几步（recursive replanning）。

## 5. 归一化：最容易踩的坑

```python
# dataset/base_dataset.py
class LinearNormalizer:
    def __init__(self): self.stats = {}
    def fit(self, data):                          # 统计均值/标准差
        self.stats = {k: (mean, std) for k, v in data.items()}
    def normalize(self, x, key):
        m, s = self.stats[key]; return (x - m) / s
    def unnormalize(self, x, key):
        m, s = self.stats[key]; return x * s + m
```

推理时**一定记得 unnormalize** 回真实量纲，否则动作会爆。

## 6. 为什么它重要

- **多步动作 + 扩散**天然表达多模态（同一观察可有多条合理动作），比高斯回归强。
- **和任务无关**：同一套代码跑机械臂、自动驾驶都行，只需要换 obs 编码器与 dataset。
- 它是后面所有「扩散驾驶策略」的方法论母体：DriveVLA-W0 的 diffusion_policy、DiffusionDrive 的 flow matching、Diffusion Planner 的轨迹扩散，都是它的变体。

## 7. 和本系列其他文章的关系

- 自动驾驶里直接套 Diffusion Policy 思路 → DiffusionDrive（生成式端到端）。
- 把动作换成轨迹、加无训练引导 → Diffusion Planner（下一篇）。
- 把扩散换成 Flow Matching、挂 VLM 后 → pi0。

---

> 个人思考：Diffusion Policy 的精髓是「把控制当生成」。它不假装自己懂物理，只是用扩散把多模态动作分布学出来。这种思路一旦进入驾驶域，立刻解决「回归式规划只会出平均轨迹」的顽疾——这也是我自己在 Flow-GRPO 里反复用到的底层直觉。
