---
title: "Diffusion Policy 代码讲解：用 DDPM 给机器人「生成」一连串动作"
date: 2026-07-20
description: "拆解 real-stanford/diffusion_policy：ConditionalUnet1D 去噪网络、观察/动作视野、LinearNormalizer 归一化、Workspace/Dataset/Policy 解耦，看通用扩散动作策略如何落地"
tags: ["扩散模型", "DDPM", "机器人", "代码讲解"]
categories: ["代码讲解"]
summary: "「Diffusion Policy 把『给一段观察、出一段动作序列』做成条件扩散生成：噪声动作被 ConditionalUnet1D 逐步去噪，训练是 DDPM 标准损失（预测加到数据上的噪声、MSE），推理是 DDIM 多步采样（20~100 步），归一化必做且推理要 unnormalize 回真实量纲。本文顺着 real-stanford 官方仓库，从名词速查、架构总览、项目结构，到 ConditionalUnet1D / diffusion / diffusion_unet1d_policy / base_dataset / train_diffusion_policy 逐文件逐函数细讲，并穿插『为什么多步动作更稳、为什么扩散天然多模态、为什么归一化是命门』等认知，最后落到它作为 DiffusionDrive / Diffusion Planner / pi0 方法论母体的位置，是一篇面向零基础、超详细、篇幅足够长的工程向代码讲解」"
---

## 写给零基础读者：读这篇之前先搞懂几个名词

在正式读代码之前，我们先把这篇文章会反复出现的几个「黑话」用一句话翻译成人话。你不用现在就完全懂，只要有个印象，后面看到它们就不会慌。

- **DDPM（去噪扩散概率模型，Denoising Diffusion Probabilistic Models）**：一种生成式 AI 的训练范式。它的训练过程是「往一段干净数据（这里是动作序列）上一点点加噪声，直到变成纯噪声；再训练一个网络学会把噪声一步步去掉、还原干净数据」。推理时，从纯噪声出发，让网络反复「去噪」，最后「雕」出一段新的数据。Diffusion Policy 把这套思路用在「生成一连串动作」上，不是生成图片。

- **DDIM（去噪扩散隐式模型，Denoising Diffusion Implicit Models）**：DDPM 的加速采样变体。它把原本必须一步一步、不能跳的随机去噪过程，改成一个确定性的、可以跳步的过程，所以能用更少的步数（比如 20 步）得到差不多的结果。可以理解为「DDPM 是慢工出细活的随机雕刻，DDIM 是能抄近路的确定性雕刻」。

- **观察视野 To（observation horizon）与动作视野 Ta（action horizon）**：To 是模型「往回看」多少步观测（比如过去 2 步的相机图 + 机器人状态），Ta 是模型「往前出」多少步动作（比如未来 16 步的关节角度或转向指令）。典型配置 To=2、Ta=16。多步动作比「只预测下一步」稳得多，因为网络一次性把一小段未来都规划出来，不会因为单步误差被放大而抖。

- **LinearNormalizer（线性归一化器）**：对观测和动作做线性缩放（均值/标准差，或最小/最大）的小工具。扩散模型对数据的「量纲」极其敏感——动作里如果同时有「角度（几度）」和「位置（几米）」，数值尺度差太大，网络就学不动。所以必须把数据压到差不多尺度再喂给扩散。它是本文最容易踩坑、也最容易被忽略的一环。

- **U-Net / 1D U-Net**：U-Net 是一种「先下采样压缩、再上采样还原」的对称网络，形似字母 U，原本用于图像分割。这里动作是一维时间序列（Ta 步），所以用「1D 版」的 U-Net（用一维卷积），而不是图像用的 2D U-Net。它的作用是「吃进带噪动作，吐出预测噪声」。

- **条件扩散（conditional diffusion）**：普通扩散是无条件的（从纯噪声生成任意东西）；条件扩散是「在给定某个条件（这里是观察）的情况下生成」。实现方式之一是用 FiLM 把观察特征以「缩放 + 偏移」的方式注入网络的每一层，相当于不停地告诉网络「你现在是为这个观察去噪，请按这个观察来调整」。

- **noise schedule（噪声调度）**：决定「第 t 步数据里噪声占多少比例」的一条曲线。常见有 linear（线性）和 cosine（余弦）两种。它控制加噪的快慢，也决定训练时每个时间步对应的噪声强度。

- **recursive replanning（递归重规划）**：模型一次输出 Ta 步动作，但执行时通常只取前几步真正发给机器人，下一步再重新看新的观察、重新生成一段 Ta 步动作。如此滚动，像人开车时不断「看一眼、动一下、再看一眼」。这能抵消误差累积，比「一次算完走到底」稳。

- **多模态动作（multimodal action）**：同一个观察，合理的动作可能不止一种（比如「向左绕」和「向右绕」都行）。扩散模型靠「从噪声出发、一次雕出一条完整轨迹」的天然属性，能表达这种多峰分布；而传统「高斯回归一个均值」只会吐出「两种开法的平均」，结果往往是不伦不类的危险轨迹。

- **RSS 2023**：Robotics: Science and Systems 2023，这篇论文（Diffusion Policy: Visuomotor Policy Learning via Action Diffusion）发表的会议。它虽然是机器人操纵领域的文章，但和自动驾驶扩散方法同源。

如果你上面这些名词都大致有印象了，下面读代码会轻松很多。我们开始。

## 为什么要讲 Diffusion Policy 的代码

Diffusion Policy（RSS 2023，作者单位 real-stanford / 哥伦比亚大学）是「把扩散模型用于动作生成」的奠基性工作，也是后续几乎所有「扩散驾驶策略」的方法论源头。它不特指自动驾驶——它原本是给机械臂、给机器人学操纵用的——但正是因为它把「给一段观察、出一段动作序列」这件事做成了一个干净、通用、和任务解耦的框架，后面 DiffusionDrive（扩散规划）、Diffusion Planner（轨迹扩散）、甚至 pi0（挂了 VLM 的流匹配动作模型）才都能在它身上找到影子。

这篇不讲公式推导，直接看源码——仓库 `real-stanford/diffusion_policy` 把模型、数据、训练循环三者拆得极干净，是学习「扩散 + 控制」工程化落地的最佳范本。

> **一句话结论**：Diffusion Policy 的本质 = 观察编码器（CNN/Transformer/Recurrent 三选一）+ 一个 1D U-Net 去噪网络（ConditionalUnet1D）+ 标准 DDPM 训练（预测噪声的 MSE）+ DDIM 多步采样推理；训练时随机加噪学去噪，推理时从纯噪声雕出整段 Ta 步动作，再 unnormalize 回真实量纲、取前几步执行并递归重规划。

## 架构总览：先看地图

Diffusion Policy 是**生成式策略**：它不从固定候选集里选动作，而是**从纯噪声出发、用扩散逐步「雕」出一段动作序列**，并且这段动作是「 conditioned on（以）观察」的。

把整条链路用一句话串起来就是：

**一段观察（过去 To 步的图/状态）→ 观察编码器 → 观察特征（global_cond）→ 作为条件注入 1D U-Net → 从纯噪声动作出发，按 DDIM 调度反向去噪 Ta 步 → 得到去噪后的动作序列 → unnormalize 回真实量纲 → 取前几步执行，下一步递归重规划。**

用文字 + Markdown 列表把这个闭环拆开看：

- 输入侧：过去 To 步观测，可以是图像（CNN 编码器）、状态向量（MLP）、或者时序状态（Recurrent / Transformer 编码器）。
- 条件注入：观察编码成固定维度向量 global_cond，通过 FiLM 加偏置注入 U-Net 每一层，实现「条件扩散」。
- 生成侧：动作是 Ta 步的一维序列，形状 `[B, Ta, action_dim]`，初始为纯高斯噪声。
- 去噪网络：ConditionalUnet1D，输入「带噪动作 + 时间步 timestep + 观察条件」，输出「预测的噪声」。
- 训练目标：标准 DDPM 损失——预测加到数据上的那一份噪声，和真实噪声做 MSE。
- 推理采样：DDIM 多步（默认 20~100 步）反向迭代，逐步把噪声动作雕成干净动作。
- 后处理：LinearNormalizer 把动作反归一化回机器人真实量纲；执行时取前 k 步，递归重规划。

> **关键认知**：Diffusion Policy 的「多步 + 扩散」组合，和「单步回归」最大的区别就是——它能表达多模态动作分布。同一个观察，网络可以从不同噪声起点雕出不同合理动作；而回归式方法只会给一个均值，遇到「左绕还是右绕」这种分叉就只会吐出危险的「中间轨迹」。这也是后面所有扩散驾驶方法要解决的同一个顽疾。

## 项目结构：先厘清边界（real-stanford/diffusion_policy）

Diffusion Policy 最值得学的是**模块拆分**。它把「模型定义」「数据拼装」「训练循环」三者完全解耦，换任务基本只改 yaml 配置和 dataset，不动网络结构。仓库根目录是 `diffusion_policy/`，下面大致长这样（用普通 Markdown 列表，避免代码块内中文被逐字符换行的问题）：

- diffusion_policy/
  - policy/
    - diffusion_policy.py：总装类，把「观察编码 + 去噪网络 + 采样器」串起来，对外提供 predict_action / compute_loss。
    - diffusion_unet1d_policy.py：基于 1D U-Net 的策略入口（本文重点），继承上面的总装。
    - diffusion_transformer_policy.py：基于 Transformer 去噪网络的策略变体（本文不展开）。
    - model/
      - noise_unet.py：ConditionalUnet1D 的定义（1D U-Net 去噪网络，本文核心）。
      - diffusion.py：DDPM / DDIM 的加噪 q_sample 与采样逻辑封装。
      - trajectory_visualizer.py：可视化工具，不重要。
  - model/
    - cfg_scheduler.py：噪声调度（cosine / linear）的封装，给扩散用。
    - obs_encoder.py：图像 / 状态 / 时序三种观察编码器。
    - common/
      - module_attr_mixin.py：给模型挂属性的一些小 mixin。
  - dataset/
    - base_dataset.py：拼 (obs, action) 对，做 LinearNormalizer 归一化 / 反归一化。
    - realpusht_state_dataset.py / image_dataset.py 等：具体任务的数据集实现。
  - workspace/
    - train_diffusion_policy.py：训练循环 + 日志 + checkpoint（基于 Hydra + 自写 TrainLoop）。
    - train_diffusion_unet_..._realpusht.yaml 等：所有超参都在 yaml。
  - configs/
    - 各种任务的 yaml 配置，集中放超参。

逐个文件用一句话说明它干嘛：

- `policy/diffusion_policy.py`：策略总装基类 `DiffusionPolicy`。定义了「观察怎么编码、噪声怎么采样、损失怎么算、动作怎么预测」的通用骨架；具体去噪网络由子类（如 `DiffusionUnet1dPolicy`）提供。
- `policy/diffusion_unet1d_policy.py`：本文主角。用 ConditionalUnet1D 当去噪网络，实现 `predict_action`（DDIM 采样出动作）和 `compute_loss`（DDPM 损失）。
- `model/noise_unet.py`：ConditionalUnet1D 的实现。1D U-Net，观察作 FiLM 条件注入，时间步作时间嵌入。
- `model/diffusion.py`：`Diffusion` 类。封装 `q_sample`（按噪声调度给干净动作加噪）、损失计算、以及 DDIM 采样所需的 step 逻辑。
- `model/cfg_scheduler.py`：把 noise schedule（线性 / 余弦）和 DDIM 步数封装好，供 diffusion 调用。
- `model/obs_encoder.py`：三种观察编码器——CNNImageEncoder（图像）、IdentityEncoder / MLP（状态）、以及基于 RNN / Transformer 的时序编码器。
- `dataset/base_dataset.py`：`BaseDataset` + `LinearNormalizer`。负责把原始 (obs, action) 拼成样本，并在训练前 fit 归一化统计量、训练时 normalize、推理时 unnormalize。
- `workspace/train_diffusion_policy.py`：训练入口。用 Hydra 读 yaml，构造 policy / dataset / optimizer，跑训练循环，存 checkpoint，记日志。

> **注意**：文件名里写着 `realpusht`、`image_dataset` 这些是论文里的演示任务（推方块、机械臂），但网络与训练框架和任务无关。换到自动驾驶，你只要新写一个 dataset（把「相机/BEV/状态」当 obs、把「未来轨迹/控制量」当 action），再改 yaml，网络一行都不用动。这就是「解耦设计」的威力。

## 一、观察编码器：obs_encoder.py（先把"看"这件事讲透）

模型首先要「看懂」观察。Diffusion Policy 支持三种观察编码器，对应三种输入模态：

- **CNN 图像编码器**：吃多视角图像，过 CNN（如 ResNet 变体）提取特征，再 flatten / pool 成固定维度向量。论文里机械臂任务多用这个。
- **状态编码器（MLP / Identity）**：观察本身就是低维状态向量（比如机器人关节角），直接过一个 MLP 或不编码（Identity），得到条件向量。
- **时序编码器（Recurrent / Transformer）**：观察是「过去 To 步的状态序列」，用 RNN 或 Transformer 把这段时序压成一个向量，能捕捉动态历史。

不管哪种，最终都汇成一个 `global_cond`（全局条件向量），形状大致 `[B, global_cond_dim]`，后面喂给 U-Net 做 FiLM 注入。下面给一个简化的结构示意（块内零中文，中文说明在块外）：

```python
# model/obs_encoder.py (simplified structure)
class CNNEncoder(nn.Module):
    def __init__(self, shape, out_dim=256):
        # shape: (C, H, W) of input image
        self.cnn = build_cnn_backbone(shape, out_dim)

    def forward(self, obs):
        # obs: (B, To, C, H, W) -> merge To into batch
        b, t = obs.shape[:2]
        x = obs.reshape(b * t, *obs.shape[2:])
        x = self.cnn(x)                 # (B*To, out_dim)
        x = x.reshape(b, t, -1)
        x = x.mean(dim=1)               # pool over To -> (B, out_dim)
        return x
```

上面这段是说：图像编码器把「To 步」当成额外 batch 维一起过 CNN，再把时间维平均池化，得到每帧一致的条件向量。状态 / 时序编码器同理，只是 backbone 不同。

> **一句话澄清**：观察编码器只负责「把观察压成一个条件向量」，它不参与去噪。去噪是 U-Net 的事，编码器只是给 U-Net 递「小纸条」告诉它现在场景长啥样。

## 二、核心去噪网络：ConditionalUnet1D（noise_unet.py）

动作是一维时间序列（Ta 步，每步 action_dim 维），所以用 **1D U-Net**（不是图像用的 2D U-Net）。它的职责非常单一：**吃进「带噪动作 + 时间步 + 观察条件」，吐出「预测的噪声」**。

### 2.1 网络结构

网络主体是一个对称的「下采样 → 中间块 → 上采样」结构，每一层用一维卷积（`Conv1d`）。下采样把序列在「时间轴 Ta」上压缩、通道数翻倍；上采样再还原回来。时间步 `timestep` 被编码成时间嵌入（time embedding），观察条件 `global_cond` 被编码成条件向量，两者都通过 FiLM 式「加偏置 / 缩放」注入每一层。

下面给一个最简化、能跑通逻辑的伪代码（块内零中文）：

```python
# model/noise_unet.py (heavily simplified)
class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=(256, 512, 1024),
                 dim_mult=(1, 2, 2), kernel_size=5):
        self.input_dim = input_dim
        self.down_dims = down_dims
        # time embedding: map scalar timestep -> feature
        self.time_emb = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        # project observation condition to same dim
        self.cond_obs = nn.Sequential(
            nn.Linear(global_cond_dim, 256), nn.SiLU(), nn.Linear(256, 256))
        # down / mid / up blocks with 1d conv
        self.down_blocks = nn.ModuleList([...])   # Conv1d + downsample
        self.mid_block = MidBlock1D(...)          # Conv1d bottleneck
        self.up_blocks = nn.ModuleList([...])     # Conv1d + upsample
        self.final_conv = nn.Conv1d(256, input_dim, 1)

    def forward(self, sample, timestep, global_cond):
        # sample: (B, Ta, input_dim) noisy action
        # timestep: (B,) integer step
        # global_cond: (B, global_cond_dim) observation feature
        t_emb = self.time_emb(timestep_embed(timestep))      # (B, 256)
        c_emb = self.cond_obs(global_cond)                   # (B, 256)
        h = sample.transpose(1, 2)                           # (B, input_dim, Ta)
        skips = []
        for down in self.down_blocks:
            h, c_emb = down(h, c_emb, t_emb)                 # FiLM inject
            skips.append(h)
        h = self.mid_block(h, c_emb, t_emb)
        for up in self.up_blocks:
            h = up(h, skips.pop(), c_emb, t_emb)             # FiLM inject
        h = self.final_conv(h)                               # (B, input_dim, Ta)
        return h.transpose(1, 2)                             # (B, Ta, input_dim) noise
```

中文说明：上面这段里，`timestep_embed` 把整数时间步变成连续向量（一般是正弦位置编码再线性变换），`time_emb` 和 `cond_obs` 各自把它俩映射到同一维度 256。`down_blocks` 逐层下采样，每个 block 内部把 `c_emb` 和 `t_emb` 以「加偏置（和条件有关的一个常数加到特征上）」的方式注入，这就是 FiLM 条件扩散的核心。最后 `final_conv` 把通道数压回 `input_dim`，输出和输入同形的「预测噪声」。

> **关键认知**：FiLM 注入为什么有效？因为 U-Net 的每一层特征都被「观察条件」调制了——网络在去噪的每一步都知道「我现在是在为这个观察服务」，于是它雕出来的动作天然贴合当前场景。这是「条件扩散」落地的最朴素实现，后面 DiffusionDrive 的 cross-attention 注入、pi0 的 conditioning 都是同一思想的变体。

### 2.2 为什么是 1D 而不是 2D

动作序列是 `(Ta, action_dim)` 的二维张量，但「时间」和「动作维度」语义不同：时间轴是要去噪铺开的轴，动作维度是每个时间步的向量。1D U-Net 在「时间轴」上做卷积和下采样，把时间维压短再还原，通道维承载 action_dim。这和图像 2D U-Net 在「高 × 宽」上卷积是平行的思路，只是从 2D 降成了 1D。

## 三、扩散过程：diffusion.py（DDPM 训练 + DDIM 推理的引擎）

`Diffusion` 类是整个扩散过程的「发动机」。它管两件事：(1) 训练时怎么给干净动作加噪（forward diffusion，即 `q_sample`）；(2) 推理时怎么按 DDIM 一步步去噪（`step`）。

### 3.1 加噪公式（DDPM 的 q_sample）

DDPM 的前向过程是一个固定的马尔可夫链：在时刻 t，干净数据 x0 被加进方差为 β_t 的噪声。闭式解（直接跳到任意 t 步）是：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$，所以 $\sqrt{\bar{\alpha}_t}$ 是「保留多少原始信号」，$\sqrt{1-\bar{\alpha}_t}$ 是「加多少噪声」。这个公式就是代码里 `q_sample` 干的事：给定干净动作 `action` 和随机噪声 `noise`，按当前 timestep 的系数线性混合，一步到位得到 `noisy_action`，不用一步步迭代。

### 3.2 训练：标准 DDPM 损失（预测噪声的 MSE）

训练目标极其简单——网络去预测「刚才加进去的那份噪声」，和真实噪声做均方误差。下面给简化伪代码（块内零中文）：

```python
# model/diffusion.py (simplified training step)
def compute_loss(self, model, obs_cond, action):
    # action: (B, Ta, action_dim) clean action
    noise = torch.randn_like(action)                  # eps ~ N(0, I)
    timestep = torch.randint(0, self.horizon, (action.shape[0],))
    noisy_action = self.q_sample(action, noise, timestep)
    # model predicts the noise added at timestep t
    pred_noise = model(noisy_action, timestep, global_cond=obs_cond)
    loss = F.mse_loss(pred_noise, noise)              # epsilon-prediction
    return loss
```

中文说明：训练时随机抽一个时间步 `timestep`（从 0 到总步数之间均匀采样），用 `q_sample` 把干净动作变成带噪动作，再让 U-Net 预测噪声，最后和「真实加进去的噪声」算 MSE。注意这是 **ε-prediction（预测噪声）**，和 DDPM 原论文完全一致。所有 Ta 个动作步、所有 action_dim 维一起联合回归，没有分步加权。

> **一句话澄清**：训练时网络「看不到」干净动作 x0，它只看到带噪动作和 timestep，被要求还原噪声。推理时反过来——从纯噪声出发，用网络反复预测噪声并减去，逐步逼近 x0。训练和推理是同一套网络、两种用法。

### 3.3 推理：DDIM 多步采样

推理用 DDIM（确定性、可跳步），从纯高斯噪声 `sample` 出发，按调度给的时间步序列反向迭代。每步让 U-Net 预测噪声，再用 DDIM 更新公式把 `sample` 往「更干净」推一步。简化伪代码（块内零中文）：

```python
# model/diffusion.py (simplified DDIM step, conceptual)
def ddim_step(self, model, sample, timestep, obs_cond, prev_timestep):
    noise_pred = model(sample, timestep, global_cond=obs_cond)
    # DDIM closed-form update (deterministic)
    alpha = self.alphas[timestep]
    alpha_prev = self.alphas[prev_timestep]
    x0_pred = (sample - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
    dir_term = torch.sqrt(1 - alpha_prev) * noise_pred
    sample = torch.sqrt(alpha_prev) * x0_pred + dir_term
    return sample
```

中文说明：上面这段是 DDIM 更新的一步逻辑（真实代码用 `diffusers.DDIMScheduler` 或自写调度，细节略有差异，但思想一致）：先由「带噪样本 + 预测噪声」反推出 x0 的估计，再按上一步的 alpha 重新组合出「更干净的样本」。反复 N 步（默认 20~100 步）后，`sample` 就近似成一段干净动作。DDIM 之所以快，是因为它允许把总步数从 DDPM 的 1000 步压缩到几十步且基本不降质量。

## 四、策略入口：diffusion_unet1d_policy.py（把一切串起来）

`DiffusionUnet1dPolicy` 是真正对外暴露 API 的类。它继承 `DiffusionPolicy` 总装，内部持有 `ConditionalUnet1D`（去噪网络）、`Diffusion`（扩散引擎）、`obs_encoder`（观察编码器）、`noise_scheduler`（DDIM 调度）。它最该被盯紧的两个方法是 `compute_loss`（训练）和 `predict_action`（推理）。

### 4.1 训练路径 compute_loss

训练时，policy 先把观察编码成 `obs_cond`，再把 (obs_cond, action) 交给 `Diffusion.compute_loss` 算 MSE。简化逻辑（块内零中文）：

```python
# policy/diffusion_unet1d_policy.py (simplified)
class DiffusionUnet1dPolicy(DiffusionPolicy):
    def __init__(self, model, obs_encoder, noise_scheduler, ...):
        self.model = model                  # ConditionalUnet1D
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler

    def compute_loss(self, batch):
        obs = batch['obs']
        action = batch['action']            # (B, Ta, action_dim)
        obs_cond = self.obs_encoder(obs)    # (B, global_cond_dim)
        # normalize actions before diffusion (handled in dataset usually)
        loss = self.diffusion.compute_loss(self.model, obs_cond, action)
        return loss
```

中文说明：这里 `obs_encoder` 把观察压成条件，`action` 是数据集里已经 normalize 过的干净动作，`compute_loss` 内部就是上一节讲的「随机加噪 + 预测噪声 + MSE」。注意归一化通常在 dataset 侧就做好了，policy 拿到的是归一化后的动作，这样扩散的量纲才安全。

### 4.2 推理路径 predict_action（DDIM 多步采样）

推理时，给定一段新观察，policy 先编码条件，再从纯噪声出发跑 DDIM 采样，最后把动作 unnormalize 回真实量纲。简化伪代码（块内零中文）：

```python
# policy/diffusion_unet1d_policy.py (simplified predict_action)
@torch.no_grad()
def predict_action(self, obs):
    obs_cond = self.obs_encoder(obs)                       # (B, global_cond_dim)
    sample = torch.randn((obs.shape[0], self.Ta, self.action_dim))
    for t in self.noise_scheduler.step_timesteps:          # DDIM reverse
        noise_pred = self.model(sample, t, global_cond=obs_cond)
        sample = self.noise_scheduler.step(noise_pred, t, sample)
    # sample now ~ clean action in normalized space
    action = self.normalizer['action'].unnormalize(sample)
    return action                                          # (B, Ta, action_dim)
```

中文说明：`step_timesteps` 是 DDIM 调度预先算好的「反向时间步序列」（比如从 999 跳到 0 的几十个离散点）。循环每步让 U-Net 预测噪声、调度器更新样本，循环结束 `sample` 就是归一化空间里的干净动作。最后用 `normalizer` 把它反归一化回机器人真实单位（度 / 米 / 弧度），这一步绝对不能漏，否则动作量纲不对、机器人直接抽风。

> **关键认知**：推理输出的动作是「整段 Ta 步」。但真正执行时，论文和工程实践都只取前 k 步（比如前 5 步）发给机器人，然后下一步重新观察、重新生成。这就是 recursive replanning（递归重规划），它能把单步误差「就地消化」，不让错误滚雪球。

## 五、归一化：最容易踩的坑（base_dataset.py 的 LinearNormalizer）

扩散模型对量纲极敏感——动作向量里如果「角度」是几度、「位置」是几米、「速度」是每秒几米，数值尺度天差地别，网络会只盯着大尺度那一项学。所以 **LinearNormalizer 是必做项**，不是可选项。

### 5.1 三种归一化模式

`LinearNormalizer` 支持几种缩放方式，最常见的是：

- **normal（均值/标准差）**：减均值除标准差，把数据拉到「均值 0、方差 1」附近。
- **min_max**：减最小值除以极差，把数据压到 [0, 1] 或 [-1, 1]。
- **identity**：不归一化（某些本来就无量纲的字段用）。

它对「观察的各个 key」和「动作 action」分别 fit 统计量。简化实现（块内零中文）：

```python
# dataset/base_dataset.py (simplified LinearNormalizer)
class LinearNormalizer:
    def __init__(self, mode='normal'):
        self.mode = mode
        self.stats = dict()          # key -> (mean, std) or (min, max)

    def fit(self, data_dict):        # data_dict: key -> Tensor
        for key, tensor in data_dict.items():
            mean = tensor.mean(dim=0)
            std = tensor.std(dim=0) + 1e-8
            self.stats[key] = (mean, std)

    def normalize(self, x, key):
        mean, std = self.stats[key]
        return (x - mean) / std

    def unnormalize(self, x, key):
        mean, std = self.stats[key]
        return x * std + mean
```

中文说明：训练前先在整个训练集上 `fit`，统计每个 key 的均值/标准差；训练时 `normalize` 把动作和观察压到标准尺度再喂给扩散；推理时 predict_action 里用 `unnormalize` 把生成的动作还原回真实单位。**千万记住：归一化和反归一化是成对出现的，fit 的统计量必须从训练集来、推理时复用，不能各自重新算。**

### 5.2 为什么归一化是「命门」

如果不归一化，扩散的损失函数会被大尺度维度主导，小尺度维度（比如细微的角度修正）几乎学不到，结果动作「大方向对、细节全废」，机器人表现为抖动或失控。论文里所有任务都做了归一化，这也是为什么 base_dataset 把这个逻辑放在数据层、和模型解耦——换任务只要换 dataset 里的 key 名，归一化框架不动。

## 六、训练循环：train_diffusion_policy.py（Hydra 驱动的样板）

训练入口用 Hydra 读 yaml，构造 policy、dataset、optimizer，然后跑标准 PyTorch 训练循环。它不是本文重点，但有几个点值得点一下：

- **配置全在 yaml**：网络维度、学习率、噪声调度类型（linear/cosine）、DDIM 步数、To/Ta、batch size 全部写在 `configs/train_diffusion_unet_*.yaml` 里，代码里几乎不硬编码。这就是「换任务只改 yaml」的底气。
- **Workspace 模式**：`TrainLoop` 负责跑 epoch、记 loss、存 checkpoint、可能还有验证集 rollout 评估。它和 policy / dataset 解耦，所以同一个训练循环能训任意 diffusion policy。
- **数据加载**：dataset 在 `__getitem__` 里返回 `(obs, action)` 样本（已经 normalize 过），dataloader 打包成 batch 喂给 `policy.compute_loss`。

简化训练骨架（块内零中文）：

```python
# workspace/train_diffusion_policy.py (simplified skeleton)
def main(cfg):
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataset.setup_normalizer()                 # fit LinearNormalizer on train set
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_normalizer(dataset.get_normalizer())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    for epoch in range(cfg.epochs):
        for batch in dataset.dataloader():
            loss = policy.compute_loss(batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        policy.save_checkpoint(epoch)
```

中文说明：上面这段把核心流程画出来了——先 fit 归一化、再把归一化器灌给 policy（保证训练和推理用同一套统计量），然后标准 `loss -> backward -> step` 循环。工程上就这么朴素，复杂的东西全藏在 yaml 配置和各个被实例化的模块里。

## 七、把一次前向 / 反向完整走一遍（串起来看）

为了让你脑子里有完整闭环，这里把「训练一步」和「推理一次」各用一句话列出来：

训练一步：

1. 取一个 batch 的 (obs, action)，action 已 normalize。
2. obs_encoder 把 obs 编码成 obs_cond。
3. 随机抽 timestep，q_sample 给 action 加噪得到 noisy_action。
4. ConditionalUnet1D 吃 (noisy_action, timestep, obs_cond) 预测噪声。
5. MSE(预测噪声, 真实噪声) 反传，更新 U-Net 参数。

推理一次：

1. 新 obs 编码成 obs_cond。
2. 从纯噪声 `sample` 出发。
3. 对 DDIM 的每个反向时间步：U-Net 预测噪声 → 调度器更新 sample。
4. 循环结束得到归一化空间干净动作。
5. unnormalize 回真实量纲。
6. 取前 k 步执行，下一步递归重规划。

> **一句话澄清**：训练和推理用的是「同一个 U-Net」，区别只在于——训练时噪声是随机加的、timestep 是随机抽的；推理时噪声是纯高斯起步、timestep 是按调度从大到小固定走的。网络本身不区分「训练 / 推理模式」，区分的是「你怎么喂它噪声」。

## 七之一、U-Net 内部到底长什么样（down / mid / up 三个块）

上一节我们把 `ConditionalUnet1D` 当成黑盒讲了输入输出，这一节把盒子打开，看看 `down_blocks`、`mid_block`、`up_blocks` 内部到底在算什么。理解了这块，你才算真正看懂「条件 1D U-Net」。

每个下采样块（DownBlock1D）通常做三件事：一维卷积提取特征 → 用条件做 FiLM 调制 → 在时间轴（Ta）上下采样（stride=2 把序列压短一半）。中间块（MidBlock1D）是瓶颈，通道数最大、序列最短，做一次或几次卷积极限提取。上采样块（UpBlock1D）则是反向：先上采样把序列拉长、再卷积、再 FiLM，并且把同层下采样时存下来的 `skip` 特征拼回来（skip connection，U-Net 的灵魂，防信息丢失）。

简化伪代码（块内零中文）：

```python
# model/noise_unet.py (simplified block internals)
class DownBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, c_emb, t_emb):
        h = self.conv1(x)
        h = h + c_emb[:, :, None] + t_emb[:, :, None]      # FiLM: add cond + time
        h = F.silu(h)
        h = self.conv2(h)
        h = h + c_emb[:, :, None] + t_emb[:, :, None]      # FiLM again
        h = F.silu(h)
        h = self.downsample(h)
        return h

class UpBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, kernel_size=5):
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)

    def forward(self, x, skip, c_emb, t_emb):
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)                    # skip connection
        h = self.conv1(h)
        h = h + c_emb[:, :, None] + t_emb[:, :, None]      # FiLM
        h = F.silu(h)
        h = self.conv2(h)
        h = h + c_emb[:, :, None] + t_emb[:, :, None]      # FiLM
        h = F.silu(h)
        return h
```

中文说明：上面这段把 FiLM 写得最直白——`c_emb` 和 `t_emb` 都被 broadcast 到「(B, 256, Ta)」的形状，然后直接加到特征上。注意 FiLM 标准做法是「缩放 gamma + 偏移 beta」两个参数，这里为简化只写了「加偏移」一种；真实代码里 `global_cond` 会被拆成 `(scale, shift)` 两路分别乘和加，但思想完全一致：「用条件去调制特征」。`skip` 拼接则是让上采样时能找回下采样时压缩掉的高频细节，否则去噪出来的动作会糊。

> **一句话澄清**：你不需要背下这些卷积维度。只要记住三件事：(1) 时间轴 Ta 在下采样时变短、上采样时变长；(2) 每一层都被观察条件和时间步调制（FiLM）；(3) skip connection 把细节从下采样层「抄近路」送回上采样层。这三点就是 1D U-Net 去噪的全部奥义。

## 七之二、三种观察编码器到底差在哪（obs_encoder.py 细看）

前面说过 Diffusion Policy 支持 CNN / 状态 MLP / 时序 RNN-Transformer 三种编码器。这里补一句每种适合什么、代码长啥样。核心差异只在 backbone，输出都会被压成同一个维度的 `global_cond`，所以下游 U-Net 完全无感。

简化伪代码（块内零中文）：

```python
# model/obs_encoder.py (three encoder variants, simplified)
class StateEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=256):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim))
    def forward(self, obs):                  # obs: (B, obs_dim)
        return self.mlp(obs)                 # (B, out_dim)

class CNNEncoder(nn.Module):
    def __init__(self, shape, out_dim=256):
        self.cnn = build_cnn(shape, out_dim)
    def forward(self, obs):                  # obs: (B, To, C, H, W)
        b, t = obs.shape[0], obs.shape[1]
        x = obs.reshape(b*t, *obs.shape[2:])
        x = self.cnn(x).reshape(b, t, -1).mean(dim=1)
        return x                             # (B, out_dim)

class RecurrentEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=256):
        self.rnn = nn.GRU(input_dim, out_dim, batch_first=True)
    def forward(self, obs):                  # obs: (B, To, obs_dim)
        _, h = self.rnn(obs)
        return h.squeeze(0)                  # (B, out_dim) last hidden
```

中文说明：`StateEncoder` 最简单，obs 本身就是低维向量，过两层 MLP 即可，适合「机器人关节角、自车状态」这类结构化输入。`CNNEncoder` 吃图像，把 To 步当 batch 维一起提特征再平均，适合「多视角相机」输入。`RecurrentEncoder` 用 GRU 把时序 obs 压成最后一步隐状态，能显式建模「过去几帧怎么变化」，适合动态场景。自动驾驶里常用 CNN（吃 BEV / 环视图）或 Recurrent（吃时序状态）两种。

> **关键认知**：观察编码器是「把任务差异关进的笼子」。U-Net 和扩散过程完全不关心你喂的是图像还是状态——它们只认 `global_cond` 这个固定维度向量。这就是为什么换任务只改 encoder + dataset，网络主体纹丝不动。

## 七之三、DDPM 训练 vs DDIM 推理：为什么能快这么多

很多人第一次见到「训练 1000 步、推理 20 步」会困惑：步数不一样，网络不会乱吗？答案是：不会，因为 DDPM 和 DDIM 共享同一套训练好的噪声预测网络，区别只在「怎么用网络」。

- **DDPM 推理**：每一步都从「带噪样本 + 预测噪声」里减掉噪声，并重新注入一点随机噪声（保持随机性），必须一步步走完（一般 1000 步）才能收敛。慢，但样本多样性好。
- **DDIM 推理**：去掉了每步重新注入的随机噪声，变成确定性映射，因此可以「跳步」——只在 20~100 个离散时间步上走，中间的直接跳过。快，且结果基本不变。

用一句话类比：DDPM 是「每步都重新掷骰子决定往哪走」的随机漫步，必须走满才能到目的地；DDIM 是「看一眼地图（网络预测的噪声方向）就直线抄近路」，所以少走几步也到得了。两者训练时学的是同一张「地图」（噪声预测器），所以共享网络不冲突。

## 七之四、多模态动作：一次推理怎么「雕」出不同开法

扩散模型表达多模态的天然机制是：**从不同的纯噪声起点出发，去噪后会收敛到不同的合理动作**。这不像回归模型「一个输入对应一个输出」，而是「一个输入 + 不同噪声种子 = 多个可能输出」。

工程上怎么用这个性质？一个常见做法是：推理时一次性采 K 个独立噪声起点，各自跑一遍 DDIM，得到 K 条候选动作序列，再用一个评判函数（比如「哪条最不容易撞」）挑一条执行。这其实就是后面 DiffusionDrive 用 20 个 anchor + argmax 选轨迹的思想雏形——只不过 Diffusion Policy 是从「纯噪声」出发、DiffusionDrive 是从「anchor 模板」出发，本质都是「多起点生成 + 择优」。

简化伪代码（块内零中文）：

```python
# sampling K multimodal candidates (conceptual)
def predict_action_multimodal(self, obs, K=8):
    obs_cond = self.obs_encoder(obs)
    candidates = []
    for k in range(K):
        sample = torch.randn((1, self.Ta, self.action_dim))   # different seed
        for t in self.noise_scheduler.step_timesteps:
            noise_pred = self.model(sample, t, global_cond=obs_cond)
            sample = self.noise_scheduler.step(noise_pred, t, sample)
        candidates.append(self.normalizer['action'].unnormalize(sample))
    return candidates                                          # list of K actions
```

中文说明：上面这段把「多模态」写成最朴素的形式——同一个 obs，跑 K 次不同随机种子，得到 K 条动作。实际仓库里 Diffusion Policy 默认只取 1 条（单样本执行 + 递归重规划），但多候选 + 筛选是扩散类方法通用的「准确率放大器」，你在 DiffusionDrive / Diffusion Planner 里会反复见到它的升级版。

## 七之五、常见坑排查清单（踩过的都在这）

- **坑 1：忘记 unnormalize**。推理出来动作直接发机器人，结果量纲错乱、机器人抽搐。永远记得 predict_action 最后一步要把动作反归一化。
- **坑 2：训练和推理用两套统计量**。归一化的 mean/std 必须从训练集 fit、推理时复用，绝不能推理时重新算。否则归一化空间错配，生成动作全偏。
- **坑 3：obs 和 action 的 key 对不上**。LinearNormalizer 是按 key 存统计量的，dataset 里 normalize 用的 key 必须和推理 unnormalize 用的 key 完全一致，差一个字母就崩。
- **坑 4：timestep 不是整数 / 越界**。q_sample 和调度器都要求 timestep 在 `[0, horizon)` 内且和调度表对齐，乱传会索引报错。
- **坑 5：DDIM 步数设太少导致动作糊**。20 步通常够，但任务复杂时降到 10 步以内可能雕不干净，表现为动作发抖。权衡速度和质量是常事。
- **坑 6：To 和 Ta 配错**。观察看太短（To 太小）会丢历史动态，动作出太长（Ta 太大）会增加去噪负担且累积误差。论文常用 To=2、Ta=16 起手。

> **一句话澄清**：这六个坑里，前三个（归一化相关）占了实际调试时间的八成。扩散对量纲的敏感不是玄学，是数学——MSE 损失会被大尺度维度主导，所以「先归一化、后扩散、再反归一化」是铁律，没有例外。

## 八、为什么它重要（方法论母体的位置）

- **多步动作 + 扩散**天然表达多模态（同一观察可有多条合理动作），比高斯回归一个均值强太多。这是 Diffusion Policy 被后续所有扩散驾驶方法继承的第一性原理。
- **和任务无关**：同一套代码跑机械臂、跑推方块、跑自动驾驶都行，只需要换 obs 编码器与 dataset。解耦设计让它成为「扩散 + 控制」的通用基线。
- 它是后面所有「扩散驾驶策略」的方法论母体：DiffusionDrive 的 truncated diffusion、Diffusion Planner 的轨迹扩散、pi0 的 flow-matching 动作模型，都是它的变体——要么改了去噪网络（Transformer / cross-attention），要么改了起点（anchor / 流匹配），要么挂了 VLM，但「观察作条件、噪声雕动作、多步输出、递归执行」的骨架一脉相承。

## 九、和本系列其他文章的关系

- 自动驾驶里直接套 Diffusion Policy 思路、把动作换成轨迹并加 anchor 加速 → **DiffusionDrive**（生成式端到端，截断扩散 2 步去噪）。
- 把动作换成轨迹、强调无训练引导和多模态规划 → **Diffusion Planner**（下一篇，轨迹扩散）。
- 把扩散换成 Flow Matching、再挂一个 VLM 当视觉-语言条件 → **pi0**（视觉-语言-动作模型）。
- 本篇（Diffusion Policy）是它们的「第 0 课」：不懂这篇的 FiLM 条件注入、DDPM 损失、DDIM 采样、归一化命门，后面三篇的加速 trick 和变体你都会看得云里雾里。

---

> **个人思考**：Diffusion Policy 的精髓是「把控制当生成」。它不假装自己懂物理、不建模动力学，只是用扩散把「多模态动作分布」老老实实学出来。这种思路一旦进入驾驶域，立刻解决「回归式规划只会出平均轨迹」的顽疾——平均轨迹在「左绕还是右绕」这种分叉点上，往往是一条既撞左车又撞右车的自杀路线。我在 Flow-GRPO 里反复用到的底层直觉，追根溯源就是这篇 RSS 2023 的 Diffusion Policy：先承认分布是多峰的，再用生成模型把它雕出来，最后用采样 / 引导去挑那条真正安全的。读懂它，后面所有扩散驾驶方法都是「换皮不换骨」。
