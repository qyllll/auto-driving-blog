---
title: "代码讲解：DriveVLA-W0 世界模型如何给 VLA 大模型补上稠密监督"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["DriveVLA-W0", "VLA", "世界模型", "端到端自动驾驶", "代码讲解", "Flow Matching", "MoE", "Emu3", "NAVSIM"]
summary: "逐行拆解 BraveGroup/DriveVLA-W0 的已开源代码，面向零基础读者：先讲清 VLA、世界模型、MoE、Flow Matching、FAST tokenizer 等黑话，再厘清仓库边界（只开源了 policy_head 与 tokenizer，VLM 主干在外部库），然后逐文件拆解 MoE 动作专家、Flow Matching 解码器、扩散策略头、动作 tokenizer 与世界模型 VQ 编码器，最后把一次前向从图像到轨迹串成 7 步、讲透两阶段训练与推理旁路世界模型的设计。一篇把代码边界到数据流到关键模块到训练梯度到推理选轨迹到个人思考全部讲明白的极详细工程向代码讲解。"
---

## 写给完全没基础的同学：先补几个最关键的名词

读 DriveVLA-W0 的代码前，先把会反复出现的「黑话」翻译成人话。你不用现在就全懂，有个印象即可。

- **VLA（Vision-Language-Action，视觉-语言-动作）模型**：给模型「一张图（有时加一句指令）」，它直接输出「车怎么开（动作/轨迹）」。它本质是「把驾驶当成语言建模问题」——图当输入 token，动作当输出 token，中间用一个大模型（VLM）串起来。DriveVLA-W0 用的是 Emu3-8B（离散视觉 token）和 Qwen2.5-VL-7B（连续视觉特征）两种主干。

- **世界模型（World Model）**：一个「学会预测未来长啥样」的模型。DriveVLA-W0 让它「看了当前画面 + 做了某个动作，就画出下一帧画面」。它不直接参与开车，只在**训练时**逼模型理解「我做了这个动作，世界会变成什么样」。

- **监督赤字（Supervision Deficit）**：本文核心痛点。给一个 7-8B 的大模型，却只拿「未来几维轨迹点」去监督它——信号太稀疏，大模型的能力被浪费。世界模型用「每像素的未来画面」提供**稠密监督**，把亏损补上。

- **MoE（Mixture of Experts，混合专家）**：把模型拆成多个「专家」子网络，每个输入只激活其中一部分。DriveVLA-W0 让一个 500M 的「动作专家（Action Expert）」和一个 7-8B 的「理解专家（Understanding Expert，即 VLM 主干）」通过 Joint Attention 耦合——小专家借用大专家的表征，却不必把大模型搬进控制回路。

- **Joint Attention（联合注意力）**：两个专家各自算 Q/K/V，沿 token 维度拼起来做一次注意力，再拆回各自专家。这是 MoE 在本文的落地方式（不是路由式 MoE，而是「拼接-融合-拆分」式）。

- **Flow Matching（流匹配）**：一种生成式训练范式。它学一个「从噪声沿直线流到真实动作」的向量场 v_theta，训练时预测这个向量场，推理时沿 ODE 积分几步把噪声「流」成轨迹。相比扩散，它走的是直线概率路径，更简洁。

- **FAST tokenizer**：把连续轨迹点（waypoints）离散化成 token 的方法（源自 OpenVLA）。连续的横向/纵向/朝向，被均匀分箱成 256 个 bin，映射到词表末尾的 256 个 token。这样轨迹就能像文字一样被自回归预测。

- **VQ（Vector Quantization，向量量化）**：把图像压缩成一串「离散视觉 token」的编码器（Emu3 的视觉 tokenizer 就是 MoVQGAN 一类）。世界模型（AR 分支）预测的就是这些离散 token，再用解码器还原成像素。

- **adaLN（adaptive LayerNorm，自适应层归一化）**：用「时间步 t + 类别 y」通过一个 MLP 生成 LayerNorm 的缩放/平移参数，让同一层在不同时间步有不同行为。DiT（扩散 Transformer）的经典技巧，本文的 MoE 专家里也用了。

- **NAVSIM**：自动驾驶规划评测基准，提供传感器数据与 PDMS 指标。DriveVLA-W0 的单目前视版本就刷在这上面。

- **PDMS（Predictive Driver Model Score）**：NAVSIM 的综合规划评分，越高越好（人类约 94.8）。

如果你这些词都有印象了，下面读代码会轻松很多。我们开始。

## 为什么要讲 DriveVLA-W0 的代码

DriveVLA-W0（ICLR 2026，中科院自动化所 × 蔚来）是 VLA 路线里**「用世界模型放大数据缩放律」**的代表作。它最反直觉的结论是：**单目前视相机，靠「预测未来画面」的稠密监督，就能打穿依赖多相机 + 激光雷达的对手**（NAVSIM v1 PDMS 93.0）。

这篇不讲公式推导，直接看 `BraveGroup/DriveVLA-W0` 的真实开源代码。但必须先讲清一个**重要边界**。

> **一句话结论**：DriveVLA-W0 = VLM 主干（Emu3-8B / Qwen2.5-VL-7B）+ 500M MoE 动作专家 + 三种可选动作解码器（Query/AR/Flow Matching）；训练时额外挂一个世界模型分支（AR 或 Diffusion）提供稠密监督，**推理时世界模型被旁路**，只走动作通路。

## 先厘清边界：仓库到底开源了什么

这一点非常关键，照着读源码才不会被坑。仓库 README 明确写道：

> Due to company policy, only the reviewed part of our codebase is available.

也就是说，**真正的大模型主干（Emu3 / Qwen2.5-VL）是外部库，仓库里没有**；开源的是 DriveVLA-W0 **自己写的那一层**——动作专家、解码器、tokenizer、世界模型的 VQ 编码工具，以及推理脚本。挂载关系如下：

- **未开源（外部库）**：`Emu3`（8B 理解专家主干）、`Qwen2.5-VL`（7B 连续特征主干）、`MoVQGAN`（视觉 tokenizer 编解码器）。
- **已开源（本仓库）**：全部在 `models/` 与 `inference/` 下，是「动作头 + tokenizer + 推理入口」这一层插件。

用普通 Markdown 列表展示仓库结构（避免代码块内中文逐字符换行）：

- DriveVLA-W0/
  - models/policy_head/ （动作策略头，作者自研）
    - moe_experts.py （MoE 动作专家，DiT 风格）
    - flow_matching.py （Flow Matching 训练/采样封装）
    - diffusion_policy.py （扩散/流匹配 PolicyHead，含编码解码器）
    - noise_schedulers.py （Flow Matching 加噪调度器）
  - models/tokenizer/ （动作与视觉 tokenizer）
    - action_tokenizer.py （FAST 动作离散化，改自 OpenVLA）
    - emu3_tokenizer_navsim.py （用 Emu3 视觉 tokenizer 把图像压成 VQ 码）
    - extract_vq_emu3_navsim.sh （批量提取 VQ 码的脚本）
  - configs/ （Emu3MoE 配置 + 归一化统计）
    - moe_fast_video.json （动作专家 + 主干超参）
    - normalizer_navsim_trainval/ （训练集归一化参数）
  - inference/vla/ （NAVSIM 推理脚本）
    - inference_action_navsim_flow_matching_vava.py （Flow Matching 版推理入口）
    - infer_navsim_flow_matching_PDMS_87.2.sh （一键推理 shell）
  - utils/datasets.py （NAVSIM 数据集定义）
  - tools/pickle_gen/ （数据预处理成 pickle）
  - Training.md （训练说明）

逐个文件一句话说明它干嘛：

- `moe_experts.py`：定义 `Emu3Experts`（DiT 风格的去噪/生成网络），含 `TimestepEmbedder`、`LabelEmbedder`、`ActionProjector`、adaLN 调制、RoPE。它是动作专家与世界模型共用的一套 Transformer 积木。
- `flow_matching.py`：把 Flow Matching 的「采样时间步 + 加噪 + 预测向量场 + MSE 损失 + ODE 采样」封装成 `FlowMatching` 类，训练与推理都靠它。
- `diffusion_policy.py`：定义 `PolicyHead`（动作头的顶层），内含 `Emu3ActionEncoder`（噪声动作编码）、`MultiLayerAttentionDecoder`（轨迹解码）、`RoPE`/`PositionalEncoding`。这是 Flow Matching 动作解码器的具体实现。
- `noise_schedulers.py`：定义 `FlowMatchingScheduler`，负责按 Beta 或 Uniform 分布采样时间步 tau，并用 `(1-tau)·x + tau·noise` 线性加噪。
- `action_tokenizer.py`：把连续轨迹点离散成词表末尾 token（FAST/OpenVLA 风格），以及反向解码回连续值。
- `emu3_tokenizer_navsim.py`：调用 Emu3 视觉 tokenizer，把 NAVSIM 前视图像编码成 VQ 离散码（世界模型 AR 分支的训练前处理）。
- `moe_fast_video.json`：Emu3MoE 的完整配置，含主干 32 层、动作专家 2 层（500M 量级）、`action_dim=7` 等关键超参。
- `inference_action_navsim_flow_matching_vava.py`：推理主脚本，逐场景调用模型吐轨迹、存 JSON。

> **读源码的边界提醒**：你会发现仓库里**没有**「把图像送进 VLM、吐出动作」的那段主前向——因为它在 Emu3/Qwen2.5-VL 库内部。我们能讲透的，是作者自研的「动作专家 / 解码器 / tokenizer / 世界模型预处理」这一层，以及它们如何被训练与目标如何计算。下面所有的代码解读都严格限定在这层。

## 架构总览：先看地图

![DriveVLA-W0 架构总览](/images/drivevla-w0/fig1.png)

![DriveVLA-W0 详细架构](/images/drivevla-w0/fig2.png)

把整条链路用一句话串起来（以 Flow Matching 解码器 + Emu3 主干为例）：

**前视图像 + 语言指令 + 历史动作 → 拼成深度交错的 [L, V, A, ..., L, V, A] 序列 → VLM 主干（理解专家）自回归编码 → 拆出条件特征 → MoE 动作专家（Action Expert）通过 Joint Attention 借用主干表征 → Flow Matching 解码器从噪声沿向量场「流」出轨迹 → 输出 `[B, T, 7]` 动作。**

训练时，同一条序列还额外喂给**世界模型分支**（AR 预测下一帧 VQ 码 / Diffusion 预测下一帧潜表示），提供稠密监督；**推理时世界模型分支整个旁路掉**。

## 一、配置：从 moe_fast_video.json 看全局超参

先读配置，建立「模型有多大」的直觉。仓库根配置 `configs/moe_fast_video.json` 的关键字段：

- architectures: ["Emu3MoE"]
- hidden_size: 4096 （主干隐维度）
- num_hidden_layers: 32 （理解专家 32 层）
- num_attention_heads: 32
- vocab_size: 184622 （Emu3 词表）
- action_experts: false （此配置下动作专家是否独立，由 action_config 决定）
- action_config.action_dim: 7 （动作维度：x, y, 朝向, 油门, 刹车 ... 共 7 维）
- action_config.num_hidden_layers: 2 （动作专家只有 2 层！）
- action_config.hidden_size: 4096
- action_config.vision_loss_weight: 5.0 （世界模型视觉损失权重，远大于动作）

**大白话**：主干是个 32 层的 4K 隐维度大模型（约 8B），而动作专家只有 **2 层**——这就是「500M 动作专家」的来源。配置里 `vision_loss_weight=5.0` 说明训练时世界模型监督的权重被刻意放大，呼应论文「alpha=0.1 最优、世界损失不宜过高」的折线（本文用 5.0 是另一组设定，但思想一致：让世界损失足够大以逼出表征，又不淹没动作）。

## 二、动作 tokenizer：连续轨迹怎么变成 token

VLA 把驾驶当语言建模，所以连续轨迹必须先「变成文字」。代码在 `models/tokenizer/action_tokenizer.py`，改自 OpenVLA：

```python
class ActionTokenizer:
    def __init__(self, tokenizer, bins=256, min_action=-1, max_action=1):
        self.bins = np.linspace(min_action, max_action, bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.last_vocab_idx = self.tokenizer.pad_token_id - 1
        self.action_token_begin_idx = self.last_vocab_idx - (bins + 1)
```

- `bins=256`：每个连续动作维度被均匀切成 256 个区间。
- `np.linspace(-1, 1, 256)`：区间端点从 -1 到 1。
- `action_token_begin_idx`：动作 token 只占词表**最后 256 个**位置（复用最少用的 token）。

编码（连续 → token id）：

```python
def __call__(self, action):
    action = np.clip(action, self.min_action, self.max_action)
    discretized = np.digitize(action, self.bins)
    return list(self.last_vocab_idx - discretized)
```

**大白话**：把一条轨迹的每一个数（比如「横向偏移 0.3」）先夹到 [-1,1]，再看它落在 256 个箱子的第几格，映射成词表末尾对应的一个 token id。反向 `decode_token_ids_to_actions` 则用 `bin_centers` 把 token 还原成连续值。这样，「自回归预测动作」就等价于「预测一串 token」，和普通 LLM 生成文字一模一样——这就是 AR 解码器能直接复用 VLM 的原因。

> **注意**：`action_dim=7` 意味着每个时刻的动作是 7 维；而 NAVSIM 评测只关心轨迹（通常 x/y/theta）。tokenizer 把整段动作都离散化了，训练时模型学的是「7 维动作序列」，推理后再按需要取轨迹部分。

## 三、MoE 动作专家：DiT 风格的积木

`models/policy_head/moe_experts.py` 定义了一套 DiT（扩散 Transformer）风格的网络 `Emu3Experts`，它是**动作专家与世界模型共用的 backbone 积木**。我们挑最关键的几块讲。

### 3.1 TimestepEmbedder：把时间步变成向量

Flow Matching / 扩散都要让网络知道「现在是第几步加噪」，用正弦位置编码 + MLP：

```python
class TimestepEmbedder(nn.Module):
    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

**大白话**：时间步 t（一个标量）经过「cos/sin 编码」变成一串和隐维度同长的向量，告诉网络「现在是噪声程度 tau 的时刻」。这和 Transformer 的位置编码同源。

### 3.2 ActionProjector：把噪声动作投影进隐空间

```python
class ActionProjector(nn.Module):
    def forward(self, x, tau):
        out1 = self.W1(x)
        out2 = self.W2(torch.cat([out1, tau], dim=-1))
        return self.W3(out2)
```

- `x`：patchify 后的噪声动作/图像块，形状 `[B, L, d]`。
- `tau`：时间步嵌入，拼到动作特征后面一起过 MLP。
- 输出仍是 `[B, L, dim]`，作为 Transformer 的输入。

**大白话**：动作（或图像块）先过一层线性，再和时间步嵌入拼接过一层线性，最终投到模型隐维度。这一步把「动作」和「现在噪声到啥程度」两件事绑在一起送进网络。

### 3.3 adaLN 调制：让每层随时间步变形

`TransformerBlock` 里用 adaLN（adaptive LayerNorm）代替普通 LN：

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
    self.adaLN_modulation(adaln_input).chunk(6, dim=1)
x = x + gate_msa * self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis)
x = x + gate_mlp * self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
```

其中 `modulate(x, shift, scale) = x * (1 + scale) + shift`。`adaln_input = t + y`（时间步嵌入 + 类别嵌入）。

**大白话**：普通 Transformer 每层归一化是固定的；adaLN 让「时间步 t」通过一个 MLP 生成 6 个参数（shift/scale/gate 各一对），动态决定这一层怎么缩放、平移、以及残差门控多大。这样同一个网络在不同噪声程度 tau 下行为不同——这是 DiT 能「按噪声程度去噪」的关键机制。动作专家与世界模型都用这套积木。

### 3.4 前向：patchify → 投影 → 拼时间步 → 过 N 层 → unpatchify

```python
def forward(self, x, t, y):
    x = self.patchify(x)
    t = self.t_embedder(t)
    y = self.y_embedder(y, self.training)
    adaln_input = (t + y).unsqueeze(1).repeat(1, x.size(1), 1)
    x = self.input_proj(x, adaln_input)
    x = torch.cat([x, adaln_input.unsqueeze(1)], dim=1)
    for layer in self.layers:
        x = layer(x)[0]
    x = x[:, 1:, :]
    x = self.final_layer(x, adaln_input)
    return self.unpatchify(x)
```

- `patchify`：把 `[B, C, H, W]` 的图像/动作图切成小块序列（世界模型分支处理的是「未来帧图像」，所以这里 x 是图）。
- 拼上 `adaln_input` 作为一个额外的 token（类 DiT 的 global token）。
- 过若干 `Emu3DecoderLayer`（来自外部 Emu3 库，本仓库只调用）。
- `final_layer` 用 adaLN 输出最终的「去噪后 / 生成后」小块，再 `unpatchify` 还原成图像。

**大白话（世界模型分支视角）**：把「未来帧该长啥样」拆成小块，加噪声后送进这个 DiT，网络预测「该往哪修」，迭代后还原出清晰未来帧。这就是论文里 AR/Diffusion 世界模型的「网络主体」。注意：论文里 AR 世界模型预测的是**离散 VQ token**（配合 `emu3_tokenizer_navsim.py` 的编码），Diffusion 世界模型预测的是**连续潜表示**——两者共用 `Emu3Experts` 这套积木，区别在预测目标与损失。

## 四、Flow Matching 解码器：动作怎么从噪声「流」出来

这是 DriveVLA-W0 三种解码器之一（论文里它在大数据下 PDMS 91.3，是连续方法里最好的）。代码分两层：`flow_matching.py`（封装）与 `diffusion_policy.py`（PolicyHead 具体实现）。

### 4.1 加噪调度：FlowMatchingScheduler

```python
class FlowMatchingScheduler:
    def add_noise(self, original_samples, noise, timesteps):
        while len(timesteps.shape) < len(noise.shape):
            timesteps = timesteps.unsqueeze(-1)
        timesteps = timesteps.expand_as(noise)
        return (1 - timesteps) * original_samples + timesteps * noise
```

**大白话**：Flow Matching 用**直线概率路径**——在 tau 时刻，样本 = `(1-tau)·真实动作 + tau·噪声`。tau=0 是真实动作，tau=1 是纯噪声。这比扩散的「逐步加噪」简单：一步线性插值就完事。`sample_method="beta"` 时用 Beta(1.5, 1.0) 分布采样 tau，让训练更关注「接近真实动作」的低噪声区间（对应推理时真正会走到的区域）。

### 4.2 训练目标：预测向量场

`flow_matching.py` 的 `forward`：

```python
def forward(self, x, cond):
    t = self.sample_t(b)
    texp = t.view(b, *([1] * (x.dim() - 1)))
    z1 = torch.randn_like(x)
    zt = (1 - texp) * x + texp * z1
    vtheta = self.model(zt, t, cond)
    batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=tuple(range(1, x.dim())))
    return batchwise_mse.mean(), ttloss
```

**大白话**：随机采一个 tau，把真实动作 x 和噪声 z1 按 `(1-tau)x + tau z1` 混成带噪样本 zt；网络 `self.model`（即 `Emu3Experts` 或 `PolicyHead`）预测向量场 `vtheta`；目标是让 `vtheta ≈ z1 - x`（即「从噪声指向真实动作的方向」）。损失就是预测向量场与真值向量场的 MSE。这正是论文里的：

$$ \mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(\psi_t(x), t, c) - (x_1 - x_0)\|^2\right] $$

### 4.3 PolicyHead：噪声动作编码 + 轨迹解码

`diffusion_policy.py` 把 Flow Matching 接到动作上：

```python
class PolicyHead(nn.Module):
    def __init__(self, action_dim, embedding_dim, width, noise_scheduler):
        self.encoder = Emu3ActionEncoder(action_dim, embedding_dim, width)
        self.decoder = MultiLayerAttentionDecoder(embedding_dim, action_dim, num_heads=4, hidden_dim=128, num_layers=1)
        self.noise_scheduler = noise_scheduler

    def forward_loss(self, action_sequence, noise, tau):
        noisy = self.noise_scheduler.add_noise(action_sequence, noise, tau)
        emb = self.encoder(noisy, tau)
        vel_pred = self.decoder(emb)
        target = action_sequence - noise
        loss = F.mse_loss(vel_pred, target)
        return {"loss": loss}
```

- `Emu3ActionEncoder`：把 `[B, T, 7]` 的噪声动作 + 时间步 tau（用正弦位置编码 + RoPE 序列位置）编码成隐式表征。
- `MultiLayerAttentionDecoder`：一层多头注意力 + MLP，把表征解码回 `[B, T, 7]` 的动作（预测向量场）。
- `target = action_sequence - noise`：即 Flow Matching 的真值向量场 `(x1 - x0)`。

**大白话**：动作头和世界模型共享「Flow Matching 训练套路」，但动作头处理的是 7 维动作序列（不是图像），输出也是动作序列。训练时只优化这个轻量 head + 动作专家，主干（理解专家）提供条件特征 `cond`/`c`。

### 4.4 推理采样：沿 ODE 把噪声「流」成轨迹

`flow_matching.py` 的 `sample`：

```python
@torch.no_grad()
def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
    dt = 1.0 / sample_steps
    images = [z]
    for i in range(sample_steps, 0, -1):
        t = i / sample_steps
        vc = self.model(z, t, cond)
        if null_cond is not None:
            vu = self.model(z, t, null_cond)
            vc = vu + cfg * (vc - vu)
        z = z - dt * vc
        images.append(z)
    return images
```

**大白话**：从纯噪声 z 出发（tau=1），按 `z = z - dt · vθ(z, t, cond)` 一步步「往真实动作方向流」，跑 `sample_steps`（默认 50，论文推理可用更少）步到 tau=0，得到最终轨迹。若给 `null_cond` 做 classifier-free guidance（cfg），则 `vc = vu + cfg·(vc - vu)` 放大条件信号。对比 DiffusionDrive 的 2 步 DDIM，这里步数更多（因从纯噪声出发），但胜在路径是直线、训练目标简单。

## 五、前向传播：一张图到一条轨迹（Flow Matching + Emu3 主线）

把上面所有模块串成一次前向（推理视角，世界模型旁路）。输入是 NAVSIM 的前视图像 + 指令 + 历史动作，输出是 `[B, T, 7]` 动作。下面用「纯英文代码块 + 块外中文解说」串步骤（代码块内零中文，避免对齐错位）。

- `inference_action_navsim_flow_matching_vava.py`
  - `load_scene(token)` → `img = [1, 3, 256, 144]` (front cam)
  - `build_vla_sequence(img, instruction, past_actions)`
    - `seq = [L, V, A, ..., L, V, A]`  (interleaved, 2VA in stage2)
  - `Emu3MoE.forward(seq)`  (understanding expert, frozen in stage2)
    - `hidden = [1, N_tokens, 4096]`
  - `ActionExpert.joint_attention(hidden, action_queries)`  (500M MoE)
    - `cond = [1, N_act, 4096]`
  - `z = randn([1, T, 7])`  (noise init)
  - `for step in range(sample_steps): v = PolicyHead(z, t, cond); z = z - dt * v`
  - `traj = z[..., :3]`  (take x, y, theta)
  - `output = [1, T, 3]`

逐步中文含义：

- `load_scene`：从 NAVSIM pickle 取一个场景的前视相机图（256×144，单目前视！）。
- `build_vla_sequence`：把语言指令 L、图像 V、历史动作 A 沿时间**深度交错**拼成 `[L, V, A, ...]` 序列。阶段 2 用 2VA（当前 + 前一帧）以降延迟。
- `Emu3MoE.forward`（理解专家）：Emu3-8B 主干自回归编码整段序列，输出统一的条件特征 `hidden`。阶段 2 主干**冻结**。
- `ActionExpert.joint_attention`：500M 动作专家通过 Joint Attention 与主干特征耦合，得到动作相关的条件 `cond`。这一步是 MoE 的精髓——小专家借用大表征。
- `z = randn`：动作从纯噪声初始化（Flow Matching 起点）。
- 循环采样：每一步 `PolicyHead` 预测向量场 `v`，`z = z - dt·v` 向真实动作流动；跑 sample_steps 步。
- `traj = z[..., :3]`：取前 3 维（x, y, theta）作为最终轨迹，形状 `[1, T, 3]` 交给 NAVSIM 评测。

> 注意：论文里动作是 7 维（`action_dim=7`），评测轨迹只取其中 x/y/theta 部分。其余维度（如油门刹车）在闭环仿真里可被 PDMS 用，但 NAVSIM 评测主要看轨迹几何。

## 六、训练：两阶段范式与目标函数

仓库只给了 `Training.md` 说明 + 上述模块，两阶段逻辑清晰：

| 阶段 | 输入序列 | 监督目标 | 可训练模块 | 作用 |
|:---|:---|:---|:---|:---|
| 阶段1 世界预训练 | 长序列 6VA（12帧） | L_action + alpha·L_WM | VLM 主干 + 世界模型 + 动作头 | 稠密信号喂出世界表征 |
| 阶段2 动作专精 | 短序列 2VA（4帧） | L_action | 仅 Action Expert + 解码器 | 瘦身，实时出轨迹 |

阶段 1 总目标（来自论文，与代码 `vision_loss_weight` 呼应）：

$$ \mathcal{L}_{\text{阶段1}} = \underbrace{\| \hat{a}_t - a_t^* \|_2}_{\text{动作模仿}} + \alpha \cdot \underbrace{\mathcal{L}_{\text{WM}}(\hat{I}_{t+1}, I_{t+1})}_{\text{世界建模}} $$

- 动作部分：`PolicyHead.forward_loss` 的 MSE（Flow Matching）或 focal+CE（AR 解码器）。
- 世界模型部分：AR 分支用交叉熵预测下一帧 VQ token（`emu3_tokenizer_navsim.py` 预编码好的码本）；Diffusion 分支用 MSE 预测下一帧潜表示。两者都提供**每像素/每 token 的稠密监督**。

阶段 2 冻结主干，只训动作专家 + 解码器，输入缩到 2VA。延迟从 117.8ms 降到 74.3ms，PDMS 反而 85.6→88.4——说明阶段 1 学到的世界表征已**蒸馏进主干权重**，去掉世界分支后仍在。

**梯度流小结**：阶段 1 主干 + 世界模型 + 动作头全可训；阶段 2 仅动作专家/解码器可训，主干与世界模型冻结（或移除）。世界模型分支**只在训练时存在**，推理时被完全旁路——这是它「不拖慢决策」的关键。

**训练/推理差异表**：

| 环节 | 训练 | 推理 |
|:---|:---|:---|
| 世界模型分支 | 参与，提供稠密监督 | 旁路（不计算） |
| 序列长度 | 6VA（阶段1）/ 2VA（阶段2） | 2VA |
| 动作解码 | Flow Matching MSE / AR CE / Query L1 | 同，但只走动作通路 |
| 采样步数 | 不需要迭代（直接监督） | Flow Matching 默认 50 步 ODE |
| 主干状态 | 阶段1 可训，阶段2 冻结 | 冻结 |

## 七、自己跑起来（最小实操）

仓库 README 给了 5 分钟示例，下面是整理后的最小路径：

1. **装环境**：

```bash
conda create -n drivevla python=3.10
conda activate drivevla
pip install -r requirements.txt
pip install "transformers[torch]"
pip install deepspeed tensorboard wandb
```

2. **下权重**：从 HuggingFace `liyingyan/DriveVLA-W0` 下载 Emu3 预训练主干与 `Emu3_Flow_Matching_Action_Expert_PDMS_87.2` 等检查点，以及 NAVSIM 的 `navsim_emu_vla_256_144_test_pre_1s.pkl`。

3. **推理（Flow Matching 版，单前视相机）**：

```bash
bash inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh
```

脚本内部调用 `inference/vla/inference_action_navsim_flow_matching_vava.py`，用 `torchrun --nproc_per_node=8` 跑，输出 JSON 动作。

4. **算 PDMS**：

```bash
bash inference/vla/eval_navsim_metric_from_json.sh
```

需要配套 NAVSIM v1.1 仓库环境。

5. **训练**：参考 `Training.md`（两阶段，8×L20 约 16 小时）。

**新手最常踩的 3 个坑**：

- 主干路径硬编码：不少脚本里 `sys.path.append("/share/project/.../Emu3")` 和模型路径是作者内网绝对路径，必须改成自己的 `EMU_HUB` 等环境变量（脚本已用 `${EMU_HUB:-...}` 留了覆盖口）。
- VQ 码未生成：世界模型 AR 分支依赖 `emu3_tokenizer_navsim.py` 提前把图像压成 VQ 码（`.npy`）；漏了这步直接报文件缺失。
- 归一化统计不匹配：动作 token 化与轨迹反归一化依赖 `configs/normalizer_navsim_trainval/norm_stats.json`，换数据集要重新算，否则轨迹尺度会崩。

## 闭环总结

| 阶段 | 入口 | 关键代码 | 训练/推理差异 |
|:----:|:------|:----------|:--------------|
| 动作表示 | `action_tokenizer.py` | FAST 离散化（256 bin） | 编码/解码对称 |
| 动作生成 | `diffusion_policy.py` PolicyHead | Flow Matching MSE 向量场 | 训练直接监督，推理 ODE 采样 |
| 专家耦合 | `moe_experts.py` Emu3Experts | adaLN + Joint Attention 积木 | 主干阶段2冻结 |
| 世界监督 | `emu3_tokenizer_navsim.py` + Emu3Experts | AR 预测 VQ / Diffusion 预测潜表示 | 仅训练，推理旁路 |
| 配置 | `moe_fast_video.json` | 主干32层 / 动作专家2层 / action_dim=7 | 超参集中 |

**一句话记住这个闭环**：连续动作经 FAST tokenizer 离散化、又经 Flow Matching 从噪声「流」出；500M 动作专家通过 MoE 借用 8B 主干表征；训练时世界模型用「预测未来帧」的稠密监督逼主干学懂动力学，推理时世界模型整支旁路，只留轻量动作通路——这就是 DriveVLA-W0「单目打穿多传感器」的秘密。

## 个人理解与思考

**1. 仓库边界本身是个重要信号。** 开源的只有「动作头 + tokenizer + 推理壳」，真正的大模型主干是外部库。这说明 DriveVLA-W0 的工程贡献重心在「如何给 VLA 接上世界模型监督」这一层，而非从零训一个大模型。对想复现的人，重点是搞懂 PolicyHead / MoE / Flow Matching 这层的接口，主干直接复用 Emu3/Qwen2.5-VL。这和 DiffusionDrive（把整套塞进 navsim 插件）的「全栈自研」风格截然不同——也提醒我们：读论文代码时先问「哪层是作者写的」，能少走很多弯路。

**2. Flow Matching 比扩散更适合「动作」这类低维连续量。** 看 `noise_schedulers.py` 与 `flow_matching.py`，Flow Matching 用一条直线路径 `(1-tau)x + tau·noise` 加噪，训练目标就是预测「指向真实动作的方向」(x1 - x0)。没有扩散那套复杂的噪声调度与多步去噪。动作是 7 维小向量，直线流足够；而对图像（世界模型分支）才需要 DiT 的多层去噪。这印证了论文「解码器优劣是数据规模的函数」之外的另一条规律：**任务维度决定生成范式**——低维用 Flow Matching 简洁高效，高维用扩散/AR 表达力强。

**3. adaLN + Joint Attention 是「小专家撬大主干」的精巧设计。** `Emu3Experts` 用 adaLN 让每层随时间步变形，`moe_experts.py` 与配置里「动作专家仅 2 层」则说明：动作专家不靠深度，而靠「和主干做联合注意力」借表征。这比把整个 8B 搬进控制回路省了 37% 延迟（117.8→74.3ms）。作为 VLA 方向工程师，我认为这种「重主干 + 轻动作头」的 MoE 是车端实时部署的现实路线——理解交给大模型（可离线/可慢），动作交给小专家（必须快）。

**4. 世界模型「仅训练时存在」是个聪明的工程取舍，但有隐患。** 它让推理零额外开销，却也意味着：世界模型学到的「动力学理解」必须**完全蒸馏进主干权重**才能惠及推理。阶段 2 去掉世界分支后 PDMS 反而涨，证明蒸馏成功；但一旦换数据分布，主干里「世界理解」的泛化能否撑住，取决于阶段 1 监督的稠密程度。我倾向于认为，未来更稳的做法是「轻量 latent 世界模型常驻」（如 Waymo 方向），而非彻底旁路——毕竟完全旁路会让推理时的「预见性」归零，只剩阶段 1 蒸馏的静态记忆。

**5. 一个延伸想法：动作 tokenizer 的 256 bin 会不会是瓶颈？** `action_tokenizer.py` 把每维切成 256 箱，quantization 误差约 2-3%。论文里 AR 解码器在小数据下受此拖累（PDMS 84.1），大数据下才靠上下文补偿回来。更值得玩味的是：论文发现「解码器优劣是数据规模的函数」——小数据下连续回归（Query/flow）赢，大数据下 AR 反超。这其实和 tokenizer 误差被数据稀释是同一回事。对从业者的启示是：**选解码器前，先问自己处在数据曲线的哪个位置**。一个 7B 团队若只有 10 万帧，上 AR 是自讨苦吃；有 7000 万帧，AR 才是上限所在。

## 延伸阅读

- 同系列对比：[DiffusionDrive 代码讲解](/posts/code/diffusiondrive代码讲解/)——另一条「anchor + 截断扩散」路线
- 论文背景：[DriveVLA-W0 论文精读](/posts/paper-reading/drivevla-w0精读/)——世界模型如何放大数据缩放律
- 榜单对照：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
