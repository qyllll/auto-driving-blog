---
title: "Cosmos 3 代码讲解：Mixture-of-Transformers 双塔统一五模态世界模型"
date: 2026-07-29
description: "拆解 NVIDIA/cosmos-framework：MoT 双塔架构让 Reasoner（自回归语义理解）和 Generator（扩散生成）共享每一层 Transformer 权重但各有独立投影；DCAE 对视频做 4×32×32 极致压缩；Rectified Flow 做生成训练；序列打包（sequence packing）把 und/gen token 拼成一条序列走一次前向"
tags: ["世界模型", "NVIDIA", "MoT", "Cosmos3", "代码讲解", "物理AI"]
categories: ["代码讲解"]
summary: "「NVIDIA Cosmos 3 用 Mixture-of-Transformers 双塔架构把语言、图像、视频、音频、动作五模态统一进一个世界基础模型。本文从 NVIDIA/cosmos-framework 源码出发，面向零基础读者逐文件讲透 MoT 双塔（PackedAttentionMoT / MoTDecoderLayer 的 dual-QKV/proj 设计）、DCAE 视频 tokenizer（4×32×32 因果 VAE）、Rectified Flow 训练（logit-normal 时间分布 + MSE 向量场损失）、序列打包（und/gen token 混排）、以及推理管线（35 步 UniPC 调度 + 多模态生成），最后与 pi0 / Diffusion Planner / AlpaMayo-R1 对照」"
---

> 本文是「代码讲解」路线的第 11 篇。Cosmos 3（NVIDIA，2026）是**物理 AI 世界基础模型**的代表作——它用一个统一的 Mixture-of-Transformers（MoT）架构把语言理解、图像/视频/音频生成、动作预测全揉进了一个模型。代码在官方 **NVIDIA/cosmos-framework**。

## 写给零基础读者：读这篇之前先搞懂几个名词

Cosmos 3 把很多前沿概念揉在了一起。如果你不是每篇都追，可能会被以下黑话劝退。我先把它们全部翻译成人话：

- **世界模型（World Foundation Model）**：一个能「理解物理世界怎么运作」的大模型。给它一段视频，它能预测接下来几秒会发生什么；给它一张图和一句话，它能生成符合物理规律的视频。Cosmos 3 追求的不只是"画面好看"，而是"画面符合物理"——车该沿着路开、球该往下落、人走路该有影子。

- **Mixture-of-Transformers（MoT，混合 Transformer）**：Cosmos 3 最核心的架构创新。传统 Transformer 每一层只有一套 QKV 投影和一套 FFN。MoT 给每一层准备了两套投影（QKV 各两套、FFN 两套）——一套给「理解（Reasoner / und）」用，另一套给「生成（Generator / gen）」用。两套路径共享同一个 Transformer 层的结构，但权重独立。就像一个人有两副眼镜：一副读书（理解），一副看远方（生成），镜片度数不同但共用同一个脑袋。

- **Reasoner 塔（理解塔，und 路径）**：负责语言理解、视觉推理、因果分析。它用**因果自注意力（causal attention）**——只能看前面的 token，不能看后面的，适合"读入一段文字并理解"这种任务。训练目标是 next-token prediction（交叉熵）。

- **Generator 塔（生成塔，gen 路径）**：负责图像、视频、音频、动作的生成。它用**双向注意力（full/bidirectional attention）**——可以看到序列里所有位置的 token，适合"从噪声里逐步雕出图像/视频"这种任务。训练目标是 Rectified Flow（向量场 MSE）+ 可选的交叉熵（离散模态）。

- **Rectified Flow（整流流）**：一种比 DDPM 更直接的生成式训练方法。DDPM 要学"预测加在上面的噪声"，Rectified Flow 直接学"从噪声指向数据的直线向量场"。公式上：给定噪声 x₀ 和真实数据 x₁，在时间 t 做线性插值 x_t = (1-t)x₀ + t x₁，然后让网络预测 v = x₁ - x₀（从噪声指向数据的方向）。用 MSE 损失。优点是推理步数更少、理论更简洁。Diffusion Planner 那篇的代码讲解里也用了类似思路，只不过 Cosmos 3 把它用在了视频/图像生成上。

- **DCAE（Dual-Channel AutoEncoder，双通道自编码器）**：Cosmos 3 自研的视频压缩器。它的作用是**把一段视频压缩成一个小得多的"潜在表示"**，模型在这个小空间里做生成，最后再用 DCAE 解码器把潜在表示还原成视频。压缩率惊人：4 倍时间压缩 × 32 倍空间压缩 = 128 倍总压缩率。一段 128×128×128 的视频（T×H×W），压缩后只剩 1×4×4——也就是 16 个格子。

- **序列打包（Sequence Packing）**：理解 token 和生成 token 需要共存于同一条序列里让 Transformer 处理。序列打包就是决定怎么把两种 token 拼在一起、各自的注意力模式（causal vs full）怎么设置。这是 MoT 架构能否跑起来的关键工程细节。

- **因果 VAE（Causal VAE）**：视频压缩器的一种，它编码当前帧时**只能用过去帧的信息，不能偷看未来帧**。这对世界模型至关重要——因为世界模型要预测未来，编码器如果偷看了未来，模型在预测时就"作弊"了。

- **mRoPE（多维旋转位置编码）**：Cosmos 3 用的位置编码。普通的 RoPE 只编码一维位置（文本token的先后顺序）；mRoPE 编码三维——时间 T、空间高度 H、空间宽度 W。这样模型在看视频 token 时，知道每个 token 在"第几帧的什么位置"。

- **UniPC 调度器**：一种多步扩散采样算法（类似 DDIM 的加速版）。Cosmos 3 推理时用 UniPC 走 35 步从噪声生成视频，比标准 DDPM 的 1000 步快 30 倍。

如果这些你都有模糊印象了，下面看代码就会顺畅很多。

## 为什么要讲 Cosmos 3 的代码

前面十篇我们把端到端自动驾驶和通用机器人 VLA 的方法全摸了一遍。但一直有个底层问题没回答：**这些方法的世界知识从哪来？**

UniAD / VAD 的世界知识来自 3D 检测和 BEV 空间；DriveVLA-W0 的世界知识来自一个显式训练的未来帧预测头；pi0 的世界知识来自冻结的 PaliGemma VLM 主干——但这些都是"隐式默认"的，不是"显式建模"的。

Cosmos 3 反了过来：它 **"世界知识"就是模型本身**。它不是为了某一个具体任务设计的，而是像 GPT 一样先在海量物理数据上训出一个世界基础模型，然后下游任务（驾驶规划、机器人操控、视频仿真）直接用或微调。这是和前面所有文章都不同的范式——**前面是专用模型，这个是通用基座**。

更重要的是，Cosmos 3 的代码结构极其清晰。NVIDIA 把整套框架开源在 `cosmos-framework` 里，模型定义、训练管线、推理引擎、tokenizer、安全 guardrail 全部在一个包内，可读性很高。拆透它，你就能看清"一个工业级世界模型该怎么搭"。

**一句话结论**：Cosmos 3 = MoT 双塔（Reasoner 因果 + Generator 双向自注意，每层两套独立投影） + DCAE 4×32×32 视频压缩 + Rectified Flow 向量场训练 + 序列打包混合 und/gen token + UniPC 35 步快速采样。读代码就从这五块入手。

## 架构总览：先看地图

在钻进任何文件之前，先把 Cosmos 3 一次完整的生成链路在脑子里过一遍。以 text-to-video 为例：

1. **输入**：一段文字 prompt（如"一辆红色汽车在雨天的高速公路上行驶"）。
2. **文本 tokenize**：Qwen2Tokenizer 把 prompt 转成 token id 序列。
3. **噪声初始化**：从标准高斯采样一个噪声视频 latent，形状 `(C, T_latent, H_latent, W_latent)`——经过 DCAE 压缩后的视频空间。
4. **序列打包**：把文本 token（走 und/reasoner 路径）和噪声视频 token（走 gen/generator 路径）拼成一条序列。
5. **MoT 前向传播**：序列进入 Transformer。und token 用因果自注意力（只往前看），gen token 用双向注意力（所有方向都看）。gen token 还可以通过 cross-attention 从 und token 取语义特征。
6. **去噪循环**：在 UniPC 调度下重复 35 步。每步 MoT 输出 gen 部分的向量场预测，调度器根据预测更新噪声 latent。
7. **解码**：35 步结束后，DCAE decoder 把 latent 还原成视频帧。可选的 AVAE decoder 同步生成音频。
8. **安全后处理**：LlamaGuard3 做 prompt/输出筛查，face blur 保护隐私。

用列表把模块边界划死：

- **理解侧（Reasoner / und）**：文本 + 图像 token → causal self-attention → next-token prediction（交叉熵损失）
- **生成侧（Generator / gen）**：噪声 latent → full bidirectional attention → Rectified Flow 向量场（MSE 损失）
- **连接件**：序列打包 + gen→und cross-attention（gen token 从 und token 取语义）
- **压缩器（预处理 + 后处理）**：DCAE encoder（视频→latent）/ decoder（latent→视频）
- **训练目标**：理解用 CE，生成用 Rectified Flow MSE
- **推理调度**：UniPC 35 步

> 关键认知：Cosmos 3 的"全模态"不是给每种模态单独准备一个网络，而是让所有模态共享同一个 MoT 骨架——视觉、语言、音频、动作都变成 token 序列，按各自的注意力方式（causal / full）在同一套权重里处理。这就是"统一"的含义。

## 项目结构：先厘清边界

Cosmos 3 的代码分两个仓库：

- **NVIDIA/cosmos**：项目主页 + cookbook（Jupyter notebook 示例），**不包含模型代码**。
- **NVIDIA/cosmos-framework**：真正的训练/推理框架，全部代码在 `cosmos_framework/` 包下。

我们要读的是后者。下面用 Markdown 嵌套列表画目录树：

- `cosmos-framework/`
  - `cosmos_framework/`
    - `scripts/`
      - `train.py` / `_train.py`：训练入口
      - `inference.py`：推理入口
    - `model/`
      - `generator/`
        - `mot/`
          - `unified_mot.py`：**MoT 核心**（PackedAttentionMoT, MoTDecoderLayer）
          - `omni_mot_model.py`：OmniMoTModel 总装
          - `attention.py`：注意力函数调度
        - `tokenizers/`
          - `dc_ae/`：DCAE 视频 tokenizer（Cosmos 自研，4×32×32 压缩）
          - `audio/avae/`：AVAE 音频 tokenizer
        - `reasoner/`
          - `qwen3_vl/`：Qwen3-VL 稠密 backbone
          - `qwen3_vl_moe/`：Qwen3-VL MoE backbone
          - `nemotron_3_dense_vl/`：Nemotron 3 稠密 VL（Edge 用）
    - `configs/`
      - `base/defaults/`
        - `model_config.py`：OmniMoTModelConfig / DiffusionExpertConfig / RectifiedFlowConfig
        - `tokenizer.py`：所有 tokenizer 注册与参数
        - `reasoner.py`：VLM backbone 配置
    - `data/`
      - `generator/`
        - `sequence_packing/runtime.py`：序列打包（und/gen token 混排）
        - `processors/`：多模态数据处理器
        - `datasets/`：JSONL / WebDataset / LeRobot
    - `inference/`
      - `args.py`：采样参数
      - `model.py`：推理引擎封装
    - `trainer/`：训练循环
    - `auxiliary/guardrail/`：安全过滤（LlamaGuard3, face blur, content filter）
  - `examples/`：SFT 示例配置文件
  - `docs/`：文档

逐文件一句话：

- `unified_mot.py`：MoT 架构的心脏——`MoTDecoderLayer` 定义了双塔共享层的结构，`PackedAttentionMoT` 实现了 und/gen 两套各自独立的 QKV/O 投影和注意力分发。
- `omni_mot_model.py`：把 MoT 层、embedding、head 总装成可调用的 OmniMoTModel。
- `dc_ae/`：DCAE 的实现——4 倍时序 × 32 倍空间压缩的因果 VAE（Causal VAE），训练带感知损失 + GAN 损失 + 时间一致性损失。
- `sequence_packing/runtime.py`：把 und token 和 gen token 按指定模式（causal/full/混合）打包成一条序列。
- `model_config.py`：定义所有模型配置类——从 tokenizer 类型、扩散参数到训练超参。
- `trainer/`：实现 FSDP 分布式训练 + callback 架构。
- `inference/`：推理引擎，支持 diffusers/vLLM/SGLang 后端切换。

> 边界提醒：下面所有代码都是简化后的示意伪代码，用来讲清楚思路。代码块里用英文，中文解释在块外。

## 一、MoT 核心：`mot/unified_mot.py`

这是 Cosmos 3 最值钱的文件。MoT 不是"两个独立的 Transformer 拼在一起"，而是一个 Transformer 层里有**两套独立的投影权重**，根据 token 的路径（und/gen）选择不同的投影做注意力和 FFN。

### 1.1 PackedAttentionMoT：双路径注意力

```python
# model/generator/mot/unified_mot.py (simplified)
import torch
import torch.nn as nn
import torch.nn.functional as F


class PackedAttentionMoT(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        # --- Understanding path projections (causal attention) ---
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

        # --- Generation path projections (full attention) ---
        self.q_proj_moe_gen = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj_moe_gen = nn.Linear(num_heads * head_dim, hidden_size)

        # Optional QK normalization
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.q_norm_moe_gen = RMSNorm(head_dim)
        self.k_norm_moe_gen = RMSNorm(head_dim)
        # Cross-attention: gen queries attending to und KVs
        self.k_norm_und_for_gen = RMSNorm(head_dim)

    def forward(self, hidden_states, attention_mask, position_ids, moe_gen_mask):
        # Split hidden states into und and gen paths
        und_mask = ~moe_gen_mask
        und_h = hidden_states[und_mask]
        gen_h = hidden_states[moe_gen_mask]

        # --- Understanding path: causal QKV ---
        und_q = self.q_norm(self.q_proj(und_h))
        und_k = self.k_norm(self.k_proj(und_h))
        und_v = self.v_proj(und_h)
        und_out = scaled_dot_product_attention(und_q, und_k, und_v, is_causal=True)

        # --- Generation path: full attention + cross-attention to und ---
        gen_q = self.q_norm_moe_gen(self.q_proj_moe_gen(gen_h))
        gen_k = self.k_norm_moe_gen(self.k_proj_moe_gen(gen_h))
        gen_v = self.v_proj_moe_gen(gen_h)
        # Self-attention among gen tokens (full, not causal)
        gen_out = scaled_dot_product_attention(gen_q, gen_k, gen_v, is_causal=False)
        # Cross-attention: gen queries attend to und key/values
        und_k_for_gen = self.k_norm_und_for_gen(self.k_proj(und_h.detach()))
        gen_cross = scaled_dot_product_attention(gen_q, und_k_for_gen, und_v.detach(), is_causal=False)

        gen_out = gen_out + gen_cross  # combine self + cross
        gen_out = self.o_proj_moe_gen(gen_out)
        und_out = self.o_proj(und_out)

        # Reassemble into original order
        output = torch.zeros_like(hidden_states)
        output[und_mask] = und_out
        output[moe_gen_mask] = gen_out
        return output
```

逐块说人话：

- **双套 QKV/O 投影**：`q_proj / k_proj / v_proj / o_proj` 给 und 用；`q_proj_moe_gen / k_proj_moe_gen / v_proj_moe_gen / o_proj_moe_gen` 给 gen 用。它们在初始化时各自独立，训练时各自更新。这是 MoT 最核心的设计——层结构共享但权重不共享。

- **QK 归一化**：两个路径各自独立的 RMSNorm（`q_norm / k_norm` vs `q_norm_moe_gen / k_norm_moe_gen`）。这是 DeepSeek 等模型验证过的技巧——对 QK 做 norm 可以稳定训练、防止 attention logits 爆炸。

- **gen→und cross-attention**：生成 token 不仅自己做 full attention，还用 gen 侧的 query 去 attend und 侧的 key/value。这相当于"动作专家回头看工程师的方案"——gen token 从 und token 取语义特征。`k_norm_und_for_gen` 是专门为跨路径交叉注意力准备的 key norm，因为 und 的 QK norm 和 gen 的 QK norm 可能统计量不同。`und_h.detach()` 确保梯度不反传到 und 路径（和 pi0 的 `no_grad` 思路一样——理解侧冻结或低 lr，生成侧才是训练主力）。

- **注意力分发**：实际实现里 `scaled_dot_product_attention` 会被替换成 `dispatch_attention_fn`，支持三种后端——密集 attention（PyTorch 原生）、neighborhood attention（NATTEN，只关注局部窗口）、以及 KV-cached 推理 attention。这是工程优化，不影响架构理解。

> 一句话澄清：`PackedAttentionMoT` 不是"两个独立的注意力层串行"，而是**一份输入根据 moe_gen_mask 拆成两路，各自走不同的投影但共享同一个注意力函数**。拆开的是权重，不是计算图。

### 1.2 MoTDecoderLayer：双塔一层

有了 PackedAttentionMoT，MoTDecoderLayer 就很简单了——就是一个标准 Pre-LN Transformer 层，但 attention 和 FFN 都是双路的：

```python
class MoTDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.input_layernorm_moe_gen = nn.LayerNorm(config.hidden_size)
        self.self_attn = PackedAttentionMoT(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm_moe_gen = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)  # und path
        self.mlp_moe_gen = MLP(config)  # gen path

    def forward(self, hidden_states, attention_mask, position_ids, moe_gen_mask):
        residual = hidden_states
        # Dual layernorm: und and gen get different norms
        und_normed = self.input_layernorm(hidden_states)
        gen_normed = self.input_layernorm_moe_gen(hidden_states)
        # Merge back: pick norm based on path
        normed = torch.where(moe_gen_mask.unsqueeze(-1), gen_normed, und_normed)
        attn_out = self.self_attn(normed, attention_mask, position_ids, moe_gen_mask)
        hidden_states = residual + attn_out

        residual = hidden_states
        # Dual post-attention norm
        und_normed = self.post_attention_layernorm(hidden_states)
        gen_normed = self.post_attention_layernorm_moe_gen(hidden_states)
        normed = torch.where(moe_gen_mask.unsqueeze(-1), gen_normed, und_normed)
        # Dual FFN
        ff_out = torch.where(
            moe_gen_mask.unsqueeze(-1),
            self.mlp_moe_gen(normed),
            self.mlp(normed),
        )
        hidden_states = residual + ff_out
        return hidden_states
```

逐块说人话：

- `input_layernorm` 给 und，`input_layernorm_moe_gen` 给 gen。`torch.where` 根据 `moe_gen_mask` 为每个 token 选择使用哪种 layernorm。
- 类似的，attention 输出后，再走各自的后注意力 layernorm，最后各自走自己的 FFN（`mlp` vs `mlp_moe_gen`）。
- 所以一个 MoTDecoderLayer 的前向就是：**输入 → 选 norm → 双路注意力 → 加残差 → 选 norm → 双路 FFN → 加残差 → 输出**。每层里 und 和 gen 各走各的权重，但共享同一个 hidden_states 张量和残差连接。

> 关键认知：`moe_gen_mask` 是一个布尔 mask，形状 `(batch * seq_len,)`，标记哪些 token 走 gen 路径、哪些走 und 路径。这个 mask 在序列打包阶段就确定好了，整个前向过程中不变。

下面这张图把 MoTDecoderLayer 的完整数据流画了出来——注意 und 和 gen 各走各的 layernorm、QKV/O 投影、FFN，但在残差连接上共享同一个 hidden_states 张量：

![MoTDecoderLayer 双塔架构](/images/cosmos3/code/mot_architecture.svg)

### 1.3 前向传播的 tensor 形状追踪

为了让前向传播更具体，我们用 Nano 规格（hidden_size=4096）的一次调用追踪各张量的形状变化。假设打包后的序列有 100 个 und token + 400 个 gen token，batch_size=1：

```python
# 形状追踪（Nano 规格：H=4096, n_heads=32, head_dim=128）
input_hidden = torch.randn(1, 500, 4096)   # (B, L, H) = (1, 100+400, 4096)
moe_gen_mask = torch.cat([                  # 前100个False, 后400个True
    torch.zeros(100, dtype=torch.bool),
    torch.ones(400, dtype=torch.bool),
])

# 拆分成两路
und_h = input_hidden[0, :100]               # (100, 4096) — und 路径
gen_h = input_hidden[0, 100:]               # (400, 4096) — gen 路径

# Und 注意力: 32 heads × 128 head_dim
und_q = q_proj(und_h).reshape(100, 32, 128)  # (100, 32, 128)
und_k = k_proj(und_h).reshape(100, 4, 128)   # (100, 4, 128) — GQA: 4 KV heads
und_v = v_proj(und_h).reshape(100, 4, 128)
# causal attention → und_out: (100, 4096)

# Gen 注意力: 同样 32 heads, 不同权重
gen_q = q_proj_moe_gen(gen_h).reshape(400, 32, 128)
gen_k = k_proj_moe_gen(gen_h).reshape(400, 4, 128)
gen_v = v_proj_moe_gen(gen_h).reshape(400, 4, 128)
# full attention → gen_self_out: (400, 4096)

# Cross-attention: gen query × und key/value
gen_cross = cross_attention(gen_q,                           # (400, 32, 128)
    k_norm_und_for_gen(k_proj(und_h)),                       # und 的 key
    v_proj(und_h))                                           # und 的 value
# → gen_cross_out: (400, 4096)

gen_out = gen_self_out + gen_cross  # 组合
# 最后 FFN: 每路各走各的 MLP
# und: MLP(4096 → 12288 → 4096)  (Nano 规格)
# gen: MLP_moe_gen(4096 → 12288 → 4096)
# 输出: (1, 500, 4096) — und 和 gen 重新拼回
```

这个形状追踪揭示了几个关键点：
- **GQA（Grouped Query Attention）**：32 个 query head 只有 4 个 KV head，每组 8 个 query head 共享同一对 KV。这是标准 GQA 做法，减少 KV 缓存显存，对长序列生成帮助极大。
- **gen_out = self_attn + cross_attn**：gen 的最终注意力输出是自注意力（gen token 之间互看）和交叉注意力（gen 看 und）直接**相加**。这意味着 gen token 同时从"自己的邻居"和"理解侧的语义特征"两个来源获取信息。
- **detach() 的实现细节**：`und_h.detach()` 切断了 gen→und 交叉注意力的梯度反传，确保 und 路径的训练信号只来自 next-token prediction 损失。这和 pi0 的 `torch.no_grad()` 思路一致，但粒度更细——detach 只在 cross-attention 分支上生效，und 路径的自注意力仍然接收自己的梯度。

### 1.4 三种模型规格的配置

MoTDecoderLayer 本身和规格无关——差异全在配置里。三个规格的 backbone 分别是：

| 规格 | Backbone | hidden_size | layers | heads | KV heads | intermediate_size |
|------|----------|:---:|:---:|:---:|:---:|:---:|
| Edge (4B) | Nemotron-3 2B | 2048 | 28 | 16 | 4 | 8192 |
| Nano (16B) | Qwen3-VL-8B | 4096 | 36 | 32 | 8 | 12288 |
| Super (64B) | Qwen3-VL-32B | 5120 | 64 | 64 | 8 | 25600 |

所有规格共享同一套 `MoTDecoderLayer` 代码，不同之处只有 `config` 里的数字。这是工程复用做的很干净的地方。

## 二、OmniMoTModel：总装

`omni_mot_model.py` 把 MoT 层、embedding、head 拼装成可调用的模型。核心逻辑：

```python
# model/generator/mot/omni_mot_model.py (simplified)
class OmniMoTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MoTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, moe_gen_mask, position_ids, attention_mask=None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask, position_ids, moe_gen_mask)
        h = self.norm(h)
        return h  # (B, L, H) final hidden states
```

但这里有个重要细节：OmniMoTModel 并不直接管理 und 和 gen 的细节配置——它只是个通用的 Transformer 骨干。真正的"理解 + 生成"双头分工是在更上层的代码里做的：

- **理解头（Reasoner head）**：Qwen3VLModel / NemotronModel 的 `lm_head`——把 hidden states 投影回 vocab 维度的 logits，做 next-token prediction。
- **生成头（Generator head）**：一个 diffusion head——把 gen token 位置对应的 hidden states 投影成向量场预测（在 latent 空间里），再用 Rectified Flow 损失监督。

这种"共享 backbone + 双头"的设计和 pi0 / AlpaMayo-R1 非常像，只不过 Cosmos 3 的双头在每一层都通过 MoT 双路径深度耦合，而不只是最后加一个 head。

```python
# 更完整的 OmniMoTModel 调用示意
class OmniMoTForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = OmniMoTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gen_head = DiffusionHead(config)  # projects gen hidden → vector field

    def forward(self, input_ids, moe_gen_mask, labels=None, gen_targets=None):
        hidden = self.model(input_ids, moe_gen_mask, ...)
        # Und tokens go to lm_head (next-token prediction)
        und_logits = self.lm_head(hidden[~moe_gen_mask])
        # Gen tokens go to gen_head (vector field prediction)
        gen_pred = self.gen_head(hidden[moe_gen_mask])
        loss = 0
        if labels is not None:
            loss += cross_entropy(und_logits, labels)  # reasoner loss
        if gen_targets is not None:
            loss += mse_loss(gen_pred, gen_targets)    # generator loss
        return loss
```

## 三、DCAE：视频 tokenizer

Cosmos 3 自研的 DCAE（Dual-Channel AutoEncoder）是所有视频生成的基础。它的核心功能：把原始视频 `(T, H, W, 3)` 压缩成 latent `(C, T/4, H/32, W/32)`。

### 3.1 压缩率为什么重要

没有 DCAE，Cosmos 3 的 Transformer 根本跑不动。一段 5 秒 24fps 的 720p 视频有 `5×24=120` 帧，每帧 `1280×720×3` 像素——直接在像素空间做 attention 的 token 数会爆炸。DCAE 把 120 帧压缩到 `120/4=30` 个时间步，每帧压缩到 `720/32 × 1280/32 ≈ 23×40 ≈ 920` 个 latent 位置——总量从天文数字降到可管理的级别。

### 3.2 DCAE 架构简化

```python
# model/generator/tokenizers/dc_ae/ (simplified)
class DCAEEncoder(nn.Module):
    def __init__(self, latent_channels=64, downsample_factor=32, temporal_downsample=4):
        super().__init__()
        # Stack of 3D convolutions with increasing channels
        self.stem = nn.Conv3d(3, 128, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            DownsampleBlock3D(128, 256,  stride=(1, 2, 2)),  # spatial /2
            DownsampleBlock3D(256, 512,  stride=(1, 2, 2)),  # spatial /4
            DownsampleBlock3D(512, 512,  stride=(1, 2, 2)),  # spatial /8
            DownsampleBlock3D(512, 512,  stride=(2, 2, 2)),  # temporal/2, spatial/16
            DownsampleBlock3D(512, latent_channels, stride=(2, 2, 2)),  # temporal/4, spatial/32
        ])
        self.causal = True  # causal padding throughout

    def forward(self, video):
        # video: (B, T, H, W, 3) normalized to [-1, 1]
        # permute to (B, C, T, H, W) for Conv3d
        x = video.permute(0, 4, 1, 2, 3)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x  # (B, latent_channels, T//4, H//32, W//32)


class DCAEDecoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        # Symmetric to encoder: upsample blocks
        self.blocks = nn.ModuleList([
            UpsampleBlock3D(latent_channels, 512, stride=(2, 2, 2)),
            UpsampleBlock3D(512, 512, stride=(2, 2, 2)),
            UpsampleBlock3D(512, 512, stride=(1, 2, 2)),
            UpsampleBlock3D(512, 256, stride=(1, 2, 2)),
            UpsampleBlock3D(256, 128, stride=(1, 2, 2)),
        ])
        self.head = nn.Conv3d(128, 3, kernel_size=3, padding=1)

    def forward(self, latent):
        for block in self.blocks:
            latent = block(latent)
        out = self.head(latent)
        return out.permute(0, 2, 3, 4, 1)  # (B, T, H, W, 3)
```

逐块说人话：

- **因果 3D 卷积**：所有卷积层都用因果 padding（即只看当前时间步及之前，不看未来），确保编码器不会"作弊"偷看未来的帧。这是世界模型生成时因果正确性的基础。
- **渐进下采样**：5 个 downsampling block，前三个只压缩空间（H, W 各 /2），第四个开始同时压缩时间（T/2）。最终达到 T/4 和 H/32, W/32。
- **多规格支持**：latent_channels 可取 64/96/128，对应不同的重建质量等级。

### 3.3 CausalVAE3D 与因果卷积细节

DCAE 的核心算子叫 `CausalVAE3D`——一个把因果性嵌入到 3D 卷积每一个算子的模块。它的关键修正在卷积的 padding 策略上：

```python
# dc_ae/modules/causal_conv3d.py (simplified concept)
class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1):
        # Standard Conv3d with causal padding
        # kernel_size assumed (k_t, k_h, k_w)
        pad_t = (kernel_size[0] - 1) * dilation[0]  # only pad past, not future
        pad_h = (kernel_size[1] - 1) * dilation[1] // 2
        pad_w = (kernel_size[2] - 1) * dilation[2] // 2
        # padding: (W_left, W_right, H_top, H_bottom, T_front, T_back)
        # T_back=0 ensures causal: current frame only sees past frames
        padding = (pad_w, pad_w, pad_h, pad_h, pad_t, 0)
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # Standard Conv3d with causal padding already set in __init__
        return super().forward(x)
```

这里最关键的一行是 `padding = (pad_w, pad_w, pad_h, pad_h, pad_t, 0)`——时间维度的 padding 只给过去（pad_t），不给未来（0）。这和 NLP 里的 causal mask 是同一个逻辑，不过实现在了卷积核层面。

CausalVAE3D 的所有下采样/上采样层、残差块、注意力层都基于这个因果 3D 卷积构建。包括其中的 GroupNorm、SiLU 激活、以及可选的 3D 位置编码（用于 mRoPE），都在因果约束下工作。

### 3.4 量化与潜空间正则化

DCAE 在 encoder 和 decoder 之间还有一个量化模块，控制 latent 的分布：

```python
# dc_ae/modules/quant.py (simplified concept)
class DCAEQuantizer(nn.Module):
    def __init__(self, latent_channels=64, quant_type="lpips"):
        super().__init__()
        self.conv_in = nn.Conv3d(latent_channels, latent_channels, 1)
        self.conv_out = nn.Conv3d(latent_channels, latent_channels, 1)
        self.quant_type = quant_type  # "lpips" or "vq"

    def forward(self, x):
        # Apply channel-wise scaling based on perceptual importance
        x = self.conv_in(x)
        if self.quant_type == "vq":
            # Vector quantization (for discrete latent space)
            x, _ = self.vector_quantization(x)
        # "lpips" mode: no actual quantization, just learnable scaling
        x = self.conv_out(x)
        return x
```

实际推理时默认用 `"lpips"` 模式——不做真正的量化，只是通过一个可学习的通道缩放来校准 latent 的空间。`"vq"` 模式则做向量量化（类似 VQ-VAE），把 latent 变成离散 token——这在需要语言模型直接生成 latent token 时使用。

### 3.5 完整的训练损失

DCAE 需要单独训练，不是免费的。它的完整损失函数：

```python
# DCAE training loss components
L_dcae = L_recon + λ_perc · L_perceptual + λ_gan · L_gan + λ_t · L_temporal + β · L_kl
```

各分量含义：

| 损失项 | 公式 | 作用 |
|--------|------|------|
| `L_recon` | L1 + L2 混合 | 像素级重建精度 |
| `L_perceptual` | LPIPS（VGG 特征空间 MSE） | 感知质量——让重建帧在"人眼感知上"和原图一致 |
| `L_gan` | PatchGAN hinge loss | 纹理真实感——补 L1/LPIPS 可能模糊的细节 |
| `L_temporal` | 光流 warp 误差 | 时间一致性——相邻帧不闪烁不跳变 |
| `L_kl` | KL 散度 | 约束 latent 接近标准正态（标准 VAE 正则项） |

PatchGAN 鉴别器也是一个 3D 卷积网络，以视频片段为输入，输出一个 patch-level 的真伪图（每个 patch 判断是否真实），而非整段视频一个分数。这有助于保持局部纹理的真实性。

DCAE 是**独立于 MoT 模型**训练的，训好后冻结使用。MoT 只在 DCAE 的 latent 空间里做生成。这是很经典的两阶段训练——先训好高效的压缩器，再基于压缩空间训生成模型。

> 关键认知：DCAE 的 4×32×32 压缩是 Cosmos 3 能跑起来的前提。没有这个压缩，任何 Transformer 都处理不了视频的 token 数量。这和 Diffusion Planner 用轨迹纬度压缩、pi0 用 action horizon 压缩是同一个逻辑——**先压缩到可计算的空间，再做生成**。

把上面这段编码-解码流程画成图，每个 down/up block 的形状变化一目了然：

![DCAE 4×32×32 压缩流程](/images/cosmos3/code/dcae_compression.svg)

## 四、Rectified Flow 训练

Cosmos 3 的生成侧用 Rectified Flow 训练，而不是 DDPM。这节讲训练配置和目标函数。

### 4.1 配置

```python
# configs/base/defaults/model_config.py (simplified)
@dataclass
class RectifiedFlowTrainingConfig:
    shift: int = 5                     # distribution shift for noise schedule
    train_time_image_distribution: str = "logitnormal"
    train_time_video_distribution: str = "logitnormal"
    train_time_action_distribution: str = "logitnormal"
    loss_scale: float = 1.0
    action_loss_weight: float = 10.0   # action loss weighted higher
    
@dataclass
class RectifiedFlowInferenceConfig:
    scheduler_type: str = "unipc"      # UniPC multi-step
    num_train_timesteps: int = 1000
    shift: int = 1
```

### 4.2 训练损失

```python
def rectified_flow_loss(model, video_latent, text_input_ids, moe_gen_mask):
    # video_latent: (B, C, T_lat, H_lat, W_lat) from DCAE encoder
    # Flatten spatial-temporal dims to sequence
    B, C, T, H, W = video_latent.shape
    x_1 = video_latent.reshape(B, C, -1).transpose(1, 2)  # (B, T*H*W, C)
    
    # Sample noise and time
    noise = torch.randn_like(x_1)
    t = sample_logit_normal((B, 1, 1), shift=5)  # logit-normal distribution
    
    # Linear interpolation: x_t = (1-t) * noise + t * data
    x_t = (1 - t) * noise + t * x_1
    
    # Forward through MoT (packed with text tokens)
    packed_input = pack_sequence(text_input_ids, x_t, moe_gen_mask)
    hidden = model(packed_input, moe_gen_mask)
    
    # Get gen token predictions (vector field)
    gen_hidden = hidden[moe_gen_mask]
    v_pred = gen_head(gen_hidden)  # (B*T*H*W, C)
    
    # Target: v = x_1 - noise (direction from noise to data)
    v_target = x_1 - noise
    
    loss = F.mse_loss(v_pred, v_target.reshape(-1, C))
    return loss
```

逐行说人话：

- `x_1` 是真实数据（从 DCAE 编码器拿到的视频 latent），展开成序列 `(B, N, C)`——N = T×H×W 是 latent token 总数。
- `t` 从 logit-normal 分布采样——它让模型花更多时间在中间噪声程度（t ≈ 0.5）上，因为那里最难学。这和 Diffusion Planner 的均匀采样不同——Cosmos 3 用了更精细的时间分布。
- `x_t = (1-t) * noise + t * x_1`：线性插值，从噪声到数据的直线路径。
- `pack_sequence`：序列打包——把文本 token（und 路径）和视频 latent token（gen 路径）拼成一条序列，附带 `moe_gen_mask` 标记哪些位置走 gen 路径。
- `v_pred`：模型预测的向量场。
- `v_target = x_1 - noise`：直线路径上的恒定向量场方向（从噪声指向数据）。
- 损失就是两者 MSE——最朴素的回归。没有 KL、没有对抗、没有任何花哨。

> 一句话澄清：Rectified Flow 比 DDPM 更直接——DDPM 预测的是加在数据上的噪声（间接），RF 预测的是从噪声到数据的方向（直接）。所以 RF 在推理时需要的步数更少（Cosmos 3 只用 35 步 vs DDPM 的 1000 步）。

## 五、序列打包：und/gen token 如何共存

序列打包是 Cosmos 3 能跑起来的关键工程模块。它在 `data/generator/sequence_packing/runtime.py` 里。

### 5.1 打包逻辑

```python
# data/generator/sequence_packing/runtime.py (simplified)
def pack_sequence(text_tokens, gen_latents, config):
    # text_tokens: (B, L_text) token ids for und path
    # gen_latents: (B, N_gen, C) flattened video/audio latent tokens for gen path
    
    # 1. Create moe_gen_mask: True for gen tokens, False for und tokens
    und_len = text_tokens.shape[1]
    gen_len = gen_latents.shape[1]
    total_len = und_len + gen_len
    
    moe_gen_mask = torch.cat([
        torch.zeros(und_len, dtype=torch.bool),
        torch.ones(gen_len, dtype=torch.bool),
    ])
    
    # 2. Concatenate text + latent along sequence dimension
    packed_input_ids = torch.cat([text_tokens], dim=1)  # text only (latents go through different path)
    
    # 3. Create combined hidden states
    packed_hidden = torch.cat([
        text_embeddings,      # (B, und_len, H)
        gen_proj(gen_latents),  # (B, gen_len, H)
    ], dim=1)
    
    return packed_hidden, moe_gen_mask
```

关键工程细节：

- **位置编码**：und token 和 gen token 使用不同的位置编码策略。und token 用标准的 1D RoPE（文本位置），gen token 用 mRoPE（3D 位置：时间 T + 空间 H + 空间 W）。最终拼接时各自加上各自的位置编码。
- **注意力 mask**：und 部分用 causal mask（每个 token 只能看自己和之前的），gen 部分用 full mask（所有 gen token 互看）加上 gen→und cross-attention mask（gen 可以看所有 und token 但 und 不能看 gen）。
- **最大 token 数**：默认 `max_num_tokens_after_packing = 13312`——经过 DCAE 压缩后，这个限制大约允许 5 秒 24fps 的视频生成。

```python
# 注意力 mask 示意
# und tokens (causal):  ✓ 每行只能看当前列及以左
# gen tokens (full):    ✓ 任意 gen 看所有 gen
# gen→und cross:       ✓ gen 看所有 und
# und→gen:             ✗ 不看（因果上 und 不能看到未来的生成结果）

# mask 形状: (total_len, total_len)
# und1 und2 gen1 gen2
# und1  ✓   ✗   ✗   ✗
# und2  ✓   ✓   ✗   ✗
# gen1  ✓   ✓   ✓   ✓
# gen2  ✓   ✓   ✓   ✓
```

把序列打包的拼接方式和注意力 mask 结构画成一张图，und/gen 各自的位置编码和注意力范围一目了然：

![序列打包与注意力 mask](/images/cosmos3/code/sequence_packing.svg)

## 六、推理管线

推理入口在 `scripts/inference.py`，核心生成循环在 `inference/` 下。看 text-to-video 的简化流程：

```python
# scripts/inference.py (simplified concept)
@torch.no_grad()
def text_to_video(prompt, model, dcae, tokenizer, num_steps=35, guidance_scale=6.0):
    # 1. Tokenize text prompt
    text_ids = tokenizer.encode(prompt)  # (B, L_text)
    
    # 2. Initialize noise in latent space
    # For 5s @24fps video: T=120 → T_lat=30 (DCAE /4), H=320→10 (/32), W=512→16 (/32)
    noise = torch.randn(1, 64, 30, 10, 16)  # (B, C, T_lat, H_lat, W_lat)
    latent = noise.clone()
    
    # 3. UniPC sampling loop
    scheduler = UniPCScheduler(num_train_timesteps=1000, num_inference_steps=35)
    timesteps = scheduler.get_timesteps()
    
    for t in timesteps:
        # Pack und (text) + gen (noisy latent) tokens
        gen_tokens = flatten_latent(latent)  # (B, N_gen, C)
        packed_hidden, moe_gen_mask = pack_sequence(text_ids, gen_tokens, ...)
        
        # MoT forward: get gen path vector field
        hidden = model(packed_hidden, moe_gen_mask)
        gen_hidden = hidden[moe_gen_mask]
        v_pred = model.gen_head(gen_hidden)
        
        # Reshape back to latent space
        v_pred = unflatten_to_latent(v_pred, latent.shape)
        
        # Classifier-free guidance
        v_uncond = ...  # forward with empty prompt
        v_guided = v_uncond + guidance_scale * (v_pred - v_uncond)
        
        # Scheduler step
        latent = scheduler.step(v_guided, t, latent)
    
    # 4. Decode latent to video
    video = dcae.decoder(latent)  # (B, T, H, W, 3)
    return video
```

逐块说人话：

- **噪声初始化**：在 DCAE latent 空间采样高斯噪声。实际分辨率取决于配置——512p 视频对应的 latent 是 64 通道 × 30 时间步 × 16 高 × 26 宽。
- **UniPC 调度**：35 步多步采样。比 DDPM 的 1000 步快 30 倍，比 DDIM 的 50 步也快。Cosmos 3 还有个 4 步蒸馏版本（fixed-step sampler），适合超低延迟场景。
- **Classifier-free guidance（CFG）**：每次前向算两次——一次有条件（带 prompt），一次无条件和空 prompt。然后外推：`v_guided = v_uncond + scale * (v_cond - v_uncond)`。scale=6.0 是默认值。这确保生成的视频和 prompt 相关度高。
- **解码**：35 步走完，DCAE decoder 把 latent 还原回像素视频。

> 关键认知：推理时 gen token 全部是双向注意力，并且可以看到所有 und token（包括完整的 prompt）。这和训练时 und token 用因果注意力只能看前文不同——推理时 gen 路径有"上帝视角"，可以基于完整的语义描述做生成。

把 Rectified Flow 训练和推理的完整流程画出来，噪声→数据直线路径、MSE 向量场损失、35 步 UniPC 调度全部一目了然：

![Rectified Flow 训练与推理](/images/cosmos3/code/rectified_flow.svg)

## 七、和本系列其他文章的关系

把 Cosmos 3 放进代码讲解坐标系里，它的位置很特别：

- **对比 Diffusion Policy / Diffusion Planner（扩散生成范式）**：那些方法只用扩散做动作/轨迹生成，backbone 是 U-Net 或标准 DiT。Cosmos 3 的 MoT 把扩散 DiT 和自回归 VLM 整合到了一个 Transformer 里，生成和理解不再是两个独立模型。

- **对比 pi0（VLM + 动作专家并联）**：pi0 是 PaliGemma VLM 主干 + Flow Matching 动作专家，两座塔权重独立、仅在推理时通过 cross-attention 交互。Cosmos 3 的 MoT 是每层都有双投影，交互更密集、更深度。两者思路同源——理解用因果、生成用双向——但 Cosmos 3 的耦合更紧。

- **对比 AlpaMayo-R1（VLM + 轨迹专家）**：AlpaMayo-R1 也是 VLM 主干 + expert 模块的并联结构，在语种上更接近 Cosmos 3 的 MoT。但 AlpaMayo-R1 的 expert 是专为驾驶轨迹设计的，Cosmos 3 的 gen 路径是通用的扩散生成器，能处理图像、视频、音频、动作。

- **对比 AutoVLA / DriveVLA-W0（离散动作 token 派）**：Cosmos 3 不用离散 token，走 Rectified Flow 连续生成路线。和 pi0 一样，它也是"连续派"的代表。

## 八、个人思考

**1. MoT 双塔本质上是"结构化的专家混合"。** 传统 MoE 在 FFN 层做路由——token 动态选择不同的 FFN 专家。MoT 在 attention 和 FFN 上同时做路由，但路由策略不是动态的（不依赖 token 内容决定走哪条路），而是静态的（und/gen 在序列打包时就定死了）。这种"静态路由"的好处是简单、稳定，但代价是灵活性不如动态路由。我认为这是一个务实的选择——世界模型的稳定性比灵活性重要。

**2. 因果 VAE + Rectified Flow 的组合是防止"世界模型作弊"的关键。** 很多视频生成模型生成的视频"看起来好看但物理错误"——那是因为生成过程没有因果约束。Cosmos 3 从编码器（DCAE causal padding）到训练目标（RF 路径）到推理（时序逐步生成），层层加了因果约束，物理一致性才得以保证。这对自动驾驶世界模型尤其重要——你不能让模型"预知"未来再生成过去。

**3. DCAE 的 4×32×32 压缩率是一个惊人的工程成就。** 128 倍的压缩几乎无损（视频质量在 FVD/PSNR 指标上接近无损压缩），这意味着 MoT 可以在一个极其紧凑的 token 空间里工作。这个思路对自动驾驶也很有启发——DrivoR 把视觉 token 压缩了 256 倍（6144→24），DCAE 把视频压缩了 128 倍——"压缩然后生成"是一个通用的物理 AI 范式。

**4. 序列打包的设计揭示了注意力模式的根本差异。** 理解需要因果（因为语言是顺序的、知识是累积的），生成需要双向（因为图像/视频的空间结构需要全局感知）。强行用同一种注意力模式处理两种任务会两头不讨好。MoT 的解决方案——同层双路径——优雅地解决了这个矛盾。这对我们设计多模态架构是一个重要参考。

**5. Cosmos 3 给"VLA + 世界模型"指出了一条融合方向。** 现有的驾驶 VLA 要么不做显式世界模型（AutoVLA/pi0），要么把世界模型当成一个单独的预测头来训（DriveVLA-W0）。Cosmos 3 展示了另一种可能：世界模型的"生成能力"和 VLA 的"理解能力"可以统一到一个共享骨干里。这在驾驶上的应用前景非常吸引人——一个模型既能"看懂"场景，又能"预测"未来，还能"生成"规划。

## 九、Config 系统：配置即代码

Cosmos 3 的配置系统在 `configs/base/defaults/model_config.py` 中。它不是简单的 YAML/JSON，而是用 Python dataclass + OmegaConf 的分层注册系统。每一类组件都有自己独立的配置类：

```python
# configs/base/defaults/model_config.py (simplified structure)
@dataclass
class OmniMoTModelConfig:
    # Transformer backbone
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    
    # MoT specific
    moe_gen_enabled: bool = True  # toggle MoT on/off
    # Use separate layernorm for gen path
    moe_gen_layernorm: bool = True
    # Cross-attention config
    moe_gen_cross_attention: bool = True
    moe_gen_k_norm_und: bool = True
    
@dataclass
class DiffusionExpertConfig:
    # Rectified Flow training
    shift: int = 5
    loss_scale: float = 1.0
    action_loss_weight: float = 10.0
    
    # Inference
    scheduler_type: str = "unipc"
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    
@dataclass
class TokenizerConfig:
    video_tokenizer: str = "dcae"  # or "cosmos_videovae"
    audio_tokenizer: str = "avae"
    tokenizer_type: str = "CausalVAE3D"
    latent_channels: int = 64
    base_channels: int = 128
    channel_multipliers: tuple = (1, 2, 4, 4)
    down_samples: tuple = (1, 2, 2, 2)  # temporal down factors
    spatial_compression: int = 32
    temporal_compression: int = 4

@dataclass
class ReasonerConfig:
    backbone: str = "qwen3_vl"  # or "nemotron_3_dense_vl"
    freeze_backbone: bool = False
    enable_gradient_checkpointing: bool = True
```

这个配置体系的核心优势：

- **分层组合**：`OmniMoTModelConfig` 定义骨干结构，`DiffusionExpertConfig` 定义生成头，`TokenizerConfig` 定义压缩器，`ReasonerConfig` 定义 VLM。最终模型通过组合这些配置类来完整描述。
- **OmegaConf 合并**：所有 dataclass 被 OmegaConf 递归注册，base config 在 `configs/base/` 里，实验配置通过 `--config-name` 覆盖特定字段。可以理解为"Python 版 Hydra"。
- **每个规格一份配置**：Edge/Nano/Super 各维护一个 YAML，只列出和 base 不同的部分：

```yaml
# configs/experiment/nano_text_to_video.yaml
defaults:
  - base_text_to_video
  - _self_

model:
  hidden_size: 4096
  num_hidden_layers: 36
  num_attention_heads: 32
  moe_gen_enabled: true

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 1
```

这种设计让配置管理的复杂度可控——base 值改一次，所有实验同步更新。

## 十、Trainer 多阶段训练管线

Cosmos 3 的训练在 `trainer/` 目录下实现，基于 PyTorch FSDP（Fully Sharded Data Parallelism）。整个训练分为三个阶段：

### 10.1 预训练阶段（Pretraining）

第一阶段的 MoT 骨干在 DCAE latent 空间上从头训练生成能力。数据来源是互联网视频 + 内部数据集，经过以下流程：

```python
# trainer/train.py (simplified concept)
class CosmosTrainer:
    def __init__(self, config):
        self.model = OmniMoTForCausalLM(config)
        self.dcae = load_frozen_dcae()  # frozen tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        
    def train_step(self, batch):
        # batch: video, text, audio, action (depending on task)
        video_latent = self.dcae.encoder(batch.video)  # (B, C, T/4, H/32, W/32)
        
        # Pack sequence: text tokens (und) + video latents (gen)
        packed, moe_gen_mask = pack_sequence(batch.text, video_latent)
        
        # Rectified Flow loss on gen path
        loss = self.compute_rf_loss(packed, moe_gen_mask, video_latent)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

关键训练参数：
- **Batch 构建**：每个 batch 包含来自不同视频的片段，text→video 对。文本用 Qwen2Tokenizer tokenize 后送入 und 路径，视频经 DCAE 压缩后送入 gen 路径。
- **FSDP 分片**：模型参数、梯度、优化器状态在所有 GPU 间分片。对于 Super（64B）规格，这几乎是强制的——单卡放不下完整模型。
- **梯度 checkpointing**：MoT 的每一层 swap 激活值到 CPU 或重新计算，用计算换显存。

### 10.2 监督微调（SFT）

预训练之后是 SFT 阶段，用高质量标注数据微调模型在特定任务上的表现：

- **text-to-video**：高质量 caption—video pair，提升 prompt 对齐度。
- **image-to-video**：首帧条件化，模型学习从静态图推断运动。
- **text-to-action**：用 LeRobot 等机器人数据集微调，让 gen 路径输出动作序列。
- **video-to-text**：用 video captioning 数据微调 und 路径，提升视频理解能力。

SFT 阶段可以只微调 gen 路径或 und 路径，也可以一起微调。配置控制：

```python
# 冻结 und 路径，只训练 gen 路径
for name, param in model.named_parameters():
    if 'moe_gen' not in name:
        param.requires_grad = False
```

这和 pi0 的"先预训专家、再微调专家"的思路高度一致。

### 10.3 RLHF + 奖励模型

Cosmos 3 还可以接 RLHF 阶段，用偏好数据进一步优化生成质量。偏好信号来自：
- **视频质量评分器**：基于 FVD / CLIP score / aesthetic score 训练的奖励模型。
- **人类偏好数据**：人工标注的视频对比对（A/B test）。
- **物理一致性检查器**：检测视频中是否出现物理不合理现象（物体穿透、重力异常等）。

整个训练管线可以用一张图总结：

```
视频 → DCAE 编码 → latent (gen 路径) ─┐
                                       ├ → MoT → gen 输出 → MSE loss → 反向传播
文本 → Tokenizer → token ids (und) ────┘
                                          ↑
                                    冻结 DCAE
```

## 十一、多模态生成模式

Cosmos 3 支持五种模态输入和多种生成模式。看推理代码中的模式分发：

```python
# scripts/inference.py (simplified routing)
def run_inference(args):
    if args.mode == "text_to_video":
        return text_to_video(args.prompt, ...)
    elif args.mode == "image_to_video":
        return image_to_video(args.image, args.prompt, ...)
    elif args.mode == "video_to_video":
        return video_to_video(args.input_video, args.edit_prompt, ...)
    elif args.mode == "text_to_action":
        return text_to_action(args.prompt, ...)
    elif args.mode == "text_to_audio":
        return text_to_audio(args.prompt, ...)
    elif args.mode == "text_and_image_to_video":
        return text_and_image_to_video(args.prompt, args.image, ...)
```

每种模式的工程差异在于序列打包策略不同：

| 模式 | und token | gen token | 注意点 |
|------|-----------|-----------|--------|
| text→video | 仅文本 | 视频 latent | 标准打包 |
| image→video | 文本 + 图像 token | 视频 latent | 图像 token 也走 und 路径（causal） |
| video→video | 文本 + 原视频 token | 编辑后的视频 latent | 原视频可以走 und 或 gen，取决于是否允许编辑因果 |
| text→action | 仅文本 | 动作序列 | 动作是连续向量，用 linear proj 映射到 hidden |
| text→audio | 仅文本 | 音频 latent | 音频用 AVAE 压缩，类似 DCAE 但处理 1D 信号 |
| text+image→video | 文本 + 图像 token | 视频 latent | 图像作为首帧条件 |

其中 image→video 最具代表性——它展示了 Cosmos 3 如何用"条件化首帧"来约束生成：

```python
# image_to_video 的核心逻辑
def image_to_video(image, prompt, model, dcae, ...):
    # 1. 把输入图过一遍 DCAE encoder，拿到 latent
    img_latent = dcae.encoder(image.unsqueeze(1))  # (B, C, 1, H_lat, W_lat)
    
    # 2. 噪声 latent 只在 t>0 的时间步上初始化
    noise = torch.randn(B, C, T_lat - 1, H_lat, W_lat)
    # 首帧直接用 img_latent，后续帧从噪声开始去噪
    latent = torch.cat([img_latent, noise], dim=2)
    
    # 3. 序列打包时，首帧 token 走 und 路径（提供条件），
    #    后续帧走 gen 路径（生成）
    # 这样 und 路径看到了"开头长什么样"，gen 路径负责"延续下去"
```

这里模型学到的是：给定首帧（und 路径看到的视觉上下文），预测后续帧在 latent 空间中的向量场。这和视频预测（Video Prediction）任务的范式完全一致——只不 Cosmos 3 用的是扩散生成而不是确定性回归。

## 十二、Guardrail 安全系统

Cosmos 3 在生产部署时集成了多层安全 guardrail。代码在 `auxiliary/guardrail/` 下：

```python
# auxiliary/guardrail/ (simplified concept)
class SafetyGuardrail:
    def __init__(self):
        self.prompt_classifier = LlamaGuard3ForSequenceClassification(...)
        self.output_classifier = LlamaGuard3ForSequenceClassification(...)
        self.face_blur = FaceBlurModule()
        self.content_filter = ContentFilter()
        
    def check_prompt(self, text):
        # 检查输入 prompt 是否包含不安全内容
        result = self.prompt_classifier(text)
        if result.unsafe:
            return False, result.violated_categories
        return True, None
    
    def filter_output(self, video):
        # 对生成的视频做后处理
        video = self.face_blur(video)       # 人脸模糊
        video = self.content_filter(video)  # 内容过滤（暴力/色情检测）
        return video
```

完整 guardrail 流程：

1. **Prompt 过滤**：用户输入的 prompt 先过 LlamaGuard3，检测是否有暴力、色情、仇恨言论等不安全内容。如果不安全，直接拒绝生成。
2. **输出过滤**：生成的视频逐帧过内容分类器，检测是否有不合规内容。
3. **人脸模糊**：检测到的所有人脸做高斯模糊——这是隐私合规的硬性要求，尤其对自动驾驶场景（路人的隐私保护）。
4. **Wurst 水印**：在生成视频上嵌入不可见数字水印，标示内容由 AI 生成。这是应对 deepfake 和版权问题的行业标准做法。

这个 guardrail 系统不是 Cosmos 3 的学术创新，但它是从"研究原型"到"可部署产品"必不可少的一步。

---

> 小结：Cosmos 3 的代码核心就六件事——MoT 双塔（`PackedAttentionMoT` / `MoTDecoderLayer` 的双套投影）、DCAE 4×32×32 因果视频压缩（`dc_ae/` + CausalConv3d + 多损失联合训练）、Rectified Flow 向量场训练（MSE loss + logit-normal t 分布）、序列打包（und/gen token 混排 + 混合注意力 mask）、UniPC 35 步推理、以及多层 Guardrail 安全过滤。它和 pi0 / Diffusion Planner 共享了"理解因果 + 生成双向 + 交叉注意力传递语义"的底层思想，但把结构从"两个独立模型"推进到了"一个共享骨架的双路径"。从配置管理（OmegaConf dataclass 分层组合）到训练管线（FSDP 预训练 → SFT → RLHF），再到推理部署（UniPC 调度 + guardrail + 水印），Cosmos 3 展示了工业级世界模型的全栈工程实践。
