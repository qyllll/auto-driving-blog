---
title: "知识点拆解｜Transformer与注意力机制基础"
date: 2026-07-19
draft: false
categories: ["知识点拆解"]
tags: ["🧠 Transformer","⚡ 注意力","👁️ 视觉","📐 架构"]
summary: "Transformer 凭借自注意力机制的长程依赖建模与并行计算能力，从 NLP 席卷至自动驾驶全栈。本文从缩放点积注意力的数学原理出发，系统梳理 Multi-Head Attention、ViT、BEVFormer 及 VLM 中 Causal Attention 的关键技术。为理解 Transformer 在自动驾驶感知与规划中的应用奠定理论基础。"
weight: 15
---

## 🧠 引言：为什么 Transformer 统治了自动驾驶？

2017 年 "Attention Is All You Need" 开启了深度学习的新纪元。Transformer 最初为 NLP 设计，但凭借其**长程依赖建模**和**并行计算**优势，迅速席卷了计算机视觉（ViT）、多模态融合（BEVFormer）、以及端到端自动驾驶（UniAD）。

本文从数学原理出发，系统梳理注意力机制的演变及其在自动驾驶中的关键技术。

---

## ⚡ Self-Attention：核心数学

### 基本公式

Self-Attention 的核心是一个**缩放点积注意力**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 Q (Query)、K (Key)、V (Value) 由输入 X 经线性变换得到：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

### 计算步骤详解

| 步骤 | 操作 | 输出维度 | 含义 |
|------|------|----------|------|
| 1 | Q × K^T | N × N | 每对token的相似度得分 |
| 2 | 除以 √d_k | N × N | 防止softmax梯度消失 |
| 3 | Softmax 行归一化 | N × N | 注意力权重（和为1） |
| 4 | × V | N × d | 加权聚合所有token信息 |

### 为什么除以 √d_k？

当 d_k 很大时，Q·K 的内积方差也大，softmax 会集中在少数极大值上，梯度极小。除以 √d_k 将方差缩放到 1，保持梯度健康。

![Scaled Dot-Product Attention：QKV计算流程](/images/transformer-basics/attention.png)

---

## 🔢 Multi-Head Attention：并行关注不同子空间

### 多头机制

单一注意力只能关注一种交互模式。**多头注意力**（Multi-Head Attention, MHA）让模型并行学习 h 个不同的注意力子空间：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

其中每个 head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)

### 头的分工

在实践中，不同 head 会自动学习关注不同类型的关系：

| Head 编号 | 关注模式 | 在自动驾驶中的对应 |
|-----------|----------|-------------------|
| Head 1-2 | 局部邻居 | 相邻车道车辆交互 |
| Head 3-4 | 全局关系 | 远处道路拓扑 |
| Head 5-6 | 语义模式 | 车道线/交通灯 |
| Head 7-8 | 时序关系 | 历史轨迹关联 |

**复杂度分析**：MHA 的计算量与 $N^2$ 成正比（N 为 token 数），这是 Transformer 在长序列场景的最大瓶颈。

---

## 📍 Positional Encoding：给 Transformer 注入位置信息

### 为什么需要？

Self-Attention 是**排列等变**（Permutation Equivariant）的——打乱输入顺序，输出也会按相同顺序打乱，但值不变。这意味着模型不知道"token 1"和"token 2"的位置关系。

### 正弦位置编码

原版 Transformer 使用固定频率的正弦/余弦函数：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**特性**：
- 任意位置都能编码
- 可以外推到更长序列
- 不同频率编码不同尺度的位置关系

### 可学习位置编码 vs 旋转位置编码

| 类型 | 特点 | 外推能力 | 常见应用 |
|------|------|----------|----------|
| 正弦编码 | 固定频率 | ✅ 强 | Transformer |
| 可学习编码 | Embedding 表 | ❌ 有限 | BERT, ViT |
| RoPE (旋转) | 旋转矩阵相乘 | ✅ 强 | **LLaMA, GPT-4** |
| ALiBi | 偏置项 | ✅ 强 | BLOOM |

**RoPE**（Rotary Position Embedding）是当前主流大模型的首选。它通过旋转矩阵将位置信息编码到 Q 和 K 中，兼具相对位置编码的外推能力和绝对位置编码的明确性。

---

## 👁️ ViT：Transformer 进军视觉

### Patch Embedding

ViT（Vision Transformer）将图像分割为 $N = H \times W / P^2$ 个 patch，每个 patch 展平后线性投影为 token。加上 **[CLS] token** 用于分类，加上位置编码保留空间信息。

### ViT vs CNN

| 维度 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 局部性 + 平移不变性 | 最少（全靠学习） |
| 感受野 | 逐层扩大 | 第一层即全局 |
| 数据效率 | 小数据集好 | **大数据集必须** |
| 计算复杂度 | O(N) | O(N²) |
| 高分辨率 | ✅ 灵活 | ❌ 平方增长 |

### Swin Transformer 的改进

Swin Transformer 引入**分层注意力 + 窗口偏移**，将计算复杂度从 O(N²) 降到 O(N)，在 COCO 检测等任务上首次超越了 CNN 基线。

---

## 🔗 Cross-Attention：多模态融合的桥梁

### 公式

Cross-Attention 与 Self-Attention 的唯一区别：**Q 来自一个模态，K 和 V 来自另一个模态**。

$$\text{CrossAttention}(Q_X, K_Y, V_Y) = \text{softmax}\left(\frac{Q_X K_Y^T}{\sqrt{d_k}}\right)V_Y$$

### 在 BEV 融合中的应用

**BEVFormer** 使用 Deformable Cross-Attention 将多视角图像特征投影到 BEV 空间：

1. **BEV Query**：BEV 网格上的可学习 query
2. **Cross-Attention**：每个 query 在图像视角上采样参考点
3. **Deformable 机制**：只关注参考点附近的稀疏位置，而非全图

这种方法比密集的全局 Cross-Attention 计算量降低了一个数量级，同时保持了感知精度。

### 其他应用场景

| 场景 | Q 来源 | K/V 来源 | 作用 |
|------|--------|----------|------|
| 多模态融合 | BEV query | 图像特征 | 2D→3D 投影 |
| 文本条件生成 | 文本 token | 图像特征 | 文生图 (Cross-Attn in SD) |
| 目标查询 | object query | 特征图 | DETR 检测器 |
| 轨迹预测 | agent query | 场景特征 | 交互建模 |

---

## 🔒 Causal Attention：自回归生成的基石

### Masked Self-Attention

**Causal Attention**（因果注意力）的核心是**掩码**：每个 token 只能关注自己及之前的 token，不能看到未来的 token。

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 M 是上三角掩码矩阵（$M_{ij} = -\infty$ 当 j > i）。

### Causal LM vs Encoder-Only

![Transformer Encoder-Decoder整体架构](/images/transformer-basics/transformer-arch.png)

| 架构 | 注意力类型 | 代表模型 | 适用任务 |
|------|-----------|----------|----------|
| Encoder-Only | 双向（全可见） | BERT | 分类, NER |
| Decoder-Only | **Causal** | GPT, LLaMA | 生成 |
| Encoder-Decoder | 双向 + Cross | T5 | 翻译, 摘要 |

### VLM 中的 Causal Attention

在视觉语言模型（VLM）中，视觉 token 和文本 token 共用 causal attention：

```
[IMG₁, IMG₂, ..., IMG_N, |, TXT₁, TXT₂, ..., TXT_M]
```

- 图像 token 之间：全双向（实际应用中有变体）
- 文本 token 之间：causal mask
- 文本 token → 图像 token：全可见

这保证了文本生成时只能看到已生成的 token 和所有图像信息。

---

## ⚡ KV Cache：推理加速的核心

### 为什么需要 KV Cache？

在自回归推理中，生成第 t 个 token 时，前 t-1 个 token 的 K 和 V 矩阵已经计算过。**KV Cache** 将这些中间结果缓存下来，避免重复计算。

| 阶段 | 无 KV Cache | 有 KV Cache |
|------|-------------|-------------|
| 预填充（第1个token） | 计算全部 QKV | 计算全部 QKV + 缓存 KV |
| 解码（后续 token） | 重新计算所有 K、V | 只需计算新 token 的 K、V |
| 复杂度 | O(N²) 每步 | O(N) 每步 |
| 加速比 | 1x | **10-100x**（长序列） |

### Multi-Query Attention (MQA) 与 Grouped-Query Attention (GQA)

为了进一步缩减 KV Cache 大小：
- **MQA**：所有 head 共享一组 K、V —— 显存降至 1/h
- **GQA**：分组共享 K、V —— MHA 和 MQA 的折中，LLaMA 2/3 采用

---

## 🧩 注意力机制在自动驾驶中的应用全景

| 任务 | 注意力类型 | 作用 | 代表方法 |
|------|-----------|------|----------|
| 2D 检测 | Multi-Head Self-Attn | 特征增强 | DETR |
| 3D 检测 | Cross-Attn | 2D→3D 投影 | BEVFormer |
| 轨迹预测 | Self + Cross | 交互建模 | Wayformer |
| 在线地图 | Cross-Attn | 地图元素检测 | MapTR |
| 端到端规划 | Causal + Cross | 时序决策 | UniAD |
| VLM 推理 | Causal | 文本生成 | DriveLM |

---

## 💭 个人思考

Transformer 在自动驾驶领域的渗透速度远超预期。从最初的感知模块开始，如今已覆盖预测、决策、规划的每个环节。

关键洞察：
1. **Attention 本质是信息路由** — 让模型学会自己决定"从哪里获取信息"，这比手工设计的融合策略更灵活
2. **计算效率是永恒瓶颈** — O(N²) 的复杂度在 BEV 大分辨率和长时序场景下极为棘手。Deformable Attention、Linear Attention、Mamba 等方案都在尝试突破
3. **Chinchilla Law 同样适用** — 自动驾驶 Transformer 也需要更大的数据和模型才能显示出 Scaling 能力

未来值得关注的方向：**Sparse Attention**（减少冗余计算）和 **多模态共用 Transformer Backbone**（统一感知-预测-规划），这些将是推动自动驾驶模型能力提升的关键杠杆。
