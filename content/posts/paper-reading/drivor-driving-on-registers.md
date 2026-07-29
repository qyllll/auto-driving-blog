---
title: "论文精读｜DrivoR: Driving on Registers — 基于寄存器token的高效端到端自动驾驶"
date: 2026-07-29
draft: false
categories: ["论文精读", "自动驾驶"]
tags: ["🚗 端到端自动驾驶", "⚡ ViT寄存器", "🧠 Token压缩", "📊 CVPR 2026"]
weight: 29
summary: "DrivoR提出了一种极简的纯Transformer端到端自动驾驶架构，利用ViT寄存器token将多相机视觉特征压缩为紧凑场景表示，仅需~40M参数即可在NAVSIM和HUGSIM上达到SOTA。"
---

## 📄 论文信息

<img src="/images/paper/2601.05083/figure1.png" alt="DrivoR架构总览" style="width:100%; max-width:900px; display:block; margin:0 auto;">
**图1: DrivoR架构概览**。由三个Transformer模块组成：感知编码器（将视觉信息压缩到相机感知寄存器中）、轨迹解码器（生成候选轨迹）、评分解码器（评估轨迹）。最终轨迹通过最高预测分数选出。

- **标题**：*Driving on Registers*（基于寄存器驾驶）
- **团队**：Valeo.ai, Paris（Ellington Kirby, Alexandre Boulch, Yihong Xu等14位作者）
- **发表**：CVPR 2026（arXiv:2601.05083）
- **关键词**：端到端自动驾驶、ViT、寄存器token、Token压缩、轨迹评分
- **一句话总结**：一个极致简洁的纯Transformer端到端自动驾驶架构，通过ViT寄存器token将数千视觉token压缩为数十场景token，仅~40M参数达到SOTA。
- **论文链接**：[arXiv:2601.05083](https://arxiv.org/abs/2601.05083)
- **项目主页**：[valeoai.github.io/driving-on-registers](https://valeoai.github.io/driving-on-registers/)
- **代码仓库**：[github.com/valeoai/DrivoR](https://github.com/valeoai/DrivoR)

---

## 🤔 要解决什么问题？

端到端自动驾驶方法的感知backbone通常主导了参数和FLOP计数。现有方法主要面临以下问题：

**1. 视觉Token爆炸**。ViT等大规模预训练模型虽然性能优越，但每帧输出数千个token。如果以ViT-Large为例，单张图像产生1024个patch token，6个相机就是6144个token——要在这些token上做数百条轨迹的交叉注意力计算，计算瓶颈极其严重。

**2. 简单的池化操作信息丢失严重**。常见的空间池化（spatial pooling）将所有空间位置视为等重要的，无法区分不同相机视角的信息差异，且对分辨率变化敏感。

**3. 端到端方法仍然复杂**。虽然UniAD、VAD等方法号称端到端，但仍然依赖检测、跟踪、建图等中间模块，增加了标注成本和部署难度。

DrivoR的核心问题非常简单直接：**到底需要多少个token才能表示一个驾驶场景？**

---

## 🏗️ 核心架构

<img src="/images/paper/2601.05083/figure2.png" alt="DrivoR编码器与解码器架构" style="width:100%; max-width:900px; display:block; margin:0 auto;">
**图2: 编码器和解码器架构**。(a)编码器：在ViT中引入相机感知寄存器，将每相机的视觉信息压缩到R个寄存器token中。(b)解码器：标准Transformer解码器，以场景token为KV进行交叉注意力，生成轨迹或分数。

DrivoR的设计哲学是"极简"——没有BEV表示、没有大规模轨迹词典、没有复杂的中间模块。整个架构由三个Transformer模块组成：

### 1. 感知编码器（Perception Encoder）

核心创新在于**相机感知寄存器token（camera-aware register tokens）**：

- 在每个相机的ViT输入中，除了标准CLS token和patch token外，额外添加**R个可学习的寄存器token**
- 这些寄存器经过ViT的前向传播后，从最后一层取出作为该相机的紧凑表示
- 所有相机的寄存器token拼接在一起，形成场景token（scene tokens），数量为 N×R（N为相机数，R为每相机寄存器数）

关键设计点：
- **相机感知**：每个寄存器与特定相机绑定，模型能区分不同视角的信息来源
- **LoRA微调**：使用LoRA高效微调ViT backbone，学习从视觉到寄存器的压缩
- **极低数量**：仅用N×R个token（如6相机×4寄存器=24个token）替代数千个patch token

### 2. 轨迹解码器（Trajectory Decoder）

标准Transformer解码器架构：
- 输入：可学习的轨迹查询（trajectory queries）+ 编码后的自车状态
- 交叉注意力：以场景token为Key/Value进行注意力
- 输出：通过MLP解码为候选轨迹

轨迹查询采用**赢家通吃（Winner-Takes-All）**训练策略——多个候选轨迹中只有与GT最匹配的一条计算回归损失，鼓励查询多样化。

### 3. 评分解码器（Scoring Decoder）

与轨迹解码器架构相同，但输入是被**分离梯度**的轨迹token（即轨迹token不参与评分训练的反向传播）：
- 每个轨迹token作为查询，场景token作为KV，输出该轨迹的**可解释子分数**
- 子分数包括：安全性、舒适性、效率等多个维度
- 训练时模仿oracle评分器（数据集提供的专家评分）

### 4. 可调节驾驶行为

DrivoR最实用的特性——**推理时无需重新训练即可调节驾驶风格**：

```
Score(τ) = λ_safety × s_safety + λ_comfort × s_comfort + λ_efficiency × s_efficiency
```

通过调整各子分数的权重λ，可以实时改变驾驶行为：
- λ_safety ↑ → 保守安全型驾驶
- λ_efficiency ↑ → 激进高效型驾驶
- 无需重新训练，一次训练适配多种风格

---

## 🔬 实验分析

### 基准测试表现

| 基准 | 指标 | DrivoR (ViT-S) | 最佳基线 | 人类表现 |
|------|------|:---:|:---:|:---:|
| NAVSIM-v1 | PDMS ↑ | **93.7** | DriveSuprim 93.5 | 94.8 |
| NAVSIM-v2 | EPDMS ↑ | **48.3** | ZTRS (ViT-L) 48.1 | - |
| HUGSIM | RC / HD-Score | **49.8 / 35.7** | UniAD 45.9 / 32.7 | - |

### 效率对比

| 模型 | 参数量 | 前向时间 (A100) | 显存峰值 |
|------|:---:|:---:|:---:|
| DrivoR (ViT-S) | **~40M** | **110ms** | **0.5GB** |
| GTRS-Dense (ViT-L) | ~300M | 400ms | 1.6GB |
| DriveSuprim (ViT-L) | ~300M | 350ms | 1.4GB |

DrivoR仅用ViT-Small（~22M backbone + ~15M decoder）即可达到超越大部分ViT-L方法的性能，速度提升3-4倍。

### 消融实验关键发现

1. **寄存器数量**：每相机4-8个寄存器即可达到接近全token的性能，32个寄存器时性能饱和
2. **LoRA微调**：使用LoRA微调ViT比全量微调性能更好（避免过拟合驾驶数据集）
3. **分离梯度**：评分解码器中轨迹token分离梯度是关键——否则生成和评分会陷入不良平衡
4. **轨迹数量**：32-64个候选轨迹即可达到最优，继续增加收益递减

---

## 💡 个人思考

### 创新点

1. **重新定义ViT在自动驾驶中的使用方式**。之前的方法要么使用CNN backbone（如ResNet），要么直接使用ViT的全部token进行昂贵计算。DrivoR首次利用ViT的寄存器token进行结构化压缩，充分利用了Transformer的灵活性。

2. **极致极简架构**。没有BEV、没有检测头、没有轨迹词典、没有复杂的多任务训练——三个Transformer模块搞定一切。这种"少即是多"的设计思路值得学习。

3. **可解释子分数 + 行为调节**。将评分分解为安全、舒适、效率等可解释子分数，并支持推理时行为调节，实用价值很高。

### 与iPad的对比

| 对比维度 | iPad | DrivoR |
|---------|------|--------|
| 核心思想 | 迭代精炼稀疏提案 | 寄存器token压缩 |
| 感知表示 | BEV提案特征 | Transformer场景token |
| 轨迹生成 | 预测-锚定-精炼循环 | 查询解码器 |
| 评分方式 | 单一分数 | 多维度子分数 |
| 行为调节 | 不支持 | 支持 |
| 参数量 | ~50M | ~40M |

两个方法代表了端到端自动驾驶轻量化的两种不同路线：iPad走"规划中心"路线，DrivoR走"token压缩"路线。

### 局限性

1. **依赖数据集提供的oracle评分**。评分训练需要oracle分数作为监督信号，限制了在缺乏评分数��的数据集上的应用。
2. **闭环测试尚未完全覆盖**。HUGSIM虽然是闭环仿真，但场景数量和多样性仍有限。
3. **寄存器token的可解释性**。虽然整体架构简单，但寄存器token具体编码了哪些视觉信息还缺乏深入分析。

---

## 📖 延伸阅读

1. **iPad: Iterative Proposal-centric End-to-End Autonomous Driving**（Guo et al., CVPR 2025）- 以规划提案为中心的端到端自动驾驶
2. **Vision Transformers Need Registers**（Darcet et al., ICLR 2024）- 寄存器token的原始提出论文
3. **An Image is Worth 32 Tokens for Reconstruction and Generation**（Yu et al., 2024）- TiTok，将图像压缩为极少量token
4. **Hydra-MDP: End-to-End Multimodal Planning with Multi-Target Hydra-Distillation**（Li et al., 2024）- 轨迹评分规划的奠基工作
5. **GTRS: Generalized Trajectory Scoring for End-to-End Multimodal Planning**（2025）- 通用轨迹评分方法
