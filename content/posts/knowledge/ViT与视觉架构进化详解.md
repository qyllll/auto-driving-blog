---
title: "知识精讲｜ViT与视觉架构进化详解"
date: 2026-07-19
draft: false
categories: ["知识点拆解"]
tags: ["👁️ ViT", "🖼️ 视觉架构", "🧠 Transformer", "📐 特征金字塔"]
summary: "从 ViT 到 ConvNeXt，视觉架构在 Transformer 与卷积之间完成了一个轮回。本文系统梳理 ViT 的核心机制（patch embedding、position encoding、class token），详解 DeiT、Swin、ConvNeXt 等关键演进，并分析多尺度特征金字塔与 VLA 视觉编码器的选型逻辑与缩放定律。"
weight: 10
---

## 📌 概述

Vision Transformer（ViT）的诞生标志着计算机视觉正式进入 Transformer 时代。2020 年 Google 团队证明：当有足够数据时，一个纯粹的 Transformer 可以直接在图像 patch 序列上做分类，效果超过当时最先进的 CNN。此后三年，视觉架构经历了 ViT → DeiT → Swin → ConvNeXt 的快速演进，最终在"卷积 vs Transformer"的争论中走向融合。

在自动驾驶的 VLA 和世界模型论文中，视觉编码器的选择直接影响模型性能——从 DriveVLM 的 ViT-L/14 到 Hydra-MDP 的 ConvNeXt，视觉架构的演进与自动驾驶感知需求密不可分。

---

## 🎯 核心概念

### Patch Embedding：图像如何变成序列

ViT 将图像分割为固定大小的 patches（如 16×16），通过线性投影将每个 patch 展平为向量，形成 Transformer 的输入 token 序列。这与 NLP 中将句子切分为词的逻辑完全一致。

具体过程：输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 被切分为 $N = HW/P^2$ 个 patches，每个 patch 通过可学习的线性投影映射到 $D$ 维：

$$z_0 = [x_p^1 E; x_p^2 E; \dots ; x_p^N E] + E_{pos}$$

其中 $E \in \mathbb{R}^{(P^2 C) \times D}$ 是 patch embedding 矩阵，$E_{pos}$ 是位置编码。

### Position Encoding：让 Transformer 知道"谁在哪"

自注意力是置换等变的——它不知道 token 的顺序。位置编码为每个 token 注入空间位置信息。ViT 使用可学习的 1D 位置编码（每个 patch 学一个位置向量），后续工作证明 2D 相对位置编码（如 Swin 中的 relative position bias）对视觉任务更有效。

### Class Token vs Patch Features

ViT 在输入序列前附加一个特殊的 [CLS] token，其最终输出用于分类。CLS token 通过自注意力与所有 patch token 交互，相当于"全局图像表示"。然而对于密集预测任务（检测、分割），patch-level 特征（每个 patch 的输出向量）携带更丰富的空间信息。现代 VLA 模型通常取所有 patch token 的输出，而非 CLS token。

---

## 🔧 技术详解

### ViT：原始 Transformer 的"暴力"移植

ViT 提供了多种规格以适应不同计算资源：ViT-Tiny（5.7M 参数，12 层，384 hidden）、ViT-Small（22M 参数，12 层，384 hidden, 6 heads）、ViT-Base（86M 参数，12 层，768 hidden, 12 heads）、ViT-Large（307M 参数，24 层，1024 hidden, 16 heads）和 ViT-Huge（632M 参数，32 层，1280 hidden, 16 heads）。ViT-B/16 包含 12 层 Transformer encoder，hidden size 768，12 个注意力头。输入图像 224×224 被切分为 14×14=196 个 16×16 patches，加上一个 [CLS] token 共 197 个 token。每一层包含 Multi-Head Self-Attention（MHSA）和 MLP 两个子层，均使用 Layernorm 和残差连接。

ViT 的高层与底层有不同的关注模式：底层关注局部纹理和边缘（类似 CNN 的低层特征），高层关注语义区域和物体部件。这一特性使 ViT 在需要全局语义理解的任务（如分类、VQA）上优于 CNN，但在需要局部精细特征的任务（如检测、分割）中需要额外设计。

在 ImageNet-21k 或 JFT-300M 等大数据集上预训练后，ViT 在下游任务上超越 ResNet；但在小数据上（如 ImageNet-1k），ViT 不如同等规模的 CNN。这是因为 Transformer 缺乏 CNN 的归纳偏置（局部连接、平移等变性），必须靠大量数据"学习"这些先验。

这一缺陷催生了 DeiT（Data-efficient Image Transformers）——通过知识蒸馏（teacher 为 RegNet）和强数据增强（RandAugment、MixUp、CutMix），DeiT 在 ImageNet-1k 上从头训练即达到 SOTA，无需外部数据。DeiT 还引入了 teacher-student 策略：student 接收强增强后的图像，teacher 接收弱增强图像，通过 token-level 的蒸馏损失传递细粒度知识。

### Swin Transformer：层次化与移动窗口

Swin Transformer 引入了两个关键设计：

1. **层次化特征金字塔**：通过 patch merging 逐步降低分辨率（4× → 8× → 16× → 32×），形成类似 ResNet 的多尺度特征图，天然适配检测/分割的 FPN 需求。
2. **移动窗口注意力（Shifted Window Attention）**：在局部窗口内计算自注意力（窗口大小 7×7），并在相邻层间移动窗口划分，实现跨窗口信息交换。计算复杂度从 ViT 的 $O(N^2)$ 降为 $O(N)$。

Swin 是第一个将 Transformer 引入检测/分割主干的主流工作，在 COCO 和 ADE20K 上大幅超越 CNN。

Swin 的窗口注意力机制实现了计算复杂度的空间换时间：窗口大小固定（通常 7×7），每个 token 只与窗口内 49 个 token 计算注意力，而非全图的全部 token。对于一张 224×224 的输入图像（patch size 4，得到 56×56=3136 个 token），ViT 的计算复杂度为 O(3136²) ≈ 9.8M，而 Swin 的窗口注意力复杂度为 O(56² × 7²) ≈ 154K，降低约 63 倍。

### ConvNeXt：用 Transformer 的设计哲学改造 CNN

ConvNeXt 反其道而行：以 ResNet-50 为起点，逐步注入 Transformer 的设计元素——使用 GELU 激活、LayerNorm 替代 BatchNorm、增大卷积核（7×7）、倒置瓶颈结构（inverted bottleneck）、去掉下采样的 ReLU 等。最终得到的 ConvNeXt 在性能上与 Swin 持平，但保留了卷积的推理效率优势。

ConvNeXt 的改进细节：
- **Stage ratio**：从 ResNet 的 (3,4,6,3) 调整为 Swin 的 (3,3,9,3)，增加了第 3 阶段的深度，因为该阶段特征图分辨率适中（14×14），最适合语义特征提取。
- **Patchify stem**：用 4×4 卷积（stride 4）替代 ResNet 的 7×7 conv + maxpool，类似 ViT 的 patch embedding 方式。
- **Separate downsampling**：在每个 stage 之间添加独立的 2×2 卷积（stride 2）进行下采样，而非在 stage 第一个 block 中隐式下采样。
- **Large kernel**：将 bottleneck 中的 3×3 depthwise conv 替换为 7×7，增大感受野。

ConvNeXt 的核心启示："架构的性能更多取决于设计理念（stage ratio、kernel size、normalization），而非 Transformer 或卷积本身。"

| 架构 | 核心创新 | 计算复杂度 | 多尺度 | 小数据性能 | VLA 使用情况 |
|------|----------|-----------|--------|-----------|-------------|
| ViT | patch embedding + Transformer | $O(N^2)$ | 否（单尺度） | 差 | DriveVLM ViT-L/14 |
| DeiT | 知识蒸馏 + 数据增强 | $O(N^2)$ | 否 | 好 | - |
| Swin | 移动窗口 + 层次化 | $O(N)$ | 是（天然 FPN） | 好 | VideoLMM 等 |
| ConvNeXt | 现代 CNN 设计 | $O(N)$ | 是 | 好 | Hydra-MDP |

### 多尺度特征金字塔

检测与分割任务天然需要多尺度特征。FPN（Feature Pyramid Network）通过自顶向下的路径和横向连接，将高层的语义信息与低层的空间信息融合，构建多尺度特征金字塔。后续工作进一步改进：

- **PANet**：在 FPN 上增加自底向上的增强路径，缩短低层特征到高层的路径。
- **BiFPN**（EfficientDet）：引入加权特征融合，不同尺度的贡献可学习。
- **NAS-FPN**：用神经架构搜索自动设计融合拓扑。

在自动驾驶中，多尺度特征对检测小目标（远处行人、锥桶）至关重要。BEVFormer 使用多尺度特征（来自 ResNet 的 C3-C5 层）作为 Transformer encoder 的输入。

---

## 📊 方法对比

### VLA 论文的视觉编码器选型

| 论文 | 视觉编码器 | 输入分辨率 | 参数量 | 选择原因 |
|------|-----------|-----------|--------|---------|
| DriveVLM | ViT-L/14 (CLIP) | 336×336 | 304M | 强语义特征 + 语言对齐 |
| Hydra-MDP | ConvNeXt-Base | - | 88M | 高效推理 + 多尺度 |
| UniAD | ResNet-101/ResNet-50 | 800×320 | 44M | 成熟稳定、推理快 |
| VAD | ResNet-50 | 800×320 | 25.6M | 轻量级基线 |
| EMMA | SigLIP ViT-L | - | - | 原生语言对齐 |
| World4Drive | DINOv2 + Depth Anything | - | - | 强对应 + 深度 |

### Scaling Laws：更大的视觉编码器更好吗？

DINOv2 提供了清晰的缩放证据：从 ViT-S（21M）到 ViT-g（1.1B），模型规模每增大一次，语义分割 AP、k-NN 分类准确率、特征对应质量都持续提升。ViT-g 在几乎所有下游任务上显著超过 ViT-L。

但更大并不总是更好——在自动驾驶中需要考虑：

1. **推理延迟**：ViT-g 在单张 A100 上的推理速度约 10 FPS，远不能满足实时要求。
2. **训练成本**：ViT-g 的预训练需要大规模蒸馏和 4 卡 × 27 天。
3. **过拟合风险**：在小规模驾驶数据集上直接微调大模型容易过拟合。

实际部署的平衡点是 ViT-L（如 DriveVLM）或 ConvNeXt-Base（如 Hydra-MDP）。此外，采用 token 压缩策略（如 Q-Former 将 256 tokens 压缩到 32）可以显著降低大模型在推理时的 LLM 解码成本，使 ViT-L 级别编码器在车端部署成为可能。

---

## 🔗 与自动驾驶的关联

### Vision Encoder 是 VLA 的感知基础

视觉编码器作为 VLA 模型的第一层和特征入口，其输出质量直接决定了 LLM 对场景理解的上限和规划决策的可靠性——如果视觉编码器漏掉了远处的小目标（如行人），后续的 LLM 和动作头无论如何也无法弥补这一信息缺失。VLA 模型的三组件（视觉编码器 → LLM → 动作头）中，视觉编码器承担了从原始像素到语义特征的转换。它直接影响模型对场景的理解质量——能否识别远距离行人、能否区分不同交通标志、能否在夜间/雨雾等退化条件下稳定工作。

视觉编码器的输出需要与 LLM 的 embedding 空间对齐。常用的对齐方式包括：

1. **MLP Projector**（如 LLaVA）：将 patch token 序列通过一个 2-3 层 MLP 映射到 LLM 的 embedding 维度。直接、简单，但对齐质量受限于 MLP 容量。
2. **Q-Former**（如 BLIP-2）：使用一组可学习的 query，通过跨注意力从视觉特征中提取与任务相关的信息。可有效压缩视觉 token 数量（从 256 压缩到 32），降低 LLM 的计算开销。
3. **Resampler**（如 Flamingo）：在视觉特征上使用 Perceiver Resampler，将可变长度的视觉 token 序列转换为固定长度的上下文 token。

在 DriveVLM 中，视觉编码器的输出通过一个轻量 adapter 直接拼接为 LLM 的输入 token 序列，每个 patch token 对应一个"视觉词"，与文本 token 交替排列送入 LLM 进行自回归推理。

### CLIP 预训练的视觉 backbone 成为标配

DriveVLM 等 VLA 模型采用 CLIP 预训练的 ViT-L/14。CLIP 的图文对比学习使视觉编码器天然具备语义对齐能力——它的特征空间与语言空间一致，便于后续与 LLM 对接。这与"视觉特征→LLM embedding"的 alignment 训练目标高度契合。

### Multi-scale 特征对 BEV 感知的必要性

BEVFormer 等感知模型依赖多尺度特征来同时检测远处的小目标和近处的大目标。ResNet 和 ConvNeXt 通过其层次化结构天然提供多尺度特征图，而 ViT 需要额外设计（如 Swin 的 patch merging 或使用不同层输出）才能加入 FPN。

### 不同编码器在 AD 中的实际推理性能

车载部署对视觉编码器有严格的延迟约束（通常要求 < 50ms 端到端感知）。ResNet-50 在 NVIDIA Orin 上推理约 5ms，ConvNeXt-Base 约 12ms，ViT-B 约 20ms，ViT-L 约 50ms（均为 FP16 精度）。因此，在延迟敏感的量产方案中，ConvNeXt 和 ResNet 仍是主力；而 ViT-L 更多用于云端方案或作为教师模型。

### 视觉架构的未来方向

视觉架构正在向三个方向演进：**多模态联合预训练**（如 CLIP、SigLIP 将视觉编码器与语言模型对齐）、**原生视频理解**（Video ViT、TimeSformer 直接在时空 token 上建模）、以及**可变形架构**（Deformable DETR 中的可变形注意力等自适应 token 机制）。这些方向都在自动驾驶的 VLA 和世界模型中得到快速应用。此外，**状态空间模型**（如 Mamba）作为 Transformer 的替代方案也开始在视觉任务中展示潜力，其线性复杂度为处理高分辨率图像提供了新的可能。

---

## 📚 延伸阅读

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
- Touvron et al., "Training data-efficient image transformers & distillation through attention", ICML 2021.
- Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020.（DETR 是 ViT 架构在检测中的首次成功应用）
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021.
- Liu et al., "A ConvNet for the 2020s", CVPR 2022.
- Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024.
- Tian et al., "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models", 2024.
- Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021.
- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.（MAE 是另一种重要的 ViT 自监督预训练方法，与 CLIP/DINOv2 并列）
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
