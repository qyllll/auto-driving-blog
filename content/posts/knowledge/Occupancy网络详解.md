---
title: "知识精讲｜Occupancy网络详解"
date: 2026-07-19
draft: false
categories: ["知识精讲"]
tags: ["🧊 Occupancy", "🗺️ 占据网络", "👁️ 感知", "🌐 世界模型", "🚗 规划"]
summary: "Occupancy 网络通过稠密的 3D 体素网格表示场景，突破了 3D 目标检测对\u201c物体\u201d范畴的限制，可表征任意形状的障碍物与自由空间。本文详解其与检测的差异、OccFormer/FlashOcc/OpenOccupancy 等代表架构，以及在 NIFF 等世界模型中 occupancy 如何作为核心场景表示驱动未来预测与规划决策。"
weight: 13
---

## 📌 概述

Occupancy 网络（占据网络）是继 3D 目标检测之后，自动驾驶感知领域的又一次范式跃迁。与检测"画框"的方式不同，occupancy 将场景表示为稠密的 **3D 体素网格（Voxel Grid）**，每个格子被标注为"占据"或"空闲"（或进一步分类为语义类别）。

Occupancy 的优势在于：它能描述任意形状的物体——翻倒的车辆、散落的货物、施工区域的锥桶——这些在"边界框"范式下难以处理。同时，occupancy 天然适合作为世界模型和规划器的场景表示，因为它直接给出"哪里能走、哪里不能走"的信息。

---

## 🎯 核心概念

### Occupancy vs Detection：为什么需要 Occupancy

| 维度 | 3D 目标检测 | Occupancy 网络 |
|------|------------|---------------|
| 表示方式 | 稀疏边界框（7 自由度） | 稠密体素网格（N×M×K） |
| 表征能力 | 仅限"物体"类别 | 任意形状障碍物 + 自由空间 |
| 泛化性 | 无法检测未见过的物体类型 | 可泛化至任意占据物 |
| 标注成本 | 需要 3D 框标注（昂贵） | 可通过 LiDAR 扫描自动生成 |
| 与规划的接口 | 需要额外转换（几何推理） | 可直接定义可行驶区域 |
| 输出分辨率 | 稀疏（几十个目标） | 稠密（数十万体素） |

Occupancy 的核心思想来源于 **Occupancy Grid Mapping**（机器人学经典方法），但用神经网络替代了传统的贝叶斯更新。

### 3D Voxel 表示

Occupancy 网络的输出是一个 3D 体素网格 $O \in \mathbb{R}^{X \times Y \times Z \times C}$，其中：

- X、Y 为 BEV 空间分辨率，通常覆盖自车周围 ±50m，分辨率 0.4m ~ 0.5m。
- Z 为高度维度，通常覆盖 -5m ~ 5m，约 16~32 层。
- C 为通道数：二值占用（C=1）或语义类别（C=num_classes，如地面、车辆、行人、建筑等）。

一个典型的输出为 200×200×16 × C 的稠密网格，对应约 64 万个体素。

---

## 🔧 技术详解

### OccFormer：Transformer 特征与 3D 卷积解码

OccFormer 是一个面向 3D 占据预测的 Transformer-based 架构。它从 BEVFormer 的 BEV 特征出发，通过两个关键模块构建 3D occupancy：

1. **Height Compression & Expansion**：先将 BEV 特征沿高度维度解码压缩，再通过分组反卷积恢复高度分辨率，形成 3D 特征体。
2. **3D Conv Decoder**：使用 3D 卷积逐步上采样到目标体素分辨率，每个体素输出占据概率和语义类别。

OccFormer 还引入了时序融合模块：将历史帧的 3D occupancy 特征通过 3D 流估计（3D flow estimation）对齐到当前帧坐标系，实现跨帧特征聚合。这一设计使 OccFormer 在动态场景中的预测更加稳定，能够有效处理遮挡和移动物体的"拖影"问题。

OccFormer 在 nuScenes Occ3D 基准上取得了 SOTA（mIoU 37.2），证明了基于 BEV 的 2D 特征可以通过精心设计的 3D 解码器有效转换为 3D 占据表示。

### FlashOcc：极简高效的 Occupancy

FlashOcc 追求推理效率最大化。其核心洞察是：**在 BEV 特征上直接预测高度方向的分布函数，替代显示构建 3D 特征体**。具体而言：

1. 对每个 BEV 网格位置，预测一个高度方向的概率分布 P(z)，表示"哪个高度被占据"。
2. 从 BEV 特征通过一个轻量 MLP 直接输出占据标签。

这种方法避免了 3D 卷积的高计算开销。FlashOcc 在单张 3090 上达到 77 FPS，精度仅略低于 OccFormer。

### OpenOccupancy：开放词汇语义占据

OpenOccupancy 将语义占据扩展到开放词汇场景。其核心思路是将 CLIP 的语义空间引入 occupancy——每个体素的特征与 CLIP 文本编码器输出进行匹配，实现任意类别的语义占据预测。

关键设计：
1. **LiDAR 引导的体素特征学习**：使用 LiDAR 点云的三维位置作为 query，从图像特征中采样，显式对齐图像-点云特征。
2. **CLIP 特征蒸馏**：在体素特征上施加 CLIP 对齐损失，使其语义空间与文本空间一致。

### BEVDet-Occ：从 BEVDet 扩展

BEVDet 是纯视觉 BEV 感知的经典框架。BEVDet-Occ 在其基础上增加了一个 occupancy head：对 BEVDet 的 BEV 特征做 3D 卷积解码（类似 OccFormer），但使用更轻量的设计。BEVDet-Occ 提供了一个统一的检测+occupancy 框架，实现了"检测框 + 占据图"的多任务输出。

### SurroundOcc：多相机环绕 3D 占据预测

SurroundOcc 进一步将 occupancy 预测扩展到 3D 空间中的任意视角。它不局限于 BEV 空间的 2D+高度分解，而是直接在 3D 体素空间中从多相机图像特征采样。关键设计包括：3D 体素 query 通过相机投影矩阵采样多视图图像特征，使用 3D 稀疏卷积逐步上采样到高分辨率。SurroundOcc 在语义 Occupancy 任务中显著超越了 OccFormer，尤其在高度方向的预测精度上有明显提升。

| 方法 | 骨干 | 核心设计 | 推理速度 | mIoU (Occ3D) | 是否检测 |
|------|------|---------|---------|-------------|---------|
| OccFormer | BEVFormer | 3D Conv Decoder + Transformer | 中等 | 37.2 | 否 |
| FlashOcc | ResNet-50 + LSS | 高度分布直接预测 | 77 FPS | 31.5 | 否 |
| OpenOccupancy | BEVFormer + CLIP | 开放语义对齐 | 慢 | 35.8 | 否 |
| BEVDet-Occ | BEVDet | Unified 检测+占据 | 快 | 33.7 | 是 |

### 时序融合

Occupancy 预测的时序融合方法与 BEV 感知类似：将历史帧的 occupancy 网格通过自车位姿变换到当前帧坐标系，然后与当前帧预测融合。但 occupancy 的 3D 特性带来了额外的挑战——变换投影需要 3D 空间中的旋转和平移，计算量远大于 2D BEV 特征的对齐。

在实际实现中，时序融合通常采用两种策略：

1. **BEV 特征对齐 + 3D 解码**：在 BEV 空间（2D）中对齐多帧特征，再进行 3D 解码恢复高度信息。计算效率高，但丢失了高度方向的时序一致性。
2. **3D 体素对齐**：直接在 3D 体素空间通过 3D 流估计（3D scene flow）对齐每一帧。精度最高，但计算开销比策略 1 大约 3-5 倍。

SurroundOcc 采用了混合策略：在 BEV 空间做粗略对齐，再在 3D 空间做精细对齐，平衡了效率与精度。OccNeRF 则将 NeRF 的体渲染技术引入 occupancy 预测，通过可微渲染从 2D 图像直接监督 3D 占据网格，实现了无需 LiDAR 标注的纯视觉 occupancy 训练。这是一种自监督的 occupancy 学习范式，有望大幅降低 occupancy 标注成本。但自监督 occupancy 在动态物体区域和远距离区域的精度仍显著低于有监督方法，离量产仍有距离。

---

## 📊 方法对比

### Occupancy 作为世界模型的核心表示

在世界模型论文中，occupancy 扮演着双重角色：

1. **场景编码器**：将当前观测编码为 3D occupancy 特征。
2. **预测目标**：模型预测未来时刻的 occupancy 变化。

NIFF（Neural Implicit Flow Fields）是这一方向的代表：它使用 occupancy 特征作为场景表示，通过一个 flow 预测网络推断未来时刻体素占据状态的变化。NIFF 不是直接预测未来 occupancy 的概率，而是预测"当前占据体素在未来时刻的位置"，使用 scene flow 作为中间表示。

### NiFF 的工作流程

1. **编码**：对 LiDAR 点云或图像估计的 occupancy 进行体素化。
2. **Flow 预测**：使用一个 3D 稀疏卷积网络，对每个占据体素预测未来 N 帧的 3D 位移向量。
3. **未来 occupancy 推理**：根据预测的 scene flow 将当前占据体素"移动"到未来位置。
4. **规划应用**：在未来 occupancy 中定义可行驶区域，输入到规划器。

这种范式将感知-预测-规划统一在 occupancy 空间，避免了"检测 → 跟踪 → 轨迹预测"的传统流水线。相比传统流水线，occupancy-based 世界模型的优势在于：

1. **端到端可微**：从传感器输入到规划输出全部可微，梯度可以直接回传优化感知部分。
2. **无需显式目标关联**：不依赖检测-跟踪的数据关联（data association）这一困难子问题。
3. **自由空间约束**：occupancy 直接给出"哪里不能走"的约束，规划器可以天然地在这个约束空间中搜索最优轨迹。

---

## 🔗 与自动驾驶的关联

### Occupancy 与规划的天然亲和

规划器需要的不是"前方 50 米有一辆车，尺寸 4.5×1.8×1.5m"，而是"哪些区域可以安全行驶"。Occupancy 输出直接回答了这个问题——可行驶区域 = 没有被占据的体素 + 被语义类别标记为"地面/道路"的区域。

由于 occupancy 提供了稠密的自由空间信息，在 occupancy 空间进行轨迹规划时，只需要做碰撞检测（voxel-level check，即检查规划轨迹上的体素是否被占据），不需要复杂的几何推理（bounding box overlap 计算）。这使得规划更简单、更安全。

### 从检测到 occupancy：感知范式的演进

自动驾驶感知的演进可以概括为：

> **2D 检测 → 3D 检测 → Occupancy → 端到端隐式表示**

每个阶段都在减少对"预定义类别"的依赖，增加场景的细粒度描述能力。Occupancy 是目前最接近"完全场景描述"的感知范式，但仍面临计算开销和远距离精度不足的挑战。

有趣的是，TPVFormer 提出了三视图（Tri-Perspective View）表示——将 3D 体素分解为三个正交平面（BEV 平面 + 两个垂直平面），融合三视图特征即可恢复完整的 3D 体素信息。TPV 将 occupancy 的计算复杂度从 O(N³) 降为 O(N²)，且精度与 full 3D 方法相当。

Occupancy 在感知中的定位可以类比为：检测是"名词"（这是什么物体），追踪是"动词"（物体如何运动），occupancy 是"形容词+空间"（哪里被占据、以什么形状占据）。两者结合才能完整描述场景。

### Voxel 表征的算力挑战与优化

200×200×16 的 3D 体素网格包含 64 万个体素。如果每个体素使用 64 维特征，特征图大小为 64 万 × 64 ≈ 40M 参数，在 3D 卷积中的计算量约为同等分辨率 2D 卷积的 16 倍（多出高度维度）。这对车载芯片的算力提出了极高要求。

工程优化方向包括：

1. **稀疏化**：只保留 BEV 空间中被占据区域的体素特征，使用稀疏卷积或稀疏 Transformer 减少计算量。大多数场景中，占据体素占比不到 10%，稀疏化可节省 5-10 倍计算量。
2. **高度分解**：如 FlashOcc 所示，将 3D 问题分解为 2D BEV 特征 + 高度分布预测，避免构建完整的 3D 特征体。
3. **多分辨率策略**：在保证安全的前提下，远距离区域使用更粗的分辨率（如远处 1.0m/voxel，近处 0.4m/voxel），自适应网格是一种高效的折中方案。

### 检测 + Occupancy 的协同

实践中，检测和 occupancy 并非零和关系。检测提供了目标的实例级信息和跟踪一致性，occupancy 提供了稠密空间理解。BEVDet-Occ 和 UniAD 都采用了"检测 + occupancy"的多任务架构，两者互补。

### Benchmarks：Occ3D nuScenes 与 SemanticKITTI

| 基准 | 场景 | 分辨率 | 语义类别 | 评估指标 | 数据规模 |
|------|------|--------|---------|---------|---------|
| Occ3D nuScenes | 城市道路，6 相机 | 200×200×16 | 17 类 | mIoU + RayIoU | 700 训练，150 验证 |
| SemanticKITTI | 城市+高速，单 LiDAR | 256×256×32 | 20 类 | mIoU | 22 序列（~43k 帧） |

Occ3D 的 RayIoU 指标是一个创新：它沿每条激光射线计算预测和真值占据状态的差异，更关注占据边界位置的精度，避免了传统 IoU 对大多数空闲体素的"无意义正确"的偏向。

### Occupancy 的安全关键性

Occupancy 网络直接决定了自动驾驶系统对"可行驶区域"的理解，其误差可能直接导致碰撞或急刹车等不安全行为。一个错误的占据预测（将障碍物预测为空闲）可能导致碰撞。因此，occupancy 的评估不仅关注平均精度，还关注**安全关键指标**：

1. **False Negative Rate（FNR）**：漏检的占据体素比例——越高表示越可能撞上未识别的障碍物。
2. **边界精度**：占据-空闲边界的位置误差——决定了规划轨迹与障碍物的安全距离。
3. **时序一致性**：逐帧占据预测的稳定性——闪烁的预测可能导致规划震荡。
4. **召回率 @ 安全距离阈值**：在自车规划路径的一定范围内（如前 20 米），占据体素的召回率——这是最直接影响碰撞风险的单向指标。

---

## 📚 延伸阅读

- Tong et al., "OccFormer: Semantic Occupancy Network via 3D Transformer with Temporal Fusion", CVPR 2024.
- Li et al., "FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-Wise Height Compression", CVPR 2024.
- Wang et al., "OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception", ICCV 2023.
- Huang et al., "BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View", 2022.
- Hu et al., "NIFF: Neural Implicit Flow Fields for Dynamic 3D Scene Forecasting", CVPR 2024.
- Tian et al., "Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving", NeurIPS 2023.
- Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences", ICCV 2019.
- Wei et al., "SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving", ICCV 2023.
- Cao et al., "OccDepth: A Depth-Aware Method for 3D Semantic Occupancy Network", 2024.
- Huang et al., "OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural Radiance Fields", 2023.
- Li et al., "TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction", CVPR 2023.（TPV 将 3D 体素分解为三个正交平面，进一步降低了 occupancy 的计算复杂度）
- Wang et al., "Scene as Occupancy", ICCV 2023.
- Song et al., "Occ-Attention: 3D Occupancy Prediction with Attention-Guided Depth Estimation", 2024.
