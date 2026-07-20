---
title: "知识精讲｜LiDAR点云处理基础"
date: 2026-07-19
draft: false
summary: "系统讲解LiDAR工作原理（ToF、机械/固态）、点云表示方法（xyz、intensity、ring）、三类主流点云检测方法（voxel-based、point-based、range-view），以及在端到端检测中的典型方案（CenterPoint、TransFusion-L）。"
tags: ["LiDAR", "点云", "3D检测", "感知"]
categories: ["知识点拆解"]
---

## 📌 概述

LiDAR（Light Detection And Ranging）是自动驾驶中提供精确三维空间信息的核心传感器。与相机不同，LiDAR 直接测量物体在空间中的距离，不受光照条件影响，在白昼、夜间甚至低照度环境下都能稳定工作。

本讲从 LiDAR 的物理工作原理出发，介绍点云的基本表示形式与预处理方法，重点讲解三大主流检测范式——**Voxel-based**、**Point-based** 与 **RangeView-based**，并对比其在精度与速度上的权衡。最后介绍 CenterPoint、TransFusion-L 等端到端检测方法的架构思路，以及当前主流 LiDAR 传感器的技术参数。

---

## 🎯 核心概念

### LiDAR 工作原理

LiDAR 通过发射激光束并测量反射光的飞行时间（Time of Flight, ToF）来计算目标距离：

$$
d = \frac{c \cdot \Delta t}{2}
$$

其中 $c$ 为光速，$\Delta t$ 为发射到接收的时间差。

**扫描方式分类**：

| 类型 | 原理 | 代表产品 | 特点 |
|------|------|----------|------|
| 机械旋转式 | 激光器+接收器在电机驱动下 360° 旋转 | Velodyne HDL-64E, Pandar64 | 360° FOV，精度高，但存在机械磨损 |
| 固态（Flash） | 单次大面积发射，面阵接收 | LeddarTech | 无运动部件，可靠性高，FOV 受限 |
| 固态（MEMS） | 微镜反射改变光束方向 | InnovizOne, RoboSense M1 | 兼顾可靠性与 FOV |
| OP（Optical Phased Array） | 光学相控阵控制光束偏转 | 未大规模量产 | 全固态、扫描速度快 |

### 点云表示

一个原始 LiDAR 点通常包含以下字段：

- **$(x, y, z)$**：三维空间坐标（通常以 LiDAR 为原点）
- **intensity**：反射强度，反映目标材质反射率
- **ring**：激光线束编号（对于多线 LiDAR），用于识别来自哪个发射器
- **timestamp**：时间戳，用于运动补偿（Motion Compensation）
- **azimuth**：水平方位角

### 坐标系约定

LiDAR 坐标系通常遵循：$+x$ 指向前方，$+y$ 指向左侧，$+z$ 指向上方（右手系）。点云在输入网络前通常被归一化到以自车为中心的坐标系中。

---

## 🔧 技术详解

### 预处理流程

1. **运动补偿**：在扫描周期内自车可能移动数十厘米，需利用 IMU/里程计将各点变换到统一时间戳
2. **地面分割**：使用 RANSAC 或 Patchwork++ 提取地面点，降低后续检测的搜索空间
3. **降采样**：使用 Voxel Grid Filter 或 Farthest Point Sampling（FPS）平衡计算量与密度
4. **区域裁剪**：去除自车顶部、后方等无用区域点云

### 三种主流检测范式

#### 1. Voxel-based（体素化方法）

将点云量化为固定大小的 3D 体素网格，使用 3D 稀疏卷积进行特征提取。

**VoxelNet（2017）**：开创性工作，将体素内的点通过 VFE（Voxel Feature Encoding）层聚合为体素级特征，再使用 3D 卷积处理。

**SECOND（2018）**：引入稀疏卷积（Sparse Convolution）替代 VoxelNet 中的密集 3D 卷积，推理速度提升数倍，成为后续工作的基座。

**PointPillars（2019）**：将体素压缩为柱体（pillar），只在 $(x, y)$ 平面划分网格，$z$ 方向不做划分。每个 pillar 内的点用简化版的 PointNet 编码，然后使用 2D 卷积处理伪图像。PointPillars 在速度和精度间取得了良好平衡，成为工业部署的首选方案之一。

#### 2. Point-based（逐点处理方法）

直接在原始点云上操作，无需量化操作。

**PointNet / PointNet++（2017）**：使用共享 MLP 逐点编码特征，通过对称函数（max pooling）实现置换不变性。PointNet++ 引入分层采样和局部聚合，提升了对局部结构的建模能力。

在 3D 检测中，PointNet++ 常作为 backbone 用于特征提取（如 PointRCNN）。

**优缺点**：
- 优点：不损失原始点云几何信息，理论上可达到最高精度
- 缺点：最近邻搜索开销大，推理速度慢，难以满足实时性要求

#### 3. RangeView-based（距离视图方法）

将点云投影到球面坐标系，得到类似图像的 2D 表示。

$$
\theta = \arcsin(z / r), \quad \phi = \arctan(y / x)
$$

每个像素保存 $(r, x, y, z, intensity, ring)$ 等特征，然后使用 2D 卷积网络处理。

**RangeDet（2021）**：提出 RangeView 下的通用检测框架，使用圆形卷积（circular padding）处理水平方向的周期性。

**优缺点**：
- 优点：计算效率高，可直接使用成熟的 2D CNN 架构
- 缺点：投影过程丢失 3D 几何信息，物体在 RangeView 中尺度变化剧烈，远距离目标像素极少

### 端到端 3D 检测：CenterPoint 与 TransFusion-L

**CenterPoint（2021）**：
- 将检测范式从 anchor-based 转为 center-based，类似 2D 检测中的 CenterNet
- 使用 voxel-based backbone 提取特征，在 BEV 空间预测目标的中心点热图（heatmap）
- 用中心点回归尺寸、朝向、速度等属性
- 两阶段版本（CenterPoint-PointRefine）使用 PointNet++ 在目标点云内进行 box refinement

**TransFusion-L（2022）**：
- LiDAR-only 版本的 TransFusion
- 使用 Transformer decoder 结构，通过 queries 和 BEV feature 的 cross-attention 直接输出目标框
- 引入自适应稀疏注意力（Adaptive Sparse Attention），在保持精度的同时降低计算量
- 在 nuScenes 数据集上达到当时 SOTA

---

## 📊 方法对比

| 方法 | Backbone | 精度（nuScens NDS） | FPS | 内存 |
|------|----------|---------------------|-----|------|
| VoxelNet | 3D 密集卷积 | ~52% | <5 | 高 |
| SECOND | 3D 稀疏卷积 | ~55% | ~20 | 中 |
| PointPillars | 2D 卷积 | ~57% | ~60 | 低 |
| CenterPoint | 稀疏卷积 + 2D | ~65% | ~30 | 中 |
| TransFusion-L | Transformer | ~68% | ~15 | 高 |
| PointRCNN | PointNet++ | ~55% | <5 | 高 |
| RangeDet | 2D CNN | ~58% | ~40 | 低 |

---

## 🔗 与自动驾驶的关联

### LiDAR 在感知中的角色

| 任务 | LiDAR 优势 | 典型方案 |
|------|-----------|----------|
| 3D 目标检测 | 精确距离信息，无尺度歧义 | CenterPoint, TransFusion-L |
| 在线地图重建 | 高精度点云匹配 | 基于 NDT 的 LiDAR SLAM |
| 障碍物穿越 | 低矮障碍物检测 | 点云高度特征分析 |
| 定位 | 点云配准，不受光照变化影响 | LiDAR localization |

### 常见 LiDAR 传感器参数

| 型号 | 线束 | 探测距离 | FOV(H/V) | 精度 | 类型 |
|------|------|---------|----------|------|------|
| Velodyne HDL-64E | 64 | 120m | 360°/26.9° | ±2cm | 机械 |
| Velodyne VLP-16 | 16 | 100m | 360°/30° | ±3cm | 机械 |
| Hesai Pandar64 | 64 | 200m | 360°/40° | ±1cm | 机械 |
| Hesai PandarXT-32 | 32 | 120m | 360°/40° | ±3cm | 机械 |
| RoboSense RS-LiDAR-128 | 128 | 200m | 360°/40° | ±2cm | 机械 |
| RoboSense M1 | — | 150m | 120°/25° | ±3cm | MEMS 固态 |
| Ouster OS1-64 | 64 | 120m | 360°/45° | ±3cm | 机械 |

### 趋势

- **从机械式到固态**：MEMS 和 Flash LiDAR 成本持续下降，预计 2026-2027 年在 L2+ 车型中普及
- **从单传感器到多模态融合**：LiDAR + Camera 的深度融合（如 BEVFusion, TransFusion）正成为主流
- **分辨率持续提升**：512 线以上 LiDAR 已进入原型阶段，将大幅提升远距离感知能力

---

## 📚 延伸阅读

1. Zhou, Y. & Tuzel, O. "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection." *CVPR*, 2018.
2. Yan, Y. et al. "SECOND: Sparsely Embedded Convolutional Detection." *Sensors*, 2018.
3. Lang, A. H. et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." *CVPR*, 2019.
4. Yin, T. et al. "Center-based 3D Object Detection and Tracking." *CVPR*, 2021.
5. Bai, X. et al. "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers." *CVPR*, 2022.
6. Fan, L. et al. "RangeDet: In Defense of Range View for LiDAR-based 3D Object Detection." *ICCV*, 2021.
