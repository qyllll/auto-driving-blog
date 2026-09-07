---
title: "论文精读｜Qwen-Drive-1.0：阿里通义首个驾驶视觉-语言基础模型"
date: 2026-09-06
draft: false
categories: ["论文精读"]
tags: ["🧠 VLA", "🏭 阿里", "🚗 自动驾驶", "⚡ Flow Matching", "🔄 RL后训练", "📐 BEV感知"]
summary: "Qwen-Drive-1.0 基于 Qwen3.5-4B VLM 外挂两个模块：BEV感知头（3D检测+占用预测+地图分割）和 Planning Expert（~1.1B 参数 Flow Matching 轨迹生成器），四阶段渐进训练。关键设计：**不设推理时打分器**，直接用 Flow Matching 单次采样输出轨迹，RL 后训练用 GRPO 式分组优势优化。NAVSIM v1.1 榜上 90.7 PDMS（RL），WOD-E2E 7.91 RFS，AlpaSim 闭环 0.37 分。"
weight: 10
---

## 📄 论文信息

- **标题**：*Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving*
- **团队**：Qwen Team × 华中科技大学 — Xin Zhou、Zongchuang Zhao、Zhibo Yang、Mingsheng Li、Shuai Bai 等
- **发表**：arXiv:2609.00111（2026 年 8 月 31 日）
- **代码**：[github.com/QwenLM/Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0)（Apache 2.0）
- **模型**：[Qwen/Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B)（HuggingFace）
- **关键词**：VLM 驾驶基础模型、BEV 感知探针、Flow Matching 轨迹生成、GRPO 式 RL 后训练、多数据集联合训练
- **一句话总结**：**不改 VLM 架构本身**，外挂两个模块（BEV 感知头 + Planning Expert），用四阶段训练把通用 VLM 改造成集 3D 感知、驾驶 VQA、运动规划于一体的驾驶基础模型。Planning Expert 用 Flow Matching 生成轨迹，**不设推理时打分器**，直接单次采样输出。

---

## 🤔 要解决什么问题？

当前驾驶 VLA 的主流做法是"拿一个通用 VLM，在驾驶数据上继续训"——看起来统一，实际有两个坑：

1. **3D 感知被文字 VQA 掩盖**：纯文本 VQA 监督不能直接约束 3D 布局/深度/占用。模型可以输出流畅的场景描述，但在 3D 空间上可能很不精确——你问它"前面那车有多远"，它答得头头是道，但实际规划时距离估错了。
2. **领域适配导致灾难性遗忘**：在驾驶数据上训太久，通用视觉语言能力就丢了。但实际部署需要"座舱+驾驶"一体——一个模型既要规划轨迹，又要能回答"前面那个施工标志什么意思"这类座舱问题。丢了通用能力就得另建一个座舱模型，成本翻倍。

Qwen 团队的回答是：**不动 VLM 架构本身**，通过外挂显式感知模块和轨迹生成模块，加上精心设计的数据混合和训练配方，让同一个 VLM 既保持通用能力，又获得驾驶专业能力。

---

## 🏗️ 架构总览：VLM + 两个外挂模块

Qwen-Drive-1.0 的架构可以用一句话概括：**"一个共享 VLM + 两个外挂专家"**。

*（架构图见论文 Figure 2）*

### 共享主干：Qwen3.5-4B VLM

| 特性 | 设计 | 作用 |
|------|------|------|
| **原生多模态** | ViT 视觉编码器 + 空间合并的视觉 token 直接交错插入文本 token 流 | 图像/视频/语言在单一 Transformer 内统一处理 |
| **混合注意力** | 多数层用门控线性注意力（GLA），间隔插入分组查询 softmax 注意力 | 长多模态序列高效编码，关键位置保留全局精推理 |
| **视角+帧标签** | 8 个视角方向（FRONT VIEW / FRONT RIGHT VIEW…）+ frame: k 帧标签 | 模型知道每张图来自哪个相机、哪个时刻 |

关键点：**VLM 架构完全不动**，不做任何修改——这是为了保持易用性和通用能力不丢失。两个外挂模块从 VLM 的不同位置读取特征。

### 外挂模块一：BEV 感知头（3D Probe）

BEV 感知头**同时做三件事**：3D 目标检测、语义占用预测、BEV 地图分割。它不是"额外任务"，而是作为**显式 3D 感知探针**——让模型的 3D 理解能力有一个可检查、可量化的出口。

**数据流**：

```
多视角图像 → 视觉编码器（冻结/可训）
           → VLM（可训）
           → 取两路特征：
             ① V前（视觉编码器特征，低层外观）
             ② V后（VLM 输出特征，高层语义）
           → 深度提升：用深度网络把 V前 投影到 3D 体素
           → BEV Transformer：V后 特征金字塔 + 体素几何先验 → BEV 特征
           → 三个解码头：
             • DETR 解码头 → 3D 检测
             • 3D UNet → 语义占用
             • UNet 解码头 → BEV 地图分割
```

**设计亮点**：
- **两路特征融合**：视觉编码器的低层外观（$F^v$）和 VLM 的高层语义（$F^m$）互补——前者有精确几何，后者有场景理解
- **深度无需监督**：深度网络直接从图像特征预测分类深度分布，不需要深度真值
- **共享 BEV 表征**：三个任务共享同一个 BEV 特征，互相提供正则化

### 外挂模块二：Planning Expert（轨迹生成器）

这是本文最核心的模块——**~1.1B 参数的 32 层扩散 Transformer**，用 Flow Matching 生成未来 5 秒轨迹。

**输出格式**：50 个路点 × $(x, y, \theta)$，5 秒 @10Hz，归一化到 $[-1,1]$（x: 165m，y: 25m，$\theta$: $\pi/2$ rad）

**架构**（32 层 Diffusion Transformer）：

| 组件 | 设计 |
|------|------|
| **条件注入** | VLM 的 8 个 softmax attention 层的 **缓存 KV**（每 4 层共享一份），每层做联合注意力 |
| **路点 token 拼接** | 带噪路点 + 历史轨迹编码 + Fourier 特征 + 流时间 + 位置编码 |
| **AdaLN 调制** | 注入流时间 $t$、导航指令 $\mathbf{n}$、当前自车状态 $\mathbf{e}$ |
| **隐藏维度** | 1024 |
| **参数量** | ~1.1B |

**核心问题：推理时怎么生成轨迹？**

> **Flow Matching + 10 步欧拉积分**
>
> 训练时：$\boldsymbol{\tau}_t = (1-t)\boldsymbol{\tau}_0 + t\boldsymbol{\tau}_1$（线性插值），模型预测干净终点 $\hat{\boldsymbol{\tau}}_1$（而非速度场）
>
> 推理时：$\boldsymbol{\tau}_0 \sim \mathcal{N}(0, I)$ → 10 步欧拉积分 → 得到 $\hat{\boldsymbol{\tau}}_1$ → 即最终轨迹

**训练损失**：

$$\mathcal{L}_{\text{plan}} = \mathcal{L}_{\text{fm}} + 2 \times 10^{-4}\,\mathcal{L}_{\Delta^1} + 2 \times 10^{-5}\,\mathcal{L}_{\Delta^2}$$

其中：
- $\mathcal{L}_{\text{fm}}$：Flow Matching 损失（预测速度场 vs 真实速度场的 MSE）
- $\mathcal{L}_{\Delta^1}$：一阶时序差分（Huber 惩罚），抑制路点抖动
- $\mathcal{L}_{\Delta^2}$：二阶时序差分，抑制加速度突变

---

## 🔑 关键问题：Flow Matching 生成多条轨迹，靠什么选？

你问的这个疑惑非常精准——**Qwen-Drive-1.0 的 Planning Expert 确实可以用不同噪声采样出多条轨迹，但它** **没有** **训练一个推理时打分器来选**。这和 Scoring-based 方法（如 CLOVER）有本质区别。

### 它到底怎么选？答案：不选，直接用单次采样

1. **推理时**：给定当前场景，从 $\mathcal{N}(0, I)$ 采一次噪声，跑 10 步欧拉积分，输出**一条**轨迹，直接执行。没有打分器、没有候选排序。

2. **"best-of-6" 是评测协议，不是推理方法**：论文在评测时会做 `best-of-6`——采 6 条轨迹，用**真值（ground truth）**选最好的那条。但这是**上界（oracle bound）**，不是实际部署方法。论文原文明确写道：
   > *"That selection uses the ground truth, so it is an upper bound on what inference-time selection could reach."*

3. **PDM 分数只在训练时用，不在推理时用**：Stage 4 RL 后训练用 PDM 分数作为奖励信号（和 GRPO 式分组优势），但这只影响训练——训练完之后，推理时照样是单次采样、直接输出。

### 为什么敢不设打分器？

这是 Qwen-Drive-1.0 和 Scoring-based 路线的**根本分歧**：

| | Scoring-based（CLOVER） | Qwen-Drive-1.0 |
|------|------|------|
| **生成方式** | 候选生成器产出 N 条候选 | Flow Matching 单次采样 |
| **选择机制** | 学一个打分器（Cross-Attention）对每条候选打子分数，选 Top-1 | **不选**，直接用生成的那条 |
| **推理计算** | N+1 次前向（N 候选 + 1 打分器） | 1 次前向（10 步积分） |
| **依赖** | 打分器质量、候选覆盖度 | 生成器本身的分布质量 |
| **RL 后训练** | 通常不做（或只做轻量对齐） | GRPO 式分组优势，8 条 rollouts 优化 |

**为什么不设打分器？** 论文没有直接解释，但可以从架构哲学推断：
- Qwen-Drive-1.0 的定位是**VLM 驾驶基础模型**，强调"不改 VLM 架构、保持通用能力"。加一个打分器会引入额外的训练目标和架构耦合，可能干扰 VLM 的通用表征。
- Flow Matching 本身的分布质量足够高——只要训练充分，单次采样就能输出合理的多模态轨迹（不同噪声 → 不同合理开法）。RL 后训练进一步把分布推向高分区域。
- 这也是一种**"生成质量 > 后处理选择"**的哲学：与其费力学打分器去选，不如直接让生成器本身学好。

### 实际效果对比

| 方法 | NAVSIM PDMS | 推理方式 |
|------|------|------|
| CLOVER（Scoring-based） | 94.5 | 生成 50 候选 → 打分器选 Top-1 |
| Qwen-Drive-1.0-RL | 90.7 | 单次 Flow Matching 采样 |
| Qwen-Drive-1.0-RL best-of-6 | 91.4 | 6 次采样 → GT 选（oracle） |

差距在 3-4 个点。Scoring-based 的打分器在固定场景（navtest）确实有优势，但 Qwen-Drive-1.0 靠 RL 后训练把分布收紧，在 WOD-E2E（7.91 RFS）和 AlpaSim 闭环上也展现了竞争力——而且推理速度更快（单次前向 vs N+1 次）。

---

## 🎓 四阶段训练配方

Qwen-Drive-1.0 的训练分四个阶段，每个阶段冻结/解冻不同模块，逐步引入任务：

### Stage 1：感知头预训练

| 项目 | 内容 |
|------|------|
| **冻结** | 视觉编码器 + VLM |
| **训练** | 仅 BEV 感知头 |
| **损失** | $\mathcal{L}_{\text{perc}} = \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{occ}} + \mathcal{L}_{\text{map}}$ |
| **目的** | 让新加入的感知头学会构建 BEV 表征 |

### Stage 2：感知 + VQA 联合训练

| 项目 | 内容 |
|------|------|
| **训练** | 视觉编码器 + VLM + BEV 感知头 |
| **损失** | 感知样本用 $\mathcal{L}_{\text{perc}}$，VQA 样本用 $\mathcal{L}_{\text{ntp}}$（下一个 token 预测） |
| **学习率** | 感知头 LR = VLM LR × 20 |
| **数据** | 1.54M 样本（9.7% 感知 + 26% 通用 VQA + 64.3% 驾驶 VQA） |
| **目的** | 让 VLM 表征适应 3D 感知，同时保留通用能力 |

**关键设计**：每个 minibatch 同时包含感知样本和 VQA 样本。不活跃的分支用 dummy 输入保持计算图一致（分布式训练需要），dummy 输出不参与损失。

### Stage 3：Planning Expert 预训练（SFT）

| 项目 | 内容 |
|------|------|
| **冻结** | 视觉编码器 + VLM |
| **训练** | 仅 Planning Expert |
| **损失** | $\mathcal{L}_{\text{plan}}$（Flow Matching + 时序正则） |
| **条件** | 可选文本推理链（$\mathbf{r}$ 或 $\varnothing$） |
| **目的** | 让 Planning Expert 学会从 VLM 表征生成轨迹 |

### Stage 4：RL 后训练（GRPO 式）

| 项目 | 内容 |
|------|------|
| **冻结** | 视觉编码器 + VLM |
| **训练** | 仅 Planning Expert |
| **方法** | 类 GRPO 分组优势优化 |
| **每场景** | 采 8 条 rollouts（共享初始噪声 + 不同低频扰动） |
| **奖励** | NAVSIM: PDMS + ADE；WOD-E2E: RFS + ADE；PAI-AV: PDMS + ADE |
| **目的** | 把轨迹分布推向高分区域，弥补模仿学习"只学一条轨迹"的局限 |

**RL 的技术细节**（论文最有新意的部分之一）：

1. **低频扰动**：不加独立路点噪声（会高频抖动），而是在 50 个路点的前 6 个余弦基上加扰动——产生**平滑、连贯的轨迹偏移**（如整体偏左、整体加速），而非点级抖动。

2. **恢复分数**：扰动后轨迹可能偏离 Flow Matching 的偏好区域，论文引入一个**近似恢复分数**（score correction），指向条件中心，稳定训练。

3. **训练时随机性集中在最后 3 步**（$k \in \{7, 8, 9\}$）：因为靠近输出端的扰动对最终轨迹影响最直接，早期扰动会被后续积分步骤衰减。

---

## 📊 数据配方：跨数据集统一

Qwen-Drive-1.0 的一个被低估的贡献是**数据工程**——它把多个异构驾驶数据集统一到同一个训练框架下。

### 感知数据：nuScenes + OpenScene

两个数据集的标注体系不同（nuScenes 手工标注，OpenScene 自动管线重建），需要做**标签统一**和**空间统一**：

**标签统一**：
- 3D 检测：nuScenes 的 5 种车辆类合并为 `vehicle`，`bicycle` + `motorcycle` 合并为 `bicycle` → 7 类
- 语义占用：用查找表把两个数据集的标签映射到共享 10 类空间
- 离线标签补全：用可靠辅助标注补全缺失类别（如 OpenScene 缺 `driveable`，用 nuPlan 地图补）

**空间统一**：
- nuScenes 占用网格：±40m，$z \in [-1.0, 5.4]$m，0.4m 体素
- OpenScene 占用网格：±50m，$z \in [-4.0, 4.0]$m，0.5m 体素
- 解决方案：**不重采样标签**，在预测特征上做可微三线性采样对齐——让同一个占用头在两个原生网格上都能预测

### 视觉语言数据：24 个公开数据集 + 自建

**公开数据集处理**：
1. 用 Qwen3.5-Plus **统一改写**每个数据集的 prompt 和 response 为对话格式
2. 用 Qwen3.5-Flash 做**一致性过滤**：语义一致才保留
3. 过滤后从 5.53M 样本降到 3.09M（保留率 55.9%）

**自建三类数据**：
1. **规划推理链（CoC）**：从 NAVSIM / Waymo / PAI-AV 的轨迹，用规则分类器推导纵向/横向动作先验，再用 Qwen3.7-Plus 生成因果推理链，多阶段审计
2. **相机排序**：打乱视角图像、去掉视角标签，让模型从视觉线索恢复前后顺序和顺时针排列
3. **内部感知 QA**：30K 中国道路场景的红绿灯定位 + 3D 检测样本

### Stage 2 数据混合

| 类别 | 比例 | 来源 |
|------|------|------|
| 3D 感知 | 9.7% | nuScenes + OpenScene |
| 通用 VQA | 26.0% | MMBench / MMMU / RealWorldQA 等 |
| 驾驶 VQA | 64.3% | 24 个公开驾驶数据集 + 自建 CoC / 相机排序 / 感知 QA |

重复策略：感知样本重复更多（增加 BEV 头更新频率），VQA 样本重复 2-3 epoch。

---

## 📈 实验结果

### 3D 感知

| 方法 | nuScenes mAP | nuScenes map mIoU | OpenScene mAP | OpenScene map mIoU |
|------|------|------|------|------|
| BEVFormerV2 | 43.24 | 56.90 | 41.25 | 69.83 |
| **Qwen-Drive-1.0** | **43.95** | **60.99** | **43.45** | **71.27** |

BEV 感知头作为"探针"，在两个数据集上都超过了专门的 BEV 感知模型——说明 VLM 表征确实编码了丰富的 3D 信息。

### 驾驶 VQA

| 方法 | LingoScore | Ego3D-Bench | VLADBench | SURDS | WaymoQA |
|------|------|------|------|------|------|
| Qwen3.5-4B（通用基线） | — | 7.66 | 73.0 | 37.9 | 75.8/74.1 |
| **Qwen-Drive-1.0-SFT** | **79.4** | **8.22** | **83.8** | **49.3** | **88.1/88.1** |

通用能力大幅提升，且从论文 Table 3 可以看到，14 个通用 VQA 基准（MMBench / MMMU / OCRBench 等）上 Qwen-Drive-1.0 的分数几乎没掉——**领域适配没有灾难性遗忘**。

### 运动规划

**NAVSIM v1.1 navtest**：

| 方法 | NC | DAC | EP | TTC | C | PDMS |
|------|------|------|------|------|------|------|
| Qwen-Drive-1.0-SFT w/o reasoning | 98.2 | 96.4 | 82.0 | 94.4 | 100.0 | 87.8 |
| Qwen-Drive-1.0-SFT w/ reasoning | 98.4 | 96.6 | 82.4 | 94.7 | 100.0 | 88.2 |
| Qwen-Drive-1.0-SFT best-of-6 | 98.7 | 97.2 | 83.2 | 95.5 | 100.0 | 89.3 |
| **Qwen-Drive-1.0-RL** | **98.6** | **98.2** | **84.8** | **95.9** | **100.0** | **90.7** |
| Qwen-Drive-1.0-RL best-of-6 | 98.8 | 98.4 | 85.5 | 96.5 | 100.0 | 91.4 |
| CLOVER（Scoring-based 天花板） | — | — | — | — | — | 94.5 |

RL 后训练从 SFT 的 88.2 涨到 90.7（+2.5），主要收益来自 EP（+2.4）和 DAC（+1.6）——说明 RL 确实把轨迹推向了"更敢开"的区域。

**WOD-E2E（Waymo 长尾评测）**：

| 方法 | ADE 3s | ADE 5s | RFS |
|------|------|------|------|
| Qwen-Drive-1.0-SFT | 1.19 | 2.65 | 7.78 |
| **Qwen-Drive-1.0-RL** | **1.19** | **2.67** | **7.91** |
| AutoVLA | 1.35 | 2.96 | 7.56 |
| NoRD | 1.25 | — | 7.71 |

**AlpaSim 闭环**：

| 方法 | CER↓ | Off-road↓ | Progress↑ | At-fault Score↑ |
|------|------|------|------|------|
| Alpamayo-R1 | 19.0 | 17.0 | 67.0 | 0.58 |
| Qwen-Drive-1.0-RL | 41.0 | 12.0 | 48.0 | 0.37 |

闭环上还不算强（Alpaamayo-R1 的 0.58 vs 0.37），说明 RL 后训练在开环/伪闭环上收益明显，但闭环交互场景仍需加强。

---

## 💻 代码仓库逐文件拆解

Qwen-Drive-1.0 的代码仓库结构清晰，核心代码在 `src/qwen_drive/` 下：

```
qwen-drive/
├── src/qwen_drive/              # 核心推理代码
│   ├── modeling_qwen_drive.py   # (17.8KB) 主模型：VLM + Planning Expert 组装
│   ├── planning_expert.py       # (18.1KB) Planning Expert：Flow Matching 轨迹生成
│   ├── scene.py                 # (15.5KB) 场景数据处理：多视角图像打包
│   ├── trajectory.py            # (6.2KB) 轨迹处理：归一化/反归一化
│   ├── metrics.py               # (6.6KB) 评测指标：PDMS/RFS/ADE
│   ├── configuration_qwen_drive.py  # (6.9KB) 模型配置字段
│   ├── benchmarks.py            # (6.9KB) 基准测试加载
│   └── visualize.py             # (6.2KB) 可视化工具
├── src/qwen_drive_perception/   # BEV 感知模式
│   └── ops/                     # CUDA 算子
├── scripts/                     # demo / 评测 / 可视化脚本
├── data/demo/                   # 捆绑的 demo 场景
├── data/benchmarks/             # 四个基准测试场景文件
└── docs/                        # 文档
```

### 模型加载：一个目录、多个子模块

```python
# 模型目录结构（9.1GB VLM + 2.1GB Planner × 2 + 0.5GB Perception）
# Qwen-Drive-1.0-4B/
#   ├── planner-sft/    # Planning Expert，模仿学习训练
#   ├── planner-rl/     # Planning Expert，RL 后训练
#   └── perception/     # BEV 感知头

# 加载方式（from src/qwen_drive/__init__.py）
from qwen_drive import InferenceMode, QwenDriveForPlanning

model = QwenDriveForPlanning.from_pretrained(
    "Qwen-Drive-1.0-4B",
    planner="Qwen-Drive-1.0-4B/planner-rl",  # 选 RL 版
    dtype=torch.bbf16,
    attn_implementation="flash_attention_2",
).to("cuda").eval()

# 可以随时切换 Planning Expert
model.load_planner("Qwen-Drive-1.0-4B/planner-sft")
```

**关键理解**：VLM 在根目录，所有任务共享。Planning Expert 是可插拔的子模块——切换 SFT/RL 版本只需换子目录。

### 推理模式：四种使用方式

```python
from qwen_drive import InferenceMode

# 1. VQA 模式（不加载 planner）
result = model.run(InferenceMode.VQA, scene=scene, question="前方车辆在做什么？")

# 2. 直接规划（不需要推理链）
result = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1)

# 3. 推理规划（先生成推理链，再生成轨迹）⭐
result = model.run(InferenceMode.REASONING_PLANNING, scene=scene, num_samples=6)
print(result.reasoning)  # "The car ahead is braking..."
print(result.trajectories.shape)  # (6, 50, 3) → 6 条轨迹 × 50 路点 × (x,y,heading)

# 4. 感知模式
from qwen_drive_perception import QwenDrivePerception
perception = QwenDrivePerception.from_pretrained("Qwen-Drive-1.0-4B/perception")
det3d, occ, bev_map = perception.run(scene)
```

### Planning Expert 核心代码拆解

**源文件**：`src/qwen_drive/planning_expert.py`（18.1KB）

```python
class PlanningExpert(nn.Module):
    """Flow Matching 轨迹生成器。
    
    核心流程：
    1. 从 VLM 的 softmax attention 层提取缓存 KV（条件）
    2. 带噪路点 + 历史轨迹 + Fourier 特征 → token 嵌入
    3. 32 层 DiT Block 交替做 self-attn（路点间）+ cross-attn（读 VLM）
    4. 输出头预测速度场
    """
    
    def __init__(self, config):
        self.num_waypoints = 50        # 5s @10Hz
        self.waypoint_dim = 3          # (x, y, heading)
        self.hidden_dim = 1024         # DiT 隐藏维度
        self.num_layers = 32           # DiT 层数
        self.num_heads = 16            # 注意力头数
        
        # 路点嵌入：noisy_traj (50×3) + history (50×3) + Fourier (50×freq_dim)
        self.waypoint_embed = nn.Linear(
            self.waypoint_dim * 2 + config.freq_dim,  # 3×2 + freq_dim
            self.hidden_dim
        )
        
        # 流时间嵌入：t → Fourier → MLP → hidden_dim
        self.time_embed = TimestepEmbedding(config.freq_dim, self.hidden_dim)
        
        # 导航指令嵌入
        self.nav_embed = nn.Linear(config.text_dim, self.hidden_dim)
        
        # 32 层 DiT Block
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                cross_attn_dim=config.text_dim,  # 读 VLM 缓存 KV
            )
            for _ in range(self.num_layers)
        ])
        
        # 输出头：hidden_dim → waypoint_dim（预测速度场）
        self.output_proj = nn.Linear(self.hidden_dim, self.waypoint_dim)
    
    def forward(self, noisy_traj, t, history, nav_emb, vlm_kv_cache):
        """
        训练时前向。
        
        Args:
            noisy_traj: (B, 50, 3) 带噪轨迹
            t: (B,) 流时间 ∈ [0, 1]
            history: (B, 50, 3) 历史轨迹
            nav_emb: (B, S, D) 导航指令嵌入
            vlm_kv_cache: list of (K, V) 来自 VLM 的 softmax attention 层
        
        Returns:
            v_pred: (B, 50, 3) 预测速度场
        """
        # Fourier 编码
        fourier = compute_fourier(noisy_traj)  # (B, 50, freq_dim)
        
        # 拼接嵌入
        x = self.waypoint_embed(torch.cat([noisy_traj, history, fourier], dim=-1))
        
        # 加时间步嵌入（broadcast 到每个路点）
        x = x + self.time_embed(t).unsqueeze(1)
        
        # 加导航指令嵌入
        x = x + self.nav_embed(nav_emb).mean(dim=1, keepdim=True)
        
        # 32 层 DiT Block
        for block in self.blocks:
            x = block(
                x,                           # self-attn: 路点间交互
                kv_cache=vlm_kv_cache,       # cross-attn: 读 VLM 缓存
                timestep_emb=self.time_embed(t),
            )
        
        # 输出速度场
        v_pred = self.output_proj(x)  # (B, 50, 3)
        return v_pred
```

### 主模型如何组装 VLM + Planning Expert

**源文件**：`src/qwen_drive/modeling_qwen_drive.py`（17.8KB）

```python
class QwenDriveForPlanning(Qwen3ForCausalLM):
    """在 Qwen3.5 VLM 基础上挂载 Planning Expert。"""
    
    def __init__(self, config):
        super().__init__(config)  # 加载 Qwen3.5-4B VLM
        
        # 加载 Planning Expert（可选 SFT 或 RL 版本）
        self.planner = PlanningExpert(config.planner_config)
        
        # VLM 的 softmax attention 层索引（每 4 层共享一份 KV）
        self.condition_layers = config.condition_layers  # [3, 7, 11, ...]
    
    def run(self, mode, scene, num_samples=1):
        """统一推理入口。"""
        if mode == InferenceMode.VQA:
            return self._run_vqa(scene)
        elif mode == InferenceMode.DIRECT_PLANNING:
            return self._run_planning(scene, num_samples, with_reasoning=False)
        elif mode == InferenceMode.REASONING_PLANNING:
            return self._run_planning(scene, num_samples, with_reasoning=True)
    
    def _run_planning(self, scene, num_samples, with_reasoning):
        """规划推理流程。"""
        # 1. 编码场景（多视角图像 + 文本）
        inputs = self._encode_scene(scene, with_reasoning)
        
        # 2. VLM 前向，同时缓存 softmax attention 层的 KV
        with torch.no_grad():
            vlm_output, kv_caches = self.forward(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # 3. 提取条件层的 KV 缓存
        condition_kvs = [kv_caches[i] for i in self.condition_layers]
        
        # 4. Planning Expert 采样（Flow Matching）
        trajectories = self.planner.sample(
            num_samples=num_samples,
            vlm_kv_cache=condition_kvs,
            nav_instruction=inputs["nav_instruction"],
            history=scene.ego_history,
        )
        
        return PlanningResult(trajectories=trajectories, reasoning=vlm_output.text)
```

### 场景数据处理

**源文件**：`src/qwen_drive/scene.py`（15.5KB）

```python
class Scene:
    """单个驾驶场景的数据结构。
    
    包含：
    - 多视角图像（最多 8 个相机）
    - 自车历史轨迹（用于 Planning Expert 的条件输入）
    - 导航指令
    - 轨迹标签（训练时）
    """
    
    def __init__(self):
        self.images: dict[str, Image]        # {"FRONT": Image, "FRONT_RIGHT": Image, ...}
        self.ego_trajectory: Tensor           # (T, 3) 历史轨迹
        self.navigation: str                  # 导航指令文本
        self.trajectory_label: Tensor | None  # (50, 3) 标注轨迹（训练时）
        self.risk_label: str | None           # 风险标注（评测时）

class ImageArchive:
    """图像打包格式（parquet 文件）。
    
    为了高效 I/O，多个场景的图像被打包到 parquet 文件中，
    通过 scene_id + frame_id 索引。
    """
    
    @staticmethod
    def open(path: str) -> "ImageArchive":
        """加载 parquet 图像归档。"""
        ...
    
    def get(self, scene_id: str, frame_id: str, camera: str) -> Image:
        """获取指定场景/帧/相机的图像。"""
        ...
```

### 轨迹归一化

**源文件**：`src/qwen_drive/trajectory.py`（6.2KB）

```python
# 归一化参数（论文 Table 中提到）
NORMALIZATION = {
    "x": {"range": 165.0, "unit": "m"},     # 前向 165m
    "y": {"range": 25.0, "unit": "m"},      # 横向 25m
    "heading": {"range": 3.14159 / 2, "unit": "rad"},  # ±π/2
}

def normalize_trajectory(traj: Tensor) -> Tensor:
    """将轨迹归一化到 [-1, 1]。"""
    # x: (x / 165.0) * 2 - 1
    # y: (y / 25.0) * 2 - 1
    # heading: (heading / (π/2)) * 2 - 1
    ...

def denormalize_trajectory(traj: Tensor) -> Tensor:
    """将归一化轨迹还原到原始坐标。"""
    ...
```

### 评测脚本

```bash
# NAVSIM 评测
python scripts/evaluate.py \
    --config configs/eval_navsim.yaml \
    --model-path Qwen-Drive-1.0-4B \
    --planner planner-rl

# WOD-E2E 评测
python scripts/evaluate.py \
    --config configs/eval_wod.yaml \
    --model-path Qwen-Drive-1.0-4B \
    --planner planner-rl

# 感知评测
python scripts/evaluate_perception.py \
    --config configs/eval_perception.yaml \
    --model-path Qwen-Drive-1.0-4B
```

---

## 🔬 个人解读与思考

### 1. "不设打分器"是设计选择，不是疏忽

Qwen-Drive-1.0 选择不在推理时用打分器选轨迹，这和 Scoring-based 路线（CLOVER 等）形成鲜明对比。这不是疏忽，而是**架构哲学的选择**：

- **Scoring-based**：依赖打分器质量 + 候选覆盖度。打分器越强，Top-1 越准；但推理计算量是 N+1 次前向。
- **Qwen-Drive**：依赖生成器本身的分布质量。Flow Matching 训练好后，单次采样就能输出合理轨迹；推理只需 1 次前向（10 步积分）。

**tradeoff**：Scoring-based 在固定场景（navtest）有精度优势（CLOVER 94.5 vs Qwen 90.7），但 Qwen 的推理效率更高、架构更简单，且 RL 后训练能持续把分布推向高分区域。

### 2. RL 后训练的低频扰动设计很精巧

论文不让噪声独立加在每个路点上（会产生高频抖动），而是限制在 6 个余弦基上——相当于只在"整体偏左/偏右""整体加速/减速"这几个低维模态上探索。这和人类驾驶的直觉一致：你不会突然在第 3 个路点拐一下，而是整体调整策略。

### 3. 跨数据集统一是真正的工程贡献

把 nuScenes 和 OpenScene 的占用标注统一到同一个训练框架（不同分辨率、不同坐标系、不同标注质量），需要做标签统一 + 空间对齐 + 离线补全。这种"脏活"在论文里通常一笔带过，但实际决定了联合训练能不能跑通。

### 4. 闭环场景仍是短板

AlpaSim 闭环上 Qwen-Drive-1.0 的 0.37 和 Alpamayo-R1 的 0.58 还有差距。可能原因：
- Flow Matching 生成的轨迹在交互场景下缺乏"反应式"调整（非反应式仿真得分高，但闭环里其他车会反应）
- RL 奖励主要基于 NAVSIM/Waymo 的开环指标，对闭环交互的覆盖不足
- 未来可能需要引入闭环仿真训练（类似 DriveWAM / SimWAM 的世界模型路线）

### 5. "基础模型"定位值得关注

Qwen-Drive-1.0 的定位不是"刷榜机器"，而是"驾驶基础模型"——强调一个模型同时覆盖 3D 感知、驾驶 VQA、运动规划、通用 VQA。这种"全能型"定位在产业界可能比"单项刷榜"更有价值：部署时只需维护一个模型，座舱和驾驶共享算力。

---

## 📝 一句话总结

**Qwen-Drive-1.0 = Qwen3.5-4B VLM + BEV 感知探针 + Flow Matching 规划专家，四阶段训练（感知预训练 → 联合 VQA → 轨迹 SFT → RL 后训练）。规划时不设打分器、不做候选排序，直接单次 Flow Matching 采样输出轨迹——靠生成器本身的质量 + RL 把分布推向高分区域。NAVSIM 90.7 PDMS，WOD-E2E 7.91 RFS，通用能力几乎无损。**
