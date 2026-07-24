---
title: "个人思考｜JEPA-DRIVE：用联合嵌入预测架构做自监督驾驶决策"
date: 2026-07-24
draft: false
categories: ["个人思考"]
tags: ["🧠 JEPA", "🚗 自动驾驶", "🎯 自监督学习", "📐 世界模型", "⚡ 推演式决策"]
summary: "JEPA-DRIVE 是一个基于 Joint Embedding Predictive Architecture 的自监督驾驶决策系统。它用结构化隐空间世界模型编码驾驶场景，从 65,536 个离散行为词汇中组合生成候选轨迹，再用 cycle energy 重建误差作为自监督评分信号。V17 在 NAVSIM 上达到 0.8839 selected_pdm，V21 正在用纯自监督 cycle energy 突破 PDM 代理标签的 64 值瓶颈（8×A800 分布式训练中）。全模型仅 3.88M 参数。"
weight: 1
---

## 1. JEPA 架构：从 LeCun 的世界模型蓝图到具体实现

### 1.1 起源：LeCun 的 2022 年蓝图

2022 年，Yann LeCun 发表了一篇影响深远的位置论文《A Path Towards Autonomous Machine Intelligence》，提出了一个完整的 AI 系统架构蓝图。JEPA（Joint Embedding Predictive Architecture）是其中的核心组件——一个用于构建世界模型的**自监督学习框架**。

LeCun 的核心理念：

> **智能的本质不是模式匹配，而是拥有一个"世界模型"——能够在抽象空间中预测行为后果的内部模型。**
>
> — Yann LeCun, 2022

### 1.2 I-JEPA（2023）：第一个具体实现

2023 年 Meta 发布了 I-JEPA（Image-based JEPA），这是 JEPA 在图像上的首次具体实现，也是后续所有 JEPA 变体的模板。

![I-JEPA 架构：从 context block 预测 target block 的隐空间表征（来自 Meta AI）](/images/jepa/ijepa-architecture.png)

**核心思想**：给定一张图像，遮挡多个**大块区域**（target blocks），只保留一个**信息丰富的上下文块**（context block），让模型在隐空间中预测被遮挡区域的表征——不需要 pixels，只需要 representations。

I-JEPA 训练流程包含 6 步：

1. 从图像中采样 **context block**（保留区域）
2. 采样多个 **target block**（遮挡区域，需大尺度）
3. **context encoder** (ViT) → context representation
4. **predictor**（轻量 Transformer）→ 预测 target representations
5. **target encoder**（动量更新）→ 真实 target representations
6. **L2 损失**：||pred - target||²（仅在隐空间，不预测像素）

**三个网络组件**：

| 组件 | 作用 | 架构 | 参数更新 |
|------|------|------|---------|
| **Context Encoder** | 编码上下文区域 → context rep | ViT | 梯度更新 |
| **Target Encoder** | 编码目标区域 → target rep（真实值） | ViT（与 context 结构相同） | **动量更新**（EMA of context encoder） |
| **Predictor** | 从 context rep 预测 target rep | 轻量 Transformer | 梯度更新 |

**两个关键设计选择**：

1. **Target blocks 必须足够大**（>15% 图像面积）→ 迫使模型学习语义级表征，而非像素级纹理
2. **Context block 必须信息丰富**（空间分布广）→ 避免模型走捷径（只预测附近区域）

**结果**：I-JEPA 在 ImageNet 上用 ViT-Huge/14 在 16 个 A100 GPU、72 小时内完成训练，下游分类、目标计数、深度估计等任务达到 SOTA，且**完全不需要数据增强**（对比 SimCLR/BYOL 依赖的随机裁剪、颜色抖动等）。

### 1.3 V-JEPA（2024）：从图像到视频

2024 年 Meta 发布了 V-JEPA，将 JEPA 从图像扩展到视频。

**核心变化**：
- 输入从图像变为视频片段
- Target blocks 从空间掩码变为**时空掩码**（遮挡某些帧的某些区域）
- 学习目标从"空间上下文预测"变为"时空上下文预测"

V-JEPA 学会了从无标注视频中理解物体运动、交互关系——比如"球滚向杯子，杯子会被撞倒"这样的物理常识。

### 1.4 V-JEPA 2（2025）：世界模型 + 机器人规划

2025 年的 V-JEPA 2 将规模推到 10 亿参数级别，训练数据超过 **100 万小时互联网视频 + 62 小时机器人视频**。更重要的是，它引入了一个**动作条件变体**（V-JEPA 2-AC）：

V-JEPA 2-AC 的机器人规划推理流程：

```
当前画面 → V-JEPA 2-AC → 预测多种未来 → 评分每种未来与目标的距离 → 选最优动作
```

核心：**在隐空间推演，不需真实执行**。

**Key insight**：V-JEPA 2-AC 在机器人零样本规划上取得了突破——在完全没有见过的环境、没有任务特定奖励的情况下，能抓取和放置从未见过的物体。这正是 JEPA "预测兼容性而非动作" 哲学的体现。

### 1.5 JEPA 家族全景

| 变体 | 年份 | 模态 | 预测目标 | 关键贡献 |
|------|------|------|---------|---------|
| **I-JEPA** | 2023 | 图像 | 被遮挡区域的隐空间表征 | 首个具体实现，验证 JEPA 可行性 |
| **MC-JEPA** | 2024 | 图像 | 运动特征 + 内容特征 | 分离运动与语义 |
| **V-JEPA** | 2024 | 视频 | 时空掩码区域的表征 | 扩展到视频，学习物理常识 |
| **V-JEPA 2** | 2025 | 视频+动作 | 以动作为条件的未来预测 | 10 亿参数，零样本机器人规划 |
| **VL-JEPA** | 2025 | 视觉+语言 | 视觉-语言联合表征 | 替代自回归解码，加速 2.85× |

**红线贯穿**：所有变体都遵循 JEPA 的核心原则——**在隐空间做预测，不在输入空间做预测**。

---

## 2. JEPA-DRIVE：JEPA 在驾驶决策中的落地

JEPA-DRIVE 将 JEPA 的"隐空间预测 + 能量函数"思想应用到驾驶决策。但与 I-JEPA/V-JEPA 不同：

| 维度 | I-JEPA / V-JEPA（标准 JEPA） | JEPA-DRIVE（我们的） |
|------|----------------------------|---------------------|
| **输入** | 图像 / 视频像素 | **结构化场景特征**（物体/车道/地图等向量） |
| **编码器** | ViT（通用视觉） | **多模态 MLP**（独立编码各场景要素） |
| **预测目标** | 被遮挡区域的表征 | **轨迹后半段的表征** |
| **训练数据** | 互联网图像 / 视频 | **NAVSIM 驾驶场景** |
| **使用方式** | 预训练 → 下游微调 | **直接用于推理决策** |
| **参数量** | ~10 亿级 | **3.88M** |

### 2.1 问题形式化

给定场景特征 $S$ 和候选轨迹集合 $\mathcal{C}$，目标是：

$$ T^* = \arg\max_{T \in \mathcal{C}} f(T, S) $$

其中 $f$ 是评分函数。JEPA-DRIVE 用**自监督重建误差**（cycle energy）作为 $f$，无需外部标签。

**NAVSIM 约束**：每个场景固定提供 8 个 SD 候选，评估 `selected_pdm`、`oracle_pdm`、`gap`。模型可以生成任意多候选用于训练，但最终只看 8 个 SD 候选中的选择。数据规模：85,109 训练场景 + 12,000 测试场景。

### 2.2 架构全景

![JEPA-DRIVE 架构全景](/images/jepa-drive/jepa-architecture.svg)

### 2.3 World Encoder：结构化场景编码

JEPA-DRIVE 接收的不是原始图像，而是感知模块输出的结构化场景特征——6 种模态分别用独立 MLP 编码到统一的 96 维隐空间：

```python
class ModalEncoder(nn.Module):
    def __init__(self, in_dim=24, out_dim=96):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
```

各模态编码后拼接为 `[65 tokens × 96 dim]`：

| 模态 | token 数 | 含义 | 编码方式 |
|------|---------|------|---------|
| Object | 32 | 周围物体位置/类别/速度 | MLP |
| Lane | 16 | 车道线几何/类型 | MLP |
| Map | 4 | 人行道/路肩等地图要素 | MLP |
| Route | 4 | 导航路径点 | MLP |
| Velocity | 1 | 自车速度 | MLP |
| Dynamics | 8 | 动态物体运动状态 | MLP |

**为什么用 MLP 而非 Transformer 编码器？** 场景要素已经高度结构化，MLP 更轻量，且保持模态分离——下游 CrossAttn 可以区分 token 的模态来源。

### 2.4 Vocabulary 系统（核心创新）

驾驶行为分解为横向（path）和纵向（vel）的组合：

```python
self.path_queries = nn.Parameter(torch.randn(512, 96))  # 512 种横向行为
self.vel_queries  = nn.Parameter(torch.randn(128, 96))  # 128 种纵向行为
```

- 512 路径：4 种基础走法 × 8 种幅度 × 16 种偏移
- 128 速度：4 种基础趋势 × 4 种幅度 × 8 种微调

**组合爆炸**：640 个 query 向量 → 512 × 128 = **65,536 种行为**。这是 3.88M 参数的秘密。

生成流程：Path/Vel Queries → CrossAttn(场景) → Cartesian Product → Predictor → 轨迹解码。

### 2.5 FineScorer V2（评分器）

```python
class FineScorerV2(nn.Module):
    # 输入: action_lat + attended + traj_lat + cycle_energy
    # ① TrajFusion: concat → MLP(288→1024→96)
    # ② CrossAttn(融合特征, world_tokens)
    # ③ SceneQuery: 跨候选全局上下文
    # ④ ScoreHead: MLP(192→1024→1024→1) → 分数
    # ⑤ ComponentHead: MLP(96→512→6) → 6 维组件
```

关键修复：V10 用全局 mean pool（所有候选共享场景），V16 改 CrossAttn（每个候选独立查询场景）→ 0.84 → 0.8839。

### 2.6 Cycle Energy：自监督评分信号

Cycle energy 将 JEPA 的能量基础模型思想具体化：

> 给定场景 $S$ 和轨迹 $T$，能量 $E(T,S)$ 越低，表示 $T$ 与 $S$ 越兼容。

![Cycle Energy 示意图](/images/jepa-drive/cycle-energy.svg)

**计算流程**（轨迹 12 帧切成两半）：

1. `action_lat = MLP(prefix)` → [96]
2. `traj_lat = MLP(full_traj)` → [96]
3. `attended = CrossAttn(action_lat, world_tokens)` → [96]
4. `pred_suffix = Decoder(concat[action_lat, attended, traj_lat])` → [6, 2]
5. `cycle_energy = MSE(pred_suffix, real_suffix)`

**相比 PDM 标签的优势**：

| 维度 | PDM（64 值瓶颈） | Cycle Energy |
|------|----------------|-------------|
| 值域 | 64 个离散值 | **连续值**，任意精度 |
| 来源 | 外部 PDM 打分器 | **自监督** |
| 数据覆盖 | 仅 9k / 85k 场景 | **全部 85k 场景** |
| 语义 | 单一分数 | **可分解** |

### 2.7 训练

**Phase A（生成器）**：100 epochs, 1×L20, 7119 有标注场景
- 损失：重建 + cycle 一致性 + 多样性 + 距离
- 收敛：E1 loss=1.30 → E100 loss=0.06

**Phase B（评分器）**：50 epochs, 8×A800, 85109 全量场景
- 冻结生成器，仅训练 FineScorerV2（3.17M/3.88M 参数）
- V17：$\mathcal{L}_{\text{listnet}}(scores, proxy) + 0.2\bar{E}_{\text{cyc}} + 0.1\mathcal{L}_{\text{BCE}}$
- V21：$\mathcal{L}_{\text{listnet}}(scores, -ce) + 0.05\mathcal{L}_{\text{GRPO}} - 0.01\mathcal{H}(\pi)$

### 2.8 实验演进

| 版本 | 关键变化 | 分数 |
|------|---------|------|
| V10 | 评分器定型，全局池化缺陷 | ~0.80 |
| V16 | 移除全局池化，改 CrossAttn | ~0.85 |
| **V17** | **架构定型：FineScorerV2** | **0.8839** |
| V18-V20 | 尝试各种改进 → 确认 64 值瓶颈 | 0.875-0.882 |
| **V21** | **Cycle Energy 自监督（训练中）** | **?** |

V18-V20 无论怎么改架构都在 0.88 徘徊——因为 PDM 标签只有 64 个离散值，评分器的理论上限在那里。Cycle energy 的连续值是突破方向。

---

## 3. 与业界路线对比

### JEPA-DRIVE vs VLA

| 维度 | VLA（DriveVLA/UniAD） | JEPA-DRIVE |
|------|----------------------|------------|
| 核心 | 模仿学习 $P(T \mid O)$ | 推演选择 $\arg\max_T P(O \mid T)$ |
| 训练 | 互联网图文 + 驾驶标注 | **仅驾驶场景，完全自监督** |
| 参数 | 7B+ | **3.88M** |
| 决策 | 单次 forward | 65,536 候选 → 评分 → 选最优 |
| OOD | 弱 | **强**（场景一致性） |

### JEPA-DRIVE vs PDM-Scorer

PDM-Scorer（NAVSIM 官方 baseline）用 MLP 对 8 个 SD 候选打分，上限 0.91-0.93。JEPA-DRIVE 突破了候选数量限制（8→65,536），并移除了对 PDM 标签的依赖。

### 优势与局限

**优势**：3.88M 参数（1800× 小于 VLA）、完全自监督、推演式 OOD 鲁棒、重建误差可解释

**局限**：推理延迟高于 VLA、依赖编码器质量、65,536 候选仍有覆盖盲区

---

## 4. 个人思考

### 4.1 标准 JEPA 与 JEPA-DRIVE 的核心差异

回顾标准 JEPA（I-JEPA/V-JEPA）和 JEPA-DRIVE 的差异：

标准 JEPA 是**通用视觉表征学习框架**——输入像素、编码器 ViT、输出可迁移的表征。它的目标是"理解世界"，不是"做决策"。

JEPA-DRIVE 是**驾驶决策系统**——输入结构化场景特征、编码器 MLP、输出轨迹排序。它的目标是"选最好的轨迹"。

两者共享相同的哲学（隐空间预测 + 能量函数），但在工程实现上完全不同。这种差异是有意为之：驾驶场景已经有成熟的感知模块输出结构化特征，不需要从像素重新学起。

### 4.2 3.88M 意味着什么

实现自动驾驶决策的核心模型只需要 3.88M 参数——因为 JEPA 不学"怎么开车"，它学的是**场景和轨迹之间的一致性**。一致性比驾驶更基础：模型不需要知道"该加速还是减速"，只需要知道"这条轨迹是否符合场景"。

### 4.3 64 值瓶颈的教训

改架构之前先确认瓶颈在哪。V18-V20 花大量时间改架构但没用——瓶颈不在模型容量，在标签质量。

### 4.4 与 LinkVLA 的对比

两者都用离散词汇表，动机不同：LinkVLA 用共享词表对齐语言-动作 embedding；JEPA-DRIVE 用组合生成覆盖最大行为空间。

---

*📖 V21 训练结果预计 7/26 完成，届时更新。*
