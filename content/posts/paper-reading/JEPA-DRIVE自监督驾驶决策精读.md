---
title: "个人思考｜JEPA-DRIVE：用联合嵌入预测架构做自监督驾驶决策"
date: 2026-07-24
draft: false
categories: ["个人思考"]
tags: ["🧠 JEPA", "🚗 自动驾驶", "🎯 自监督学习", "📐 世界模型", "⚡ 推演式决策"]
summary: "JEPA-DRIVE 是一个基于 Joint Embedding Predictive Architecture 的自监督驾驶决策系统。它用结构化隐空间世界模型编码驾驶场景，从 65,536 个离散行为词汇中组合生成候选轨迹，再用 cycle energy 重建误差作为自监督评分信号。V17 在 NAVSIM 上达到 0.8839 selected_pdm，V21 正在用纯自监督 cycle energy 突破 PDM 代理标签的 64 值瓶颈（8×A800 分布式训练中）。全模型仅 3.88M 参数。"
weight: 1
---

## TL;DR

**JEPA-DRIVE** 是一个完全不同于 VLA 范式的驾驶决策系统：

| 维度 | 传统 VLA | JEPA-DRIVE |
|------|---------|------------|
| 核心思想 | 模仿学习 $P(T \mid O)$ | 推演式选择 $\arg\max_T P(O \mid T)$ |
| 参数 | 7B+ | **3.88M**（相差 1800 倍） |
| 训练数据 | 互联网图文 + 驾驶标注 | 仅驾驶场景数据，**完全自监督** |
| 决策方式 | 单次 forward 生成 | 65,536 候选 → 评分 → 选最优 |
| OOD 处理 | 弱（统计相关性） | 强（场景一致性推理） |
| 可解释性 | 黑盒 | 重建误差可视化 |

---

## 目录

1. [JEPA 哲学：为什么不在像素空间做预测？](#1-jepa-哲学为什么不在像素空间做预测)
2. [自动驾驶决策的问题形式化](#2-自动驾驶决策的问题形式化)
3. [JEPA-DRIVE 架构详解](#3-jepa-drive-架构详解)
4. [Cycle Energy：自监督评分信号](#4-cycle-energy自监督评分信号)
5. [两阶段训练方法](#5-两阶段训练方法)
6. [实验演进：V1→V21](#6-实验演进v1v21)
7. [与业界路线对比](#7-与业界路线对比)
8. [个人思考](#8-个人思考)

---

## 1. JEPA 哲学：为什么不在像素空间做预测？

### 1.1 从 LeCun 的"蛋糕"比喻说起

Yann LeCun 在 2022 年的 AAAI 上提出了一个著名的比喻：

> **纯强化学习是蛋糕上的樱桃，监督学习是蛋糕上的糖霜，自监督学习是蛋糕本身。**
> — Yann LeCun, 2022

JEPA（Joint Embedding Predictive Architecture）是他的"世界模型"构想的核心实现。要理解 JEPA，需要先理解一个根本问题：**为什么"预测"是智能的基础？**

### 1.2 预测 vs 分类：两种学习范式

传统监督学习做的是**分类/回归**：给定输入 $x$，预测标签 $y$。模型学到的是 $P(y|x)$。

JEPA 做的是**隐空间预测**：给定观测 $x$ 的部分信息，在抽象表征空间中预测 $x$ 的剩余部分。模型学到的是两个表征之间的**兼容性函数**。

| 范式 | 输入 | 输出 | 表征空间 |
|------|------|------|---------|
| 监督学习 | 图像 | 类别标签 | 离散的、任务特定的 |
| 自监督（MAE） | 部分像素 | 完整像素 | 像素级（高维、冗余） |
| **JEPA** | 部分表征 | **隐空间表征** | **抽象的、结构化的** |

### 1.3 为什么像素空间预测不行？

像素级预测（如 MAE、扩散模型）有两个根本问题：

**问题 1：维度灾难**

$$ \text{像素空间维度：} 224 \times 224 \times 3 = 150,528 $$

其中 90% 以上是与驾驶决策无关的信息——天空的纹理、路面的颗粒、远处树木的晃动。模型被迫把所有无关细节都学一遍，计算量巨大。

**问题 2：统计相关性 ≠ 因果推理**

像素级模型学到的是"像素 A 旁边通常是像素 B"这样的统计相关性。当遇到分布外场景时（如施工区域），相关性坍塌——模型没见过"路栏+消失车道线"的组合，就不知道该怎么走。

### 1.4 JEPA 的解决方案

JEPA 的核心主张就一句话：

> **不要在像素空间做预测，要在抽象的隐空间做预测。**

具体做法：

1. **编码器 $f_\theta$**：将输入 $x$ 映射到隐空间表征 $s = f_\theta(x)$
2. **预测器 $g_\phi$**：在隐空间中预测缺失部分的表征 $s_{pred} = g_\phi(s_{obs})$
3. **能量函数**：衡量预测表征与实际表征之间的**兼容性**（而非像素差）

```
像素空间（高维、冗余）          隐空间（低维、结构化）
        │                              │
   ┌────┴────┐                   ┌────┴────┐
   │ 编码器   │                   │ 编码器   │
   │ f(x)    │                   │ f(x)    │
   └────┬────┘                   └────┬────┘
        │                              │
   ┌────┴────┐                   ┌────┴────┐
   │ 像素    │                   │ 隐空间  │
   │ 预测    │ ← 冗余、易过拟合  │ 预测    │ ← 抽象、可推理
   └─────────┘                   └─────────┘
```

JEPA-DRIVE 将这一思想具体化为驾驶决策系统：用结构化的隐空间（65 tokens × 96 维）编码场景，在隐空间中"推演"轨迹的未来，用重建误差（cycle energy）作为轨迹质量的评分信号。

---

## 2. 自动驾驶决策的问题形式化

### 2.1 数学定义

给定：
- 场景特征 $S = \{ \text{objects}, \text{lanes}, \text{map}, \text{route}, \text{velocity}, \text{dynamics} \}$
- 候选轨迹集合 $\mathcal{C} = \{T_1, T_2, ..., T_N\}$，每条轨迹 $T_i \in \mathbb{R}^{12 \times 2}$
- 评分函数 $f: (T, S) \rightarrow \mathbb{R}$

目标是：
$$ T^* = \arg\max_{T \in \mathcal{C}} f(T, S) $$

### 2.2 VLA 路线的局限

VLA 直接学习 $\pi_\theta(O) \rightarrow T$，本质是一条**单一路径**：
- 输入 $\rightarrow$ 编码 $\rightarrow$ LLM $\rightarrow$ 解码 $\rightarrow$ 一条轨迹
- 没有"如果这样走会怎样"的推演
- 没有"另一种走法是否更好"的比较

JEPA-DRIVE 走的是**推演路径**：
- 生成大量候选（65,536 种不同走法）
- 对每个候选问"这个走法在场景中合理吗？"
- 选最合理的那个

### 2.3 NAVSIM 基准的约束

NAVSIM 给每个场景固定提供 8 个 SD（Scene Description）候选轨迹。评估指标：

| 指标 | 含义 |
|------|------|
| **selected_pdm** | 模型从 8 个候选中选的轨迹的 PDM 分数 |
| **oracle_pdm** | 8 个候选中最高 PDM 分数 |
| **gap** | oracle_pdm - selected_pdm |

**关键矛盾**：模型可以生成任意多的候选来训练，但最终评估只看从 8 个 SD 候选中的选择质量。

---

## 3. JEPA-DRIVE 架构详解

### 3.1 架构全景

![JEPA-DRIVE 架构全景](/images/jepa-drive/jepa-architecture.svg)

上图展示了 JEPA-DRIVE 的完整数据流。三个关键设计：

### 3.2 结构化场景编码（World Encoder）

与 VLA 将图像送入 CLIP/ViT 不同，JEPA-DRIVE 接收的是**结构化场景特征**——由感知模块输出的 6 种模态的向量：

```python
class ModalEncoder(nn.Module):
    """单模态编码器：将每种场景要素映射到统一的 96 维隐空间"""
    def __init__(self, in_dim=24, out_dim=96):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
```

每种场景要素独立编码后拼接为 `[65 tokens × 96 dim]`：

| 模态 | 输入维度 | token 数 | 编码方式 |
|------|---------|---------|---------|
| Object | 32 × 24 | 32 | MLP 独立编码 |
| Lane | 16 × 24 | 16 | MLP 独立编码 |
| Map | 4 × 24 | 4 | MLP 独立编码 |
| Route | 4 × 24 | 4 | MLP 独立编码 |
| Velocity | 1 × 24 | 1 | MLP 独立编码 |
| Dynamics | 8 × 24 | 8 | MLP 独立编码 |

**为什么用 MLP 而不是 Transformer 编码器？**

场景要素已经足够结构化（物体位置、车道线方向等），不需要 Transformer 的全局交互。MLP 更轻量、更快，且保持模态分离——下游 CrossAttn 可以知道每个 token 来自什么模态。

### 3.3 词汇表系统（核心创新）

**核心思想**：驾驶行为可以被分解为两个独立维度的组合——横向（path）和纵向（vel）。

```python
self.path_queries = nn.Parameter(torch.randn(512, 96))  # 512 种横向行为
self.vel_queries  = nn.Parameter(torch.randn(128, 96))  # 128 种纵向行为
```

**为什么是 512 × 128 = 65,536？**

- 512 种路径：4 种基础走法（直行/左转/右转/变道）× 8 种幅度 × 16 种细粒度偏移
- 128 种速度：4 种基础趋势（加速/巡航/减速/停止）× 4 种幅度 × 8 种微调

**候选生成流程**：

```
Path Queries [512, 96]    Vel Queries [128, 96]
      │                          │
      ▼                          ▼
CrossAttn(场景)             CrossAttn(场景)
      │                          │
      ▼                          ▼
Path Features [512, 96]    Vel Features [128, 96]
      │                          │
      └────────┬─────────────────┘
               │
               ▼
   Cartesian Product: 512 × 128 = 65,536
               │
               ▼
   Predictor → MLP → 轨迹解码 [65536, 24]
```

**关键优势**：组合爆炸。512 种路径 × 128 种速度 = 65,536 种行为模式，但只需要学习 512 + 128 = 640 个 query 向量。这是 JEPA-DRIVE 能够做到 3.88M 参数的秘密——用组合代替记忆。

### 3.4 CrossAttn 解码器

词汇表中的 query 只是"种子"，生成实际轨迹需要结合场景信息：

```
① Path/Vel Queries → Query Projection (适配场景交互)
       │
② CrossAttn(Q=query, K=V=world_tokens)
       │
③ 场景感知的 path_features / vel_features
       │
④ Cartesian Product → comb_latent [512, 128, 96]
       │
⑤ Predictor MLP(96 → target_dim) → Pred Target
       │
⑥ Trajectory Decoder MLP(target_dim → 24) → 最终轨迹
```

### 3.5 FineScorer V2（评分器）

Phase B 唯一训练的部分，经历了 20+ 版本的迭代：

```python
class FineScorerV2(nn.Module):
    def __init__(self, L=96, hidden=1024, dropout=0.1):
        # ① TrajFusion: concat[action_lat, attended, traj_lat] → MLP → 96
        # ② CrossAttn: 候选-场景交互
        # ③ SceneQuery: 跨候选全局上下文
        # ④ ScoreHead: MLP(192 → 1024 → 1024 → 1)
        # ⑤ ComponentHead: MLP(96 → 512 → 6)
```

评分器的设计经历了一个关键修复：**从全局池化到 CrossAttn**。

V10 用全局 mean pool 提取场景特征，所有候选**共享**同一个场景向量——丢失了"某条轨迹是否适合场景的某一部分"这样的精细信息。V16 改用 CrossAttn 让每个候选独立查询场景，这是从 0.84 → 0.8839 的关键突破。

---

## 4. Cycle Energy：自监督评分信号

### 4.1 思想来源

Cycle energy 来源于 JEPA 的**能量基础模型（EBM）**思想：

> 给定场景 $S$ 和轨迹 $T$，能量 $E(T, S)$ 越低，表示 $T$ 与 $S$ 越兼容。

在 JEPA-DRIVE 中，能量函数具体化为**重建误差**：轨迹的后半段能否被前半段+场景准确预测。

![Cycle Energy 示意图](/images/jepa-drive/cycle-energy.svg)

### 4.2 具体计算

```
Prefix（前 6 帧）              Suffix（后 6 帧，待预测）
    ●────●────●────●────●────●    ○────○────○────○────○────○
  t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9  t=10 t=11

① action_lat = MLP(Prefix)             → [96]  轨迹动作编码
② traj_lat = MLP(全轨迹)               → [96]  轨迹全段编码
③ attended = CrossAttn(action_lat, world_tokens) → [96] 场景-动作交互
④ pred_input = concat[action_lat, attended, traj_lat]
⑤ pred_suffix = Decoder(pred_input)    → [6, 2] 预测后半段
⑥ cycle_energy = MSE(pred_suffix, real_suffix)  → 标量
```

### 4.3 为什么重建误差能衡量轨迹质量？

**直觉**：如果一条轨迹在场景中是合理的，那么场景隐空间中包含了"这条路接下来会怎样"的信息，解码器可以准确预测后半段。反之，如果轨迹不合理（例如在有障碍物的车道上直行），场景中没有支持这条轨迹继续前进的信息，解码器只能胡猜——重建误差大。

**数学**：cycle energy 衡量的是**条件似然** $P(\text{suffix} \mid \text{prefix}, S)$ 的负对数。高似然 = 低能量 = 轨迹合理。

### 4.4 相比 PDM 标签的三大优势

| 维度 | PDM 标签 (host_scores) | Cycle Energy（我们的） |
|------|----------------------|----------------------|
| **值域** | 64 个离散值 {0.00, 0.17, ..., 12.00} | **连续值**，任意精度 |
| **来源** | 外部 PDM 打分器（需标注） | **自监督**，从数据中产生 |
| **数据覆盖** | 仅 9k/85k 场景有标注 | **全部 85k 场景**可用 |
| **语义** | 单一分数，无法解释 | **可分解**（位置/速度/方向误差） |

这就是 V21 要回答的核心问题：**cycle energy 的连续值能否突破 PDM 64 值瓶颈？**

---

## 5. 两阶段训练方法

### 5.1 Phase A：生成器训练

```yaml
Epochs: 100 | Batch: 2 | LR: 1e-4 | GPU: 1×L20 (48GB)
数据: 7119 navtrain 场景（有候选标注用于多样性损失）
冻结: 无（全模型训练）
```

**损失函数**：

$$\mathcal{L}_{\text{Phase A}} = \mathcal{L}_{\text{im\_s}} + \mathcal{L}_{\text{im\_w}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{dst}}\mathcal{L}_{\text{dst}}$$

| 分量 | 权重 | 作用 |
|------|------|------|
| 图像重建（强/弱） | 1.0 | 候选轨迹在场景中的特征重建 |
| Cycle 一致性 | 1.0 | 编解码循环中的自一致性 |
| 多样性 | 0.01 | 鼓励 64 个候选多样化 |
| 距离损失 | 0.10 | 候选接近数据集轨迹 |

**收敛曲线**：
```
Epoch  1: loss=1.30, cyc=0.22, div=0.77, dst=2.83
Epoch 50: loss=0.09, cyc=0.005, div=0.92, dst=0.47
Epoch100: loss=0.06, cyc=0.003, div=0.92, dst=0.36
```

### 5.2 Phase B：评分器训练

```yaml
Epochs: 50 | Batch: 8×8GPUs=64 | LR: 1e-4 | GPU: 8×A800-80GB
数据: 85109 navtrain 全量场景（cycle energy 实时计算）
冻结: 生成器全部冻结，仅训练 FineScorerV2（3.17M/3.88M 参数）
```

**V17 损失**（PDM 标签）：
$$\mathcal{L}_{\text{V17}} = \mathcal{L}_{\text{listnet}}(scores, proxy) + 0.2 \cdot \bar{E}_{\text{cyc}} + 0.1 \cdot \mathcal{L}_{\text{BCE}}(components)$$

**V21 损失**（Cycle Energy 自监督）：
$$\mathcal{L}_{\text{V21}} = \mathcal{L}_{\text{listnet}}(scores, -ce\underline{}norm) + 0.05 \cdot \mathcal{L}_{\text{GRPO}} - 0.01 \cdot \mathcal{H}(\pi)$$

V21 中 GRPO 只有 0.05 权重，只是"保底"防止 cycle energy 初始阶段出错。随着训练进行，listnet 主导学习。

---

## 6. 实验演进：V1→V21

### 6.1 完整版本演进

```
V1-V9  (7/9-7/11)   原型探索期 → 基础 JEPA 验证
V10    (7/11)        架构验证期 → ~0.80 (全局池化缺陷)
V11-V14(7/12-7/13)  baseline优化 → 0.82-0.84
V15    (7/14)        容量升级 → ~0.84 (修复 hidden)
V16    (7/15-7/16)   关键修复 → ~0.85 (移除全局池化)
─────────────────────────────────────────────
V17    (7/17-7/19)   里程碑 🏆 → 0.8839 (PDM 标签，但有数据泄露)
V18-V20(7/19-7/23)   瓶颈确认 → 0.875-0.882 (64 值瓶颈锁定)
─────────────────────────────────────────────
V21    (7/23-至今)   自监督突破 🔄 → 训练中 (cycle energy 替代 PDM)
```

### 6.2 V17 为何是里程碑

V17 的 0.8839 验证了三件事：
1. **JEPA 架构可行**：隐空间世界模型能有效编码驾驶场景
2. **推演式决策有效**：生成候选 → 评分 → 选最优，计算范式成立
3. **FineScorerV2 设计正确**：CrossAttn + SceneQuery 是关键

但也发现了两个问题：
1. **数据泄露**：训练在 navtest 上 → 0.8839 有水分
2. **64 值瓶颈**：V18-V20 无论怎么改架构，分数都在 0.88 徘徊

### 6.3 64 值瓶颈的数学理解

```python
# 每步训练中:
scores = model(batch)        # [B, 64] 评分器输出，连续值
proxy  = get_nn_proxy()      # [B, 64] 从 8 个 host_scores 映射
# proxy 本质上只有 8 个不同的值（被广播到 64 个候选）
# 评分器只能学这 8 个档次之间的排序
```

PDM 的打分精度就是评分器的理论上限。Cycle energy **没有这个限制**——每个候选有自己独立的 cycle energy 值，64 个候选 × B 场景 = 128 个独特的连续标签。

### 6.4 V21 当前状态

| 步骤 | 状态 | 详情 |
|------|------|------|
| Phase A | ✅ 完成 | 100 epochs, final loss=0.06 |
| Phase B | 🔄 训练中 | 8×A800 DDP, ~55min/epoch |
| 预计完成 | 7/26 | 50 epochs |
| 首次评估 | 7/25 E10 | 期待突破 0.89 |

---

## 7. 与业界路线对比

### 7.1 全景对比

![JEPA vs VLA](/images/jepa-drive/jepa-vs-vla.svg)

### 7.2 与 PDM-Scorer 对比

| 维度 | PDM-Scorer | JEPA-DRIVE |
|------|-----------|------------|
| 候选来源 | 8 个固定 SD 候选 | **65,536 个**场景自适应候选 |
| 评分方式 | PDM MLP 回归 | **Cycle Energy 自监督 + FineScorer** |
| 参数 | 依赖 PDM 精度 | ×64 值瓶颈 |
| 泛化 | 强于 VLA | **最强**（推演式） |

PDM-Scorer 是 NAVSIM 的官方 baseline，用 MLP 对 8 个 SD 候选打分。它的上限是 0.91-0.93。JEPA-DRIVE 如果能突破 0.89，就接近了 PDM-Scorer 的水平——但 JEPA-DRIVE 是**自监督**的，不需要 PDM 标签。

### 7.3 与 DriveVLA 对比

| 维度 | DriveVLA | JEPA-DRIVE |
|------|---------|------------|
| 预训练 | CLIP（互联网图文） | **SJEPA 自监督**（仅驾驶数据） |
| 参数量 | 7B+ | **3.88M** |
| 决策 | 单次 forward | **推演 + 选择** |
| OOD | 弱 | **强** |
| 延迟 | ~100ms | ~220ms（可优化至 50-100ms） |

### 7.4 优劣势总结

**JEPA-DRIVE 的核心优势**：
1. **极轻量**：3.88M 参数 vs VLA 的 7B+，差 1800 倍
2. **完全自监督**：不需要互联网数据、不需要人工标注、不需要 PDM
3. **OOD 鲁棒**：推演式决策天生处理未见场景
4. **可解释**：每个决策可以追溯到重建误差的来源
5. **诚实**：不会"幻觉"出不合理轨迹（重建误差会惩罚它们）

**核心劣势**：
1. **计算量大**：推理需多次 decoder forward
2. **依赖场景编码质量**：编码器漏了关键信息，cycle energy 就不准
3. **候选覆盖有限**：65,536 虽然大，仍有覆盖不到的行为
4. **工程复杂**：两阶段训练、DDP、候选采样

---

## 8. 个人思考

### 8.1 JEPA-DRIVE 最让我兴奋的点

不是 0.8839 这个分数，而是 **3.88M 参数 vs 7B 参数**这个对比。

3.88M 意味着什么？一个 ResNet-50 有 25M 参数，比 JEPA-DRIVE 大 6 倍。GPT-2 有 124M 参数，大 32 倍。而实现自动驾驶决策的核心模型，只需要 3.88M 参数——因为 JEPA 不学"怎么开车"，它学的是**场景和轨迹之间的一致性**。

一致性是一个比驾驶更基础、更简单的问题。模型不需要知道"现在应该加速还是减速"，只需要知道"这条轨迹是否符合场景"。后者更容易学习，更少参数，更好泛化。

### 8.2 64 值瓶颈的教训

V18-V20 花了大量时间改架构但没用：加大评分器、换损失函数、升级整个模型——分数全在 0.875-0.882 之间徘徊。

这告诉我们：**改架构之前先确认瓶颈在哪**。如果瓶颈在标签质量，改模型容量是徒劳的。这个教训适用于所有数据驱动项目。

### 8.3 与同期的 LinkVLA 对比

LinkVLA（CVPR 2026）走的是"共享离散词表"路线——把连续轨迹离散化为 token，与文本 token 共享 LLM embedding。JEPA-DRIVE 也用了离散词汇表（512 path × 128 vel = 65,536 组合），但动机不同：

- LinkVLA：为了对齐语言和动作的 embedding 空间
- JEPA-DRIVE：为了用组合生成替代记忆，用最小参数覆盖最大行为空间

两者都验证了**离散化是 VLA 系统的重要设计维度**，但 LinkVLA 把离散化当作"对齐工具"，JEPA-DRIVE 把离散化当作"组合工具箱"。

### 8.4 未来方向

如果 V21 验证了 cycle energy 能打破 64 值瓶颈，后续有几个自然的方向：

1. **端到端联合训练**：cycle energy 反向传播到生成器，真正的 JEPA 端到端自监督
2. **词汇表扩展**：1024×256 = 262k 候选，覆盖更细粒度的行为
3. **Cycle Energy 改进**：多步 cycle energy、条件 cycle energy
4. **联合 LinkVLA**：在 JEPA-DRIVE 的候选上应用 LinkVLA 的共享词表对齐

---

*📖 本文是对 JEPA-DRIVE 项目的技术总结与个人思考。V21 训练结果预计 7/26 完成，届时更新。*