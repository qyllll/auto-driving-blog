---
title: "个人思考｜JEPA-DRIVE：用联合嵌入预测架构做自监督驾驶决策"
date: 2026-07-24
draft: false
categories: ["个人思考"]
tags: ["🧠 JEPA", "🚗 自动驾驶", "🎯 自监督学习", "📐 世界模型", "⚡ 推演式决策"]
summary: "JEPA-DRIVE 是一个基于 Joint Embedding Predictive Architecture 的自监督驾驶决策系统。它用结构化隐空间世界模型编码驾驶场景，从 65,536 个离散行为词汇中组合生成候选轨迹，再用 cycle energy 重建误差作为自监督评分信号。V17 在 NAVSIM 上达到 0.8839 selected_pdm，V21 正在用纯自监督 cycle energy 突破 PDM 代理标签的 64 值瓶颈（8×A800 分布式训练中）。全模型仅 3.88M 参数。"
weight: 1
---

## 1. JEPA 架构核心概念

### 1.1 什么是 JEPA？

JEPA（Joint Embedding Predictive Architecture）由 Yann LeCun 团队提出，核心主张就一句话：

> **不要在像素空间做预测，要在抽象的隐空间做预测。**

传统自监督方法（MAE、扩散模型）在像素空间做预测——输入部分像素，预测完整像素。JEPA 换了一条路：输入部分信息的**隐空间表征**，在隐空间中预测缺失部分的表征。

![JEPA vs VLA 哲学对比](/images/jepa-drive/jepa-vs-vla.svg)

### 1.2 JEPA 的三个核心组件

JEPA 由三个组件构成，理解它们是理解 JEPA-DRIVE 的前提：

```
                    ┌──────────┐
                    │  输入 x   │
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │   编码器 f_θ(x)     │  ← 将输入压缩为隐空间表征
              │  (MLP / Transformer)│
              └──────────┬──────────┘
                         │
                    ┌────▼────┐
                    │ 隐空间  │
                    │ 表征 s  │
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │   预测器 g_φ(s)     │  ← 在隐空间中预测缺失部分
              │  (MLP / Transformer)│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   能量函数 E(s,pred)│  ← 衡量预测与实际的兼容性
              │  (MSE / Cosine)     │     低能量 = 预测准确
              └─────────────────────┘
```

| 组件 | 作用 | 在 JEPA-DRIVE 中的对应 |
|------|------|----------------------|
| **编码器 $f_\theta$** | 将输入映射到隐空间 | World Encoder（6 模态 MLP → 65 tokens × 96 dim） |
| **预测器 $g_\phi$** | 在隐空间中预测 | CrossAttn 解码器 + Trajectory Decoder |
| **能量函数 $E$** | 衡量兼容性 | Cycle Energy（MSE 重建误差） |

### 1.3 JEPA 的核心优势

**为什么隐空间预测比像素预测更好？**

| 维度 | 像素空间预测（MAE/扩散） | 隐空间预测（JEPA） |
|------|------------------------|-------------------|
| 维度 | 224×224×3 = 150,528 | **65 × 96 = 6,240**（24 倍压缩） |
| 冗余 | 90%+ 与任务无关（天空、纹理） | 全部与驾驶相关（物体、车道、速度） |
| 抽象级别 | 像素级，统计相关性 | **语义级，因果推理** |
| 计算量 | 巨大（需建模所有细节） | **小（3.88M 参数）** |
| 泛化 | 学统计规律，OOD 脆弱 | **学场景-行为一致性，OOD 鲁棒** |

**JEPA 不是学"怎么做"（$P(\text{action} \mid \text{scene})$），而是学"什么合理"（$\text{compatibility}(\text{action}, \text{scene})$）。** 后者更容易学习、更少参数、更好泛化。

---

## 2. JEPA-DRIVE：JEPA 在驾驶决策中的落地

JEPA-DRIVE 将 JEPA 的"隐空间预测 + 能量函数"思想具体化为驾驶决策系统：

- **场景** → 结构化编码为 65 tokens × 96 dim 隐空间表征
- **轨迹** → 从词汇表中组合生成 65,536 个候选
- **能量** → Cycle Energy（重建误差）衡量轨迹与场景的兼容性
- **决策** → 选能量最低（最兼容）的轨迹

### 2.1 问题形式化

给定场景特征 $S$ 和候选轨迹集合 $\mathcal{C}$，目标是：

$$ T^* = \arg\max_{T \in \mathcal{C}} f(T, S) $$

其中 $f$ 是评分函数。JEPA-DRIVE 的独特之处在于 $f$ 基于**自监督重建误差**，而非外部标签。

**NAVSIM 约束**：每个场景固定 8 个 SD 候选，评估 `selected_pdm`（选中的轨迹的 PDM 分数）、`oracle_pdm`（最高 PDM 分数）、`gap`（两者之差）。模型可以生成任意多候选训练，但评估只看 8 个 SD 候选中的选择质量。

---

## 3. 架构详解

### 3.1 全景

![JEPA-DRIVE 架构全景](/images/jepa-drive/jepa-architecture.svg)

整个架构分为两个阶段训练：**Phase A（生成器）** 和 **Phase B（评分器）**。

### 3.2 World Encoder：结构化场景编码

JEPA-DRIVE 接收的不是原始图像，而是感知模块输出的**结构化场景特征**——6 种模态的向量，每种模态用独立 MLP 编码到统一的 96 维隐空间：

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

| 模态 | 输入维度 | token 数 | 含义 |
|------|---------|---------|------|
| Object | 32 × 24 | 32 | 周围物体位置/类别/速度 |
| Lane | 16 × 24 | 16 | 车道线几何/类型 |
| Map | 4 × 24 | 4 | 地图元素（人行道/路肩等） |
| Route | 4 × 24 | 4 | 导航路径点 |
| Velocity | 1 × 24 | 1 | 自车速度 |
| Dynamics | 8 × 24 | 8 | 动态物体运动状态 |

**为什么用 MLP 而不是 Transformer？** 场景要素已经高度结构化，MLP 更轻量、更快，且保持模态分离——下游 CrossAttn 可以区分每个 token 的来源模态。

### 3.3 Vocabulary 系统（核心创新）

驾驶行为分解为两个独立维度的组合——**横向（path）** 和 **纵向（vel）**：

```python
self.path_queries = nn.Parameter(torch.randn(512, 96))  # 512 种横向行为
self.vel_queries  = nn.Parameter(torch.randn(128, 96))  # 128 种纵向行为
```

- 512 种路径：4 种基础走法 × 8 种幅度 × 16 种偏移
- 128 种速度：4 种基础趋势 × 4 种幅度 × 8 种微调

**组合爆炸**：640 个 query 向量 → 512 × 128 = **65,536 种行为**。这是 3.88M 参数的秘密——用组合代替记忆。

候选生成流程（均带场景 CrossAttn）：

1. Path Queries [512,96] → CrossAttn(场景) → Path Features
2. Vel Queries [128,96] → CrossAttn(场景) → Vel Features
3. Cartesian Product → 65,536 组合 → MLP 解码 → 轨迹

### 3.4 FineScorer V2（评分器）

Phase B 唯一训练的部分：

```python
class FineScorerV2(nn.Module):
    # 输入: action_lat [B,N,96] + attended [B,N,96] + traj_lat [B,N,96]
    # ① TrajFusion: concat → MLP(288→1024→96)
    # ② CrossAttn(融合特征, world_tokens) → 场景感知
    # ③ SceneQuery: 跨候选全局上下文
    # ④ ScoreHead: MLP(192→1024→1024→1) → 分数
    # ⑤ ComponentHead: MLP(96→512→6) → 6 维组件
```

关键修复：V10 用全局 mean pool（所有候选共享场景），V16 改 CrossAttn（每个候选独立查询场景）→ 0.84 → 0.8839 的关键突破。

---

## 4. Cycle Energy：自监督评分信号

### 4.1 思想

Cycle energy 来源于 JEPA 的能量基础模型思想：**给定场景 $S$ 和轨迹 $T$，能量 $E(T,S)$ 越低，表示 $T$ 与 $S$ 越兼容。**

在 JEPA-DRIVE 中，能量函数具体化为**重建误差**：轨迹的后半段能否被前半段 + 场景准确预测。

![Cycle Energy 示意图](/images/jepa-drive/cycle-energy.svg)

### 4.2 计算流程

轨迹 12 帧（3 秒）切成两半，用前 6 帧 + 场景预测后 6 帧：

1. `action_lat = MLP(prefix)` → 轨迹前半段编码 [96]
2. `traj_lat = MLP(full_traj)` → 轨迹全段编码 [96]
3. `attended = CrossAttn(action_lat, world_tokens)` → 场景-轨迹交互 [96]
4. `pred_suffix = Decoder(concat[action_lat, attended, traj_lat])` → 预测后半段
5. `cycle_energy = MSE(pred_suffix, real_suffix)` → 重建误差

### 4.3 相比 PDM 标签的优势

| 维度 | PDM 标签（64 值瓶颈） | Cycle Energy（自监督） |
|------|---------------------|---------------------|
| 值域 | 64 个离散值 | **连续值**，任意精度 |
| 来源 | 外部 PDM 打分器 | **自监督**，从数据中产生 |
| 数据覆盖 | 仅 9k / 85k 场景 | **全部 85k 场景** |
| 语义 | 单一分数 | **可分解**（位置/速度/方向） |

这就是 V21 的核心问题：**cycle energy 的连续值能否突破 PDM 64 值瓶颈？**

---

## 5. 训练方法

### 5.1 Phase A：生成器

```yaml
Epochs: 100 | Batch: 2 | LR: 1e-4 | GPU: 1×L20 | 数据: 7119 有标注场景
```

损失：$\mathcal{L}_{\text{im-s}} + \mathcal{L}_{\text{im-w}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{dst}}\mathcal{L}_{\text{dst}}$

| 分量 | 权重 | 作用 |
|------|------|------|
| 图像重建 | 1.0 | 候选在场景中的特征重建 |
| Cycle 一致性 | 1.0 | 编解码自一致性 |
| 多样性 | 0.01 | 鼓励候选多样化 |
| 距离损失 | 0.10 | 候选接近数据集轨迹 |

收敛：E1 loss=1.30 → E100 loss=0.06（~17 小时）

### 5.2 Phase B：评分器

```yaml
Epochs: 50 | Batch: 8×8GPUs=64 | LR: 1e-4 | GPU: 8×A800-80GB | 数据: 85109 全量场景
冻结: 生成器冻结，仅训练 FineScorerV2（3.17M / 3.88M 参数）
```

V17（PDM 标签）：$\mathcal{L}_{\text{listnet}}(scores, proxy) + 0.2 \cdot \bar{E}_{\text{cyc}} + 0.1 \cdot \mathcal{L}_{\text{BCE}}$

V21（Cycle Energy）：$\mathcal{L}_{\text{listnet}}(scores, -ce) + 0.05 \cdot \mathcal{L}_{\text{GRPO}} - 0.01 \cdot \mathcal{H}(\pi)$

GRPO 仅 0.05 权重作为"保底"，listnet 主导学习。

---

## 6. 实验演进：V1→V21

### 版本时间线

| 版本 | 时间 | 关键变化 | 分数 |
|------|------|---------|------|
| V1-V9 | 7/9-7/11 | 原型探索，基础 JEPA 验证 | — |
| V10 | 7/11 | 评分器定型，但有全局池化缺陷 | ~0.80 |
| V11-V14 | 7/12-7/13 | Baseline 优化 | 0.82-0.84 |
| V15 | 7/14 | 评分器容量修复（hidden 192→512） | ~0.84 |
| V16 | 7/15-7/16 | **关键修复**：移除全局池化，改 CrossAttn | ~0.85 |
| **V17** | **7/17-7/19** | **架构定型**：FineScorerV2 + CrossAttn + 1024 hidden | **0.8839** |
| V18-V20 | 7/19-7/23 | 各种改进尝试 → 确认 64 值瓶颈 | 0.875-0.882 |
| **V21** | **7/23-至今** | **Cycle Energy 自监督替代 PDM** | **训练中** |

### V17 里程碑

V17 的 0.8839 验证了 JEPA 架构可行、推演式决策有效、FineScorerV2 设计正确。但也暴露了两个问题：数据泄露（训练在 navtest）和 64 值瓶颈。

### 64 值瓶颈

V18-V20 无论怎么改架构——加大评分器、换损失函数、升级模型——分数全在 0.875-0.882 徘徊。

```python
# 每步训练中:
scores = model(batch)   # [B, 64] 评分器输出，连续值
proxy  = get_nn_proxy() # [B, 64] 只有 8 种不同的值（被广播到 64 个候选）
# 评分器只能学这 8 个档次之间的排序
```

PDM 的打分精度就是评分器的理论上限。Cycle energy **没有这个限制**——每个候选有自己独立的连续值标签。

### V21 状态

| 步骤 | 状态 | 详情 |
|------|------|------|
| Phase A | ✅ 完成 | 100 epochs, loss=0.06 |
| Phase B | 🔄 训练中 | 8×A800 DDP, ~55min/epoch |
| 预计完成 | 7/26 | 50 epochs |

---

## 7. 与业界路线对比

### 全景：JEPA-DRIVE vs VLA

| 维度 | VLA（DriveVLA/UniAD） | JEPA-DRIVE |
|------|----------------------|------------|
| 核心思想 | 模仿学习 $P(T \mid O)$ | 推演式选择 $\arg\max_T P(O \mid T)$ |
| 训练数据 | 互联网图文 + 驾驶标注 | **仅驾驶场景，完全自监督** |
| 参数 | 7B+ | **3.88M**（1800 倍差距） |
| 决策 | 单次 forward 生成 | 65,536 候选 → 评分 → 选最优 |
| OOD 处理 | 弱（统计相关性） | **强**（场景一致性推理） |
| 可解释性 | 黑盒 | 重建误差可视化 |

### 优劣势总结

**优势**：
- 3.88M 参数 vs VLA 的 7B+，差 1800 倍
- 完全自监督，无需互联网数据、人工标注、PDM 标签
- 推演式决策天生处理 OOD 场景
- 每个决策可追溯到重建误差的来源

**劣势**：
- 推理需多次 decoder forward，延迟高于 VLA
- 依赖编码器质量——漏了关键信息，cycle energy 就不准
- 65,536 候选仍可能有覆盖不到的行为
- 两阶段训练、DDP、候选采样等工程复杂

---

## 8. 个人思考

### 3.88M 意味着什么

一个 ResNet-50 有 25M 参数，比 JEPA-DRIVE 大 6 倍。GPT-2 有 124M 参数，大 32 倍。

实现自动驾驶决策的核心模型只需要 3.88M 参数——因为 JEPA 不学"怎么开车"，它学的是**场景和轨迹之间的一致性**。一致性比驾驶更基础、更简单：模型不需要知道"现在该加速还是减速"，只需要知道"这条轨迹是否符合场景"。

### 64 值瓶颈的教训

V18-V20 花了大量时间改架构但没用——因为瓶颈不在模型容量，在标签质量。**改架构之前先确认瓶颈在哪。**

### 与 LinkVLA 的对比

LinkVLA（CVPR 2026）也用离散词汇表（共享词表），但动机不同：

- LinkVLA：为了对齐语言和动作的 embedding 空间
- JEPA-DRIVE：为了用组合生成替代记忆，最小参数覆盖最大行为空间

两者都验证了**离散化是 VLA 系统的重要设计维度**，但 LinkVLA 把离散化当作"对齐工具"，JEPA-DRIVE 把离散化当作"组合工具箱"。

### 未来方向

1. 端到端联合训练：cycle energy 反向传播到生成器
2. 词汇表扩展：1024×256 = 262k 候选
3. 条件 / 多步 cycle energy
4. 联合 LinkVLA 共享词表对齐

---

*📖 V21 训练结果预计 7/26 完成，届时更新。*
