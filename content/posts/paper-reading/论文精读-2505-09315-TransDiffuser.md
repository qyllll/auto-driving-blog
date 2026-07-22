---
title: "论文精读｜TransDiffuser：去相关多模态表示驱动的扩散式轨迹生成"
date: 2026-07-22
draft: false
categories: ["论文精读"]
tags: ["🎨 扩散模型", "🚗 自动驾驶", "📐 轨迹规划", "🔗 端到端", "🧠 表示学习"]
summary: "TransDiffuser 由 LiAuto 提出，用编码器-解码器架构将扩散模型引入端到端轨迹规划。核心创新是多模态表示去相关（Multi-modal Representation Decorrelation）机制——在去噪解码器和动作解码器之间插入一个轻量正则化项，强制不同特征维度的相关性矩阵趋近对角矩阵。无需任何预定义锚点轨迹或场景先验，以 ResNet-34 + LiDAR 配置在 NAVSIM 上达到 PDMS 94.85。"
weight: 7
---

![TransDiffuser 整体架构：编码器-解码器扩散模型](/images/transdiffuser/figure2_architecture.jpg)

![TransDiffuser 架构与去相关机制示意图](/images/transdiffuser/decorrelation_mechanism.svg)

## 📄 论文信息

- **标题**：*TransDiffuser: Diverse Trajectory Generation with Decorrelated Multi-modal Representation for End-to-end Autonomous Driving*
- **arXiv**：[2505.09315](https://arxiv.org/abs/2505.09315)（2025年5月，2025年9月更新 v2）
- **作者**：Xuefeng Jiang（中科院计算所），Yuan Ma（清华），Pengxiang Li（清华），Kun Zhan（理想汽车）等
- **团队**：LiAuto + 中科院计算所 + 清华大学
- **一句话总结**：用 **多模态表示去相关** 解决扩散轨迹生成中的 **模式坍缩（mode collapse）**，在不依赖任何锚点轨迹/场景先验的前提下实现 SOTA 规划性能。

---

## 一、动机：扩散模型轨迹生成的模式坍缩困境

### 1.1 端到端规划的范式迁移

端到端自动驾驶规划经历了三个阶段：

| 阶段 | 范式 | 代表工作 | 特点 |
|------|------|---------|------|
| **确定性回归** | 自回归（AR）预测单条轨迹 | Transfuser、UniAD | 简单直接，但无法表达多模态 |
| **打分式（Scoring）** | 从固定轨迹词表中选最优 | VADv2、Hydra-MDP | 依赖预定义锚点，泛化受限 |
| **生成式（Generative）** | 扩散/流匹配采样多候选 | DiffusionDrive、GoalFlow | 多模态好，但模式坍缩严重 |

生成式模型的核心优势本应是**多模态**——同一场景下有多条合理轨迹（让行/变道/缓行），扩散模型的随机采样应该能覆盖这些模式。但实际中，**不同随机噪声经过去噪后，往往收敛到相似的轨迹分布**，这就是模式坍缩（mode collapse）。

### 1.2 现有方案的局限性

| 方法 | 解决模式坍缩的手段 | 局限 |
|------|------------------|------|
| **DiffusionDrive** | 用锚点高斯分布替代标准高斯噪声（截断扩散） | 需预定义轨迹词表，引入归纳偏置 |
| **GoalFlow** | 用目标点稠密词表约束生成过程 | 需预计算场景先验，对未见场景泛化难 |
| **TrajHF** | 偏好优化 + 拒绝采样 | 候选轨迹数量大（100条），推理效率低 |

这些方法都引入了**额外的先验信息**——锚点轨迹或场景先验。虽然它们确实缓解了模式坍缩，但也带来了归纳偏置：在未见过的场景中，预定义的锚点或目标点可能不适用。

### 1.3 TransDiffuser 的核心洞察

TransDiffuser 的出发点是：**模式坍缩的根本原因可能不在于扩散模型的采样过程，而在于多模态条件特征没有得到充分利用**。

![端到端自动驾驶规划的主要方法对比](/images/transdiffuser/figure1_comparison.jpg)

具体来说，去噪解码器的条件输入包含来自不同模态的多个特征（图像特征、LiDAR 特征、BEV 特征、历史轨迹嵌入、当前状态嵌入）。当这些特征的**维度之间存在冗余或耦合**时，条件信号的信息量被压缩，导致不同的随机噪声在去噪过程中倾向于收敛到相同的输出。TransDiffuser 的解法就是**在特征进入动作解码器之前，强制让不同特征维度去相关**，使条件信号更丰富地覆盖潜在空间。

---

## 二、TransDiffuser 架构详解

### 2.1 整体架构：编码器-解码器扩散模型

TransDiffuser 是一个编码器-解码器架构，包含三个核心组件：

1. **Scene Encoder**（冻结）：基于 Transfuser 骨干，处理相机 + LiDAR 多模态感知
2. **Motion Encoder**：编码历史轨迹和当前自车状态
3. **Denoising Decoder**：DDPM 扩散解码器，以编码特征为条件生成轨迹

#### 问题形式化

模型输出 8 个 waypoint 覆盖未来 4 秒（8Hz），轨迹表示为：

$$\mathbf{x} = \{s_1, s_2, \dots, s_T\}, \quad T=8$$

其中 $s_\tau$ 是自车中心坐标系下的 waypoint 位置。每个 waypoint 通过**动作投影**（action space projection）连接相邻点以缓解异方差性：

$$\hat{x}_\tau = s_\tau - s_{\tau-1}, \quad \hat{x}_1 = s_1$$

这样，模型预测的是**动作序列**（相邻帧的位移），轨迹可以通过动作累积得到。这个变换是可逆的——动作到轨迹的映射是简单累加。

### 2.2 Scene Encoder：多模态融合

Scene Encoder 使用被 Nav-train 预训练权重冻结的 **Transfuser 骨干**。它处理两种传感器输入：

| 传感器 | 处理方式 | 输出特征 |
|--------|---------|---------|
| 前视多视角相机 | CNN backbone | $F_{img}$ |
| LiDAR 点云 | CNN backbone | $F_{LiDAR}$ |
| 两者融合 | **多阶段 Transformer 跨模态注意力** | $F_{bev}$（BEV 特征） |

**多阶段融合**是这里的关键设计：图像和 LiDAR 分支在每个阶段通过 Transformer block 进行跨模态交互，而不是只在最后阶段做一次融合。这让多模态信息在多个尺度上相互增强。

论文特别强调**Scene Encoder 被冻结**。消融实验（Table IV）显示，当 Scene Encoder 参与全量训练时，PDMS 反而下降（94.9 → 93.5）。这可能有几个原因：
- 预训练的感知特征已经足够好
- 保持感知特征不变能防止规划任务"扭曲"感知表示
- 冻结感知层减少了可训练参数量（62.8M / 251M ≈ 25%）

### 2.3 Motion Encoder：运动上下文

Motion Encoder 由两个独立的 MLP 组成，分别处理两种运动信息：

| 编码器 | 输入 | 输出 |
|--------|------|------|
| **Action Encoder** | 历史自车轨迹（多帧 waypoint） | $Emb_{action}$ |
| **Ego Status Encoder** | 当前自车状态（速度、加速度等） | $Emb_{ego}$ |

编码后的特征组为：

$$feat = \{F_{bev}, F_{img}, F_{LiDAR}, Emb_{action}, Emb_{ego}\}$$

这 5 个特征通过去噪解码器中的**多头交叉注意力**顺序融合。

### 2.4 Denoising Decoder：DDPM 轨迹生成

去噪解码器采用 **Denoising Diffusion Probabilistic Model（DDPM）** 框架。

#### 前向过程（加噪）

在前向过程中，噪声被逐步添加到真实的动作标签 $\mathbf{x}_0$：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

#### 反向过程（去噪）

模型学习去噪的反向过程，从高斯噪声 $\mathbf{x}_T$ 逐步恢复出有效的动作 $\mathbf{x}_0$：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t, feat)\right) + \sigma_t \mathbf{z}$$

其中 $\epsilon_\theta$ 是去噪解码器网络，以当前噪声动作 $\mathbf{x}_t$、去噪步数 $t$ 和特征组 $feat$ 为条件预测噪声。

| 参数 | 含义 | 配置 |
|------|------|------|
| $T$ | 总去噪步数 | 10 步（训练和推理一致） |
| $\mathbf{x}_0$ | 原始动作序列 | 8 步 waypoint 的 Δ位移 |
| $\epsilon$ | 预测的噪声 | 与 GT 噪声做 MSE 损失 |

#### 推理采样

推理时，从标准高斯噪声出发，经过 T=10 步去噪，生成动作序列，再累加得到未来 4s 的 8 个 waypoint。模型生成 **N=30** 条候选轨迹，然后通过拒绝采样过滤（基于运动学可行性），最终选择 PDMS 评分最高的轨迹。

对比其他方法生成的候选数量：

| 方法 | 候选轨迹数 | 
|------|-----------|
| GoalFlow | 128 ~ 256 条 |
| TrajHF | 100 条 |
| **TransDiffuser** | **30 条**（更少但仍保持有效） |
| DiffusionDrive | 20 条（但依赖锚点先验） |

TransDiffuser 用更少的候选轨迹就达到了 SOTA，说明生成的轨迹质量更高、探索效率更好。

---

## 三、核心创新：多模态表示去相关

### 3.1 问题定义

TransDiffuser 的核心洞察是：**条件特征 $feat$ 的利用效率是模式坍缩的关键瓶颈**。

![多模态表示去相关过程](/images/transdiffuser/figure3_decorrelation.jpg)

当特征组 $feat = \{F_{bev}, F_{img}, F_{LiDAR}, Emb_{action}, Emb_{ego}\}$ 被多头交叉注意力融合后，融合表示 $\mathbf{M}$ 的不同维度之间存在相关性。如果这种相关性过高，表示空间的**有效维度**会远低于原始维度，导致条件信号的信息量下降。在生成模型中，这意味着不同的噪声输入得到相似的轨迹输出。

### 3.2 去相关正则化

TransDiffuser 在每个训练 batch 中，对融合后的多模态表示矩阵 $\mathbf{M} \in \mathbb{R}^{B \times d}$ 施加一个去相关正则化项：

```python
# Algorithm 2: Computation of L_reg (简化)
def decorrelation_loss(M):
    # Step 1: Normalize each dimension
    mean = M.mean(dim=0, keepdim=True)
    std  = M.std(dim=0, keepdim=True)
    M_norm = (M - mean) / sqrt(std**2 + 1e-8)

    # Step 2: Compute correlation matrix (d × d)
    corr = M_norm.T @ M_norm

    # Step 3: Remove diagonal elements (we only penalize off-diagonal)
    corr_off = corr - diag(diag(corr))

    # Step 4: Mean squared off-diagonal elements
    return mean(corr_off**2) / batch_size
```

最终损失函数结合了扩散损失的监督项和这个正则化项：

$$\mathcal{L} = \mathcal{L}_{diff} + \beta \cdot \mathcal{L}_{reg}$$

其中 $\beta = 0.02$ 在整个训练过程中固定。

### 3.3 去相关为什么有效

论文从**奇异值分解**的角度分析了去相关的作用（Figure 3(d) 展示了前 100 个最大奇异值的分布）：

| 机制 | 去相关前 | 去相关后 |
|------|---------|---------|
| 相关矩阵分布 | 对角线集中 + 大量非零非对角元 | **趋近对角矩阵** |
| 奇异值分布 | 能量集中在前几个主成分 | **能量分布更均匀** |
| 有效表示维度 | 低 | **高** |
| 生成轨迹多样性 | 低（模式坍缩） | **高** |

去相关让表示空间中的每个维度携带更多**独立**信息，从而使条件信号能够更精细地指导去噪过程，让不同的噪声输入收敛到不同的轨迹。

### 3.4 与对比学习等方法的对比

| 方法 | 实现方式 | batch 需求 | 计算开销 |
|------|---------|-----------|---------|
| 对比学习（SimCLR 等） | 正负样本对拉近/推远 | 大 batch（4096+） | 高 |
| **去相关正则化** | 强制相关矩阵对角化 | **任意 batch** | **< 1‰ FLOPs** |

与对比学习不同，去相关正则化不需要构造正负样本对，也没有大 batch 需求——它在每个 batch 内自监督地优化表示分布。论文指出，其计算量仅占前向传播的不到千分之一（< 1‰ FLOPs），几乎可以忽略。

---

## 四、模式坍缩的量化度量

为了定量评估模式坍缩程度，TransDiffuser 引入了一个**多样性分数** $D$：

$$D = 1 - \frac{1}{N} \sum_{i=1}^{N} \frac{\text{Area}(\tau_i \cap \bigcup_{j=1}^{N} \tau_j)}{\text{Area}(\tau_i \cup \bigcup_{j=1}^{N} \tau_j)}$$

其中 $\tau_i$ 是第 $i$ 条去噪轨迹，$N$ 是采样总数。$D$ 衡量各条轨迹与全部轨迹并集之间的 mIoU——$D=100$ 表示完全多样（所有轨迹都互不相同），$D=0$ 表示完全坍缩（所有轨迹重合）。这个指标最先在 DiffusionDrive 中被提出，TransDiffuser 沿用了它。

---

## 五、实验与结果

### 5.1 NAVSIM 性能对比

| 方法 | 范式 | LiDAR | Anchor/先验 | PDMS ↑ |
|------|------|:----:|:----------:|:------:|
| Transfuser | AR | ✓ | — | 83.9 |
| UniAD | AR | — | — | 83.4 |
| PARA-Drive | AR | ✓ | — | 84.0 |
| VADv2 | Scoring | — | ✓ | 80.9 |
| Hydra-MDP | Scoring | ✓ | ✓ | 91.3 |
| DiffusionDrive | Diffusion | ✓ | ✓ (锚点) | 88.1 |
| GoalFlow | Diffusion | ✓ | ✓ (场景先验) | 90.3 |
| TrajHF | Diffusion | ✓ | — | 94.0 |
| **TransDiffuser** | Diffusion | ✓ | **—** | **94.9** |

**关键发现**：TransDiffuser 是第一个**完全不依赖任何锚点轨迹或场景先验**仍能达到 SOTA 的扩散式规划方法。与 TrajHF（同样无先验）相比，PDMS 提升约 0.9 分，且 TrajHF 使用了更大的 ViT 图像编码器而 TransDiffuser 仅用 ResNet-34。

### 5.2 多样性提升

| 方法 | 候选数 | 多样性 D ↑ |
|------|:-----:|:---------:|
| DiffusionDrive | 20 | 74（受益于锚点初始化） |
| **TransDiffuser (w/o decorr)** | 30 | 66 |
| **TransDiffuser (w/ decorr)** | 30 | **70** |
| **TransDiffuser** | 15 | 63 |
| **TransDiffuser** | 10 | 56 |

去相关正则化将多样性从 66 提升至 70。虽然仍比 DiffusionDrive 的 74 略低，但注意 DiffusionDrive 的多样性来自于**预定义的锚点分布**（先验信息），而 TransDiffuser 单纯通过优化表示就能接近这个水平。

### 5.3 消融实验

#### 超参敏感性

| 参数 | 配置范围 | 最优 | 结论 |
|------|---------|:----:|------|
| **去噪步数 T** | 5 / 10 / 20 | **10** | 更多步提升多样性，但 PDMS 饱和 |
| **Batch size B** | 32 / 64 / 128 | **64** | 太小去相关不稳定，太大无额外收益 |
| **权重 β** | 0 / 0.02 / 0.05 / 0.1 | **0.02** | β=0 多样性最低（66→70），太大反而下降 |

#### 组件消融

| 变体 | PDM Score | 多样性 D | 说明 |
|------|:---------:|:--------:|------|
| **Baseline (30 candidates)** | **94.9** | 70 | 完整方法 |
| w/o decorrelation | 94.3 | 66 | β=0，PDMS 和多样性均下降 |
| Inner decorrelation | 93.5 | 69 | 去相关放在去噪循环内部 |
| Fully-trained | 93.5 | 68 | Scene Encoder 参与训练 |

**关键发现 1：** 去相关放在去噪循环 "内部"（每个去噪步都做）的效果不如放在外部（去噪完整结束后、动作解码前）。论文分析认为：内部去相关可能干扰了去噪过程的稳定性，而外部去相关专注于优化**最终表示的质量**。

**关键发现 2：** 冻结 Scene Encoder 反而不损性能。这在工程实践上是好消息——预训练的感知特征足够通用，无需在规划任务上微调即可保持高质量。

### 5.4 跨方法泛化

论文将去相关模块应用到 **Transfuser**（自回归模型）上，发现 PDMS 从 78.0 提升到了 78.8。这说明去相关**不限于扩散模型**——它对任何使用多模态条件特征的规划方法都有帮助，具有通用性。

### 5.5 定性分析

![单模态轨迹可视化对比](/images/transdiffuser/figure4_single_traj.png)

![多模态多样化轨迹可视化](/images/transdiffuser/figure5_multi_traj.jpg)

在简单场景下，TransDiffuser 能给出比 Transfuser 更**积极**的驾驶方案（更早加速、更平滑的路径）。在复杂场景下，其生成的候选轨迹能覆盖比 GoalFlow **更广的可行驾驶区域**——去相关让条件表示更丰富，探索了更多可能性。

---

## 六、总结与启发

### 贡献总结

1. **编码器-解码器扩散规划架构**：用冻结的 Transfuser 骨干编码场景，用 DDPM 解码器生成多候选轨迹
2. **多模态表示去相关机制**：首个在端到端驾驶规划中利用表示去相关缓解模式坍缩的工作，无需任何锚点或先验
3. **SOTA 性能**：PDMS 94.85（ResNet-34 + LiDAR），超越了同期依赖锚点/先验的方法
4. **通用性**：去相关模块可插拔，对自回归模型同样有效

### 与同类方法的深度对比

| 维度 | DiffusionDrive | GoalFlow | TrajHF | **TransDiffuser** |
|------|:------------:|:--------:|:-----:|:---------------:|
| 缓解模式坍缩 | 锚点高斯初始化 | 目标点词表约束 | 偏好优化 | **表示去相关** |
| 依赖先验 | ✓ (锚点) | ✓ (场景先验) | — | **—** |
| 可训练参数量 | — | — | — | **62.8M / 251M** |
| 候选数 | 20 | 128~256 | 100 | **30** |
| 训练时长 | — | — | — | **2h @4×H20** |
| PDMS (ResNet-34) | 88.1 | 90.3 | 94.0 | **94.9** |

### 个人思考

TransDiffuser 最打动我的点在于它对"模式坍缩"这个问题的归因视角。之前的工作都把模式坍缩归咎于"扩散模型本身容易坍缩"，因此解法要么是约束初始分布（DiffusionDrive 的截断扩散），要么是约束生成路径（GoalFlow 的目标点限制）。**TransDiffuser 的攻击点完全不同——它认为模式坍缩的核心原因是"条件信息不够丰富"，而不是"生成过程不够受控"**。这个视角转变在方法论上非常优雅：不需要动采样过程，只需要优化表示质量。

从工程角度看，去相关正则化的极低开销（< 1‰ FLOPs）和即插即用特性使其非常适合作为**任何多模态融合方法的标配组件**——无论是扩散模型还是自回归模型。论文将其移植到 Transfuser 上验证了这一点。

另外一个值得注意的工程设计是**冻结 Scene Encoder**。虽然论文提到"full training 反而降分"可能有个案因素，但这至少在实践上确认了：感知预训练的特征在规划任务上已是强特征，过度适配可能对多任务泛化反而有害。这对于 VLA（Vision-Language-Action）模型的训练策略也有参考价值——**不是所有模块都需要随任务微调**。

说两个不足：第一，多样性 D=70 相比于 DiffusionDrive 的 74 仍有差距，说明去相关虽然有效但还未完全消除与锚点方法的多样性鸿沟；第二，论文仅在 NAVSIM 开环评测下验证，缺乏闭环仿真测试和多场景泛化分析，实际的驾驶安全性还需更多验证。

---

*📖 论文精读系列。本文基于 arXiv:2505.09315v2 撰写。*
