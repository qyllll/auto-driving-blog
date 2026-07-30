---
title: "损失函数完全指南：从 L1 到 GRPO——端到端自动驾驶中的 Loss 设计哲学"
date: 2026-07-30
draft: false
categories: ["技术指南"]
tags: ["📉 Loss", "🔧 工程实践", "🧠 深度学习", "🚗 自动驾驶"]
summary: "读论文时被各种 Loss 搞晕了？L1、L2、Focal、KL 散度、碰撞 Loss、GRPO……为什么同一个任务在不同文章里用的 Loss 不一样？这篇文章帮你把端到端自动驾驶里的所有损失函数理清楚——每种 Loss 的数学形式、直觉理解、适用场景，以及它们之间的演进关系。"
weight: 1
math: true
---

## 一个困惑

看端到端自动驾驶的论文时，你会发现同一个任务在不同的文章里用了完全不同的 Loss：

| 任务 | UniAD | VADv2 | DiffusionDrive |
|------|-------|-------|---------------|
| 轨迹规划 | L2 回归 + 碰撞 Loss | KL 散度（分布匹配） | 扩散去噪 Loss + 碰撞/边界/舒适 |
| 检测 | Focal Loss | Focal Loss | — |
| 占用预测 | 二值交叉熵 + Dice | — | — |

这还不是最让人晕的——同一篇文章内部，Loss 也在"打架"：UniAD 的 Planner 用 L2 回归，但 MotionFormer 用 L1 回归；VADv2 的分布 Loss 对 4096 条轨迹做分类，而 SparseDrive 对锚点做分类 + 残差回归。

这些 Loss 背后遵循什么设计逻辑？什么场景该用 L1、什么时候用 L2、什么时候该抛弃回归转向分布匹配？这篇文章试图提供一个统一的答案。

---

## 一、损失函数的本质

在端到端自动驾驶中，损失函数的核心作用是**把"什么样的驾驶是好的"这个模糊目标，翻译成神经网络能理解的数学信号**。

这个翻译的质量，直接决定了模型能学到什么：

```
模糊目标："把车开好"
    ↓ 翻译成 Loss
L2 回归 → 模型学"轨迹和人类尽可能像"
碰撞 Loss → 模型学"轨迹要安全"
KL 散度 → 模型学"动作分布要和人类分布一致"
```

所以理解 Loss，本质上是在理解**我们到底想让模型学什么**。

---

## 二、回归损失：找"最优答案"

### 2.1 L1 Loss / MAE

\[
\mathcal{L}_1 = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
\]

- **直觉**：预测值和真值的绝对差距
- **梯度**：处处为 ±1（常数），大误差处不会爆炸
- **谁在用**：UniAD MotionFormer 的轨迹回归、MapFormer 的点集回归

### 2.2 L2 Loss / MSE

\[
\mathcal{L}_2 = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
\]

- **直觉**：大误差平方放大，小误差忽略
- **梯度**：正比于误差（|y−ŷ|），收敛快但对异常值敏感
- **谁在用**：UniAD Planner 的轨迹回归

### 2.3 Smooth L1

```
ℒ_smooth = 0.5 * x²       if |x| < 1
ℒ_smooth = |x| - 0.5     if |x| ≥ 1
```

- **直觉**：小误差类 L2（平滑收敛），大误差类 L1（对异常值鲁棒）
- **谁在用**：Faster R-CNN 框回归，许多现代检测器（Sparse4D 等）

### 2.4 对比：什么时候用什么？

![L1 vs L2 vs Smooth L1 对比](/images/loss-guide/l1-l2-comparison.svg)

**核心规律**：

| 场景 | 推荐 Loss | 原因 |
|------|----------|------|
| 训练数据干净，需要快速收敛 | L2 / MSE | 大误差梯度大，收敛快 |
| 数据有离群值 / 标注噪声 | L1 | 对异常值不敏感 |
| 框回归（检测） | Smooth L1 | 兼顾小误差平滑 + 大误差鲁棒 |
| 轨迹回归（规划） | L1 | 轨迹标注本身噪声大，用 L2 会被离群轨迹带偏 |

为什么 UniAD 的 MotionFormer 用 L1 而 Planner 用 L2？这是**目标不同**导致的合理差异。Motion 预测未来轨迹——3 秒后的不确定性天然大，标注本身就是近似，用 L1 不让大误差主导梯度。而 Planner 输出自车轨迹——这是**执行信号**，需要精确控制，用 L2 让大误差得到更多关注。

---

## 三、分类损失：选"哪个答案"

### 3.1 交叉熵

\[
\mathcal{L}_{CE} = -\sum_{c=1}^C y_c \log(\hat{y}_c)
\]

- **直觉**：模型给正确答案分配的概率越高，Loss 越低
- **谁在用**：几乎所有分类任务——目标检测类别、交通灯状态、驾驶意图

### 3.2 Focal Loss

\[
\mathcal{L}_{focal} = -\alpha (1 - \hat{y}_c)^\gamma \log(\hat{y}_c)
\]

- **直觉**：在交叉熵基础上加了一个调制因子 \((1-\hat{y}_c)^\gamma\)——模型已经分得很好的样本（概率高）权重被压低，难样本（概率低）权重保持
- **谁在用**：UniAD 检测 + MapFormer、几乎所有自动驾驶检测头（正负样本极度不平衡）

### 3.3 BCE / sigmoid

\[
\mathcal{L}_{BCE} = -y \log(\sigma(s)) - (1-y) \log(1-\sigma(s))
\]

- **直觉**：每个类独立判断是与否
- **谁在用**：VADv2 的规划词表评分、OccFormer 的占用预测（每个像素独立二分类）

### 3.4 交叉熵 vs Focal Loss vs BCE

| 损失 | 适用场景 | 特点 |
|------|---------|------|
| 交叉熵 | 多分类（N选1） | 标准分类 Loss |
| Focal Loss | 类别极度不平衡 | 压住易分样本，聚焦难样本 |
| BCE | 多标签分类（可同时选多个） | 各维度独立，互不排斥 |

这是理解 VADv2 设计的关键：**它用 sigmoid 而不是 softmax**，不是因为技术约束，而是因为驾驶多模态性的哲学判断——多个合理动作不是互斥的，"直行"概率高不意味着"变道"概率就应低。

---

## 四、分布损失：学"正确答案的分布"

当回归损失和分类损失都面临根本缺陷时——**回归收敛到平均，分类被互斥性约束**——分布损失提供了一条新路。

### 4.1 KL 散度

\[
D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
\]

- **直觉**：衡量两个分布之间的距离
- **谁在用**：VADv2 的分布 Loss、知识蒸馏

### 4.2 从交叉熵到 KL 散度的视角切换

这里有一个重要的认知跃迁：

```
回归范式：
    场景 → 一条最优轨迹 → 回归 Loss → 模型学"平均" → 多模态场景下崩坏

分类范式：
    场景 → K 条候选轨迹 → 选一个 → 分类 Loss → 模型学"选哪条"

分布范式：
    场景 → K 条候选轨迹 → 每条独立打分 → KL Loss → 模型学"每条该得多少分"
```

VADv2 做的就是这个跃迁。它的 Loss 被设计成三条腿：

![VADv2 三重损失](/images/loss-guide/vadv2-loss.svg)

### 4.3 扩散 Loss

\[
\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
\]

- **直觉**：让模型学"从纯噪声到数据轨迹"的去噪路径
- **谁在用**：DiffusionDrive、Diffusion Planner、Gen-Drive
- **扩散 Loss 的特殊性**：它既不是回归也不是分类——它通过**预测噪声**来隐式地学习数据分布

### 4.4 Flow Matching Loss

\[
\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right]
\]

- **直觉**：学一条从噪声到数据的直线路径（扩散是曲线路径）
- **谁在用**：π0、最新一批扩散规划方法

### 4.5 分布损失总结

| 方法 | Loss | 本质 |
|------|------|------|
| VADv2 | KL 散度（分布匹配） | 离散词表分类 |
| DiffusionDrive | 扩散简单 Loss (ε 预测) | 连续去噪 |
| Flow Matching | 条件流匹配 | 直线路径去噪 |
| 扩散 Planner | 扩散 Loss + 引导 | 联合预测-规划 |

---

## 五、安全约束 Loss：让模型学会"绝对不能做什么"

这是端到端自动驾驶最有特色的 Loss 类别——**纯模仿学习不管安全**，必须显式把安全约束加进去。

### 5.1 碰撞 Loss

UniAD 的碰撞 Loss：

\[
\mathcal{L}_{collision} = \sum_t \max(0, \text{dist}(\tau_t, O_t) - \text{margin})
\]

用占用预测结果 \(O_t\) 来惩罚与障碍物重叠的规划轨迹。margin 是一个安全缓冲——轨迹不需要离障碍物无限远，但太近就要被惩罚。

### 5.2 边界 / 冲突 Loss

VADv2 的冲突 Loss：

\[
\mathcal{L}_{conflict} = \sum_{a \in V} \mathbb{1}_{\text{conflict}}(a) \cdot \log p_{\text{pred}}(a)
\]

任何与碰撞 / 道路边界冲突的动作被标记为负样本，概率被压低。

### 5.3 DiffusionDrive 的安全 Loss 组合

```
ℒ_total = ℒ_diff + λ_collision ℒ_collision + λ_boundary ℒ_boundary + λ_comfort ℒ_comfort
```

- 扩散 Loss 负责"学得像人类"
- 碰撞 Loss 负责"别撞上"
- 边界 Loss 负责"别出车道"
- 舒适 Loss 负责"别急刹急转"

### 5.4 安全 Loss 设计原则

| 原则 | 说明 |
|------|------|
| **可微** | 安全 Loss 必须可微才能反向传播 |
| **可叠加** | 不同安全约束的 Loss 权重独立调参 |
| **非零安全** | 不能把安全 Loss 加到零——需要 margin 允许一定容忍度 |
| **先验 vs 学习** | 碰撞是硬几何约束（适合硬编码），舒适度是软感受（适合学习） |

---

## 六、多任务 Loss 平衡

端到端模型有多个 Loss 同时优化，怎么让它们不打架？

![UniAD 多任务 Loss 设计](/images/loss-guide/multitask-loss.svg)

### 6.1 常见问题

- **梯度尺度不一致**：规划的 L2 Loss 在 1 量级，跟踪的分类 Loss 在 0.01 量级，如果不加处理，规划的梯度会淹没感知
- **任务冲突**：最优跟踪框不一定最有利于规划，多任务 Loss 本质上是**帕累托优化**

### 6.2 几种平衡策略

**手动加权**：最简单的办法，给每个 Loss 一个权重系数。UniAD 就是手动调的。

**不确定性加权**：

\[
\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i
\]

可学习的 \(\sigma_i\) 自动调节每个任务的权重——任务噪声大（不确定性高），权重自动降低。

**梯度归一化**：GradNorm / PCGrad——检测每个任务的梯度方向，冲突时做投影/裁剪。

### 6.3 两阶段训练：最实用的工程经验

直接从头联合训练端到端模型几乎必然失败。UniAD 的**两阶段策略**已经被后续几乎所有工作（VAD、SparseDrive 等）采用：

1. **阶段一**：各模块分别预训练（感知→预测→规划逐个训好）
2. **阶段二**：联合微调（以规划 Loss 为主导，回传至所有上游模块）

VAD 更进一步，做到了**一次性端到端训练**——因为它把规划约束写进了可微的向量化 Loss，不再需要分阶段。

---

## 七、强化学习 Loss：从"模仿"到"探索"

当监督学习学到的是"人类怎么做"，RL 学到的是"怎么做更好"。这是 Loss 设计思想的又一次跃迁。

### 7.1 PPO-Clip

\[
\mathcal{L}_{PPO} = -\mathbb{E} \left[ \min\left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A, \text{clip}\left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon \right) A \right) \right]
\]

- **直觉**：新策略不能比旧策略差太多（clip 约束更新幅度）
- **谁在用**：多智能体交互策略

### 7.2 GRPO（Group Relative Policy Optimization）

```
Advantage(a_i) = (score_i - mean(score_group)) / std(score_group)
```

- **直觉**：在候选轨迹组内算相对优势，不需要价值网络
- **谁在用**：AlphaDrive-GRPO、Flow-GRPO、AutoVLA、DiffusionDriveV2
- **为什么 GRPO 流行**：因为端到端自动驾驶天然产出一组候选轨迹，组内相对比较比绝对评分更稳

### 7.3 DiffusionDriveV2 的截断 GRPO

\[
\mathcal{L}_{RL} = -\frac{1}{N_{anchor}} \sum_{k=1}^{N_{anchor}} \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_{trunc}} \sum_t \gamma^{t-1} \log \pi_\theta(\tau_{t-1}^{k,i} \mid \tau_t^{k,i}) A^{k,i}
\]

只对去噪过程的最后几步（T_trunc=2）算 RL Loss，大大降低计算量。

---

## 八、排序 / 对比 Loss

### 8.1 Margin-Rank Loss

\[
\mathcal{L}_{rank} = \sum_{i,j} \max(0, -\alpha \cdot (\hat{s}_i - \hat{s}_j) + m)
\]

- **直觉**：排在前面的轨迹分数要明显高于后面的
- **谁在用**：DiffusionDriveV2 的两阶段选择器

### 8.2 对比 Loss

\[
\mathcal{L}_{contrast} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\sum \exp(\text{sim}(q, k_i) / \tau)}
\]

- **直觉**：拉近正样本对，推远负样本对
- **谁在用**：JEPA-DRIVE、表征学习

---

## 九、全景总结：Loss 设计的演进脉络

![损失函数全景](/images/loss-guide/loss-taxonomy.svg)

### 演进路线

```
回归 Loss (L1/L2)
    ↓  确定性回归 → 多模态场景崩坏
分类 Loss (交叉熵/Focal)
    ↓  离散化动作空间 → 词表大小受限
分布 Loss (KL/扩散/Flow)
    ↓  概率建模 → 安全无法保证
安全约束 Loss (碰撞/边界)
    ↓  被动兜底 → 不够主动
RL Loss (PPO/GRPO)
    ↓  探索不够 → 依赖奖励设计
多任务平衡 (不确定性加权/两阶段)
    ↑  最终形态：各 Loss 协同优化
```

### 十条设计原则

1. **回归适合精确控制，不适合多模态** — 轨迹回归用 L1，框回归用 Smooth L1
2. **分类解决多模态，但受限于词表覆盖** — Focal Loss 解决正负样本不平衡
3. **分布匹配是"不丢信息"的回归替代方案** — KL 散度允许分布呈任意形状
4. **扩散 Loss 同时做生成 + 分布建模** — 简单 Loss 即可学复杂分布
5. **安全约束不能靠学，要硬编码进 Loss** — collision loss / conflict loss 是必须的
6. **多任务 Loss 需要显式平衡** — 两阶段训练是最可靠的工程经验
7. **RL Loss 让模型超越人类** — GRPO 是当前端到端规划的主流选择
8. **不同 Loss 服务于不同的优化目标** — 理解目标比理解公式更重要
9. **没有万能 Loss** — 每个 Loss 都是对驾驶某个维度的建模
10. **Loss 进化的方向：从"找答案"到"学分布"再到"定策略"**

---

## 十、写在最后

读完这么多 Loss，你可能会觉得：一个规划的 Loss 怎么可以这么复杂？

其实把视角拉高一点就清楚了。端到端自动驾驶是一个**多目标优化问题**——要像人（模仿）、要安全（碰撞）、要舒适（jerk）、要合法（车道）、还要能处理不确定性（多模态）。每个目标都是一个 Loss，每个 Loss 都有适合它的数学形式。

一个好的 Loss 设计，不是在选"哪个 Loss 最好"，而是在搞清楚**这个 Loss 让模型把注意力放在哪里**——是让模型"猜得准"（L2），还是"选得对"（分类），还是"不犯错"（碰撞 Loss），还是"敢于尝试"（GRPO）。

理解了这一点，你就理解了端到端自动驾驶 Loss 设计的全部哲学。
