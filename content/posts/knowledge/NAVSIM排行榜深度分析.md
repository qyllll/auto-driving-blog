---
title: "NAVSIM 排行榜深度分析：谁在统治端到端规划？架构、创新与分数全解"
date: 2026-07-19
draft: false
categories: ["产业实践"]
tags: ["NAVSIM", "排行榜分析", "端到端自动驾驶", "PDMS", "EPDMS", "规划", "产业实践"]
summary: "NAVSIM 已成为端到端规划的事实标准benchmark，PDMS/EPDMS排行榜上群雄逐鹿。本文系统梳理navtest/navhard双榜Top方法，从Scoring-based、Diffusion-based、VLA、World Model 四大技术路线解构各家架构设计与核心创新，严格区分官方排行榜已录结果与arXiv宣称结果，并给出技术趋势判断与个人思考。数据截至2026年7月。"
---

## 📄 背景：NAVSIM 为何成为必争之地

NAVSIM 自 NeurIPS 2024 发布以来，已成为端到端自动驾驶规划的事实标准benchmark。它用**非反应式仿真**（non-reactive simulation）在开环框架下计算闭环相关性更强的指标，v1 用 PDMS，v2 升级为 EPDMS（扩展了车道保持、红绿灯合规、方向合规和扩展舒适度等子指标）。v2 还推出 **navhard** 两阶段评测（Stage 1 在原始场景评估 + Stage 2 在3DGS生成的偏移场景上再评估），专门测试规划器的鲁棒性。

截至 2026 年 7 月，NAVSIM 排行榜上的竞争已经白热化——Top 方法的 PDMS 从 2024 年的 66 分飙升到 94+，navhard 从个位数冲到 55。这不是一两个技巧的堆叠，而是整个技术路线的快速迭代。

**一个重要提醒**：本文区分两类分数来源：
- **官方排行榜（Leaderboard）**：通过 HuggingFace 提交、由官方评估服务器计算的分数，navtest 排名来自 [AGC2024-P/e2e-driving-navtest](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)，navhard 排名来自 [AGC2025/e2e-driving-navhard](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)
- **arXiv 宣称（Reported）**：论文中自我报告的结果，可能未提交到官方排行榜或提交了但未公开

这两者可信度不同——官方榜单一视同仁、可复现；arXiv 宣称可能有实验环境差异。

## 📊 排行榜全景

### NAVSIM v1 navtest（PDMS 排名）

| 排名 | 方法 | PDMS | 来源 | 路线 | 发表处 |
|:----:|------|:----:|:----:|:----:|:------:|
| 1 | CLOVER | **94.5** | 官方榜 | Scoring-based | arXiv 2605.15120 |
| 2 | ExploreVLA | **93.7** | arXiv | VLA+WM+RL | arXiv 2604.02714 |
| 3 | Centaur | 92.1 | 官方榜 | Scoring+TTT | arXiv 2503.11650 |
| 4 | SparseDriveV2 | 92.0 | 官方榜 | Scoring-based | arXiv 2603.29163 |
| 5 | Hydra-SE | 91.87 | 官方榜 | Scoring-based | 2025 |
| 6 | Hydra-MDP | 91.26 | 官方榜 | Scoring-based | NeurIPS 2024 |
| 7 | **DriveFuture** | 90.7 | **arXiv** | World Model | arXiv 2605.09701 |
| 8 | DriveVLA-W0 | 90.2 | 官方榜 | VLA+WM | ICLR 2026 |
| 9 | AutoVLA | 89.1 | arXiv | VLA | NeurIPS 2026 |
| 10 | WorldRFT | 87.8 | arXiv | WM+RL | AAAI 2026 |

### NAVSIM v2 navtest（EPDMS 排名）

| 排名 | 方法 | EPDMS | 来源 | 路线 | 备注 |
|:----:|------|:-----:|:----:|:----:|:----:|
| 1 | Human | 94.5 | 官方榜 | 人类 | 上限参考 |
| 2 | **Metis** | **90.3** | 官方榜 | WAM | arXiv 2606.15869 |
| 3 | CLOVER | 90.4 | 官方榜 | Scoring-based | 87.2 Variant |
| 4 | AutoDrive-P³ | 89.9 | 官方榜 | VLA+CoT+RL | perception=true |
| 5 | **DriveFuture** | 89.9 | **arXiv** | World Model | 未在官方榜出现 |
| 6 | EponaV2 | 88.9 | 官方榜 | WM+Flow | perception=false |
| 7 | ExploreVLA | 88.8 | arXiv | VLA+WM+RL | 未在官方榜出现 |
| 8 | DriveLaW | 88.6 | 官方榜 | VLA | - |
| 9 | PWM | 88.2 | 官方榜 | - | - |
| 10 | DriveWorld-VLA | 86.8 | 官方榜 | VLA+WM | - |

### NAVSIM v2 navhard（EPDMS 排名，两阶段）

| 排名 | 方法 | EPDMS | 来源 | 路线 |
|:----:|------|:-----:|:----:|:----:|
| 1 | **DriveFuture** | **55.5** | arXiv宣称为#1 | World Model |
| 2 | DrivoR | 54.6 | arXiv | - |
| 3 | SimScale | 53.2 | arXiv | 数据增强 |
| 4 | GuideFlow | 51.5 | arXiv | Diffusion |
| 5 | Expert | 51.3 | 官方榜 | 人类专家上限 |
| 6 | ZTRS | 45.5 | arXiv | - |
| 7 | World4Drive | 34.9 | 官方榜 | World Model |
| 8 | Metis | 32.2 | 官方榜 | WAM |
| 9 | MindDrive | 30.9 | arXiv | - |
| 10 | LTF | 24.4 | 官方榜 | Baseline |

navhard 的分数整体很低（Human Expert 也才 51.3），因为 Stage 2 的 3DGS 生成场景引入了观测偏移，会大量暴露规划器在分布外场景下的脆弱性。DriveFuture 以 55.5 超越人类上限本身就是一件有意思的事。

## 🏗️ 四大技术路线架构详解

下面按 PDMS 从高到低，逐一拆解 Top 方法的架构设计和核心创新。

### 路线一：Scoring-based（评分排序范式）

这类方法占据排行榜头部最多席位。核心哲学：**不直接生成轨迹，而是先生成大量候选，再学习打分器选出最优**。

#### 1. CLOVER（94.5 PDMS / 90.4 EPDMS）
- **机构**：清华 AIR × 中科大 × 北航
- **架构**：轻量级生成器（DINOv2 ViT-S）+ 打分器（交叉注意力）
- **核心创新**：
  - **伪专家覆盖训练（Stage 1）**：从多类可解释动作族（横向偏移、加减速等）构造候选池，用真实评估器 $R$ 过滤 + FPS 采样，形成高质量伪专家集合。把单轨迹模仿升级为集合级覆盖监督
  - **保守闭环自蒸馏（Stage 2）**：交替优化打分器（逼近真实子分数）和生成器（向教师选出的 Top-k 与 Pareto 目标学习），加稳定性正则防止分布漂移
  - **理论保障**：证明打分器选中的目标在真实评估器下统计优于当前分布时，保守蒸馏必然提升高分区概率质量
- **官方榜 vs arXiv**：**官方榜**（navtest 可见）

#### 2. SparseDriveV2（92.0 PDMS / 90.1 EPDMS）
- **机构**：—
- **架构**：Scoring-based，因子化轨迹词汇表
- **核心创新**：
  - **可扩展词汇表**：将轨迹分解为几何路径 + 速度剖面，组合覆盖动作空间——Dense anchor 比固定离散更优
  - **两级评分**：粗粒度因子化评分（路径×速度）→ 细粒度组合评分，平衡效率与精度
- **官方榜 vs arXiv**：**官方榜**（navtest 可见）

#### 3. Hydra-MDP 系列（91.26 PDMS / Hydra-SE 91.87）
- **机构**：NVIDIA
- **架构**：Transformer 编码器 → 固定 anchor 词汇表 → 多头打分器
- **核心创新**：
  - **多目标 Hydra 蒸馏**：用多个专家（规则评估器）的分数监督多个 score head，每个 head 学习一种偏好
  - **Hydra-SE**：引入 Cluster Entropy 作为不确定性度量，在多个 anchor 轨迹的分数分布上计算熵，评估决策置信度
- **官方榜 vs arXiv**：**官方榜**

### 路线二：VLA + 世界模型（Vision-Language-Action）

#### 4. ExploreVLA（93.7 PDMS / 88.8 EPDMS）
- **机构**：—
- **架构**：ViT 编码器 → 统一 backbone → 规划头 + 世界模型头（RGB + Depth 预测）
- **核心创新**：
  - **密集世界建模**：除轨迹外还预测未来 RGB + Depth 图像，提供细粒度视觉与几何监督
  - **不确定性驱动的探索奖励**：世界模型的图像预测不确定性作为内在探索信号——高不确定性 = 分布外但安全 = 值得学习的策略
  - **安全门控 GRPO**：只对既高不确定性又高 PDMS 的轨迹给予探索奖励，防止危险探索
- **官方榜 vs arXiv**：**arXiv 宣称**（未出现在官方 navtest 排行榜）

#### 5. AutoDrive-P³（89.9 EPDMS，perception=true）
- **机构**：北京大学，**ICLR 2026**
- **架构**：VLM + P³-CoT（感知-预测-规划链式思维）
- **核心创新**：
  - **P³-CoT 数据集**：构建"感知→预测→规划"三层结构化的推理链标签
  - **P³-GRPO**：分层渐进式强化微调——奖励从规划反传至感知和预测模块，Stage 1 优化感知 → Stage 2 优化预测 → Stage 3 优化规划，最终联合
  - **双思维模式**：Detailed Thinking（完整 CoT，高精度）/ Fast Thinking（直接输出，高效率）
- **官方榜 vs arXiv**：**官方榜**（perception=true 分类下可见）

#### 6. DriveVLA-W0（90.2 PDMS / 86.1-86.9 EPDMS）
- **机构**：中科院自动化所，**ICLR 2026**
- **架构**：VLA backbone + 世界模型（autoregressive / diffusion 双版本）
- **核心创新**：
  - **世界模型作为密集监督**：未来图像预测为稀疏的动作标签补充密集监督信号，显著放大数据 scaling law
  - **双版本实现**：离散 visual token 用 autoregressive WM，连续视觉特征用 diffusion WM
  - **轻量级动作专家（MoE）**：推理时 bypass 世界模型，只走动作分支，降低延迟
- **官方榜 vs arXiv**：**官方榜**

### 路线三：纯 World Model / WAM（世界动作模型）

#### 7. DriveFuture（90.7 PDMS / 89.9 EPDMS / navhard 55.5 #1）
- **机构**：—
- **架构**：Encoder → 未来潜在预测器 → Cross-Attention 精炼 → Diffusion 规划器
- **核心创新**：
  - **未来条件化潜在世界模型**：用 GT 未来潜在状态训练时做条件，推理时用预测的未来状态接替。区别于"把未来当预测目标"，而是"把未来当决策条件"
  - **统一训练-推理范式**：训练和推理都用未来条件化，避免了 train-inference mismatch
  - **navhard 表现突出**：55.5 超越人类专家，因世界模型天然擅长处理观测偏移（Stage 2 的 3DGS 合成场景）
- **官方榜 vs arXiv**：navtest **arXiv 宣称**（未出现在 navtest 官方榜）；navhard **arXiv 称 #1**（navhard 官方榜不可见具体排名）

#### 8. Metis（90.3 EPDMS / 32.2 navhard）
- **机构**：复旦大学 × 理想汽车 × 伦敦帝国理工
- **架构**：Mixture-of-Transformers（视频生成 expert + 动作预测 expert）
- **核心创新**：
  - **解耦视频-动作建模**：分离视频生成和动作预测，各自用独立的 Transformer expert，避免表征混叠
  - **非对称注意力掩码**：训练时两个 expert 联合训练，推理时动作 expert 可跳过视频生成，大幅降低延迟
  - **世界动作模型（WAM）**：统一"世界模型"和"动作模型"两个概念，在潜在空间做前向推演
- **官方榜 vs arXiv**：**官方榜**（navtest 和 navhard 都可见）

#### 9. EponaV2（88.9 EPDMS，perception-free）
- **机构**：—
- **架构**：自回归扩散世界模型 + 未来深度/语义预测
- **核心创新**：
  - **综合未来推理**：不只是预测下一帧 RGB，还预测未来 depth map 和 semantic map——3D 几何 + 语义双重理解
  - **Flow Matching GRPO**：用 flow matching 替代传统扩散，配合 GRPO 优化规划精度
  - **Perception-free**：不依赖人工感知标注，便于数据 scaling
- **官方榜 vs arXiv**：**官方榜**

### 路线四：Diffusion-based / Flow Matching 范式

#### 10. DiffusionDrive 系列（85.6-88.2 EPDMS）
- **机构**：华中科大，**CVPR 2025**
- **架构**：截断扩散策略（truncated diffusion policy）
- **核心创新**：
  - 从 anchor 高斯分布去噪而非标准高斯 → 减少去噪步数（4-8 步）
  - 天然多模态输出
- **官方榜 vs arXiv**：**官方榜**（多版本）

#### 11. GuideFlow（51.5 navhard）
- **机构**：—，**CVPR 2026**
- **架构**：Flow Matching 引导
- **核心创新**：
  - 用 flow matching 生成候选轨迹，引入引导机制提升多模态表达能力
- **官方榜 vs arXiv**：**arXiv**（navhard）

## 🔬 技术路线对比

| 维度 | Scoring-based | Diffusion-based | VLA+WM | Pure WM/WAM |
|:----|:-------------:|:---------------:|:------:|:-----------:|
| **代表方法** | CLOVER, SparseDriveV2, Hydra-MDP | DiffusionDrive, GuideFlow | ExploreVLA, AutoDrive-P³ | DriveFuture, Metis, EponaV2 |
| **最高 PDMS** | 94.5 | ~88 | 93.7 | 90.7 |
| **最高 EPDMS** | 90.4 | ~88 | 89.9 | 90.3 |
| **navhard EPDMS** | — | 51.5 | — | 55.5 |
| **推理速度** | 快（一次前向） | 中（4-8步去噪） | 慢（VLM 解码） | 中-慢 |
| **多模态支持** | 依赖 vocab 覆盖 | ✅ 天然 | ✅ 天然 | ✅ 天然 |
| **是否需额外数据** | 否 | 否 | 否 | 是（未来帧标注） |
| **理论深度** | 高（蒸馏理论） | 中 | 中 | 高（潜在动力学） |
| **工程复杂度** | 低-中 | 中 | 高 | 高 |
| **代码开源** | ✅ CLOVER | ✅ DiffusionDrive | ❌ 大部分否 | ✅ Metis |

### 关键洞察

**1. Scoring-based 在常规场景仍然最稳。**
CLOVER 的 94.5 PDMS 和 SparseDriveV2 的 90.1 EPDMS 说明，当你的评测环境是确定的（navtest），"生成候选 + 打分排序"这个范式几乎不可战胜。它的优势在于：
- 打分器可以直接用真实评估器的分数做监督，降维打击
- 候选生成可离线优化，推理时只做排序，延迟极小
- 集合级训练天然对抗 mode averaging

**2. World Model / WAM 在难例场景异军突起。**
navhard 上 DriveFuture 的 55.5 大幅领先（人类也只有 51.3），说明世界模型在面对分布外场景时（3DGS 偏移观测）的鲁棒性更强。这也是直觉上正确的方向——世界模型学会了环境动力学，当观测偏移时，它可以"想象"出合理的未来状态再做规划。

**3. VLA + GRPO 是上升最快的方向。**
ExploreVLA（93.7 PDMS）和 AutoDrive-P³（89.9 EPDMS）都用 GRPO 做 RL 后训练，而且都引入了某种形式的"世界模型"或"链式思维"作为推理增强。这是 2026 年最显著的趋势——**纯模仿学习已经到顶，RL 后训练成为标配**。

**4. Perception-free + WM 有望突破数据瓶颈。**
EponaV2 不需要人工感知标注，只靠未来深度/语义预测的自监督就能达到 88.9 EPDMS。如果真的把数据 scaling 跑起来，这条路线有潜力追上甚至超越依赖标注的方案。

## ⚠️ 排行榜来源透明化

一个重要的事实：**很多高分的 arXiv 论文并没有出现在官方排行榜上**。具体来说：

| 方法 | 宣称分数 | 官方榜可见 | 说明 |
|:----|:--------:|:---------:|------|
| CLOVER | 94.5 PDMS | ✅ 可见 | 官方榜 Top |
| ExploreVLA | 93.7 PDMS | ❌ 未出现 | arXiv-only |
| DriveFuture | 90.7 PDMS / 55.5 navhard | ❌ navtest 未出现，navhard 宣称 #1 | arXiv-only；navhard 官方榜未公开具体排名 |
| Metis | 90.3 EPDMS | ✅ 可见 | 官方榜 Top |
| AutoDrive-P³ | 89.9 EPDMS | ✅ 可见 | Perception=true 分类 |
| SparseDriveV2 | 92.0 PDMS | ✅ 可见 | 官方榜可见 |
| Centaur | 92.1 PDMS | ✅ 可见 | 官方榜可见 |

为什么 arXiv 宣称和官方榜有差异？可能原因：
1. **评估协议差异**：perception=true/false、传感器配置、训练数据范围等
2. **排行榜提交限制**：部分论文发表时排行榜已关闭提交
3. **分数时效性**：排行榜持续更新，论文报告的是某个时间点数据
4. **submit 意愿**：有些组只发了论文懒得提交

因此我的建议：**引用分数时优先参考官方排行榜的成绩**，arXiv 宣称的分数需谨慎对待，尤其是那些宣称 "SOTA" 但未提交到官方榜的。

## 💭 个人思考

### 1. 技术收敛了吗？

看起来 Scoring-based 和 World Model / WAM 正在形成两条互补的技术路线：
- **简单场景 → Scoring-based**：高确定性、高速度、已接近人类上限
- **难例/分布外 → World Model / WAM**：具备动力学理解能力，但计算开销大

但我认为最终会收敛到**世界模型 + 评分排序**的混合范式。DriveFuture 已经在尝试这条路（用世界模型做未来条件化，再用 scorer 选轨迹）。CLOVER 如果引入世界模型做伪专家生成，可能直接冲到 96+。

### 2. GRPO 是万能药吗？

2026 年几乎每篇高分论文都在用 GRPO。但我有两个担忧：
- **Reward hacking 风险**：当 reward 是规则化指标（PDMS/EPDMS）而非真实驾驶质量时，策略会找到投机取巧的方式最大化分数
- **在线 RL 的仿真 gap**：NAVSIM 是非反应式仿真，规划器不与环境交互。闭环 GRPO（真正在仿真器里开车学）还没大规模验证

CLOVER 的保守蒸馏理论提供了一个更安全的替代方案——不需要在线交互就能利用评估器反馈。

### 3. navhard 的意义

navhard 是一个被低估的评测维度。55.5 分超过人类上限这件事让我反思：
- 3DGS 生成的"观测偏移"可能包含了评测本身的漏洞（某些偏移在物理上不可能）
- 但也确实筛选出了真正懂动力学的模型（DriveFuture）和普通模仿模型（LTF 23.1）之间的差距

未来如果 navhard 被更广泛接受，"navtest 刷分"的军备竞赛可能会降温。

### 4. 对 VLA 工程师的启示

结合前面的 JD 分析，从 NAVSIM 排行榜可以看到用人单位到底在找什么能力：
- **RL 后训练**（GRPO/PPO）—— ExploreVLA、AutoDrive-P³、WorldRFT 都证明了 RL 的价值
- **世界模型** —— 从 DriveFuture 到 EponaV2，世界模型正在从"研究玩具"变成"部署必需品"
- **Scoring-based 框架** —— CLOVER 的蒸馏理论代表了对"损失函数设计"的深度理解，这正是工程团队需要的
- **数据 scaling 思维** —— DriveVLA-W0 的 scaling law 分析和 EponaV2 的 perception-free 范式，指向"如何经济地获取更多有效数据"

头部方法的代码和数据很少完全开源，但从论文中提取的架构设计思路是可以复现的。把这篇文章里列出的核心创新点逐个实现一遍，你的 VLA 技能栈会非常扎实。

## 📚 延伸阅读

如果想进一步深入，可以配合本博客的以下文章一起看：

- **论文精读 - DriveFuture**（2605.09701）：世界模型未来条件化的完整实现细节
- **论文精读 - ExploreVLA**（2604.02714）：不确定性驱动探索奖励与安全门控 GRPO 的工程实践
- **论文精读 - AutoDrive-P³**（2603.28116，ICLR 2026）：P³-CoT 与分层渐进式强化微调
- **产业实践 - 前沿 VLA 模型全景速览**：从产品视角横向对比主流 VLA 方案的定位与取舍
- **产业实践 - VLA 大模型训练技巧**：GRPO、蒸馏、数据 scaling 的实操经验

数据截至 2026 年 7 月，排行榜仍在快速变化。本文所有分数建议以官方 HuggingFace 排行榜为准，arXiv 宣称结果仅供参考。
