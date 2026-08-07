---
title: "端到端自动驾驶模型架构全解：从感知到规划的五大范式与我的思考"
date: 2026-08-07
draft: false
description: "把博客里精读过的所有自动驾驶模型统一进一个坐标系：按感知/预测/决策/规划四个架构维度横向对比 UniAD、VAD、SparseDrive、DiffusionDrive、CLOVER、TransDiffuser、DriveVLA-W0、Cosmos 等二十余个模型，逐条盘点创新点与 PDMS/EPDMS 分数，最后给出我对端到端自动驾驶演进方向的个人思考。"
categories: ["个人思考"]
tags: ["端到端自动驾驶", "UniAD", "VAD", "SparseDrive", "DiffusionDrive", "CLOVER", "VLA", "世界模型", "GRPO", "NAVSIM", "PDMS", "架构综述"]
math: true
---

## 一、为什么写这篇文章

本博客陆续精读了大量自动驾驶模型——从最早的感知基座，到 BEV / 稀疏 query，再到扩散生成式规划、VLA 大语言模型、世界模型与 GRPO 强化学习。这些文章单独看是一篇篇论文精读，但放在一起，端到端自动驾驶这条技术线在 **感知 → 预测 → 决策 → 规划** 四个架构维度上的演进脉络其实非常清晰。

这篇文章想做的，就是把博客里所有精读过的模型拉进同一个坐标系，回答三个问题：

1. **感知**：从稠密 BEV 到稀疏 query 再到纯视觉 token，一路在换什么？
2. **预测/决策/规划**：从显式管线到打分排序、扩散生成、LLM 推理、世界模型，谁在取代谁？
3. **谁真正有效**：把彼此在 NAVSIM 上的 PDMS/EPDMS 分数和架构创新对照起来，看看分数到底来自技巧还是来自架构。

> 本文覆盖的模型按技术路线分组，每一节给出"创新点 → 感知 → 预测 → 决策 → 规划"五维拆解，并把有 PDMS/EPDMS 分数的模型单独汇成一张表（见文末）。分数严格区分**官方排行榜**与 **arXiv 自报**，口径不一的地方我会注明。

---

## 二、先立坐标系：五大流派一张图

我把博客里出现过的端到端规划方法粗略归为**五大流派**，它们的分界线正是"感知表征"和"规划输出方式"这两个维度：

| 流派 | 代表模型 | 核心思想 | 感知表征 | 规划输出 |
|------|---------|---------|---------|---------|
| **① 显式端到端管线** | UniAD、VAD、SparseDrive | 感知/预测/规划在同一个可训练网络里串起来 | 稠密 BEV → 稀疏 query | 单条/多条轨迹回归 |
| **② 评分排序式 (Scoring)** | CLOVER、Hydra-MDP、SparseDriveV2、DriveSuprim、GTRS | 先大量生成候选，再学打分器选出最优 | 稀疏特征 / 轨迹词汇表 | 从候选集合中选 Top-1 |
| **③ 扩散/Flow 生成式** | DiffusionDrive、DiffusionPlanner、TransDiffuser、DiffusionDriveV2 | 把规划当成从噪声里去噪 | 稀疏感知 / BEV | 生成多样候选轨迹 |
| **④ VLA 大模型式** | DriveVLM、EMMA、AutoVLA、DriveVLA-W0、AlpaMayo-R1、ReCogDrive | 用 LLM/VLM 做感知+推理+决策 | 视觉 token / 稀疏 query | 语言决策或轨迹 token |
| **⑤ 世界模型式** | Cosmos、DriveFuture、DreamerV3、Drive-JEPA、JEPA-DRIVE | 建模环境动力学/未来状态 | 隐潜/视频 token | 以未来状态为条件规划 |

> 一句话直觉：**① 是"手拉手串行"，② 是"多打几遍再挑"，③ 是"从噪声里一遍遍画"，④ 是"读得懂交通法规再动手"，⑤ 是"猜完未来再决定"**。五种流派不是互相淘汰，而是在不同评测维度上各占一块地盘。

---

## 三、流派①：显式端到端管线——从 BEV 到稀疏 query

这类方法把感知、预测、规划放在**同一个网络**里联合训练，是端到端自动驾驶的"正统"。它们的重点在于：**用什么中间表征承载"场景理解"，以及感知输出怎么喂给规划。**

### 1. UniAD——显式端到端的开山之作（CVPR 2023 最佳论文）

- **创新点**：第一次把检测 → 跟踪 → 建图 → **运动预测 → 规划** 串成一个端到端、可联合训练的网络（不再是模块拼接），并用统一的 **query 接口**在不同任务间传特征（特征级、可微、不丢信息）。
- **感知**：用 **BEVFormer** 在 BEV 网格上做稠密感知——检测目标 agent（TrackFormer）、建向量化地图（MapFormer）；注意是稠密 BEV 特征（200×200×256）。
- **预测**：**MotionFormer**，对每个 agent 输出 K=6 条多模态轨迹；还叠加 **OccFormer**（占位预测）做稠密安全兜底。
- **决策/规划**：规划 head 用 ego query 交叉注意力到 BEV + motion + occupancy，直接回归 ego 未来轨迹；加"碰撞损失 + 推理时 occupancy 梯度优化"。
- **PDMS**：无（nuScenes 开环 L2 0.44/0.99/1.71，碰撞率 0.56）。它证明了范式可行，但**重**（~7-10 FPS，需要两阶段训练保收敛）。

> **我的判断**：UniAD 的意义不是分数，而是把"规划"第一次变成整个网络的**集体目标**——后续所有流派（包括世界模型）都在做"如何让规划驱动感知"这件事。

### 2. VAD——向量化的"轻量版 UniAD"

- **创新点**：把稠密占用/语义栅格**全部换成向量化表示**（地图折线与 agent 运动向量），彻底扔掉稠密栅格，快 2.5-9.3×（4.5-16.8 FPS）。
- **感知**：仍用 BEV 编码器（BEVFormer 风格），但输出的是向量化 3D box（agent）和向量化地图（map）。
- **预测**：Motion Head 出 agent 多模态轨迹，用模式嵌入 + agent-map 交叉注意力让轨迹贴车道。
- **决策/规划**：三个可微的**向量化规划约束**（ego-agent 碰撞、ego-边界、ego-车道方向）+ 用离线专家（Hybrid A*）做 **规划 KD** 保安全。VADv2 把规划升级为 M 条候选轨迹 + softmax 概率（分类-回归混合）。
- **PDMS**：无（nuScenes L2 0.72，碰撞 0.22%）。

> **我的判断**：VAD 证明"向量化"在效率和安全性上双赢——用**"更廉价的表示 + 显式可微约束"**替代"稠密高算力 + 隐式安全"，是 UniAD 到 SparseDrive 的中间形态。

### 3. SparseDrive——"完全稀疏"的显式管线

- **创新点**：砍掉稠密 BEV 编码层，让 **instance query 直接与多视角图像特征交互**（deformable attention 采样）；感知-预测-规划共享特征但**并行双支**（motion 和 planning 独立输出），打破 UniAD 的串行误差级联。
- **感知**：完全稀疏——计算量与**对象数成正比**而不是 BEV 分辨率平方；检测 + 追踪 + 向量化地图统一在 query 流里。
- **预测**：为每个 instance 做多模态轨迹预测（用已预测的 ego pose 做 anchoring）。
- **规划**：平行分支直接出未来 waypoints，配数据驱动 anchor + 碰撞/舒适/合规打分选择。
- **PDMS**：无（nuScenes 开环第一梯队）。

> **我的判断**：SparseDrive 解决了 UniAD 的计算瓶颈（query 数 vs BEV 分辨率平方），是"从显式管线走向稀疏 query"的**承上启下**一环——后面 SparseDriveV2 把它从"串行稀疏管线"升级成"Scoring 式词汇表"。

### 4. 感知基座：BEVFormer / Sparse4D

这两篇不是完整规划器，而是**感知基底**，但它们决定了上面所有管线"看到什么"：

- **BEVFormer**：稠密 BEV 的"标准定义"——spatiotemporal Transformer，可学习 BEV grid query + 空间交叉注意力（SCA，deformable 采样多相机）+ 带 ego 运动补偿的时序自注意力（TSA）。nuScenes NDS 56.9%，是 UniAD/VAD 的感知原型。
- **Sparse4D**：完全不建 BEV，稀疏 anchor 叠加 **4D keypoint 采样**（一个采样动作同时做空间+时序融合），随时间循环压缩特征；推理时给 instance 分配 ID 就能当 tracker（V3 AMOTA 67.7%）。这是稀疏感知的另一条路，也是 SparseDrive 稀疏化思想的感知层源头。

> **我的判断**：BEVFormer vs Sparse4D，本质是"**算力换语义**"（BEV 稠密、语义全但算贵）vs"**语义换算力**"（稀疏、算便宜但依赖 query 表达力）。显式管线在这两者之间拉锯，而 SparseDrive 之后逐渐倒向后者。

---

## 四、流派②：Scoring-based——"多生成几遍，再打分挑"

典型想法：**不直接"开"，而是"先想一万个开法，再让打分器挑最好的那个"**。在 NAVSIM 这种规则化、确定性的评测下，几乎是最稳的路线。

### 1. CLOVER（NAVSIM v1 navtest 官方榜第一：94.5 PDMS）

- **创新点**：**伪专家覆盖训练 + 保守闭环自蒸馏**。用真实评估器过滤出"伪专家集合"（从可解释动作族构造超大候选池 + FPS 采样保证覆盖），让打分器给伪专家集合打高分、给池外打低分；再用蒸馏把打分器的**完整排序偏好**（不只是选中的一条）回传给生成器。
- **感知/决策/规划**：冻结 DINOv2 ViT-S 提视觉特征 → 轻量生成器（Transformer decoder）出 N 条候选轨迹 → Scorer（Cross-Attention，候选↔场景特征）对各条预测子分数 → 选 Top-1。打分器的训练监督是**评估器在候选上算出的真实分数**——它直接在学习排行榜的判分函数。
- **理论保障**：论证"打分器选中目标在真实评估器下显著优于当前分布时，保守蒸馏必提升高分区概率质量"——不是 trick，是收敛理论。
- **PDMS**：navtest **PDMS 94.5（官方榜 #1）**、navtest v2 EPDMS 90.4、NavHard 48.3。

> **我的判断**：CLOVER 是"Scoring 流派天花板"。它的关键不是打分器多准，而是**让打分器与评估器对齐、再用打分器把生成器推向评估器偏好的闭合回路**——这本质是一种在线自我改进/蒸馏，和 RL 的 actor-critic 同构。

### 2. Hydra-MDP / Hydra-SE（NVIDIA）

- **创新点**：**多目标 Hydra 蒸馏**——用一个总分监督会把多目标"压扁"，于是用多个规则评估器各自算出的子分数，分别监督对应的 score head；推理时按权重聚合（或动态加权）。Hydra-SE 再加"**簇熵**（cluster entropy）"当不确定性度量，高熵时触发安全兜底。
- **架构**：固定 anchor 词表（K 条预定义轨迹）+ 多头打分器。推理极快、训练稳定。
- **PDMS**：Hydra-MDP 91.26（官方榜）、Hydra-SE 91.87（官方榜）。

> **我的判断**：Hydra-MDP 是"**用多专家当老师、多 head 学多维偏好**"的源头，最工程可控——几乎成为后续所有 Scoring 方法的对照基点。

### 3. SparseDriveV2（swc-17）

- **创新点**：把"整条轨迹"分解为**几何路径(path) × 速度剖面(velocity)** 两个因子词汇表，组合出候选轨迹集——动作空间指数级覆盖，而词汇量只等于两者之和（factorized vocab）；配两级评分（先粗筛 path/velocity，再细粒度联合打分）。另外还有检测/规划解耦、质量感知记忆库等工程升级。
- **感知/规划**：稀疏 query 检测 + 地图 → factorized 打分 → 选最优。
- **PDMS**：navtest **PDMS 92.0（官方榜）**、v2 EPDMS 87.35（随 path 锚点 1024→16384 单调提升；记忆从 9.5GB 涨到 38.9GB）。

> **我的判断**：SparseDriveV2 贡献了"**把动作空间解剖成可解耦轴**"的建模技巧，证明因子化打分上限受 vocab 规模约束。它是"从显式管线进入 Scoring"的完成体。

### 4. iPad、DriveSuprim——候选覆盖与选择

- **iPad**（arXiv 2505.15111）：proposal-centric + 多轮迭代规划——Scene Encoder + ProFormer（proposal 当 query 对 proposal-centric 图像特征做 deformable attention，迭代 K 轮自修正）+ 打分选优。proposal 数/迭代数/数据量都呈对数增长 PDM。NAVSIM v1 camera（R34）**PDMS 91.7**（NC 98.6 / DAC 98.3 / TTC 94.9 / Comf. 100 / EP 88.0）。
- **DriveSuprim**（arXiv 2506.06659）：把"选择"做到极致——coarse-to-fine 两级选择 + 旋转数据增强打破直行偏置 + soft label 蒸馏。R34 **PDMS 89.9**、EVA-ViT-L **93.5**。

### 5. GTRS（CVPR 2025 自动驾驶挑战赛 E2E 冠军）

- **创新点**：在**超密集轨迹词汇表（16384）**上训练打分器 + 词表 dropout，让打分器能泛化到推理时稀少的动态候选集合——统一"静态大词表"与"动态小候选"两个流派。
- **架构**：扩散策略生成细粒度动态候选 → 打分器 + 词表 dropout → 传感器旋转增强 + top-k 细化自蒸馏。
- **PDMS**：NavHard 榜 EPDMS **49.4**，逼近依赖真值感知的 PDM-Closed。

> **我的判断**：GTRS 抓到了 Scoring 的核心痛点——**打分器只能在"见过"的轨迹上可靠，但真正好的候选可能是"没见过"的**。词表 dropout 强迫打分器学习"潜力"，是方法论的关键。

---

## 五、流派③：扩散 / Flow 生成式规划

### 1. 奠基：Diffusion Policy（RSS 2023，机器人侧）

- **创新点**：把动作生成变成条件去噪过程（DDPM），建模**完整多模态分布，避免回归的 mode averaging**；可视化 + FiLM 条件注入。
- **规划**：receding horizon（生成 T 步、执行前 K 步、重规划）；推理时 DDIM 100→10→5 步。
- **PDMS**：无（机器人基准 +46.9%）。**它是整个扩散生成流派的方法论母体。**

### 2. Diffusion Planner（ICLR 2025 Oral）

- **创新点**：把规划定义为"**ego + M 邻居未来轨迹联合去噪**"——预测与规划合一，交互自然涌现；推理时用**训练免费的 DPS（Diffusion Posterior Sampling）分类器引导**注入碰撞/可行驶/车辆动力学能量约束，无需重训，一个模型多种驾驶风格。
- **规划**：DiT + MLP-Mixer 生成 ego+N 邻居整条轨迹；多轨迹采样 + 规则打分选优。
- **PDMS**：nuPlan 闭环 SOTA（无 NAVSIM PDMS）。

### 3. DiffusionDrive（CVPR 2025）/ DiffusionDriveV2

- **创新点（V1）**：**截断扩散 + 轨迹 anchor 化**——从 20 个 k-means 聚类 anchor 附近加噪声（而非标准高斯），去噪压缩到 **2 步**，实时可控；多候选 + 成本选择。
- **创新点（V2）**：在 DiffusionDrive 生成器上做 **Intra-Anchor GRPO**（组内比较保多模态稳定）+ **Inter-Anchor Truncated GRPO**（用碰撞做全局安全先验，截断负优势）+ 两级 Mode Selector；用尺度自适应乘性探索噪声保持轨迹几何平滑。
- **PDMS**：V1 **88.1**；V2 **PDMS 91.2**（EP 87.5、DAC 97.3）、v2 EPDMS 85.5；Top-10 质量均匀（PDMS@1 94.9 → @10 84.4，RL 抬高质量下限）。

> **我的判断**：DiffusionDrive 解决的是"**扩散太慢**"（截断 + 锚点）和"扩散太散"（V2 用 RL 抬高生成质量下限）。它证明扩散生成式在 NAVSIM 上可行，但天花板比 Scoring 略低（91.2 vs 94.5），原因在于没有显式打分器那层"安全筛选"。

### 4. TransDiffuser（arXiv 2505.09315，LiAuto）

- **创新点**：**多模态表示去相关（Decorrelation）**——强制多模态条件特征维度去相关，让条件信号更丰富地覆盖潜空间，无需锚点或场景先验也能生成多样化轨迹，从根上解决扩散轨迹生成的模式坍缩。
- **PDMS**：ResNet-34 + LiDAR 配置 NAVSIM **PDMS 94.85**，是目前 NAVSIM 上的最高分之一。

> **我的判断**：TransDiffuser 用"去相关"替代"锚点"解决多样性，是扩散生成方法的集大成者，拿到了几乎只有 Scoring 才能拿到的最高分。它与 CLOVER 并列，说明"覆盖 + 选择"与"生成分布本身足够好"是两条殊途同归的路。

### 5. 扩散流派横向对比

| 方法 | 生成方式 | 多样性保证 | RL | PDMS |
|------|---------|-----------|-----|------|
| Diffusion Planner | 联合 ego+邻居去噪 + 训练-free 引导 | 多模态采样 | ✗ | nuPlan 闭环 SOTA |
| DiffusionDrive | 20 锚点截断扩散 | 锚点覆盖 | GRPO（V2） | V1: 88.1 / V2: 91.2 |
| TransDiffuser | 去相关特征 + 无锚点 | 特征去相关 | ✗ | **94.85** |

---

## 六、流派④：VLA 大模型式——用语言模型做驾驶

这一流派的核心变化是：**规划不再是一个回归 head，而是一个会"读、会想、会推理"的骨干（LLM）+ 一个动作出口**。它天然适合长尾与交规语义，但**推理延迟**一直是大症结。

### 1. DriveVLM（SJTU × NIO）

- **创新点**：结构化 4 步 CoT（描述→分析→抽取→推理），慢 VLM 低频深思考 + 轻量快系统维持 10Hz 实时；用语言做**可审计的决策日志**，并用 GPT-4V 构造规模化 CoT 标注飞轮。
- **感知/决策**：VLM 视觉 token + 语言；输出**高层语言决策意图**（语义向量注入快系统）；快系统最终回归轨迹。
- **PDMS**：无（CARLA/自定义评测）。

> **我的判断**：DriveVLM 提出了一条到今天仍存在的**根本分工**：慢 VLM 决定"该咋办"、快系统负责"怎么干"。语言的"可编程性"可能就藏在这种"语言作为决策接口"里。

### 2. EMMA（Waymo，Gemini 系）

- **创新点**：把感知/预测/规划/VQA 全部译成**自然语言 VQA**，同一个 MLLM 一套权重；用**文本 token 化 3D 坐标**（字符序列）最大化复用 Gemini 世界知识；4 级 CoT 推理链（场景→关键目标带坐标→元决策→动作）。
- **PDMS**：无（nuScenes/WOMD 自报 SOTA），延迟是最大弱点。

> **我的判断**：EMMA 是"极简统一"的极端代表，但纯文本坐标 + 自回归解码，延迟巨大——它是"学术上优雅、工程上难落地"的典型。

### 3. AutoVLA（NeurIPS 2025，UCLA）

- **创新点**：**物理动作 token**——用 **KMeans 码本**（K=256-512）把连续轨迹离散成可执行的"驾驶 token"，让模型直接"说路线"；提供 **Fast（只出轨迹）/ Slow（CoT 推理 + 轨迹）双思维模式**，用 **GRPO/RFT** 让推理变成可开关的成本（简单场景自动省掉推理）。
- **感知/规划**：Qwen2.5-VL 3B backbone + 3 前置相机；自回归解码码本 token → 重建 8-step 轨迹。
- **PDMS**：navtest **PDMS 89.1**，RFT/GRPO + Best-of-N 后 **92.1**。

### 4. DriveVLA-W0 / DriveVLM-W0（CASIA × NIO，ICLR 2026）

- **创新点**：诊断 VLA"监督密度不足"——7-8B VLA 只监督几条轨迹维度浪费容量，于是加 **AR 世界模型（未来视觉 token）+ Diffusion 世界模型（连续特征）** 给 backbone 像素级密集监督；**MoE** 结构（7-8B 理解专家 + 500M 轻量 Action Expert 联合注意力），推理时绕过世界模型只走动作专家（74.3ms，−37% 延迟）。
- **数据规模反转**：动作解码器在 10万帧时连续回归胜、到 70M 帧时 AR token 反超（84.1→93.0）——**数据规模决定 tokenizer 选择**。
- **PDMS**：navtest **93.0**（AR + Best-of-N，单相机）、navtest v2 EPDMS 86.1；Query 动作头 90.1、FM 91.3。

> **我的判断**：W0 是"**世界模型 = 密集监督、推理只留动作专家**"的最高示范，也是"训练贵、推理便宜"哲学的代表作。

### 5. AlpaMayo-R1（NVIDIA）/ AlphaDrive / ReCogDrive / Senna-VLM

- **AlpaMayo-R1**（arXiv 2511.00088）：VLA + **Chain of Causation（CoC）** 决策锚定结构化推理（8 横向 × 8 纵向闭集决策 + 开放集关键组件）+ VLM 推理、2B 扩散 Expert 动作解码 + GRPO 三阶段训练。真车端到端 **99ms**，是少数过真车延迟关的 VLA。
- **AlphaDrive**（HUST × 地平线）：把规划改成**高维 meta-action 分类**（横向 × 纵向，3×4 类）而非轨迹回归，用四重乘积奖励 + GRPO 强化推理——两阶段（30k SFT 预热 → 110k GRPO）出现"一场景多解"的涌现式多模态规划。
- **ReCogDrive**（HUST × 小米）：VLM 认知（CoT）→ 扩散 planner → DiffGRPO 三阶训练；关键是把 VLM **最后隐状态**注入扩散规划器而非解码文本（7.8× 更快、35% 更平滑）。NAVSIM **PDMS 91.2**、Bench2Drive 闭环 DS 78.2。
- **Senna-VLM**：VLM 与 E2E 规划器**解耦协同**——VLM 说人话出高层决策，E2E 在该决策条件下生成精确轨迹，绕开 LVLM 不擅长数值预测的软肋。这是"VLM 决策 + 规划器执行"的低延迟原型。

### 6. 对 VLA 流派的反思

| 维度 | 优点 | 短板 |
|------|------|------|
| 语义推理 | 能读懂交规/场景语义 | 延迟高、自回归慢 |
| 监督 | 世界模型/CoT 密集监督 | 依赖数据规模、易过拟合 token |
| 泛化 | 语义跨域更强 | 闭环可靠度存疑 |
| 分数 | 单相机可达 90-93 PDMS | 推理成本 vs Scoring 有劣势 |

---

## 七、流派⑤：世界模型——"预测未来再决策"

世界模型派认为：**不能只看当前帧，要在脑子里把"未来几秒会发生什么"先模拟出来，再据此决策**。它是 navhard / OOD 场景的王者，也是 NAVSIM 从"开环刷分"走向"闭环鲁棒"的技术底层。

### 1. DreamerV3（MBRL 世界模型基石）

- **创新**：单一超参数适配 150+ 任务；RSSM 离散潜变量；在**潜空间想象 rollout** 里训练 actor-critic，首次从零拿到 Minecraft 钻石。

### 2. UniSim（交互式视频仿真世界模型）

- **创新**：把异构数据（互联网视频、机器人数据、仿真渲染）统一到"action→视频"，`p(o_t|h,a)` 建模，autoregressive 拼接历史帧；用它训练的策略可 **0-shot 迁移真实 robot**。**价值：世界模型 → 数据增强 / 闭环训练环境。**

### 3. DriveDreamer / DLWM / DriveLaW 等驾驶世界模型

- **DriveDreamer**：首个完全从真实驾驶数据训练的世界模型，两阶段 curriculum（先学"HDMap/3D box/text ↔ 外观"的结构约束，再学未来视频 + 动作预测），ControlNet 式零卷积注入多模态条件。
- **DLWM**（HKUST × 华为，CVPR 2026）：双潜世界模型——Gaussian-flow 引导（占位/4D 预测）+ ego 规划引导（轨迹可行性评估），用 3D 语义 Gaussians 做场景表征，L2 ↓16%、碰撞 ↓40%。
- **DriveLaW**（小米）：video-in + Flow Matching 世界模型 + DiT 动作，用 **Noise Reinjection** 恢复扩散丢失的结构/时序一致性；视频预训练 7.6M 帧把 PDMS 从 85.9 拉到 89.1。

### 4. Cosmos（NVIDIA 世界基础模型）

- **创新**：**MoT（Mixture of Transformers）双塔**——Reasoner（因果自回归）+ Generator（双向扩散 DiT）共享层，通过各自 QKV/FFN 投影 + 共享跨模态注意力协同；五模态（语言/图像/视频/音频/动作）统一；物理对齐后训练（PAIBench 显示，用其生成驾驶场景训练的策略闭环通过率提升）。

### 5. 驾驶侧世界模型的应用形态

| 模型 | 世界模型用途 | 规划方式 | NAVSIM / PDMS |
|------|------------|----------|---------------|
| **JEPA-DRIVE**（博客课题组自研） | JEPA 自监督 + cycle-energy 打分（生成-打分之理，512 路径 × 128 速度 = 65,536 候选轨迹） | reasoning-by-simulation 生成 + 打分 | navtest selected_pdm 0.8839 |
| **WoTE**（ICCV 2025） | BEV 空间世界模型给每条候选轨迹 rollout 未来 BEV 状态 | 奖励模型打分挑最高 | 88.3 PDMS |
| **DriveFuture**（arXiv 2605.09701） | **未来潜状态条件化**（训练给 GT 未来、推理用预测未来接替，砍掉 train/inference gap） | 条件化 Diffusion 规划 | **navhard 55.5（#1）**、navtest PDMS 90.7 / EPDMS 89.9 |
| **Uni-World VLA**（arXiv 2603.27287） | 逐帧交错生成未来帧 + 动作，世界建模与控制闭环 | 自回归交错规划 | 89.4 PDMS（纯单目） |
| **ExploreVLA**（arXiv 2604.02714） | 未来 RGB + Depth 密集世界建模，预测不确定性当内在探索奖励 | 安全门控 GRPO | 93.7 PDMS（arXiv） |
| **World4Drive** | 物理隐世界模型 + 多模态驾驶意图 | 世界模型想象 + 打分选优 | NavSim 闭环 PDMS 85.1 |

> 这条线索最核心的思想是 **DriveFuture 的"未来条件化"**：未来状态不是预测输出，而是**决策条件**——训练时给 GT 未来、推理时换预测未来接替，训练/推理共用同一"条件化"接口，gap 小、泛化强。这正是世界模型能在 navhard（观测偏移）上反超人类的原因。

---

## 八、决策/规划的"重头戏"：GRPO 与 RL 后训练

一个贯穿博客始终的大趋势：**模仿学习已到极限，RL 后训练成为标配**。AlpaMayo-R1、AutoVLA、AlphaDrive、DiffusionDriveV2、ExploreVLA、NoRD、ReCogDrive、DriveTeach-VLA、LaST-VLA 全用 GRPO 或其变体，只是 reward 不同（PDMS、meta-action F1、VLM 裁判、世界模型能量、规则奖励）。

- **Flow-GRPO**（Tencent ARC，arXiv 2505.05470）：ODE→SDE（Girsanov）让 flow 的 log-prob 可解；Denoising Reduction（短时间窗）省内存 10-20×；GRPO 组相对 advantage（无 critic，省内存 40-50%）；LoRA 更新——把 flow/diffusion 生成器直接对任意 reward 训练，是世界模型闭环的 **RL 底层**。
- **NoRD**（弱策略修正）：指出标准 GRPO 在弱策略上会因**难度偏差**（Difficulty Bias）失效，提出 **Dr.GRPO** 后从 0.67% 提升扭转为 +11.68%，且只用 60% 数据 + 零推理标注。
- **DriveTeach-VLA**（ECCV 2026）：用 GRPO + PDMS reward 做 VLA 后训练，2D 轨迹引导提示（TGP）把 PDMS 从 83.1（文本 CoT）拉到 91.5。
- **LaST-VLA**（ICML 2026）：物理锚定的 latent CoT（Cosmos 动力学 + VGGT 几何双教师蒸馏）+ GRPO，8B 达 PDMS 91.3 / EPDMS 87.1。

> **我的判断**：GRPO 不是万能药——reward hacking、开环仿真的 gap、稀疏 reward 都是真实风险（"navsim-pdms-90"里也实测了"纯 RL + 扩散样本效率低"）。它更适合做安全**后训练段**的刀具，而不是主学习器；且 reward 设计远比算法本身重要。

---

## 九、PDMS / EPDMS 分数总表

> 区分来源：**[官方]** = 官方排行榜；**[宣称]** = arXiv 自报。口径差异（navtest/navval、backbone、相机数、LiDAR）见各模型原文，横向对比需谨慎。

### NAVSIM v1（PDMS）

| 模型 | 流派 | PDMS | 来源 | 备注 |
|------|------|:----:|:----:|------|
| TransDiffuser | 扩散 | **94.85** | 宣称 | ResNet-34 + LiDAR |
| CLOVER | Scoring | **94.5** | 官方 #1 | DINOv2 冻结 + 闭式蒸馏 |
| ChainFlow-VLA | VLA + Flow | 94.8 | 宣称 | Chain+DiT+VLM 三阶段 |
| Drive-JEPA (ViT-L) | 世界模型 + JEPA | 93.7 | 宣称 | V-JEPA 预训练 |
| DriveSuprim (EVA-L) | Scoring | 93.5 | 宣称 | coarse-to-fine 选择 |
| ExploreVLA | 世界模型 + RL | 93.7 | 宣称 | 未来 RGB+Depth 密集监督 |
| DriveVLA-W0 | VLA + 世界模型 | 93.0 | 宣称 | AR 动作头 + 单相机 |
| AutoVLA (+anchor) | VLA | 92.1 | 宣称 | RFT/GRPO + Best-of-N |
| SparseDriveV2 | Scoring | 92.0 | 官方榜 | factorized vocab |
| iPad | 显式（proposal 迭代） | 91.7 | 宣称 | ProFormer 迭代自修正 |
| DiffusionDrive V2 | 扩散 + GRPO | 91.2 | 宣称 | Intra/Inter-Anchor GRPO |
| ReCogDrive | VLA + 扩散 | 91.2 | 宣称 | VLM 隐状态注入扩散 |
| LaST-VLA (8B) | VLA + latent CoT | 91.3 | 宣称 | Cosmos+VGGT 双教师 |
| Hydra-SE | Scoring | 91.87 | 官方榜 | 簇熵不确定性 |
| DriveTeach-VLA | VLA + GRPO | 91.5 | 宣称 | 2D-TGP 空间提示 |
| Hydra-MDP | Scoring | 91.26 | 官方榜 | 多目标蒸馏 |
| DriveFuture | 世界模型 | 90.7 | 宣称 | 未来条件化 |
| CoWorld-VLA | VLA + latent CoT | 89.8 | 宣称 | 四专家 token |
| AutoVLA | VLA | 89.1 | 宣称 | 双思维模式 |
| Uni-World VLA | VLA + 世界模型 | 89.4 | 宣称 | 交错生成 + 单目深度 |
| DriveLaW | 世界模型 + Flow | 89.1 | 宣称 | Noise Reinjection |
| DiffusionDrive | 扩散 | 88.1 | 宣称 | 截断扩散 |
| WoTE | 世界模型 | 88.3 | 宣称 | BEV rollout 打分 |
| LTF | 显式基线 | 83.8 | 官方基线 | TransFuser latent |

### NAVSIM v2（EPDMS）

| 模型 | 流派 | EPDMS | 来源 |
|------|------|:----:|:----:|
| CLOVER | Scoring | **90.4** | 官方 |
| Metis | WAM（世界动作模型） | 90.3 | 官方 |
| AutoDrive-P³ | VLA + RL | 89.9 | 官方 |
| DriveFuture | 世界模型 | 89.9 | 宣称 |
| EponaV2 | 世界模型 + Flow | 88.9 | 官方（perception-free） |
| Drive-JEPA | 世界模型 + JEPA | 87.8 | 宣称 |
| SparseDriveV2 | Scoring | 87.35 | 官方 |
| DriveVLA-W0 | VLA + 世界模型 | 86.1 | 宣称 |
| DiffusionDrive V2 | 扩散 + GRPO | 85.5 | 宣称 |

### NAVSIM v2 navhard（EPDMS，两阶段 3DGS 偏移场景）

| 模型 | 流派 | EPDMS | 来源 |
|------|------|:----:|:----:|
| DriveFuture | 世界模型 | **55.5** | 宣称 #1（人类专家 51.3） |
| DrivoR | 显式 Transformer | 54.6 | 宣称 |
| SimScale | 数据增强 | 53.2 | 宣称 |
| GuideFlow | 扩散 | 51.5 | 宣称 |
| GTRS | Scoring | 49.4 | 宣称 |
| LTF | 显式基线 | 24.4 | 官方基线 |

> **重要提醒**：这些分数来自不同来源（官方 vs 宣称、相机 vs LiDAR、backbone 级别），只能做"数量级 + 趋势"参考，跨行对比 `±2` 分没有意义。**最可信的是官方榜单同列直接对比的部分**（如 CLOVER 94.5 vs SparseDriveV2 92.0 vs Hydra-MDP 91.26）。

---

## 十、我的个人思考（核心部分）

### 思考 1：感知架构的"去稠密化"是不可逆的

从 UniAD 的稠密 BEV → VAD 的向量化 → SparseDrive 的完全稀疏 → Sparse4D 的稀疏 query → DriveVLA 的视觉 token，本质是**"用语义换算力"**：不被需要的稠密栅格都是浪费。到今天，稀疏 query / 视觉 token 几乎成了所有高分模型的一致选择。我的判断：**感知层不会回到稠密 BEV，只会往更"因果、更稀疏、更世界级"的方向走。**

### 思考 2：规划输出经历了三个时代

- **显式轨迹回归**（UniAD/VAD/SparseDrive）——简单，但被多模态问题绑定（mode averaging）。
- **Scoring + 多候选**（CLOVER/Hydra/GTRS）——靠"覆盖 + 挑选 + 评估器对齐"实现了**规则化评测下的最强**，是目前 leaderboard 最稳的路线。
- **扩散生成**（DiffusionDrive/TransDiffuser）与**世界模型**（DriveFuture）——提供多样性与分布外鲁棒，但要么需要 RL 后训补质量，要么计算重。

**我的结论**：短中期，"**Scoring + 扩散/Flow 生成器 + RL 后训**"会融合——扩散出多样候选、打分器挑、RL 拉质量下限（CLOVER 的蒸馏和 DiffusionDriveV2 的 GRPO 其实已经在朝这个方向走）。

### 思考 3：VLA / 世界模型的真正位置

VLA 在**交规语义、场景推理、长尾理解**上无出其右，但在 NAVSIM 这种"固定规则打分"上反而被 Scoring 压一头——因为打分函数本身就是"符合规则"。而**真实世界里"规则之外"的合理、长尾、安全避险**，正是 VLA/世界模型的用武之地。

> 当前的 leaderboard 本质是 **"规则匹配的斯巴达式竞争"**，世界模型/VLA 的"想象力"在 navhard / OOD 上才真正显现。**评价必须同时看 PDMS 和 navhard/OOD，只看 PDMS 会让技术路线选择失焦。**

### 思考 4：GRPO 是方向盘而不是引擎

GRPO 系方法解决的是"**模仿学习打不开专家分布之外的高质量策略**"。但它不是"开关级"的工具，关键在 reward 设计——用 PDMS（规则分数）还是真实安全（闭环/世界模型），决定了是安全 RL 还是在刷模拟器。我认为**真正可信的 RL 是"世界模型提供的 reward + sim-to-real 验证闭环"**，这恰好是世界模型学派的终局位置。

### 思考 5：给课题组 / 研究者的建议

1. **想上线**：Scoring（冻结 DINOv2 + Hydra 式多目标蒸馏）是"规则化评测下最稳"的路径，可配合一步扩散生成器 + RL 后训。
2. **想做有洞见的研究**：世界模型/VLA 的"泛化 + RL 对齐"方向最有长期价值，但要做好理论 + 开放评测，别只刷 navtest。
3. **别迷信单数字**：同榜对标比 arXiv 自报 + 单 backbone 对比更可信。**PDMS 90+ 只代表"规则遵守得好"，不代表真的会开车**——开环 vs 闭环、规则 vs 语义，这两对矛盾始终存在。

---

## 结语

端到端自动驾驶经过这几年，已经从"能不能端到端"进化成"感知表征、规划范式、学习策略如何组织"的设计问题。五大流派不是终点，而是一张不断互相借力的网。希望这篇综述能把博客里散落的精读串成一张可以随时"拉起来"的架构图。

*本文基于本博客全部模型精读文章的横向整合，PDMS/EPDMS 数据截至 2026 年 8 月。单块对标时以官方排行榜为准，arXiv 数字仅供参考。*