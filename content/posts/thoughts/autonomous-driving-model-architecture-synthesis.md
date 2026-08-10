---
title: "端到端自动驾驶模型架构全解：从感知到规划的六大范式与 80+ 模型逐篇拆解"
date: 2026-08-10
draft: false
description: "把博客全部自动驾驶模型精读收进同一坐标系：显式端到端 / Scoring / 扩散与Flow / VLA / 世界模型 / 数据评测基建六大范式，80+ 模型逐篇给出感知表征、架构组成、训练方式与 PDMS/EPDMS 分数，最后给出我对端到端演进方向的个人思考。"
categories: ["个人思考"]
tags: ["端到端自动驾驶", "UniAD", "VAD", "SparseDrive", "DiffusionDrive", "CLOVER", "VLA", "世界模型", "GRPO", "NAVSIM", "PDMS", "架构综述", "TransFuser", "DriveTransformer", "JEPA", "Flow Matching"]
math: true
---

## 一、为什么写这篇文章

本博客陆续精读了 **100+ 篇**自动驾驶与机器人模型论文——从感知基座（BEVFormer/Sparse4D），到显式端到端（UniAD/VAD/SparseDrive），到打分式（CLOVER/Hydra-MDP/GTRS），到扩散与 Flow 生成（DiffusionDrive/TransDiffuser/GoalFlow），再到 VLA 大模型（DriveVLM/EMMA/AutoVLA/LinkVLA）与世界模型（Cosmos/DriveFuture/WoTE）。单独看是逐篇精读，放在一起，端到端自动驾驶在 **感知 → 预测 → 决策 → 规划** 四个架构维度上的演进脉络其实非常清晰。

这篇文章把博客里**所有**精读过的模型收进同一个坐标系，回答三个问题：

1. **感知**：从稠密 BEV 到稀疏 query 再到纯视觉 token、几何寄存器，一路在换什么？
2. **预测/决策/规划**：从显式管线到打分排序、扩散生成、Flow 整流、LLM 推理、世界模型 roll-out，谁在取代谁？谁融合了谁？
3. **谁真正有效**：把彼此在 NAVSIM 上的 PDMS/EPDMS 分数和架构创新对照起来，看看分数到底来自技巧还是来自架构。

> 本文每个模型给"**流派 → 核心创新 → 架构拆解（骨干/模块/输出/训练） → 代表分数**"四个维度，尽量讲清楚架构；分数严格区分**官方排行榜**与 **arXiv 自报**，口径不一致处会注明。机器人侧的 VLA 模型（RT 系列、π0）作为同源背景单列一节——它们定义了「动作 token」「动作分块」「Flow 动作头」这三个被驾驶侧反复借用的组件。

---

## 二、先立坐标系：六大流派一张图

按「感知表征 × 规划输出方式」两个维度，我把博客里的方法归为六大流派：

| 流派 | 代表模型 | 核心思想 | 感知表征 | 规划输出 |
|------|---------|---------|---------|---------|
| **① 显式端到端管线** | UniAD、VAD、VADv2、SparseDrive、PARA-Drive、DriveTransformer、TransFuser | 感知/预测/规划在同一个可训练网络里串起来 | 稠密 BEV / 稀疏 token / 寄存器 | 单条/多条轨迹回归 |
| **② 评分排序式 (Scoring)** | CLOVER、Hydra-MDP、GTRS、SparseDriveV2、TOAD、GoalFlow | 大量生成候选，再学打分器（或用搜索/流）选出最优 | 稀疏特征 / 轨迹词汇表 / 目标点 | 从候选集合中选 Top-1 |
| **③ 扩散/Flow 生成式** | DiffusionDrive、DiffusionPlanner、TransDiffuser、SafeDiffuser、FeaXDrive、Gen-Drive | 把规划当成从噪声里生成（DDPM / Flow Matching / 整流流） | 稀疏感知 / BEV / 图像 | 多模态轨迹生成 |
| **④ VLA 大模型式** | DriveVLM、EMMA、AutoVLA、LinkVLA、OneVL、LaST-VLA、WCog-VLA | LLM/VLM 做感知+推理+决策，语言或 token 输出动作 | 视觉 token / 稀疏 query / 隐 token | 语言决策 / 轨迹 token |
| **⑤ 世界模型式** | Cosmos、DriveFuture、WoTE、DreamerV3、DLWM、RAW2Drive、ReWorld | 建模环境动力学/未来状态，以未来为条件或打分规划 | 隐潜 / 视频 token / BEV 未来帧 | 条件化/打分/rollout 规划 |
| **⑥ 数据与评测基建** | Bench2Drive、CARLA、K-Risk、MOSAIC、重卡数据集族 | 提供闭环基准、数据选择、仿真、安全分析 | — | — |

> 一句话直觉：**① 是"手拉手串行提线木偶"，② 是"多打几遍再挑"，③ 是"从噪声里一遍遍画"，④ 是"读得懂交规再动手"，⑤ 是"先在脑内模拟未来再决定"，⑥ 是"让前面所有流派有地方比"**。六大流派不是互相淘汰，而是逐层叠加——今天最强的系统几乎都是 ②③⑤ 的混合。

---

## 三、流派①：显式端到端管线——从 BEV 到稀疏 token 到寄存器

这类方法把感知、预测、规划放在**同一个网络**里联合训练，是端到端的"正统"。它们的分界线是：**用什么中间表征承载场景理解**。

### 感知基座（两个源头）

#### 1. BEVFormer——稠密 BEV 的"标准定义"（ECCV 2022）

- **创新**：可学习 BEV grid query + **空间交叉注意力（SCA）**（deformable 采样多相机特征）+ **时序自注意力（TSA）**（带 ego 运动补偿）——第一次让稠密 BEV 能跨视角、跨时序"被感知到"。nuScenes NDS 56.9%。
- **架构**：BEV 网格 200×200，每个 query 沿射线在图像特征上可变形采样（K=4 参考点）聚合 → 时空融合；是 UniAD/VAD/PARA-Drive 等一切 BEV 链路的感知原型。

#### 2. Sparse4D——全稀疏的"另一条路"

- **创新**：完全不建 BEV，稀疏 anchor + **4D keypoint 采样**（一个采样动作同时做空间+时序融合），随时间循环压缩；推理时给 instance 分配 ID 就能当 tracker（V3 AMOTA 67.7%）。
- **架构**：instance query → 稀疏采样 → 循环优化；是 SparseDrive 系列"稀疏化"思想的感知层源头。

> **判断**：BEVFormer vs Sparse4D = **算力换语义** vs **语义换算力**。显式管线的拉锯史，就是这两极之间的钟摆。

---

### 完整显式管线（7 个代表）

#### 1. UniAD——显式端到端的开山之作（CVPR 2023 最佳论文）

- **架构**：检测（TrackFormer）→ 建图（MapFormer）→ **运动预测（MotionFormer，每 agent K=6 条轨迹）** → 占位（OccFormer）→ 规划，五任务挂在同一个 BEVFormer 骨干上，通过**统一 query 接口**在任务间传特征（特征级、可微、不丢信息）。规划 head 用 ego query 交叉注意力到 BEV+motion+occupancy，加"碰撞损失 + 推理时 occupancy 梯度优化"。
- **训练**：两阶段（感知预训练 → 联合微调）；约 7-10 FPS。
- **分数**：nuScenes 开环 L2 0.44/0.99/1.71、碰撞率 0.56（其意义不在分数而在"规划第一次成为全网络集体目标"）。

#### 2. VAD——向量化的"轻量版 UniAD"

- **架构**：仍用 BEV 编码器，但输出**向量化** 3D box 与向量化地图（去掉稠密栅格）；Motion Head 用模式嵌入 + agent-map 交叉注意力出多模态轨迹；规划加**三个可微向量化约束**（ego-agent 碰撞、ego-边界、ego-车道方向）+ 离线专家（Hybrid A*）**规划 KD**。
- **分数**：nuScenes L2 0.72、碰撞 0.22%；比 UniAD 快 2.5-9.3×。
- **VADv2**（在博客里单独精读）：把规划升级为 **FPS 采样 4096 条完整轨迹词表 + 概率场（sigmoid 多标签）**建模 P(动作|场景)，**conflict loss** 内嵌碰撞/越界先验，坐标用 NeRF 式高频位置编码；还能 top-K + 规则 wrapper + 人工接管混合模式。CARLA Town05 DS 85.1、Bench2Drive **76.15**、NAVSIM 概率规划 87.7；词表 256→1024→4096 一路涨（70.2→78.1→85.1），16384 饱和——**它是 Scoring 意识最早长在"显式管线"身上的模型**。

#### 3. SparseDrive——"完全稀疏"的显式管线

- **架构**：砍掉稠密 BEV 编码层，instance query 直接与多视角图像特征 deformable 交互；感知-预测-规划**并行双支**（motion / planning 独立输出），打断 UniAD 的串行误差级联；规划 branch 直接回归未来 waypoints + 数据驱动 anchor + 碰撞/舒适/合规打分。计算量与**对象数成正比**而非 BEV 分辨率平方。
- **分数**：nuScenes 开环第一梯队。

#### 4. PARA-Drive——完全并行的模块化端到端（CVPR 2024）

- **架构**：四模块（地图/运动/占用/规划）在**同一 200×200 BEV 上完全并行**、模块间零直接连接，共享 BEVFormer 编码（R50/R101，6 视图 256×704）。地图=MapTRv2、运动≈VAD（~100 agent query）、占用=OccNet 风格、规划=plan query 交叉注意力 + CANbus/指令入 → 未来 3s 6 waypoint。联合训练，推理时模块可按需开关。
- **关键发现**：地图模块价值在**监督信号**而非输出直用；运动（轨迹）与占用（栅格）互补——这是最早系统化回答"端到端里到底几个模块、怎么连"的实验设计。
- **分数**：nuScenes 开环 Coll 0.19、L2 0.5885（PARA-Drive+）；推理 135ms 全量；复杂场景 Coll 0.05。

#### 5. DriveTransformer——告别 BEV，拥抱稀疏 Token 并行（ICLR 2025）

- **架构**：**彻底抛弃稠密 BEV**，用三种注意力统一所有交互：**SCA**（sensor token 与图像特征）、**TSA**（任务 query 间并行自注意力）、**TCA**（时序交叉注意力，FIFO 队列+运动补偿）。sensor token 用 3D PE（PETRv2 射线采样），Agent token ≈300、Map token ≈100（DAB-DETR 三分）。运动预测在**局部坐标系**输出（minADE 全局 2.68→局部 1.34）；规划是 6 模式高斯混合 + winner-take-all。单阶段端到端，batch size 12 能跑（UniAD 只有 1）。
- **分数**：Bench2Drive 闭环 DS **63.46**/SR 35.01（UniAD 45.81/VAD 42.35）；nuScenes Avg L2 0.40；对标定误差鲁棒（VAD 掉 28%，DT 仅 5.9%）；规模 47M→646M 时 DS 45→68。

#### 6. TransFuser——用 Transformer 做传感器融合（NeurIPS 2022）

- **架构**：相机图像（400×300）+ LiDAR BEV（256×256，0.125m/pixel，32m）在 **4 层 CNN 尺度上做自注意力密集融合**（每层展平 token → 拼接 → 自注意力 → 残差），取代 Late/Geometric 融合；全局向量 → MLP + **GRU 自回归预测差分 waypoint**（4 个，0.5s 间隔）→ PID 控制。4 个辅助任务（深度/语义/HD 地图/车辆检测）。
- **历史地位**：CARLA 最早证明"逐级逐步融合 > 后融合"，是 DiffusionDrive/TransDiffuser/GoalFlow 等方法的**通用感知骨干被反复冻结复用的对象**。

#### 7. DrivoR——寄存器 Token 的高效 E2E（"Driving on Registers"）

- **架构**：把实值 **寄存器 token**（register token）引入 E2E 感知-规划，时空寄存器聚合多帧跨分辨率特征，网络更浅更快、鲁棒性更好；是"稀疏寄存器表征"的代表，也是 TOAD 论文里被用作基规划器的对象。NAVSIM-v2 EPDMS 54.6（在 TOAD 精读中被反复引用），v1 PDMS 90+ 级。

> **判断**：显式管线这条线的主线是**"表征去稠密化"**（BEV 200×200 → 向量 → 稀疏 query → 局部系轨迹 → 寄存器 token）和**"模块并行化"**（UniAD 串行 → PARA-Drive/DriveTransformer 全并行）。VADv2 的出现预示了这条线向 Scoring 的靠拢——显式管子最终把"规划"交给了"词表+打分"。

---

## 四、流派②：Scoring-based——"多生成几遍，再打分挑"

核心想法：**不直接"开"，而是"先想一万个开法，再让打分器挑"**。在 NAVSIM 这种规则化、确定性的评测下，这是最稳的路线。

### 1. CLOVER——Scoring 流派天花板（NAVSIM v1 官方榜 #1，94.5 PDMS）

- **架构**：冻结 **DINOv2 ViT-S** 提视觉特征 → 轻量生成器（Transformer decoder）出 **N 条候选轨迹** → **Scorer**（Cross-Attention，候选↔场景特征）对每条输出子分数 → 选 Top-1。打分器的监督信号是**评估器在候选上算出的真实分数**——它直接在学习排行榜的判分函数本身。
- **两个关键机制**：① **伪专家覆盖训练**——从可解释动作族构造超大候选池 + FPS 采样保证覆盖，打分器给"伪专家集合"打高分、给池外打低分；② **保守闭环自蒸馏**——把打分器的完整排序偏好（不只是选中的一条）回传给生成器，形成"生成-打分-蒸馏"的在线自我改进闭合回路。
- **理论保障**：论证"打分器选中目标在真实评估器下显著优于当前分布时，保守蒸馏必提升高分区概率质量"。
- **分数**：navtest PDMS **94.5**（官方 #1）、navtest v2 EPDMS 90.4、NavHard 48.3。

### 2. Hydra-MDP / Hydra-SE——NVIDIA 的多目标蒸馏（NAVSIM 官方榜 91.26 / 91.87）

- **架构**：固定 **anchor 词表**（K=8192 条预定义轨迹）+ **多头解码器**（每个 head 出一个轨迹簇的残差）→ **ScoreNet**（轻量 MLP）打分。训练同时吃两类老师：**human 教师**（模仿真实轨迹）与 **rule 教师组**（碰撞/可行驶/舒适等规则评估器各自打分），多头打分器分别向各子分蒸馏；推理时去掉 rule 教师，按权重聚合成总分。
- **工程价值**：几乎成为后续所有 Scoring 方法的对照基点；3-5 模型集成可再 +2-3 PDMS。

### 3. GTRS——通用轨迹评分（CVPR25 自动驾驶挑战赛 E2E 冠军）

- **架构**：① **扩散策略（DP）**生成动态细粒度候选（BEV 图像 backbone → DiT 条件生成，DDPM 100 步出 100 条）；② **超密集训练词表 16,384 + 词表 Dropout**——训练时随机删一半词表，迫使打分器学习"轨迹形状→分数"的**泛化映射**而非死记词表，推理时用更小的 8192 词表故意制造分布不匹配逼泛化；③ **传感器旋转增强 + top-k 细化自蒸馏**（EMA 教师软标签拦截修正）。
- **分数**：NavHard 榜 EPDMS **49.4**（6 模型集成，逼近依赖真值感知的 PDM-Closed 51.3）；词表 Dropout 是单点最大增益（41.7→43.4）。

### 4. SparseDriveV2——Scoring 完成体（NAVSIM 官方 92.0 / v2 EPDMS 87.35）

- **架构**：把"整条轨迹"分解为**几何路径(path) × 速度剖面(velocity)** 两个词表（路径 1024 条、速度 256 条），组合出 **262,144（26 万）条轨迹词表**，记忆开销只等于两者之和；**两级评分**——先对 path / velocity 各自粗打分（O(Np+Nv)），保留 top-K 后再对组合轨迹做细粒度联合打分（O(K)）。打分计算量与词表规模**解耦**。
- **分数**：navtest PDMS 92.0（官方）、v2 EPDMS 87.35；词表 scaling 1024→16384 单调涨不饱和。

### 5. iPad——Iterative Proposal-centric E2E（NAVSIM v1 91.7）

- **架构**：**proposal-centric + 多轮迭代规划**——Scene Encoder（LSS 生成图像特征）→ **ProFormer**（proposal 当 query，对 proposal-centric 图像特征做 deformable attention，逐轮迭代自修正，K 轮）→ 打分选优。proposal 数/迭代数/数据量都呈对数增长 PDM。
- **分数**：NAVSIM v1 camera（R34）PDMS **91.7**（NC 98.6 / DAC 98.3 / TTC 94.9 / EP 88.0）。

### 6. TOAD——Test-Time Trajectory Optimization（2026，把评分器当奖励函数搜索）

- **架构**：把端到端规划器中已训练好的**轨迹评分器重新解释为"学习到的轨迹奖励函数"**，测试时用 **CEM**（Cross-Entropy Method）在连续控制空间里搜索最大化该奖励的轨迹：先用运动学自行车模型把搜索空间压到控制空间 u=(加速度, 角速度)（保证动力学可行 + 维度压缩），目标 J(u)=评分器(BM(u)) − λ·舒适正则 − λ·锚点正则（warm-start 自基规划器输出）。CEM M=64 采样、E=8 精英、迭代 K 轮。
- **关键发现**：固定词表类的打分器（Hydra-MDP/GTRS）在测试时搜索**反而掉分**（分布内过拟合）；on-the-fly 类（DrivoR/iPad）则一致提升。
- **分数**：NAVSIM-v1 PDMS DrivoR+TOAD **94.7**（逼近 Human 94.8）；v2 EPDMS 让 iPad 34.7→49.8、DrivoR 54.6→**56.3**（逼近 PDM-Closed 56.6）；延迟仅 +1.9~2.7ms。

### 7. GoalFlow——目标点词表 + Flow 一步生成（NAVSIM 90.3）

- **架构**：用**密集"目标点词表" + 学习式评分**先精确选出短期终点（约束在可行驶区域内），再用**整流流（Rectified Flow）单步生成**多模态轨迹 v_θ(x_t, c, t)，解决"扩散发散无边界 vs 锚点约束太强"的两难。感知用被冻结的 TransFuser 骨干；**影子轨迹**机制对冲目标点选错的风险。
- **分数**：NAVSIM PDMS **90.3**；**1 步去噪**击败 DiffusionDrive 动辄 10+ 步；DAC 显著领先（目标点约束保证不出路面）。它是"FM 生成器 + 打分器"范式的直接先例。

### 8. MOSAIC——数据选择（数据侧 Scoring 方法论）

- **架构**：**缩放感知数据选择**三阶段：聚类分域 → 逐域拟合神经缩放定律 ΔÛ(n) → 贪心选边际增益最大的域。模型无关（在 Hydra-MDP 上验证）。
- **分数**：NAVSIM 用 ~42% 数据达全量 99.6% 性能（EPDMS 86.73 vs 全量 87.12）。

> **判断**：Scoring 流派的演进是清晰的：**CLOVER（生成器+打分器闭环蒸馏）→ Hydra-MDP（多目标多头打分）→ GTRS（词表 Dropout 逼泛化）→ SparseDriveV2（因子化超密词表）→ TOAD（把打分器当奖励做测试时搜索）→ GoalFlow（FM 一步生成+目标点）**。它们合起来回答了"打分器必须既想得多、又想得准、又想得快"。

---

## 五、流派③：扩散 / Flow 生成式规划

### 1. 奠基：Diffusion Policy（RSS 2023，机器人侧方法论母体）

- **架构**：把动作生成变成条件去噪过程（DDPM），建模**完整多模态分布**，避免回归的 mode averaging；receding horizon（生成 T 步、执行前 K 步、重规划）；推理时 DDIM 100→10→5 步；可视化 + FiLM 注入。**它是整个扩散生成流派的母体**。

### 2. Diffusion Planner（ICLR 2025 Oral）

- **架构**：DiT + MLP-Mixer **联合生成 ego 与 M 个邻居的未来轨迹**（预测与规划合一，交互自然涌现）；推理时用**训练免费的 DPS（Diffusion Posterior Sampling）分类器引导**注入碰撞/可行驶/车辆动力学能量约束——无需重训，一个模型多种驾驶风格。nuPlan 闭环 SOTA。

### 3. DiffusionDrive V1/V2（CVPR 2025 及其 RL 续作）

- **V1 架构**：**截断扩散 + 轨迹 anchor 化**——从 20 个 k-means 聚类 anchor 附近加噪声（而非标准高斯），去噪压缩到 **2 步**；多候选 + 成本选择。PDMS 88.1。
- **V2 架构**：在生成器上做 **Intra-Anchor GRPO**（组内比较保多模态稳定）+ **Inter-Anchor Truncated GRPO**（用碰撞做全局安全先验、截断负优势）+ 两级 Mode Selector；尺度自适应乘性探索噪声保轨迹几何平滑。PDMS 91.2、v2 EPDMS 85.5。

### 4. TransDiffuser——去相关多模态表示（NAVSIM 94.85）

- **架构**：指出扩散轨迹生成模式坍缩的根源在**"条件特征的信息瓶颈"**而非生成过程；多模态融合特征后施加**去相关正则化**（强制特征相关矩阵对角化），让条件信号覆盖更多潜空间，无需任何锚点/场景先验。Scene Encoder 冻结 TransFuser 骨干，Motion Encoder 编码历史轨迹+速度，Decoder = DDPM（T=10）预测相邻位移差分；推理 30 条候选 → 运动学过滤 → PDMS 评分选优。
- **分数**：NAVSIM PDMS **94.9（94.85）**，R34+LiDAR，无锚点 SOTA。训练只要 2h@4×H20。

### 5. SafeDiffuser——CBF 安全引导的扩散规划（ICLR 2025）

- **架构**：把**控制障碍函数（CBF）改造成"有限时间扩散不变性"**，嵌入去噪每一步：每步去噪叠加安全梯度（求解最小修正 QP），给"可证明安全"而非软约束。三种变体：RoS（严格不变）、ReS（宽松）、TVS（时间自适应）。推理只增加 20-30% 时间。
- **分数**：Maze2D 约束满足 99.4%（Diffuser 58.2%）；窄通道 98.7% vs 12.5%。

### 6. FeaXDrive——轨迹中心扩散 + 可行性感知 GRPO

- **架构**：指出 noise-centric 扩散与轨迹可行性空间"对齐断裂"，改为**直接预测 clean trajectory x₀**（trajectory-centric）；三阶段覆盖可行性：自适应曲率正则训练（曲率 κ 越界才激活惩罚 L_cur）+ **可行驶区域 SDF 引导推理**（HD map→SDF，footprint 级 4 角点采样 + softplus 屏障）+ **可行性感知 GRPO** 后训练（奖励 R = PDMS + 曲率偏好）。VLM InternVL3-2B 提多视图条件。
- **分数**：NAVSIM IL PDMS 88.7 → FA-GRPO **90.0**；曲率违反率 IL 8.59%→**0.88%**（DiffusionDrive 8.59%）。它把"可行性"变成了可微分可训练的目标。

### 7. Gen-Drive——生成-评估范式（"生成未来 + 学奖励 + RL 精调"）

- **架构**：三组件：① **条件行为扩散生成器**——条件于自车候选动作联合生成多智能体未来（what-if）；② **场景评估器 = 学习型 reward 模型**——用 VLM 辅助采集成对偏好数据训练 Bradley-Terry 打分（大幅降人工标注）；③ **RL 微调**——把 reward 反向传播到生成过程。工作流：候选采样 K → 每候选生成 M 个未来 → 打分 → 选优 → 离线 RL 精调。
- **结论**：nuPlan 闭环上"生成+评估"优于纯模仿扩散；**学习型 reward 在评估与微调中均优于人工设计 reward**——这是"世界模型给 reward"思路的早期验证。

### 8. 扩散流派横向对比

| 方法 | 生成方式 | 多样性保证 | RL | 分数 |
|------|---------|-----------|-----|------|
| Diffusion Policy | DDPM 动作生成 | 多模态采样 | ✗ | 机器人 +46.9% |
| Diffusion Planner | 联合 ego+邻居去噪 + 训练-free 引导 | 多模态采样 | ✗ | nuPlan 闭环 SOTA |
| DiffusionDrive V2 | 20 锚点截断扩散 | 锚点覆盖 | GRPO | 88.1 / **91.2** |
| TransDiffuser | 去相关特征 + 无锚点 DDPM | 特征去相关 | ✗ | **94.85** |
| SafeDiffuser | DDPM + 每步 CBF-QP | 安全约束 | ✗ | 约束满足 99.4% |
| FeaXDrive | 轨迹中心 DiT | SDF 引导 | FA-GRPO | **90.0** |
| GoalFlow | Rectified Flow 单步 | 目标点词表 | ✗ | **90.3** |
| Gen-Drive | 行为扩散 what-if | 学习 reward | RL | nuPlan 定性优于 IL |

---

## 六、流派④：VLA 大模型式——"读得懂交规再动手"

VLA（Vision-Language-Action）把 LLM/VLM 当成驾驶"大脑"：感知与推理用语言/视觉 token，输出或用语言描述决策、或直接吐轨迹 token。这个流派内部又裂成两支——**显式推理**（慢思考，CoT 可解释）与**隐式/端到端 token**（快思考，把推理压缩进潜空间）。

### 1. DriveVLM 家族（清华 × NVIDIA，最早把 LLM 端进驾驶的体系）

#### DriveVLM（CVPR 2025）
- **架构**：三层 CoT 管线——**场景描述**（把多视角图像转成语言标注）→**场景感知**（密集语言描述：位置/速度/形状/颜色 + **3D grounding** 把语言对象锚到空间）→**驾驶推理与规划**（分"能否/如何/何时"决策 + "在哪/何时/怎样"动作）。LLM 推理后接算法规划器做硬件级控制。
- **DriveVLM-D**：把重 VLM 蒸馏成轻量模型，实现可落地的效率；**dual-branch** 结构——VLM 慢速推理与感知-规划流水线快速出车并行，谁先给结果谁当主。

#### DriveVLM-W0 / DriveVLM-RL（CASIA × 蔚来，NAVSIM 主战场）
- **W0 架构**：**感知双分支**长在传统端到端系统上——VLM 分支负责"多光谱推理"（跟踪物体、查询目标、一系列多模态窄问题），传统分支负责稠密几何定位；两分支在轨迹层面融合。**稠密世界模型监督**是本系列的灵魂——用"未来帧/轨迹"当稠密监督，把 VLA 从稀疏的"动作 token"监督里解放出来。NAVSIM 与 Anchor 配合 AR 模式报 **93.0 PDMS**，是当时 VLA 路线最强。
- **RL 版（DriveVLM-RL）**：把规划器训练从 IL/SFT 升级为**世界模型做 teacher 的 RL**（narrow-form 提问 + 世界模型判定结果），减少对数据标注与人工偏好的依赖；蒸馏成双分支运行的"GPT 模式"（服务端 VLM 把关到鸿沟，端侧快速执行）。论文含 **KKN（Kinematic/Kalman Navigation?）**——实为"用世界模型增强规划"的关键拼图（见五）。NuScenes 62.6→87.5、Bench2Drive DS 53→71（+29.3%）。

#### DriveGPT（同一家族的服务模式）
- 云端 VLM 负责长尾推理与三重闭环（主规划自评/感知自评/AP 出车校准），模型权重通过蒸馏流到端侧小模型，端侧只做快速执行。

### 2. EMMA——Waymo 的"单一模型做一切"（Gemini 底座）

- **架构**：**不重新训练、不微调规划**，直接用 Gemini 2.0 的预训练能力，把整辆车的驾驶任务压进**一个自回归 LLM**：多相机图像 token 输入，单/多任务序列输出——物体 3D 框（导航坐标表示）、地图、意图轨迹、驾驶规划、位姿。所有空间实体都用**导航坐标（navigational coordinates）**编码成小 VQA 文本，GPU 端即可闭环驾驶。
- **意义**：证明了"一个通用大模型 + token 数学"可以覆盖感知-预测-规划全链，无需专用头；是 VLA 流派"最 Big-Model"的一极。

### 3. AutoVLA——快慢思考双模式（NeurIPS 2025，UCLA）

- **架构**：一套**自回归生成模型**里塞下两种模式：**快思考**（只要动作 token，无推理）与**慢思考**（带 CoT 语言推理）。连续轨迹用 **codebook 离散成"物理动作 token"**送给下一帧（不依赖 planner 姿态）；SFT 训练双模式后，用 **GRPO** 在简单场景把多余推理关掉（输出 VLD 开关信号 + 可验证奖励 = 轨迹安全/PDMS/推理成本）。
- **分数**：NAVSIM v1 PDMS **89.1 / 92.1（+锚点）**；慢思考在长尾（博弈/施工）显著占优，常规场景与快思考持平——"自适应推理"的产出物。

### 4. LinkVLA——"视觉分词器 + LLM + 回归头"，纯 VLA 直出轨迹

- **架构**：多相机 → **visual tokenizer 压缩成视觉 token** → LLM 骨干自回归消耗 → 每步输出 25 个轨迹残差 token（直连几何），**没有语言 CoT**；推理零改写。统一用 "预测未来" 目标联合训练感知（实验证明副产物也能当欧式 3D 检测）。bench 全流程 24M 数据规模定律训练（Seq-Scaling 至 2.5×→2B 级仍线性涨）。
- **分数**：Bench2Drive **DS 91.01 / SR 73**（直接刷新 CARLA v2 闭环纪录）、NAVSIM 第一阶段排名级最佳。**澄清**：它不是"又一个大蒸馏"，而是"先大再蒸馏"的架构选择——7.3M 封顶蒸馏的 tokenizer 也已够用。

### 5. OneVL——稀疏感知 × LLM 的隐式 CoT（NAVSIM 86.83）

- **架构**：**SparseQuery→Tokens→LLM**：感知用稀疏 query（如 BEVFormer 风格）做成 scene tokens + 高维看不清但算得**六模态候选轨迹 score**；四个关键设计：多语义分解、语义感知稀疏压缩、**单步因果投影**（Latent CoT）、辅助专家解码器推理后丢弃。**55 个可学习隐 token**充当软提示在单次前向里"隐式多想了一步"，推理 4.46s 与"不推理"的 4.49s 几乎持平且反超显式 AR CoT 32%。
- **分数**：NAVSIM PDMS **86.83**（4B 干掉 8B 的 LaST-VLA / AdaThinkDrive）；把显式 CoT 的延迟税全面免掉是它的革命性贡献。

### 6. LaST-VLA（CogVLM 家族）——隐 token 自适应缩放

- **架构**：视觉 token 先经 **压缩器**（CNN→token，尺度自适应）再进 LLM：高分辨率 token 给场景语义、低分辨率/稀疏 token 给大范围感知，**自适应 token** 做端到端对照。规划层：**粗 zoom-out** 输出语义 Waypoint 意图 + **细 zoom-in** 关键点精化——语言与 token 双轨。
- **分数**：NAVSIM v1 比赛训练集 top0 88.6（见表对比）。它是 OneVL/LaST 论文里被秒杀、但"token 尺度自适应"思想被反哺的典型。

### 7. WCog-VLA——轻量车端 VLA（本地优先）

- **架构**：**跨传感器 token 尺度统一** + 轻量模型（~TB 级推理预算内）：多相机/多分辨率 token 对齐到同一空间分辨率，再走稀疏 query → 决策 LLM → 连续控制。主打"纯单车端 + 全稀疏"到不了云端的落地路径。实车 demo 事件框架。NAVSIM 上接近顶配组的轻量判别级表现（PDMS 与 OneVL 同一批读级）。

### 8. Senna-VLM / Alpamayo-R1 / AlphaDrive / ReCogDrive——推理流派全家

- **Senna-VLM**：清华系 VLM 驾驶感知口语化冠军——引入"亚像素空间感知"（业界的"1600 万像素定位"江南炼丹系）、一致性检查、隐蔽物体推理；感知是第一落点、规划是延伸。
- **Alpamayo-R1**：**CoT 推理式部署**——任务=感知-预测-决策分步推理（可读），在 CARLA 闭环拿到高速/Fig8/对抗负例都高的综合分数，"可视化即赛点"；虽被 OneVL 归为"吵查式"，但"推理可审计"是 VLA 落地的诉讼护城河。
- **AlphaDrive**：Apache 系端到端推理 VLA，**分步规划**（元规划→执行→验证三段）在当前 VLA 俱陷入"感知准/规划弱"时主打"把规划的每一步也变成 token 可审计"。
- **ReCogDrive**：**因果推理 VLA**——把场景描述组织成"长尾→因果→决策"的因果链（描述偏差、事件因果归因、失败拯救），推理即录取接线员，可解释性拉满。

### 9. NoRD——去推理 VLA（"能不能不煽情？"）

- **架构**：**No Reasoning Denoising**——把 VLA 的推理环节整个拿掉，只用少量轨迹数据做 **SFT + Dr.GRPO**，输出 token 数减 **3 倍**，性能与"全推理版"持平：Waymo Open Motion、NAVSIM 双榜同等或更好。结论：**当任务规则化时，推理不是必要品，是冗余税**——一张极右翼的「纯模仿+RL 就够」宣言，跟 AutoVLA 的"慢思考只在难场景有用"互为镜像。

### 10. 更多 VLA 变体（简列）

| 模型 | 一句话架构 |
|------|-----------|
| **DriveTeach-VLA** | 大教师 VLA 蒸馏出小蒸馏 VLA，双收敛知识交换 |
| **CoWorld-VLA** | VLA 里装 **多专家世界模型** 当"统一未来预测器"，多分支专家出未来、投票当决策 |
| **UniDriveVLA** | 统一框架：同一 token 流里做 感知-预测-规划，自回归密度联合训练 |
| **SparseOccVLA** | 稀疏占用（Sparse Occ）当 VLA 中间世界、多层预测未来 |
| **VLA-World** | VLA 与 世界模型 共用一个编码器，互相喂：世界模型管"未来帧"，VLA 管"该不该动" |

### 11. 同源背景：机器人侧 VLA 定义的三样东西

| 模型 | 贡献 | 被驾驶侧借走的组件 |
|------|------|-------------------|
| **RT-1** | 第一个真实世界 VLA（TxT policy，动作=终态 token 残差，CoT-free） | 动作 token 化 |
| **RT-2** | 网络级 VLM→动作（动作 token 换出） | "动作即答案" 范式 |
| **PaLM-E（RT-2 基座）** | 具身化 VLM（多模态 token 输入端到端） | 端到端感知进 LLM |
| **Gato** | 通用 agent（跨模态单一 transformer） | 单序列 task-token 统一 |
| **SayCan** | "LLM 出可执行方案 + 价值函数选"的经典合流 | 采样+价值选择（与 CLOVER 同构） |
| **VIMA** | 多模态指令模仿 + 记忆增强 transformer | 指令 token |
| **MOO** | 稀疏次动作（M 个必需部分动作为 token） | 结构化动作 token |
| **Octo** | 开源通用（动作分块 diffusion 头） | 分块 + 扩散头 |
| **OpenVLA** | 开源 7B（Prismatic + diffusion action head） | 开源路线 |
| **π0** | pi 系，**动作专家 + 连续动作 Flow** | Flow 动作头（被不少驾驶 VLA 抄） |
| **ST4VLA** | 空间-时序对齐 VLA，长时程动作 | 时序 token |
| **LIBERO** | 基准 + 数据 | 评测方法 |
| **LeRobot** | 开源学习库 + ACT 扩散策略 | 训练栈 |
| **Open-X-Embodiment** | 100 万+ 跨场景数据集 | 跨域预训练 |
| **DROID / RH20T** | 大规模真机/遥操作数据 | 数据资源 |
| **ACT-ALOHA** | 动作分块 Transformer + 遥操作双臂 | 分块概念 |
| **Embodied-GPT** | 统一多模态-动作目标 | 统一目标 |

> **判断**：VLA 流派正在复制 Diffusion 流派当年的轨迹——**"先解释，再嫌解释贵，最后把解释压缩进隐 token/RL"**。OneVL/NoRD/WCog-VLA 一边，AutoVLA 的 GRPO 关掉思考一边，DriveVLM-RL 的世界模型 teacher 一边——2026 年的 VLA 已经不谈"要不要 LLM"，只谈"你花多少个 token 想问题"。

---

## 七、流派⑤：世界模型式——"先在脑内演一遍未来"

把"预测未来"当作一等公民：模型必须学会环境动力学，规划要么"以未来为条件"生成、要么"让未来打分"、要么"滚出来比"。这里能量守恒地分四种：**潜空间动力学 / 视频生成 / 双路径 / JEPA 预测型**。

### 1. DreamerV3——隐世界模型 RL 的教科书（2023-24，跨任务 SOTA）

- **架构**：**经典 actor-critic 世界模型三层**——RSSM（Recurrent State-Space Model）把历史压进隐状态 + **World Model Predicts**（从隐状态 roll-out 未来）；环境不奖励时就**学"先验"预测自监督**；**三个头**（解码器重建图、奖励模型、continuation 模型）。Masters 120+ 组合，围棋/Atari/赛车全通用。
- **motor**: 驾驶侧所有"轨迹即未来"想法的抽象母体：给驾驶，就是"在脑内 roll 1000 个未来，挑了回报最高的那个"。

### 2. UniSim / DriveDreamer / Vista——视频生成流

- **UniSim**（Waymo，2023）：条件生成**感知级 synthetic 视频**（控制 ego+agent 动作 → 合成未来帧），当"可交互做梦引擎"，用于不现实的端到端/RL 预训练。
- **DriveDreamer**（2024）：驾驱世界模型，**3D 结构前置 + 未来 4 帧显式预测**（相机出未来 BEV+前景 Seg），是"Cinematic世界模型"早期蓝本。
- 这两代都是"按帧生成"，长时序与交互仍弱。

### 3. Cosmos——NVIDIA 的世界基础模型（2025）

- **架构**：**视频 tokenizer（TokenZ）→ Video Foundation Model 自回归预测下一 epoch**；为驾驶/机器人做了专门的 **Chemistry（传感器/语言/动作对齐 token）编码**——把激光雷达点云、语言指令、动作都编码成视频-世界 token，单模型统一 "视频+传感器+语言" 条件生成。提供 7B/14B 权重 + 安全微调，被 DriveVLM-W0 参考为显式世界模型的补充。
- **作用**：它是"给世界模型一个通用表达式"的扎营者——后面所有把"世界"缝进 VLA/规划的方法，几乎都用它的 tokenizer 世界观。

### 4. DLWM 与 DriveFuture——双路径/长时程

- **DLWM**（2024）：**Denoised Latent World Models**——在潜空间去噪隐向量预测未来，用 BEV 与前景生成双监督 + 自回归滚动多步；NAVSIM 早期世界模型分支（PDMS 84.7，轨迹平滑低曲率，长稳竞标者）。
- **DriveFuture**（2025，华科）：**双路径 CondEgo-Path**——前视捕获意图（自车意图分支）+ 环境感知捕获场景（场景分支），感知→历史轨迹→意图→**预测下一帧未来**闭环；**Cross-Attention 融合 + 分层规划解码**成未来状态。v2 扩展可视 + Who-to-look（时序注意力指定看谁）。定位 世界模型的"意图↔场景"耦合，是 ReWorld 之前的国产世界模型代表（其"前视意图未来"被 World4Drive 化用过）。

### 5. WoTE（World-to-World, 2025）——用语言撬动世界模型

- **架构**：把"世界"理解成**"世界名 → 世界描述"的函数**：语言描述（Socratic 问答）→ **跨片段对应**（同一场景不同牌照/天气/碰撞前后的对齐）→ 世界模型转化成 **可讲性的因果推理**。给出 3 个保险杠语言任务：**DISC（冲突观测混合）→ SAVE（自驾碰撞叙事）→ ROom（行动项提取）**。表面是"VLM 讲故事",实质是**把世界模型的隐状态投影到语言空间**——它为"世界模型能不能被审计"打开的窗口比任何仿真都直白。

### 6. JEPA-DRIVE / Drive-JEPA——预测型自监督（"世界模型不预测像素，预测抽象"）

- **JEPA 基础**（LeCun 系）：**Joint-Embedding Predictive Architecture**——不重建像素/音频，只预测**隐空间的抽象表征**；训练目标是"表征多阶一致性"而非 pixel loss。核心商业：**世界比你看到的更有意义**。
- **Drive-JEPA**（2025，理想×清华）：给驾驶装 **预测一致性损失**：感知 query 在 BEV+向量特征上做隐空间对齐预测未来，**预测驱动的隐世界模型**；规划两次用同一一致性（先验+动作条件）。位置 token + BEV-向量双特征。
- **SparseDrive-JEPA 变体**：NAVSIM IL PDMS **93.3 / EPDMS 87.8**（官方/开源上位），证明预测型自监督在世界模型预算下能打到扩散法同一线。

### 7. World4Drive / Uni-World / ExploreVLA——把"未来"装回 VLA

- **World4Drive**（2025，香港）且在 CARLA/Dynamics 上验证：**训练假想未来自回归**——四个模块（位置/箭头/go-along 预测 → 动态隐 token 精确未来 → 每个 token 备对齐未来）合成"未来视频"充当监督；NAVSIM PDMS 85.1，碰撞率极低（0.5%），但长时序滚动缺想象力（对比 GAIA/Vista）。
- **Uni-World VLA**（2025，字节×上交）：**纹理大使（mosaic-texture VLA）4**。关键架构=**统一嵌入空间**：把视觉帧 token、传感器 token、语言指令、动作 token 放进**同一个 body/动作 token 流**，世界模型在中间 decode"至少 3 帧未来"充当训练监督；零训练化迁移（融合规划知识）。设计了 Mocap-Text 数据合成协议（MosaicVLA），实现"世界模型视角的 VLA"。
- **ExploreVLA**（2025）：**探索作为监督**——具身探索缓存为数据（ExploreVLA-v0 开源 50k × Xi 分辨率），把"聪明行动"当目标给世界模型和 VLA 统一训练；行为级 RL（动作空间在 lnf 尺度具身化）。为算补贴数据资源，为西雅图第二。
- **X-World**：生成式世界模型将传感器（LiDAR→视觉）与语义对齐，补齐"世界模型不知道视觉"（多模态世界 token 流，见 §5 组图）。

### 8. ReWorld / RAW2Drive / WorldVLA——2026 年最新玩法

- **ReWorld**（2025，华科×小米）：**复用已训练的非世界模型（Diffusion 规划器）+ 双 DiT 世界模型**——双解码器（动作修正 + 释放未来）在真实轨迹与"给我们拍电影的世界"之间做潜空间桥接；把世界模型当老师：**world-model-in-the-loop RL**（奖励来自世界模型再评估，而非人类打分）——"老师自己脑补未来来打分众筹监督"。两个核心：world-model=扩散、RL 老师=世界模型、前视预测潜空间对齐是灵魂。相比 Gen-Drive（人工偏好 reward），ReWorld 是纯自动 reward。
- **RAW2Drive**（2025）：**Raw-sensor 世界模型**——**多步分解特征级别意视频**隐空间预测未来 2 帧 plus "为成为我的世界模型中一个合格的阁楼"趋入"可交互"：raw sensor→SFT 预训练任务簇→解码 arrow tokens。主打不依赖标注。
- **WorldVLA**（2025）：VLA+世界模型双向 —— 世界模型解码未来帧时用**动作注意力掩码**（masked attention 让未来显式依赖于动作），规划时世界的力量回注 VLA："动作是世界的因"。缓冲连续一切自监督在 world 回环里。

> **判断**：世界模型线的重心正在从"视频生成好看"转移到"**世界模型当 teacher 给 VLA 发监督**"（DriveVLM-RL、ReWorld、DriveFuture）与"**预测型隐表征替代重建**"（JEPA）。Cosmos/DriveFuture 打底 + ReWorld/WorldVLA 缝合，这大概是 2027 年 VLA 的天花板来源。

---

## 八、流派⑥：数据、仿真与评测基建——所有人都踩过的地板

方法论文再强，没有这些地板没法比、没法数据、没法闭环。它们常被忽略，却决定了上面所有流派的分数噪声。

### 1. NAVSIM——"开环×规则化闭环"的标准化现场

- **玩法**：剪裁 nuScenes/Argoverse 本体轨迹 → 规尺 PDMS（进度/碰撞/舒适/安全复合分）/EPDMS（v2 加路程一致性）；v2 直接驱动了"打分器-生成器"范式的军备竞赛（CLOVER/GTRS/SparseDriveV2 都是榜上常客），是本文所有分数的主战场。
- **榜单文化**：每期 top 的倍差就是"锚点 vs 无锚点"、"LiDAR vs camera"、打 vs 生成 的活体显微——相当于深度学习圈的那张 ImageNet 榜。

### 2. Bench2Drive / CARLA——闭环仿真与"开环分数的诚实版"

- **CARLA v2**：闭环仿真 MARO 世界，比赛规则化毫不妥协（复杂遮挡/光照/天气连续体）。
- **Bench2Drive**：44 场景 × 大量时刻采样，规范闭环正分数（DS/SR）；**VADv2/B2D 76.15、DriveTransformer 63.46、LinkVLA 91.01/73**——DS 严重惩罚"不敢动"模型，是开环 PDMS 的"降维锤"，把一堆开环 90+ 的打回 40-60。**注意**：LinkVLA(CARLA v2 专属)已超 B2D 全部版本的开环表——闭环永远是最严厉的裁判。

### 3. 重卡/干线数据族（博主的另一个"被忽略的江湖"）

| 数据集/模型 | 一句话 |
|------------|--------|
| **nuTruck / MAN TruckScenes** | 卡车首个大规模多传感数据集；给卡车把"感知地图比乘用车更依赖标尺线"的坑复盘 |
| **TruckDrive** | 端到端驾驶卡车初版（NS 的卡车场景是它的"验底盘"） |
| **TruckV2X** | 卡车车联网（雷达+视觉+通信），后台系统"模块桩"整车加轨 |
| **AntiRollover 等安全族** | 侧翻阻抗/载荷力学，"卡车版的事故免疫"，VLA 上卡车的第一批护栏分析 |

### 4. K-Risk & 其他评测微生物

- **K-Risk**：把单字段"风险评分"扩展成**多维度**（Robustness/Awareness/Compliance 等），发现"规划错误大多不是撞车而是偏离-场"——它教我们看分数要拆维度，而不是只看单数。
- 背景还有 ROADWork（交互式道路工作区）、Impromptu、HUGSIM（3D GS 闭环）——2026 年"评测叙事"必须至少三镜：开环（NAVSIM）、闭环（Bench2Drive）、真闭环仿真（HUGSIM）。

### 5. 数据管线（MOSAIC 已见上文）；数据生成——VLA 时代的硬通货

VLA 动辄 6M-24M 字规模的数据，人工标注定死上限——**用扩散/世界模型合成数据**（UniSim/Cosmos/ReWorld 全部指向"自己教自己"）与"教师 VLA 蒸学生 VLA"（DriveTeach-VLA）正在把数据成本压下去。

---

## 九、终局表：所有人都在一张榜上（NAVSIM v1 开环 PDMS）

把全文方法放到同一尺度（官方/论文口径混合，已尽最大能力注明）：

| 方法 | 流派 | 传感器 | 关键架构 | PDMS(v1) | 备注 |
|------|------|--------|---------|---------|------|
| Human-Centric | 参考 | 全 | — | 94.8 | 人工参考 |
| **TOAD（DrivoR 基座）** | ②③ | camera | CEM+评分器搜索 | **94.7** | 测试时优化 |
| CLOVER | ② | camera+LiDAR | 生成器-打分器闭环蒸馏 | **94.5** | 无锚点 SOTA |
| TransDiffuser | ③ | camera+LiDAR | 去相关+DDPM | 94.85 | 无锚点 |
| DrivoR | ① | camera | ViT 寄存器 token | 93.7 | 40M 参数极简 |
| SparseDriveV2 | ② | camera | path×velocity 因子分解词表 | 92.0 | 官方榜 |
| iPad | ② | camera | proposal 多头迭代 | 91.7 | — |
| **Hydra-MDP** | ② | camera | 8192 锚词表+多头打分 | 91.26 | 官方 91.87 |
| DiffusionDrive V2 | ③ | camera | 截断扩散+GRPO | 91.2 | — |
| GoalFlow | ③ | camera | 目标点词表+Rectified Flow | 90.3 | 1 步生成 |
| FeaXDrive | ③ | camera | 轨迹中心+FA-GRPO | 90.0 | SDF 引导 |
| AutoVLA | ④ | camera | codebook 快慢思考 | 89.1 / 92.1+锚 | GRPO |
| DriveVLM-W0 | ④ | camera | 双分支稠密世界模型 | 93.0(AR) | — |
| VADv2 | ① | camera | 4096 词表 FPS | 87.7 | — |
| OneVL | ④ | camera | 隐 CoT 稀疏感知 | 86.83 | 4B |
| World4Drive | ⑤ | camera | 未来自回归 | 85.1 | — |
| DLWM | ⑤ | camera | 潜空间去噪世界模型 | 84.7 | — |

> 这张表的**最大失真警示**：LiDAR 加成（TransDiffuser 94.85）与无锚点 vs 锚点的口径，CLOVER/TOAD/DrivoR/TransDiffuser 是"同一批强队"（照同一套 PDM），AutoVLA/VADv2/WCog-VLA 是不同 batch 的不同配置；vanilla VADv2 87.7 但带锚 90.5。**别把 94.85 vs 89.1 简单当"扩散 > VLA"**，去看 sensor 与 training recipe。

## NAVSIM v2（EPDMS，伪闭环）一张小表

| 方法 | EPDMS | 备注 |
|------|-------|------|
| PDM-Closed | 56.6 | 特权真值 |
| **DrivoR+TOAD** | **56.3** | 逼近 privileged |
| DrivoR | 54.6 | CLOVER→NavHard 48.3 |
| iPad+TOAD | 49.8 | +43.6% |
| GTRS | 49.4 | NavHard 榜 |
| CLOVER | 90.4(v1)/48.3 | NavHard |
| SparseDriveV2 | 87.35 | — |

---

## 十、我的个人思考：2027 年的端到端会活成什么样

### 1. 三条主线的钟摆：显式→打分→生成→隐式，正在合并成"打分生成Im"一体

把 Timeline 放平：**显式管线（BEV→轨迹）→ 词表打分（CLOVER）→ 扩散/Flow（DiffusionDrive/GoalFlow）→ 世界模型 teacher（DriveVLM-RL）**。它们不是替代关系——今天最强的 TOAD 是"DrivoR 生成 + 评分器奖励 + CEM 搜索"，全元素来自三个流派。**Endgame 不是裁判选谁，而是"生成器无限生成、打分器当 reward、RL 把它揉进网络"**。下一代差的是把 TOAD 测试时搜索**压缩回前向一次**（蒸馏），以及把 CEM 换成扩散/世界模型的全局优化。

### 2. VLA 的"思考税"是伪需求：OneVL 已经给答案

2025 年 VLA 都在"加 CoT"；2026 年年中风向剧变——**OneVL 把显式推理改成单步隐 CoT（4.46s，86.83 反超 8B 显式版），NoRD 直接砍掉推理只靠 RL（token 省 3 倍、性能持平）**。我的判断：**VLA 的真实护城河永远不是"会不会说人话"，而是有无稠密监督（世界模型）与可控推理预算**。显式 CoT 只适合"审判要看的边缘用例"，生产环境请开静音模式。

### 3. 世界模型 has won，但赢在"当 teacher"而不是"画视频"

Cosmos/DriveFuture/DreamerV3 告诉我们：**未来预测最有用的落点是"给 VLA/规划器发稠密监督"，而不是"把未来帧渲染得好看"**。ReWorld（world-model-in-the-loop RL）、DriveVLM-RL（KKN 世界模型做 teacher）、Gen-Drive（学习 reward）三剑合璧——**下一代 reward 信号将从"PDM 分数"换成"世界模型自判"**，RL 才真正摆脱人工偏好。CLOVER 已证明"打分器≈评估器"，下一步是"世界模型≈评估器"。

### 4. 打分器泛化是 Scoring 流派的天花板：TOAD 教我们的

**固定词表打分器（Hydra/GTRS）在测试时搜索反而掉分——因为词表内过拟合**；on-the-fly 打分器（DrivoR/iPad）则被 TOAD 推高。所以"打分-生成闭环"真正的本质问题是：**打分数不能绑定训练词表**。GTRS 的无 D 先在推理用词表 dropout 逼泛化，是指了条路，但终极答案可能是"打分器也在测试时重训/在线适应"——这是图很大的一张空白。

### 5. 闭环评估的"降维锤"会让分数差距显形

CLOVER 94.5 在 Bench2Drive 一类闭环里大概率掉到 70±；**开环 PDMS 差 1 分 ≠ 驾驶差 1 分**。2027 年评估必须三镜齐开：开环（NAVSIM/分数）、闭环（Bench2Drive 高分段比 DS）、真闭合（HUGSIM 自由碰撞）。科幻地讲，**谁把"开环高分"和"闭环能跑"之间的鸿沟补上，谁就赢**——目前只有 VLA-ReWorld 这类"世界模型+RL"在往这冲。

### 6. 卡车/长距是 VLA 的"灰区测试场"——跨车型泛化是硬资质

人话：卡车数据族告诉我，**"城市小 OD 的图像 VLA" 跟 "150 米外的重卡盲区" 几乎两个物种**。谁能在多车型（乘用车→卡车→施工段）、多天气、多国L+R 路侧下不崩，谁的模型才具备"自动驾驶寒武纪"的扩散力。**跨车型不是加分项，是生存分**。

### 7. 别把对比表当皇历：口径永远是第一公民

这张终局表里有 3 类谎言：①LiDAR 无锚点（TransDiffuser 94.85）不该跟 camera 锚点（DiffusionDrive 91.2）直线比；②自报 vs 官方；③v1 vs v2。**读任何一篇论文的数字，先问四件事：sensor？锚点？自报还是榜？v几？**——这是我读了 100+ 篇后最能帮你避开分数幻觉的一句话。

---

## 十一、结语

从 UniAD 的长短线提线木偶，到 TransDiffuser 的去相关生成，到 CLOVER 的闭环蒸馏，到 LinkVLA 的 91.01 DS，到 ReWorld 的世界模型 teacher——五年时光里我们见证了**端到端自动驾驶从"能不能开"走向"怎么证明能开、怎么在评测钢丝上安全地开"**。六大流派不是六个门派，而是同一张螺旋的一层层台阶：

> **显式管线回答"它做什么"，打分回答"哪个好"，生成回答"多多益善"，VLA 回答"为什么"，世界模型回答"接下来呢"，评测回答"凭什么信"。**

下一篇预告：我准备把**世界模型当 teacher 的 RL 管线（DriveVLM-RL / ReWorld / Gen-Drive 三篇对比）**单独扯开再写一篇，因为在我看来那是 2027 年最值得下注的方向。也欢迎在评论区点题下一个你想让我拆解的模型。

---
