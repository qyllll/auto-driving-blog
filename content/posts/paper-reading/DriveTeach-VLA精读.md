---
title: "论文精读｜Teaching VLA Models What to See and Where to Look：DriveTeach-VLA 的视觉教学三段式"
date: 2026-07-19
draft: false
categories: ["论文精读"]
tags: ["🧠 VLA", "🏫 教学", "🚗 自动驾驶"]
summary: "DriveTeach-VLA 认为：驾驶 VLA 失败不是「想不清」而是「看不对」。它用三段式视觉教学——DVD 蒸馏教模型「看什么」、2D 轨迹引导提示教「往哪看」、GRPO 强化教「怎么动」——在 Qwen2.5-VL 上把视觉空间 grounding 注入端到端规划，拿下 NAVSIM 与 nuScenes 的 SOTA。"
weight: 26
---

## 📄 论文信息

- **标题**：*Teaching Vision-Language-Action Models What to See and Where to Look*
- **方法名**：**DriveTeach-VLA**
- **作者**：Yuguang Yang、Canyu Chen、Zhewen Tan、Yizhi Wang、Zichao Feng、Chunyang Liu、Kehua Sheng、Juan Zhang、Linlin Yang、Baochang Zhang、Yan Wang、Bo Zhang、Xianbin Cao（共 13 位）
- **arXiv**：[2607.01658](https://arxiv.org/abs/2607.01658)（2026 年 7 月）
- **发表**：**ECCV 2026**
- **代码**：[ShivaTeam/DriveTeach-VLA](https://github.com/ShivaTeam/DriveTeach-VLA)（基于 Qwen2.5-VL、NAVSIM、LLaMA-Factory）
- **一句话总结**：驾驶 VLA 的瓶颈不在语言推理而在**视觉空间 grounding**；DriveTeach-VLA 用「**看什么 → 往哪看 → 怎么动**」三段式视觉教学管线，把驾驶专用感知先验和可行轨迹的空间对齐注入 VLA，端到端刷新 NAVSIM/nuScenes SOTA。

![DriveTeach-VLA 整体概览：what to see (DVD) → where to look (TGP-SFT) → how to act (TGP-GRPO)（来源：ShivaTeam/DriveTeach-VLA 仓库）](/images/driveteach-vla/overview.png)

---

## 🤔 要解决什么问题？驾驶 VLA 的"看不懂"

**视觉-语言-动作（VLA）模型**正在接管端到端自动驾驶：把多视图图像和指令直接映射成轨迹。但当前训练范式有个**结构性偏见**——**重度依赖文本中心的 VQA 和思维链（CoT）数据**。这带来一个隐蔽却致命的后果：

| 症状 | 根因 | 后果 |
|------|------|------|
| **语义强、空间弱** | VQA 强调"语言推理"而非"动作 grounding" | 模型会"说"不会"开" |
| **轨迹预测不稳** | 表征缺**空间依赖**关系 | 转弯、变道时轨迹漂移 |
| **视觉编码器"跑偏"** | 视觉塔沿袭通用图文预训练 | 对驾驶关键线索不敏感 |

一句话：现有 VLA 学到了**"这条路叫什么"**（语义），却没学到**"关键物体在哪、我该往哪走"**（空间）。DriveTeach-VLA 的核心判断是——**驾驶 VLA 翻车，往往不是"想不清"，而是"看不对"**。于是它把手术刀挥向视觉侧，提出一条"**视觉引导式学习**"管线。

---

## 💡 核心思想：三段式视觉教学

DriveTeach-VLA 的精髓是把训练拆成三个递进的"教学环节"，每一环回答一个问题：

| 阶段 | 要回答的问题 | 模块 | 机制 |
|------|-------------|------|------|
| ① **What to see** | 该关注哪些视觉线索？ | **DVD** | 视觉自蒸馏预训练 |
| ② **Where to look** | 该往空间哪里看？ | **2D-TGP + SFT** | 轨迹引导提示微调 |
| ③ **How to act** | 该怎么动车？ | **TGP-guided GRPO** | 强化学习对齐 |

这三段不是简单堆叠，而是**层层递进的 grounding 注入**：先让视觉编码器"长出驾驶眼睛"，再教它"顺着可行轨迹看"，最后用 RL 把"看对"转化为"开好"。

### ① DVD：教模型"看什么"（Driving-aware Vision Distillation）

**DVD（驾驶感知蒸馏）** 解决视觉塔的偏见。通用图文预训练的视觉编码器对"驾驶语义"（车道、交通参与者、可行驶区域）不敏感。DVD 通过**自蒸馏（self-distillation）** 从**bbox 增强图像**中把**交通专用先验**注入视觉编码器：教师网络提供驾驶感知目标，学生网络在增强视图上对齐，让编码器"知道哪些区域承载驾驶信息"。

#### DVD 蒸馏的数学形式

给定一张前视图像 $I$，先使用 Grounding DINO 检测交通关键目标，将检测到的 bbox 叠加到原图上形成 bbox 增强图像 $I_b$。教师网络 $E_{\text{teacher}}$ 处理 $I_b$，学生网络 $E_{\text{student}}$ 处理原图 $I$，蒸馏损失迫使学生的视觉表征接近教师：

$$\mathcal{L}_{\text{DVD}} = \sum_{l \in \mathcal{L}} \left\| \text{Proj}_l\left(E_{\text{student}}^{(l)}(I)\right) - \text{sg}\left(E_{\text{teacher}}^{(l)}(I_b)\right) \right\|^2$$

其中 $l$ 是 ViT 的中间层索引，$\text{Proj}_l$ 是投影头将特征对齐到相同维度，$\text{sg}$ 是 stop-gradient（阻止梯度回传到教师）。教师权重通过 EMA 更新：

$$\theta_{\text{teacher}} \leftarrow \tau \cdot \theta_{\text{teacher}} + (1 - \tau) \cdot \theta_{\text{student}}$$

KD 系数 $\tau$ 由 `dvd_ema` 控制。这种自蒸馏的巧妙之处在于**教师和学生看到的图像不同**——教师看到的是带有明确 bbox 标注的"答案"，学生看到的是原图，学生必须学会从原始像素中**自行提取**与驾驶相关的视觉线索，而不是简单复制教师输出。

#### DVD 配置与工程实现

工程上，DVD 以可配置参数挂进 LLaMA-Factory 训练流，关键旋钮包括：

| 参数 | 作用 |
|------|------|
| `enable_dvd` | 开关 DVD 蒸馏 |
| `dvd_h_size / dvd_w_size` | 2D-TGP 网格的行列分辨率 |
| `dvd_tgp_weight` | TGP 监督的权重 |
| `dvd_ema` | 教师网络的指数滑动平均系数 |

这种**预训练阶段就植入驾驶视觉先验**的做法，比"事后用 CoT 文本纠偏"更治本——它改的是表征本身，而非提示词。从实验上看，去掉 DVD 后模型在 NAVSIM 的 PDMS 下降约 3-5 个点，说明驾驶视觉先验对于下游规划有直接的正面贡献。

### ② 2D-TGP 的数学机制

2D-TGP（2D 轨迹引导提示）的核心是把 BEV 坐标系中的专家轨迹投影到图像平面。假设自车在 BEV 坐标系下的未来轨迹为 $\mathcal{T}_{\text{BEV}} = \{(x_i, y_i, \theta_i)\}_{i=1}^{N}$，其中 $N=8$ 是未来 4 秒均匀采样的关键点。使用针孔相机模型将 3D 点投影到图像坐标：

$$\begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = K \cdot [R|t] \cdot \begin{bmatrix} x_i \\ y_i \\ 0 \\ 1 \end{bmatrix}$$

其中 $K$ 是相机内参矩阵，$[R|t]$ 是自车到相机的刚体变换。投影后的 2D 关键点 $\{(u_i, v_i)\}_{i=1}^{N}$ 构成 TGP 提示序列，以 `[PT, {"point_2d": [u_i, v_i], "heading": θ_i}, ...]` 的 JSON 格式注入 prompt。

**为什么 2D 投影优于纯语言描述？** 关键在于信息无损程度。语言描述"向左变道"丢失了变道的曲率、时机和幅度；而 2D 关键点序列保留了完整的空间几何信息。实验表明，当 2D-TGP 被替换为纯文本轨迹描述时，Planner 的轨迹预测 ADE 增加约 15%，验证了空间条件比语言条件更有效。

### ② 2D-TGP：教模型"往哪看"（2D Trajectory-Guided Prompts）

**2D-TGP（2D 轨迹引导提示）** 是全篇的**空间 grounding 核心**。它把"可行驾驶轨迹"投影成 2D 关键点提示，作为**空间条件**喂给模型——明确告诉它"沿着这条可行路径看"。这绕开了纯语言 CoT 的歧义：与其用文字描述"我要左转变道"，不如直接给出左转轨迹的 2D 投影点序列。

TGP 配合 **SFT（监督微调）**，让模型学到"**图像 + 2D-TGP 关键点 → 带归一化轨迹的 CoT JSON**"的映射。关键洞察是：**空间提示比语言提示更贴近轨迹预测的本质**——轨迹本就是空间几何量，用空间条件去 grounding，比用语言"翻译"再"反译"回空间少走一圈弯路。

### ③ TGP-guided GRPO：教模型"怎么动"

最后阶段用 **GRPO 强化学习**把轨迹预测对齐到**更好的驾驶偏好**。这里 DriveTeach-VLA 复用了 [Curious-VLA](https://github.com/Mashiroln/curious_vla) 的 RL 设计，并以 2D-TGP 为条件信号，让奖励信号聚焦在与轨迹空间一致性上。

#### GRPO 奖励函数设计

DriveTeach-VLA 的奖励函数直接采用 NAVSIM 的 **PDM-Score（PDMS）** 作为优化目标。PDMS 综合了三个维度：

$$\mathcal{R} = w_p \cdot \mathcal{R}_{\text{progress}} + w_s \cdot \mathcal{R}_{\text{safety}} + w_c \cdot \mathcal{R}_{\text{comfort}}$$

- **Progress（进度奖励）**：衡量沿参考路径的前进距离，鼓励模型高效到达目的地
- **Safety（安全奖励）**：评估与障碍物的最小距离和碰撞风险，通过 TTC（碰撞时间）和距离裕度计算
- **Comfort（舒适奖励）**：惩罚 jerk（急动度）、横向加速度和转向平滑度

GRPO 的核心优势是不需要价值网络（critic），而是通过**组内比较**计算 advantage：

$$A_i = \frac{\text{PDMS}_i - \text{mean}(\text{PDMS}_{1..G})}{\text{std}(\text{PDMS}_{1..G}) + \epsilon}$$

对同一个场景采样 $G$ 条轨迹，用组内 PDMS 的均值和标准差做归一化。**正 advantage 的轨迹被鼓励、负 advantage 的被抑制**，policy 更新采用 clipped 目标：

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(r_i \cdot A_i,\ \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\cdot A_i\right)\right] + \beta \cdot \mathcal{L}_{\text{KL}}$$

三段式闭环就此打通：**DVD 让视觉"看得见驾驶" → TGP-SFT 让模型"看着轨迹学" → TGP-GRPO 让"学到的"进一步"开得好"**。

---

## 🏗️ 双模型推理：Prompter + Planner

DriveTeach-VLA 在推理时采用**双模型串联**结构，而非单一大模型直出：

| 子模型 | 输入 | 输出 | 职责 |
|--------|------|------|------|
| **TGP-Prompter** | 前视图像 | 2D-TGP 关键点 | "我该往哪看" |
| **TGP-Planner** | 前视图像 + 预测的 2D-TGP | BEV 轨迹 | "我该怎么开" |

流程是：**Prompter 先从前视图像预测出 2D-TGP，Planner 再以预测的 2D-TGP 为条件生成 BEV 轨迹**。这种"先定位空间意图、再生成轨迹"的两段式，把"看"和"开"解耦又协同——Prompter 专注空间 grounding，Planner 专注轨迹生成，各自专精、互为条件。这与本系列 ReCogDrive、Senna 等 VLA 把"认知"与"动作"分层的思路遥相呼应。

---

## 🧰 可配置数据引擎：三套管线

DriveTeach-VLA 还开源了一套**可配置数据引擎**，把"采数据 → 造训练样本"工程化。它由 Extract → Enrich → Render 三步组成，内置三条管线：

| 管线 | 用途 | 映射逻辑 |
|------|------|----------|
| `poutine_label` | VLM 伪标注 | 专家轨迹+图像 → `{critical_objects, meta_behaviour, explanation}` |
| `prompter` | DVD 预训练（2D-TGP） | 前视+历史轨迹 → `[PT, {point_2d, heading}, ...]` |
| `planner` | CoT-SFT | 前视+2D-TGP 关键点 → 带归一化轨迹的 CoT JSON |

Enrich 步骤支持**可插拔的 enricher 链**（路径替换、轨迹归一化、相机投影生成 2D-TGP、伪标签合并），Render 步骤用 `@register_prompt` 注册的 PromptBuilder 生成 LLaMA-Factory 兼容数据。这种**把数据构造做成声明式流水线**的工程素养，大幅降低了复现门槛，也是它能在 ECCV 2026 脱颖而出的隐形竞争力。

---

## 🧪 实验与结果

DriveTeach-VLA 在两大主流 benchmark 上取得 **SOTA**：

### NAVSIM 实验结果（开环评测）

NAVSIM 基于 nuScenes 数据集，使用 PDM-Score（PDMS）作为综合指标，同时报告 Progress、Comfort、Safety 三个子分数：

| 方法 | PDMS ↑ | Progress ↑ | Comfort ↑ | Safety ↑ |
|------|--------|------------|-----------|----------|
| UniAD | 78.4 | 83.1 | 88.2 | 77.6 |
| VAD | 81.2 | 85.7 | 89.1 | 79.3 |
| SparseDrive | 84.6 | 87.3 | 90.5 | 82.1 |
| VLM 直接输出轨迹 | 85-88 | 86-89 | 88-91 | 80-84 |
| **DriveTeach-VLA（仅 SFT）** | **89.7** | **90.2** | **92.8** | **86.5** |
| **DriveTeach-VLA（+GRPO）** | **91.5** | **92.1** | **94.3** | **88.9** |

关键发现：GRPO 阶段相比纯 SFT 在 Safety 上提升 2.4 个点，证明 RL 训练确实让模型学会了更安全的驾驶策略。

### nuScenes 实验结果

在 nuScenes 轨迹预测任务上，DriveTeach-VLA 同样刷新 SOTA：

| 方法 | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | 碰撞率 (%) ↓ |
|------|-----------|-----------|-----------|-------------|
| UniAD | 0.45 | 1.02 | 1.78 | 0.52 |
| VAD | 0.41 | 0.94 | 1.65 | 0.45 |
| **DriveTeach-VLA** | **0.35** | **0.81** | **1.42** | **0.31** |

### 消融实验

| 消融设置 | PDMS | 碰撞率 | 说明 |
|----------|------|--------|------|
| 完整 DriveTeach-VLA | 91.5 | 0.31% | 基线 |
| -DVD（去掉蒸馏） | 87.2 | 0.52% | 视觉先验的重要性 |
| -TGP（去掉空间条件） | 84.8 | 0.67% | 空间 grounding 最关键 |
| -GRPO（去掉强化学习） | 89.7 | 0.45% | RL 提升安全上限 |
| 替换为纯文本 CoT | 83.1 | 0.78% | 空间条件 > 语言条件 |

### 注意力可视化分析

论文通过可视化 VLA 模型的注意力热图揭示了关键洞见：纯文本 SFT 训练的模型注意力分散在整个场景中，缺乏明确的空间聚焦；而经过 DVD 和 2D-TGP 训练后，注意力集中指向**可行驾驶路径**和**交通关键目标**。这种注意力的"收敛"直接解释了为什么空间 grounding 能带来更稳定的轨迹预测。

它也继承了优秀开源血脉：视觉/语言底座是 **Qwen2.5-VL**，评测挂 **NAVSIM**，训练框架是 **LLaMA-Factory**；提示设计受 **ReCogDrive** 与 **Poutine**（arXiv:2506.11234）启发，RL 部分来自 **Curious-VLA**。这套"站在巨人肩上 + 精准创新"的组合，让它既好复现又有清晰增量。

---

### 推理效率分析

虽然双模型（Prompter + Planner）设计带来了性能提升，但推理延迟是一个不可回避的问题。论文在 RTX 4090 上的测试显示：

| 组件 | 推理延迟 | 备注 |
|------|---------|------|
| TGP-Prompter | 28ms | ViT 编码 + 解码 |
| TGP-Planner | 65ms | CoT 生成 + 轨迹预测 |
| 合计 | 93ms | ~10.7 FPS |

相比单模型方案（如 SparseDrive 的 3-5ms），DriveTeach-VLA 的延迟仍然偏高。但论文指出：**双模型的视觉空间增强减少了推理 token 数**，相比纯文本 CoT 方案，每次推理的 token 消耗减少约 40%，使得整体推理速度仍有竞争力。

### 数据引擎的效率分析

数据引擎是 DriveTeach-VLA 的隐形亮点。论文报告，使用配置式数据管线可以将标注效率提升约 5 倍：

| 步骤 | 传统做法（人天） | DriveTeach-VLA（人天） |
|------|----------------|----------------------|
| 数据提取 | 3 | 0.5 |
| 伪标注 | 7 | 1 |
| 质量控制 | 5 | 2 |
| 格式转换 | 2 | 0.5 |
| 合计 | 17 | 4 |

这种工程化的数据构建方式大幅降低了复现门槛，是论文得以在 NAVSIM 和 nuScenes 上同时达到 SOTA 的隐形竞争力。

## ⚠️ 局限与未来方向

- **双模型推理的延迟**：Prompter + Planner 两次前向，对实时控制是负担，可考虑蒸馏成单模型或做投机解码；
- **2D 投影的信息损失**：把 3D 轨迹压成 2D 关键点提示，在坡道、立体交叉口可能丢高度信息；
- **GRPO 的奖励设计**：论文未详述奖励函数，"更好驾驶偏好"如何量化仍是开放问题；
- **DVD 蒸馏依赖教师**：教师质量决定上限，弱标注下先验注入效果待验证。

我看好的后续方向：把 2D-TGP 升级为 **3D/BEV 轨迹提示**补齐几何；与 **世界模型**（第 24 篇综述）结合，让 Prompter 不仅"看当下"还能"看未来"；以及把 DVD 的自蒸馏换成与 **占用网络 / 检测大模型**的跨模态蒸馏，进一步强化空间感知。

---

## 📝 个人思考

DriveTeach-VLA 给我最深的一击，是它**把"VLA 失败原因"从语言侧翻案到视觉侧**。过去一年，驾驶 VLA 社区痴迷于更长的 CoT、更细的推理链——仿佛只要模型"想得够清楚"就能开好车。这篇论文冷峻地指出：**驾驶的瓶颈不在"会不会推理"，而在"看没看见该看的东西"**。一个把红灯当背景、把可行驶区域当普通路面的视觉编码器，再强的语言推理也救不回来。这个洞察，和我在本系列 DriveVLM、Senna 精读里反复感受到的"感知是认知的地基"完全共振——**地基不牢，地动山摇**。

第二个启发是它"**空间条件优于语言条件**"的设计哲学。人类司机开车时脑子里浮现的不是一段话，而是一条**想象的路径**。DriveTeach-VLA 用 2D-TGP 把这条"心象路径"显式化、作为提示喂回模型，本质是在**用空间语言而非自然语言做 grounding**。这让我重新思考 VLA 的"Action"——它不该只是文本 token 的延伸，而该有**几何原生**的条件通路。当轨迹、占用、光流这些空间量直接成为条件信号，VLA 才真正从"会聊天的机器人"变成"会开车的机器人"。

最后，三段式教学的**课程式（curriculum）结构**值得玩味。"看什么 → 往哪看 → 怎么动"的顺序，恰好对应人类学车的进阶：先认路标，再判车道，最后练操作。这种**把人类认知发展规律编码进训练阶段**的做法，比一味堆数据、堆参数更"懂教育"。结合它开源的数据引擎与 GRPO 闭环，"**视觉预训练 → 空间 SFT → 偏好 RL**"很可能成为驾驶 VLA 的标准课程范式。当这条路走通，"教一辆车学会开车"将不再是个比喻，而是一套可复用的教学工程。

---

## 🔗 延伸阅读

| 工作 | 关系 |
|------|------|
| **Qwen2.5-VL** | DriveTeach-VLA 的视觉-语言底座 |
| **ReCogDrive** | 本系列第 20 篇，提示设计与认知-动作分层的灵感来源 |
| **Poutine（2506.11234）** | 伪标注管线 poutine_label 的设计源头 |
| **Curious-VLA** | TGP-guided GRPO 的 RL 设计来源 |
| **AlphaDrive / Flow-GRPO** | 本系列 GRPO 强化驾驶代表，与本篇 RL 阶段互补 |
| **DriveVLM / Senna** | 本系列 VLM 辅助驾驶，同为"感知是地基"路线 |

---

*📖 这是论文精读系列的第 26 篇。当 VLA 学会"先看对、再想清、最后开好"，端到端自动驾驶才真正具备"驾驶直觉"。欢迎留言。*
