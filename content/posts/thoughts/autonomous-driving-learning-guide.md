---
title: "个人思考｜自动驾驶技术全景学习指南：从基础到前沿的六范式知识体系"
date: 2026-07-27
draft: false
categories: ["个人思考"]
tags: ["🧠 VLA", "🔗 端到端", "🔮 世界模型", "⚡ Scoring", "🧩 JEPA", "🎮 GRPO", "📊 学习指南"]
weight: 1
summary: "一份面向自动驾驶从业者的全景式学习指南。系统梳理六大技术范式（端到端E2E、VLA、世界模型、Scoring规划、JEPA、RL+生成）的核心概念、代表工作、联系脉络和知识依赖。附自测清单和推荐阅读路线，帮助你定位自己的知识短板。"
---

## 📍 为什么需要这份指南

2026 年的自动驾驶技术已经不再是"模块化 vs 端到端"的二元争论。VLA、世界模型、Scoring-based 规划、JEPA、Flow-GRPO……新技术范式在短短两年内集中爆发，每一条路线都有自己完整的论文谱系、工程实践和社区生态。

对从业者来说，问题已经不是"该学什么"，而是**"太多东西要学，不知道该从哪里入手"**。

这份指南的目标不是替代已有的知识点拆解文章（博客里每篇都写得很细了），而是提供一份**导航地图**——告诉你每个知识点在整体图景中的位置、它依赖什么前置知识、它能通向什么前沿方向。

---

## 🧱 第一层：基础基石

这些是理解所有上层范式的前置知识，**绕不过去**。

### 1.1 Transformer 与注意力机制

这是所有现代自动驾驶模型的底层构件。从 UniAD 到 VLA 到世界模型，无一例外。

**必须搞懂的核心概念：**
- Scaled Dot-Product Attention 的计算流程
- Cross-Attention 在模态融合中的作用（图像 token 作为 Query，场景 token 作为 Key/Value）
- KV-Cache 加速推理的原理
- Positional Encoding（绝对位置 vs RoPE vs ALiBi）

**博客文章**：[Transformer与注意力机制详解](/posts/knowledge/transformer与注意力机制详解/)

**自测**：能写出 Cross-Attention 的前向代码，并解释 $O(n^2)$ 复杂度从哪来。

### 1.2 BEV 感知

BEV（Bird's Eye View）是 2022-2025 年自动驾驶感知的事实标准。即使最新的稀疏化方案（SparseDrive）也在师承 BEV 的思想。

**必须搞懂的核心概念：**
- LSS（Lift-Splat-Shoot）视角变换
- Transformer-based BEV 融合（BEVFormer 的 Temporal Self-Attention）
- BEV 特征 vs 稀疏 query 的 trade-off

**博客文章**：[BEV感知技术详解](/posts/knowledge/bev感知技术详解/)

**自测**：能解释"为什么 BEV 特征比图像特征更适合做规划输入"。

### 1.3 扩散模型与 Flow Matching

生成式范式（VLA 的动作头、世界模型的视频生成、Scoring 的候选采样）全部建立在这两个技术上。

**必须搞懂的核心概念：**
- DDPM 的前向加噪和反向去噪过程
- Flow Matching 的直线轨迹 vs 扩散的弯曲轨迹
- Classifier-Free Guidance（CFG）的原理
- ODE vs SDE 采样

**博客文章**：[扩散模型基础详解](/posts/knowledge/扩散模型基础详解/) + [Flow-Matching入门详解](/posts/knowledge/flow-matching入门详解/)

**自测**：能数学上写出 Flow Matching 的条件向量场损失函数。

### 1.4 强化学习基础

从 Flow-GRPO 到 DiffGRPO 到 Dreamer V3，RL 正在渗透自动驾驶的每一个角落。

**必须搞懂的核心概念：**
- Policy Gradient 定理和 REINFORCE 算法
- PPO 的 clipped objective 和 advantage 估计
- On-policy vs Off-policy 的核心差异
- GRPO 的组内优势计算（不需要 critic）

**博客文章**：[强化学习基础详解](/posts/knowledge/强化学习基础详解/) + [GRPO强化学习方法详解](/posts/knowledge/grpo强化学习方法详解/)

**自测**：能推导 PPO 的 ratio 公式，并解释为什么 GRPO 不需要 critic。

### 1.5 VLM 与视觉语言对齐

VLA 的基础——如果不理解 VLM，就无法理解 VLA。

**必须搞懂的核心概念：**
- CLIP 的对比学习目标和双塔架构
- LLaVA 的视觉投影层和指令微调范式
- 多模态 token 拼接和自回归生成

**博客文章**：[VLM视觉语言模型入门详解](/posts/knowledge/vlm视觉语言模型入门详解/)

### 1.6 评测体系

没有评测就没有进步。NAVSIM、nuScenes、Bench2Drive 的指标设计直接影响方法论走向。

**必须搞懂的核心概念：**
- 开环（Open-Loop）vs 闭环（Closed-Loop）的根本区别
- PDMS 的五维指标：NC(碰撞) × DAC(可行驶区域) × (5TTC + 2C + 5EP)/12
- EPDMS 的扩展：新增 DDC(方向)/TL(红绿灯)/LK(车道保持)/EC(扩展舒适)
- Bench2Drive 的 Driving Score 和 Success Rate

**博客文章**：[NAVSIM基准详解](/posts/knowledge/navsim基准详解/) + [开环评测vs闭环评测深度解析](/posts/knowledge/开环评测vs闭环评测深度解析/)

---

![自动驾驶知识体系依赖图谱](/images/guide/knowledge_tree.svg)

---

## 🏛️ 第二层：六大核心范式

### 范式一：端到端自动驾驶（E2E）

**一句话定位**：用一个大网络打通传感器到控制，消除模块间信息损失。

**发展脉络**：

从 2016 年 NVIDIA PilotNet 的"单目图像→方向盘"的直接回归开始，E2E 经历了三个阶段的演进：

1. **隐式 E2E（2016-2022）**：CNN 直接回归控制量，黑盒、难训练、效果不稳定。代表：PilotNet、Conditional Affordance Learning、Learning by Cheating。

2. **显式 E2E（2023-2024）**：网络内部保留感知/预测/规划的任务结构，但梯度贯通联合优化。UniAD（CVPR 2023 Best Paper）是开山之作。VAD 实现了向量化效率革命。SparseDrive 把稀疏化做到极致。

3. **生成式 E2E（2024-2026）**：引入扩散模型做多模态轨迹生成。DiffusionDrive 用 Flow Matching 替代确定性回归，DiffusionDrive V2 引入 RL 约束增强安全性。

**关键论文序列**：
```
UniAD(2023) → VAD(2024) → SparseDrive(2025) → DiffusionDrive(2025)
→ DiffusionDrive V2(2025) → SparseDriveV2(2026)
```

**与其他范式的关系**：
- E2E → 进化出 **VLA**（加入 LLM 推理能力）
- E2E → 需要 **World Model** 提供数据增强和闭环训练环境
- E2E × **Scoring** = SparseDriveV2（生成候选 + 评分选优）

**推荐博客文章**：[端到端自动驾驶演进](/posts/knowledge/端到端自动驾驶演进/) + [VAD向量化端到端精读](/posts/paper-reading/vad向量化端到端精读/)

---

### 范式二：VLA（Vision-Language-Action）

**一句话定位**：给 E2E 装上 LLM 的"脑子"，让系统边推理边开车。

**发展脉络**：

VLA 经历了动作表征的三代演进：

1. **G1 离散 Token（2023-2024）**：RT-2 把动作离散化成 256 个 bin 当作文字生成。精度受限于量化误差，但证明了"VLM 可以直接输出动作"的可行性。

2. **G2 扩散/Flow Matching（2024-2025）**：π0 用 Flow Matching 替代离散 tokenization，10 步采样即达 30Hz 控制频率。Octo 开源了扩散策略标杆。

3. **G3 CoT 推理（2025-2026）**：GR-3 引入 Chain-of-Thought 推理（"看到行人→判断意图→规划制动→输出动作"），碰撞率降低 45%。EPM 把规划显式嵌入 token 生成过程。BridgeVLA 用"桥接监督"弥合预训练和微调之间的 gap。

**动作头的三角权衡**：

| 方案 | 精度 | 速度 | 多模态支持 | 代表 |
|------|------|------|-----------|------|
| 离散 Token | 中（量化误差） | 慢（自回归） | 弱 | RT-2, OpenVLA |
| 连续回归 | 中（mode averaging） | **快** | 弱 | 早期 SFT 头 |
| 扩散/Flow Matching | **高** | 中（可优化） | **强** | π0, Octo, DiffusionDrive |

**三阶段训练管线**：

```
VLM预训练(图文对齐) → SFT(图+指令+动作) → RLHF/GRPO(偏好对齐)
```

**2026 年最新进展**：
- **VLA-World**（CVPR 2026）：统一 VLA 预测想象与反思推理——用动作引导未来帧生成，再用 GRPO 优化。
- **Orion**：端到端 VLA，语言指令直接驱动轨迹生成。
- **OpenDriveVLA**：开源 VLA 驾驶模型，在 nuScenes 上 SOTA。

**推荐博客文章**：[什么是VLA模型](/posts/knowledge/什么是vla模型/) + [前沿VLA模型全景速览](/posts/knowledge/前沿vla模型全景速览/)

---

### 范式三：世界模型（World Model）

**一句话定位**：学会"世界怎么运作"，在脑子里预演未来。

**三大预测空间**：

| 空间 | 信息密度 | 优点 | 缺点 | 代表 |
|------|---------|------|------|------|
| 像素空间 | 最高 | 可可视化、数据增强 | 计算量大、不保证物理一致 | GAIA-1, Cosmos |
| 潜在空间 | 中 | 计算高效、可服务规划 | 不可直接可视化 | World4Drive, ReWorld |
| 占用空间 | 低（语义级） | 显式碰撞检测 | 无纹理细节 | OccWorld, NIFF |

**三大应用场景**：

1. **数据增强**：生成训练集中稀缺的组合场景（如夜间暴雨+行人横穿）。代表：DriveDreamer、MagicDrive。

2. **闭环仿真**：根据自车动作实时生成下一帧，实现互动式训练。代表：Vista（DiT 骨干 + 动作条件）。

3. **Model-based Planning**：在推演中选最优轨迹。代表：Drive-WM（世界模型+MPPI）、ReWorld（重建误差即 reward）。

**与其他范式的关系**：
- World Model → 为 **VLA** 提供"想象力"（VLA-World 路线）
- World Model → 为 **E2E** 提供训练数据和闭环仿真环境
- World Model × **JEPA** = 表征式世界模型（在隐空间预测，不重建像素）

**推荐博客文章**：[什么是世界模型](/posts/knowledge/什么是世界模型/) + [驾驶世界模型全景详解](/posts/knowledge/驾驶世界模型全景详解/)

---

### 范式四：Scoring-based 规划

**一句话定位**：生成多个候选轨迹，学一个评分器选最好的。判别比生成更可靠。

**三大范式对比**：

| 范式 | 核心思路 | 优点 | 缺点 | 代表 |
|------|---------|------|------|------|
| Vocabulary-based | 预定义轨迹模式库 | 推理快、可解释 | 覆盖有限 | VADv2, Hydra-MDP |
| Dynamic Generation | CEM 自适应采样 | 覆盖任意轨迹 | 推理慢 | TOAD, DrivoR |
| Diffusion-based | 扩散生成多样性候选 | 质量高、模式多 | 多步采样慢 | DiffusionDrive |

**SparseDriveV2 的关键突破**（2026.03）：

清华 + 地平线证明了一个反直觉的结论：**"Scoring is All You Need"**——只要词汇表够大、评分器够好，静态词汇表可以匹敌甚至超越动态生成方法。

- **因子化词汇表**：1024 路径 × 256 速度 = 262,144 候选
- **层级评分**：coarse scorer 粗筛 top-128×top-64 → fine scorer 精评 400 条
- **ResNet-34 轻量骨干**，NAVSIM PDMS 92.0，Bench2Drive DS 89.15

**Scorer 损失函数**：PDM label（开环 approach）→ Cycle Energy（自监督 approach，JEPA-DRIVE）

**推荐博客文章**：[Scoring-based规划范式详解](/posts/knowledge/scoring-based规划范式详解/)

---

### 范式五：JEPA（联合嵌入预测架构）

**一句话定位**：在隐空间做预测，不碰像素。让模型学"会发生什么"，而非"画得像什么"。

**核心三件套**：

| 组件 | 作用 | 参数更新 |
|------|------|---------|
| Context Encoder | 编码已知信息到隐空间 | 梯度下降 |
| Target Encoder | 编码待预测目标作为答案 | EMA（动量更新）+ stop-gradient |
| Predictor | 从已知预测未知的隐空间表征 | 梯度下降 |

**为什么在隐空间预测？**

传统方法（MAE/Diffusion）在像素空间预测，模型可以作弊——用周围像素颜色模糊填充，不需要理解"被遮的是什么"。JEPA 把预测搬到隐空间，**天然丢弃了像素级纹理细节，迫使模型理解语义**。

**演进路线**：

```
I-JEPA(2023,图像) → V-JEPA(2024,视频) → V-JEPA 2(2025,世界模型)
                                                    ↓
                                            DRIVE-JEPA(2026,驾驶特征提取)
                                            JEPA-DRIVE(2026,自监督评分)
```

**两个自动驾驶 JEPA 路线的区别**：

| 维度 | DRIVE-JEPA（XPENG） | JEPA-DRIVE（我们的项目） |
|------|--------------------|------------------------|
| JEPA 角色 | 视觉编码器（特征提取） | 世界模型 + 自监督评分信号 |
| 决策范式 | 端到端模仿学习 | 推演式决策（生成+评分+选择） |
| 评分信号 | PDM 标签 | Cycle Energy 自监督 |
| 参数量 | ~300M+ | 3.88M |

**推荐博客文章**：[JEPA联合嵌入预测架构详解](/posts/knowledge/jepa联合嵌入预测架构详解/)

---

### 范式六：RL + 生成式规划

**一句话定位**：用强化学习优化生成式策略，让模型超越"模仿人类"的上限。

**为什么需要 RL？**

模仿学习（Behavior Cloning）有三个根本局限：
1. **分布偏移**：模型自己输出偏离训练数据时，没有信号拉回来
2. **无法超越专家**：上限就是训练数据的水平
3. **目标不可微**：碰撞率、舒适性无法写成监督标签

RL 的解法：**优化"结果好不好"，不是"像不像数据"**。

**四大范式谱系**：

| 范式 | 方法 | 代表 |
|------|------|------|
| 1️⃣ Reward Guidance | 推理时用奖励梯度引导采样 | Diffusion Planning |
| 2️⃣ Rejection Sampling | 生成 N 条候选，挑最好的训练 | TrajRL, RGT |
| 3️⃣ Policy Gradient | 通过 log-prob 把奖励反馈到生成网络 | Flow-GRPO, DiffGRPO |
| 4️⃣ World Model RL | 在世界模型的"想象"中训练策略 | Dreamer V3, Drive-WM |

**Flow-GRPO 的核心改造**：

把 GRPO 从离散 token（语言模型）迁移到连续去噪（Flow Matching），需要三个关键改造：

1. **去噪转移 = action**：从 $x_t$ 到 $x_{t-1}$ 的 latent 转移作为 RL 的 action
2. **ODE → SDE**：纯 ODE 确定性的，没有概率密度就写不出 log-prob。注入随机性：$x_{t-1} = \mu(x_t, t) + \sigma \cdot \epsilon$
3. **终点 reward 广播**：最终结果的 reward 复制到每个去噪步的 log-prob ratio 上

**加速技巧**：Flow-GRPO-Fast 只训练 1-2 个窗口步，速度数倍提升；GRPO-Guard 用 RatioNorm + Gradient Reweight 防过优化。

**推荐博客文章**：[Flow-GRPO详解](/posts/knowledge/flow-grpo详解/) + [PPO算法深度拆解](/posts/knowledge/ppo算法深度拆解/)

---

![六大范式全景关系图](/images/guide/paradigm_overview.svg)

---

## 🔗 第三层：范式间的联系

这六大范式不是孤立的，它们之间有**依赖、进化、互补**三重关系。

### 依赖关系

```
Transformer/Attention ──────┬────→ E2E(UniAD/VAD)
                            ├────→ VLA(LLM backbone)  
                            ├────→ World Model(DiT/Transformer decoder)
                            ├────→ Scoring(ScoreNet CrossAttn)
                            └────→ JEPA(ViT backbone)

扩散/Flow Matching ───────┬────→ VLA(动作头: π0/bridgeVLA)
                           ├────→ World Model(视频生成: Cosmos/Vista)
                           ├────→ Scoring(扩散候选采样)
                           └────→ RL(Flow-GRPO 的去噪链)

RL(Policy Gradient) ───────┬────→ Flow-GRPO(流匹配策略优化)
                            ├────→ DiffGRPO(扩散策略优化)
                            ├────→ Dreamer V3(世界模型+RL)
                            └────→ Scoring(RL fine-tune ScoreNet)
```

### 进化关系

```
E2E(确定性回归) ──→ VLA(加入LLM推理) ──→ VLA+World Model(推演+决策)
                      ↑                    ↑
Scoring(静态词汇表) ──→ SparseDriveV2(因子化词汇表+层级评分)
                      ↓
JEEP(PDM监督评分) ──→ JEPA-DRIVE(自监督评分)
```

### 互补关系

| 范式组合 | 解决的问题 | 代表工作 |
|---------|-----------|---------|
| VLA + 世界模型 | VLA 缺乏"预言"能力 | VLA-World (CVPR 2026) |
| E2E + Scoring | E2E 的多模态困境 | SparseDriveV2 |
| 世界模型 + RL | 模仿学习上限 | Dreamer V3 |
| JEPA + Scoring | PDM 标签依赖 | JEPA-DRIVE |
| Flow-GRPO + VLA | 流匹配策略的 RL 优化 | DiffGRPO / ReCogDrive |

---

## 📄 第四层：论文时间线与关键脉络

### 2023 年：基础奠基

| 论文 | 范式 | 机构 | 为什么重要 |
|------|------|------|-----------|
| I-JEPA | JEPA | Meta | 首个 JEPA 实现，证明隐空间预测有效 |
| UniAD | E2E | 清华+商汤 | CVPR Best Paper，定义显式 E2E 范式 |
| RT-2 | VLA | Google | 首个 VLM→VLA，证明可行性 |
| GAIA-1 | WM | Wayve | 首个十亿参数驾驶世界模型 |

### 2024 年：范式爆发

| 论文 | 范式 | 机构 | 为什么重要 |
|------|------|------|-----------|
| VAD | E2E | 华中科大 | 向量化效率革命 |
| V-JEPA | JEPA | Meta | 扩展到视频，时空掩码 |
| π0 | VLA | Physical Int. | Flow Matching 动作头标杆 |
| DriveDreamer | WM | 上海 AI Lab | 结构化条件可控生成 |
| DriveVLM | VLA | 上交+蔚来 | 首个驾驶 VLM 思维链 |
| OccWorld | WM | 上海 AI Lab | 首个占用空间世界模型 |
| VADv2 | Scoring | 华中科大 | 轨迹词汇表概念 |
| Hydra-MDP | Scoring | — | 多头规划器+独立评分器 |
| DiffusionDrive | E2E+Scoring | 华中科大 | 扩散模型规划头 |
| SparseDrive | E2E | 清华+地平线 | 稀疏化极致设计 |

### 2025 年：工程深化

| 论文 | 范式 | 机构 | 为什么重要 |
|------|------|------|-----------|
| V-JEPA 2 | JEPA | Meta | 自回归预测器，物理世界基础模型 |
| V-JEPA 2-AC | JEPA | Meta | 动作条件世界模型，零样本机器人规划 |
| GR-3 | VLA | Google | CoT 推理驱动，碰撞率降 45% |
| Cosmos | WM | NVIDIA | 2000 万小时视频预训练 |
| SparseDriveV2 | Scoring | 清华+地平线 | "Scoring is All You Need" |
| Flow-GRPO | RL+Gen | 字节跳动 | 流匹配+GRPO 范式 |
| GenAD | E2E | — | 生成式端到端框架 |

### 2026 年：融合收敛

| 论文 | 范式组合 | 机构 | 为什么重要 |
|------|---------|------|-----------|
| DRIVE-JEPA | JEPA+E2E | XPENG+VT+Purdue | V-JEPA 预训练+轨迹蒸馏，NAVSIM SOTA |
| JEPA-DRIVE | JEPA+Scoring | 我们 | Cycle Energy 自监督评分，3.88M 参数 |
| VLA-World | VLA+WM | 上交+华为 | CVPR 2026，统一预测想象与推理 |
| EMMA | VLA | Wayve | Gemini 风格多任务端到端 |
| BridgeVLA | VLA | Stanford | 桥接监督弥合预训练-微调 gap |
| EPM | VLA+Planning | MIT | 规划即推理，多步状态预测 |
| SparseDriveV2 | Scoring+E2E | 清华+地平线 | PDMS 92.0, Bench2Drive DS 89.15 |

---

## 🧭 第五层：学习路线推荐

### 按角色推荐

**如果你是感知背景出身**：
```
BEV感知 → Occupancy Network → SparseDrive(稀疏化) → 世界模型(占用空间)
→ JEPA(表征式世界模型的另一种思路)
```

**如果你是规划/控制背景**：
```
端到端演进 → Scoring-based规划(SparseDriveV2精髓) → VLA动作头设计 
→ Flow-GRPO(用RL优化规划策略)
```

**如果你是 LLM/VLM 背景**：
```
VLM基础 → VLA训练管线(SFT+RLHF) → CoT推理(GR-3/EPM) 
→ VLA+世界模型融合(VLA-World)
```

**如果你是系统工程/部署背景**：
```
工业方案对比(FSD/Momenta/华为/小鹏) → 影子模式/数据飞轮 
→ 车端部署优化(量化/KV-cache/action chunking)
```

### 按时间投入推荐

**1 周入门**：E2E演进 + VLA基础 + 世界模型概念 + 评测体系

**1 个月系统学**：以上 + VAD精读 + π0/DriveVLM精读 + Scoring范式详解

**3 个月深入**：以上 + JEPA详解 + Flow-GRPO源码 + SparseDriveV2复现 + JEPA-DRIVE理解

**6 个月专家**：以上 + 能梳理任意新论文的范式归属 + 能设计新范式组合方案

---

## 📋 第六层：知识自测清单

每道题按表计分，**60 分及格，80 分良好，90 分以上是专家水平**。

### 基础层（每题 2 分，共 20 分）

- [ ] 能写 Transformer 的 Cross-Attention 前向代码
- [ ] 能解释 BEVFormer 的 Temporal Self-Attention 作用
- [ ] 能对比扩散模型和 Flow Matching 的核心公式差异
- [ ] 能推导 PPO 的 clipped objective
- [ ] 能对比开环评测和闭环评测的优缺点
- [ ] 能解释 PDMS 指标的五维组成
- [ ] 能说出 CLIP 的训练目标和双塔架构
- [ ] 能解释 LSS 视角变换的核心步骤
- [ ] 能说出 GRPO 相比 PPO 的关键差异
- [ ] 能解释 CFG（Classifier-Free Guidance）的作用

### 核心层（每题 4 分，共 40 分）

- [ ] 能梳理端到端三代架构（隐式/显式/生成式）的演进脉络
- [ ] 能比较 UniAD / VAD / SparseDrive 三者的核心设计差异
- [ ] 能比较三种 VLA 动作头（离散/回归/扩散/Flow）的 trade-off
- [ ] 能说出世界模型三大预测空间及其适用场景
- [ ] 能对比 Scoring 三大范式（Vocabulary/Dynamic/Diffusion）的优劣
- [ ] 能解释 SparseDriveV2 的因子化词汇表和层级评分设计
- [ ] 能画出 JEPA 三件套（Context Encoder/Target Encoder/Predictor）架构
- [ ] 能说出 JEPA 为什么不需要数据增强（对比 SimCLR/BYOL）
- [ ] 能对比四种 RL+生成范式（Reward Guidance/Rejection/Policy Gradient/World Model RL）
- [ ] 能解释 Flow-GRPO 的三个核心改造（action=SDE/latent transition/log-prob）

### 进阶层（每题 5 分，共 25 分）

- [ ] 能对比 FSD / Momenta / 华为 ADS / 小鹏 XNGP 四家方案的核心差异
- [ ] 能说出 VLA-World (CVPR 2026) 的"预测想象+反思推理"具体怎么做
- [ ] 能对比 DRIVE-JEPA 和 JEPA-DRIVE 的 JEPA 角色差异
- [ ] 能解释 Cycle Energy 和 JEPA 预测误差的关系
- [ ] 能说出 Flow-GRPO-Fast 和 GRPO-Guard 的加速/防过优化原理

### 专家层（每题 5 分，共 15 分）

- [ ] 能用一个框架统一解释 E2E / VLA / World Model / Scoring / JEPA / RL+Gen 六个范式的关系
- [ ] 能判断一篇新论文属于哪个范式，并说出它解决了什么核心问题
- [ ] 能为一个具体问题（如"重卡VLA"）设计范式组合方案

### 评分对照

| 分数 | 定位 | 建议行动 |
|------|------|---------|
| ≤40 | 基础薄弱 | 先读博客的知识点拆解系列，打牢 Transformer/BEV/RL/VLM 基础 |
| 40-60 | 入门 | 读 E2E 演进 + VLA 详解 + 世界模型全景三篇 |
| 60-80 | 良好 | 深入 Scoring/JEPA/Flow-GRPO 三篇精读，做论文对照表 |
| 80-90 | 优秀 | 能复现关键论文思路，可指导范式选择 |
| ≥90 | 专家 | 你在创造知识，不是在学习知识 |

---

## 🔮 未来趋势判断

### 已确认的趋势（共识级）

1. **Scoring 范式成为规划主流**：SparseDriveV2 证明"静态词汇表+层级评分"可以匹敌动态生成。JEPA-DRIVE 证明评分可以自监督，不需要 PDM。

2. **VLA + 世界模型走向融合**：VLA-World (CVPR 2026) 是标志性工作——VLA 做推理，世界模型做想象，共享骨干。

3. **RL 成为 Post-training 标配**：Flow-GRPO → DiffGRPO → ReCogDrive，RL 正在从图像生成渗透到驾驶策略优化。

4. **表征式世界模型崛起**：JEPA 路线证明了在隐空间预测的有效性，"不重建像素"正在成为世界模型的新共识。

### 仍在争议的方向

1. **像素 vs 潜在 vs 占用**：三者各有适用场景，尚未出现"赢家通吃"。

2. **端到端 One-Model vs 模块化**：FSD 走 one-model，其他三家走"端到端化模块"，谁最终胜出取决于安全验证体系。

3. **仿真数据 vs 真实数据**：Tesla 走仿真为主（65%），Momenta 走真实为主。数据策略的分化在扩大。

4. **CoT 推理的价值**：CoT 在碰撞率上提升明显（GR-3 降低 45%），但引入了"幻觉推理"的新失败模式（~1.3%）。是否值得额外延迟？

### 个人判断

1. **3.88M vs 7B 不是参数量的竞争，是范式的竞争**。JEPA-DRIVE 证明：如果你改变范式（从"映射"到"选择"），小模型也能做大事。

2. **自监督评分是 2026 年的关键突破口**。Cycle Energy（JEPA-DRIVE）和 ReWorld（重建误差即 reward）印证了同一个方向——评分信号可以从数据中自监督地产生。

3. **2027 年的方向是"三层融合"**：VLA（推理层）+ 世界模型（推演层）+ Scoring/RL（优化层）三层融合，共享隐空间表征，形成真正像人一样开车的系统。

---

## 📚 参考文献索引

本文依赖博客已有的系列文章体系。建议按以下顺序阅读：

**基础系列**：
1. Transformer与注意力机制详解
2. BEV感知技术详解
3. 扩散模型基础详解 + Flow-Matching入门详解
4. 强化学习基础详解 + GRPO强化学习方法详解
5. VLM视觉语言模型入门详解

**核心范式系列**：
6. 端到端自动驾驶演进 + VAD向量化端到端精读
7. 什么是VLA模型 + 前沿VLA模型全景速览
8. 什么是世界模型 + 驾驶世界模型全景详解
9. Scoring-based规划范式详解
10. JEPA联合嵌入预测架构详解
11. Flow-GRPO详解

**产业实践系列**：
12. 主流自动驾驶方案对比（FSD/Momenta/华为/小鹏）
13. NAVSIM基准详解 + NAVSIM排行榜深度分析
14. 端到端驾驶评测指标全景

---

*📖 本文是对博客已有知识体系的一次全景梳理与路线图绘制。如果你能答出 80/100 分自测题，说明你已经掌握了 2026 年自动驾驶技术的核心知识；如果你能答出 90+，你就是这个领域的专家。*