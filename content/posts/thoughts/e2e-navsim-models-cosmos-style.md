---
title: "纯视觉端到端模型在 Cosmos 风格迁移加持下的 NAVSIM 大规模评测：11 个开源模型全景"
date: 2026-08-07
draft: false
description: "课题组新 benchmark：用 Cosmos 做雨/雪/晴风格迁移构建 NAVSIM 视觉压力测试，批量评测 11 个纯视觉（camera-only）端到端规划模型。本文系统梳理 SparseDriveV2、iPad、DrivoR、ChainFlow-VLA、AutoVLA、ReCogDrive、DriveVLA-W0、Drive-JEPA、DriveSuprim、DriveLaW、LTF 的论文架构（附 arXiv 原文架构图）、开源资源、评测配置与 NAVSIM PDMS/EPDMS 成绩。"
categories: ["个人思考"]
tags: ["NAVSIM", "端到端规划", "纯视觉", "Cosmos", "风格迁移", "VLA", "世界模型", "评测基准", "PDMS"]
math: true
---

## 0. 写在前面：我们需要一份"会晒黑的评测"

现阶段启用的 benchmark 大多默认天气晴朗、光照良好、纹理干净，视觉端到端模型的感知在这个"温室"里表现优秀。但当复用习惯了晴天的视觉 planner 进入真实世界——阴雨路面积水反光、雪天车道线被覆盖、黄昏光照死黑——它的 No-at-fault Collision 和 Drivable Area Compliance 往往断崖式下跌。**"能不能开"和"换了个风格还能不能开"是两种能力。**

课题组这一轮想做的，就是用 **Cosmos（NVIDIA 的世界基础模型）做光照/天气/纹理的风格迁移**（rain / snow / 日夜 / 模糊），在不改变场景拓扑、导航指令和 agents 语义的前提下，把同一份 NAV 场景"换皮"，得到一组**视觉域被扰动、但语义标签不变**的评测集，然后再跑纯视觉 end-to-end 模型在 NAV 上的 PDMS。这样我们拿到的不只是一个数字，而是一张"模型在位姿、外观扰动下的鲁棒性画像"，可以回答：**哪个模型在风格漂移下最硬？哪个只在晴天强？**

### 0.1 选型逻辑：为什么是这 11 个

入选标准明确，缺一不可：

1. **camera-only（纯视觉）**：必须只吃相机输入，不能依赖 LiDAR 点云。这样"风格迁移只改图像"才能干净地、真实地作用到模型输入上。
2. **NAV 可跑**：官方有 NAV 相关权重或可复现配置，能进入我们的批量评测管线（Agent API 封装）。
3. **论文开源**：能从 arXiv 拿到原文架构与打分细节。
4. **覆盖技术谱系**：从规则化生成型（LTF）到高分辨率稀疏语义 (SparseDriveV2)、proposal 中心式（iPad）、扩散/飞流式（DrivoR、Drive-JEPA、DriveLaW）、再到 VLA 语言世界模型（ChainFlow-VLA、AutoVLA、ReCogDrive、DriveVLA-W0）——我们想测的是"在哪一种范式上，风格迁移造成的伤害最小"。

于是，最终敲定了下面 11 个：**SparseDriveV2、iPad、DrivoR、ChainFlow-VLA、AutoVLA、ReCogDrive、DriveVLA-W0、Drive-JEPA、DriveSuprim、DriveLaW、LTF（Latent TransFuser）**。

下面我会给每个模型：**一句话定位 → 架构解读（附 arXiv 原文架构图）→ 官方资源（仓库/HF）→ 评测配置 → NAV 得分 → 在我们风格迁移评测里值得画的重点**。

---

## 1. SparseDriveV2（感知-规划稀疏联合）

> **一句话定位**：把高阶"时空轨迹"分解为"几何路径 + 速度谱"，用稀疏 vocab 打分选出最优，属于 generation/selection 范式的稀疏语义代表。

- **arXiv**: 2603.29163（修正：原表误写 2603.29159）
- **官方仓库**: [github.com/swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2)
- **权重**: HuggingFace `wenchaosun/SparseDriveV2`
- **论文图 1（架构）**：

{{< figure src="/images/e2e-navsim-models/sparsedrivev2_arch.png" title="SparseDriveV2 总体架构：轨迹因式分解为几何路径与速度谱，再重建轨迹（arXiv 2603.29163 Figure 1）"  width="100%" >}}

### 架构解读

SparseDriveV2 的核心是把一条端到端轨迹 $\tau = \{(x_t, y_t)\}_{t=1}^{T}$ 做**因式分解**：

$$
\tau \xrightarrow{\mathcal{D}} (p, v),\quad p=\{(x_i,y_i)\}_{i=1}^S,\ v=\{v_t\}_{t\in T}
$$

- 路径 $p$ 是空间上的锚点序列，速度谱 $v$ 是逐时刻速度标量；合成器 $\mathcal{C}(p,v)=\tau$ 可无损重建轨迹。
- 稀疏 vocab 由 $N_p$ 个 path 锚点 与 $N_v$ 个 velocity 锚点张成，组合出候选轨迹集 $\mathcal{T}=\{\mathcal{C}(p_i,v_j)\}$。
- 打分体系是三重（score-path / score-vel / score-traj）加权，最终 `Softmax(-λ·d)` 的路径、速度、轨迹损失 + 语义度量 BCE 一起训。

### 评测配置与得分

- **NAVSIM v1 (navtest) PDMS：92.0**（ResNet-34 后端）——表格里高于 DiffusionDrive 88.1、DriveSuprim（R34）89.9，与 iPad 91.7 略高/持平。
- **NAVSIM v2 EPDMS**: 随着路径锚点数从 1024→16384，EPDMS 从 85.02 提升到 **87.35**（内存从 9.5GB 涨到 38.9GB），说明 factorized vocab 规模直接决定上限。

| #Anchors | EPDMS | Memory (MB) |
|---|---|---|
| 1024 | 85.02 | 9531 |
| 4096 | 86.33 | 15513 |
| 16384 | **87.35** | 38877 |

**风格迁移视角**：SparseDriveV2 是稀疏语义的强代表，但它的前置感知依赖网络是否在高泛化性上保持语义稳定（路径锚点对车道线/障碍遮挡较敏感），在风格偏移下"几何路径"需要持续被推到正确区域，值得重点观察。

---

## 2. iPad（Iterative Proposal-centric，自拍式规划）

> **一句话定位**：把"一次性生成"换成了"用 proposal 迭代自拍"——场景编码器 + ProFormer（proposal-centric BEV）+ 多轮 refinement，NAV 上 PDM 随 proposal 数/迭代数/数据对数增长。

- **arXiv**: 2505.15111
- **官方仓库**: [github.com/Kguo-cs/iPad](https://github.com/Kguo-cs/iPad)
- **权重**: 官方提供 Google Drive 下载（仓库内入口）
- **论文图 2（框架总览）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/ipad_overview.png" alt="iPad 框架" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">iPad 框架：Scene Encoder（灰）+ ProFormer（蓝）+ proposal 迭代（arXiv 2505.15111 Figure 2）</figcaption>
</figure>

- **论文图 5（ProFormer 详解）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/ipad_proformer.png" alt="ProFormer 架构" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">ProFormer 详细架构：proposal 作为 query 对 proposal-centric 图像特征做 deformable 交叉注意力更新（arXiv 2505.15111 Figure 5）</figcaption>
</figure>

### 架构解读

iPad 把 end-to-end 规划从"密集 grid-centric 一步出" 改成 **proposal-centric + 多轮迭代**：

1. **Scene Encoder** 取多相机图 + 自车状态，输出场景特征。
2. **ProFormer**：先初始化 $N$ 个 BEV proposal（即位姿/速度），再把 proposal 当 query 对"proposal-centric 图像特征"（yellow branch）做 deformable attention，**迭代 $K$ 轮**更新位姿和 proposal 特征。
3. 每轮都会推出一系列候选轨迹并在 score 头上评分，多轮后逐步逼近可行解——类似于收敛迭代，但对象是"位姿子集"（proposal）。
4. 最终选分最高的轨迹作为输出。

关键超参数（论文原文）：proposal 数 $N=64$、迭代数 $K=4$、规划步长间隔 0.5s、通道 256。

### 评测配置与得分

- **NAVSIM v1（camera, ResNet-34）**: 各子分数 NC 98.6 / DAC 98.3 / TTC 94.9 / Comf. 100 / EP 88.0，**PDMS 91.7**。

结果：proposal 数/迭代数/训练数据三者都呈对数增长 PDM——论文 Figure 3 的 *Scaling Law*。消融里"加 Proposal Refinement + ProFormer + proposal-centric BEV + proposal-centric prediction"逐步把 PDMS 从 78.5 拉 91.7。

**风格视角**：iPad 的 proposal self-correction 依赖图像里的可行驶区域和交互物件都能被 deformable attention 稳定 Query 到，一旦风格赶走了车道线/交通灯语义，iteration 是否有"收敛容错"值得重点看——它可能是"对扰动鲁棒"的最大黑马。

---

## 3. DrivoR（轨迹与打分双解码端 Transformer）

> **一句话定位**：纯视觉的标准 three-block（1 编码 + 2 解码）Transformer，用 DINOv2 初始化 + scene tokens / sensor registers + 多轨迹最优点成绩，NAV 跑的 state-of-the-art 纯视觉基线。

- **arXiv**: 2601.05083
- **官方仓库**: [github.com/valeoai/DrivoR](https://github.com/valeoai/DrivoR)
- **权重**: GitHub Releases（仓库内入口）
- **论文图 1（架构）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/drivor_arch.png" alt="DrivoR 架构" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">DrivoR 架构：一个感知编码器（含 sensor registers）+ 轨迹解码器 + 打分解码器（arXiv 2601.05083 Figure 1）</figcaption>
</figure>

- **论文图 2（编码/解码块）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/drivor_blocks.png" alt="DrivoR 感知块" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">Encoder/Decoder 块内含 sensor registers 作为 scene token（arXiv 2601.05083 Figure 2）</figcaption>
</figure>

### 架构解读

- **感知编码器**：DINOv2 初始化卷积 ViT + Registers。Encoder 输出一组通用的 **scene tokens（≈20-64/场景，可压缩）**。
- **轨迹解码器**：自回归/多 token 输出候选轨迹。
- **打分解码器**：与轨迹头解耦（disentangled），输出每个子度量（NC/DAC/TTC/C/EP）的 score，用于选优——"disentangled scoring"是一套**候选生成 + 打分选择**的 scorer 结构。

关键发现（论文 Table 4）：压缩为 registers 的 16/64 个 token 的性能逼近 250x 很多 token；初始化 DINOv2 比 ImageNet 21k 好；多轨迹 WTA（winner-take-all）or multi-target（目标 $t{T+T}$ 与 $t{T+T'}$ 联合）进一步提升。

### 评测配置与得分

DrivoR 的关键结论来自其对 **navval（val set）** 的报道：

| 初始化 | Random | ImageNet 21k | DINOv2 |
|--------|---------|--------------|--------|
| PDMS (navval) | 70.1 | 87.5 | **90.0** |

- **NAVSIM v2 navhard-two-stage（EPDMS）**: DrivoR (ViT-S) **45.3**；加 185k SimScale 数据可达 **52.3**。
- 场景 token 数影响：16/64 场景 token 性能逼近 250 倍 token 数。
- 轨迹数：1→8→64→128，PDMS 80.1 到 90.0（大轨迹数收益递减，128 封顶）。

| 核心消融                  | PDMS |
|---------------------------|------|
| DINOv2 初始               | 90.0 |
| ImageNet-21k 初始         | 87.5 |
| 无 registers/压缩         | 88.2 → 90.0 |
| 1 轨迹 vs 8/64/128 轨迹 | 80.1 → 90.0 |
| disjointed scoring | 90.0 |

**风格视角**：DrivoR 对视觉编码器依赖很大，官方强调"Registers + DINO"。如果 Cosmos 风格迁移把视觉分布推开，ViT-based registers 应该是敏感点；但它的**多轨迹 + 打分器解耦**可能在扰动下有去钝性。是"图像域鲁棒性"重点考察对象。

---

## 4. ChainFlow-VLA（自回归 Chain + Flow 飞集 + VLM）

> **一句话定位**：把 "DiffusionDrive / LEAD-drive" 的无序扩散、结合 VLM 语言指导，做出"自回归轨迹生成（Chain）" + "DiT 增量精化（Flow）" + "VLM 推理引导"的三段式 VLA 规划，是 NAV 上最"爆"的一篇。

- **arXiv**: 2605.23270
- **官方仓库**: [github.com/AFARI-Research/ChainFlow-VLA](https://github.com/AFARI-Research/ChainFlow-VLA)
- **权重**: HuggingFace `AFARI-Research/ChainFlow-VLA`
- **论文图 2（整体架构）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/chainflow_framework.png" alt="ChainFlow-VLA 框架" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">ChainFlow-VLA 框架：先用 Autoregressive Trajectory Generation 产出 K 条因果 proposal，再经 VLM-Guided Residual Diffusion Refiner 修缮（arXiv 2605.23270 Figure 2）</figcaption>
</figure>

### 架构解读

三段式生成链路：

1. **Chain（自回归轨迹生成）**：不一次性铺所有 future，而按时刻步、做自回归（autoregressive）方式逐步生成 $K$ 条因果 trajectory proposal。数学上：$P(Y^{\mathrm{AR}}\mid \mathcal{O}) = \prod_t P(y_t\mid y_{<t}, \mathcal{O})$，同时用 Bicycle 运动学推进。得到 $\{Y^{(k)}_{\mathrm{AR}}\}_{k=1}^K$。
2. **Flow（DiT 修正候选的增量）**：对每条 AR proposal，用 Flow / DiT refiner 去噪 $P(\Delta Y_k \mid Y^{(k)}_{\mathrm{AR}}, h_{\mathrm{VLM}})$，而不是从噪声重构整条，修的是"残差"。步数/块数可调。
3. **VLM 指导**：VLM 输出推理 `h_VLM` 指导解码去噪（作为 guidance/conditioning）。

### 评测配置与得分（NAVSIM 主消融）

| ID | Chain | DiT | VLM | PDMS ↑ |
|----|-------|-----|-----|--------|
| 0 | ✗ | ✗ | ✗ | 93.7 |
| 1 | ✓ | ✗ | ✗ | 94.0 (+0.3) |
| 2 | ✓ | ✓ | ✗ | 94.1 (+0.4) |
| 3 | ✓ | ✓ | ✓ | **94.8 (+1.1)** |

- 在 DiffusionDrive 上兼容：88.1 → +ChainFlow **88.9**；iPad 91.7 → +ChainFlow **92.7**。
- DiT 12 block vs 8 block：94.72 vs 94.64（几乎无差异，选少即可）。

**风格视角**：ChainFlow 明确把 DrivoR 定为 baseline 之一（ID 0 = DrivoR），所以它是"最纯的视觉迭代流" ——VLM 语言指导在风格迁移时可能保持推理稳定，但前提是它依赖的视觉 backbone 在"扰动图像"下还能给出可靠的 scene tokens。这就把"VLM 的跨域稳定性"放大成关键变量。

---

## 5. AutoVLA（VLA + 自回归 action token + GRPO / RFT）

> **一句话定位**：走"VLA 世界模型"路线的代表——用预训练小 VLM 做 backbone，把轨迹离散成 driving tokens 预测，再用 GRPO（RL）直接优化 PDMS；关注"Fast/Slow Thinking"两种推理模式。

- **arXiv**: 2506.13757（修正：用户表中"2605.13757"有误）
- **官方仓库**: [github.com/ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA)
- **权重**: HuggingFace `Zewei-Zhou/AutoVLA`
- **论文图 3（模型与训练流程）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/autovla_overview.png" alt="AutoVLA 总览" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">AutoVLA：预训练小 VLM 骨干 + 多相机流/系统提示 + 世界知识融入策略；训练含 iSFT 与 GRPO RFT（arXiv 2506.13757 Figure 3）</figcaption>
</figure>

### 架构解读

- **Backbone**: 预训练小 VLM（语言世界知识合集）。
- **输出**: 轨迹被量化为 token 序列（多视角相机 + 系统提示），自回归解码出轨迹。
- **训练三个阶段**：LM loss + action loss（SFT），其中加重 CoT 文本权重；随后 **GRPO / RFT**（RL）直接以 PDMS 作为稀疏 reward；同时有 `.CoT` 文本 reward。
- **Fast / Slow Thinking**: 提供两种推理模式，Slow 更多推理间隔，可视为探索更全面的推理能力。

### 评测配置与得分

- **NAVSIM v1（camera, 3x）**: NC 98.4 / DAC 95.6 / TTC 98.0 / Comf. 99.9 / EP 81.9，**PDMS 89.1**。
- **RFT/GRPO 后（同表 BiF）**: 能达到更高水平（表 14 中 Best-of-N 提升至 92.1，97.1、87.6 EP 等）。

**风格视角**：AutoVLA 的成败几乎全压在"语言世界模型 + 推理"之上；事实证明它在空间推理、交通规则上很强，但对视觉的低级纹理依赖小——**纯视觉漂移到来时，它可能是最抗打的一档**，也是必须重点看的 VLA 对照组。

---

## 6. ReCogDrive（VLM + 扩散规划 + 三阶训练）

> **一句话定位**：典型"感知-认知-动作"三级 VLM 驱动：VLM 给出认知（trajectory/reasoning），扩散 planner 在认知 token 条件下去噪生成轨迹，三阶训练（预训练、IL、RL）逐步提稳。

- **arXiv**: 2506.08052（修正：用户表中"2605.08052"有误）
- **官方仓库**: [github.com/xiaomi-research/recogdrive](https://github.com/xiaomi-research/recogdrive)
- **权重**: HuggingFace 集合 `owl10/recogdrive`
- **论文图 1（总览）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/recogdrive_overview.png" alt="ReCogDrive 总览" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">ReCogDrive 总览：含丰富驾驶先验，VLM 与扩散 planner 协同（arXiv 2506.08052 Figure 1）</figcaption>
</figure>

- **论文图 3（训练管线 + 模型架构）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/recogdrive_pipeline.png" alt="ReCogDrive 训练管线与架构" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">ReCogDrive 训练管线：VLM 将输入编码为认知 token 引导轨迹去噪，分三阶段——驾驶预训练、模仿学习、强化学习（arXiv 2506.08052 Figure 3）</figcaption>
</figure>

### 架构解读

- VLM 输入（相机图、导航、ego 状态）→ 输出**认知 tokens（cognitive tokens）** + 文本推理。
- 扩散路径：$q(\mathbf x_t|\mathbf x_{t-1})=\mathcal{N}(\sqrt{1-\sigma_t^2}\,\mathbf{x}_{t-1},\sigma_t^2\mathbf{I})$；DiT：$\mathbf{x}_{t-1}=D_{\mathrm{act}}(\mathrm{DiT}_\theta(z_t;F_h;S_{ego};t))$，融合历史、自我、VLM feature。
- 训练三阶段：Driving Pre-training → IL → RL（扩散 RL）。

### 评测配置与得分

- **NAVSIM v1（camera, 3x）**: NC 98.2 / DAC 97.8 / TTC 95.2 / Comf. 99.8 / EP 83.5，**PDMS 89.6**。
- 另一结果 ReCogDrive（R34）EP 87.3 → PDMS **90.8**（与 DriveVLA-W0 邻近配置）。

**风格视角**：ReCogDrive 和 iPad 一样，成功依赖 VLM 的认知文本 + 扩散的噪声鲁棒；风格迁移如果把图像推到 VLM 之外、或 diffusion 的噪声分布改变，轨迹去噪可能"跑偏"。是纯视觉 + VLM 的"鲁棒敏感中等"代表。

---

## 7. DriveVLA-W0（世界模型 + MoE VLA）

> **一句话定位**：把 VLA 从"只有动作监督"升级为"原生预测未来"——AR World Model（视觉 token）+ Diffusion World Model（连续未来）两条线，再用 MoE 把大 VLA + 轻量 Action Expert 联合优化效率。

- **arXiv**: 2510.12796
- **官方仓库**: [github.com/BraveGroup/DriveVLA-W0](https://github.com/BraveGroup/DriveVLA-W0)
- **权重**: HuggingFace `liyingyan/DriveVLA-W0`
- **论文图 2（架构）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/drivevlaw0_arch.png" alt="DriveVLA-W0 架构" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">DriveVLA-W0：AR World Model 预测离散视觉 token，Diffusion World Model 预测未来连续模态，MoE 配对 Action Expert（arXiv 2510.12796 Figure 2）</figcaption>
</figure>

### 架构解读

- **AR World Model**：自回归预测未来视觉 token，让世界知识注入。
- **Mixture-of-Experts 架构**：一个重型 VLA Expert + 一个轻量 Action Expert（query-based flow matching）。推理时主要用 Action Expert 走快车道。
- **双路径世界建模**：AR（离散视觉 token）与 Diffusion（连续未来）互补。

### 评测配置与得分

- **NAVSIM v1（camera，单相机）**: 基础 **PDMS 88.4**；NuPlan 预训练 + NAVSIM 微调后 **90.2**（表 6 的 6VA/2VA 配置）。

| 序列设计 (pretrain/finetune) | NC | DAC | PDMS |
|---------------------|----|-----|------|
| VA/VA | 96.8 | 92.7 | 83.3 |
| 2VA/2VA | 97.3 | 93.2 | 84.2 |
| 6VA/2VA | 98.3 | 93.8 | 85.6 |

- 更长 action 序列与更长视野持续提升 PDMS（NAVSIM 微调后到 88-90）。

**风格视角**：DriveVLA-W0 用 world model 去"想象未来"，在漂移下理论较稳（不依赖逐像素局部感知）；值得把"world model 能否在雨/雪/夜里保持未来连贯"列为测评指标。

---

## 8. Drive-JEPA（联合世界模型预测 + JEPA 视觉预训练）

> **一句话定位**：把 V-JEPA 式自监督视频预训练搬到驾驶规划上，配合多轨迹发散蒸馏（MTD）和多模态伪标签，走 camera-only 且全图不依赖 LiDAR，取得高性能。

- **arXiv**: 2601.22032
- **官方仓库**: [github.com/linhanwang/Drive-JEPA](https://github.com/linhanwang/Drive-JEPA)
- **权重**: HuggingFace 数据集 `datasets/LinHanWang/Drive-JEPA`
- **论文图 2（总览）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/drivejepa_arch.png" alt="Drive-JEPA 总览" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">Drive-JEPA：Driving Video Pretraining 用 V-JEPA 学 ViT encoder，再以 memory / 世界未来预测接口双优化（arXiv 2601.22032 Figure 2）</figcaption>
</figure>

### 架构解读

- **视频预训练**：`min ||P_phi(Δ, E_θ(x)) - sg(E(y))||`，用自监督 V-JEPA 学到一个 ViT encoder。
- **感知 / 世界模型**：Transformer decoder 做目标/未来预测的 query；双 WADA attention 处理轨迹 + 世界 token。
- **MTD（Multimodal Trajectory Distillation）**：用多个 pseudo-teacher 轨迹使 proposal 保持多模态，避免 mode collapse。

### 评测配置与得分

- **NAVSIM v1（ViT-S, camera）**：NC 98.7 / DAC 96.2 / EP 82.9 / TTC 95.5 → **PDMS 89.0**。
- **NAVSIM v1（ViT-L, camera）**：可达 **PDMS 93.7**。

| 方法 | 后端 | 输入 | PDMS |
|------|------|------|------|
| Epona | ResNet | camera | 86.2 |
| iPad (R34) | ResNet | camera | 91.1 |
| DriveSuprim (R34) | ResNet | camera | 89.9 |
| Drive-JEPA (R34) | ResNet | camera | 91.5 |
| **Drive-JEPA (ViT-L)** | ViT-L | camera | **93.7** |

**风格视角**：因为 JEPA 用 V-JEPA 自监督（而非纯有监督/纯 VLM），它对**视觉语义的底层特征**更有通用性——理论上对风格迁移的 invariance 较好，是我们最想验证"预训练范式 vs 域漂移"假设的模型。

---

## 9. DriveSuprim（Top-K 粗到细 + 旋转增强 + 候选蒸馏）

> **一句话定位**：聚焦"**Selection（候选选择）**"范式，用 coarse-to-fine 两步选择最 top 的轨迹，并用旋转数据增强打破"直行偏置"，配合蒸馏。

- **arXiv**: 2506.06659（修正：用户表"2606.06659"有误）
- **官方仓库**: [github.com/William-Yao-2000/DriveSuprim](https://github.com/William-Yao-2000/DriveSuprim)
- **权重**: HuggingFace `alkaid-2000/DriveSuprim`
- **论文图 2（架构）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/drivesuprim_arch.png" alt="DriveSuprim 架构" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">DriveSuprim 架构：coarse-to-fine 选择区分 hard negatives；粗选择出候选，并在此基础上 RefineDec 修；用旋转数据增强减小训练集"直行偏置"（arXiv 2506.06659 Figure 2）</figcaption>
</figure>

### 架构解读

- 学习由 粗选（coarse）+ 精选（refine）两级组成。粗选大 candidate 池，用 metric score 逐子项打分，筛出 top-K；精用 refinement decoder 对候选轨迹细修。
- 关键创新：**soft label 蒸馏（teacher 高分轨迹作为软标签，结合旋转增强）** + 旋转图像增强缓解"大数据里直行比例失衡、转向丢失"。
- 损失：粗选 + 精选 + 增强 + soft（`L = L_ori + L_aug + L_soft`）。

### 评测配置与得分

- **NAVSIM v1（R34, camera）**: **PDMS 89.9**；**NAVSIM v1（EVA-ViT-L 后端）**：PDMS **93.5**。

**风格视角**：DriveSuprim 是把"选择"做到极致的模型，它的增强（旋转）对方向偏置有用，但对"风格/纹理"替换是否敏感——取决于它的 coarse selector 是否仍能挑出对冲的 top-K。如果 K 内全部被风格污染，coarse-to-fine 也无救。

---

## 10. DriveLaW（时空一致的世界模型 + Flow 生成）

> **一句话定位**：离线 video-in + flow matching 世界模型 + DiT 动作，用 "Noise Reinjection" 恢复由扩散丢失的结构/时间一致性，再生成轨迹。

- **arXiv**: 2512.23421
- **官方仓库**: [github.com/xiaomi-research/drivelaw](https://github.com/xiaomi-research/drivelaw)
- **权重**: HuggingFace `tz2026/DriveLaW`
- **论文图 1（总览架构）**：

<figure align="center">
  <img src="/images/e2e-navsim-models/drivelaw_arch.png" alt="DriveLaW 总览" style="max-width:100%;">
  <figcaption style="font-size:0.9em; color:#888;">DriveLaW 整体架构：先把历史观测（图/动作）编码进统一 latent world，再经 diffusion/flow 预测；配合 Noise Reinject 保结构时序一致（arXiv 2512.23421 Figure 1）</figcaption>
</figure>

### 架构解读

- 世界模型把历史观测统一编码到 latent world，输出未来。
- **噪声重新注入（Noise Reinjection）**：`L' = L_t + σ'_t·M⊙ε_t`，用 mask M 把丢失的时空一致信息还原，避免生成的未来帧"散架"。
- 动作用 Flow Matching / DiT action（`f_θ(a_t,t) = DiT_act([h_act;t] | h_ctx, {f_i})`），`L_FM = ||f_θ(a_t,t) - (a_0−ε)||`。

### 评测配置与得分（视频预训练缩放）

| Video P.T. | PDMS |
|-----------|------|
| 0 (scratch) | 85.9 |
| 76k | 87.0 |
| 3.8M | 87.8 |
| **7.6M** | **89.1** |

**风格视角**：DriveLaW 是世界模型家族的代表，几乎不直接吃图像原始像素——它是靠 latent world 生成未来。如果风格迁移改变图像视觉分布，世界模型的"雨雪后的未来帧真实"就成了主要脆弱点——**正好命中我们评测里最核心的"视觉域 → 世界模型"链路**。

---

## 11. LTF / Latent TransFuser（官方纯视觉 NAV 基线）

> **一句话定位**：NAVSIM 官方 image-only 基线的代表——TransFuser 的 latent 变体，用 latent positional embedding 替代 LiDAR BEV，是 NAV 基准里经过官方验证的纯视觉 baseline。

- **arXiv**: 2406.15349（NAVSIM 论文本身）
- **官方仓库**: 官方 [github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim)（`navsim_baselines`目录，`ltf`）
- **权重**: HuggingFace `autonomousvision/navsim_baselines`（`ltf`）

> 架构图：LTF 是 TransFuser 的 latent 变体，其体系架构图沿用 TransFuser 架构（见 `static/images/transfuser/fig2_architecture.png`）。其特点是把 TransFuser 的 LiDAR BEV 通道替换成 **latent positional embedding**，从而 camera-only 化。

### 架构与配置

- 它不依赖 VLA、不依赖世界模型，走经典 CNN/Transformer 融合（latent variant）。
- 论文原论文 Figure 1 展示整个 NAVSIM benchmark（GT / metric 定义），Fig 2 展示 hard-frames filtering，Fig 5 是 NAVSIM Challenge 界面。
- **评价指标取向**：NAVSIM 用它作为 **官方 image-only baseline**——就是我们评测"纯视觉"的**控制组最低参考**。

### 评测配置与得分

| 数据（真实照片相机） | NC | DAC | TTC | Comf. | EP | PDMS |
|-------------------|----|-----|-----|-------|-----|------|
| LTF (camera) | 97.4 | 92.8 | 92.4 | 100 | 79.0 | **83.8** |

（在 DrivoR 的 Table S4.T1 中同列为 LTF baseline 83.8）

**风格视角**：因为它就是官方最简单纯视觉基线，风格迁移下它最能体现"一个朴素 camera-only 模型在 Cosmos 风格下到底能差到哪、以及提升空间多大"，是我们整个评测的**下界锚**。

---

## 12. 横向对比表 & 选型建议

### 12.1 全模型横向对比（NAVSIM v1 navtest / val）

| 模型 | arXiv | 类别 | 视觉后端 | 仓库 | PDMS (相机) |
|------|-------|------|------|------|------------|
| SparseDriveV2 | 2603.29163 | 稀疏 vocab 分解 | ResNet-34 | github+HF | **92.0** |
| iPad | 2505.15111 | proposal 迭代 | ResNet-34 | github/GD | **91.7** |
| DrivoR | 2601.05083 | Transformer 三块（enc+2 dec） | ViT-S (DINOv2) | github | **90.0** (val) |
| ChainFlow-VLA | 2605.23270 | Chain + Flow + VLM | DrivoR 后端 | github+HF | **94.8** |
| AutoVLA | 2506.13757 | VLA + GRPO | VLM | github+HF | **89.1** |
| ReCogDrive | 2506.08052 | VLM + Diffusion | ResNet-34 | github+HF | **89.6 / 90.8** |
| DriveVLA-W0 | 2510.12796 | World Model + MoE VLA | VLM | github+HF | **90.2** |
| Drive-JEPA | 2601.22032 | JEPA + 多模态蒸馏 | ViT-S / ViT-L | github+HF | **89.0 / 93.7** |
| DriveSuprim | 2506.06659 | coarse-to-fine 选择 | R34 / EVA-ViT-L | github+HF | **89.9 / 93.5** |
| DriveLaW | 2512.23421 | 世界模型 + Flow | latent world | github+HF | **89.1** |
| LTF | 2406.15349 | TransFuser latent | ResNet-34 | navsim 官方 | **83.8** |

> 注：各模型论文所用评测集（navtest / navval）与 backbone 略有差异，分数仅供横向参考，跨行直接对比需注意统计口径。

### 12.2 给课题组的评测排期建议

我的观点（供组长参考，选型已定）：

1. **如果你想验证"哪类范式最抗视觉风格漂移"**：非 VLA 的强纯视觉组合（iPad / DriveSuprim / Drive-JEPA / DrivoR）值得优先看——它们对"像素→具体语义"的依赖较重，风格一旦把语义视觉打碎，这类最容易暴露差距，能放大"迁移"的观感。
2. **如果你想验证"世界模型 / VLA 的泛化是否缓解漂移"**：AutoVLA、ReCogDrive、DriveVLA-W0 / DriveLaW。它们的强项本来就是"链视觉 → 推理/未来"，如果它们对风格偏移抗性好，会是一个强信号：**VLA/世界模型天然鲁棒**。
3. **LTF 必须当基线放进去**——它就是官方 image-only baseline，整个评测需要一个"无超参下界"。

**总之一句话**：因为我们已经敲定这 11 个，以下是"评测排序"的建议（不是改选型）：

**评测建议：按 3 队列跑**
- 队列 A（纯视觉基线）：LTF、DrivoR / SparseDriveV2
- 队列 B（proposal/扩散范式）：iPad、ChainFlow、Drive-JEPA、DriveSuprim
- 队列 C（VLA/世界模型）：AutoVLA、ReCogDrive、DriveVLA-W0、DriveLaW

对每个模型，在 NAV 天气/风格迁移下跑出 **{PDMS, EPDMS, 每子项，以及关闭 vs 开启风格的两级差距}**，然后归一化到"**风格鲁棒指数 Σ_{天气≠晴} ΔPDMS**"，就能得到我们评测的核心输出。

---

## 13. 下一步（评测管线怎么看）

具体我们评测还要落地行动：

1. **搭 Agent API 封装**：给 11 个模型写统一的 `Agent` wrapper（输入→轨迹输出），统一输出到 NAVSIM 的 `EgoPlanner` 输入。
2. **Cosmos 风格迁移流水线**：先固定"同一场景"（scene-id），对 camera 做 rain/snow/daycycle，闭环到 PDMS。
3. **统计口径**：PDMS = NC × DAC × (5·EP+5·TTC+2·C)/12（当 NC、DAC 满分时 EP 是唯一连续变量），重点盯 NC、DAC 两个乘性项在迁移下的塌陷。
4. **输出**：每模型一张 "style 热力矩阵"（× 天气 × 难易度），最终汇总成"模型 × 通用鲁棒性"雷达图 —— 这就是我们新 benchmark 最有价值的产出。

---

*本文基于 2025-2026 arXiv 原文、官方仓库与 NAVSIM 官网数据整理，所有分数以论文原文为准（不同论文同一基线存在 ±0.2~1 的统计差异）。数据截止 2026 年 8 月。*