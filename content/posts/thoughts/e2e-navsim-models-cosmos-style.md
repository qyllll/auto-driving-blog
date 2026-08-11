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

## 14. 个人疑问：这些"VLA"到底在干什么？VLM 和 ResNet/DINOv2 差在哪？VLM 就更好吗？

写完上面 11 个模型后，我自己绕不开的一个疑问是：**这 11 个模型里既有纯 ResNet-34、又有 DINOv2、又有世界模型，名字里还混着 4 个 "VLA"——它们到底凭什么分成两类？"VLA" 的 L 到底是什么？用 VLM 就比用 ResNet 高级吗？** 这个疑问不解决，对比表只是堆数字。下面把话一次说透。

### 14.1 先把这 11 个模型的"视觉端到底是谁"钉死

我读论文时一个现实的痛点是：**不少 VLA 论文从不点名自己视觉塔的型号**（只写 "a pretrained VLM"），而官repo 常常又是重度改动过的。结合本文各模型论文原文与已核实的开源仓库，11 个模型的"视觉端"实况如下：

| 模型 | 表面类别 | 视觉端到底是什么 | 有没有 LLM / 语言空间 |
|------|---------|-----------------|----------------------|
| SparseDriveV2 | 稀疏打分 | **ResNet-34**（经典 CNN 特征提取） | ❌ 无 LLM，纯几何打分 |
| iPad | proposal 迭代 | **ResNet-34** Scene Encoder | ❌ 无 LLM |
| DrivoR | 寄存器 Transformer | **DINOv2 初始化的 ViT-S**（自监督视觉预训练） | ❌ 无 LLM |
| ChainFlow-VLA | "VLA" | DrivoR 同款视觉后端 + **VLM 语言指导** | ✅ LLM 参与语言指导 |
| AutoVLA | VLA | **预训练 VLM 骨干**（UCLA 自称 "小 VLM"）| ✅ 显式 CoT + 动作 token |
| ReCogDrive | VLA | **VLM 认知模块** 视觉塔 + 扩散规划器融合 | ✅ VLM 出认知 token |
| DriveVLA-W0 | VLA/世界模型 | **VLM 骨干** + AR/Diffusion 世界模型 | ✅ 隐式预测式 CoT |
| Drive-JEPA | JEPA | **V-JEPA 自监督视频预训练 ViT**（S/L） | ❌ 无 LLM，纯预测型表征 |
| DriveSuprim | coarse-to-fine | **ResNet-34 / EVA-ViT-L**（CLIP 蒸馏视觉塔） | ❌ 无 LLM |
| DriveLaW | 世界模型 | **latent world 编码器 + DiT**（非 VLM，纯世界模型流） | ❌ 无 LLM |
| LTF | 官方基线 | **TransFuser 系 (ResNet 融合)** | ❌ 无 LLM |

一眼可见：**4 个 "VLA" 不是"视觉塔不同"，而是"视觉塔上面叠加了一个 LLM（语言空间）"**。视觉塔本身彼此同源——ResNet、ViT、甚至 EVA/SigLIP 这类 CLIP 蒸馏塔，VLM 里也在用。

### 14.2 先补个课：ViT 是什么？

上面一直在说"ResNet、ViT、视觉塔"，但 **ViT（Vision Transformer，视觉 Transformer）** 这几个字对第一次看的人来说是最大的一道坎，这里用三句话说透：

- **它是什么**：ViT 就是"把图像当句子读"的模型。把一张图切成固定大小的小块（比如 `16×16` 像素一个 patch，224×224 的图切成 14×14=196 块），每块展平成一个向量，再像 Transformer 处理文字一样，让这些小块的向量互相做 **自注意力（self-attention）**——每个小块根据其他所有小块的信息更新自己，从而"看懂"整张图的上下文关系。**公式上它和 GPT/大语言模型完全同一族（都是 Transformer），只是输入从"文字 token"换成了"图像 patch token"。**

- **为什么它重要**：2020 年谷歌提出 ViT 时颠覆了 CV——以前图像必须靠 CNN（卷积，如 ResNet，滑窗扫描提取局部特征）；ViT 第一次证明**纯 Transformer 不用卷积也能达到同样水平，且越大越强、天然和 LLM 同构**。这正是今天几乎所有 VLM/VLA 都选"ViT 系视觉塔"（DINOv2、CLIP/SigLIP、EVA 全是 ViT）而不是纯 CNN 的根本原因——**图塔和语言塔同构，才能无缝拼进一个模型**。

- **在本文里看到它的地方**：DrivoR 的"DINOv2 初始化的 ViT-S"、Drive-JEPA 的 V-JEPA ViT、DriveSuprim 的 EVA-ViT-L，以及 4 个 VLA 内部嵌的视觉塔——全是 ViT。以及表格里出现的 `ViT-S / ViT-L` 中的 S/L 指 **Small / Large 两个尺寸档**（参数量几十 M vs 几百 M），这套命名沿自语言模型。

> 一句话记忆：**ResNet = 用"扫描仪"看图像（卷积），ViT = 把图像拆成词用"阅读理解"看（Transformer），而 VLM/VLA 全都站在第二条路上。**

### 14.3 普通视觉 backbone 和 VLM 到底差在哪（说人话版）

把两者拆开，一共就三个差异：

1. **"看"的部分其实差不多**。不管是 ResNet、DINOv2 的 ViT，还是嵌入式在 VLM 里的 InternViT / SigLIP / EVA，干的都是同一件事：**把图像 patch 编码成特征 token 序列**。区别只在**预训练方式**不同——有的是监督分类（ImageNet 预训练 ResNet），有的是自监督（DINOv2、V-JEPA 自己跟自己比），有的是图文对齐（CLIP/SigLIP 让图塔和文塔空间对齐）。但这三者都属于"**普通视觉 backbone**"，都不带推理。

2. **区别的核心：有没有"语言空间"这个中间层**。普通 backbone 的特征往下直接喂给动作头（回归/打分/扩散），**中间产物是不可读的几何特征**。VLM 则是在视觉塔和输出之间塞了一个在互联网级**图文数据上预训练的 LLM**——图像 token 和文本 token 一起进 LLM，LLM 既能"描述场景"（可读），又能"推理该不该动"（常识），最后要么吐语言决策、要么吐动作 token。

3. **"预训练知识量"完全不同**。ResNet/DINOv2 只在（相对）窄的图像/视频域上学特征；LLM 在大规模语言+图文上预训练，**自带世界知识、常识、因果与规则理解**。这就是 VLM 相对普通 backbone 唯一实在的"增量"：**多了一个会推理、懂常识的大脑袋**。

一句话：**普通 backbone = 图塔；VLM = 图塔 + LLM。视觉能力来源可以一样，"会不会想"才是分界线。**

### 14.4 VLM 就更好吗？——分数说明根本不是

把本文分数摊开，**VLM 在开环 PDMS 上不但没赢，多数还更差**：

- 无语言/无 LLM 阵营：TransDiffuser 94.85、CLOVER 94.5、DrivoR 93.7、Drive-JEPA 93.7、DriveSuprim 93.5、SparseDriveV2 92.0、iPad 91.7 —— 全线 91+，天花板 94.85。
- VLA 阵营：AutoVLA 89.1（+锚 92.1）、ReCogDrive 89.6/90.8、DriveVLA-W0 90.2 —— **最高也只有 90.8**。

理由很硬：(1) **语言 tokenizer 是离散的**，把连续轨迹切成 token 有量化误差，开环指标天然吃亏，ReCogDrive 的扩散头、EMMA 的坐标 token 苦难都是佐证；(2) **LLM 又大又慢**，推理/部署成本高；(3) 开环 PDMS 基本是"几何+规则"打分,**不太需要常识**，正是普通背书的强项。

**所以 VLM 的战场不在开环排行榜**，而在：
- **长尾/未知场景**（没见过的障碍、烂天气、逆光——需要常识兜底而不是匹配训练分布）；
- **可解释性与安全审计**（能说一句"前方车道被雪覆盖，我沿车辙走"给规控/监管看）；
- **闭环决策质量**（跟车博弈、换道时机这类"靠想不靠抄"的能力）。
这正好是这篇 Bench2Drive / Cosmos 迁移评测该盯的地方——**VLM 的"溢出"能力在分布外才体现，开环高分测不出来**。

### 14.5 为什么叫 VLA 却不一定"说人话"？—— L 是训练遗产，不是运行义务

"VLA" 这四个字母（Vision-Language-Action）指的是**架构声明**：模型里有一个 **LLM 骨干**，哪怕推理时不吐一个语言 token。所以"名字带 VLA"≠"运行中在说人话"。L 的参与可以分三档，你拿这篇文章的 4 个模型就能对号入座：

| 等级 | L 参与方式 | 例子 |
|------|-----------|------|
| **① 显式 L** | 推理时真的输出语言决策/CoT，"想给谁看就给谁看" | AutoVLA（慢思考模式）、DriveVLM 家族 |
| **② 隐式 L** | 语言不用现形，但 LLM 在 token 层做隐性推理（隐式 CoT / 认知 token） | DriveVLA-W0（预测式 CoT）、ReCogDrive（认知 token） |
| **③ 遗产式 L** | 只用 VLM 的预训练视觉塔+整体初始化，推理纯出动作 token，语言彻底不上线 | 相当一部分轻量 "VLA" 实际如此 |

关键认知：**"VLA" 是"一个滑杆"，不是"一个阵营"**。L 的作用是预训练图文数据给的全局表示强度，而不是"推理时嘴巴动了没"。你甚至可以理解成：**叫 VLA 的模型，本质是"图塔 + 一个能记常识的大模型 + 一条动作输出"**，L 参与几成是个工程选择。

### 14.6 那么 L 在自动驾驶里到底有什么用？（能说清的用途只有这几个）

1. **导航指令跟随**——"前方 300 米右转"这种自然语言指令，普通 backbone 根本没有入口，LLM 天然能解。
2. **规则/常识注入**——红灯停、让救护车、路权判断这类"文字级规则"，LLM 预训练里见过，SFT 一点就通。
3. **长尾兜底与可解释性**——模型可以"描述+推理+决策"三段式输出，出问题可归因、可对法规问责。
4. **强迁移先验**——海量图文预训练的表示，在某些分布外/域漂移场景的鲁棒性优于纯 ImageNet/自监督特征（这正是 Cosmos 迁移评测想验证的假设）。
5. **慢思考/自适应**——AutoVLA 用 GRPO 把"想不想"交给模型，让它"该想才想"，是对部署成本的工程妥协。

而"L 的代价"也不含糊：**token 自定义精度、慢、贵、开环分数被稀释**——这也是为什么 ReCogDrive 会把动作交给扩散头、DriveSuprim/SparseDriveV2 干脆不走 LLM。

### 14.7 一句话理解框架（送给自己，也送给读者）

> **遇到任何模型先问三个问题：① 视觉塔是谁（ResNet / DINOv2 / CLIP系 / 视频自监督）？② 塔上面有没有 LLM（有语言空间 / 无）？③ 如果有，L 参与几成（显式 CoT / 隐式 token / 只有预训练遗产）？**

用这套框架重新看本文 11 个模型：**SparseDriveV2、iPad、DrivoR、Drive-JEPA、DriveSuprim、LTF 是"纯图塔队"（无非"自监督还是超分辨率塔"的区别）；ChainFlow-VLA、AutoVLA、ReCogDrive、DriveVLA-W0 是"图塔+LLM 队"，分水岭只在 L 参与几成；DriveLaW 是"不带 LLM 的世界模型队"，独立一档。** 这样再看那张 91+/93+ 开环高分表，就不会再困惑"VLM 怎么没霸榜"——**它们在等分布外和闭环的考试，不在开环这一场。**

---

## 15. 再补一个坑：感知到底在感知什么？"稠密 BEV"和"稀疏语义"是什么？为什么有的模型"只用相机"？

这个坑我掉进去过很久，趁这次把它彻底填平。你的疑惑我复述一遍：**"大家好像都有'感知'，但有的稠密 BEV、有的稀疏语义，有的又搞检测/建图/运动预测/占位一堆任务，有的却只用相机就完事——感知到底是为了得到什么？它们之间是什么关系？"**

先说结论：**你把三条不相干的线捆在一起了，它们其实是三件独立的事——① 用什么传感器看（输入）、② 脑子里用什么结构表示场景（中间表示）、③ 显式吐出哪些理解结果（输出任务）。** 下面一条条拆。

### 15.1 第一条线：传感器（输入）——"只用相机"是指这个

- **camera-only（纯视觉）**：只吃摄像头图像，如本文 11 个模型全部如此。图便宜、图泛化、图能"看到颜色/文字/语义"。
- **camera + LiDAR**：再加激光雷达点云。几何精确（直接给 3D 距离），但贵、怕雨雪、缺语义（不知道"那是一块没信号的牌子"）。
- **LiDAR-only / 多模态**：工程车量产更常见，评测榜少见。

**关键澄清**："只用相机"说的是**输入通道**，不是"不需要感知"。相机照样要感知——只是感知的**难度更高**（要从 2D 像素猜 3D 距离）。别再以为"只用相机=没有感知"。

### 15.2 第二条线：中间表示——"稠密 BEV" 和 "稀疏语义" 是两种"脑子里的地图"

这是你问题里最大的坎，也是核心。所有感知模型都要回答同一个问题：**"把多路相机的 2D 像素，变成'模型好用的场景表示'"**。答案分两大流派：

#### 稠密 BEV：把世界摊成一张"格子纸"

- **全名**：Dense BEV（Bird's Eye View，鸟瞰图）。就是把自车周围（比如前后各 50m、左右各 50m）的矩形区域**划成一个个规则小格子**（如 200×200，每格 0.5m），**每个格子里存一个特征向量**，表示"这个位置有什么"。
- **怎么做**（两种主流做法）：
  - **LSS（Lift-Splat-Shoot）派**：先把每个像素"抬升"成一根根视锥线（猜测深度），再按外参把这些视锥线"摊"到地面上对应的格子里（splat），同格累加得到特征 → 一张 `200×200×C` 的**稠密特征图**。
  - **BEVFormer 派**：不逐像素抬升，而是**每个格子学一个"查询"（query）**，靠可变形注意力去多相机图像上采样自己要看的东西，再按时序自注意力融合上一帧 → 同样得到 `200×200×C` 稠密特征图。
- **特点**：**像地图一样铺满全场**——每个位置都有格子，天然适合"哪里能走/哪里被占"这种位置问答；一格一信息，密集但**算力按面积平方涨**，远处格子往往是浪费。

#### 稀疏语义 / 稀疏查询：只要"重要的点"

- **全名**：Sparse Queries / Sparse Tokens。不再铺满格子，而是**只维护 N 个"有意义的点"**（N 通常 100~900 个），每个点是一个可学习 query，代表"一个潜在的物体/车道线/区域中心"。
- **怎么做**：像 DETR 那样，用 N 个初始化 query 在图像特征上做注意力，逐步"收敛"到 N 个真正的物体上；每个 query 自己带位置、类别、状态，且能**跨帧持续追踪同一个对象**（自带 ID）。
- **代表**：Sparse4D、SparseDrive、DrivoR 的 register token。
- **特点**：**算力跟"场景里有多少东西"成正比，而不是跟"场地面积"成正比**；但问题在于——**它必须先"猜"哪里有东西**（靠 query 迭代），网格之外没 cover 到的地方可能漏检。

#### 一张表分清

| 维度 | 稠密 BEV | 稀疏语义/query |
|------|---------|---------------|
| 场景表示 | 规则网格，每个格子一个特征 | 离散 N 个可学习 query |
| 算力 | 跟场地面积平方走 | 跟场景物体数走 |
| 优势 | 全覆盖、好做"位置问答"、省脑筋 | 高效、自带实例 ID、跨帧稳 |
| 劣势 | 远格子浪费算力、网格分辨率死板 | 要"猜哪里有东西"，可能漏 |
| 代表 | LSS、BEVFormer、UniAD/VAD 的感知层 | Sparse4D、SparseDrive、DETR 系 |

> **记忆钩子**：稠密 BEV 像"**铺满全城的监控摄像头**"（哪里都看，浪费但稳）；稀疏 query 像"**只派探子盯着关键目标**"（省力但要探子找得到）。

### 15.3 第三条线：输出任务——检测/建图/运动预测/占位，它们是从哪来的？

这是你第二个坎：**"为什么有的信息从稠密 BEV 得到、有的从稀疏语义得到？"**

答案是：**任务本身对"位置型 vs 实例型"的需求不同，选择什么表示最顺手就用什么——两边常常混用。**

| 任务 | 它要回答什么 | 稠密 BEV 的玩法 | 稀疏语义的玩法 |
|------|-------------|----------------|---------------|
| **检测**（物体在哪） | 前方有车/人/桩，框在哪 | 在稠密特征图上做 anchor/heatmap 回归 | query 直接回归每个框（DETR 路线） |
| **建图**（道路结构） | 车道线、路沿、中心线形状 | 每个格子分类"是否车道线"→连线成图 | 专门的 map query 直接生成折线 |
| **运动预测**（别人会去哪） | 旁边那车未来 3 秒轨迹 | 占位流：稠密预测"每个格子未来有没有东西在动" | 对每个实例 query 出 K 条轨迹 |
| **占位/占用（occupancy）** | 每个体素/格子未来有没有障碍 | **天生就是稠密的**——逐格预测被占概率 | SparseOcc 类用稀疏 query 预测稀疏占用 |
| **规划**（我开哪） | 未来 3 秒自车轨迹 | 稠密特征做 cost / 打分 | 稀疏特征做打分 / 生成 |

**重点**：不是"BEV 只能出检测、稀疏只能出轨迹"——两边都能出任何任务，**只是哪个更顺手**。稠密擅长"哪里被占"（占位几乎必须稠密）；稀疏擅长"每个个体是谁、持续盯它"（追踪/多模态轨迹）。所以像 SparseDrive 这类模型其实是"**稀疏为主体**"，但为了算运动/占用也常悄悄再补一点稠密/网格结构。

### 15.4 那感知到底"为了得到什么"？——把它想成"给规划端一个可用的世界理解"

剥掉任务名，感知统一在做一件事：

> **把"几路摄像头的 2D 像素 + 自车运动状态"压缩成一份"模型能用来做决策的世界理解"——这份理解至少要回答：车在哪、路在哪、谁要动、哪里能走、危险在哪。**

- 有了它，规划头（不管打分/扩散/VLA）才有东西可以"想"。
- 有的模型**显式**把这理解拆成可读任务（检测框、车道线、轨迹、占位——比如 UniAD、SparseDrive），好处是**可解释、可分开调、可审计**，坏处是模块串行可能出错。
- 有的模型**隐式**不吐出任何任务，直接在"图像→轨迹"之间自己学出这层理解（比如 DrivoR、大部分扩散/VLA 模型），好处是简单、端到端、少误差累积，坏处是**你看不到它到底理解了啥**（黑盒）。

> **所以"有的感知复杂、有的简单"不是理解深度不同，而是"把理解显式吐出来 vs 藏在网络内部"的工程选择不同。** 理解一定都在，只是可见不可见。

### 15.5 回到本文 11 个模型，把这三条线标清楚

| 模型 | 传感器 | 中间表示 | 显式感知任务 |
|------|--------|---------|-------------|
| SparseDriveV2 | 相机 | 稀疏 query 为主 | 显式（感知-规划联合，路径+速度打分） |
| iPad | 相机 | 介于两者（proposal 中心 BEV） | 显式（proposal 迭代即预测） |
| DrivoR | 相机 | 稀疏（ViT 特征→register token） | **隐式**（无显式任务输出） |
| ChainFlow-VLA | 相机 | 稀疏视觉 token + VLM | 隐式 |
| AutoVLA | 相机 | 稀疏视觉 token | 隐式 |
| ReCogDrive | 相机 | 稀疏认知 token + 扩散 | 半显式（认知 token 即理解） |
| DriveVLA-W0 | 相机 | 稀疏视觉 token + 世界模型 | 隐式（世界模型预测未来） |
| Drive-JEPA | 相机 | 稀疏（V-JEPA 自监督特征） | 隐式 |
| DriveSuprim | 相机 | 稀疏（图像特征→轨迹选择） | 隐式 |
| DriveLaW | 相机 | latent world（隐空间） | 隐式 |
| LTF | 相机 | TransFuser 融合（BEV 风格） | 半显式（辅助任务） |

### 15.6 给你的最后一张图（一个统一的视角）

```
相机/点云像素
     │  ① 传感器（输入）—— 只用相机 = 从 2D 猜 3D，难度高但省成本
     ▼
场景表示（稠密 BEV 格子纸 / 稀疏 query 探子）
     │  ② 中间表示 —— 怎么把像素整理成"能理解的地图"
     ▼
显式/隐式理解：检测·建图·运动预测·占位
     │  ③ 输出任务 —— 显式=吐出来可读；隐式=藏在网络里
     ▼
规划（打分/扩散/VLA 挑出未来轨迹）
```

**三个"为什么"一句话收尾**：为什么有的稠密有的稀疏？——**因为"格子纸"适合位置问答、"探子"适合实例追踪，按任务挑。** 为什么有的显式一堆任务有的啥都不吐？——**因为显式可审计、隐式更简洁，是工程取舍。** 为什么都"只用相机"？——**那只是说输入只用相机，感知该有的理解一个不少，只是更难罢了。**

---

*本文基于 2025-2026 arXiv 原文、官方仓库与 NAVSIM 官网数据整理，所有分数以论文原文为准（不同论文同一基线存在 ±0.2~1 的统计差异）。数据截止 2026 年 8 月。*