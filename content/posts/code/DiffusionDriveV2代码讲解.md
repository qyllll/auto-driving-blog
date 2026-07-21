---
title: "代码讲解：DiffusionDriveV2 — 用 GRPO 强化学习给截断扩散的多样轨迹「上安全锁」"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["DiffusionDriveV2", "GRPO", "强化学习", "扩散规划", "端到端自动驾驶", "代码讲解", "NAVSIM"]
summary: "「DiffusionDriveV2 在 DiffusionDrive 的 anchor 截断扩散之上，补了一套 GRPO 强化学习微调：用 scale-adaptive 乘性噪声做探索、Intra-Anchor GRPO 保住多模态不坍缩、Inter-Anchor Truncated GRPO 用碰撞惩罚把低质量轨迹压下去，最后加两级 Mode Selector 精排，在 NAVSIM v1 上冲到 91.2 PDMS。本文基于论文 Algorithm 还原成逐文件逐函数伪代码，从 cold start 权重加载写到 rollout → advantage → loss 的完整 RL 训练循环。」"
---

## 写给完全没基础的同学：先补几个最关键的名词

在看 DiffusionDriveV2 的代码之前，有几个名词是必须先搞懂的。如果你已经读过本博客的 DiffusionDrive 代码讲解，前一半名词你已经熟了——这里用更短的方式过一遍，重点讲新东西。

### 旧朋友（快速过）

- **端到端自动驾驶**：给模型传感器输入，直接输出行驶轨迹，中间不拆成独立模块。
- **轨迹（Trajectory）**：未来 T 个时刻的 (x, y, θ) 序列。本文里每条轨迹是 8 个点，每点 3 个值。
- **anchor（锚轨迹）**：KMeans 从训练集里聚出来的 20 种典型驾驶姿势模板。训练时冻结（不改），作为扩散起点。
- **截断扩散（Truncated Diffusion）**：不从纯高斯噪声出发，而是从 anchor 出发、只加一点点噪声（t=8），然后去噪 2 步就修好。这是 DiffusionDrive 的核心加速 trick，DiffusionDriveV2 完全继承它，**没改结构**。
- **去噪网络（Denoiser / TrajectoryHead）**：2 层 CustomTransformerDecoder，每层做 grid_sample 交叉注意力 + agent/map 交叉注意力 + FiLM 时间步调制 + 残差 offset 回归。
- **DDIM / DDPM**：DDIM 是确定性快速采样（η=0），DDPM 是随机慢采样（η=1）。DiffusionDriveV2 训练探索时用 DDPM（η=1）获取随机性，推理时用 DDIM（η=0）确定性地出结果。
- **NAVSIM 评测指标**：PDMS 是综合驾驶得分（包含碰撞、舒适度、进度等），EP 是自车前进距离，DAC 是可行驶区域合规率。

### 新名词（重点看）

- **GRPO（Group Relative Policy Optimization）**：DeepSeek 提出的强化学习算法——不给模型训一个单独的价值网络（critic），而是在一组（group）样本内部做归一化算优势。免去了训价值网络的不稳定。公式上，组内第 i 个样本的优势 = (r_i - mean(r_group)) / std(r_group)。
- **Intra-Anchor GRPO**：把 GRPO 的「组」定义为「来自同一个 anchor 的若干条轨迹」。右转和直行的轨迹不放在同一个组里比——因为它们代表不同驾驶意图，不应该互相竞争。只在同一个意图内比「这条转得好不好」。
- **Inter-Anchor Truncated GRPO**：在 Intra-Anchor GRPO 的基础上加全局视角。对于没碰撞的轨迹，只保留正优势（鼓励更好）、截断负优势（不惩罚保守）；对于碰撞的轨迹，直接给 -1 重罚。用两行判断解决「矮子里拔将军」和「安全硬约束」。
- **乘性探索噪声（Multiplicative Exploration Noise）**：RL 需要探索，探索需要加噪声。标准做法是给每个 (x,y) 加独立高斯（加性噪声），但轨迹近端和远端尺度差异大，加出来毛刺多。乘性做法只生成两个随机因子（纵向/横向），乘到整条轨迹上，保持几何形状不变。
- **冷启动（Cold Start）**：DiffusionDriveV2 的生成器不是从零训，而是直接加载 DiffusionDrive 的预训练权重。RL 只负责微调安全偏好，不需要重新学怎么开车。
- **Mode Selector（模式选择器）**：和 DiffusionDrive 的分类头（一层 MLP）不同，V2 换成一个独立的两级 scorer：先粗筛保留 top-10，再细排选最优。训练时加 Margin-Rank loss 让模型学「相对排序」而非绝对分值。
- **Margin-Rank Loss**：一种排序损失——如果轨迹 A 的真实分数高于 B，但模型预测的 A 分数低于 B，就产生惩罚。它让 scorer 更关注「谁比谁好」而不是「绝对分数是多少」。
- **REINFORCE 梯度**：策略梯度最基础的形式。对于不可微的奖励函数，用它避开求导：梯度 = 期望(log_prob * advantage)。DiffusionDriveV2 的 RL 更新就是用这个。
- **Denoising Discount γ**：给去噪链中早期步（噪声大、梯度信号弱）一个折扣权重，避免早期步的高噪声破坏训练。值在 0~1 之间，离 t=0 越近折扣越小。

## 为什么要讲 DiffusionDriveV2 的代码

DiffusionDrive（CVPR 2025 Highlight）用 anchor 截断扩散把多样轨迹生成做得很好，但有个「屋里的大象」：**多样性有了，质量参差不齐——好轨迹和会撞的坏轨迹一起出，全靠下游分类头去挑。** 训练时 IL 只监督了离真值最近的那一个正样本 anchor，剩下 19 个负样本 anchor 完全没被约束，于是它们爱往哪走往哪走、经常出事故。分类头参数少、泛化弱，一旦在 OOD 场景里选错，就是事故。

论文原文用一个类比说得很清楚：IL 的正样本监督像「只告诉模型什么是正确做法」，但不告诉它「其他做法有多危险」。结果模型虽然能产出高质量的轨迹，但同时也产出大量未受约束的低质量——甚至是碰撞的——轨迹。这些坏轨迹全靠下游 selector 去过滤，而 selector 通常比 generator 参数少得多，泛化能力更弱。一旦遇到分布外场景，selector 选错一条碰撞轨迹，系统就出事故。

DiffusionDriveV2（hustvl × Horizon Robotics，arXiv 2512.07745）在同一个生成器上加了 **RL 微调**——不是简单加奖励，而是精确地设计了「不让 anchor 间互相比较导致坍缩」的 GRPO 改版。它用 Intra-Anchor GRPO 维持多模态（同 anchor 内比、不同 anchor 不比），用 Inter-Anchor Truncated GRPO 加上全局惩罚（碰撞的杀无赦、正优势保留）。最终在 NAVSIM v1 上拿到 91.2 PDMS 新 SOTA。

下图对比了三种方法的多模态轨迹质量（论文 Figure 1，三张子图依次为 a/b/c）：
![Figure 1：三种方法多模态轨迹质量对比——(a) Vanilla Diffusion 模式坍缩到单条轨迹，(b) DiffusionDrive 多样性好但碰撞轨迹多，(c) DiffusionDriveV2 用 RL 约束后既多样又安全](/images/diffusiondrivev2/model_comparison.png)

**图片讲解**：这是从 NAVSIM 验证集中选的一个多模态场景（交叉路口，存在左转/直行/右转多种合理走法）。每张子图都是一个俯视 BEV 视角，蓝色线是规划轨迹簇，颜色深浅代表轨迹密度。

- **子图 (a) Vanilla Diffusion**：标准扩散模型不加 anchor，直接从一个纯高斯噪声去噪生成轨迹。由于没有显式的多模态先验，模型在推理时把所有可能性「平均」成一条保守直线——左转和右转都被抹平了，多样性完全丧失。这就是论文说的 **mode collapse（模式坍缩）**。
- **子图 (b) DiffusionDrive**：加了 20 个 anchor 做截断扩散，每条轨迹从不同 anchor 出发，所以能生成左转、直行、右转等多样轨迹。但其中多条轨迹直接扎进对面车道——它们会碰撞（图中红色高亮标注）。这是 IL 监督只覆盖正样本 anchor 的后果：其余 19 个 anchor 没被约束，爱怎么走怎么走。
- **子图 (c) DiffusionDriveV2**：用同样的生成器但加了 RL 微调。碰撞轨迹被 GRPO 的优势函数打上负分、从参数上「推开」，安全轨迹的正优势鼓励模型走得更果断。结果是该左转的左转、该直行的直行，没有碰撞，且轨迹更贴合道路曲率。

> **一句话结论**：DiffusionDriveV2 = DiffusionDrive 生成器（完全不变，冷启动加载）+ 19 个负样本 anchor 终于也被 RL 管住了（Intra-Anchor GRPO 不坍缩 + Inter-Anchor 碰撞惩罚）+ 两级 Mode Selector 精排，在 NAVSIM v1 拿下 91.2 PDMS 新 SOTA。

## 架构总览

整体架构见论文 Figure 2：
![Figure 2：DiffusionDriveV2 整体架构——输入传感器数据 → Encoder → Anchored Truncated GRPO → Mode Selector 输出精修轨迹](/images/diffusiondrivev2/architecture_overview.png)

**图片讲解**：这张图是理解 DiffusionDriveV2 最关键的一张，建议配合下文「推理链路」列表一起看。架构分三大区域，从左往右读：

1. **左半（Encoder）**：和 DiffusionDrive 完全共享。多传感器数据（图像 + LiDAR BEV）先各自过 ResNet-34，再用 Cross-Attention 融合成场景特征 `fused_feat`。右下角的 N× 表示 20 个 anchor 各自独立展开一条扩散链。
2. **中间（Truncated Diffusion Decoder + Multi Noise）**：灰色框里的 `A_11`...`A_NG` 矩阵代表 N 个 anchor × G 条探索轨迹。每一条轨迹都是从 anchor 出发，先加乘性噪声（Multiplicative Noise，图中 Multi Noise 标注），再用截断扩散 decoder 走 2 步生成。彩色实线 vs 虚线表示同一条 anchor 内 G 次探索的不同结果（实线 = 高质量，虚线 = 低质量）。
3. **右半（Anchored Truncated GRPO + Mode Selector）**：这就是 V2 的核心新增。G 条探索轨迹先在 Intra-Anchor 组内归一化算优势（同 anchor 的轨迹互相比较、不同 anchor 不比），再经 Inter-Anchor 碰撞截断（负的没撞的截为 0，撞的直接 -1）。最后策略梯度更新 decoder 参数。Mode Selector 是一个独立的两级 scorer，从 N×G 条候选里精选出最终一条轨迹输出（Refined Trajectories）。

- **感知主干（Perception Backbone）**：和 DiffusionDrive 一模一样的 ResNet-34 双路编码器。图像（三相机拼接全景 1024×256）过一个 ResNet-34，LiDAR BEV 图过另一个 ResNet-34，两者做 cross-attention 融合，输出场景特征 fused_feat。这是 TransFuser 主干，没改任何结构。
- **截断扩散生成器（Truncated Diffusion Generator）**：和 DiffusionDrive 一模一样的 TrajectoryHead（20 anchor + 2 步 DDIM 去噪 + 分类头）。**完全复用 DiffusionDrive 的预训练权重**，不随机初始化。
- **RL 微调 + Mode Selector（V2 新增部分）**：在生成器之上，把训练流程从纯 IL 改成 RL（IL 做冷启动，RL 做安全微调），并新增一个两级 scorer 做最终轨迹选择。

整条推理链路（用文字版 Markdown 列表描述）：

- 图像 3 路输入 → 裁剪拼接成 1024×256 全景图
- LiDAR 点云 → 投影成 BEV 图（栅格化）
- 全景图过 ResNet-34 → 图像特征
- LiDAR BEV 过 ResNet-34 → 激光特征
- 图像特征与激光特征做 Cross-Attention 融合 → fused_feat（B, N_token, C）
- 20 个冻结 anchor → 加 t=8 截断噪声 → 2 步 DDIM 去噪（每步做 cross-attention 与场景特征交互）→ 20 条候选轨迹
- 训练时：候选轨迹送入 RL 奖励函数 → GRPO 算优势 → 更新 decoder 参数
- 推理时：候选轨迹送 Mode Selector（两级 scorer）→ 选最优 1 条输出

## 项目结构（基于论文 Algorithm 还原）

官方仓库 `hustvl/DiffusionDriveV2` 承诺开源但截至本文尚未 release。论文给出了完整的 Algorithm 描述（含 reward 定义、advantage 计算、loss 公式），以下目录结构是基于论文 + DiffusionDrive 已有代码推测的**最可能结构**。标注 `(new)` 的文件是相对 DiffusionDrive 新增的，标注 `(mod)` 是修改的。

- `navsim/agents/diffusiondrivev2/`（新增 Agent 插件）（new）
  - `diffusiondrivev2_agent.py`（new）：Agent 接口层，继承 navsim 的 AbstractAgent。和 TransfuserAgent 基本一致，区别在于调用的模型类是 DiffusionDriveV2Model 而非 V2TransfuserModel。对外暴露 compute_trajectory(agent_input)。
  - `diffusiondrivev2_model.py`（new）：主模型文件。定义 DiffusionDriveV2Model，包含感知主干（复用 V2TransfuserModel）+ 截断扩散 decoder（复用 DiffusionDrive 的 TrajectoryHead）+ ModeSelector（新增两级 scorer）。
  - `diffusiondrivev2_config.py`（new）：超参配置文件。RL lr=2e-4, batch=512, G=8, gamma=0.99, lambda_il=0.1。
  - `modules/grpo_loss.py`（new）：GRPO 损失计算核心，含 Intra-Anchor 优势归一化和 Inter-Anchor 截断逻辑。
  - `modules/exploration_noise.py`（new）：乘性探索噪声的实现（两个高斯因子纵向/横向）。
  - `modules/reward_fn.py`（new）：奖励函数封装，调用 NAVSIM 的 PDM 仿真器算碰撞/舒适度/进度。
  - `modules/mode_selector.py`（new）：两级 scorer，粗筛保留 top-10，细排 MLP 出分，含 Margin-Rank loss。
  - `train_rl.py`（new）：RL 训练入口脚本，加载 DiffusionDrive 预训练权重后做 rollout → reward → advantage → loss 循环。
- `navsim/planning/script/`（未改动）
  - `run_create_submission_pickle.py`（复用官方，不修改）
  - `run_pdm_score.py`（复用官方，不修改）
- `scripts/`（新增）
  - `run_rl.sh`（new）：RL 训练启动脚本。

逐文件一句话说明：

- `diffusiondrivev2_agent.py`：Agent 接口。在初始化时加载 DiffusionDrive 预训练权重（cold start），并新建 ModeSelector。
- `diffusiondrivev2_model.py`：总装。以 DiffusionDrive 的 V2TransfuserModel 作为 backbone + decoder，额外挂载 ModeSelector 作为最终选择模块。
- `modules/grpo_loss.py`：核心 RL 损失。输入 B×N_anchor×G 条轨迹及其奖励，先按 anchor 分组做 intra 归一化，再对碰撞样本做 inter 截断，最后返回策略梯度 loss。
- `modules/exploration_noise.py`：只用两行代码：`eps_long = randn(B,20,1,1); eps_lat = randn(B,20,1,1)`，然后 `noise = cat([eps_long, eps_lat], dim=-1)`。
- `modules/reward_fn.py`：封装 pdm_simulator 的 compute_score，把 PDMS 各子项加权成标量 reward。论文用 NAVSIM 官方的 PDM 仿真器，未自定义 reward shaping。
- `modules/mode_selector.py`：独立的两级 scorer。训练时用 BCE + Margin-Rank loss，推理时 argmax。
- `train_rl.py`：训练入口。不是 PyTorch-Lightning，而是自定义训循环。每步做：backbone 前向 → decoder rollout G 条 → reward → advantage → loss → 反向传播。

## 一、核心创新逐块拆解（伪代码形式）

### 1.1 截断扩散生成器：完整复用 DiffusionDrive

生成器不做任何结构改动，核心代码和 DiffusionDrive 完全一致。以下两个关键片段直接来自 DiffusionDrive：

Anchor 加载和冻结：

```python
plan_anchor = np.load("kmeans_navsim_traj_20.npy")
self.plan_anchor = nn.Parameter(
    torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False,
)
```

训练时的截断加噪（t 在 0~50 范围内随机）：

```python
timesteps = torch.randint(0, 50, (B,), device=device)
noise = torch.randn_like(anchor)
noisy = diffusion_scheduler.add_noise(anchor, noise, timesteps)
```

推理时的截断去噪（固定 t=8 起，2 步 DDIM）：

```python
anchor = self.plan_anchor.unsqueeze(0).repeat(B, 1, 1, 1)
img = self.norm_odo(anchor)
noise = torch.randn(img.shape, device=device)
t_start = torch.ones((B,), dtype=torch.long, device=device) * 8
img = self.diffusion_scheduler.add_noise(img, noise, t_start)
for step in range(2):
    offset = denoiser(current, fused_feat, t)
    current = current + offset
plan_reg = concat([current, theta_head(current)], dim=-1)
```

DiffusionDriveV2 和 DiffusionDrive 在这段逻辑上**一个字都不改**。唯一的训练差异在调度器的 η 参数：RL 训练探索时 η=1（DDPM，随机采样），而在冷启动的 IL 阶段和推理时 η=0（DDIM，确定性）。对应代码改动：

```python
# inference / evaluation: deterministic DDIM
self.scheduler = DDIMScheduler(num_train_timesteps=1000, ..., eta=0)
# RL training exploration: stochastic DDPM
self.scheduler_explore = DDIMScheduler(num_train_timesteps=1000, ..., eta=1)
```

### 1.2 Scale-Adaptive 乘性探索噪声

这是 RL 探索阶段的关键改动。为什么不用标准加性噪声？因为轨迹的近端和远端尺度差异大——近端坐标变化 0.1 米和远端变化 5 米，用同一个 σ 加噪声会让远端抖动远大于近端，轨迹变成「折断的线」，失去运动学合理性。

论文 Figure 3 清楚展示了加性噪声和乘性噪声的区别：
![Figure 3：加性噪声 vs 乘性噪声对比——(a) 加性噪声使轨迹锯齿状断裂，(b) 乘性噪声保持轨迹几何形状不变](/images/diffusiondrivev2/noise_comparison.png)

**图片讲解**：图中绿色实线是原始轨迹（某 anchor 生成的标准走法），蓝色和红色虚线是加噪声后的探索轨迹。

- **子图 (a) Additive Noise**：对每个 (x,y) 点独立加高斯噪声 `N(0, σ²)`。由于轨迹近端坐标变化小（如 0→0.1m）、远端变化大（如 5→15m），相同的 σ 对远端产生的抖动远超近端，结果就是蓝色/红色线呈现出锯齿状「折断线」——这种轨迹在运动学上不合理，会进入 PDM 仿真器里被碰撞/舒适度指标严厉惩罚，无法提供有价值的探索信号。
- **子图 (b) Multiplicative Noise**：不逐点加噪声，而是生成两个高斯随机因子——一个纵向 `(1 + ϵ_long)` 控制整条轨迹往前走多远，一个横向 `(1 + ϵ_lat)` 控制整条轨迹向左/右偏多少。乘到整条轨迹上，每个点同比例缩放，轨迹整体放大或缩小而不会产生锯齿。这样得到的探索轨迹保持原始的几何平滑性，PDM 仿真器才能给出有意义的奖励信号。

```python
# additive noise: each (x,y) gets independent Gaussian
eps_add = torch.randn_like(traj)
traj_noisy = traj + sigma * eps_add
# result: jagged trajectory, distal points jitter much more than proximal
```

乘性噪声的正确做法：只生成**两个**随机因子，一个纵向、一个横向，乘到整条轨迹上。

```python
def add_multiplicative_noise(traj):
    # traj: (B, 20, T, 2) normalized trajectory coordinates
    eps_long = torch.randn(B, 20, 1, 1, device=traj.device)  # longitudinal
    eps_lat = torch.randn(B, 20, 1, 1, device=traj.device)   # lateral
    noise = torch.cat([eps_long, eps_lat], dim=-1)            # (B,20,1,2)
    traj_noisy = traj * (1 + sigma * noise)
    return traj_noisy
```

注意 `traj_noisy` 的形状没变——仍然是 (B, 20, T, 2)。关键不同是：B×20 条轨迹每条只分享两个随机数，而不是 B×20×T×2 个独立随机数。所以整条轨迹的几何形状被整体缩放（如「整条线同时加速」或「整体向右偏」），不会出现「前段直、后段乱」。

作者在消融实验里验证了乘性比加性 PDMS 高 0.4（89.7 → 90.1），且轨迹平滑度显著更好。这个结果在论文 Table 4 中列出。

### 1.3 Intra-Anchor GRPO：保住多模态，不让 anchor 间互相内卷

这是 DiffusionDriveV2 最精巧的设计。要理解它为什么必要，先看「如果直接套标准 GRPO 会怎样」。

**标准 GRPO 的问题**：GRPO 把所有采样轨迹放在一个组里归一化。假设我们从 20 个 anchor 各采样 8 条，共 160 条轨迹放在一起算均值/标准差。问题在于：anchor-5 代表右转，anchor-10 代表直行。直行轨迹的 EP（前进距离）天然比右转高，于是所有右转轨迹都拿到负优势——模型就会放弃右转，导致 mode collapse。

**Intra-Anchor GRPO 的解决**：不给不同意图的 anchor 互相比较。每个 anchor 自己成为一个 group，只在自己组内算优势。这样右转轨迹只和右转轨迹比，直行轨迹只和直行轨迹比，互不干扰。

```python
def compute_intra_anchor_grpo_loss(model, batch, G=8):
    N_anchor = 20
    total_loss = 0

    for k in range(N_anchor):
        # for anchor k, generate G candidate trajectories
        # with multiplicative exploration noise
        trajs_k = []
        for i in range(G):
            # add multiplicative noise to anchor k
            noise = add_multiplicative_noise(anchor[k])
            # truncate at t=8 and denoise 2 steps
            noisy = scheduler.add_noise(anchor[k], noise, t=8)
            traj = denoiser(noisy, fused_feat, t_steps=[8, 4, 0])
            trajs_k.append(traj)

        # compute rewards via NAVSIM PDM simulator
        # reward_fn is non-differentiable
        rewards = [compute_reward(t.detach()) for t in trajs_k]

        # group-relative advantage: normalize within anchor k only
        r_mean = mean(rewards)
        r_std = std(rewards) + 1e-8
        advantages = [(r - r_mean) / r_std for r in rewards]

        # policy gradient update with REINFORCE
        for i in range(G):
            # log_prob of trajectory i under current policy
            log_prob = compute_log_prob(denoiser, trajs_k[i], fused_feat)
            # REINFORCE: gradient = -log_prob * advantage
            total_loss += - advantages[i] * log_prob * gamma_weight(t)

    return total_loss / N_anchor
```

代码里的几个关键实现细节：

- `compute_log_prob` 算的是在当前策略参数下，生成这条轨迹的对数似然。DDPM 的每一步去噪 (τ_{t-1} | τ_t) 是高斯分布，pdf 有闭解——直接用 Gaussian 的 log_prob 即可。论文 Eq.5。
- `gamma_weight(t)`：去噪折扣系数，给早期步更小权重，因为早期步噪声大、梯度信号不可靠。论文记为 γ^{t-1}。
- `compute_reward`：调用 NAVSIM 的 PDM 仿真器计算奖励，包含碰撞惩罚、舒适度、进度等。该函数**不可微**，所以 RL 用 REINFORCE 绕过。论文没有自定义 reward shaping，直接用 PDMS 指标。

### 1.4 Inter-Anchor Truncated GRPO：加一道全局安全底线

Intra-Anchor GRPO 有一个缺陷：每组内的「最好的」可能是「矮子里的将军」。比如 anchor-7 组全是好轨迹（都不撞），优势在 0 附近；anchor-13 组全是烂轨迹（都撞），但相对最好的那个优势也可能是正的——模型会错误地鼓励它。

Inter-Anchor Truncated GRPO 的修正逻辑极其**简单但有效**：

```python
def truncate_advantage(advantages, collisions):
    truncated = []
    for adv, coll in zip(advantages, collisions):
        if coll:
            truncated.append(-1.0)       # collision: heavy penalty
        else:
            truncated.append(max(0, adv))  # non-collision: keep only positive
    return truncated
```

翻译成人话两条规则：

- **没碰撞**：正优势保留（这条比别人好，值得学），负优势截断（这条太保守但安全，不惩罚）。
- **撞了**：直接给 -1 重罚，不管组内相对优势是多少。这是安全硬约束。

这比你想像的更巧妙。如果只做「碰撞的直接惩罚」，那 model 可能学出「只要不撞就好，越保守越好」的锁死状态。但这里保留了正优势的鼓励信号——对于没碰撞的轨迹，如果它比别人走得更远更高效，它的正优势会引导模型往「既安全又高效」的方向学。去掉负优势避免了保守轨迹被错误地压下去。

论文把这个修改记为 Eq.10：

$$ A_{trunc}^{k,i} = \begin{cases} -1 & \text{if collision}, \\ \max(0, A^{k,i}) & \text{otherwise}. \end{cases} $$

### 1.5 完整 RL 损失函数

论文 Eq.7 把上面两个设计统一成最终损失：

$$ L_{RL} = -\frac{1}{N_{anchor} G T_{trunc}} \sum_{k=1}^{N_{anchor}} \sum_{i=1}^{G} \sum_{t=1}^{T_{trunc}} \gamma^{t-1} \log \pi_\theta(\tau_{t-1}^{k,i} | \tau_t^{k,i}) A_{trunc}^{k,i} $$

此外，为了防止 RL 训飞，加一个 IL loss 做正则（Eq.9）：

$$ L = L_{RL} + \lambda L_{IL} $$

其中 λ ∈ (0,1)，消融实验大概取 0.1。IL loss 就是 DiffusionDrive 原来的 multi-modal focal+L1 loss（对正样本 anchor 监督）。它的作用是保持模型「还会开车」，不让 RL 过于激进地探索导致连基本驾驶能力都忘了。

### 1.6 Mode Selector：两级 scorer 精排

DiffusionDrive 的分类头 `plan_cls` 只有一层 MLP，参数极少，训练时只被「谁是正样本 anchor」这个二分类信号监督，不直接优化轨迹质量的排序。DiffusionDriveV2 换成一个更强大的独立选择器。

```python
class ModeSelector(nn.Module):
    def __init__(self):
        # coarse scorer: stage 1
        self.coarse_scorer = nn.Sequential(
            DeformableCrossAttention(d_model=256),
            nn.Linear(256, 1),
        )
        # fine scorer: stage 2 (same structure, separate weights)
        self.fine_scorer = nn.Sequential(
            DeformableCrossAttention(d_model=256),
            nn.Linear(256, 1),
        )

    def forward(self, trajs, fused_feat, agent_map_queries):
        # Stage 1: coarse, select top-10
        scores_c = self.coarse_scorer(trajs, fused_feat).squeeze(-1)
        topk_idx = scores_c.topk(10).indices
        trajs_topk = trajs.gather(1, topk_idx)

        # Stage 2: fine, pick best among top-10
        scores_f = self.fine_scorer(trajs_topk, fused_feat).squeeze(-1)
        best_idx = scores_f.argmax()
        return trajs_topk[best_idx]
```

训练 scorer 的损失函数（启发自 DriveSuprim）：

```python
def selector_loss(pred_scores, gt_scores, trajs):
    # 1. BCE loss: binary good/bad classification
    gt_binary = (gt_scores > threshold).float()
    bce_loss = F.binary_cross_entropy_with_logits(pred_scores, gt_binary)

    # 2. Margin-Rank loss: enforce relative ordering
    rank_loss = 0
    for i, j in all_pairs:
        sign = torch.sign(gt_scores[i] - gt_scores[j])
        margin = pred_scores[i] - pred_scores[j]
        rank_loss += max(0, -sign * margin + m)

    return bce_loss + lambda_rank * rank_loss
```

Margin-Rank loss 的精髓：它不要求模型精确回归绝对分数（那很难），只要求相对排序正确——如果轨迹 A 的真实 PDMS 高于轨迹 B，那模型的预测分数也应该 A > B。这种「相对约束」通常更容易学、泛化更好。

## 二、前向传播：逐步骤跟踪张量形状

下面把一整次 inference 的前向传播拆成 9 步，用纯英文伪代码 + 块外中文说明的方式呈现。每一步标注张量形状。与 DiffusionDrive 相比，变化集中在步骤 6~9（RL 差异在训练时而非推理时）。

### 步骤 1：构建输入特征

```python
def build_features(agent_input):
    cameras = agent_input.cameras[-1]
    l0 = cameras.cam_l0.image[28:-28, 416:-416]
    f0 = cameras.cam_f0.image[28:-28]
    r0 = cameras.cam_r0.image[28:-28, 416:-416]
    stitched = np.concatenate([l0, f0, r0], axis=1)
    img = cv2.resize(stitched, (1024, 256))
    img = transforms.ToTensor()(img)
    lidar_bev = project_lidar_to_bev(agent_input.lidar)
    ego_token, map_token = encode_ego_map(agent_input)
    return {"img": img, "lidar_bev": lidar_bev,
            "ego_token": ego_token, "map_token": map_token}
```

输出形状：
- `img`：[3, 256, 1024] 三相机拼接的全景图
- `lidar_bev`：[C, H, W] LiDAR 投影的 BEV 特征图
- `ego_token`：[1, D] 自车状态编码向量
- `map_token`：[N_map, D] 地图 token 序列

### 步骤 2：TransFuser 主干（感知融合）

```python
img_feat = ResNet34(img)                         # [B, C1, H1, W1]
bev_feat = ResNet34(lidar_bev)                   # [B, C2, H2, W2]
fused_feat = cross_attention(img_feat, bev_feat) # [B, N, D]
fused_feat = fused_feat + ego_token + map_token  # [B, N, D]
```

输出 `fused_feat = [B, N_token, D]`：融合了图像、激光雷达、自车状态、地图信息的场景表征。这是后面所有轨迹 query 的「条件信息源」。

### 步骤 3：加载 20 个 anchor query

```python
plan_anchor = self.plan_anchor                   # [20, T, 2]
anchor = plan_anchor.unsqueeze(0).repeat(B,1,1,1) # [B, 20, T, 2]
anchor = self.norm_odo(anchor)                   # normalize to ~[-1, 1]
```

输出 `anchor = [B, 20, 8, 2]`：20 个冻结的驾驶模式模板。T=8 是未来 8 个时刻，2 是 (x, y) 坐标。朝向 θ 在后续由 theta_head 单独回归。

### 步骤 4：截断加噪（推理时固定 t=8）

```python
noise = torch.randn_like(anchor)                 # [B, 20, 8, 2]
t_start = torch.full((B,), 8, dtype=torch.long)  # fixed t=8
noisy = diffusion_scheduler.add_noise(
    original_samples=anchor, noise=noise, timesteps=t_start)
noisy = torch.clamp(noisy, min=-1, max=1)       # [B, 20, 8, 2]
```

输出 `noisy = [B, 20, 8, 2]`：从 anchor 加轻微截断噪声后的初始样本。和 pure Gaussian 不同——它看起来已经像合理的轨迹（只是「略微歪」）。这就是为什么只需要 2 步去噪。

### 步骤 5：2 步 DDIM 去噪循环

```python
current = noisy
for step_idx, t in enumerate(timesteps):  # [8, 4] or [8, 0]
    offset = self.denoiser(
        traj=current,              # [B, 20, 8, 2]
        scene=fused_feat,          # [B, N, D]
        timestep=t,                # scalar
    )
    current = current + offset     # residual regression, [B, 20, 8, 2]
```

去噪网络输出的是 **offset（偏移量）** 而非直接坐标。这称为残差回归——网络只预测「相对当前 noisy 轨迹需要修正多少」，通常修正量很小（±0.5 以内），比直接回归绝对坐标更容易学。

### 步骤 6：补全朝向角 + 分类分数

```python
theta = self.theta_head(current)                # [B, 20, 8, 1]
plan_reg = torch.cat([current, theta], dim=-1)  # [B, 20, 8, 3]
plan_cls = self.cls_head(current, fused_feat)   # [B, 20]
```

输出：
- `plan_reg`：[B, 20, 8, 3] → 20 条候选轨迹，每条 8 个点 (x,y,θ)
- `plan_cls`：[B, 20] → 20 个候选的置信度分数

### 步骤 7：Mode Selector 选最优（推理时替代 argmax）

```python
# DiffusionDrive original: cls argmax
# best_idx = plan_cls.argmax(dim=-1)
# best_traj = plan_reg.gather(1, best_idx)

# DiffusionDriveV2: use ModeSelector
best_traj = self.mode_selector(plan_reg, fused_feat, agent_map_queries)
```

输出 `best_traj = [B, 8, 3]`：选中的最优轨迹，交给 NAVSIM 评测。

### 步骤 8：训练时的 RL 分支（仅在训练时）

如果是训练模式（且是 RL 阶段），步骤 5~7 替换为 RL 流程：

```python
# Step 5-RL: use multiplicative noise + DDPM eta=1 for stochastic sampling
noise_mul = add_multiplicative_noise(anchor)
noisy = scheduler_explore.add_noise(anchor, noise_mul, t_start)
for step_idx in range(2):
    offset = denoiser(current, fused_feat, timestep)
    current = current + offset + gaussian_noise * eta  # eta=1

# Step 6-RL: compute reward for rollouts
trajs_all = current                                  # [B, 20, 8, 2]
rewards = [pdm_simulator(t) for t in trajs_all]      # [B*20]

# Step 7-RL: Intra-Anchor advantage normalization
# reshape to [B, 20, G] and normalize per anchor
adv_intra = intra_anchor_normalize(rewards)           # [B, 20, G]

# Step 8-RL: Inter-Anchor collision truncation
collisions = [pdm_simulator.is_collision(t) for t in trajs_all]
adv_final = truncate_advantage(adv_intra, collisions) # [B, 20, G]

# Step 9-RL: policy gradient loss
log_probs = compute_log_probs(model, fused_feat, trajs_all, t_steps)
loss_rl = -mean(log_probs * adv_final * gamma_weights)
loss_il = imitation_loss(model, batch)                # keep driving skill
loss = loss_rl + lambda_il * loss_il
```

### 张量形状一览表

| 步骤 | 变量 | 形状 | 说明 |
|------|------|------|------|
| 1 | img | [B, 3, 256, 1024] | 拼接全景图 |
| 1 | lidar_bev | [B, C, H, W] | LiDAR 俯视图 |
| 2 | fused_feat | [B, N, D] | 融合场景特征 |
| 3 | anchor | [B, 20, 8, 2] | 冻结 anchor 模板 |
| 4 | noisy | [B, 20, 8, 2] | 加截断噪声后 |
| 5 | offset | [B, 20, 8, 2] | 残差修正量 |
| 6 | plan_reg | [B, 20, 8, 3] | 20 条候选轨迹 |
| 6 | plan_cls | [B, 20] | 候选置信度 |
| 7 | best_traj | [B, 8, 3] | 选中轨迹 |
| 7-RL | trajs_all | [B, 20, G, 8, 2] | RL rollout 多条 |
| 7-RL | rewards | [B, 20, G] | 每条的奖励 |
| 8-RL | adv_final | [B, 20, G] | 截断后优势 |

## 三、训练流程详解

DiffusionDriveV2 的训练分两大阶段。和 DiffusionDrive 的「纯 IL」相比，这是最大的不同。

### Stage 0：IL 冷启动（加载 DiffusionDrive 权重）

```python
pretrained_ckpt = torch.load("diffusiondrive_final.pth", map_location="cpu")
# only load backbone + decoder weights; skip missing keys (e.g. mode_selector)
model.load_state_dict(pretrained_ckpt, strict=False)
# mode_selector is randomly initialized (not in pretrained)
```

这一步不执行训练，只是加载权重。感知主干 + 截断扩散 decoder 全部复用 DiffusionDrive 的 final checkpoint。

### Stage 1：RL 微调（10 epochs）

```python
optimizer = AdamW(model.parameters(), lr=2e-4)
gamma = 0.99          # denoising discount
G = 8                 # group size per anchor
lambda_il = 0.1       # IL regularization weight

for epoch in range(10):
    for batch in dataloader:
        fused_feat = backbone(batch["img"], batch["lidar_bev"])

        # roll out G candidates per anchor with multiplicative noise
        trajs_all, log_probs_all = rollout_with_exploration(
            model.decoder, model.anchor, fused_feat, G=8)

        # compute reward via NAVSIM simulator
        rewards = [pdm_simulator.compute_reward(t) for t in trajs_all]

        # Intra-Anchor: normalize within each anchor group
        adv_intra = intra_anchor_normalize(rewards, anchor_ids)

        # Inter-Anchor: truncate with collision penalty
        collisions = [pdm_simulator.is_collision(t) for t in trajs_all]
        adv_final = truncate_advantage(adv_intra, collisions)

        # RL loss (Eq.7 in paper)
        loss_rl = 0
        for k in range(N_anchor):
            for i in range(G):
                for t in range(T_trunc):
                    log_pi = log_probs_all[k][i][t]
                    loss_rl -= gamma**(t-1) * log_pi * adv_final[k][i]
        loss_rl /= (N_anchor * G * T_trunc)

        # IL loss as regularization (Eq.9)
        loss_il = imitation_loss(model, batch)
        loss = loss_rl + lambda_il * loss_il

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

这段循环的运行时复杂度主要由 `rollout_with_exploration` 决定：每步要在 B×N_anchor×G 个轨迹上各跑一次 2 步去噪。假设 B=64, N_anchor=20, G=8，那就是 64×20×8 = 10240 条轨迹 / batch。每条轨迹 2 步去噪 → ~2 万次 denoiser 前向 / batch。论文用 8 张 L20 GPU 分布式训练来控制时间。

### Stage 2：Mode Selector 训练（20 epochs）

Mode Selector 单独训练（生成器冻结），用 RL 微调后的模型来收集轨迹数据：

```python
model.eval()  # freeze generator
selector_optim = AdamW(mode_selector.parameters(), lr=2e-4)

for epoch in range(20):
    for batch in dataloader:
        with torch.no_grad():
            fused_feat = backbone(batch["img"], batch["lidar_bev"])
            trajs_all = decoder.forward_test(model.anchor, fused_feat)
            gt_scores = [pdm_simulator(t) for t in trajs_all]

        pred_scores = mode_selector(trajs_all, fused_feat, ...)
        loss = selector_loss(pred_scores, gt_scores, trajs_all)
        loss.backward()
        selector_optim.step()
```

### 训练对比表：DiffusionDrive vs DiffusionDriveV2

| 维度 | DiffusionDrive | DiffusionDriveV2 |
|------|------|------|
| 预训练 | 无（从头训） | 加载 DiffusionDrive 权重冷启动 |
| 训练范式 | 纯 IL（BC，监督 GT 轨迹） | IL 冷启动 + RL 微调 |
| 监督信号 | 只监督正样本 anchor | 所有 anchor 都被 RL 奖励约束 |
| 噪声类型 | 标准加性高斯（训练 t~0~50） | 乘性（纵向/横向因子）+ DDPM η=1 探索 |
| 去噪调度 η | 0（训练推理都用 DDIM） | 1（探索用 DDPM）、0（推理用 DDIM） |
| 负样本 anchor | 无监督（随便生成） | RL 奖励约束 + 碰撞惩罚 |
| 选择器 | 简单 cls head（1 层 MLP） | 两级 scorer（cross-attn + Rank loss） |
| 训练 epochs | ~100 epochs | 10（RL）+ 20（scorer） |
| 主要开销 | backbone + 2 步去噪 | backbone + 2 步去噪 × G 倍 rollout |

## 四、推理流程

推理时关闭 RL 探索、DDIM η=0 回到确定性、使用 Mode Selector：

```python
def compute_trajectory(agent_input):
    features = build_features(agent_input)
    fused_feat = backbone(features["img"], features["lidar"])

    # same truncated diffusion as DiffusionDrive (2 steps, DDIM eta=0)
    trajs, cls_scores = decoder.forward_test(model.anchor, fused_feat)

    # use ModeSelector instead of original cls argmax
    best_traj = mode_selector(trajs, fused_feat, features["agent_map"])

    return best_traj   # (8, 3) trajectory
```

推理开销：主干（一次 ResNet-34 前向）+ 2 步 DDIM 去噪 + 两级 scorer（两次 cross-attention）。比 DiffusionDrive**多一个 scorer 前向**（两级 cross-attn），但 4090 上仍在实时范围内。论文 DiffussionDrive 报 45 FPS，V2 估计 ~35~40 FPS（多了 scorer 但 anchor 数量不变）。

## 五、实验结果精华

论文在 NAVSIM v1（navtest 闭评测）上的结果：

| 方法 | Img. Backbone | PDMS ↑ | EP ↑ | DAC ↑ |
|------|------|------|------|------|
| DiffusionDrive | ResNet-34 | 88.1 | 82.2 | 96.2 |
| DriveSuprim | ResNet-34 | 89.9 | 86.7 | 97.3 |
| Hydra-MDP | V2-99 | 90.3 | 86.5 | 97.8 |
| GoalFlow | V2-99 | 90.3 | 85.0 | 98.3 |
| **DiffusionDriveV2 (ours)** | **ResNet-34** | **91.2** | **87.5** | **97.9** |

三个值得注意的点：

- **ResNet-34 小骨干打平甚至超过 V2-99 大骨干**：V2 只用 21.8M 参数，就超过了 Hydra-MDP 和 GoalFlow（96.9M 参数）。这说明 RL 微调带来的收益比加参数量更大。
- **EP 涨了 5.3**：从 82.2 到 87.5。说明 RL 不仅让轨迹更安全（不撞），还让路线更高效（走得更远）。
- **PDMS@Top-10（原始输出质量）涨了 9.1**：从 75.3 到 84.4。这直接证明 RL 把负样本 anchor 的轨迹质量整体抬高了。不再是「好轨迹 + 一大把会撞的」。Mode Selector 选错的概率也大幅下降。

论文还做了详尽的消融：

| 消融项 | PDMS | 说明 |
|------|------|------|
| Baseline (DDV1 IL only) | 88.6 | DiffusionDrive 的 IL baseline |
| + 乘性噪声 | 90.1 | 比加性噪声高 0.4 |
| + Intra-Anchor GRPO | 90.5 | 保住多模态 |
| + Inter-Anchor Truncated | 91.0 | 加上全局碰撞惩罚 |
| + Mode Selector | 91.2 | 两级 scorer 再补 0.2 |

每项都有独立贡献，没有凑数的。

## 六、个人思考

**1. Intra-Anchor GRPO 的分层思想值得迁移。** 把策略空间先按「意图」分层再做 RL，本质是 hierarchical RL 的一种轻量实现。这种思路可以迁移到其他有多模态策略的问题上：比如先决定「加速/减速/转向」，再决定执行细节。关键在于「不让不同意图的样本放在同一个组里算优势」。

**2. Inter-Anchor 的截断信号是「安全 RL」的教科书案例。** 两条规则（碰撞= -1，没撞= max(0, adv)）用极简的设计同时解决了「安全硬约束」和「保守锁死」两个矛盾。相比之下，很多安全 RL 工作用复杂的约束优化（Lagrangian、Lyapunov），效果不一定比这个干净的两行判断好。

**3. 乘性噪声的几何直觉：不破坏轨迹的结构。** 加性噪声破坏局部连续性（让轨迹毛刺），模型需要额外的能力去「修复」这些毛刺——这是不必要的。保持轨迹的几何结构参与探索，模型探索到的样本更可能是有物理意义的。这和课程学习的思路类似：先让模型在合理的轨迹附近探索，再逐渐扩大范围。

**4. 冷启动是 RL 在驾驶上落地的关键。** 如果从零训 RL，20 个 anchor × 高维动作空间的探索成本极高，而且很容易训飞。先 IL 训一个「会开车」的 baseline，再用 RL 做安全偏好对齐——这个两阶段范式是端到端驾驶 RL 最可行的路径。

**5. 和本系列其他文章的呼应。** AutoVLA 和 DriveVLA-W0 也用 GRPO，但它们在离散动作 token 空间上做，DiffusionDriveV2 在连续扩散轨迹上做。这使得它的探索更自然（加噪声就是探索），但也需要 Intra-Anchor 的处理来避免 mode collapse。如果你在对比 GRPO 在不同架构上的应用，DiffusionDriveV2 和 AutoVLA 是很好的对照——一个连续、一个离散。

## 七、自己跑起来（等代码开源后）

截至本文写作时 `github.com/hustvl/DiffusionDriveV2` 尚未 release，以下是最小可行路径的预期操作：

1. 环境准备（和 DiffusionDrive 一样，基于 navsim 开发套件）：

```bash
git clone https://github.com/autonomousvision/navsim.git
cd navsim && pip install -e .
```

2. 下载 DiffusionDrive 预训练权重：

```bash
# download from huggingface: https://huggingface.co/hustvl/DiffusionDrive
mkdir -p ckpts && wget <weight-link> -O ckpts/diffusiondrive.pth
```

3. 克隆 DiffusionDriveV2 代码并 RL 训练：

```bash
git clone https://github.com/hustvl/DiffusionDriveV2.git
cd DiffusionDriveV2
# symlink DiffusionDrive pretrained weights
ln -s ../ckpts/diffusiondrive.pth ckpts/diffusiondrive_pretrained.pth
```

```bash
bash scripts/run_rl.sh
```

该脚本内部大致等价于：

```bash
python train_rl.py \
    agent=diffusiondrivev2_agent \
    experiment_name=ddv2_rl \
    rl.num_epochs=10 rl.batch_size=512 rl.G=8 rl.lr=2e-4 \
    rl.cold_start_ckpt=ckpts/diffusiondrive_pretrained.pth
```

4. 推理出 submission：

```bash
python navsim/planning/script/run_create_submission_pickle.py \
    agent=diffusiondrivev2_agent experiment_name=ddv2_submission
```

5. 本地算分 / 上榜：用官方 `run_pdm_score.py` 看 PDMS。

## 和本系列其他文章的关系

- **DiffusionDrive**（前一篇文章）：用 IL 训的截断扩散基线，V2 的生成器完全照搬它。理解 DiffusionDrive 是读 V2 的前提。
- **SparseDriveV2**：检索式规划的对比面——不走扩散、靠 26 万词表查表。两者对照能看清「生成 vs 检索 + RL vs IL」的交错。
- **AutoVLA / DriveVLA-W0**：也用 GRPO 但走离散动作 token。DiffusionDriveV2 证明 GRPO 在连续扩散轨迹上同样有效，但要处理 anchor 间的 mode collapse 问题。
