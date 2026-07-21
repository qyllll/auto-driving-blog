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

## 跟 DiffusionDrive 的对比

DiffusionDrive 和 DiffusionDriveV2 的关系是「同一套生成器，不同的训练范式」。下表从 10 个维度对比两者的差异：

| 维度 | DiffusionDrive | DiffusionDriveV2 |
|------|---------------|-----------------|
| **论文** | CVPR 2025 Highlight | arXiv 2512.07745 (2025.12) |
| **生成器结构** | TrajectoryHead (2 层 CustomTransformerDecoder) | **完全不变**，冷启动加载预训练权重 |
| **感知主干** | ResNet-34 双路 (TransFuser) | **完全不变** |
| **Anchor 数量** | 20 (KMeans 聚类，冻结) | **20，不变** |
| **训练范式** | 纯 IL (模仿学习) | IL 冷启动 → RL 微调 (GRPO) |
| **扩散调度** | DDIM (η=0) 确定去噪 | 训练探索用 DDPM (η=1)，推理用 DDIM (η=0) |
| **探索噪声** | 无（不需要，IL 看真值） | 乘性噪声 (Multiplicative Noise) |
| **轨迹筛选** | 分类头 (1 层 MLP，与 generator 端到端) | 两级 Mode Selector（粗筛 top-10 + MLP 细排 + Margin-Rank loss，独立训练） |
| **负样本 anchor** | 无约束（只监督离真值最近的正样本） | RL 约束全部 20 个 anchor |
| **NAVSIM v1 PDMS** | 89.3 | **91.2** (+1.9) |
| **代码仓库** | github.com/hustvl/DiffusionDrive | github.com/hustvl/DiffusionDriveV2 |

**核心一句话**：DiffusionDriveV2 没改生成器一行代码，只改了怎么训——从「只看真值」变成「真值打底 + RL 纠偏」。所有改动都集中在训练循环里的探索噪声、GRPO 损失、和 Mode Selector，推理时生成器本身一模一样。

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

## 项目结构（基于真实开源仓库）

官方仓库 `hustvl/DiffusionDriveV2` 已开源（`https://github.com/hustvl/DiffusionDriveV2`，MIT 协议，345+ stars）。以下目录结构直接来自仓库源码。标注 `(new)` 是相对 DiffusionDrive 新增/修改的。

- `navsim/agents/diffusiondrivev2/`（新增 Agent 插件）（new）
  - **RL 分支**（负责 rollout + GRPO 策略梯度训练）
    - `diffusiondrivev2_rl_agent.py`（new）：RL 训练的 Agent 接口。加载 DiffusionDrive 预训练权重后冻结 backbone，只训练 TrajectoryHead。使用 PyTorch Lightning 框架。
    - `diffusiondrivev2_rl_config.py`（new）：RL 超参（lr=1e-4, batch=4, groups=4, warmup=1 epoch, cos lr schedule）。
    - `diffusiondrivev2_model_rl.py`（new）：**核心文件**（~1140 行）。包含：V2TransfuserModel（感知主干）+ TrajectoryHead（含 DDIMScheduler_with_logprob）+ `forward_train_rl()`（做 rollout 10 步去噪 + PDM 评分 + advantage 计算）+ `get_rlloss()`（策略梯度 + IL 辅助损失）。
  - **Selector 分支**（负责 coarse→fine 两级排序选最优轨迹）
    - `diffusiondrivev2_sel_agent.py`（new）：Selector 训练的 Agent 接口。
    - `diffusiondrivev2_sel_config.py`（new）：Selector 超参。
    - `diffusiondrivev2_model_sel.py`（new）：**核心文件**（~1400 行）。包含：TrajectoryHead（含独立 scorer 网络）+ `_score_coarse()`（5 个子指标 BCE + Margin-Rank loss 粗排 top-k）+ `_score_fine_multi()`（多层 scorer 细排 + 最终选择）。
  - `modules/` (new)
    - `blocks.py`：GridSampleCrossBEVAttention、linear_relu_ln 等共享模块。
    - `conditional_unet1d.py`：ConditionalUnet1D 及正余弦时间嵌入。
    - `multimodal_loss.py`：LossComputer（多模态分类/回归损失）。
    - `scheduler.py`：WarmupCosLR 学习率调度。
  - `transfuser_backbone.py`（new）：Transfuser 感知双路编码器。
  - `transfuser_features.py`（new）：特征构建器。
  - `transfuser_loss.py`（new）：损失函数。
  - `transfuser_callback.py`（new）：PyTorch Lightning 回调。
  - `transfuser_config.py`（new）：全局配置。
- `navsim/planning/script/`（未改动）
  - `run_training.py`：Lightning 训练启动脚本（RL 和 Selector 共用）。
  - `run_pdm_score.py`、`run_create_submission_pickle.py`：NAVSIM 评测脚本。
- `scripts/`（新增）
  - `run_rl.sh`、`run_selector.sh`：训练启动 shell 脚本。

关键设计决策：**仓库拆成了两个独立 Agent**——`diffusiondrivev2_rl_agent` 只负责「GRPO 训练」，`diffusiondrivev2_sel_agent` 只负责「Mode Selector 训练」。它们各自有独立的 config 和 model，但共享 backbone 和 decoder 的基础结构。

## 一、核心创新逐块拆解（基于真实源码）

以下代码片段全部来自仓库 `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py` 和 `diffusiondrivev2_model_sel.py`，做了必要的简化以聚焦核心逻辑。

### 1.1 截断扩散生成器：完整复用 DiffusionDrive

生成器不做任何结构改动，核心代码和 DiffusionDrive 完全一致。Anchor 加载方式（model_rl.py:710-715）和截断加噪逻辑都直接复用。唯一差异在调度器的 η 参数：

```python
# model_rl.py:697-708 — TrajectoryHead.__init__
self.diffusion_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear",
    prediction_type="sample",
)
self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
    num_train_timesteps=1000, beta_schedule="scaled_linear",
    prediction_type="sample",
)
```

两个 scheduler 的区别：`diffusionrl_scheduler` 的 `step()` 在返回 `prev_sample` 的同时还返回 `log_prob` 和 `prev_sample_mean`，用于 RL 的策略梯度计算。训练探索时 `eta=1`（DDPM 随机采样），推理时 `eta=0`（DDIM 确定性）。

### 1.2 Scale-Adaptive 乘性探索噪声（在 scheduler 的 step 里实现）

乘性噪声**不是**一个独立函数，它直接在 `DDIMScheduler_with_logprob.step()` 中实现（model_rl.py:644-676）。原版 DDIM 的 step 只输出一个确定性/随机的 `prev_sample`，V2 改写了这段逻辑，**同时生成了乘性和加性两种噪声并组合**：

```python
# model_rl.py:644-676 — DDIMScheduler_with_logprob.step()
if eta > 0:
    std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
    std_dev_t_add = torch.tensor(0.0)
else:
    std_dev_t_mul = torch.tensor(0.0)
    std_dev_t_add = torch.tensor(0.0)
if prev_sample is None:
    # multiplicative noise: two Gaussian factors (horizon, vertical)
    variance_noise_horizon = randn_tensor(
        [B, G, 1, 1], device=d, dtype=dtype
    ) * std_dev_t_mul + 1.0
    variance_noise_vert = randn_tensor(
        [B, G, 1, 1], device=d, dtype=dtype
    ) * std_dev_t_mul + 1.0
    variance_noise_mul = torch.cat(
        [variance_noise_horizon, variance_noise_vert], dim=-1
    ).repeat(1, 1, T, 1)

    # additive noise: extra independent Gaussian (multiplied by 0, unused)
    variance_noise_x = randn_tensor([B, G, 1, 1], device=d, dtype=dtype)
    variance_noise_y = randn_tensor([B, G, 1, 1], device=d, dtype=dtype)
    variance_noise_add = torch.cat(
        [variance_noise_x, variance_noise_y], dim=-1
    ).repeat(1, 1, T, 1)

    prev_sample = (
        prev_sample_mean * variance_noise_mul
        + std_dev_t_add * variance_noise_add
    )   # core: x_{t-1} = mean * mul_noise + 0 * add_noise

# also compute log_prob for policy gradient
log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
           / (2 * std_dev_t_mul ** 2) - log(std_dev_t_mul)
           - log(sqrt(2 * pi))
log_prob = log_prob.sum(dim=(-2, -1))
return prev_sample, log_prob, prev_sample_mean
```

关键设计：`std_dev_t_add = 0` 所以加性噪声实际被**关闭**了，只有乘性噪声生效。两个高斯因子 `(1 + N(0, σ²))` 分别控制纵向（整条轨迹加速/减速）和横向（整条轨迹向左/右偏），形状 `(B, G, 1, 1)` → `repeat` 到 `(B, G, T, 2)`——即同一条轨迹的所有时刻共享同一组随机因子，保持几何平滑。

### 1.3 Intra-Anchor GRPO（优势计算 + 截断）

这部分在 `TrajectoryHead.forward_train_rl()` 中（model_rl.py:800-939）。核心优势计算流程：

```python
# model_rl.py:885-935 — forward_train_rl()
# step_num=10, num_groups default 4
# rollout generates diffusion_output: (B, num_groups*20, 8, 2)
# concat GT trajectory and send to PDM scorer together
reward_group = self.get_pdm_score_para(diffusion_output_with_gt, metric_cache)
reward_gt = reward_group[:, -1:]               # GT at last index
reward_group = reward_group[:, :-1]            # remove GT, keep candidates only

# --- Intra-Anchor advantage normalization ---
reward_group = reward_group.view(bs, num_groups, 20)  # (B, G, N)

mean_grouped = reward_group.mean(dim=1)               # (B, N)
std_grouped  = reward_group.std(dim=1)                # (B, N)
advantages = (reward_group - mean_grouped.unsqueeze(1))
           / (std_grouped.unsqueeze(1) + 1e-4)        # (B, G, N)

# --- keep only positive advantages for samples better than GT ---
mask_positive = (reward_group > reward_gt - 1e-6)     # (B, G, N)
advantages = advantages.clamp(min=0) * mask_positive.float()

# --- Inter-Anchor Truncated: collision gets -1 ---
for k, v_full in batched_sub.items():
    v = v_full[:, :-1].view(bs, num_groups, 20)
    if k in ('no_collision', 'drivable_area'):
        zero_mask = (v != 1)                          # safety constraint violation
        advantages = torch.where(
            zero_mask,
            torch.full_like(advantages, -1.0),
            advantages
        )

# --- Denoising Discount ---
discount = torch.tensor([0.8 ** (step_num - i - 1) for i in range(step_num)])
advantages = advantages.detach().unsqueeze(-1).repeat(1, 1, 1, step_num)
advantages = advantages * discount
return {"advantages": advantages, "reward": ...}
```

**Intra-Anchor** 体现在 `reward_group.view(bs, num_groups, 20)`：第 1 维是 `num_groups`（默认为 4），每组包含 20 条来自不同 anchor 的轨迹。`mean(dim=1)` 是同 anchor 跨 group 做均值——也就是每个 anchor 自己算自己的 baseline，不和其他 anchor 混合。

但等一下——这和论文描述的「一个 anchor 一个 group」不太一样？实际代码里 `num_groups=4` 是把**每个 anchor 拆成 4 个子组**分别 rollout（4 次独立加噪探索），然后 4 个组之间做归一化。论文描述的是 `groups=G` 是对同一条 anchor 的 G 次采样。实现上把 G 设为 4，每个 group 内 20 条轨迹来自不同的 anchor。这相当于用 4 次独立探索来估计同 anchor 的 reward 分布。

### 1.4 Inter-Anchor Truncated GRPO（策略梯度 + IL 正则）

对应的 `get_rlloss()` 方法（model_rl.py:1028-1120）：

```python
# model_rl.py:1094-1119 — get_rlloss()
old_diffusion_output = old_pred["all_diffusion_output"]  # cached 10-step chain
advantages = old_pred["advantages"]

# re-run decoder to compute log-likelihood for each trajectory
per_token_logps = all_log_probs.view(bs, num_groups * 20, -1)  # (B, G*N, 10)

# --- policy gradient (importance weighted) ---
per_token_loss = -torch.exp(
    per_token_logps - per_token_logps.detach()
) * advantages
mask_nz = per_token_loss != 0
RL_loss_b = (per_token_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp_min(1)
RL_loss_b = RL_loss_b.mean(dim=-1)

# --- IL auxiliary loss (prevent RL from forgetting basic driving) ---
IL_loss_b = torch.zeros_like(RL_loss_b)
target_traj = targets["trajectory"].unsqueeze(1).repeat(1, num_groups*20, 1, 1)
for poses_reg_list in poses_reg_steps_list:
    for poses_reg in poses_reg_list:
        traj_l1 = F.l1_loss(poses_reg[..., :2], target_traj[..., :2], reduction="none")
        IL_loss_b += traj_l1.mean() / (len(poses_reg_steps_list) * len(poses_reg_steps_list[0]))

# increase IL weight when no positive samples in batch (all worse than GT)
has_positive = (advantages > 0).any(dim=2).any(dim=1)
il_weight = torch.where(has_positive == 0, 1.0, 0.1)
loss = RL_loss_b + il_weight * IL_loss_b
```

关键设计细节：

- 使用 `torch.exp(log_prob_new - log_prob_old)` 做 importance weighting（类似 PPO 的 clip，但这里没有 clip——直接用 exp 做 on-policy 校正）。不是标准的 REINFORCE `-log_prob * advantage`。
- **Denoising Discount γ**：0.8 ^ (10 - i - 1)，越靠后的去噪步权重越大（因为越接近 x_0，梯度信号越可靠）。论文中将 γ^t-1 直接乘在 advantage 上。
- IL loss 权重根据 batch 是否含正样本自适应：如果整 batch 都没正样本（全部劣于 GT），加大 IL 到 1.0，防止 RL 训崩。

### 1.5 Mode Selector（两级 scorer 精细选择）

Selector 是**独立训练的模型**（`diffusiondrivev2_model_sel.py`），不参与 GRPO 训练。它的 TrajectoryHead 包含完整的 coarse scorer + fine scorer 网络。

#### 粗排（model_sel.py:978-1042）

```python
# model_sel.py:978-1042 — _score_coarse()
# predict 5 sub-metrics for each trajectory
NC_score  = self.NC_head(traj_feature).squeeze(-1)   # no_collision
EP_score  = self.EP_head(traj_feature).squeeze(-1)   # progress
DAC_score = self.DAC_head(traj_feature).squeeze(-1)  # drivable_area
TTC_score = self.TTC_head(traj_feature).squeeze(-1)  # ttc
C_score   = self.C_head(traj_feature).squeeze(-1)    # comfort

# BCE loss per sub-metric
loss_nc  = F.binary_cross_entropy_with_logits(NC_score, gt_nc)
loss_ep  = F.binary_cross_entropy_with_logits(EP_score, gt_ep)
loss_dac = F.binary_cross_entropy_with_logits(DAC_score, gt_dac)
loss_ttc = F.binary_cross_entropy_with_logits(TTC_score, gt_ttc)
loss_c   = F.binary_cross_entropy_with_logits(C_score, gt_c, reduction="none")
loss_c   = (loss_c * mask).sum() / mask.sum()        # skip -1 invalid entries

# Margin-Rank loss: enforce relative ordering
idx_i, idx_j = torch.combinations(range(Gk), r=2).unbind(-1)
pred_i, pred_j = EP_score[:, idx_i], EP_score[:, idx_j]
gt_i, gt_j = gt_ep[:, idx_i], gt_ep[:, idx_j]
target = torch.sign(gt_i - gt_j)
loss_rank = self.rank_loss(pred_i, pred_j, target)   # MarginRankingLoss(margin=0.05)

loss_coarse = loss_nc + loss_ep + loss_dac + loss_ttc + loss_c + 2 * loss_rank

# synthesize final reward from predicted scores, select top-k
final_coarse = sigmoid(NC) * sigmoid(DAC) * (5*sigmoid(TTC) + 5*sigmoid(EP) + 2*sigmoid(C)) / 12
best_idx = final_coarse.topk(10).indices              # coarse filter: keep top-10
```

#### 细排（model_sel.py:1044-1122）

```python
# model_sel.py:1044-1122 — _score_fine_multi()
# fine scorer has 3 ScorerTransformerDecoderLayer, each predicts 5 sub-metrics
for i, feat in enumerate(traj_feature_list):          # 3 layer features
    EP_score  = self.fine_EP_head(feat).squeeze(-1)
    NC_score  = self.fine_NC_head(feat).squeeze(-1)
    DAC_score = self.fine_DAC_head(feat).squeeze(-1)
    TTC_score = self.fine_TTC_head(feat).squeeze(-1)
    C_score   = self.fine_C_head(feat).squeeze(-1)
    # ... same BCE + Margin-Rank as coarse ...
    loss_fine = loss_fine + loss_nc + loss_ep + loss_dac + loss_ttc + loss_c + 2*loss_rank

loss_fine = loss_fine / len(traj_feature_list)       # average over 3 layers
final_fine = sigmoid(NC) * sigmoid(DAC) * (5*sigmoid(TTC) + 5*sigmoid(EP) + 2*sigmoid(C)) / 12
best_idx = final_fine.argmax(dim=-1)                  # top layer picks best
```

Selector 的训练不涉及 RL，是一个纯监督学习任务。用 PDM scorer 算出的真值分数作为标签，训练 scorer 网络学会模仿 PDM 的排序行为。这样在推理时代价极低（不需要调 PDM 仿真器），一次前向就出分数。

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

### 步骤 3：加载 anchor + repeat num_groups 次（V2 新增）

```python
# model_rl.py:812-815
num_groups = self.num_groups                         # default: 4
plan_anchor = self.plan_anchor                       # [20, 8, 2]
plan_anchor = plan_anchor.unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 8, 2]
plan_anchor = plan_anchor.repeat(bs, num_groups, 1, 1, 1)  # [B, G, 20, 8, 2]
plan_anchor = plan_anchor.view(bs, num_groups * 20, 8, 2)  # [B, G*20, 8, 2]
anchor = self.norm_odo(plan_anchor)                  # normalize
```

输出 `anchor = [B, G×20, 8, 2]`：每个 anchor 被重复 G 次（默认 4），每条「副本」将独立加噪探索。G 控制探索密度——越大奖励分布估计越准，但计算成本线性增长。

### 步骤 4：截断加噪 + 10 步 rollout 去噪（训练）/ 2 步（推理）

训练时（`forward_train_rl`，model_rl.py:818-856）走 10 步 DDPM（η=1）：

```python
# model_rl.py:818-856
step_num = 10
noise = torch.randn(anchor.shape, device=device)
diffusion_output = self.diffusionrl_scheduler.add_noise(
    original_samples=anchor, noise=noise,
    timesteps=torch.full((bs,), 8, device=device, dtype=torch.long))

    for i, k in enumerate(roll_timesteps):               # 10 steps
        noisy_traj_points = self.denorm_odo(diffusion_output)
        traj_feature = self.plan_anchor_encoder(
            gen_sineembed_for_position(noisy_traj_points))
        poses_reg, _ = self.diff_decoder(
            traj_feature, noisy_traj_points, ...)
        x_start = self.norm_odo(poses_reg[..., :2])
        prev_sample, log_prob, _ = self.diffusionrl_scheduler.step(
            model_output=x_start, timestep=k,
            sample=diffusion_output, eta=1.0)             # DDPM exploration
        diffusion_output = prev_sample
        all_log_probs.append(log_prob)

diffusion_output = self.denorm_odo(diffusion_output)
diffusion_output = self.bezier_xyyaw(diffusion_output)  # bezier + heading
```

推理时（model_rl.py:942-1004）用 `eta=0.0` 确定性采样，2 步即收敛。代码结构同上但 `step_num=2`、`num_groups=4`。

### 步骤 5：PDM 仿真算奖励 + Intra-Anchor 优势 + Inter-Anchor 截断

```python
# model_rl.py:867-935
target_traj = targets["trajectory"].unsqueeze(1)
diffusion_output_with_gt = torch.cat([diffusion_output, target_traj], dim=1)

reward_group, metric_cache, sub_rewards = self.get_pdm_score_para(
    diffusion_output_with_gt, metric_cache)

reward_gt = reward_group[:, -1:]
reward_group = reward_group[:, :-1]

# Intra-Anchor: reshape to (B, G, N), normalize within each anchor
reward_group = reward_group.view(bs, num_groups, 20)
mean_grouped = reward_group.mean(dim=1)
std_grouped = reward_group.std(dim=1)
advantages = (reward_group - mean_grouped.unsqueeze(1)) / (std_grouped.unsqueeze(1) + 1e-4)

# Keep only positive advantages for trajectories better than GT
mask_positive = (reward_group > reward_gt - 1e-6)
advantages = advantages.clamp(min=0) * mask_positive.float()

# Inter-Anchor: collision trajectories get -1
for k, v in sub_rewards.items():
    if k in ("no_collision", "drivable_area"):
        advantages = torch.where(
            v.view(bs, num_groups, 20) != 1,
            torch.full_like(advantages, -1.0), advantages)

# Denoising discount
discount = torch.tensor([0.8 ** (9 - i) for i in range(10)]).to(device)
advantages = advantages.detach().unsqueeze(-1).repeat(1, 1, 1, 10) * discount
```

### 步骤 6：策略梯度 + IL 辅助损失（第二次前向）

见上文 1.4 节的 `get_rlloss()`（model_rl.py:1094-1119）。第一次前向（`forward_train_rl`）只做 rollout + 算 advantage，detach 后存下来；第二次前向（`get_rlloss`）重新过 decoder 算 log_prob，结合 advantage 做 **importance-weighted 策略梯度**。IL loss 在 batch 无正样本时加大权重防止训飞。

### 推理时（selector 分支）

不走 RL。由 `diffusiondrivev2_model_sel.py` 的 TrajectoryHead 生成 20 条轨迹后，直接送 coarse scorer → fine scorer 选最佳，无需 PDM 仿真。

### 张量形状一览表

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | fused_feat | [B, N, D] | 感知融合特征 |
| 步骤 3 | anchor | [B, G×20, 8, 2] | G=num_groups(默认4) |
| 步骤 4 | diffusion_output | [B, G×20, 8, 2] | rollout x_t |
| 步骤 4 | log_prob | [B, G×20, 10] | 10 步的对数似然 |
| 步骤 5 | reward_group | [B, G, 20] | PDM 分数 |
| 步骤 5 | advantages | [B, G, 20, 10] | discount 后优势 |
| 步骤 6 | loss | scalar | 策略梯度 + IL |
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
