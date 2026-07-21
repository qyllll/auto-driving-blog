---
title: "代码讲解：DiffusionDriveV2 — 用 GRPO 强化学习给截断扩散的多样轨迹「上安全锁」"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["DiffusionDriveV2", "GRPO", "强化学习", "扩散规划", "端到端自动驾驶", "代码讲解", "NAVSIM"]
summary: "「DiffusionDriveV2 在 DiffusionDrive 的 anchor 截断扩散之上，补了一套 GRPO 强化学习微调：用 scale-adaptive 乘性噪声做探索、Intra-Anchor GRPO 保住多模态、Inter-Anchor Truncated GRPO 用碰撞惩罚把低质量轨迹压下去，最后加两级 Mode Selector 精排，在 NAVSIM v1 上冲到 91.2 PDMS。本文基于 paper 详尽的算法描述改写成伪代码，逐行讲清 RL 训练循环怎么写、优势怎么算、分类头怎么训。」"
---

## 写给完全没基础的同学：先补几个最关键的名词

在看 DiffusionDriveV2 的代码之前，有几个名词是必须先搞懂的。如果你已经读过本博客的 [DiffusionDrive 代码讲解](/posts/code/diffusiondrive代码讲解/)，那前一半名词你已经熟了——这里用更短的方式过一遍，重点讲新东西。

- **截断扩散（Truncated Diffusion）**：不是从纯高斯噪声出发，而是从 k-means 聚好的 anchor（模板轨迹）出发、只加一点点噪声（t=8），然后去噪 2 步就修好。DiffusionDriveV2 的生成器完全继承它，没改结构。
- **anchor（锚轨迹）**：训练集里聚类出来的 20 种典型驾驶姿势（直行 / 左转 / 超车…），作为扩散的起点。每个 anchor 代表一种「驾驶意图」。
- **GRPO（Group Relative Policy Optimization）**：DeepSeek 提出的强化学习算法——不给模型训一个单独的价值网络（critic），而是在一组（group）样本内部做归一化算优势，避免训价值网络带来的不稳定。DiffusionDriveV2 把 GRPO 改成「先按 anchor 分组（Intra），再加全局碰撞惩罚（Inter）」，适配到带 anchor 的截断扩散模型。
- **优势函数（Advantage）**：RL 里衡量「这个动作比平均水平好多少」的值。正优势 = 这个轨迹比同组其他好；负优势 = 比同组差。DiffusionDriveV2 用 Intra-Anchor GRPO 算优势（只跟同 anchor 的其他样本比），然后用 Inter-Anchor Truncated GRPO 把正优势保留、负优势截断成 0、碰撞的直接给 -1。
- **乘性探索噪声（Multiplicative Exploration Noise）**：给轨迹加噪声时，不是在每个点 (x,y) 上加独立高斯噪声（加性，会把轨迹变得毛刺），而是统一在纵向/lateral 方向各乘一个随机因子（乘性），保持轨迹的平滑和物理合理性。
- **Mode Selector（模式选择器）**：生成 20 条候选轨迹后，需要一个「最终评分工」挑最好的那条。DiffusionDriveV2 专门训了一个两级 scorer（粗筛 top-k → 细排），用 BCE loss + Margin-Rank loss 学怎么打分。
- **冷启动（Cold Start）**：先用 DiffusionDrive 的 IL 预训练权重初始化生成器，再做 RL 微调。不是从零训。
- **EP（Ego Progress）**：NAVSIM 的一个子指标，衡量自车在 8 秒内前进了多远。DiffusionDriveV2 的 EP 比 v1 高了 5.3（从 82.2 到 87.5），说明 RL 不仅让轨迹更安全（不撞），还让路线更高效（走得远）。
- **DDPM / DDIM**：DDPM 是加随机噪声的去噪过程，DDIM 是确定性快速采样。训练探索时用 DDPM 加噪（`η=1`）获取随机性，推理时用 DDIM（`η=0`）确定性地出结果。

如果你不记得 anchor 怎么聚、截断扩散怎么 2 步去噪、TrajectoryHead 长啥样——建议先读 DiffusionDrive 那篇再回来。这篇直接假设你已经认识它们。

## 为什么要讲 DiffusionDriveV2 的代码

DiffusionDrive（CVPR 2025 Highlight）用 anchor 截断扩散把多样轨迹生成做得很好，但它有个「屋里的大象」：**多样性有了，但质量参差不齐——好轨迹和会撞的坏轨迹一起出，全靠下游分类头去挑。** 训练时 IL 只监督了「离真值最近」的那一个正样本 anchor，剩下 19 个负样本 anchor 完全没被约束，于是它们爱往哪走往哪走、经常出事故。分类头参数少、泛化弱，一旦在 OOD 场景里选错，就是事故。

DiffusionDriveV2（hustvl × Horizon Robotics，arXiv 2512.07745）在同一个生成器上加了 **RL 微调**——不仅仅是加奖励，而是精确地设计了「不让 anchor 间互相比较导致坍缩」的 GRPO 改版。它用 Intra-Anchor GRPO 维持多模态（同 anchor 内比、不同 anchor 不比），用 Inter-Anchor Truncated GRPO 加上全局惩罚（碰撞的杀无赦、正优势保留）。

官方代码仓库 `hustvl/DiffusionDriveV2` 承诺开源但截至本文写作时尚未 release，不过论文给出的算法描述极为详实（含伪代码级别的 RL loss 公式、噪声注入方式、scorer 结构），足以写出可运行的伪代码。这篇**基于 paper 的算法描述，还原成「代码讲解」格式**，让你读完就有能力自己实现。

> **一句话结论**：DiffusionDriveV2 = DiffusionDrive 生成器（完全不变，冷启动加载）+ 19 个负样本 anchor 终于也被 RL 管住了（Intra-Anchor GRPO 不坍缩 + Inter-Anchor 碰撞惩罚），最后加两级 Mode Selector 精排，在 NAVSIM v1 拿下 91.2 PDMS 新 SOTA。

## 架构总览

DiffusionDriveV2 的整体结构，分三块：

- **感知主干（Perception Backbone）**：和 DiffusionDrive 一模一样的 ResNet-34 双路（图像 + LiDAR BEV）+ TransFuser 融合，输出场景特征 fused_feat。
- **截断扩散生成器（Truncated Diffusion Generator）**：和 DiffusionDrive 一模一样的 TrajectoryHead（20 anchor + 2 步去噪 + cls 分类头）。**完全复用 DiffusionDrive 的预训练权重**。
- **RL 微调 + Mode Selector（本文件新增）**：在生成器之上，改训练流程（不再是纯 IL，而是 IL + RL 混合），并新增一个两级 scorer 做最终选轨迹。

数据流：

- 图像 + LiDAR → ResNet-34 主干 → fused_feat
- 20 个 anchor 加 t=8 截断噪声 → TrajectoryHead 去噪 2 步 → 20 条候选轨迹 + 20 个 cls 分数
- 训练时：候选轨迹送入 RL 奖励函数算 reward → GRPO 算优势 → 更新 decoder
- 推理时：候选轨迹送 Mode Selector（两级 scorer）→ 选最优 1 条输出

## 项目结构（推测，基于官方仓库结构约定）

DiffusionDriveV2 同样是 navsim 插件，和 DiffusionDrive 共用同一套 navsim fork，预期新增/修改的文件如下（标注 `(new)` 或 `(mod)`）：

- `navsim/agents/diffusiondrivev2/`（新增）
  - `diffusiondrivev2_agent.py`（new）：Agent 接口，继承 AbstractAgent，基本照抄 DiffusionDrive 的 TransfuserAgent，改模型类为 DiffusionDriveV2Model。
  - `diffusiondrivev2_model.py`（new）：主模型文件。含 `V2TransfuserModel`（复用 DiffusionDrive 的生成器）+ `ModeSelector`（新增两级 scorer）。
  - `diffusiondrivev2_runner.py`（new）：RL 训练入口。负责做 rollout（生成候选轨迹）、算 reward、算 GRPO 优势、反向传播。不再是 PyTorch-Lightning 的标准训练循环，而是自定义 RL 循环。
  - `modules/exploration_noise.py`（new）：乘性噪声实现（纵向横向各一个高斯因子）。
  - `modules/mode_selector.py`（new）：两级 scorer（粗筛 → 细排），含 deformable cross-attention + MLP 打分 + Margin-Rank loss。
  - `modules/grpo_loss.py`（new）：Intra-Anchor GRPO + Inter-Anchor Truncated GRPO 的 loss 计算。
  - `modules/reward.py`（new）：NAVSIM 可微/NON-differentiable 奖励函数封装（PDMS 相关指标拆解：碰撞罚、舒适度、进度）。
  - `diffusiondrivev2_config.py`（new）：超参。（RL lr = 2e-4, batch = 512, GRPO group size G, 等）。
- `scripts/training/run_rl.sh`（new）：启动 RL 训练的脚本。先加载 DiffusionDrive 预训练权重，再跑 RL 微调。

## 一、核心创新逐块拆解（伪代码形式）

### 1.1 扩散生成器：和 DiffusionDrive 一模一样（复用）

生成器不做任何结构改动。核心代码和 DiffusionDrive 完全一致（见 [DiffusionDrive 代码讲解第 1.2 节](/posts/code/diffusiondrive代码讲解/#12-截断扩散策略trajectoryhead-本文核心)）：

```python
plan_anchor = load_anchor("kmeans_navsim_traj_20.npy")
self.plan_anchor = nn.Parameter(tensor(plan_anchor), requires_grad=False)
```

```python
timesteps = torch.randint(0, 50, (B,))
noise = torch.randn_like(anchor)
noisy = diffusion_scheduler.add_noise(anchor, noise, timesteps)
```

训练时加噪 t 在 0~50 随机；推理时固定 t=8，2 步 DDIM 去噪。

唯一区别：**RL 训练时把 DDIM 调度器的 η 从 0（确定性）改成 1（随机性），以获得探索随机性**：

```python
# inference / evaluation: deterministic DDIM
self.scheduler = DDIMScheduler(..., eta=0)
# RL training exploration: stochastic DDPM
self.scheduler_explore = DDIMScheduler(..., eta=1)
```

### 1.2 Scale-Adaptive 乘性探索噪声

为什么不用标准加性噪声？因为轨迹的近端和远端尺度差异大（近端点可能只偏离 0.1 米，远端偏离 5 米），独立加噪声会让轨迹变得毛刺（jagged），失去运动学合理性。

加性噪声的坑：

```python
# additive noise: each (x,y) gets independent Gaussian
eps_add = torch.randn_like(traj)
traj_noisy = traj + sigma * eps_add
# result: jagged trajectory, distal points jitter much more than proximal
```

乘性噪声的正确做法：只加两个随机因子，一个纵向（long），一个横向（lat），乘到整个轨迹上：

```python
# multiplicative noise: preserves smoothness
def add_multiplicative_noise(traj):
    eps_long = torch.randn(B, 20, 1, 1, device=traj.device)
    eps_lat = torch.randn(B, 20, 1, 1, device=traj.device)
    # long on x, lat on y
    noise = torch.cat([eps_long, eps_lat], dim=-1)
    traj_noisy = traj * (1 + sigma * noise)
    return traj_noisy
```

注意只生成 B×20×1×1 个随机数（不是 B×20×T×2），所以同一个轨迹的所有步共享同一个纵向/横向因子——轨迹的几何形状被整体缩放，不会出现「前端直、后端歪」的情况。

作者在消融里验证了乘性比加性 PDMS 高 0.4（89.7 → 90.1），且轨迹平滑度显著更好。

### 1.3 Intra-Anchor GRPO：保住多模态，不让 anchor 间互相内卷

这是最精巧的设计。标准的 GRPO 把所有采样的轨迹放在一个组里算优势，但对 anchor 模型会出事：比如 anchor-5 是「右转」、anchor-10 是「直行」，把它们的样本混在一起比，「右转」的轨迹肯定比「直行」进度慢，于是「右转」的样本全被判负优势，模型就不出右转轨迹了——这就叫 **mode collapse**。

DiffusionDriveV2 的做法是：**每个 anchor 自己成为一个 group，只在自己组内算优势**。

```python
def compute_intra_anchor_grpo_loss(model, batch, G=8):
    N_anchor = 20
    total_loss = 0

    for k in range(N_anchor):
        trajs_k = []
        for i in range(G):
            noise = add_multiplicative_noise(anchor[k])
            noisy = scheduler.add_noise(anchor[k], noise, t=8)
            traj = denoiser(noisy, fused_feat, t_steps=[8,4,0])
            trajs_k.append(traj)

        rewards = [compute_reward(t.detach()) for t in trajs_k]

        r_mean = mean(rewards)
        r_std = std(rewards) + 1e-8
        advantages = [(r - r_mean) / r_std for r in rewards]

        for i in range(G):
            log_prob = compute_log_prob(denoiser, trajs_k[i], fused_feat)
            total_loss += - advantages[i] * log_prob * gamma_weight(t)

    return total_loss / N_anchor
```

关键细节：

- `compute_log_prob` 算的是「在当前策略下生成这条轨迹的似然」。DDPM 的每一去噪步都是高斯分布，对数似然有闭解（Eq.5 in paper）。实际实现时对每步 (t-1|t) 的 Gaussian 概率密度取 log。
- `gamma_weight(t)`：一个折扣系数，给早期去噪步更小的权重，因为早期步的噪声大、梯度信号不可靠。论文用 `gamma_{t-1}` 表示。
- `compute_reward` 调用 NAVSIM 的 PDM 仿真器算奖励——包含碰撞惩罚、进度奖励、舒适度等。奖励函数**不可微**，所以 RL 用 REINFORCE 梯度绕过它。

> **关键认知**：Intra-Anchor GRPO 的精髓是「不同意图不打架」。右转和直行不应该被比较——它们都是合理的驾驶模式，只是适用于不同场景。只有在同一种驱动意图下，才谈得上「这条轨迹好、那条轨迹差」。

### 1.4 Inter-Anchor Truncated GRPO：加一道全局安全底线

Intra-Anchor GRPO 只有一个问题：每组内的「最好的」可能是「矮子里的将军」。比如 anchor-7 组里全是好轨迹（都不撞），优势都在 0 附近；anchor-13 组里全是烂轨迹（都撞），但相对最好的那个优势也可能是正的——模型会错误地去学它。

Inter-Anchor Truncated GRPO 的修正极其简单：

```python
def truncate_advantage(advantages, collisions):
    truncated = []
    for adv, coll in zip(advantages, collisions):
        if coll:
            truncated.append(-1.0)       # collision: heavy penalty
        else:
            truncated.append(max(0, adv))  # keep only positive advantage
    return truncated
```

翻译成人话：

- **没撞**：只保留正优势（鼓励），去掉负优势（不惩罚——因为可能只是这条轨迹「过于保守」而不是真的不好）。
- **撞了**：直接 -1 重罚，不管你的组内优势是多少。

这三行代码是全文最关键的工程 trick。它给模型一个干净的学习信号：**「不撞是底线，在不撞的基础上做得更好」**。去掉负优势避免了「太保守但安全的轨迹」被压下去；重罚碰撞则确保安全是硬约束。

### 1.5 Mode Selector：两级 scorer 精排

DiffusionDrive 的分类头（plan_cls）只有一层，参数少，训练时只监督「谁是正样本 anchor」，不直接优化轨迹质量排序。DiffusionDriveV2 换成一个**独立的两级 scorer**：

```python
class ModeSelector(nn.Module):
    def __init__(self):
        # coarse scorer (first stage)
        self.coarse_scorer = nn.Sequential(
            DeformableCrossAttention(d_model=256),
            nn.Linear(256, 1),
        )
        # fine scorer (second stage, same structure, separate params)
        self.fine_scorer = nn.Sequential(
            DeformableCrossAttention(d_model=256),
            nn.Linear(256, 1),
        )

    def forward(self, trajs, fused_feat, agent_map_queries):
        # Stage-1: coarse, keep top-10
        scores_c = self.coarse_scorer(trajs, fused_feat).squeeze(-1)
        topk_idx = scores_c.topk(10).indices
        trajs_topk = trajs.gather(1, topk_idx)

        # Stage-2: fine
        scores_f = self.fine_scorer(trajs_topk, fused_feat).squeeze(-1)
        best_idx = scores_f.argmax()
        return trajs_topk[best_idx]
```

训练 scorer 用两种 loss：

```python
def selector_loss(pred_scores, gt_scores, trajs):
    # 1. BCE loss: binary classification good/bad
    gt_binary = (gt_scores > threshold).float()
    bce_loss = F.binary_cross_entropy_with_logits(pred_scores, gt_binary)

    # 2. Margin-Rank loss: preserve relative ordering
    rank_loss = 0
    for i, j in all_pairs:
        sign = torch.sign(gt_scores[i] - gt_scores[j])
        margin = pred_scores[i] - pred_scores[j]
        rank_loss += max(0, -sign * margin + m)

    return bce_loss + lambda_rank * rank_loss
```

Margin-Rank loss 防止模型只学绝对分值（难回归），而是学「相对排序」——这通常更容易泛化。

## 二、训练流程：从 IL 冷启动到 RL 微调

训练分两步走：

**Step 1：加载 DiffusionDrive 预训练权重（冷启动）**

```python
state_dict = torch.load("diffusiondrive_pretrained.pth")
model.load_state_dict(state_dict, strict=False)
```

所有感知主干 + 截断扩散 decoder 的参数都复用，不随机初始化。scorer 是新加的，随机初始化。

**Step 2：RL 微调（10 epochs）**

```python
for epoch in range(10):
    for batch in dataloader:
        fused_feat = backbone(batch["img"], batch["lidar_bev"])

        # roll out G candidates per anchor with multiplicative noise
        # paper uses G=8
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
        optimizer.step()
```

超参（paper Sec 5.2）：

- optimizer: AdamW, lr=2e-4
- batch size: 512（分布式 8×L20 GPU）
- GRPO group size G: 8
- gamma (denoising discount): 0.99（推测，具体值未在 paper 出现但此类方法常用）
- lambda_il: paper 说 lambda in (0,1)，消融里用 0.1 左右
- epochs: 10（RL）+ 20（scorer）

**和 DiffusionDrive 训练的关键差异一览**：

| 维度 | DiffusionDrive | DiffusionDriveV2 |
|------|------|------|
| 训练范式 | 纯 IL（BC，监督 GT 轨迹） | IL 预训练 + RL 微调 |
| 监督信号 | 只监督正样本 anchor | 所有 anchor 都被 RL 奖励约束 |
| 噪声类型 | 标准加性高斯 | 乘性（纵向/横向因子） |
| 去噪调度器 η | 0（训练时也用确定性） | 1（探索时随机 DDPM） |
| 是否有 scorer | 简单 cls head（一层 MLP） | 两级 scorer（粗筛+细排+Rank loss） |

## 三、推理流程

推理时 RL 探索关闭、DDIM η=0 回到确定性：

```python
def compute_trajectory(agent_input):
    features = build_features(agent_input)
    fused_feat = backbone(features["img"], features["lidar"])
    trajs, cls_scores = decoder.forward_test(model.anchor, fused_feat)
    # use ModeSelector instead of original cls argmax
    best_traj = mode_selector(trajs, fused_feat, features["agent_map"])
    return best_traj
```

推理开销：主干（一次前向）+ 2 步去噪（同 DiffusionDrive）+ 两级 scorer（两次 cross-attention），在 4090 上仍在实时范围内。Paper 未报具体 FPS，但和 DiffusionDrive 架构基本一致，45 FPS 应该仍有。

## 四、实验结果速览

| 指标 | DiffusionDrive | DiffusionDriveV2 | 提升 |
|------|------|------|------|
| PDMS（NAVSIM v1） | 88.1 | 91.2 | +3.1 |
| EP（Ego Progress） | 82.2 | 87.5 | +5.3 |
| DAC（Drivability） | 96.2 | 97.9 | +1.7 |
| PDMS@Top-1（原始输出） | 93.5 | 94.9 | +1.4 |
| PDMS@Top-10（原始输出） | 75.3 | 84.4 | +9.1 |

PDMS@Top-10 涨了 9.1——这直接证明 RL 把负样本 anchor 的轨迹质量整体抬高了，不再是「好轨迹 + 一大把会撞的」。这就是本文标题说的「上安全锁」。

## 五、个人思考

**1. Intra-Anchor GRPO 的设计哲学值得学。** 它本质上是「分层强化学习」的一种体现：先决定「意图」（选哪个 anchor），再决定「具体走法」（在同意图内做 RL）。很多端到端模型在加入 RL 时直接把所有样本混在一起训，忽略了「不同驾驶模式之间不应该直接比较优劣」。DiffusionDriveV2 这个「分而治之」的思路，对任何有多模态策略的 RL 场景都有借鉴意义。

**2. Inter-Anchor 的截断信号的设计很干净。** 它只有两个数字：`max(0, adv)` 和 `-1`。没有复杂的 reward shaping，没有层级权重——但刚好解决了「矮子里拔将军」和「安全硬约束」这两个问题。说明在 RL 里，**信号的设计比信号的大小重要**。

**3. 乘性噪声让人想起「课程学习（Curriculum Learning）」的直觉。** 加性噪声破坏轨迹的局部结构，模型学起来吃力；乘性噪声保持整体几何，模型学得轻松。这启发我们：在轨迹强结构化的任务里，噪声的几何意义比统计性质更重要。

**4. Mode Selector 其实才是「保底安全」的最后一道防线**，但这篇的工作把防线往前提了——靠 RL 让生成器本身就不出太烂的轨迹，分类头选错的风险就小了。这是比「训一个更强的 scorer」更根本的解法。

**5. 一个延伸问题：冷启动的 DiffusionDrive 权重有多重要？** 如果 DiffusionDriveV2 从零训（不用 IL 预训练），RL 直接探索 20 个 anchor 的高维空间，收敛可能非常慢甚至发散。冷启动相当于给了 RL 一个「已经会开车」的起点，RL 只需要做「微调安全偏好」——这验证了 IL → RL 两步走仍然是端到端驾驶 RL 的最实用路径。

## 附：自己跑起来（等代码开源后）

截至本文写作时 `github.com/hustvl/DiffusionDriveV2` 尚未 release，以下是最小可行路径的预期操作：

1. **环境准备**：安装 navsim 开发套件（和 DiffusionDrive 一样），下载 NAVSIM v1 数据集。

```bash
git clone https://github.com/autonomousvision/navsim.git
cd navsim && pip install -e .
```

2. **下载 DiffusionDrive 预训练权重**：

```bash
# download from huggingface: https://huggingface.co/hustvl/DiffusionDrive
mkdir -p ckpts && wget <weight-link> -O ckpts/diffusiondrive.pth
```

3. **克隆 DiffusionDriveV2 代码并 RL 训练**：

```bash
git clone https://github.com/hustvl/DiffusionDriveV2.git
cd DiffusionDriveV2
# symlink DiffusionDrive pretrained weights
ln -s ../ckpts/diffusiondrive.pth ckpts/diffusiondrive_pretrained.pth
```

```bash
bash scripts/training/run_rl.sh
```

4. **推理出 submission**：

```bash
python navsim/planning/script/run_create_submission_pickle.py \
    agent=diffusiondrivev2_agent experiment_name=ddv2_submission
```

5. **本地算分 / 上榜**：用官方 `run_pdm_score.py` 看 PDMS。

## 和本系列其他文章的关系

- **DiffusionDrive**（前一篇文章）：是用 IL 训的截断扩散基线。理解了它，才懂 DiffusionDriveV2 改了哪里、为什么这样改。
- **SparseDriveV2**：是对立面——检索式规划，不走扩散。两者对比能看清「生成 vs 检索 + RL vs IL」的交错。
- **AutoVLA / DriveVLA-W0**：也用 GRPO 但不走扩散、走离散动作 token。DiffusionDriveV2 证明了 GRPO 在连续扩散轨迹上也能奏效。

---

> 个人总结：DiffusionDriveV2 的贡献不是创造了一个新架构，而是**给已有的锚点扩散生成器配了一套刚好不坍缩的 RL 训练方法**。它最值得学习的是「认清问题所在——多模态 IL 的负样本监督缺失」然后「设计刚好对症的方案——分 anchor 做 GRPO + 干净的两行截断信号」。这种精准打补丁、而不是从头造轮子的思路，是工程落地最需要的素质。
