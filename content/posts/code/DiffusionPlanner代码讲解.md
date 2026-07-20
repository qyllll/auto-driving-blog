---
title: "Diffusion Planner 代码讲解：轨迹级扩散 + 免训练能量引导"
date: 2026-07-20
description: "拆解 ZhengYinan-AIR/Diffusion-Planner：DiT + MLP-Mixer 去噪轨迹、DDPM 训练、classifier guidance 做碰撞/可达性无训练约束，看轨迹扩散规划如何做到 SOTA 且即插即用"
tags: ["扩散模型", "轨迹规划", "能量引导", "代码讲解"]
categories: ["代码讲解"]
summary: "「Diffusion Planner 把规划对象从『动作』升级为『整条轨迹』，用 DiT + MLP-Mixer 去噪；亮点是在 DDPM 采样时注入 classifier guidance（碰撞/可行驶区域/运动学能量），无需重新训练就能把轨迹压进可行域，本文顺着官方仓库讲清 guidance 模块」"
---

> 本文是「代码讲解」路线的第 8 篇，是 Diffusion Policy 在「规划」上的专门化（ICLR 2025 Oral，OpenDriveLab / ZhengYinan-AIR）。代码在 **ZhengYinan-AIR/Diffusion-Planner**。

## 0. 名词速查

- **轨迹级扩散**：直接对「未来 T 步 (x,y,heading) 序列」做扩散，而不是逐步动作。规划即生成一整条路径。
- **DiT（Diffusion Transformer）**：用 Transformer 块做去噪网络，替代 U-Net，长序列建模更强。
- **MLP-Mixer**：在轨迹 token 之间做「模态混合」，负责把轨迹点之间的时序相关性揉顺（运动学平滑）。
- **Classifier Guidance / DPS**：采样时额外加一个能量项梯度（如碰撞代价），把样本「推离」不可行区域。免训练，灵活可插拔。
- **多模态规划**：同一初始状态可采出多条不同风格轨迹（保守/激进），下游再选。

## 1. 仓库结构

- `Diffusion-Planner/`
  - `diffusion_planner/`
    - `model/`
      - `dit.py`：DiT 去噪主干
      - `mixer.py`：MLP-Mixer 时序混合
      - `guidance/`：免训练能量引导
        - `collision.py`：碰撞能量
        - `drivable.py`：可行驶区域能量
        - `kinematic.py`：运动学平滑能量
    - `scheduler/ddpm.py`：加噪/去噪调度
    - `env/nuplan_wrapper.py`：闭环评测环境
    - `configs/`
  - `scripts/`
    - `train.py`
    - `eval_closed_loop.py`

## 2. 去噪网络：DiT + MLP-Mixer

轨迹是 (T, D) 序列，先切成 token，过 DiT 块提取结构，再过 MLP-Mixer 做时序混合。

```python
# model/dit.py（简化）
class DiTBlock(nn.Module):
    def __init__(self, dim):
        self.attn = nn.MultiheadAttention(dim, 8)   # 轨迹 token 互看
        self.mixer = MLPMixer(dim)                  # 时序/特征混合
        self.adaLN = AdaLNNorm(dim)                 # 时间步条件注入
    def forward(self, x, t_emb):
        x = x + self.attn(self.adaLN(x, t_emb))
        x = x + self.mixer(x)
        return x

class TrajectoryDiT(nn.Module):
    def forward(self, noisy_traj, t, cond):
        t_emb = self.timestep_emb(t)
        h = self.input_proj(noisy_traj) + self.pos_embed   # (B, T, D)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.head(h)                                # 预测噪声
```

`cond` 是场景编码（ego 状态 + 邻居轨迹 + 地图），通过 cross-attention 或 FiLM 注入。

## 3. 训练：标准 DDPM，但目标是轨迹

```python
# 训练一步（简化）
def train_step(model, scene, traj):
    noise = torch.randn_like(traj)
    t = torch.randint(0, T, (B,))
    noisy = q_sample(traj, noise, t)
    pred = model(noisy, t, cond=encode_scene(scene))
    loss = F.mse_loss(pred, noise)        # 预测加在轨迹上的噪声
    return loss
```

训练数据来自 nuPlan 等真实驾驶日志，轨迹是专家驾驶真值。

## 4. 推理亮点：免训练能量引导（DPS）

这是 Diffusion Planner 最实用的一招。标准 DDPM 采样只靠学到的分布，可能出碰撞轨迹。它在每步去噪时**额外减去能量项梯度**，把轨迹推离不可行区：

```python
# scheduler/ddpm.py 采样 + guidance（简化）
def sample(self, model, scene):
    x = torch.randn((B, T, D))
    for t in reversed(range(T)):
        eps = model(x, t, cond=scene)                 # 预测噪声
        x = self.posterior_step(x, eps, t)            # 标准去噪
        # 免训练引导：计算能量梯度并反向推
        for guide in self.guides:                     # collision/drivable/kinematic
            E = guide.energy(x, scene)                # 标量能量
            g = torch.autograd.grad(E, x)[0]          # dE/dx
            x = x - self.guide_scale * g              # 沿能量下降方向移
    return x
```

各 energy 的含义：
- `collision.py`：轨迹点与其他 agent 未来位置的距离惩罚，近则能量高。
- `drivable.py`：轨迹点落到不可行驶区域（如马路牙子外）的惩罚。
- `kinematic.py`：相邻步之间曲率/加速度超限的平滑惩罚。

因为这些都在推理时算，所以**换地图、换规则不用重训模型**，即插即用——这对部署极其友好。

## 5. 多模态：一条场景采多条

```python
# eval_closed_loop.py
candidates = [planner.sample(scene) for _ in range(M)]  # M 条候选轨迹
scores = [rule_score(traj, scene) for traj in candidates]
best = candidates[int(torch.stack(scores).argmax())]
```

和 VADv2 的「多模态概率」思路一致，但轨迹是用扩散**采样**出来的，覆盖更自然的分布。

## 6. 为什么比回归式规划强

- 回归式（UniAD/VAD）只出一条平均轨迹，遇到歧义场景（可左可右）会「犹豫」出危险折中。
- 扩散直接建模多模态轨迹分布，再用能量引导筛掉不可行项，安全率和舒适度都更高。
- 代价是推理要跑多步去噪（虽比图像扩散轻得多，轨迹才几十个点）。

## 7. 在本系列的位置

- 方法论母体 → Diffusion Policy（上篇）。
- 同属「扩散用于驾驶」→ DiffusionDrive（生成式端到端，flow matching 版）。
- 离散动作的对照 → AutoVLA（把动作做成 token 而非连续扩散）。

---

> 个人思考：Diffusion Planner 的「免训练能量引导」是我最想抄到 Flow-GRPO 里的设计——它把「安全约束」从训练期解放到推理期，意味着规则变了不用重训。结合 GRPO 的奖励信号，其实可以统一成「采样 + 可微约束」，这正是当前端到端规划很有前景的方向。
