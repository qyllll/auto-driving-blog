---
title: "pi0 代码讲解：通用 VLA 的「VLM 主干 + Flow Matching 动作专家」"
date: 2026-07-20
description: "拆解 Physical-Intelligence/openpi：PaliGemma VLM 当通用感知-语义主干，独立的 Flow Matching action expert 在 10 步 ODE 内生成连续动作，看通用机器人 VLA 如何端到端输出动作"
tags: ["VLA", "Flow Matching", "机器人", "代码讲解"]
categories: ["代码讲解"]
summary: "「pi0 用 PaliGemma 视觉语言模型当『懂场景』的主干，再挂一个独立的 Flow Matching 动作专家，在 10 步 ODE 内把噪声动作流到真实动作；本文顺着 Physical-Intelligence 官方 openpi 仓库讲清两塔结构与动作生成」"
---

> 本文是「代码讲解」路线的第 9 篇（收官篇）。pi0（Physical Intelligence，2024）是**通用机器人 VLA** 的代表，但它和 AutoVLA / DriveVLA-W0 共享同一套思想：VLM 理解世界 + 动作头生成执行。代码在官方 **Physical-Intelligence/openpi**。

## 0. 名词速查

- **PaliGemma**：Google 的轻量 VLM（SigLIP 视觉塔 + Gemma 语言塔），pi0 用它当通用感知-语义主干。
- **Action Expert（动作专家）**：pi0 里一个**独立Transformer**，专门把「VLM 特征 + 噪声动作」去噪成真实动作。它和 VLM 不是同一个权重，是并联的两塔。
- **Flow Matching（流匹配）**：不像 DDPM 逐步加噪再去噪，而是学一个**从噪声到数据的向量场（ODE）**，沿 ODE 积分几步就到数据。pi0 只用 10~50 步，比 DDPM 快一个数量级。
- **交叉注意力注入**：动作专家通过 cross-attention 从 VLM 主干「取」语义特征，而不是把 VLM 整个重训。
- **多机器人通用**：同一份架构，换不同的 action 投影头和 tokenizer，就能跑机械臂、折叠、收拾等不同任务。

## 1. 仓库结构

- `openpi/`
  - `src/openpi/models/`
    - `pi0.py`：pi0 总装，VLM + Action Expert
    - `paligemma.py`：PaliGemma 主干封装
    - `action_expert.py`：Flow Matching 动作专家
  - `src/openpi/training/`
    - `config.py`：训练超参
    - `trainer.py`：训练循环
  - `src/openpi/inference/runtime.py`：推理（ODE 采样）
  - `src/openpi/transforms/`：数据预处理 / tokenizer
  - `examples/aloha/`：具体机器人适配示例
  - `scripts/`：`train.py` / `infer.py`

## 2. 两塔结构：VLM 主干 + 动作专家

pi0 的精髓是**动作专家独立成塔**，VLM 只负责理解，不被动作梯度污染。

```python
# models/pi0.py（简化）
class PI0(nn.Module):
    def __init__(self, paligemma, action_dim, action_horizon):
        self.vlm = paligemma                       # 冻结或低 lr 的 VLM 主干
        self.action_expert = ActionExpert(
            inp_dim=action_dim + vlm_hidden,
            out_dim=action_dim,
            horizon=action_horizon,
        )

    def forward(self, images, lang, noisy_actions, t):
        vlm_feat = self.vlm(images, lang)          # (B, L, H) 语义特征
        # 动作专家：把噪声动作 + VLM 特征一起过
        act_out = self.action_expert(noisy_actions, vlm_feat, t)
        return act_out                             # 预测向量场 v(x,t)
```

## 3. 动作专家：Flow Matching 去噪

动作专家是一个 Transformer，输入是「噪声动作序列 + 时间 t + VLM 特征」，输出是**指向真实动作的向量场**。

```python
# models/action_expert.py（简化）
class ActionExpert(nn.Module):
    def __init__(self, inp_dim, out_dim, horizon, n_layer=8):
        self.tok_emb = nn.Linear(inp_dim, D)
        self.blocks = nn.ModuleList([ExpertBlock(D) for _ in range(n_layer)])
        self.head = nn.Linear(D, out_dim)

    def forward(self, noisy_actions, vlm_feat, t):
        x = self.tok_emb(torch.cat([noisy_actions, vlm_feat], dim=-1))
        for blk in self.blocks:
            x = blk(x, t, cross=vlm_feat)          # 自注意 + 对 VLM 交叉注意
        return self.head(x)                        # 向量场 v
```

训练时，Flow Matching 的损失是「预测向量场」与「真值流（数据-噪声方向）」的回归：

```python
# training/trainer.py 中（简化）
def fm_loss(model, obs, action):
    noise = torch.randn_like(action)
    t = torch.rand((B,))                           # 连续时间 [0,1]
    x_t = (1 - t) * noise + t * action             # 线性插值路径
    v_target = action - noise                      # 真值向量场
    v_pred = model(obs, x_t, t)                    # 网络预测
    return F.mse_loss(v_pred, v_target)
```

## 4. 推理：10 步 ODE 积分

Flow Matching 不需要上百步去噪，沿 ODE 积分十几步即可：

```python
# inference/runtime.py
def predict(self, images, lang, n_step=10):
    vlm_feat = self.vlm(images, lang)
    x = torch.randn((B, Ta, action_dim))           # 从噪声起步
    for i in range(n_step):
        t = i / n_step
        v = self.action_expert(x, vlm_feat, t)
        x = x + (1.0 / n_step) * v                 # Euler 积分一步
    return x                                       # 真实动作序列
```

Euler 一步就逼近，所以 pi0 实时性远好于 DDPM 类方法（Diffusion Policy 要 20~100 步）。

## 5. 数据 / tokenizer：怎么把动作接进去

不同机器人动作维度不同，pi0 用 `transforms` 把观测（图像+语言）和动作分别 tokenize：

```python
# transforms/ 中
obs = transform_obs(raw_images, lang_prompt)   # → VLM 输入
action = transform_action(robot_state)         # → (Ta, action_dim)
```

训练时动作头只训动作专家 + 轻量适配层，VLM 主干大多冻结或极低学习率，避免灾难性遗忘通用语义。

## 6. 和自动驾驶 VLA 家族的对照

- **AutoVLA**：动作做成**离散 token**，用 GRPO 对齐；pi0 用**连续 Flow Matching**，不用 RL。
- **DriveVLA-W0**：强调**世界模型**（预测未来帧）；pi0 不做显式世界模型，只做「观测→动作」。
- 共同点：都是「VLM 理解 + 独立动作头生成」的两段式，证明这条路在机器人/驾驶都通。

## 7. 收官：本系列九篇串起来

```text
端到端回归：   UniAD（全栈 query） → VAD（轻量向量化，VADv2 多模态概率）
生成式扩散：   Diffusion Policy（通用动作扩散） → DiffusionDrive（驾驶端到端）
              → Diffusion Planner（轨迹扩散 + 能量引导）
VLA 路线：     AutoVLA（离散动作 token + GRPO） → DriveVLA-W0（VLA + 世界模型）
通用 VLA：     pi0（VLM 主干 + Flow Matching 动作专家）
```

## 8. 个人总结

九篇写下来，端到端自动驾驶的方法论收敛成两条互补路线：
- **生成式**（扩散 / Flow Matching）解决「多模态 + 安全分布」；
- **语言化**（VLA）解决「可解释 + 常识推理 + 易对齐」。

pi0 站在通用机器人视角告诉我们：动作完全可以是**连续 Flow Matching 生成**，不一定非得像 AutoVLA 那样离散成 token。两者各有取舍，而真正落地的系统大概率是「VLM 语义主干 + 连续/离散动作专家 + 强化/规则对齐」的混合体——这也是我后续在 Flow-GRPO 上想继续推进的方向。

---

> 思考题：如果你要设计一个「既能用语言解释决策、又能实时输出连续平滑轨迹、还能用规则/奖励在线约束」的驾驶模型，会从这九篇里各借哪一块？欢迎在评论区交流。
