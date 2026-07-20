---
title: "AutoVLA 代码讲解：把「开车」变成「说动作 token」并用 GRPO 微调"
date: 2026-07-20
description: "拆解 ucla-mobility/AutoVLA：动作 codebook 离散化、快慢思考两阶段训练、SFT + GRPO 拒绝采样微调，看 VLA 如何端到端输出可执行驾驶动作"
tags: ["VLA", "GRPO", "动作token", "代码讲解"]
categories: ["代码讲解"]
summary: "「AutoVLA 用离散 codebook 把连续驾驶动作压缩成 token，让 Qwen2.5-VL 像说话一样『说出』动作；训练分 SFT 模仿与 GRPO 强化两阶段，本文顺着官方仓库讲清 action_token_cluster 与 run_rft 的关键实现」"
---

> 本文是「代码讲解」路线的第 6 篇，紧接前面的 UniAD / VAD（端到端回归范式）和 DriveVLA-W0（VLA + 世界模型）。AutoVLA（NeurIPS 2025，ucla-mobility）把端到端推进到 **「大模型直接输出离散动作 token」** 并用强化学习微调。代码在 **ucla-mobility/AutoVLA**。

## 0. 名词速查

- **动作 codebook（动作词表）**：把连续驾驶动作（转向、油门、刹车等）通过聚类离散化成有限个「动作词」，每个词对应一个 token id。模型输出 token 就等于输出动作。
- **快慢思考（Slow-Fast Thinking）**：先让模型用自然语言「想一遍」（slow，推理/规划理由），再输出动作 token（fast，执行）。推理时慢思考可以被旁路，只留快思考提速。
- **SFT（监督微调）**：用专家驾驶数据模仿学习，让模型先会输出合理动作 token。
- **GRPO / RFT（Rejection sampling Fine-Tuning）**：用规则奖励筛选高质量轨迹样本再微调，等于一种拒绝采样的强化学习，避免训飞。
- **VQA 数据构造**：把驾驶场景问答题（如「前方是否可左转」）合成进训练，增强常识推理。

## 1. 仓库结构

- `AutoVLA/`
  - `auto_vla/model/`：模型定义（Qwen2.5-VL + 动作头）
    - `modeling_auto_vla.py`
    - `action_head.py`：动作 token 投影/解码
  - `auto_vla/data/`：数据集与 action codebook 构造
    - `build_codebook.py`
    - `action_token_dataset.py`
  - `auto_vla/train/`：SFT 与 RFT 训练脚本
    - `sft.py`
    - `grpo_trainer.py`
  - `auto_vla/infer/`：推理（含慢思考旁路）
    - `run_infer.py`
  - `scripts/`
    - `action_token_cluster.sh`：聚类生成 codebook
    - `run_sft.sh`
    - `run_rft.sh`：GRPO / 拒绝采样微调
  - `configs/`

## 2. 动作 codebook：连续动作 → 离散 token

这是 AutoVLA 的核心。先用 KMeans 把大规模驾驶动作聚成 K 个簇，每个簇中心就是一个「动作词」。

```python
# data/build_codebook.py（简化）
from sklearn.cluster import KMeans
actions = load_all_actions()            # (N, D) 连续动作，如 [steer,油门,刹车]
kmeans = KMeans(n_clusters=K, random_state=0).fit(actions)
codebook = kmeans.cluster_centers_      # (K, D)  K 个动作原型
# 保存 codebook，并把每个样本映射到最近簇 id
token_ids = kmeans.predict(actions)     # 连续动作 → 离散 token id
np.save("codebook.npy", codebook)
```

推理时模型输出 token id，再用 codebook 反查成连续动作执行：

```python
# infer 时把 token 还原为连续动作
def token_to_action(token_id, codebook):
    return codebook[token_id]           # (D,)
```

## 3. 模型：在 Qwen2.5-VL 上挂动作头

AutoVLA 基于 **Qwen2.5-VL-3B**，把图像/文本编码进 LLM，再在输出侧接一个动作投影，把隐藏状态映射到 codebook 维度的 logits。

```python
# model/modeling_auto_vla.py（简化）
class AutoVLA(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, num_action_tokens=K):
        super().__init__(config)
        self.action_head = nn.Linear(config.hidden_size, num_action_tokens)

    def forward(self, pixel_values, input_ids, labels, action_labels=None):
        out = super().forward(pixel_values=pixel_values,
                              input_ids=input_ids, labels=labels)
        last_hidden = out.hidden_states[-1]           # (B, L, H)
        action_logits = self.action_head(last_hidden) # (B, L, K)
        if action_labels is not None:
            loss_a = F.cross_entropy(action_logits.view(-1, K),
                                     action_labels.view(-1))
            return out.loss + loss_a
        return action_logits
```

文本部分照常用 next-token 预测（语言/推理理由），动作部分额外做一个分类损失。

## 4. 快慢思考的两阶段训练

### 4.1 Stage-1：SFT 模仿（会开车）

```bash
bash scripts/run_sft.sh
```

SFT 数据里每条样本既包含「慢思考」自然语言推理，也包含最终动作 token 监督。模型先学会「看到场景 → 说出理由 → 输出正确动作 token」。

```python
# train/sft.py 关键（简化）
for batch in dataloader:
    loss = model(
        pixel_values=batch["images"],
        input_ids=batch["input_ids"],      # 含 <reason>...<action> 模板
        labels=batch["text_labels"],
        action_labels=batch["action_token"],
    ).loss
    loss.backward(); optimizer.step()
```

### 4.2 Stage-2：GRPO / RFT 强化（开得更稳）

SFT 容易学出「平均值」动作，危险场景不够果断。AutoVLA 用 **RFT（拒绝采样微调）**：用当前模型 rollout 多条轨迹，按规则奖励（不碰撞、遵守交规、舒适）筛选高奖励样本，再拿这些样本继续 SFT。

```bash
bash scripts/run_rft.sh
```

```python
# train/grpo_trainer.py（RFT 思路，简化）
policy = load_sft_model()
for iter in range(RFT_ROUNDS):
    rollouts = [env.rollout(policy) for _ in range(M)]   # 每条样本采 M 条
    rewards = [rule_reward(traj) for traj in rollouts]   # 规则打分
    keep = [t for t, r in zip(rollouts, rewards) if r > threshold]
    policy = sft(policy, keep)                            # 只在好样本上微调
```

这本质是用 GRPO 类思想（无价值网络、靠组内相对比较）做对齐，但实现上退化成「拒绝采样 + 再监督」，工程上更稳。

## 5. 推理：慢思考可旁路

推理时若追求低延迟，可以跳过慢思考只跑快思考：

```python
# infer/run_infer.py
def infer(model, image, fast_only=True):
    if fast_only:
        prompt = "Describe the scene briefly and output the action token."
    else:
        prompt = "Think step by step about the driving decision, then output action token."
    logits = model(pixel_values=image, input_ids=tokenize(prompt))
    action_id = logits[..., -1, :].argmax(-1)
    return token_to_action(action_id, codebook)
```

## 6. 和 DriveVLA-W0 的对照

- 两者都是「VLA + 离散动作 token + 强化微调」。
- AutoVLA 明确用 **codebook 聚类** 做动作离散化，并配快慢思考；DriveVLA-W0 更强调 **世界模型（未来帧预测）** 做「想清楚再动」。
- 微调范式上 AutoVLA 用 RFT/GRPO 拒绝采样，DriveVLA-W0 用 Flow-GRPO 风格的流式强化。

## 7. 值得抄的点

- **动作 codebook 是连接「语言模型」和「控制器」的桥**：把连续控制问题转成离散分类，直接复用 LLM 训练栈。
- **快慢思考**让同一模型兼顾可解释（推理）与低延迟（旁路）。
- **RFT 比纯 GRPO 好落地**：不需要精确的奖励模型，规则奖励 + 拒绝采样就能显著提安全率。

---

> 个人思考：AutoVLA 是「端到端回归（UniAD/VAD）→ 生成式扩散（DiffusionDrive）→ VLA 离散动作（本文）」这条线上的代表。它最妙的地方是把驾驶动作塞进 LLM 的 token 空间，于是「强化学习对齐」可以几乎原样复用 NLP 的 GRPO 套路。下一站 pi0 会告诉我们：动作不一定非得离散，Flow Matching 直接对连续动作做生成也很好用。
