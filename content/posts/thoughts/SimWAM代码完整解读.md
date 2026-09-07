---
title: "SimWAM 代码完整解读：从 Video DiT 到 Action DiT，从 SFT 到 FlowGRPO 的逐行拆解"
date: 2026-09-06
draft: false
categories: ["个人思考"]
summary: "从 README 第 1 行到 trainer_grpo.py 最后 1 行，逐模块拆解 SimWAM 的完整代码逻辑：Video Expert（Wan2.2-5B DiT）怎么加载？Action Expert（ActionDiT）怎么初始化？联合 Flow Matching 训练怎么跑？Isolated Attention Mask 怎么实现？FlowGRPO RL 后训练的 SDE 采样、PDM 奖励、LoRA 微调怎么串起来？每段代码都标注了源文件路径。"
tags: ["SimWAM", "World Action Model", "Flow Matching", "GRPO", "Wan2.2", "自动驾驶", "端到端"]
math: true
weight: 98
---

## 入门篇：SimWAM 到底是什么？（小白版）

> 如果你第一次听说"世界动作模型""Video DiT""Action DiT""FlowGRPO"——**先读这一篇**。
> 这一篇不讲任何代码，只帮你建立**直觉**。

### 0.1 一句话版

> **SimWAM = 一个冻结的视频生成模型（Wan2.2-5B）+ 一个轻量动作专家（ActionDiT），用联合 Flow Matching 训练，让视频预测的"运动先验"流入轨迹生成——但推理时不需要真的生成未来帧，只用动作专家直接输出轨迹。**

论文：*SimWAM: A Simple World Action Model for End-to-End Autonomous Driving*（arXiv:2608.07468，华中科技大学 × 东风）。

它和传统做法的本质区别：

| 传统端到端（UniAD / VAD） | World Model（DriveWAM / DriveLaW） | SimWAM |
|---|---|---|
| 直接从观测映射到轨迹 | 先想象未来帧，再规划 | **训练时**用视频预测传递运动先验，**推理时**不需要生成未来帧 |
| 无显式时序建模 | 显式预测未来（慢） | 隐式传递时序信息（快） |
| 轨迹质量依赖模仿学习 | 轨迹质量依赖未来帧质量 | 轨迹质量 = 运动先验 + RL 后训练 |

一句话概括动机：**视频生成模型里藏着丰富的"物体怎么运动"的先验——SimWAM 把这个先验"蒸馏"到一个轻量动作专家里，让它直接输出好轨迹，而不需要在推理时真的花时间生成未来帧。**

### 0.2 核心设计：Isolated Attention Mask

SimWAM 最精巧的设计是一个**注意力掩码**——它让动作 token 和未来帧 token 都能看到当前观测，但**互相看不到**：

```
训练时：
  观测帧 tokens ← 被 Video DiT 和 Action DiT 都看到
  未来帧 tokens ← 只被 Video DiT 看到（预测未来视频）
  动作 tokens   ← 只被 Action DiT 看到（生成轨迹）
  未来帧 tokens ←→ 动作 tokens：互相隔离！

推理时：
  直接扔掉 Video DiT，只用 Action DiT
  因为动作 tokens 从来不依赖未来帧 tokens，所以不需要生成未来帧
```

这就是"**训练时借力、推理时独立**"的精髓。

### 0.3 两个核心概念

| 名字的一部分 | 对应概念 | 负责的事 |
|---|---|---|
| **Sim** | Simple（简洁） | 不搞复杂模块，用现成视频模型 |
| **WAM** | World Action Model | 视频世界模型 + 动作预测联合训练 |

两个专家：

| 专家 | 身份 | 参数量 | 训练时 | 推理时 |
|---|---|---|---|---|
| **Video Expert** | Wan2.2-TI2V-5B DiT（冻结/微调） | ~5B | 联合训练，预测未来帧 | **不用** |
| **Action Expert** | ActionDiT（从头训/LoRA） | ~0.2B-1B | 联合训练，生成轨迹 | **只用这个** |

---

## 📁 代码结构总览

SimWAM 的代码结构非常清晰：

```
SimWAM/
├── src/simwam/                    # 核心代码
│   ├── models/
│   │   └── wan22/
│   │       ├── simwam.py          # ⭐ 主模型（53KB）：SimWAM 类，联合前向、损失、推理
│   │       ├── simwam_grpo.py     # ⭐ GRPO 模型（22KB）：SimWAMGRPO 类，RL 训练
│   │       ├── action_dit.py      # ⭐ Action DiT（13KB）：动作专家架构
│   │       ├── mot.py             # ⭐ MOT（24KB）：Mixture-of-Transformers 联合注意力
│   │       ├── wan_video_dit.py   # Wan2.2 Video DiT 封装
│   │       ├── wan_video_vae.py   # Wan2.2 Video VAE 封装
│   │       ├── wan_video_text_encoder.py  # T5 文本编码器
│   │       ├── lora.py            # LoRA 适配器实现
│   │       └── schedulers/        # Flow Matching 调度器
│   ├── datasets/                  # 数据加载
│   │   └── navsim/               # NAVSIM 数据集 + PDM 奖励
│   ├── runtime.py                 # SFT 训练入口
│   ├── runtime_grpo.py            # GRPO 训练入口
│   ├── trainer_grpo.py            # ⭐ GRPO 训练器（41KB）
│   └── trainer_tensorboard.py     # TensorBoard 日志
├── configs/
│   ├── model/simwam_navsim.yaml   # 模型配置
│   ├── task/
│   │   ├── navsim_uncond_front_384x672_1e-4.yaml  # SFT 任务
│   │   └── navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml  # GRPO 任务
│   └── train.yaml / train_grpo.yaml
├── scripts/
│   ├── train.py                   # SFT 训练启动脚本
│   ├── train_grpo.py              # GRPO 训练启动脚本
│   └── train_navsim_zero1_torchrun.sh  # 多机分布式启动
├── experiments/navsim/
│   ├── eval_navsim.py             # 评测脚本
│   └── run_eval_navsim.sh
└── navsim/                        # NAVSIM devkit
```

**阅读顺序建议**：`configs/model/simwam_navsim.yaml` → `action_dit.py` → `mot.py` → `simwam.py` → `simwam_grpo.py` → `trainer_grpo.py` → `runtime_grpo.py`。

---

## 🧠 模块一：Action DiT（动作专家）

**源文件**：`src/simwam/models/wan22/action_dit.py`（13KB）

Action DiT 是 SimWAM 的核心创新——一个轻量级的 Diffusion Transformer，专门负责从 VLM 表征生成轨迹。

### 1.1 架构参数（来自 configs/model/simwam_navsim.yaml）

```yaml
action_dit_config:
  action_dim: 3          # (x, y, heading)
  hidden_dim: 1024       # 隐藏维度
  ffn_dim: 4096          # FFN 中间维度
  num_heads: 24          # 注意力头数
  attn_head_dim: 128     # 每头维度（1024/8=128）
  num_layers: 30         # Transformer 层数
  text_dim: 4096         # 文本条件维度（与 T5 对齐）
  freq_dim: 256          # 时间步 Fourier 编码维度
  eps: 1.0e-06           # LayerNorm eps
```

**参数量计算**：30 层 × (self-attention + cross-attention + FFN) ≈ **~0.2B-1B**（取决于是否用 LoRA）。

### 1.2 核心类结构

```python
class ActionDiT(nn.Module):
    """
    轻量 Diffusion Transformer，从噪声轨迹生成干净轨迹。
    条件输入：VLM 的缓存 KV + 时间步 + 导航指令 + 自车状态。
    """
    def __init__(self, action_dim, hidden_dim, ffn_dim, num_heads,
                 attn_head_dim, num_layers, text_dim, freq_dim, ...):
        # 1. 轨迹嵌入：把 (x, y, heading) 的 noisy 轨迹 + Fourier 特征 → hidden_dim
        self.traj_embed = nn.Linear(action_dim * 2 + freq_dim, hidden_dim)  # 轨迹 + Fourier
        
        # 2. 时间步嵌入：Flow time → Fourier → MLP → hidden_dim
        self.time_embed = TimestepEmbedding(freq_dim, hidden_dim)
        
        # 3. 文本嵌入：导航指令（T5 输出）→ project → hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 4. 自车状态嵌入：当前速度/加速度等 → project → hidden_dim
        self.ego_proj = nn.Linear(ego_dim, hidden_dim)
        
        # 5. 30 层 Transformer Block
        self.blocks = nn.ModuleList([
            ActionDiTBlock(hidden_dim, ffn_dim, num_heads, attn_head_dim, ...)
            for _ in range(num_layers)
        ])
        
        # 6. 输出头：hidden_dim → (action_dim × 2)  # 预测速度场
        self.out_proj = nn.Linear(hidden_dim, action_dim * 2)
```

### 1.3 ActionDiTBlock 的内部结构

每个 Block 包含三个核心操作：

```python
class ActionDiTBlock(nn.Module):
    def forward(self, x, t_emb, kv_cache, mask):
        # 1. Self-Attention（动作 token 之间互相看）
        #    - 动作 token 是 50 个路点（5s @10Hz），每个 3 维 (x, y, heading)
        #    - 加入因果掩码：后面的路点可以看到前面的，但前面的看不到后面的
        x = x + self.self_attn(self.norm1(x), t_emb, mask=self_causal_mask)
        
        # 2. Cross-Attention（动作 token 看 VLM 的缓存 KV）
        #    - 这是"世界模型信息流入动作专家"的唯一通道
        #    - VLM 的 KV 是从当前观测帧编码的，包含了场景理解
        x = x + self.cross_attn(self.norm2(x), kv_cache, mask=cross_mask)
        
        # 3. FFN（前馈网络）
        x = x + self.ffn(self.norm3(x))
        
        return x
```

**关键设计**：Cross-Attention 的 `kv_cache` 来自 VLM 的 softmax attention 层——这就是"世界模型信息流入动作专家"的管道。动作 token 通过 cross-attention 读取 VLM 编码的场景表征，但**看不到未来帧 token**（因为 isolated attention mask）。

---

## 🔗 模块二：MOT（Mixture-of-Transformers 联合注意力）

**源文件**：`src/simwam/models/wan22/mot.py`（24KB）

MOT 是 SimWAM 的"胶水层"——它让 Video DiT 和 Action DiT 共享注意力接口，但通过掩码隔离信息流。

### 2.1 核心思想

```
传统 Mixture-of-Transformers：
  Video tokens 和 Action tokens 混在一起做联合注意力
  → 信息自由流动，Action 会看到未来帧

SimWAM 的 Isolated MOT：
  Video tokens 和 Action tokens 仍然拼在一起做注意力
  → 但用掩码阻止 Action tokens 看到未来帧 tokens
  → 同时允许两者都看到当前观测 tokens
```

### 2.2 掩码矩阵的设计

```python
def build_isolated_mask(video_len, action_len, obs_len):
    """
    构建 isolated attention mask：
    - 观测 tokens：所有 token 都能看到
    - 未来帧 tokens：只有 Video DiT 能看到
    - 动作 tokens：只有 Action DiT 能看到
    """
    total_len = obs_len + video_len + action_len
    
    # 初始化全 1 矩阵（允许所有注意力）
    mask = torch.ones(total_len, total_len, dtype=torch.bool)
    
    # 观测 tokens（前 obs_len 个）：所有 token 都能看到
    mask[:, :obs_len] = True
    
    # 未来帧 tokens（obs_len 到 obs_len+video_len）：
    #   - 只能被自己（Video DiT）看到
    #   - 不能被动作 tokens 看到
    mask[obs_len+video_len:, obs_len:obs_len+video_len] = False  # 动作→未来：禁止
    
    # 动作 tokens（最后 action_len 个）：
    #   - 只能被自己（Action DiT）看到
    #   - 不能被未来帧 tokens 看到
    mask[obs_len:obs_len+video_len, obs_len+video_len:] = False  # 未来→动作：禁止
    
    return mask
```

### 2.3 推理时的关键优化

```python
# 推理时：直接扔掉 Video DiT 部分，只跑 Action DiT
# 因为动作 tokens 从来不依赖未来帧 tokens
# 所以不需要生成未来帧，大幅减少计算量

# 训练时：两个 DiT 都跑，联合优化
# 推理时：只跑 Action DiT，直接输出轨迹
# → 这就是"训练时借力、推理时独立"
```

---

## 🏗️ 模块三：SimWAM 主模型

**源文件**：`src/simwam/models/wan22/simwam.py`（53KB）

这是最大的文件，包含了完整的前向传播、损失计算和推理逻辑。

### 3.1 模型初始化流程

```python
class SimWAM(nn.Module):
    @classmethod
    def from_wan22_pretrained(cls, model_id, ...):
        """
        从 Wan2.2 预训练权重初始化 SimWAM。
        
        步骤：
        1. 加载 Wan2.2 Video DiT 权重（视频生成 backbone）
        2. 加载 Wan2.2 Video VAE（把图像压缩到 latent）
        3. 加载 T5 文本编码器（编码导航指令）
        4. 初始化 ActionDiT（从预训练权重或随机初始化）
        5. 构建 MOT 联合注意力层
        """
        # 1. Video DiT
        video_dit = WanVideoDiT.from_pretrained(model_id)
        
        # 2. Video VAE
        video_vae = WanVideoVAE.from_pretrained(model_id)
        
        # 3. T5 文本编码器
        text_encoder = WanVideoTextEncoder.from_pretrained(tokenizer_model_id)
        
        # 4. Action DiT
        action_dit = ActionDiT(**action_dit_config)
        if action_dit_pretrained_path:
            action_dit.load_state_dict(torch.load(action_dit_pretrained_path))
        
        # 5. MOT（共享注意力接口）
        mot = MOT(video_dit, action_dit, ...)
        
        return cls(video_dit, action_dit, video_vae, text_encoder, mot, ...)
```

### 3.2 训练时的前向传播

```python
def forward(self, images, trajectories, text_input, ...):
    """
    训练时的完整前向传播。
    
    输入：
      - images: 当前帧 + 未来帧的图像 (B, T, C, H, W)
      - trajectories: 专家轨迹 (B, 50, 3)  # 50 路点 × (x, y, heading)
      - text_input: 导航指令的 token IDs
    
    输出：
      - video_loss: 未来帧预测损失
      - action_loss: 轨迹生成损失
    """
    # 1. VAE 编码：图像 → latent
    video_latents = self.video_vae.encode(images)  # (B, T, C', H', W')
    
    # 2. 文本编码：导航指令 → embedding
    text_emb = self.text_encoder(text_input)  # (B, seq_len, 4096)
    
    # 3. 采样时间步 t（Flow Matching）
    t = torch.rand(B, device=devices[0])
    
    # 4. 构造噪声轨迹（Flow Matching 插值）
    noise = torch.randn_like(trajectories)
    noisy_traj = (1 - t) * noise + t * trajectories  # 线性插值
    
    # 5. 联合前向传播（MOT + isolated mask）
    video_pred, action_pred = self.mot(
        video_latents=video_latents,
        noisy_traj=noisy_traj,
        text_emb=text_emb,
        t=t,
        mask=isolated_mask,  # ⭐ 关键：隔离掩码
    )
    
    # 6. 计算损失
    video_loss = F.mse_loss(video_pred, video_latents)
    action_loss = F.mse_loss(action_pred, trajectories - noise)  # 速度场
    
    total_loss = self.lambda_video * video_loss + self.lambda_action * action_loss
    return total_loss, video_loss, action_loss
```

### 3.3 推理时的轨迹生成

```python
@torch.no_grad()
def generate_trajectory(self, images, text_input, num_steps=10):
    """
    推理时：只用 Action DiT，不需要 Video DiT。
    
    流程：
    1. VAE 编码当前帧
    2. T5 编码导航指令
    3. 从高斯噪声初始化轨迹
    4. 10 步欧拉积分 → 干净轨迹
    """
    # 1. 编码当前帧
    obs_latent = self.video_vae.encode(images[:, :1])  # 只编码当前帧
    
    # 2. 编码导航指令
    text_emb = self.text_encoder(text_input)
    
    # 3. 初始化噪声轨迹
    traj = torch.randn(B, 50, 3, device=device)  # 50 路点 × 3
    
    # 4. 10 步欧拉积分
    for i in range(num_steps):
        t = i / num_steps
        # Action DiT 预测速度场
        v_pred = self.action_dit(traj, t, text_emb, obs_latent)
        # 欧拉步
        traj = traj + v_pred * (1.0 / num_steps)
    
    return traj  # (B, 50, 3)  # 最终轨迹
```

**注意**：推理时**完全没有 Video DiT**——因为动作 token 从来不依赖未来帧 token，所以不需要生成未来帧。这就是 SimWAM 推理速度快的原因。

---

## 🎓 模块四：SFT 训练流程

**源文件**：`src/simwam/runtime.py` + `scripts/train.py`

### 4.1 训练入口

```python
# scripts/train.py
@hydra.main(config_path="../configs", config_name="train")
def main(cfg):
    run_training(cfg)  # → simwam.runtime.run_training

# simwam/runtime.py
def run_training(cfg):
    # 1. 构建模型
    model = instantiate(cfg.model)  # SimWAM.from_wan22_pretrained(...)
    
    # 2. 构建数据集
    train_dataset = instantiate(cfg.data.train)  # NAVSIM 数据集
    
    # 3. 构建训练器
    trainer = Trainer(model=model, dataset=train_dataset, cfg=cfg)
    
    # 4. 开始训练
    trainer.train()
```

### 4.2 训练配置（configs/task/navsim_uncond_front_384x672_1e-4.yaml）

```yaml
batch_size: 2
learning_rate: 1e-4
num_epochs: 100
lr_scheduler_type: "cosine"
weight_decay: 1e-2
gradient_accumulation_steps: 1

model:
  video_dit_config:
    hidden_dim: 3072      # Video DiT 隐藏维度
    num_layers: 30         # Video DiT 层数
  action_dit_config:
    hidden_dim: 1024       # Action DiT 隐藏维度
    num_layers: 30         # Action DiT 层数
```

### 4.3 损失函数

```python
# simwam.py 中的损失计算
loss = lambda_video * MSE(video_pred, video_target)  # 未来帧预测
     + lambda_action * MSE(action_pred, action_target)  # 轨迹速度场预测

# 默认 lambda_video=1.0, lambda_action=1.0
# 两个损失等权重，让视频预测和轨迹生成同时学习
```

---

## 🔑 模块五：FlowGRPO RL 后训练

**源文件**：`src/simwam/trainer_grpo.py`（41KB）+ `src/simwam/runtime_grpo.py`

这是 SimWAM 最复杂的部分——用 GRPO 式强化学习微调 Action Expert。

### 5.1 GRPO 训练配置

```yaml
# configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml
grpo:
  lora:
    enabled: true          # 只训练 LoRA 适配器
    r: 16                  # LoRA rank
    alpha: 32.0            # LoRA alpha
    target_modules: [q, k, v, o]  # 只在注意力层加 LoRA
  
  sample:
    sde_mode: rigorous     # 严格的 SDE 采样（score-corrected）
    noise_level: 0.1       # SDE 噪声级别
  
  train:
    ppo_clip_range: 0.02   # PPO clip 范围
    adv_clip_max: 5.0      # advantage 截断
    num_inner_epochs: 4    # 每批 rollout 复用 4 次
    rollout_buffer_batches: 1
  
  reward:
    # NAVSIM PDM 奖励（NC × DAC × (5·TTC + 5·EP + 2·C) / 12）
```

### 5.2 GRPO 训练流程

```python
# simwam/runtime_grpo.py
def run_grpo_training(cfg):
    # 1. 加载 IL 预训练模型
    model = create_simwam_grpo(cfg.model)
    model.load_checkpoint(cfg.model.checkpoint_path)  # 加载 SFT 权重
    
    # 2. 配置 LoRA
    model.configure_grpo(cfg.grpo)  # 冻结 Video DiT，只给 Action DiT 加 LoRA
    
    # 3. 构建数据集和奖励
    train_ds = instantiate(cfg.data.train)
    reward = NavSimPDMReward(**cfg.grpo.reward)  # NAVSIM PDM 评分器
    
    # 4. 构建 GRPO 训练器
    trainer = SimWAMGRPOTrainer(model=model, train_dataset=train_ds, 
                                 reward=reward, cfg=cfg)
    trainer.train()
```

### 5.3 GRPO 训练器核心逻辑

**源文件**：`src/simwam/trainer_grpo.py`

```python
class SimWAMGRPOTrainer:
    def train(self):
        for step in range(self.max_steps):
            # ===== Phase 1: 采样（Rollout）=====
            # 从当前策略采样 G=8 条轨迹
            rollout_batch = self.sample_rollouts()
            
            # 计算每条轨迹的奖励
            rewards = self.reward(rollout_batch)  # NAVSIM PDM 分数
            
            # 计算组内归一化 advantage
            advantages = self.compute_advantages(rewards)  
            # advantage_i = (r_i - mean) / (std + eps)
            
            # ===== Phase 2: 训练（Update）=====
            # 复用每批 rollout 做 num_inner_epochs 次更新
            for inner_epoch in range(self.num_inner_epochs):
                self.update_policy(rollout_batch, advantages)
```

### 5.4 SDE 采样（FlowGRPO 的核心）

```python
def sample_rollouts(self):
    """
    FlowGRPO 的 SDE 采样：在确定性 ODE 上加随机扰动，
    让模型探索不同的轨迹方向。
    
    关键：只在最后 3 步加扰动（k ∈ {7, 8, 9}），
    因为靠近输出端的扰动对最终轨迹影响最直接。
    """
    # 1. 初始化噪声轨迹
    traj = torch.randn(B, 50, 3)
    
    # 2. 10 步欧拉积分，最后 3 步加 SDE 扰动
    for k in range(10):
        t = k / 10
        
        # Action DiT 预测速度场
        v_pred = self.model.action_dit(traj, t, ...)
        
        # ODE 步
        traj = traj + v_pred * (1.0 / 10)
        
        # 如果是最后 3 步（k ∈ {7, 8, 9}），加 SDE 扰动
        if k in [7, 8, 9]:
            # 低频余弦基扰动（不是独立路点噪声）
            noise = self.compute_low_freq_noise(traj)  # 6 个余弦基
            score_correction = self.compute_score_correction(traj, t)
            traj = traj + sigma * noise + 0.5 * sigma**2 * score_correction
    
    return traj
```

### 5.5 PDM 奖励计算

```python
class NavSimPDMReward:
    def __call__(self, trajectories):
        """
        计算 NAVSIM PDM 分数。
        
        PDMS = NC × DAC × (5·TTC + 5·EP + 2·C) / 12
        
        NC: 无责碰撞（0/0.5/1）
        DAC: 可行驶区域（0/1）
        TTC: 碰撞时间（0/1）
        EP: 前进进度（0-1 连续）
        C: 舒适度（0/1）
        """
        rewards = []
        for traj in trajectories:
            # 把轨迹转换成 NAVSIM 格式
            navsim_traj = self.convert_to_navsim(traj)
            
            # 调用 NAVSIM 评测器
            pdms = self.pdm_scorer(navsim_traj)
            
            rewards.append(pdms)
        
        return torch.tensor(rewards)
```

### 5.6 LoRA 微调

```python
# simwam/models/wan22/lora.py
class LoRALinear(nn.Module):
    """
    低秩适配器：只训练 A 和 B 两个小矩阵，
    冻结原始权重 W。
    
    W' = W + α * B @ A
    """
    def __init__(self, in_features, out_features, r=16, alpha=32.0):
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.alpha = alpha
        self.scaling = alpha / r
    
    def forward(self, x):
        # 原始输出 + LoRA 增量
        return F.linear(x, self.weight) + self.scaling * self.lora_B(self.lora_A(x))
```

**GRPO 训练时的冻结策略**：
- Video DiT：**完全冻结**（不训练）
- Action DiT 的 self-attention：**加 LoRA**（只训练 LoRA 参数）
- Action DiT 的 cross-attention：**加 LoRA**
- Action DiT 的 FFN：**冻结**
- 输出头：**训练**（全参数）

---

## 📊 模块六：评测流程

**源文件**：`experiments/navsim/eval_navsim.py`

### 6.1 评测入口

```bash
# 评测命令
CKPT=./runs/grpo/.../checkpoints/weights/step_2000.pt \
TASK=navsim_grpo_action_pdm_384x672_flowgrpo_lora \
NPROC_PER_NODE=8 \
bash experiments/navsim/run_eval_navsim.sh
```

### 6.2 评测脚本核心逻辑

```python
# experiments/navsim/eval_navsim.py
def evaluate(checkpoint_path, task_config):
    # 1. 加载模型
    model = load_model(checkpoint_path)
    
    # 2. 加载 NAVSIM navtest 数据集
    test_dataset = NavSimDataset(split="navtest")
    
    # 3. 对每个场景生成轨迹
    for scene in test_dataset:
        traj = model.generate_trajectory(scene.images, scene.navigation)
        
        # 4. 用 NAVSIM PDM 评测器打分
        pdms = pdm_scorer(traj, scene)
        
        # 5. 保存结果
        results.append({
            "scene": scene.token,
            "trajectory": traj,
            "nc": pdms.nc,
            "dac": pdms.dac,
            "ep": pdms.ep,
            "ttc": pdms.ttc,
            "comfort": pdms.comfort,
            "pdms": pdms.score,
        })
    
    # 6. 汇总统计
    avg_pdms = np.mean([r["pdms"] for r in results])
    print(f"PDMS: {avg_pdms:.1f}")
```

---

## 🔬 个人解读与思考

### 1. "训练时借力、推理时独立"的设计哲学

SimWAM 最聪明的地方在于：**训练时让 Video Expert 和 Action Expert 联合学习，但推理时只用 Action Expert**。这解决了 World Model 路线的核心痛点——推理时生成未来帧太慢。

具体实现靠 Isolated Attention Mask：
- 训练时：Video tokens 和 Action tokens 拼在一起做联合注意力，但掩码阻止 Action 看到未来帧
- 推理时：直接扔掉 Video DiT，只跑 Action DiT

这个设计的好处：
1. **训练效率**：Video Expert 的运动先验通过共享注意力流入 Action Expert
2. **推理效率**：不需要生成未来帧，大幅减少计算量
3. **灵活性**：Video Expert 可以替换成任何视频生成模型，Action Expert 独立可扩展

### 2. 低频余弦基扰动的精巧设计

FlowGRPO 在 RL 训练时不加独立路点噪声（会产生高频抖动），而是限制在 6 个余弦基上——相当于只在"整体偏左/偏右""整体加速/减速"这几个低维模态上探索。

这和人类驾驶的直觉一致：你不会突然在第 3 个路点拐一下，而是整体调整策略。论文原文：

> *"Independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity."*

### 3. LoRA 微调的工程选择

GRPO 训练时只给 Action DiT 的注意力层加 LoRA（rank=16），冻结 FFN 和整个 Video DiT。这样做的好处：
1. **显存效率**：只训练 ~5% 的参数
2. **稳定性**：LoRA 不会破坏预训练权重
3. **可恢复性**：如果 RL 训练崩了，可以回退到 SFT 权重

### 4. 和其他 WAM 的对比

| | DriveWAM | DriveLaW | SimWAM |
|---|---|---|---|
| 视频 backbone | 自训 | 自训 | **Wan2.2-5B（预训练）** |
| 推理时需要生成未来帧？ | 是 | 是 | **否** |
| 推理延迟 | 高 | 高 | **低** |
| PDMS | 90.1 | 89.1 | **91.5** |
| RL 后训练 | 无 | 无 | **FlowGRPO** |

SimWAM 的核心优势：**用预训练视频模型的运动先训，但推理时不需要生成未来帧**——既借了视频模型的力，又不背它的包袱。

### 5. 闭环仍是短板

和 Qwen-Drive-1.0 一样，SimWAM 在 AlpaSim 闭环上（0.30 at-fault score）还不如 Alpamayo-R1（0.58）。可能原因：
- 非反应式仿真训练的轨迹在闭环交互场景下缺乏"反应式"调整
- PDM 奖励主要基于开环指标，对闭环交互的覆盖不足
- 未来可能需要引入闭环仿真训练

---

## 📝 一句话总结

**SimWAM = Wan2.2-5B Video DiT（冻结）+ ActionDiT（可训）+ Isolated Attention Mask（训练时隔离、推理时独立）+ 联合 Flow Matching（SFT）+ FlowGRPO（RL 后训练，LoRA 微调）。训练时借视频模型的运动先验，推理时只用动作专家直接输出轨迹——NAVSIM 91.5 PDMS，推理延迟远低于其他 WAM。**
