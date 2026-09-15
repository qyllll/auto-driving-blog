---
title: "SimWAM 代码完整解读：从 Video DiT 到 Action DiT，从 SFT 到 FlowGRPO 的逐行拆解"
date: 2026-09-06
draft: false
categories: ["个人思考"]
summary: "从 configs/model/simwam_navsim.yaml 第 1 行开始，逐层追踪 SimWAM 的完整逻辑链：两个专家怎么协作？Isolated Attention Mask 怎么实现训练时借力、推理时独立？联合 Flow Matching 训练的损失怎么算？FlowGRPO 的 SDE 采样、PDM 奖励、LoRA 微调怎么串起来？每段代码都标注了源文件路径。"
tags: ["SimWAM", "World Action Model", "Flow Matching", "GRPO", "Wan2.2", "自动驾驶", "端到端"]
math: true
weight: 98
---

## 入门篇：SimWAM 到底是什么？（小白版）

> 如果你第一次听说「世界动作模型」「Video DiT」「Action DiT」「FlowGRPO」这些词，或者听到就头大——**先读这一篇**。
> 这一篇不讲任何代码，只帮你建立**直觉**。等脑子里有画面了，再进入下面的代码逐行拆解。

### 0.1 一句话版

> **SimWAM = 一个冻结的视频生成模型（Wan2.2-5B）+ 一个轻量动作专家（ActionDiT），用联合 Flow Matching 训练，让视频预测的「运动先验」流入轨迹生成——但推理时不需要真的生成未来帧，只用动作专家直接输出轨迹。**

论文：*SimWAM: A Simple World Action Model for End-to-End Autonomous Driving*（arXiv:2608.07468，华中科技大学 × 东风）。

它和传统做法的本质区别：

| 传统端到端（UniAD / VAD） | World Model（DriveWAM / DriveLaW） | SimWAM |
|---|---|---|
| 直接从观测映射到轨迹 | 先想象未来帧，再规划 | **训练时**用视频预测传递运动先验，**推理时**不需要生成未来帧 |
| 无显式时序建模 | 显式预测未来（慢） | 隐式传递时序信息（快） |
| 轨迹质量依赖模仿学习 | 轨迹质量依赖未来帧质量 | 轨迹质量 = 运动先验 + RL 后训练 |

一句话概括动机：**视频生成模型里藏着丰富的「物体怎么运动」的先验——SimWAM 把这个先验「蒸馏」到一个轻量动作专家里，让它直接输出好轨迹，而不需要在推理时真的花时间生成未来帧。**

### 0.2 先看思维导图

![SimWAM 思维导图](/images/simwam/simwam_mindmap.svg)

图里有 5 个分支：**是什么、为什么要 WAM、两个专家、训练方法、推理优势**。先让它们在脑子里占个位置，下面逐个展开。

### 0.3 两个必须搞懂的核心概念

把名字拆开就是它的两个引擎：

| 名字的一部分 | 对应概念 | 负责的事 |
|---|---|---|
| **Sim** | Simple（简洁） | 不搞复杂模块，用现成视频模型 |
| **WAM** | World Action Model | 视频世界模型 + 动作预测联合训练 |

两个专家：

| 专家 | 身份 | 参数量 | 训练时 | 推理时 |
|---|---|---|---|---|
| **Video Expert** | Wan2.2-TI2V-5B DiT（冻结/微调） | ~5B | 联合训练，预测未来帧 | **不用** |
| **Action Expert** | ActionDiT（从头训/LoRA） | ~0.2B-1B | 联合训练，生成轨迹 | **只用这个** |

#### 概念一：从「噪声」到「轨迹」——Flow Matching

和所有扩散模型一样，SimWAM 的 ActionDiT 也是「从噪声生成轨迹」：

> **生成轨迹 = 从一个纯噪声出发，一步步把它「整形」成一条干净的轨迹。**

**扩散模型（Diffusion）**的做法是"一点点擦马赛克"：每一步去掉一点噪声，走 20~50 步。

**Flow Matching**是扩散的"直连版"：不弯弯绕绕，而是在「纯噪声」和「轨迹」之间**拉一条直线**，训练模型学习这条线上的**速度场（velocity field）**——告诉它"在任意一个中间点，下一步该往轨迹方向走多少"。步数可多可少（大步走也行），所以更快、更适合反复采样的强化学习。

![Flow Matching 概念](/images/simwam/simwam_flow_matching.svg)

> 小知识：SimWAM 的 Video DiT（Wan2.2）用的也是 Flow Matching 家族（rectified-flow），所以两个专家天然兼容。

#### 概念二：Isolated Attention Mask——「训练时借力、推理时独立」

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

![Isolated Attention Mask](/images/simwam/simwam_isolated_mask.svg)

> 为什么"相对比较"这么重要？因为视频模型的「运动先验」通过共享注意力流入动作专家——但动作专家不需要知道"未来帧长什么样"，只需要知道"当前场景里物体在怎么运动"。Isolated Mask 恰好保留了有用信息、屏蔽了无用信息。

### 0.4 概念关系图：五个名词怎么拼起来

把上面串起来：

- **蓝色（视频专家）**：Wan2.2-5B DiT → 预测未来帧 → 提供场景理解和运动先觉
- **红色（动作专家）**：ActionDiT → 从噪声生成轨迹 → 推理时独立工作
- **紫色（联合训练）**：MOT + Isolated Mask → 训练时共享注意力，推理时独立
- **绿色（优化算法）**：SFT（模仿学习）→ FlowGRPO（RL 后训练）→ PDM 奖励

三者合起来就是一个完整的 **SimWAM** 训练流程。

### 0.5 和自动驾驶 / 世界模型有什么关系？

你在这个博客里读过的很多 VLA（视觉语言行动模型）工作，底层就是这套组合拳的变体：

- **UniAD / VAD**：直接从观测映射到轨迹，无显式时序建模
- **DriveWAM / DriveLaW**：先生成未来帧，再规划——推理时必须生成未来帧（慢）
- **SimWAM**：训练时借视频模型的运动先觉，推理时只用动作专家——**既借了力，又不背包袱**

所以这篇讲的 SimWAM 不是"只跟视频有关"的孤立技巧：**「视频模型 + 动作专家 + Isolated Mask + RL 后训练」这套配方，就是 2025-2026 年端到端驾驶领域最有竞争力的范式之一。** 读懂了它，等于读懂了那一批工作的共同骨架。

### 0.6 接下来怎么读？

后面内容沿着**一条执行主线**走：

```
初始化 → SFT 联合训练 → FlowGRPO RL 后训练 → 推理 → 评测
```

建议带着入门篇建立的三个"心智锚点"去读：

1. **Video Expert = Wan2.2-5B DiT**（提供运动先觉）；
2. **Action Expert = ActionDiT**（生成轨迹，推理时独立）；
3. **Isolated Mask = 训练时借力、推理时独立的开关**。

---

## 先回答：SimWAM 代码仓库结构是什么？

### 这是一份可运行的代码仓库

SimWAM **不是一篇只能读的论文，而是一份完整的开源代码仓库**，代码就在 `src/simwam/` 目录下：

![SimWAM 代码仓库结构](/images/simwam/simwam_code_structure.svg)

### 三个入口脚本怎么分工？

读代码前先分清三个入口文件：

| 文件 | 作用 | 对应配置 |
|------|------|---------|
| `scripts/train.py` | SFT 训练（模仿学习） | `configs/task/navsim_uncond_*.yaml` |
| `scripts/train_grpo.py` | FlowGRPO 训练（RL 后训练） | `configs/task/navsim_grpo_*.yaml` |
| `experiments/navsim/eval_navsim.py` | 评测（NAVSIM navtest） | `experiments/navsim/run_eval_*.sh` |

### 两个核心 Python 文件

| 文件 | 行数 | 干什么 |
|------|------|--------|
| `src/simwam/runtime.py` | ~200 行 | SFT 训练的运行时入口：构建模型、数据集、训练器 |
| `src/simwam/runtime_grpo.py` | ~200 行 | GRPO 训练的运行时入口：加载 IL checkpoint、配置 LoRA、构建 PDM 奖励 |

而**真正的核心逻辑**在这两个文件里：

| 文件 | 大小 | 干什么 |
|------|------|--------|
| `src/simwam/models/wan22/simwam.py` | 53KB | SimWAM 主模型：前向传播、损失计算、推理 |
| `src/simwam/trainer_grpo.py` | 41KB | GRPO 训练器：SDE 采样、PDM 奖励、PPO 更新 |

而**模型架构**在这几个文件里：

| 文件 | 大小 | 干什么 |
|------|------|--------|
| `src/simwam/models/wan22/action_dit.py` | 13KB | ActionDiT：动作专家架构 |
| `src/simwam/models/wan22/mot.py` | 24KB | MOT：Mixture-of-Transformers 联合注意力 |
| `src/simwam/models/wan22/wan_video_dit.py` | 31KB | Wan2.2 Video DiT 封装 |
| `src/simwam/models/wan22/lora.py` | 5KB | LoRA 适配器实现 |

### 本文的讲解方式

**不从概念讲起，而是从配置文件的入口出发，沿着执行路径一步一步深入。**

我们追踪一辆"数据快车"：

```
configs → runtime_grpo.py → simwam_grpo.py → action_dit.py → mot.py → trainer_grpo.py → 回到采样
```

每到一个关键节点，我都会：
1. 标注代码位置（文件名 + 行号）
2. 说明"这个函数接收什么、输出什么"
3. 解释背后的数学/算法动机
4. 配逻辑分支图

---

## 第 0 步：先理解总图

![SimWAM 架构总览](/images/simwam/simwam_architecture.svg)

**整个框架只有两个大阶段，反复交替：**

| 阶段 | 发生位置 | 是否更新参数 | 产出 |
|------|---------|------------|------|
| **SFT 训练** (模仿学习) | `runtime.py` → `trainer_tensorboard.py` | ✅ 更新全部参数 | IL checkpoint |
| **FlowGRPO 训练** (RL 后训练) | `runtime_grpo.py` → `trainer_grpo.py` | ✅ 只更新 LoRA 参数 | RL checkpoint |

这是**两阶段训练**的标志：先 SFT 学会基本能力，再 FlowGRPO 用奖励信号微调改进。

---

## 第 1 步：配置文件——一切的起点

### 1.1 模型配置 (`configs/model/simwam_navsim.yaml`)

```yaml
# configs/model/simwam_navsim.yaml
# 这个文件定义了 SimWAM 的所有模型参数

# === Video Expert ===
model_id: "Wan-AI/Wan2.2-T2V-14B"     # 预训练视频模型
tokenizer_model_id: "Wan-AI/Wan2.2-T2V-14B"  # T5 文本编码器
tokenizer_max_len: 512                  # 文本最大长度
load_text_encoder: false                # 训练时不加载文本编码器（冻结）

# === Video DiT 配置 ===
video_dit_config:
  hidden_dim: 3072        # Video DiT 隐藏维度
  ffn_dim: 12288          # FFN 中间维度
  num_heads: 24           # 注意力头数
  attn_head_dim: 128      # 每头维度（3072/24=128）
  num_layers: 40          # Video DiT 层数
  freq_dim: 256           # 时间步 Fourier 编码维度

# === Action Expert 配置 ===
action_dit_config:
  action_dim: 3           # (x, y, heading) 三维度
  hidden_dim: 1024        # ActionDiT 隐藏维度
  ffn_dim: 4096           # FFN 中间维度
  num_heads: 24           # 注意力头数
  attn_head_dim: 128      # 每头维度
  num_layers: 30          # ActionDiT 层数（比 Video 少）
  text_dim: 4096          # 文本条件维度（与 T5 对齐）
  freq_dim: 256           # 时间步 Fourier 编码维度

# === 调度器配置 ===
video_scheduler:
  train_shift: 5.0        # 训练时的噪声偏移
  infer_shift: 5.0        # 推理时的噪声偏移
  num_train_timesteps: 1000  # 训练时间步数

action_scheduler:
  train_shift: 1.0        # ActionDiT 的训练噪声偏移
  infer_shift: 1.0        # ActionDiT 的推理噪声偏移
  num_train_timesteps: 10  # 10 步欧拉积分
  prediction_type: "velocity"  # 预测速度场
  sigma_clamp_min: 0.1    # SDE 噪声下限
```

**关键理解：** Video DiT 有 40 层、3072 维（~5B 参数），ActionDiT 只有 30 层、1024 维（~0.2B 参数）。ActionDiT 比 Video DiT 轻得多——这就是"轻量动作专家"的含义。

### 1.2 SFT 任务配置 (`configs/task/navsim_uncond_front_384x672_1e-4.yaml`)

```yaml
# configs/task/navsim_uncond_front_384x672_1e-4.yaml
# SFT 训练的超参数

batch_size: 2              # 每 GPU 批大小
learning_rate: 1e-4        # 学习率
num_epochs: 100            # 训练轮数
lr_scheduler_type: "cosine"  # 余弦退火
weight_decay: 1e-2         # 权重衰减
gradient_accumulation_steps: 1  # 梯度累积

model:
  proprio_dim: 4           # 自车状态维度（速度、加速度等）
  loss:
    lambda_video: 1.0      # 视频损失权重
    lambda_action: 1.0     # 动作损失权重（等权重）
```

### 1.3 GRPO 任务配置 (`configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml`)

```yaml
# configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml
# FlowGRPO 训练的超参数

model:
  checkpoint_path: "path/to/sft/checkpoint.pt"  # SFT 权重

grpo:
  # === LoRA 配置 ===
  lora:
    enabled: true          # 只训练 LoRA 适配器
    r: 16                  # LoRA rank
    alpha: 32.0            # LoRA alpha（缩放系数 = alpha/r = 2）
    target_modules:        # 只在注意力层加 LoRA
      - q
      - k
      - v
      - o
  
  # === 采样配置 ===
  sample:
    sde_mode: rigorous     # 严格的 SDE 采样（score-corrected）
    noise_level: 0.1       # SDE 噪声级别
    num_samples: 8         # 每个场景采样 G=8 条轨迹
  
  # === 训练配置 ===
  train:
    ppo_clip_range: 0.02   # PPO clip 范围 ε
    adv_clip_max: 5.0      # advantage 截断到 ±5
    num_inner_epochs: 4    # 每批 rollout 复用 4 次
    learning_rate: 5e-6    # RL 学习率（比 SFT 小 20 倍）
  
  # === 奖励配置 ===
  reward:
    # NAVSIM PDM 奖励
    # PDMS = NC × DAC × (5·TTC + 5·EP + 2·C) / 12
```

**关键理解：** FlowGRPO 的学习率（5e-6）比 SFT（1e-4）小 20 倍——RL 微调必须小心，步子太大会"忘记"SFT 学到的能力。

---

## 第 2 步：入口脚本——从 train_grpo.py 开始

### 2.1 入口 (`scripts/train_grpo.py`)

```python
# scripts/train_grpo.py（整个框架的入口）
import hydra
from omegaconf import DictConfig
from simwam.runtime_grpo import run_grpo_training

@hydra.main(config_path="../configs", config_name="train_grpo", version_base="1.3")
def main(cfg: DictConfig):
    run_grpo_training(cfg)

if __name__ == "__main__":
    main()
```

**一句话：** `train_grpo.py` 只是一个启动器，真正的逻辑在 `runtime_grpo.py` 的 `run_grpo_training` 里。

### 2.2 runtime_grpo.py 的完整逻辑 (`src/simwam/runtime_grpo.py`)

```python
# src/simwam/runtime_grpo.py
def run_grpo_training(cfg: DictConfig):
    # 1. 日志和输出目录
    misc.register_work_dir(cfg.output_dir)
    setup_logging(...)
    
    # 2. 设备和精度
    device = _resolve_train_device()
    model_dtype = _mixed_precision_to_model_dtype(cfg.mixed_precision)
    
    # 3. 构建模型（通过 Hydra instantiate）
    checkpoint_path = cfg.model.get("checkpoint_path", None)
    model_container = OmegaConf.to_container(cfg.model, resolve=True)
    model_container.pop("checkpoint_path", None)
    model = instantiate(model_container, model_dtype=model_dtype, device=device)
    
    # 4. 加载 SFT 权重（warm-start）
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        model.load_checkpoint(str(ckpt), optimizer=None)
    
    # 5. 配置 GRPO（注入 LoRA、配置采样器）
    model.configure_grpo(cfg.grpo)
    
    # 6. 构建数据集和奖励
    train_ds = instantiate(cfg.data.train)
    reward = instantiate_reward(cfg.grpo)  # NavSimPDMReward
    
    # 7. 构建 GRPO 训练器并开始训练
    trainer = SimWAMGRPOTrainer(model=model, train_dataset=train_ds, reward=reward, cfg=cfg)
    trainer.train()
```

**对应流程图：**

```
train_grpo.py → run_grpo_training
  ├─ instantiate(model) → SimWAMGRPO.from_wan22_pretrained(...)
  ├─ load_checkpoint(sft_weights) → 加载 SFT 权重
  ├─ configure_grpo(cfg.grpo) → 注入 LoRA、配置采样器
  ├─ instantiate(data) → NAVSIM 数据集
  ├─ instantiate_reward → NavSimPDMReward
  └─ trainer.train() → 开始 GRPO 循环
```

### 2.3 create_simwam_grpo 工厂函数

```python
# src/simwam/runtime_grpo.py:24-87
def create_simwam_grpo(
    model_id, tokenizer_model_id, video_dit_config,
    tokenizer_max_len, load_text_encoder, proprio_dim,
    action_dit_config, action_dit_pretrained_path,
    skip_dit_load_from_pretrain, video_scheduler,
    action_scheduler, loss, mot_checkpoint_mixed_attn,
    redirect_common_files, model_dtype, device,
):
    """构建 SimWAMGRPO（和 runtime.create_simwam 参数一样）。"""
    from .models.wan22.simwam_grpo import SimWAMGRPO
    
    # 把 DictConfig 转成 dict
    video_dit_config = _as_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = _as_dict(action_dit_config, "action_dit_config")
    action_scheduler = _as_dict(action_scheduler, "action_scheduler", required=True)
    
    # 校验 action_scheduler 必需的 key
    required_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_keys - set(action_scheduler.keys())
    if missing:
        raise ValueError(f"`action_scheduler` missing keys: {sorted(missing)}.")
    
    # 调用 SimWAMGRPO.from_wan22_pretrained
    return SimWAMGRPO.from_wan22_pretrained(
        device=device, torch_dtype=model_dtype,
        model_id=model_id, tokenizer_model_id=tokenizer_model_id,
        # ... 其余参数
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
```

**关键理解：** 这个工厂函数负责把配置文件里的参数"翻译"成模型对象。核心调用是 `SimWAMGRPO.from_wan22_pretrained`——它会加载 Wan2.2 预训练权重、初始化 ActionDiT、构建 MOT 联合注意力层。

---

## 第 3 步：ActionDiT——动作专家架构

**源文件**：`src/simwam/models/wan22/action_dit.py`（13KB）

### 3.1 核心类结构

```python
# src/simwam/models/wan22/action_dit.py
class ActionDiT(nn.Module):
    """
    轻量 Diffusion Transformer，从噪声轨迹生成干净轨迹。
    条件输入：VLM 的缓存 KV + 时间步 + 导航指令 + 自车状态。
    """
    def __init__(self, action_dim=3, hidden_dim=1024, ffn_dim=4096,
                 num_heads=24, attn_head_dim=128, num_layers=30,
                 text_dim=4096, freq_dim=256, eps=1e-6):
        super().__init__()
        
        # 1. 轨迹嵌入：把 (x, y, heading) 的 noisy 轨迹 + Fourier 特征 → hidden_dim
        # 输入维度 = action_dim × 2 + freq_dim = 3×2 + 256 = 262
        self.traj_embed = nn.Sequential(
            nn.Linear(action_dim * 2 + freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 2. 时间步嵌入：Flow time → Fourier → MLP → hidden_dim
        self.time_embed = TimestepEmbedding(freq_dim, hidden_dim)
        
        # 3. 文本嵌入：导航指令（T5 输出）→ project → hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 4. 自车状态嵌入：当前速度/加速度等 → project → hidden_dim
        self.ego_proj = nn.Linear(proprio_dim, hidden_dim)
        
        # 5. 30 层 Transformer Block
        self.blocks = nn.ModuleList([
            ActionDiTBlock(hidden_dim, ffn_dim, num_heads, attn_head_dim, ...)
            for _ in range(num_layers)
        ])
        
        # 6. 输出头：hidden_dim → (action_dim × 2)  # 预测速度场
        self.out_proj = nn.Linear(hidden_dim, action_dim * 2)
```

**参数量计算：** 30 层 × (self-attention + cross-attention + FFN) ≈ **~0.2B-1B**（取决于是否用 LoRA）。

### 3.2 ActionDiTBlock 的内部结构

```python
# src/simwam/models/wan22/action_dit.py
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

**关键设计：** Cross-Attention 的 `kv_cache` 来自 VLM 的 softmax attention 层——这就是"世界模型信息流入动作专家"的管道。动作 token 通过 cross-attention 读取 VLM 编码的场景表征，但**看不到未来帧 token**（因为 isolated attention mask）。

### 3.3 轨迹嵌入的数学

ActionDiT 的输入不是原始轨迹，而是经过 Fourier 编码的轨迹：

```python
# 轨迹嵌入的输入维度 = action_dim × 2 + freq_dim = 3×2 + 256 = 262
# 其中：
#   - action_dim = 3: (x, y, heading)
#   - × 2: 原始值 + Fourier 编码
#   - freq_dim = 256: Fourier 特征维度

# Fourier 编码的数学：
# pos_enc(x) = [sin(2π·f₁·x), cos(2π·f₁·x), ..., sin(2π·f₂₅₆·x), cos(2π·f₂₅₆·x)]
# 其中 fᵢ = 1/10000^(i/128) 是频率
```

**为什么需要 Fourier 编码？** 因为 (x, y, heading) 是低维连续值，直接输入 Transformer 表达能力不够。Fourier 编码把低维映射到高维空间，让模型更容易学习非线性关系。

---

## 第 4 步：MOT——联合注意力层

**源文件**：`src/simwam/models/wan22/mot.py`（24KB）

### 4.1 核心思想

MOT（Mixture-of-Transformers）是 SimWAM 的"胶水层"——它让 Video DiT 和 Action DiT 共享注意力接口，但通过掩码隔离信息流：

```
传统 Mixture-of-Transformers：
  Video tokens 和 Action tokens 混在一起做联合注意力
  → 信息自由流动，Action 会看到未来帧

SimWAM 的 Isolated MOT：
  Video tokens 和 Action tokens 仍然拼在一起做注意力
  → 但用掩码阻止 Action tokens 看到未来帧 tokens
  → 同时允许两者都看到当前观测 tokens
```

### 4.2 掩码矩阵的构建

```python
# src/simwam/models/wan22/mot.py
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

### 4.3 推理时的关键优化

```python
# 推理时：直接扔掉 Video DiT 部分，只跑 Action DiT
# 因为动作 tokens 从来不依赖未来帧 tokens
# 所以不需要生成未来帧，大幅减少计算量

# 训练时：两个 DiT 都跑，联合优化
# 推理时：只跑 Action DiT，直接输出轨迹
# → 这就是"训练时借力、推理时独立"
```

---

## 第 5 步：SimWAM 主模型——联合前向传播

**源文件**：`src/simwam/models/wan22/simwam.py`（53KB）

### 5.1 模型初始化流程

```python
# src/simwam/models/wan22/simwam.py
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

### 5.2 训练时的前向传播

```python
# src/simwam/models/wan22/simwam.py
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

**对应公式：**

$$\mathcal{L} = \lambda_{\text{video}} \cdot \text{MSE}(v_{\text{video}}, v_{\text{target}}) + \lambda_{\text{action}} \cdot \text{MSE}(v_{\text{action}}, v_{\text{target}})$$

其中 $v_{\text{target}} = x_1 - x_0$（正确速度），默认 $\lambda_{\text{video}} = \lambda_{\text{action}} = 1.0$（等权重）。

### 5.3 推理时的轨迹生成

```python
# src/simwam/models/wan22/simwam.py
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

**注意：** 推理时**完全没有 Video DiT**——因为动作 token 从来不依赖未来帧 token，所以不需要生成未来帧。这就是 SimWAM 推理速度快的原因。

---

## 第 6 步：FlowGRPO 训练循环

![FlowGRPO 训练循环](/images/simwam/simwam_grpo_loop.svg)

**源文件**：`src/simwam/trainer_grpo.py`（41KB）

### 6.1 训练器核心逻辑

```python
# src/simwam/trainer_grpo.py
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

### 6.2 SDE 采样——FlowGRPO 的核心

这一节是整个 FlowGRPO 最关键的部分——**为什么确定性 ODE 需要变成随机 SDE？怎么变？变完之后 log_prob 怎么算？** 每一步都拆开讲。

#### 6.2.1 为什么需要 SDE？——从"确定性"到"可微概率"

Flow Matching 推理时用 ODE（常微分方程）：

$$x_{k+1} = x_k + v_\theta(x_k, t) \cdot \Delta t$$

这个过程是**完全确定性的**——给定同一个初始噪声 $x_0$，每次都生成同一条轨迹。没有随机性。

但 GRPO 需要算 PPO 的 ratio $r_i = \pi_{\text{new}} / \pi_{\text{old}}$——这要求策略 $\pi$ 是一个**概率分布**，能输出"生成这条轨迹的概率"。ODE 没有概率（概率 = 1 在确定性路径上，其他路径概率 = 0），所以**无法算 ratio**。

解法：**在 ODE 上加噪声，变成 SDE**。SDE 有随机性，同一条轨迹的概率 $p_\theta(x_{k+1} | x_k)$ 就不再是 0 或 1，而是一个连续的高斯概率——可以求导、可以算 log、可以算 ratio。

#### 6.2.2 ODE→SDE 的三步数学推导

**第一步：SDE 的漂移项是什么？**

一般形式的 SDE：

$$dx = f_{\text{SDE}}(x, t)\,dt + g(t)\,dW$$

其中 $f_{\text{SDE}}$ 是漂移项（决定"往哪走"），$g(t)\,dW$ 是扩散项（加噪声）。

我们要让 SDE 和原始 ODE 产生**相同的边际分布**（即 $p_t(x)$ 不变）。通过 Fokker-Planck 方程的逆推，可以得到漂移项：

$$f_{\text{SDE}} = v_\theta - \frac{\sigma_{\text{noise}}^2}{2}\nabla\log p_t(x)$$

其中 $\nabla\log p_t(x)$ 是**score function**——"概率密度在 $x$ 处的梯度方向"。

**第二步：Score function 的闭合形式（Rectified Flow 特有）**

对于 Rectified Flow 的插值 $x_t = (1-t)x_0 + tx_1$，score function 有解析解：

$$\nabla\log p_t(x) = -\frac{x}{t} - \frac{1-t}{t}\,v_\theta(x, t)$$

直觉：score 告诉你"从当前点 $x$ 出发，往哪个方向走能到达更高概率的区域"。它等于**当前位置的反方向**（把 $x$ 拉向原点）加上**速度场的修正**。

**第三步：Euler-Maruyama 离散化**

把 SDE 写成离散步：

$$x_{k+1} = x_k + \underbrace{\left[v_\theta + \frac{\sigma_{\text{noise}}^2}{2t}\big(x_k + (1-t)v_\theta\big)\right]}_{\text{mean}_\theta}\Delta t + \underbrace{\sigma_{\text{noise}}\sqrt{-\Delta t}\,\epsilon}_{\text{随机噪声}}$$

其中 $\epsilon \sim \mathcal{N}(0, I)$。拆解 mean 项：

$$\text{mean}_\theta = v_\theta + \frac{\sigma_{\text{noise}}^2}{2t}\big(x_k + (1-t)v_\theta\big)$$

- 第一项 $v_\theta$：原始 ODE 的速度场（"模型认为该往哪走"）
- 第二项 $\frac{\sigma^2}{2t}(x_k + (1-t)v_\theta)$：score correction（"概率梯度的修正"，把轨迹拉向高概率区域）

#### 6.2.3 为什么需要 Score Correction？——直觉解释

想象一下：你用 ODE 从噪声走到轨迹，中间路径是确定的。现在加了噪声（SDE），路径会偏离原来的 ODE 路径。问题是：**偏离之后，轨迹可能跑到低概率区域**（模型从没见过的奇怪形状）。

Score correction 就是一个"引力"——它指向概率密度更高的方向，把偏离的轨迹拉回来。没有它，SDE 的采样会发散，质量崩塌。

#### 6.2.4 Log_prob 的闭合形式——PPO ratio 的基础

SDE 的单步转移是一个高斯分布：

$$p_\theta(x_{k+1} \mid x_k) = \mathcal{N}(x_{k+1}; \text{mean}_\theta, \sigma^2 I)$$

所以 log_prob 有解析解：

$$\log p_\theta(x_{k+1} \mid x_k) = -\frac{\|x_{k+1} - \text{new}_\theta\|^2}{2\sigma^2} - \log(\sigma\sqrt{2\pi})$$

PPO 的 ratio 就是：

$$r_i = \exp\big(\log p_{\text{new}}(x_{k+1} \mid x_k) - \log p_{\text{old}}(x_{k+1} \mid x_k)\big)$$

因为新旧策略的 mean 不同（参数 $\theta$ 不同），但采样的 $x_{k+1}$ 相同（用 $\theta_{\text{old}}$ 采的），所以 ratio 衡量的是"新策略下这条转移的概率是变高了还是变低了"。

#### 6.2.5 代码逐行拆解

```python
# src/simwam/trainer_grpo.py
def sample_rollouts(self):
    """完整 SDE 采样流程（带详细注释）。"""
    
    # ========== 第一步：初始化噪声 ==========
    # 从标准正态分布采样初始轨迹（50个路点 × 3维：x, y, heading）
    # 这就是 Flow Matching 的起点 x_0 ~ N(0, I)
    traj = torch.randn(B, 50, 3)  # (batch=8, waypoints=50, dim=3)
    
    # ========== 第二步：10步欧拉积分 ==========
    for k in range(10):
        t = k / 10  # 时间步 t ∈ {0.0, 0.1, ..., 0.9}
        
        # --- 纯 ODE 步 ---
        # Action DiT 预测当前时刻的速度场 v_θ(x_k, t, c)
        # 输入：带噪轨迹 + 时间步 + VLM 条件（cross-attention KV）
        v_pred = self.model.action_dit(traj, t, **conditions)  # (B, 50, 3)
        
        # 确定性 ODE 步：x_{k+1} = x_k + v_θ · Δt
        traj_ode = traj + v_pred * (1.0 / 10)  # Δt = 1/10 = 0.1
        
        # --- SDE 步（仅 k ∈ {7, 8, 9}）---
        if k in [7, 8, 9]:
            # 计算噪声级别 σ（可能随时间步变化）
            sigma = self.compute_sigma(t)  # 通常 σ = noise_level = 0.1
            
            # 计算低频余弦基噪声（不是独立路点噪声！）
            noise = self.compute_low_freq_noise(traj)  # (B, 50, 3)
            
            # 计算 score correction（概率梯度修正）
            score_correction = self.compute_score_correction(traj, t, v_pred)
            # score_correction ≈ -[x_k + (1-t)·v_θ] / t
            
            # SDE 更新：ODE步 + 噪声 + score修正
            traj = traj_ode + sigma * noise + 0.5 * sigma**2 * score_correction
        else:
            # 非SDE步：直接用ODE结果
            traj = traj_ode
    
    return traj  # (B, 50, 3) → 8条采样轨迹
```

#### 6.2.6 低频余弦基噪声：为什么比独立噪声好？

独立路点噪声的问题：

```python
# ❌ 独立噪声：每个路点独立扰动
noise_independent = torch.randn(50, 3)  # 50个路点各自独立
# 结果：第3个路点突然左拐0.5m，第4个又右拐0.3m → 高频抖动
# 物理上不可执行：车辆的转向系统无法做这种突变
```

低频余弦基噪声：

```python
# ✅ 低频余弦基：用6个平滑基函数组合
def compute_low_freq_noise(traj):
    """用余弦基函数生成平滑噪声。"""
    # 路点索引 i = 0, 1, ..., 49
    i = torch.arange(50).float()
    
    # 6个基函数：cos(π·i/50 · k), k=1,2,...,6
    # 每个基函数都是全局平滑的
    bases = torch.stack([torch.cos(torch.pi * i / 50 * k) for k in range(1, 7)])
    # bases shape: (6, 50)
    
    # 随机权重
    weights = torch.randn(6)  # 6个权重
    
    # 组合：noise(i) = Σ w_k · cos(π·i·k/50)
    noise_1d = weights @ bases  # (50,)
    
    # 扩展到3维（x, y, heading 分别用不同权重）
    noise = torch.stack([noise_1d] * 3, dim=-1)  # (50, 3)
    
    return noise
```

直觉对比：

| | 独立噪声 | 余弦基噪声 |
|---|---|---|
| 形状 | 锯齿状（高频） | 正弦曲线（低频） |
| 物理含义 | "第3个路点突然拐" | "整条轨迹整体偏左/偏右" |
| 可执行性 | ❌ 车辆无法跟随 | ✅ 平滑转弯 |
| 探索模态 | 50×3=150 维 | **6 维**（6个基函数的权重） |

**为什么是 6 个基？** 论文实验发现 6 个基已经覆盖了驾驶中常见的操纵模态：
1. 整体偏左/偏右（平移）
2. 整体加速/减速（速度缩放）
3. 左转弯 / 右转弯（曲率变化）
4. S 形变道（组合曲率）
5. 近端急转（局部曲率）
6. 远端调整（全局微调）

#### 6.2.7 为什么只在最后 3 步加扰动？

Flow Matching 的 10 步积分有一个特性：**早期步的扰动会被后续步"吸收"**。

直觉理解：
- 第 0 步加噪声 → 第 1 步的速度场会"纠正"这个偏差 → 第 2 步继续纠正 → ... → 最终偏差被衰减到很小
- 第 9 步加噪声 → 直接影响最终输出，没有后续步来纠正

数学解释：每一步 ODE 积分相当于一个线性变换，扰动经过多次变换后会被雅可比矩阵的特征值衰减。对于 Flow Matching 的直线路径，这个衰减因子大约是 $(1-\Delta t)^{n}$，n 步后衰减到很小。

论文的消融实验验证了这一点：

| SDE 步数 | PDMS |
|------|------|
| 所有 10 步都加 | 89.2 |
| 只加前 5 步 | 89.5 |
| 只加后 5 步 | 90.8 |
| **只加后 3 步** | **91.5** |

最后 3 步效果最好——既保证了足够的探索，又避免了早期扰动被衰减的浪费。

### 6.3 PDM 奖励计算——每个子指标怎么打分

PDM 分数是 SimWAM FlowGRPO 的**唯一奖励信号**。这一节把每个子指标的计算方式拆开讲。

#### 6.3.1 总公式

$$\text{PDMS} = \text{NC} \times \text{DAC} \times \frac{5 \cdot \text{TTC} + 5 \cdot \text{EP} + 2 \cdot \text{C}}{12}$$

注意结构：**NC 和 DAC 是乘性门槛**（任何一个 = 0，总分 = 0），TTC/EP/C 是**加权平均**（分值在 0-1 之间）。

#### 6.3.2 NC（No-Collision）：无责碰撞

```python
def compute_nc(ego_traj, agent_trajs, agent_metadata):
    """
    NC = 0.0  如果自车是碰撞的主责方
    NC = 0.5  如果自车是碰撞的次责方
    NC = 1.0  如果没有碰撞，或碰撞完全由对方造成
    
    判断依据：碰撞点的位置 + 双方的速度方向
    """
    for agent in agent_trajs:
        # 检测 OBB（有向包围盒）重叠
        collision_point = detect_obb_overlap(ego_traj, agent)
        
        if collision_point is not None:
            # 判断责任：谁的运动方向指向碰撞点？
            ego_heading_to_collision = angle(ego_vel, collision_point - ego_pos)
            agent_heading_to_collision = angle(agent_vel, collision_point - agent_pos)
            
            if ego_heading_to_collision < agent_heading_to_collision:
                return 0.0  # 自车主责
            else:
                return 0.5  # 自车次责
    
    return 1.0  # 无碰撞
```

**为什么 NC 是乘性的？** 因为碰撞是最严重的安全问题——如果你撞了人，轨迹再平滑也没用。所以 NC=0 直接把总分清零。

#### 6.3.3 DAC（Drivable Area Compliance）：可行驶区域

```python
def compute_dac(ego_traj, road_boundaries):
    """
    DAC = 1.0  所有路点都在可行驶区域内
    DAC = 0.0  任何一个路点越界
    
    判断方法：每个路点 vs 道路边界的多边形包含测试
    """
    for waypoint in ego_traj:
        if not is_inside_drivable_area(waypoint, road_boundaries):
            return 0.0
    return 1.0
```

**为什么 DAC 也是乘性的？** 越界 = 可能撞护栏/行人/对向车道，同样是安全红线。

#### 6.3.4 TTC（Time-to-Collision）：碰撞时间

```python
def compute_ttc(ego_traj, agent_trajs):
    """
    TTC ∈ [0, 1]
    
    TTC = 1.0  TTC > 5s（非常安全）
    TTC = 0.0  TTC < 0.5s（即将碰撞）
    TTC = 中间值  线性插值
    
    计算方法：对每对(自车, 障碍物)，找最近的碰撞时间
    """
    min_ttc = float('inf')
    
    for agent in agent_trajs:
        # 简化模型：假设双方匀速运动，求最近距离时刻
        rel_pos = ego_pos - agent.pos
        rel_vel = ego_vel - agent.vel
        
        # 最近时刻 t* = -rel_pos · rel_vel / |rel_vel|²
        t_star = -dot(rel_pos, rel_vel) / (dot(rel_vel, rel_vel) + 1e-6)
        t_star = max(0, t_star)
        
        # 最近距离
        min_dist = norm(rel_pos + rel_vel * t_star)
        
        # 碰撞时间：最近距离 < 安全阈值时的最早时刻
        if min_dist < safety_threshold:
            ttc = t_star
            min_ttc = min(min_ttc, ttc)
    
    # 归一化到 [0, 1]
    if min_ttc < 0.5:
        return 0.0
    elif min_ttc > 5.0:
        return 1.0
    else:
        return (min_ttc - 0.5) / 4.5  # 线性插值
```

#### 6.3.5 EP（Ego Progress）：前进进度

```python
def compute_ep(ego_traj, max_progress=60.0):
    """
    EP ∈ [0, 1]
    
    EP = 自车在规划窗口内前进的距离 / 最大期望前进距离
    
    例：规划 5s，自车前进了 30m，最大期望 60m → EP = 0.5
    """
    # 计算自车在规划窗口内的纵向位移
    progress = ego_traj[-1].y - ego_traj[0].y  # y轴 = 前进方向
    
    # 归一化
    ep = min(progress / max_progress, 1.0)
    return max(ep, 0.0)
```

**EP 为什么是连续的？** 因为"前进多少"是程度问题——不是"要么走要么不走"。RL 需要这个连续信号来学习"怎么开得更快但不违规"。

#### 6.3.6 C（Comfort）：舒适度

```python
def compute_comfort(ego_traj, dt=0.1):
    """
    C = 1.0  横向加速度 < 0.5m/s² 且 纵向加速度 < 1.5m/s²
    C = 0.0  任何一个超阈值
    
    实际是二值的：舒适 or 不舒适
    """
    for i in range(1, len(ego_traj)):
        # 横向加速度（转向离心力）
        lat_acc = compute_lateral_acceleration(ego_traj, i, dt)
        # 纵向加速度（急加速/急刹车）
        lon_acc = compute_longitudinal_acceleration(ego_traj, i, dt)
        
        if abs(lat_acc) > 0.5 or abs(lon_acc) > 1.5:
            return 0.0
    return 1.0
```

#### 6.3.7 加权组合的直觉

| 子指标 | 权重 | 含义 |
|---|---|---|
| TTC | 5 | 碰撞时间越长越好（安全核心） |
| EP | 5 | 前进越多越好（效率核心） |
| C | 2 | 舒适度（体验核心） |
| 归一化因子 | 12 = 5+5+2 | 保证加权平均 ∈ [0, 1] |

权重设计的直觉：**安全（TTC）和效率（EP）同等重要**（各 5 分），舒适度是锦上添花（2 分）。NC 和 DAC 是"一票否决"——安全红线不能碰。

#### 6.3.8 完整打分流程

```python
# src/simwam/datasets/navsim/pdm_reward.py
class NavSimPDMReward:
    def __call__(self, trajectories):
        """对每条采样轨迹打 PDM 分数。"""
        rewards = []
        for traj in trajectories:
            # 1. 坐标转换：SimWAM 格式 → NAVSIM 格式
            navsim_traj = self.convert_to_navsim(traj)
            
            # 2. 逐项计算
            nc = self.compute_nc(navsim_traj)
            dac = self.compute_dac(navsim_traj)
            ttc = self.compute_ttc(navsim_traj)
            ep = self.compute_ep(navsim_traj)
            c = self.compute_comfort(navsim_traj)
            
            # 3. 加权组合
            pdms = nc * dac * (5*ttc + 5*ep + 2*c) / 12.0
            
            rewards.append({
                "pdms": pdms,
                "nc": nc, "dac": dac, "ttc": ttc, "ep": ep, "c": c
            })
        
        return torch.tensor([r["pdms"] for r in rewards])
```

### 6.4 Advantage 计算——从奖励到梯度方向

#### 6.4.1 为什么需要 Advantage？——直接用奖励有什么问题？

假设同一个场景采了 8 条轨迹，PDM 奖励分别是：

```
[0.85, 0.72, 0.91, 0.68, 0.83, 0.77, 0.88, 0.79]
```

如果直接用原始奖励作为 advantage：
- 好轨迹的 advantage 都是正的（~0.8）
- 差轨迹的 advantage 也是正的（~0.7）
- **没有"差"的信号！** 梯度会让所有轨迹的概率都增大。

需要 advantage 来做**组内归一化**：把"比平均好"变成正 advantage，"比平均差"变成负 advantage。

#### 6.4.2 数值走一遍

```python
# 8条轨迹的PDM奖励
rewards = torch.tensor([0.85, 0.72, 0.91, 0.68, 0.83, 0.77, 0.88, 0.79])

# 第一步：算均值和标准差
mean = rewards.mean()  # = 0.80375
std = rewards.std() + 1e-4  # = 0.08214

# 第二步：组内归一化
advantages = (rewards - mean) / std
# advantages = [ 0.56, -1.02,  1.29, -1.48,  0.32, -0.41,  0.93, -0.20]
```

| 轨迹 # | PDM 奖励 | Advantage | 含义 |
|------|------|------|------|
| #2 | 0.91 | **+1.29** | 比平均好很多 → 大幅提高概率 |
| #7 | 0.88 | **+0.93** | 比平均好 → 提高概率 |
| #1 | 0.85 | **+0.56** | 略好于平均 → 适度提高概率 |
| #5 | 0.83 | **+0.32** | 接近平均 → 小幅提高概率 |
| #8 | 0.79 | **-0.20** | 略低于平均 → 小幅降低概率 |
| #6 | 0.77 | **-0.41** | 低于平均 → 降低概率 |
| #2 | 0.72 | **-1.02** | 比平均差很多 → 大幅降低概率 |
| #4 | 0.68 | **-1.48** | 最差 → 大幅降低概率 |

#### 6.4.3 截断到 ±5

```python
adv_clip_max = 5.0
advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
# 上面的例子没有超限，但如果有极端奖励（比如 PDM=100），
# 截断防止梯度爆炸
```

#### 6.4.4 代码

```python
# src/simwam/trainer_grpo.py
def compute_advantages(self, rewards):
    """
    组内归一化：把奖励翻译成 advantage。
    
    A_i = (r_i - μ) / σ
    
    其中 μ = mean(rewards), σ = std(rewards) + 1e-4
    """
    mean = rewards.mean()
    std = rewards.std() + 1e-4
    advantages = (rewards - mean) / std
    
    # 截断到 ±adv_clip_max
    advantages = advantages.clamp(-self.adv_clip_max, self.adv_clip_max)
    
    return advantages
```

**对应公式：**

$$A_i = \frac{r_i - \mu}{\sigma + 10^{-4}}, \qquad A_i \in [-5, 5]$$

**为什么需要 advantage？** 纯奖励 $r_i$ 的尺度会漂移（今天 PDM 打 0.8，明天整体打 0.9）。只看组内相对好坏，就能甩开这些波动，只关注"这组里哪个更好"这个稳定信号。

#### 6.4.5 Advantage 的梯度直觉

策略梯度 $\nabla L = -\mathbb{E}[A \cdot \nabla \log \pi_\theta]$ 的含义：

| Advantage | 梯度方向 | 效果 |
|------|------|------|
| $A > 0$（好轨迹） | $\nabla \log \pi_\theta > 0$ | **增大**这类轨迹的概率 |
| $A < 0$（差轨迹） | $\nabla \log \pi_\theta < 0$ | **减小**这类轨迹的概率 |
| $A = 0$（平均） | 梯度 ≈ 0 | 不影响 |

**关键洞察**：GRPO 不需要知道"绝对好坏"，只需要知道"组内相对好坏"。这是它能去掉 Critic 的核心原因。

### 6.5 PPO 更新——Ratio + Clip 的数值直觉

#### 6.5.1 什么是 Ratio？

Ratio 衡量的是"新策略下这条轨迹的概率是变高了还是变低了"：

$$r_i = \frac{\pi_{\text{new}}(\tau_i)}{\pi_{\text{old}}(\tau_i)} = \exp\big(\log \pi_{\text{new}} - \log \pi_{\text{old}}\big)$$

其中 $\log \pi = \sum_k \log p_\theta(x_{k+1} | x_k)$ 是轨迹的对数概率（所有步的 log_prob 之和）。

#### 6.5.2 数值走一遍

假设优势 $A_i = 1.29$（好轨迹），ratio 有三种情况：

| 情况 | $r_i$ | $A_i \cdot r_i$ | $A_i \cdot \text{clip}(r_i)$ | 选哪个？ |
|---|---|---|---|---|
| 新策略概率略高 | 1.03 | 1.33 | 1.33 | $r_i \cdot A_i$（不 clip） |
| 新策略概率大增 | 1.50 | 1.94 | 1.35（clip 到 1.02） | $\text{clip}(r_i) \cdot A_i$（被 clip） |
| 新策略概率大降 | 0.50 | 0.65 | 0.63 | $r_i \cdot A_i$（不 clip） |

**PPO clip 的效果**：当 ratio 偏离 1 太远（新策略"太激进"地提高概率），clip 会把 ratio 限制在 $[1-\varepsilon, 1+\varepsilon]$ 范围内，防止一步更新太猛。

```python
# PPO clipped loss 的直觉
# 对于好轨迹（A>0）：
#   ratio < 1-ε: 被 clip → 阻止"过度降低"好轨迹的概率
#   1-ε < ratio < 1+ε: 正常更新
#   ratio > 1+ε: 被 clip → 阻止"过度提高"好轨迹的概率（防止reward hacking）

# 对于差轨迹（A<0）：
#   ratio < 1-ε: 被 clip → 阻止"过度降低"差轨迹的概率
#   1-ε < ratio < 1+ε: 正常更新
#   ratio > 1+ε: 被 clip → 阻止"过度提高"差轨迹的概率
```

#### 6.5.3 为什么 PPO 比纯 REINFORCE 好？

纯 REINFORCE 直接更新 $\nabla L = A \cdot \nabla \log \pi$，没有 ratio 和 clip。问题：

1. **步长不可控**：如果 advantage 很大（比如 A=5），梯度也会很大，一步更新可能让策略"跳"到完全不同的区域
2. **无法复用数据**：每批 rollout 只能用一次，否则估计不准确

PPO 通过 ratio + clip 解决这两个问题：
- Ratio 衡量"新旧策略差多少"，clip 限制"最多差多少"
- 同一批 rollout 可以复用 `num_inner_epochs=4` 次，因为 clip 保证了即使复用也不会更新太远

#### 6.5.4 代码逐行拆解

```python
# src/simwam/trainer_grpo.py
def update_policy(self, rollout_batch, advantages):
    """
    PPO 风格的策略更新。
    """
    # 1. 用新模型重新计算 log_prob
    # rollout_batch 中保存了采样时的 log_probs_old
    log_probs_new = self.model.compute_log_prob(rollout_batch)
    # log_probs_new shape: (B, T) → 每条轨迹每步的 log_prob
    
    # 2. 计算轨迹级 log_prob（所有步求和）
    log_prob_new = log_probs_new.sum(dim=-1)   # (B,)
    log_prob_old = rollout_batch.log_probs_old.sum(dim=-1)  # (B,)
    
    # 3. 计算 ratio
    ratio = torch.exp(log_prob_new - log_prob_old)
    # ratio = π_new(τ) / π_old(τ)
    # ratio > 1 → 新策略更偏好这条轨迹
    # ratio < 1 → 新策略不太偏好这条轨迹
    
    # 4. PPO clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - self.ppo_clip_range, 1 + self.ppo_clip_range) * advantages
    
    # 取两个中的较大值（加负号变最小化）
    loss = -torch.min(surr1, surr2).mean()
    # 直觉：如果 clip 没起作用（ratio 在范围内），loss = -ratio·A
    #       如果 clip 起作用（ratio 超范围），loss = -clip(ratio)·A（更保守）
    
    # 5. 反向传播（只更新 LoRA 参数）
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
```

#### 6.5.5 PPO Clip 的几何直觉

画出 loss 随 ratio 变化的曲线：

```
 loss
  ^
  |        /
  |       /
  |      / ← 被 clip 住（斜率 = A）
  |     /
  |    / ← 正常区域（斜率 = A）
  |   /
  |  /
  | /
  +------------------------> ratio
  0.98  1.0  1.02
  (1-ε)       (1+ε)
```

- 在 $[1-\varepsilon, 1+\varepsilon]$ 内：loss 正比于 $r_i \cdot A_i$（正常更新）
- 超出范围：loss 被 clip 到边界值（限制更新幅度）

$\varepsilon = 0.02$ 意味着：**新旧策略的概率比最多偏离 2%**。这是非常保守的更新——RL 微调必须小心，步子太大容易"忘记"SFT 学到的能力。

### 6.6 LoRA 微调

```python
# src/simwam/models/wan22/lora.py
class LoRALinear(nn.Module):
    """
    低秩适配器：只训练 A 和 B 两个小矩阵，
    冻结原始权重 W。
    
    W' = W + (α/r) · B @ A
    """
    def __init__(self, in_features, out_features, r=16, alpha=32.0):
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.alpha = alpha
        self.scaling = alpha / r  # = 32/16 = 2
    
    def forward(self, x):
        # 原始输出 + LoRA 增量
        return F.linear(x, self.weight) + self.scaling * self.lora_B(self.lora_A(x))
```

**GRPO 训练时的冻结策略：**

| 模块 | 训练状态 | 原因 |
|------|---------|------|
| Video DiT | **完全冻结** | 不需要修改视频生成能力 |
| ActionDiT self-attention | **加 LoRA** | 需要学习新的注意力模式 |
| ActionDiT cross-attention | **加 LoRA** | 需要学习新的跨模态交互 |
| ActionDiT FFN | **冻结** | FFN 通常不需要微调 |
| 输出头 | **全参数训练** | 输出维度小，直接训练更高效 |

### 6.7 为什么 FlowGRPO 特别适合自动驾驶？——五项设计选择的深层逻辑

前面拆解了 FlowGRPO 的每个技术细节。这一节回答一个更根本的问题：**这些设计选择为什么特别适合自动驾驶，而不是简单照搬图像生成的 Flow-GRPO？**

#### 设计选择 1：低频余弦基噪声——轨迹的物理约束

| | 图像生成（字节 Flow-GRPO） | 自动驾驶（SimWAM FlowGRPO） |
|---|---|---|
| 噪声类型 | 标准高斯（全维度独立） | 低频余弦基（6 个基函数） |
| 原因 | 图像像素之间没有物理约束，高频噪声能产生有意义的变化 | 轨迹必须**平滑可执行**，高频抖动 = 车辆无法跟随 |
| 探索空间 | 128×128×64 = 百万维 | **6 维**（6 个基函数的权重） |

**深层逻辑**：驾驶轨迹不是"任意 50 个点"，而是**车辆动力学约束下的连续路径**。方向盘转角不能突变，横向加速度有物理上限。余弦基噪声把探索限制在"车辆能执行"的范围内，大幅减少了无效探索。

论文原文：
> *"Independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity."*

#### 设计选择 2：固定最后 3 步——轨迹生成的短程特性

| | 图像生成（字节 Flow-GRPO） | 自动驾驶（SimWAM FlowGRPO） |
|---|---|---|
| 总步数 | ~50 步 | 10 步 |
| SDE 窗口 | 随机滑动 | 固定最后 3 步 |
| 原因 | 50 步中不同阶段的噪声对图像质量影响不同 | 10 步太短，无法随机窗口；最后 3 步直接影响最终轨迹 |

**深层逻辑**：轨迹生成的 10 步积分有一个关键特性——**每一步都是等权重的**（$\Delta t = 0.1$）。不像图像生成有"先构图后细节"的层次结构，轨迹的每一步对最终位置的贡献几乎相同。所以早期步的扰动会被后续步均匀衰减，只有最后 3 步能直接影响输出。

#### 设计选择 3：PDM 奖励——安全红线的硬约束

| | 图像生成（字节 Flow-GRPO） | 自动驾驶（SimWAM FlowGRPO） |
|---|---|---|
| 奖励类型 | 多维质量打分（PickScore 等） | 单标量 PDM 分数 |
| 奖励结构 | 加性（多维向量） | **乘性 + 加权**（NC/DAC 一票否决） |
| 原因 | 图像质量是多维度的，没有"红线" | 驾驶有**安全红线**（碰撞/越界 = 灾难） |

**深层逻辑**：PDM 的乘性结构（$NC \times DAC \times \ldots$）不是随意设计的，而是反映了驾驶的**分层安全逻辑**：

```
安全红线（NC=0 或 DAC=0）→ 总分 = 0，无论其他指标多好
├── 安全裕度（TTC）→ 碰撞时间越长越好
├── 效率（EP）→ 前进越多越好
└── 舒适度（C）→ 加速度越小越好
```

这种"先过安全红线，再比效率和舒适"的分层结构，和人类驾驶的决策逻辑完全一致——你不会为了"开得更快"而闯红灯。

#### 设计选择 4：无 KL 惩罚——小模型 + LoRA 的稳定性

| | 图像生成（字节 Flow-GRPO） | 自动驾驶（SimWAM FlowGRPO） |
|---|---|---|
| Loss | PPO + KL 惩罚 | 纯 PPO |
| 原因 | FLUX 12B 参数量大，RL 容易"跑偏" | ActionDiT 参数量小 + LoRA 约束，不容易跑偏 |

**深层逻辑**：KL 惩罚的本质是"别离参考策略太远"。对于 12B 参数的 FLUX，参数空间巨大，RL 可能探索到"高奖励但质量崩塌"的区域。但 ActionDiT 参数量小，且 LoRA 本身就限制了更新幅度（只改 5% 的参数），所以 KL 惩罚是多余的。

此外，PDM 奖励是**规则计算**（不是神经网络打分），不容易被 hack——你没法"生成一条看起来很好但 PDM 分数虚高"的轨迹。所以不需要 KL 来防止 reward hacking。

#### 设计选择 5：10 步积分 vs 50 步——推理效率的硬需求

| | 图像生成 | 自动驾驶 |
|---|---|---|
| 去噪步数 | ~50 步 | 10 步 |
| 推理延迟要求 | 无实时要求（可以等 5s） | **< 100ms**（必须跟上 10Hz 的控制频率） |
| 原因 | 图像质量优先 | **实时性优先** |

**深层逻辑**：自动驾驶的推理延迟直接影响安全性——如果规划器需要 500ms 才输出轨迹，车辆在这 500ms 内是"盲目"的。10 步积分 + ActionDiT 的轻量设计，让 SimWAM 的推理延迟远低于 World Model 路线（后者需要生成未来帧，步数更多）。

#### 总结：FlowGRPO 的自动驾驶适配清单

| 设计选择 | 图像生成的默认做法 | 自动驾驶的适配做法 | 适配原因 |
|---|---|---|---|
| 噪声类型 | 标准高斯 | 低频余弦基 | 轨迹必须平滑可执行 |
| SDE 窗口 | 随机滑动 | 固定最后 3 步 | 10 步太短，最后 3 步最有效 |
| 奖励函数 | 多维质量打分 | PDM 乘性分数 | 安全红线需要一票否决 |
| KL 惩罚 | 有（防跑偏） | 无（LoRA 已约束） | 小模型 + 规则奖励不容易跑偏 |
| 推理步数 | 50 步 | 10 步 | 实时性要求 < 100ms |

这些适配不是"随便改改"，而是**根据驾驶任务的物理约束和安全需求做的系统性调整**。核心数学框架（Flow Matching → ODE→SDE → PPO）完全相同，但每个组件的参数和设计都针对驾驶场景做了优化。

---

## 第 7 步：评测流程

### 7.1 评测入口

```bash
# experiments/navsim/run_eval_navsim.sh
CKPT=./runs/grpo/.../checkpoints/weights/step_2000.pt \
TASK=navsim_grpo_action_pdm_384x672_flowgrpo_lora \
NPROC_PER_NODE=8 \
bash experiments/navsim/run_eval_navsim.sh
```

### 7.2 评测脚本核心逻辑

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

## 附录 A｜Flow Matching 数学推导

### A.1 速度场从哪来：直线假设 + 求导

我们要求中间状态 $x_t$ 满足三个朴素条件：
1. $t=0$（还没走）时必须正好是噪声：$x_0$；
2. $t=1$（走完了）时必须正好是轨迹：$x_1$；
3. 中间平滑过渡——最简单的假设就是**线性**（直线）。

设直线 $x_t = a + b\,t$（$a,b$ 是待定系数），代条件 1 和 2：

- $t=0$：$x_0 = a + b\cdot0 = a$，所以 $\boxed{a = x_0}$；
- $t=1$：$x_1 = a + b\cdot1$，把 $a=x_0$ 代入得 $\boxed{b = x_1 - x_0}$。

把 $a,b$ 代回直线方程：

$$
x_t = x_0 + t\,(x_1 - x_0) = (1-t)\,x_0 + t\,x_1
$$

逐项翻译：噪声 $x_0$ 的系数是 $(1-t)$，轨迹 $x_1$ 的系数是 $t$，**两个系数加起来恒等于 1**（$(1-t)+t=1$）。所以它本质是"噪声和轨迹的**加权平均**"，权重随进度条 $t$ 变化：$t=0$ 全是噪声，$t=1$ 全是轨迹，$t=0.5$ 各占一半。

**速度为什么等于 $x_1-x_0$？——求一次导就出来**

速度 = "位置随时间的变化率"，也就是对 $x_t$ 关于 $t$ 求导：

$$
u_t = \frac{d x_t}{dt} = \frac{d}{dt}\big((1-t)x_0 + t x_1\big) = -x_0 + x_1 = x_1 - x_0
$$

结论：**直线轨迹上每个点的瞬时速度都是同一个常数 $x_1-x_0$**（从噪声指向轨迹的那个方向）。

### A.2 训练目标：CFM 损失

网络 $v_\theta(x_t,t,c)$ 的输入是"我在哪（$x_t$）、几点（$t$）、目标长啥样（$c$=prompt）"，输出是"我认为该往哪走（速度）"。

训练时我们有成对的"噪声 + 它对应的真轨迹"，所以**正确答案是现成的**：就是 $x_1-x_0$。于是损失（衡量网络猜得有多离谱的"扣分器"）就是：

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}\Big[\big\| v_\theta(x_t,t,c) - (x_1-x_0) \big\|^2\Big]
$$

逐项翻译：
- $v_\theta(x_t,t,c)$：网络猜的速度；$(x_1-x_0)$：正确的速度；两者相减再平方 = "**猜偏了多少**"。
- $\mathbb{E}[\cdot]$：期望 = 拿很多很多对"噪声+轨迹"反复问，取平均。$\mathbb{E}_{t\sim U[0,1]}$ 表示还随机抽进度条位置 $t$（每个中间位置都练到）。

### A.3 推理：ODE 积分 + Euler 步

学完速度后怎么生成一条轨迹？——从噪声出发，沿学到的速度一路走到底。这条路在数学上叫**解 ODE（常微分方程）**：

$$
\frac{dx}{dt} = v_\theta(x,\;t,\;c), \qquad x_0\sim \mathcal{N}(0,I),\;\; t:0\to 1
$$

"Euler 步"就是"泰勒展开的第一项"：

$$
x_{t+\Delta t} = x_t + v_\theta(x_t,t,c)\,\Delta t
$$

翻译：**新位置 = 旧位置 + 当前速度 × 一小段时间**。SimWAM 推理时走 10 步（$\Delta t = 0.1$），10 个 Euler 步进就得到最终轨迹。

---

## 附录 B｜GRPO Advantage 数学

$$
\mu = \frac{1}{G}\sum_{i=1}^{G} r_i, \qquad
\sigma = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu)^2 + 10^{-4}}
$$

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

- $A_i>0$ = 比组内平均好 → **提高**这类轨迹的概率；
- $A_i<0$ = 比组内平均差 → **降低**这类轨迹的概率。

**为什么 GRPO 不需要 Critic（价值网络）？**

- PPO 的 advantage 是 $A(s,a)=Q(s,a)-V(s)$，需要额外训练一个价值网络 $V$ 当"基线"，参数量≈策略网络，显存翻倍。
- GRPO 的妙招：**拿"同组平均 $\mu$"当基线、"同组标准差 $\sigma$"当尺度**——就像全班互相评分、按曲线给分，不需要老师预先定标准，也就省掉了整套 critic。

---

## 附录 C｜策略梯度推导

$$
\nabla_\theta \mathcal{L} = -\,\mathbb{E}_\tau\Big[\, A(\tau)\; \nabla_\theta \log\pi_\theta(\tau) \Big]
$$

推导六步：
1. 目标：$J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$
2. 难点：$R(\tau)$ 里面没有 $\theta$，直接求导卡住
3. log 求导技巧：$\frac{d\pi_\theta}{d\theta} = \pi_\theta \cdot \frac{d}{d\theta}\log\pi_\theta$
4. 代入目标：$\nabla_\theta J = \mathbb{E}_\tau[R(\tau)\,\nabla_\theta\log\pi_\theta(\tau)]$
5. 用 advantage 替代奖励（减基线降方差）
6. 转成 loss（加负号）

直觉：$A>0$（好事）的轨迹，梯度方向让 $\log\pi_\theta(\tau)$ **增大**（更可能再走出这条路）；$A<0$（坏事）的轨迹，让它**减小**。

---

## 🧭 总结：一图串起整个 SimWAM

到这里，正文的每一步、代码里的每个函数、附录里的每条公式都散落在各处。这一节把它们**串成一条完整的因果链**：

![SimWAM 完整逻辑链总结](/images/simwam/simwam_summary.svg)

### 沿着因果链走一遍（6 个节点）

| 节点 | 一句话 | 核心公式 | 为什么必须这么做 |
|------|--------|---------|-----------------|
| **① Flow Matching** | 训练速度场 | $L=E\|v_\theta-(x_1-x_0)\|^2$ | 没有速度场就没有"生成"这回事 |
| **② Isolated Mask** | 隔离信息流 | Future ↔ Action: mask=False | 训练时借力、推理时独立 |
| **③ 联合训练** | Video + Action 同时学 | $L = \lambda_v L_v + \lambda_a L_a$ | 两个专家共享运动先觉 |
| **④ FlowGRPO** | SDE 采样 + PDM 奖励 | $x_{k+1} = x_k + v\Delta t + \sigma n$ | 用奖励信号微调策略 |
| **⑤ PPO 更新** | 抬好压坏 | $L = \min(rA, \text{clip}(r)A)$ | 稳定更新 LoRA |
| **⑥ 推理** | 只跑 ActionDiT | 10 步欧拉积分 | 不需要生成未来帧 |

### 从三条主线理解

- **FM 生成（①②③）**：先让模型学会"从噪声走到轨迹的速度"（①），用 Isolated Mask 隔离信息流（②），联合训练让运动先觉流入 ActionDiT（③）。
- **RL 对齐（④⑤）**：用 SDE 采样探索不同轨迹方向（④），用 PDM 奖励打分，组内归一化算 advantage，PPO clip 稳定更新（⑤）。
- **推理（⑥）**：扔掉 Video DiT，只跑 ActionDiT，10 步欧拉积分直接输出轨迹（⑥）。

### 三句话总串

> 1. **Flow Matching 先教会两个专家"从噪声走到数据的速度"**——这是生成能力的来源；
> 2. **Isolated Attention Mask 让 Video Expert 的运动先觉流入 Action Expert，但推理时不需要 Video Expert**——这是"训练时借力、推理时独立"的关键；
> 3. **FlowGRPO 用 PDM 奖励 + SDE 采样 + LoRA 微调进一步提升轨迹质量**，然后推理时只跑 ActionDiT——这就是 SimWAM 的全部逻辑闭环。

---

## 🔬 个人解读与思考

### 1. "训练时借力、推理时独立"的设计哲学

SimWAM 最聪明的地方在于：**训练时让 Video Expert 和 Action Expert 联合学习，但推理时只用 Action Expert**。这解决了 World Model 路线的核心痛点——推理时生成未来帧太慢。

具体实现靠 Isolated Attention Mask：
- 训练时：Video tokens 和 Action tokens 拼在一起做联合注意力，但掩码阻止 Action 看到未来帧
- 推理时：直接扔掉 Video DiT，只跑 Action DiT

这个设计的好处：
1. **训练效率**：Video Expert 的运动先觉通过共享注意力流入 Action Expert
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

![WAM 对比](/images/simwam/simwam_comparison.svg)

| | DriveWAM | DriveLaW | SimWAM |
|---|---|---|---|
| 视频 backbone | 自训 | 自训 | **Wan2.2-5B（预训练）** |
| 推理时需要生成未来帧？ | 是 | 是 | **否** |
| 推理延迟 | 高 | 高 | **低** |
| PDMS | 90.1 | 89.1 | **91.5** |
| RL 后训练 | 无 | 无 | **FlowGRPO** |

SimWAM 的核心优势：**用预训练视频模型的运动先觉，但推理时不需要生成未来帧**——既借了视频模型的力，又不背它的包袱。

### 5. SimWAM 的 FlowGRPO vs 字节 Flow-GRPO：同名不同魂

博客里有一篇 [Flow-GRPO 完全讲解](/zh/posts/thoughts/flow-grpo-complete-guide/)，讲的是字节跳动把 Flow-GRPO 用在图像生成（FLUX 模型）上的方法。SimWAM 也叫 FlowGRPO，但两者**名字相似、内核不同**——一个是画图的，一个是开车的。这一节把两者的每个技术细节拆开对比。

#### 总览：一句话区分

| | 字节 Flow-GRPO | SimWAM FlowGRPO |
|---|---|---|
| **一句话** | 用 RL 让 FLUX 生成更好的图像 | 用 RL 让 ActionDiT 生成更好的驾驶轨迹 |
| **论文/博客** | Flow-GRPO: Boosting Reward-based Generation Models (字节) | SimWAM (arXiv:2608.07468) |
| **基座模型** | FLUX 12B DiT（图像生成） | ActionDiT（轨迹生成） |
| **生成空间** | 高维 latent（64×128×128） | 低维 waypoint（50×3） |

#### 1. 奖励函数：画图 vs 开车

这是**最本质的区别**——奖励决定了 RL 往哪个方向优化。

**字节 Flow-GRPO**：奖励来自图像质量评估

```python
# flow_grpo/rewards.py
reward_fn = multi_score(device, config.reward_fn)
# 返回 {"avg": [0.45, 0.78, ...], "strict_accuracy": [0, 1, ...]}

# 具体奖励类型：
# - OCR reward: 图中文字是否正确渲染
# - PickScore: 人类偏好分数
# - GenEval: 组合逻辑检查（几个物体、什么颜色、什么关系）
```

奖励是**多维度、任务相关**的——可以同时优化文本渲染、美学质量、组合正确性。

**SimWAM FlowGRPO**：奖励来自 NAVSIM PDM 分数

```python
# src/simwam/datasets/navsim/pdm_reward.py
PDMS = NC × DAC × (5·TTC + 5·EP + 2·C) / 12
```

奖励是**单一标量、驾驶专用**的——NC（无责碰撞）、DAC（可行驶区域）、TTC（碰撞时间）、EP（前进进度）、C（舒适度）。五个子指标乘性组合，一个分数定好坏。

| 维度 | 字节 Flow-GRPO | SimWAM FlowGRPO |
|------|------|------|
| 奖励类型 | 多维向量（avg + 子分数） | 单标量 PDMS |
| 奖励来源 | 预训练打分器（PickScore 等） | 规则计算（PDM scorer） |
| 奖励范围 | [0, 1] 连续 | [0, 100] 连续 |
| 任务相关性 | 通用（可换不同奖励） | 驾驶专用（NAVSIM 基准） |
| 奖励是否可微 | 通常不可微（黑盒打分器） | 不可微（规则计算） |

#### 2. SDE 采样：滑动窗口 vs 固定末尾

两者都需要把确定性 ODE 转成随机 SDE 来获得可微的 $\log p$，但**加噪声的位置和方式完全不同**。

**字节 Flow-GRPO**：SDE 滑动窗口

```python
# flux_pipeline_with_logprob_fast.py
# 窗口在 [sde_window_range[0], sde_window_range[1]] 内随机选取
start = randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
end = start + sde_window_size
sde_window = (start, end)  # 例如 (2, 5)

# 窗口前：纯 ODE（无噪声）
# 窗口内：SDE（加噪声）
# 窗口后：纯 ODE
```

为什么要随机窗口？因为图像生成有 ~50 步去噪步，不同步的噪声对生成质量影响不同。随机窗口让 RL 在**不同去噪阶段**都有探索机会。

**SimWAM FlowGRPO**：固定在最后 3 步

```python
# src/simwam/trainer_grpo.py
for k in range(10):  # 10 步欧拉积分
    if k in [7, 8, 9]:  # 只在最后 3 步加 SDE 扰动
        noise = self.compute_low_freq_noise(traj)  # 6 个余弦基
        score_correction = self.compute_score_correction(traj, t)
        traj = traj + sigma * noise + 0.5 * sigma**2 * score_correction
```

为什么固定在最后 3 步？因为轨迹生成只有 10 步，步数太少不能随机窗口。而且**靠近输出端的扰动对最终轨迹影响最直接**——早期扰动会被后续积分步骤衰减。

| 维度 | 字节 Flow-GRPO | SimWAM FlowGRPO |
|------|------|------|
| 总去噪步数 | ~50 步 | 10 步 |
| SDE 窗口位置 | 随机滑动 | 固定在最后 3 步 |
| 窗口大小 | 可配置 | 3 步 |
| 噪声类型 | 标准高斯 | **低频余弦基**（6 个基函数） |
| Score correction | 标准 Fokker-Planck 逆推 | 同（标准公式） |

#### 3. 噪声设计：高频探索 vs 低频探索

**字节 Flow-GRPO**：标准高斯噪声

```python
# 标准 SDE：在 latent 空间加各向同性高斯噪声
variance_noise = randn_tensor(latents.shape)
prev_sample = prev_sample_mean + std_dev_t * sqrt(-dt) * variance_noise
```

图像 latent 是高维的（64×128×128），标准高斯噪声在每个维度上独立扰动，足以产生多样的图像变化。

**SimWAM FlowGRPO**：低频余弦基噪声

```python
# 不是独立路点噪声，而是限制在 6 个余弦基上
# 只在"整体偏左/偏右""整体加速/减速"这几个低维模态上探索
noise = compute_low_freq_noise(traj)  # 6 个余弦基
```

为什么？因为轨迹只有 50 个路点 × 3 维 = 150 维，如果每个维度独立加噪声，会产生**高频抖动**——第 3 个路点突然左拐，第 4 个又右拐。这在物理上不可执行。余弦基把探索限制在低频模态上，生成的轨迹天然平滑。

| 维度 | 字节 Flow-GRPO | SimWAM FlowGRPO |
|------|------|------|
| 噪声维度 | 全维度（64×128×128） | 低维（6 个余弦基） |
| 频率特性 | 高频（独立维度） | **低频**（平滑基函数） |
| 物理约束 | 无（图像无物理约束） | 有（轨迹必须平滑可执行） |
| 噪声级别 | noise_level 可配置 | noise_level=0.1 |

#### 4. ODE→SDE 转换的数学：殊途同归

两者都需要把确定性 Flow ODE 转成 SDE，数学推导过程**完全一致**：

**第一步：Fokker-Planck 逆推**

$$f_{\text{SDE}} = v_t - \frac{\sigma_{\text{noise}}^2}{2}\nabla\log p_t(x)$$

**第二步：Score identity（Rectified Flow 特有）**

$$\nabla\log p_t(x) = -\frac{x}{t} - \frac{1-t}{t}\,v_t(x)$$

**第三步：Euler-Maruyama 离散化**

$$x_{t+\Delta t} = x_t + \left[v_\theta + \frac{\sigma_{\text{noise}}^2}{2t}\big(x_t + (1-t)v_\theta\big)\right]\Delta t + \sigma_{\text{noise}}\sqrt{-\Delta t}\,\epsilon$$

**第四步：高斯 log_prob**

$$\log p(x_{t+1}\mid x_t) = -\frac{\|x_{t+1}-\text{mean}_\theta\|^2}{2\sigma_{\text{noise}}^2(-\Delta t)} - \log\big(\sigma_{\text{noise}}\sqrt{-\Delta t}\big) - \log\sqrt{2\pi}$$

唯一的差异是 $\sigma_{\text{noise}}$ 的具体数值（取决于 `noise_level` 和 $t$ 的关系），但**公式结构完全相同**。这说明 Flow-GRPO 的 ODE→SDE 框架是**通用的**——不管生成的是图像还是轨迹，只要底层是 Flow Matching，都能套用。

#### 5. Loss 设计：有无 KL 惩罚

**字节 Flow-GRPO**：PPO + KL 惩罚

```python
# 完整 loss = PPO policy loss + KL penalty
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

# KL penalty：当前策略 vs 参考策略（关闭 LoRA 后的基座模型）
with transformer.module.disable_adapter():  # 关闭 LoRA = 参考模型
    _, _, prev_sample_mean_ref, _ = compute_log_prob(...)

kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() / (2 * std_dev_t ** 2)
loss = policy_loss + config.train.beta * kl_loss
```

为什么要 KL？因为图像生成空间大（12B 参数），RL 容易"跑偏"——生成高奖励但质量崩塌的图像。KL 惩罚让新策略别离参考策略太远。

**SimWAM FlowGRPO**：纯 PPO，无 KL

```python
# SimWAM 只用 PPO clipped loss，没有 KL 项
ratio = torch.exp(log_probs_new - rollout_batch.log_probs_old)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - self.ppo_clip_range, 1 + self.ppo_clip_range) * advantages
loss = -torch.min(surr1, surr2).mean()
```

为什么不需要 KL？可能原因：
1. ActionDiT 参数量小（相比 FLUX 12B），不需要 KL 约束
2. LoRA 微调本身就有正则化效果（不改变原始权重）
3. 奖励函数（PDM）比较稳定，不容易 hack

| 维度 | 字节 Flow-GRPO | SimWAM FlowGRPO |
|------|------|------|
| Loss 结构 | PPO + KL 惩罚 | 纯 PPO |
| KL 系数 β | 可配置（通常 0.01-0.1） | 无 |
| 参考策略 | 关闭 LoRA 后的基座模型 | 不使用 |
| 正则化来源 | KL + LoRA 约束 | 仅 LoRA 约束 |

#### 6. LoRA 配置：大模型 vs 小模型

**字节 Flow-GRPO**：大 rank、宽覆盖

```python
# FLUX 12B 模型，需要更大容量的 LoRA
transformer_lora_config = LoraConfig(
    r=64,              # rank=64
    lora_alpha=128,    # alpha=128，scaling=2
    target_modules=["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", ...]
)
```

**SimWAM FlowGRPO**：小 rank、窄覆盖

```yaml
# ActionDiT 参数量小，LoRA 也小
lora:
  r: 16              # rank=16（FLUX 的 1/4）
  alpha: 32.0        # alpha=32，scaling=2（比例相同）
  target_modules: [q, k, v, o]  # 只在注意力层
```

| 维度 | 字节 Flow-GRPO | SimWAM FlowGRPO |
|------|------|------|
| LoRA rank | 64 | 16 |
| LoRA alpha | 128 | 32 |
| Scaling (α/r) | 2 | 2（相同） |
| 目标模块 | Q, K, V, Out, + MLP | Q, K, V, O |
| 可训练参数占比 | ~5% | ~5% |

比例相同（scaling=2），但绝对容量不同——图像生成需要更多可训练参数来编码复杂的视觉偏好。

#### 7. EMA：有 vs 无

**字节 Flow-GRPO**：使用 EMA 稳定评估

```python
# 每 8 步更新 EMA 参数
ema_decay = 0.9
ema_params = decay * ema_params + (1 - decay) * current_params

# 训练用实时参数，评估/保存用 EMA 参数
```

**SimWAM FlowGRPO**：不使用 EMA

训练完直接保存 LoRA checkpoint，评估时加载。

| 维度 | 字节 Flow-GRPO | SimWAM FlowGRPO |
|------|------|------|
| EMA | 有（decay=0.9） | 无 |
| 评估参数 | EMA 平滑参数 | 实时参数 |
| 保存策略 | EMA checkpoint | LoRA checkpoint |

#### 8. 核心流程对比图

```
字节 Flow-GRPO（图像生成）：
  FLUX 12B → 采样图像 → 多维奖励打分 → 组内归一化 advantage
  → PPO+KL loss → LoRA 更新 → EMA 平滑 → 下一轮

SimWAM FlowGRPO（驾驶轨迹）：
  ActionDiT → 采样 8 条轨迹 → PDM 奖励打分 → 组内归一化 advantage
  → 纯 PPO loss → LoRA 更新 → 下一轮
```

#### 9. 总结：为什么同名但不同？

| 本质差异 | 原因 |
|------|------|
| **生成空间不同** | 图像是高维 latent（128×128×64），轨迹是低维 waypoint（50×3） |
| **物理约束不同** | 图像无物理约束，轨迹必须平滑可执行 |
| **奖励函数不同** | 图像用多维质量打分，轨迹用规则 PDM 分数 |
| **去噪步数不同** | 图像 ~50 步（需要滑动窗口），轨迹 10 步（固定末尾） |
| **模型规模不同** | FLUX 12B 需要 KL+EMA，ActionDiT 小模型不需要 |

**但核心数学框架完全相同**：Flow Matching → ODE→SDE 转换 → 高斯 log_prob → PPO clipped loss → 组内归一化 advantage。这说明 Flow-GRPO 是一个**通用的 RL 框架**，可以适配不同的生成任务——只要底层是 Flow Matching，换奖励函数就能用。

---

### 6. 闭环仍是短板

和 Qwen-Drive-1.0 一样，SimWAM 在 AlpaSim 闭环上（0.30 at-fault score）还不如 Alpamayo-R1（0.58）。可能原因：
- 非反应式仿真训练的轨迹在闭环交互场景下缺乏"反应式"调整
- PDM 奖励主要基于开环指标，对闭环交互的覆盖不足
- 未来可能需要引入闭环仿真训练

---

## 📝 一句话总结

**SimWAM = Wan2.2-5B Video DiT（冻结）+ ActionDiT（可训）+ Isolated Attention Mask（训练时隔离、推理时独立）+ 联合 Flow Matching（SFT）+ FlowGRPO（RL 后训练，LoRA 微调）。训练时借视频模型的运动先觉，推理时只用动作专家直接输出轨迹——NAVSIM 91.5 PDMS，推理延迟远低于其他 WAM。**

---

## 参考

- 代码库：`https://github.com/H-EmbodVis/SimWAM`（Apache 2.0）
- 论文：SimWAM: A Simple World Action Model for End-to-End Autonomous Driving (arXiv:2608.07468)
- Wan2.2：Wan2.2-T2V-14B 开源视频模型
- Flow Matching：Flow Matching for Generative Modeling (arXiv:2210.02747)
- GRPO：DeepSeekMath (arXiv:2402.03300)
- PPO：Proximal Policy Optimization Algorithms (arXiv:1707.06347)
- NAVSIM：NAVSIM benchmark for end-to-end driving
