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

```python
# src/simwam/trainer_grpo.py
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

**对应公式：**

$$x_{k+1} = x_k + v_\theta(x_k, t, c) \cdot \Delta t + \sigma_{\text{noise}} \cdot \text{noise} + \frac{1}{2}\sigma_{\text{noise}}^2 \cdot \text{score\_correction}$$

**为什么只在最后 3 步加扰动？** 论文实验发现：靠近输出端的扰动对最终轨迹影响最直接。早期步的扰动会被后续步"吸收"，效果不明显。

**为什么用低频余弦基？** 独立路点噪声会产生高频抖动（第 3 个路点突然拐一下）。低频余弦基（6 个基）只在"整体偏左/偏右""整体加速/减速"这几个低维模态上探索——和人类驾驶的直觉一致。

### 6.3 PDM 奖励计算

```python
# src/simwam/datasets/navsim/pdm_reward.py
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

### 6.4 Advantage 计算

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

### 6.5 PPO 更新

```python
# src/simwam/trainer_grpo.py
def update_policy(self, rollout_batch, advantages):
    """
    PPO 风格的策略更新。
    
    核心：用 ratio = π_new / π_old 控制更新幅度，
    加 clip 防止一步更新太猛。
    """
    # 计算新模型下的 log_prob
    log_probs_new = self.model.compute_log_prob(rollout_batch)
    
    # PPO ratio
    ratio = torch.exp(log_probs_new - rollout_batch.log_probs_old)
    
    # PPO clipped loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - self.ppo_clip_range, 1 + self.ppo_clip_range) * advantages
    
    loss = -torch.min(surr1, surr2).mean()
    
    # 反向传播（只更新 LoRA 参数）
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
```

**对应公式：**

$$L = -\mathbb{E}\left[\min\left(r_i \cdot A_i, \;\text{clip}(r_i, 1-\varepsilon, 1+\varepsilon) \cdot A_i\right)\right]$$

其中 $r_i = \frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$ 是新旧策略的概率比，$\varepsilon = 0.02$ 是 clip 范围。

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

### 5. 闭环仍是短板

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
