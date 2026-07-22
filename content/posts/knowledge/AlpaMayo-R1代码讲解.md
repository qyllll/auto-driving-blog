---
title: "知识点拆解｜AlpaMayo-R1 代码讲解：VLA 三组件架构与推理流程"
date: 2026-07-22
draft: false
categories: ["知识点拆解"]
tags: ["🔧 代码实践", "🚗 自动驾驶", "🧠 VLA", "🎮 强化学习", "⚡ GRPO"]
summary: "深度拆解 NVIDIA AlpaMayo-R1 的开源代码：VLM + Expert + Diffusion 三组件架构、单轮动力学动作空间编码、Flow Matching 扩散解码器、推理流程的完整数据流。从 config 到采样，带你逐文件弄懂这套 VLA 模型的工程实现。"
weight: 6
---

![AlpaMayo-R1 端到端架构](/images/alpamayo-r1/architecture.png)

![AlpaMayo-R1 代码结构总览](/images/alpamayo-r1/code_structure.svg)

## 一句话理解 AlpaMayo-R1 代码架构

> **AlpaMayo-R1 的推理 = VLM 自回归生成 CoC 推理链 → Expert 模块基于 KV cache 做扩散去噪 → 单轮动力学模型把 (accel, kappa) 解码成轨迹点——三个独立模块串成一条 VLA 流水线。**

代码仓库 [github.com/NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) 只包含推理阶段；SFT 和 RL 后训练脚本在 [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes)。本文只拆解推理代码，这是所有后续训练的基础。

![AlpaMayo-R1 推理流程](/images/alpamayo-r1/inference_flow.svg)

---

## 一、项目结构总览

```
alpamayo/
├── src/alpamayo_r1/
│   ├── config.py                     # AlpamayoR1Config: 扩散/动作空间/Expert 配置
│   ├── helper.py                     # 消息构造 + 处理器初始化 + 设备递归转换
│   ├── test_inference.py             # 端到端推理入口
│   ├── load_physical_aiavdataset.py  # 从 Physical AI AV Dataset 加载数据
│   ├── models/
│   │   ├── alpamayo_r1.py            # AlpamayoR1: VLM + Expert + Diffusion 三组件
│   │   ├── base_model.py             # ReasoningVLA 基类 + TrajectoryFusionMixin
│   │   ├── action_in_proj.py         # PerWaypointActionInProj: Fourier 编码 + MLP
│   │   ├── token_utils.py            # StopAfterEOS / 文本提取 / 填充替换
│   │   └── delta_tokenizer.py        # DeltaTrajectoryTokenizer (Δxyz → 离散 token)
│   ├── action_space/
│   │   ├── action_space.py           # ActionSpace 抽象基类
│   │   ├── unicycle_accel_curvature.py # 单轮动力学模型 (accel, kappa → traj)
│   │   └── discrete_action_space.py  # DiscreteTrajectoryTokenizer
│   ├── diffusion/
│   │   ├── base.py                   # BaseDiffusion + StepFn Protocol
│   │   └── flow_matching.py          # FlowMatching: Euler 积分采样
│   ├── geometry/                     # 旋转矩阵/角度工具
│   └── common/                       # 日志工具
```

关键设计原则：**模块化**。每个组件（VLM、动作空间、扩散、Expert）都有独立的配置系统，通过 `hydra.utils.instantiate` 从 config dict 中注入——替换任意组件不需要改模型代码。

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| `alpamayo_r1.py` | ~260 | 整合 VLM+Expert+Diffusion，暴露 `sample_trajectories_from_data_with_vlm_rollout` |
| `base_model.py` | ~290 | ReasoningVLA 基类：轨迹融合、tokenizer 初始化、模型注册 |
| `flow_matching.py` | ~180 | Flow Matching 的 Euler 积分采样 + 训练数据构造 |
| `unicycle_accel_curvature.py` | ~260 | 单轮动力学模型的 traj↔action 双向映射 |
| `action_in_proj.py` | ~140 | 傅里叶特征编码 + MLP 投影 |
| `test_inference.py` | ~55 | 端到端推理示例，**最简入口** |

---

## 二、推理入口：从 clip_id 到轨迹

### 2.1 test_inference.py：15 行核心逻辑

```python
# test_inference.py（精简核心逻辑）
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
messages = helper.create_message(data["image_frames"].flatten(0, 1))

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")

inputs = processor.apply_chat_template(messages, tokenize=True, ...)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True,
    )
print("CoC:\n", extra["cot"][0])  # ← 打印推理链
```

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `t0_us` | 采样时刻（历史1.6s+未来6.4s的锚点） | 5.1M µs (即 clip 第5.1秒) |
| `num_traj_samples` | 每个场景采样的轨迹数 | 1(默认)~6+ |
| `max_generation_length` | VLM 生成的最大 token 数 | 256 |
| `top_p` / `temperature` | 文本生成的随机性控制 | 0.98 / 0.6 |

### 2.2 load_physical_aiavdataset.py：坐标变换

数据加载的关键是将世界坐标系的自车轨迹转换到**自车 t0 时刻的局部坐标系**：

```python
# 核心坐标变换
t0_xyz = ego_history_xyz[-1]  # t0 时刻位置
t0_rot = spt.Rotation.from_quat(ego_history_quat[-1])  # t0 时刻朝向
t0_rot_inv = t0_rot.inv()

ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
ego_future_xyz_local  = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
```

**为什么要在局部坐标系？** 模型学的是"自车在当前朝向下的相对运动模式"。如果直接用世界坐标，同一个路口不同行驶方向对应完全不同的 (x,y) 数值，模型要额外学一个"朝向"的隐式编码。局部坐标让所有训练样本的"直行"都对应 `x ≈ 正方向，y ≈ 0`，大大降低学习难度。

数据输出张量形状：

| 张量 | 形状 | 说明 |
|------|------|------|
| `image_frames` | (4, 4, 3, H, W) | 4相机 × 4帧（t-3 ~ t0），按 cam index 排序 |
| `ego_history_xyz` | (1, 1, 16, 3) | 历史 1.6s @10Hz |
| `ego_future_xyz` | (1, 1, 64, 3) | 未来 6.4s @10Hz |
| `relative_timestamps` | (4, 4) | 各帧相对时间的秒数 |

4 个相机按固定顺序排列：`[cross_left, front_wide, cross_right, front_tele]`（即索引 [0, 1, 2, 6]）。

### 2.3 helper.py：对话消息构造

```python
def create_message(frames: torch.Tensor):
    num_traj_token = 48  # 历史轨迹 placeholder 数量
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    return [
        {"role": "system", "content": [{"type": "text",
            "text": "You are a driving assistant..."}]},
        {"role": "user", "content":
            [{"type": "image", "image": frame} for frame in frames] +
            [{"type": "text",
              "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning..."]},
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]
```

**关键设计**：历史轨迹用 `<|traj_history|>` 占位 token（48 个），推理时通过 `base_model.py` 的 `fuse_traj_tokens()` 方法替换为实际轨迹的离散 token。这保持了 VLM 的"视觉+文本"原生输入格式，同时让轨迹信息以离散 token 的方式融入自回归生成。

---

## 三、核心模型：AlpamayoR1 三组件架构

### 3.1 模型初始化

```python
class AlpamayoR1(ReasoningVLA):
    def __init__(self, config, pretrained_modules=None, original_vocab_size=None):
        super().__init__(config, pretrained_modules, original_vocab_size)

        # 组件1: Expert — 轻量 Transformer，只处理动作 token
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        # ... 应用 expert_cfg 覆盖
        self.expert = AutoModel.from_config(expert_config)
        del self.expert.embed_tokens  # Expert 不独立 embed，复用 VLM 的

        # 组件2: Action Space — 连续动作的物理表示
        self.action_space = hyu.instantiate(config.action_space_cfg)

        # 组件3: Diffusion — Flow Matching 采样器
        self.diffusion = hyu.instantiate(config.diffusion_cfg,
                                          x_dims=self.action_space.get_action_space_dims())

        # 动作投影模块: action_in_proj / action_out_proj
        self.action_in_proj = hyu.instantiate(config.action_in_proj_cfg, ...)
        self.action_out_proj = hyu.instantiate(config.action_out_proj_cfg, ...)
```

**三组件各司其职：**

| 组件 | 类名 | 输入 | 输出 | 权重规模 |
|------|------|------|------|---------|
| **VLM** (背板) | `Qwen3VLForConditionalGeneration` | 图像+文本token | CoC 文本 + KV cache | ~8B |
| **Expert** (动作专家) | `AutoModel.from_config(text_config)` | 去噪中的动作 token 嵌入 | 预测的速度场 v | ~2B |
| **Diffusion** (采样器) | `FlowMatching` | 随机噪声 + step_fn | 去噪后的动作 | 0（无参） |

"Expert" 是三类组件中最巧妙的设计。它直接从 VLM 的 `text_config` 实例化（与 VLM 的 LLM backbone 结构一致），但**删除了 embed_tokens**（不独立做 embedding，由 action_in_proj 投影后的向量直接当输入）。这样 Expert 可以重用 VLM 的 hidden_size 和注意力模式，却能独立处理动作序列的去噪迭代。

### 3.2 推理采样：VLM rollout + Expert 去噪

`sample_trajectories_from_data_with_vlm_rollout` 是整个推理管线的核心方法，分为**两个阶段**：

```python
def sample_trajectories_from_data_with_vlm_rollout(self, data, ...):
    # === 阶段1: VLM 自回归生成 CoC 推理链 ===
    input_ids = data["tokenized_data"]["input_ids"]
    traj_data_vlm = {"ego_history_xyz": ..., "ego_history_rot": ...}
    input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)

    vlm_outputs = self.vlm.generate(
        input_ids=input_ids,
        generation_config=...,
        stopping_criteria=StopAfterEOS(eos_token_id=traj_future_start_id),
        logits_processor=ExpertLogitsProcessor(...),  # mask 掉离散轨迹 token
    )

    # === 阶段2: Expert + Diffusion 去噪生成轨迹 ===
    def step_fn(x, t):
        # 把噪声 latent 投影为 Expert token 嵌入
        future_token_embeds = self.action_in_proj(x, t)
        # Expert 在 KV cache 上做前向，只处理未来 token
        expert_out = self.expert(inputs_embeds=future_token_embeds,
                                  past_key_values=prompt_cache, ...)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        pred = self.action_out_proj(last_hidden)
        return pred  # 预测的速度场 v

    sampled_action = self.diffusion.sample(batch_size=..., step_fn=step_fn, ...)

    # 解码: (accel, kappa) → (xyz, yaw)
    pred_xyz, pred_rot = self.action_space.action_to_traj(sampled_action, ...)
    return pred_xyz, pred_rot
```

**ExpertLogitsProcessor 的作用：** 在 VLM 自回归阶段，离散轨迹 token（`<i0>` ~ `<i767>`）的 logit 被 mask 为 `-inf`，强制 VLM 只生成文本 token。这个设计保证了：**CoC 推理链是纯文本，不与轨迹 token 混淆**。

**KV cache 共享：** VLM 生成的 KV cache 直接传给 Expert 的 forward。这意味着 Expert 可以"看到"VLM 生成的图像特征和 CoC 文本上下文，而不需要重新编码——这正是"推理链约束轨迹生成"的实现机制。

### 3.3 TrajectoryFusionMixin：轨迹 token 注入

```python
def fuse_traj_tokens(self, input_ids, traj_data):
    if traj_data is None: return input_ids
    hist_idx = tokenize_history_trajectory(self.hist_traj_tokenizer, traj_data,
                                            self.hist_token_start_idx)
    input_ids = replace_pad_token(input_ids, hist_idx,
                                   self.config.traj_token_ids["history"])
    return input_ids
```

`tokenize_history_trajectory` 将历史轨迹 `(xyz, rot)` 编码为离散 token，替换掉占位的 `<|traj_history|>`。编码器用的是 `DeltaTrajectoryTokenizer`——对 Δxyz 做均匀量化，`num_bins=1000`，16 个历史点 × 3 维 = 48 个 token。

---

## 四、Action Space：单轮动力学动作表示

这是 AlpaMayo-R1 **最巧妙的工程设计**之一。它避免了直接回归 waypoint (x,y)，转而用**加速度 + 曲率**作为动作表示。

### 4.1 为什么用 (accel, kappa) 而不是 (x, y)

| 动作表示 | 优点 | 缺点 |
|---------|------|------|
| 原始 waypoint (x,y) | 直观、无信息损失 | 对传感器噪声敏感、收敛慢、轨迹可能不物理 |
| **加速度+曲率 (accel, kappa)** | 物理合理、平滑性好 | 需要额外解码步骤 |
| 航向角+速度 (theta, v) | 常用 | 积分漂移问题 |

AlpaMayo-R1 的 `UnicycleAccelCurvatureActionSpace` 用单轮动力学模型做正反向映射：

```python
class UnicycleAccelCurvatureActionSpace(ActionSpace):
    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)  # (64, 2): 每步 (accel, kappa)

    def action_to_traj(self, action, traj_history_xyz, traj_history_rot, t0_states=None):
        accel, kappa = action[..., 0], action[..., 1]
        # 反标准化
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        # 从历史状态估计 t0 时刻的速度
        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        v0 = t0_states["v"]

        # 单轮动力学积分: (accel, kappa) → (x, y, theta)
        dt = self.dt  # = 0.1s
        velocity = v0 + cumsum(accel * dt)
        theta   = cumsum(kappa * velocity * dt)    # 航向角
        x       = cumsum(velocity * cos(theta) * dt)  # x 坐标
        y       = cumsum(velocity * sin(theta) * dt)  # y 坐标
        return traj_future_xyz, traj_future_rot
```

**核心映射流程：**

```
动作空间: (accel, kappa)  ← Normalized, 模型预测的是这个
    ↓ 反标准化
物理量:  加速度 ≈ [-9.8, +9.8] m/s², 曲率 ≈ [-0.2, +0.2] m⁻¹
    ↓ 单轮动力学积分
轨迹:    (x, y, yaw)  ← 64 个 waypoint @10Hz
```

### 4.2 反向映射：轨迹 → 动作 (训练用)

```python
def traj_to_action(self, traj_history_xyz, traj_history_rot,
                   traj_future_xyz, traj_future_rot):
    # 从连续轨迹点中求解控制量
    dxy = diff(full_xy)               # 位置差分 → 速度估计
    v = dxy_theta_to_v(dxy, theta, v0)  # 用正则化最小二乘求解速度
    accel = finite_diff(v)            # 速度差分 → 加速度
    kappa = dtheta / (v * dt)          # 航向变化率 → 曲率
    return normalize(accel, kappa)     # 标准化到模型输出范围
```

这个反向映射在 SFT 第一阶段（动作模态注入）使用——把 GT 轨迹转成 (accel, kappa) 做监督学习。

### 4.3 离散轨迹 Tokenizer

```python
class DiscreteTrajectoryTokenizer:
    def encode(self, ...):
        action = self.action_space.traj_to_action(hist, rot, fut_xyz, fut_rot)
        # 均匀量化到 num_bins 个离散值
        action = (action - dims_min) / (dims_max - dims_min) * (num_bins - 1)
        return action.round().long().reshape(batch_size, -1)

    def decode(self, hist_xyz, hist_rot, tokens, ...):
        action = tokens.float() / (num_bins - 1)
        action = action * (dims_max - dims_min) + dims_min
        return self.action_space.action_to_traj(action, hist_xyz, hist_rot)
```

`num_bins=1000`，每条轨迹 64 步 × 2 维 = **128 个离散 token**。这些 token 和文本 token 一起在 VLM 的自回归过程中生成（但被 ExpertLogitsProcessor mask 掉，实际上由 Expert 处理）。

---

## 五、Flow Matching 扩散解码器

### 5.1 FlowMatching 采样

```python
class FlowMatching(BaseDiffusion):
    def sample(self, batch_size, step_fn, device,
               inference_step=10, ...):
        x = torch.randn(batch_size, *self.x_dims, device=device)  # 高斯噪声
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)

        for i in range(inference_step):
            dt = time_steps[i+1] - time_steps[i]
            v = step_fn(x=x, t=time_steps[i])  # 模型预测速度场
            x = x + dt * v                     # Euler 积分
        return x  # 去噪后的动作 (accel, kappa)
```

**为什么是 Flow Matching 而不是 DDPM？** Flow Matching 的采样轨迹是**直线**（从噪声到数据的直线路径），DDPM 是弯曲 SDE。直线路径意味着：

| 维度 | DDPM | Flow Matching |
|------|------|--------------|
| 生成路径 | 弯曲 SDE 反向链 | 直线 ODE |
| 步数需求 | 20~1000 步 | **10 步即可** |
| 采样方式 | Langevin + 噪声预测 | **Euler 积分** |
| 时间步含义 | 噪声调度 | **线性插值系数** |

这就是 `num_inference_steps=10` 的含义——从噪声到数据的 10 步直线插值。

### 5.2 训练数据构造

```python
def construct_training_data(self, x):
    # beta 分布采样时间步（偏重于高噪声端）
    t = self.beta_dist.sample((batch_size,))
    t = self.beta_scale_constant - t * self.beta_scale_constant

    noise = torch.randn_like(x)
    noisy_x = t * x + (1 - t) * noise  # 线性插值噪声
    return {"noisy_x": noisy_x, "timesteps": t, "noise": noise, "x": x}

def compute_loss_from_pred(self, training_data, pred):
    x = training_data["x"]
    noise = training_data["noise"]
    target = x - noise  # 速度场目标
    return F.mse_loss(target, pred)
```

时间步采样用的是 `Beta(1.5, 1.0)` 分布，偏重于高噪声端（靠近 1 的 t），这鼓励模型在接近数据时做更精准的预测。

---

## 六、动作投影模块：从 latent 到 token

```python
class PerWaypointActionInProjV2(nn.Module):
    def __init__(self, in_dims, out_dim, num_enc_layers=4, hidden_size=1024, ...):
        # 每个动作维度独立做 Fourier 编码
        self.sinus = nn.ModuleList([
            FourierEncoderV2(dim=num_fourier_feats, max_freq=100.0)
            for _ in range(in_dims[-1])  # 2: accel, kappa
        ])
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=100.0)
        # MLP 编码器把 Fourier 特征映射到 Expert 的 hidden_size
        self.encoder = MLPEncoder(num_input_feats=..., out_dim=out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, timesteps):
        # 对每个动作维度做 Fourier 编码
        action_feats = [s(x[:, :, i]) for i, s in enumerate(self.sinus)]
        # 时间步也做 Fourier 编码
        timestep_feats = self.timestep_fourier_encoder(timesteps)
        # 拼接 → MLP → LayerNorm
        x = torch.cat([*action_feats, timestep_feats], dim=-1)
        return self.norm(self.encoder(x))
```

**FourierEncoderV2** 是关键组件——它用**对数间隔的频率**做正弦/余弦编码：

```python
class FourierEncoderV2(nn.Module):
    def __init__(self, dim, max_freq=100.0):
        freqs = torch.logspace(0, math.log10(max_freq), steps=dim // 2)
        self.freqs = freqs[None, :]

    def forward(self, x):
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
```

| 输入 | Fourier 编码输入 | 输出 |
|------|-----------------|------|
| `x[..., 0]` (accel) | → 20 维 Fourier 特征 | |
| `x[..., 1]` (kappa) | → 20 维 Fourier 特征 | → 拼接 → 40 维 |
| `timesteps` | → 20 维 Fourier 特征 | → 总计~60 维 → MLP(1024) |
| | | → 投影到 Expert hidden_size |

Fourier 编码的作用：**把连续值映射到高频空间**。直觉上，神经网络很难直接处理一个 `[0, 1]` 小数的细微变化，但 Fourier 编码把 `x=0.5` 变成 `[sin(ω₁·0.5), cos(ω₁·0.5), ...]` 这样一组高频信号，网络更容易捕捉到数值变化。

---

## 七、基础知识：模型配置系统

### 7.1 config.py 与 hydra 注入

```python
class AlpamayoR1Config(ReasoningVLAConfig):
    def __init__(self,
                 diffusion_cfg=None,        # → hyu.instantiate 创建 FlowMatching
                 action_space_cfg=None,     # → hyu.instantiate 创建 UnicycleAccelCurvatureActionSpace
                 action_in_proj_cfg=None,   # → hyu.instantiate 创建 PerWaypointActionInProjV2
                 action_out_proj_cfg=None,  # → nn.Linear
                 expert_cfg=None,           # → 覆盖 text_config 属性
                 expert_non_causal_attention=True,
                 keep_same_dtype=True,
                 ...):
```

所有组件都是**可插拔**的，通过 config dict 配置。例如 action_space 从 YAML 解析后的 dict 注入：

```python
# 配置示例（简化）
action_space_cfg = {
    "_target_": "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace",
    "accel_mean": 0.0, "accel_std": 1.0,
    "curvature_mean": 0.0, "curvature_std": 1.0,
    "dt": 0.1, "n_waypoints": 64,
}
```

### 7.2 模型注册机制

```python
class AlpamayoR1(ReasoningVLA):
    config_class = AlpamayoR1Config
    base_model_prefix = "vlm"

AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
```

通过 HuggingFace 的 `AutoModel.register` 机制注册自定义模型，使 `AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B")` 可以直接加载 HF hub 上的权重。这是 22 GB 权重能一键加载的技术基础。

---

## 八、数据流总览：从输入到输出的完整路径

```
用户输入: clip_id + t0_us[5.1s]
    │
    ▼
load_physical_aiavdataset()
    ├── 加载 4 相机 × 4 帧图像         → image_frames (4,4,3,H,W)
    ├── 加载自车历史轨迹 (1.6s)         → ego_history_xyz (1,1,16,3)
    ├── 加载自车未来轨迹 (6.4s)         → ego_future_xyz  (1,1,64,3)
    └── 坐标变换: 世界系 → 自车 t0 局部系
    │
    ▼
helper.create_message()
    └── 构造 [system, user, assistant] 对话模板
    │
    ▼
processor.apply_chat_template()
    └── 图像 → ViT patch embedding; 文本 → token ID
    │
    ▼
AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout()
    │
    ├── Stage 1: VLM 自回归生成
    │   ├── fuse_traj_tokens(): 历史轨迹 → 离散 token → 替换 placeholder
    │   ├── vlm.generate() with ExpertLogitsProcessor
    │   │   ├── 图像 token + 历史轨迹 token + 文本 prompt
    │   │   ├── → 自回归生成 CoC 推理链文本
    │   │   ├── → 遇到 <|traj_future_start|> 停止
    │   │   └── → 得到: 推理链文本 + KV cache (含图像&文本上下文)
    │   │
    │   └── 构建 Expert 输入
    │       ├── position_ids 对齐 (去除填充, 加上 rope_delta)
    │       └── attention_mask 只关注 KV cache 的有效部分
    │
    └── Stage 2: Expert + Diffusion 去噪
        ├── init: x = randn(B, 64, 2)  # 高斯噪声 (accel, kappa)
        │
        └── for t in [0.0, 0.1, ..., 1.0] × 10 steps:
            ├── action_in_proj(x, t)
            │   ├── Fourier 编码 (accel_i, kappa_i, t)
            │   └── MLP → Expert hidden_size
            ├── expert(inputs_embeds, past_key_values=KV_cache)
            │   └── Expert 在 VLM 上下文上做前向 → 预测速度场 v
            ├── action_out_proj(last_hidden)  # hidden → (accel, kappa)
            └── x = x + dt * v  # Euler 积分
            │
            ▼
        sampled_action (accel, kappa) → (64, 2)
        │
        ▼
    action_space.action_to_traj()
        ├── 反标准化: normed_accel → 物理 accel ±9.8
        ├── 单轮动力学积分: (accel, kappa) → (x, y, yaw)
        └── 输出: pred_xyz (64,3) + pred_rot (64,3,3)
        │
        ▼
    额外输出: extra["cot"] = CoC 推理链文本
```

---

## 九、总结：记住这五点

1. **三组件架构**：VLM（8B 背板）生成 CoC 推理链 + KV cache；Expert（2B 动作专家）在 KV cache 上迭代去噪；Diffusion（Flow Matching）用 10 步 Euler 积分从噪声生成动作。
2. **动作空间设计**：用单轮动力学模型的 (accel, kappa) 替代直接回归 (x,y)，且通过标准化让模型输出的数值范围可控（均 0 方 1），数值稳定性好。
3. **推理分两阶段**：VLM 先生成纯文本推理链（离散轨迹 token 被 mask），然后 Expert 利用 VLM 的 KV cache 做扩散去噪——推理链和轨迹生成完全解耦但上下文共享。
4. **Fourier 编码 + MLP 投影**：把连续的 (accel, kappa, t) 映射到高频特征空间再投影到 Expert 的 hidden_size，让神经网络更容易捕捉数值变化。
5. **模块化配置**：所有组件通过 hydra + config dict 注入，action_space、diffusion、expert_cfg 都可以独立替换不修改模型代码。这是后续 SFT 和 RL 训练复用的基础。

---

*📖 知识点拆解系列。本文基于 [github.com/NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) commit 17 个（main 分支）的代码分析。SFT 和 RL 后训练代码在 [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes)。*
