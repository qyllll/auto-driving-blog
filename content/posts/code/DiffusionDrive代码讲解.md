---
title: "代码讲解：DiffusionDrive 从架构到训练推理的完整闭环"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["DiffusionDrive", "扩散规划", "端到端自动驾驶", "代码讲解", "NAVSIM", "Truncated Diffusion"]
summary: "逐行拆解 hustvl/DiffusionDrive 的真实源码：从架构图与全局数据流出发，讲清截断扩散策略如何把去噪步数从几十步压到 2 步、anchor 高斯如何替代标准高斯、训练与推理如何复用 navsim 官方管线。一篇把项目挂载方式→数据流→前向传播→训练梯度→推理选轨迹→个人思考讲透的工程向代码讲解。"
---

## 为什么要讲 DiffusionDrive 的代码

DiffusionDrive（CVPR 2025，hustvl）是端到端规划里**扩散策略**路线的开创者，也是 NAVSIM 排行榜上扩散类方法的基线。它最被人津津乐道的工程 trick 是：**把标准扩散从几十步去噪压到 2~8 步，还能天然支持多模态轨迹**。这篇不讲公式推导，直接看源码——仓库 `hustvl/DiffusionDrive` 把整套 navsim 内联了，DiffusionDrive 自身代码全部在 `navsim/agents/diffusiondrive/` 下。

> **一句话结论**：DiffusionDrive 的本质 = ResNet 主干 + 20 个 k-means anchor 轨迹 + 从 anchor 加"截断噪声"的轻量扩散解码器；训练从 anchor 加噪、推理从 anchor 去噪 2 步，最后用分类头 argmax 选一条轨迹。

## 架构总览：先看地图

DiffusionDrive 是**生成式规划**：它不从固定候选集里选，而是**从 anchor 出发、用扩散逐步"雕"出一条轨迹**。下面这张图（DiffusionDrive 论文原图）给出了整体范式：

![DiffusionDrive 架构总览](/images/diffusiondrive/architecture.png)

为把源码里的数据流向讲清楚，我画了一张**数据流图**：

```
                输入 (navsim AgentInput)
  ┌───────────────────────────────────────────────┐
  │ 多视角相机 cam_l0/cam_f0/cam_r0 → 拼接全景图    │
  │ + LiDAR（BEV 特征，可选）                      │
  │ + ego 状态 / 高精度地图元素                     │
  └───────────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ TransFuser 融合主干              │  图像 ResNet-34
        │ (img + LiDAR BEV → 统一特征)     │  LiDAR ResNet-34
        └─────────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ TrajectoryHead (扩散解码器)      │  ★ 核心
        │  query = 20 个冻结 anchor 轨迹    │
        │  denoiser = 2 层 CustomTransformer│
        └─────────────────┬───────────────┘
            ┌─────────────┴──────────────┐
            │ 训练: anchor+噪声 → 回归 GT  │
            │ 推理: anchor+截断噪声→2步去噪 │
            └─────────────┬──────────────┘
                          │ 分类头 argmax
                          ▼
            输出轨迹 (4s, 8 poses, x/y/θ)
```

> **关键认知**：和 SparseDriveV2（检索式）相反，DiffusionDrive 是**生成式**——它能在 20 个 anchor 附近"生成"字典外的轨迹，泛化更灵活；代价是推理要走去噪循环（虽已压到 2 步）。两篇对照读，能看清 NAVSIM 榜单上"生成 vs 检索"两条路线的分野。

## 项目结构：它如何挂在 navsim 上（先厘清边界）

和 SparseDriveV2 一样，DiffusionDrive 也是 navsim 的一个 fork，挂载关系分两层：

- **新增的模型代码**（作者写的部分）：全部在 `navsim/agents/diffusiondrive/` 下，作为 navsim 的一个 `Agent` 插件接入。
- **训练 / 评测入口**：完全复用 navsim 官方的 `run_training.py`、`run_create_submission_pickle.py`，通过 Hydra 的 `agent=diffusiondrive_agent` 把自研模型"挂"上去。
- **官方管线文件**（fork 内联，基本不动）：`navsim/planning/script/run_training.py` 等。

```
navsim/                                  # ← fork 自 autonomousvision/navsim（官方管线，内联）
├── agents/diffusiondrive/               # ★ 作者新增的模型代码（挂在 navsim 上的"插件"）
│   ├── transfuser_agent.py        # Agent（继承 AbstractAgent）
│   ├── transfuser_model_v2.py     # ★ 主模型 + 截断扩散策略 TrajectoryHead
│   ├── transfuser_backbone.py     # 图像/LiDAR 主干（TransFuser 融合）
│   ├── transfuser_features.py     # 特征/目标构建（环视拼接在这）
│   ├── transfuser_loss.py         # 顶层损失聚合
│   ├── transfuser_config.py       # 全部超参
│   └── modules/
│       ├── multimodal_loss.py     # ★ 多模态 cls+reg 损失
│       ├── conditional_unet1d.py
│       └── scheduler.py
└── planning/script/run_training.py  # ← 复用官方训练入口（不修改）
```

> **注意**：文件名沿用 `transfuser_*`，但类 `V2TransfuserModel` / `TransfuserAgent` 实际实现的是 DiffusionDrive——这是历史遗留命名（从 TransFuser 改过来的），读源码时别被名字误导。挂载关系一句话：**`run_training.py`（官方）→ `agent=diffusiondrive_agent` → `TransfuserAgent` → `V2TransfuserModel`（内含扩散 TrajectoryHead）**。

## 一、架构：截断扩散到底截在哪

### 1.1 主干与环视图像处理

图像 encoder 用 **ResNet-34**（LiDAR 分支也是 resnet34，通过 `timm` 加载）：

```python
# transfuser_config.py:16-17
image_architecture: str = "resnet34"
lidar_architecture: str = "resnet34"
```

关键在**多视角环视的处理方式**——不是逐相机编码，而是把左/前/右三路裁剪后横向拼成一张 4:1 全景图再送 encoder：

```python
# transfuser_features.py:62-73
cameras = agent_input.cameras[-1]
# Crop to ensure 4:1 aspect ratio
l0 = cameras.cam_l0.image[28:-28, 416:-416]
f0 = cameras.cam_f0.image[28:-28]
r0 = cameras.cam_r0.image[28:-28, 416:-416]
# stitch l0, f0, r0 images
stitched_image = np.concatenate([l0, f0, r0], axis=1)
resized_image = cv2.resize(stitched_image, (1024, 256))   # 1024x256 全景
tensor_image = transforms.ToTensor()(resized_image)
```

> **工程要点**：把多相机拼成单张全景图，省掉了逐相机独立 backbone + 跨相机融合的复杂度，代价是相机间几何关系被"压扁"进 2D 拼接，依赖后续 transformer 自己学回来。

### 1.2 截断扩散策略（TrajectoryHead）—— 本文核心

实现类 `TrajectoryHead`（`transfuser_model_v2.py:382-558`），使用 `diffusers` 的 `DDIMScheduler`。

**anchor 定义**：20 个 k-means 聚类锚轨迹（预生成 `kmeans_navsim_traj_20.npy`），作为不可训练参数。这 20 个 anchor 对应 20 种驾驶模式（直行/左转/跟车…）：

```python
# transfuser_model_v2.py:400-412
self.diffusion_scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample",
)
plan_anchor = np.load(plan_anchor_path)
self.plan_anchor = nn.Parameter(
    torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False,
)  # 20,8,2
```

**训练：从 anchor 加"截断噪声"**。标准扩散从标准高斯 `N(0,1)` 出发、timestep 取满 1000；DiffusionDrive 改成从 anchor 出发、timestep **截断到 50**：

```python
# transfuser_model_v2.py:462-476
# 1. add truncated noise to the plan anchor
plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
odo_info_fut = self.norm_odo(plan_anchor)
timesteps = torch.randint(0, 50, (bs,), device=device)          # 截断到 50
noise = torch.randn(odo_info_fut.shape, device=device)
noisy_traj_points = self.diffusion_scheduler.add_noise(
    original_samples=odo_info_fut, noise=noise, timesteps=timesteps,
).float()
noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
noisy_traj_points = self.denorm_odo(noisy_traj_points)
```

**推理：只去噪 2 步**，初始样本 = anchor 加固定截断时刻 `t=8` 的噪声（而非纯高斯）：

```python
# transfuser_model_v2.py:504-520
def forward_test(...):
    step_num = 2                                # 去噪步数 = 2
    self.diffusion_scheduler.set_timesteps(1000, device)
    step_ratio = 20 / step_num
    roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1] ...
    # 1. add truncated noise to the plan anchor
    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
    img = self.norm_odo(plan_anchor)
    noise = torch.randn(img.shape, device=device)
    trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8   # 截断起点 t=8
    img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
```

为什么"从 anchor 出发 + 截断 timestep"能大幅提速？因为 anchor 已经把轨迹拉到了"合理驾驶模式"附近，扩散只需要做小幅修正，不需要从纯噪声一步步生成——所以 2 步就够。这是 DiffusionDrive 相比标准 DDPM 扩散（几十步）的**核心加速来源**。

去噪网络是 `CustomTransformerDecoder`（2 层，非标准 UNet1D），每层做：轨迹点在 BEV 上 grid_sample 交叉注意力 → 与 agent/ego query 交叉注意力 → 时间步 FiLM 调制 → 回归 offset：

```python
# transfuser_model_v2.py:341（残差回归形式）
poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
```

### 1.3 多模态轨迹表示

20 个 anchor 一一对应 20 个模式（`ego_fut_mode = 20`），每个模式回归 `num_poses×3 (x,y,θ)`：

```python
# transfuser_model_v2.py:182-228
plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
```

**没有独立 scorer 模块**；选择由分类头 `plan_cls`（每模式一个置信度）承担，取 argmax 即最终轨迹：

```python
# transfuser_model_v2.py:555-557（推理）
mode_idx = poses_cls.argmax(dim=-1)
mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
```

## 二、前向传播：一张图对应的代码路径

把上面所有模块串成一次 `forward`。输入是 navsim 的 `AgentInput`，输出是 `(B, 8, 3)` 的轨迹（8 个 pose）。完整时序：

```
compute_trajectory(agent_input)                 # AbstractAgent 接口（推理）
  └─> _build_features(agent_input)              # 相机拼接 + LiDAR/BEV + ego/map
  └─> V2TransfuserModel.forward(features)
        ├─ TransFuser 主干(img, lidar) → fused_feat
        ├─ 构造 20 个 anchor query (冻结)
        └─ TrajectoryHead.forward / forward_test
             训练: anchor + randint(0,50) 噪声 → denoiser → reg offset → GT
             推理: anchor + t=8 噪声 → 2 步 DDIM → reg offset → 20 模式
        └─> plan_cls (20 置信度) + plan_reg (20 模式轨迹)
  └─> argmax(plan_cls) → 选 1 条
```

**输入维度小结**：
- 图像：`[B, 3, 256, 1024]`（三相机拼接后的全景图）
- LiDAR/BEV：`[B, C, H, W]`（BEV 特征，可选）
- anchor：`[20, 8, 2]`（冻结，20 模式×8 pose×x/y）
- ego/map  token：若干向量 token

**输出**：`plan_reg [B, 20, 8, 3]`（20 条候选轨迹）+ `plan_cls [B, 20]`（每条置信度），推理时 argmax 取 1 条。

## 三、训练：怎么挂上 navsim

DiffusionDrive **不写自己的训练入口**，直接复用 navsim 官方 `run_training.py`（PyTorch-Lightning），只通过 Hydra 配置 `agent=diffusiondrive_agent` 挂载自己的模型：

```bash
# docs/train_eval.md:18-27
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain split=trainval trainer.params.max_epochs=100 ...
```

agent 配置 `navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`：

```yaml
_target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent
config:
  _target_: navsim.agents.diffusiondrive.transfuser_config.TransfuserConfig
checkpoint_path: null
lr: 6e-4
```

Agent 类继承 `AbstractAgent`（`transfuser_agent.py:32`），优化器用 `AdamW + WarmupCosLR`，图像 encoder 学习率 ×0.5。

### 3.1 损失函数（梯度流向）

顶层聚合在 `transfuser_loss.py:11-53`：

```python
loss = (
    config.trajectory_weight * trajectory_loss      # 12.0
    + config.diff_loss_weight * diffusion_loss       # 20.0（当前 diffusion_loss=0）
    + config.agent_class_weight * agent_class_loss
    + config.agent_box_weight * agent_box_loss
    + config.bev_semantic_weight * bev_semantic_loss
)
```

真正的**多模态监督**在模型内部逐 decoder 层计算（深监督），本体在 `modules/multimodal_loss.py`：用 GT 轨迹到 20 个 anchor 的 L2 距离找最近模式作为分类标签 → focal loss 分类 + 对该模式做 L1 回归：

```python
# modules/multimodal_loss.py:117-163
dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
mode_idx = torch.argmin(dist, dim=-1)                    # 最近 anchor = 正模式
loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(poses_cls, target_classes_onehot, gamma=2.0, alpha=0.25)
reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)
ret_loss = loss_cls + reg_loss
```

> **注意**：仓库里 `diff_loss_weight` 路径当前 `diffusion_loss=0`——即训练时**直接监督 anchor 回归 + 分类，并不反向传播扩散重建损失**。扩散仅作为推理时的轨迹生成/修正机制。这是读源码时容易踩的坑。

**梯度流小结**：只有主干、`TrajectoryHead`（denoiser + cls/reg head）参数可训；`plan_anchor` 全程冻结。多模态 focal+L1 损失逐层加权求和后反向传播，AdamW 优化。

## 四、推理：怎么跑出 submission

### 4.1 复用官方提交脚本

DiffusionDrive 直接复用 navsim 官方 `run_create_submission_pickle.py` 生成 `submission.pkl`（未自定义）。该脚本对每个 token 调 agent 的 `compute_trajectory`：

```python
# run_create_submission_pickle.py:47-53
agent.initialize()
trajectory = agent.compute_trajectory(agent_input)
```

`compute_trajectory` 本身继承 `AbstractAgent`（`abstract_agent.py:62`），内部 build features → `forward` → 取 `predictions["trajectory"]`。真正的"采样几步 + 选轨迹"落在 `TrajectoryHead.forward_test`：

- 去噪 **2 步**（`step_num=2`），从 anchor + 截断噪声（`t=8`）出发；
- 每步 DDIM `step` 更新；
- 按分类分数 argmax 从 20 个模式选一条输出。

### 4.2 本地算分 / 上榜

评测用官方 `run_pdm_score.py`；正式上榜把 `submission.pkl` 传 HuggingFace 官方空间（navtest / navhard 榜）。

## 闭环总结

| 阶段 | 入口 | 关键代码 | 训练/推理差异 |
|:----:|:------|:----------|:--------------|
| 架构 | `transfuser_model_v2.py` TrajectoryHead | anchor 高斯 + 截断 timestep（训练 50 / 推理 t=8、2 步） | anchor 冻结，两侧共用 |
| 训练 | navsim `run_training.py` + `agent=diffusiondrive_agent` | 多模态 focal+L1 损失（深监督） | 随机 t∈[0,50) 加噪回归 |
| 推理 | navsim `run_create_submission_pickle.py` | `forward_test` 去噪 2 步 + cls argmax 选轨迹 | 固定 t=8 起，2 步去噪 |

**一句话记住这个闭环**：训练时让 20 个 anchor 学会"覆盖所有驾驶模式 + 对每个模式做残差回归"，推理时从 anchor 加一点截断噪声、2 步去噪、分类头挑最好的那条——这就是 DiffusionDrive 又快又多模态的秘密。

## 个人理解与思考

**1. "截断扩散"是对自动驾驶场景的精准定制。** 标准 DDPM 从纯噪声生成，需要几十步才能收敛；但驾驶轨迹不是"任意图像"，它有强结构（平滑、符合运动学、贴合车道）。DiffusionDrive 用 k-means anchor 把起点从"纯噪声"拉到"合理驾驶模式附近"，于是扩散只需做小幅修正——2 步足矣。这启示我们：**扩散的起点选择比步数更重要**，给模型一个好的先验，能极大压缩生成开销。

**2. 训练和推理的目标不一致，是个值得警惕的信号。** 训练时 `diffusion_loss=0`，模型实际是在做"anchor 回归 + 分类"的硬监督，扩散重建损失没参与。这意味着推理时的扩散去噪，本质上是在一个"没被扩散损失调过"的 decoder 上跑——它能 work，靠的是 anchor 先验足够强 + 残差回归 head 已经学好了。但严格说，训练/推理的分布是有 gap 的。若想把扩散用得更充分（比如支持更长 horizon、更自由的多模态），应该把扩散重建损失也打开。

**3. 全景图拼接是取舍鲜明的工程选择。** 三相机拼一张 1024×256 全景图，省掉跨相机 fusion 模块，简洁高效；但几何畸变和信息损失不可忽视（侧视相机被压扁）。在 NAVSIM 这种前视为主的评测里够用，放到需要全向感知的开放场景，可能不如真正多相机 token + cross-attention 稳健。

**4. 和 SparseDriveV2 的对比给人的启发。** SparseDriveV2 是"检索式"——在 26 万字典里选，推理无迭代、贴合榜单指标；DiffusionDrive 是"生成式"——从 anchor 扩散出轨迹，擅长覆盖字典外的多模态。两者都挂在 navsim 上、都复用官方管线，但哲学相反：**前者相信"好答案在被枚举的候选里"，后者相信"好答案该被生成出来"**。作为 VLA 方向工程师，我认为实车部署更看重 SparseDriveV2 式的可解释与确定性（输出必在可行集），而算法探索阶段 DiffusionDrive 式的生成灵活性更有想象空间。

## 延伸阅读

- 同系列对比：[SparseDriveV2 代码讲解](/posts/code/sparsedrivev2代码讲解/)——另一条"词汇表 + 两级评分"路线
- 榜单背景：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
