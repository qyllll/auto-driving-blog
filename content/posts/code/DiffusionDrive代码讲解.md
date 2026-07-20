---
title: "代码讲解：DiffusionDrive 从架构到训练推理的完整闭环"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["DiffusionDrive", "扩散规划", "端到端自动驾驶", "代码讲解", "NAVSIM", "Truncated Diffusion"]
summary: "逐行拆解 hustvl/DiffusionDrive 的真实源码：截断扩散策略如何把去噪步数从几十步压到 2 步、anchor 高斯如何替代标准高斯、训练与推理如何复用 navsim 官方管线。一篇把架构→训练→推理闭环讲清楚的工程向代码讲解。"
---

## 为什么要讲 DiffusionDrive 的代码

DiffusionDrive（CVPR 2025，hustvl）是端到端规划里**扩散策略**路线的开创者，也是 NAVSIM 排行榜上扩散类方法的基线。它最被人津津乐道的工程 trick 是：**把标准扩散从几十步去噪压到 2~8 步，还能天然支持多模态轨迹**。这篇不讲公式推导，直接看源码——仓库 `hustvl/DiffusionDrive` 把整套 navsim 内联了，DiffusionDrive 自身代码全部在 `navsim/agents/diffusiondrive/` 下。

> **一句话结论**：DiffusionDrive 的本质 = ResNet 主干 + 20 个 k-means anchor 轨迹 + 从 anchor 加"截断噪声"的轻量扩散解码器；训练从 anchor 加噪、推理从 anchor 去噪 2 步，最后用分类头 argmax 选一条轨迹。

## 项目结构：它如何挂在 navsim 上

仓库是 navsim 的一个 vendored fork，DiffusionDrive 只在 `navsim/agents/diffusiondrive/` 下新增自己的 agent，训练/推理入口全部复用 navsim 官方脚本：

```
navsim/agents/diffusiondrive/
├── transfuser_agent.py        # Agent（继承 AbstractAgent）
├── transfuser_model_v2.py     # ★ 主模型 + 截断扩散策略 TrajectoryHead
├── transfuser_backbone.py     # 图像/LiDAR 主干（TransFuser 融合）
├── transfuser_features.py     # 特征/目标构建（环视拼接在这）
├── transfuser_loss.py         # 顶层损失聚合
├── transfuser_config.py       # 全部超参
└── modules/
    ├── multimodal_loss.py     # ★ 多模态 cls+reg 损失
    ├── conditional_unet1d.py
    └── scheduler.py
```

注意文件名沿用 `transfuser_*`，但类 `V2TransfuserModel` / `TransfuserAgent` 实际实现的是 DiffusionDrive——这是历史遗留命名。

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

## 二、训练：怎么挂上 navsim

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

### 2.1 损失函数

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

## 三、推理：怎么跑出 submission

### 3.1 复用官方提交脚本

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

### 3.2 本地算分 / 上榜

评测用官方 `run_pdm_score.py`；正式上榜把 `submission.pkl` 传 HuggingFace 官方空间（navtest / navhard 榜）。

## 闭环总结

| 阶段 | 入口 | 关键代码 |
|:----:|:------|:----------|
| 架构 | `transfuser_model_v2.py` TrajectoryHead | anchor 高斯 + 截断 timestep（训练 50 / 推理 t=8、2 步） |
| 训练 | navsim `run_training.py` + `agent=diffusiondrive_agent` | 多模态 focal+L1 损失（深监督） |
| 推理 | navsim `run_create_submission_pickle.py` | `forward_test` 去噪 2 步 + cls argmax 选轨迹 |

**一句话记住这个闭环**：训练时让 20 个 anchor 学会"覆盖所有驾驶模式 + 对每个模式做残差回归"，推理时从 anchor 加一点截断噪声、2 步去噪、分类头挑最好的那条——这就是 DiffusionDrive 又快又多模态的秘密。

## 延伸阅读

- 同系列对比：[SparseDriveV2 代码讲解](/posts/code/sparsedrivev2代码讲解/)——另一条"词汇表 + 两级评分"路线
- 榜单背景：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
