---
title: "代码讲解：SparseDriveV2 因子化轨迹词汇表与两级评分的完整闭环"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["SparseDriveV2", "端到端自动驾驶", "代码讲解", "NAVSIM", "因子化词汇表", "两级评分"]
summary: "逐行拆解 swc-17/SparseDriveV2 的真实源码：几何路径×速度剖面如何因子化组合成 26 万条轨迹词汇表、两级评分怎么把候选从 26 万粗筛到 200 条再精排、score 蒸馏损失如何用 PDM 当老师。一篇把架构→训练→推理闭环讲清楚的工程向代码讲解。"
---

## 为什么要讲 SparseDriveV2 的代码

SparseDriveV2（swc-17）是 NAVSIM 排行榜上 Scoring-based 路线的代表之一（navtest 官方榜 92.0 PDMS）。它和 DiffusionDrive 思路完全不同：**不扩散、不生成，而是预先建一个巨大的"轨迹词汇表"，再学打分器挑最好的**。它的核心创新是**因子化轨迹词汇表（path × velocity）**和**两级评分（coarse-to-fine）**。这篇直接看真实源码。

> **一句话结论**：SparseDriveV2 = 冻结的 path(1024) × velocity(256) 组合词汇表（26 万条轨迹）+ 两级评分（先按 path/vel 粗筛到 200 条，再用 PDM 蒸馏的 metric head 精排 argmax）。它把"规划"变成了一个"从超大离散候选集里检索最优"的问题。

## 项目结构：它如何挂在 navsim 上

SparseDriveV2 直接 fork 了 autonomousvision/navsim，核心模型代码放在 `navsim/agents/sparsedrive/`，训练/评测复用 navsim 管线：

```
navsim/agents/sparsedrive/
├── sparsedrive_agent.py      # Agent（继承 AbstractAgent）
├── sparsedrive_model.py      # 模型 + 加载冻结词汇表
├── sparsedrive_backbone.py   # ResNet-34 主干
├── custom_decoder.py         # ★ 两级评分 + 组合 + 损失
└── scorer/get_pdm_score_v2.py# score 蒸馏用的 PDM 打分
scripts/cluster/cluster_anchor.py   # ★ 离线构建 path/velocity 词汇表
```

Agent 定义：`navsim/agents/sparsedrive/sparsedrive_agent.py:22` `class SparseDriveAgent(AbstractAgent)`。

## 一、架构：因子化轨迹词汇表

### 1.1 它不是"instance query 做检测+规划"

注意一个常见误解：原版 SparseDrive 用 instance query 同时做检测和规划；**SparseDriveV2 是纯规划（planning-only）**——查询是**静态的轨迹词汇表**（path × velocity 候选），没有检测 head。模型入口：

```python
# sparsedrive_model.py:14
class SparseDriveModel:
    self._backbone = SparseBackbone(config)
    self._status_encoding = nn.Linear(4 + 2 + 2, config.d_model)
    self._trajectory_head = TrajectoryHead(
        num_poses=config.trajectory_sampling.num_poses,
        d_ffn=config.d_ffn, d_model=config.d_model, config=config)
```

`TrajectoryHead` 加载**冻结的** path/velocity/trajectory 词汇表（`requires_grad=False`），作为 query 送入 `CustomTransformerDecoder`。

### 1.2 词汇表构建（离线 KMeans）

path 候选：对训练集所有路径做 KMeans，`K_PATH = 1024`；velocity 候选：`K_VELOCITY = 256`：

```python
# scripts/cluster/cluster_anchor.py
path_cluster = KMeans(n_clusters=K_PATH).fit(paths_flatten).cluster_centers_
path_cluster = path_cluster.reshape(K_PATH, num_pts, 3)
velocity_cluster = KMeans(n_clusters=K_VELOCITY).fit(velocities).cluster_centers_
```

**组合**：遍历 `i in K_PATH × j in K_VELOCITY`，按速度积分沿路径弧长插值，得到 `(1024, 256, 8, 3)` 的轨迹张量：

```python
# scripts/cluster/cluster_anchor.py
trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 3))
trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))
# ... interp by cumsum(velocity*DT) along path arc-length ...
np.savez(f"{CKPT_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.npz",
         trajectory=trajectory, trajectory_mask=trajectory_mask)
```

**Vocab 大小 = 1024 × 256 = 262,144 条组合轨迹**（README 所谓"32× denser"）。运行时加载：

```python
# sparsedrive_model.py:73
self.path_vocab = nn.Parameter(torch.from_numpy(np.load(config.path_anchor)).float(), requires_grad=False)
self.vel_vocab  = nn.Parameter(torch.from_numpy(np.load(config.velocity_anchor)).float(), requires_grad=False)
trajectory_data = np.load(config.trajectory_anchor)
self.traj_vocab = nn.Parameter(torch.from_numpy(trajectory_data["trajectory"]).float(), requires_grad=False)
self.traj_mask  = nn.Parameter(torch.from_numpy(trajectory_data["trajectory_mask"]).float(), requires_grad=False)
```

> **为什么因子化？** 直接枚举 26 万条完整轨迹当固定 query 太浪费参数；拆成 path(往哪走) × velocity(多快走) 两个因子，组合时只需把两个嵌入相加即可指数级覆盖动作空间，vocab 尺寸只是两者之和。这是"在动作空间结构上做归纳偏置"。

### 1.3 两级评分（Two-Level Scoring）

实现于 `custom_decoder.py` 的 `CustomTransformerDecoderLayer`。

**第一级（粗粒度，因子化）**：分别对 path 与 velocity 打分，各自 top-k 后同步过滤组合词汇表：

```python
# custom_decoder.py:167-211
topk_path_scores, topk_path_indices = torch.topk(path_scores, self._config.path_filter_num[self.decoder_idx], dim=1)
filter_traj_vocab = torch.gather(filter_traj_vocab, 1, topk_path_indices[:, :, None, None, None].expand(...))
# velocity 同理
topk_vel_scores, topk_vel_indices = torch.topk(vel_scores, self._config.velocity_filter_num[self.decoder_idx], dim=1)
filter_traj_vocab = torch.gather(filter_traj_vocab, 2, topk_vel_indices[:, :, :, None, None].expand(...))
```

**组合在 decoder 中的实现**（最后一层）：path 嵌入 ⊕ velocity 嵌入逐元素相加，展平成组合轨迹 query：

```python
# custom_decoder.py:215
traj_emed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
traj_emed = traj_emed.flatten(1, 2)
```

**第二级（细粒度，组合）**：仅最后一层，对粗筛后的 ~200 条组合轨迹用 metric head 打分：

```python
# custom_decoder.py:214, 231
if self.decoder_idx == self.decoder_num_layers - 1:
    traj_scores = self.traj_mlp(...)   # 组合评分
    # 每个 metric 一个 head（custom_decoder.py:144-150）
```

### 1.4 主干与多视角图像

Backbone 是 **ResNet-34**（timm）+ FPN，注意**未复用**原 SparseDrive 的 Swin 编码器，这里是轻量 ResNet-34。多视角默认 `cam_l0/cam_f0/cam_r0` 三路，forward 时压平送 backbone 再 reshape 回相机维度，供 deformable attention 按内外参采样：

```python
# sparsedrive_backbone.py:47-69
img = img.flatten(end_dim=1)   # [B, N_cam, 3, H, W] -> [B*N_cam, 3, H, W]
feat = self.backbone(img)
feat = feat.reshape(B, N_cam, C, h, w)
```

## 二、训练：怎么挂上 navsim + score 蒸馏

训练入口直接复用 navsim `run_training.py`：

```bash
# scripts/training/sparsedrive_navsimv1.sh
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name default_training \
    agent=sparsedrive_agent experiment_name=$agent \
    train_test_split=navtrain use_cache_without_dataset=True \
    cache_path=exp/data_cache_navtrain \
    dataloader.params.batch_size=16 trainer.params.max_epochs=10 \
    agent.lr=0.0001 +agent.config.dataset_version=v1 \
    +agent.config.metrics=[...] +agent.config.velocity_filter_num=[64,20]
```

注意它用 `use_cache_without_dataset=True`（走 `CacheOnlyDataset`，直接按 `train_logs` 索引 cache），`batch_size=16`、`max_epochs=10`——比 navtrain 全量训练轻很多。

### 2.1 损失函数（含 score 蒸馏损失）

损失全在 `custom_decoder.py:236-288`：

- **Path 分类损失**（soft target，对最近 GT path 做 softmax 距离）：
  ```python
  # custom_decoder.py:242-252
  dist = (path_vocab - target_path[:, None])[..., :2].pow(2).sum(-1) * mask
  dist = dist.sum(-1) / valid_cnt * path_sigmas * len_path
  path_loss = F.cross_entropy(path_scores, (-dist).softmax(1))
  ```
- **Velocity 分类损失**、**组合轨迹分类损失（IMI imitation）** 同理。

- **Score 蒸馏损失（核心）**：用 **PDM 模拟器**对组合轨迹打分当 GT，训练每个 metric head（BCE with logits）：
  ```python
  # custom_decoder.py:270-288
  sub_scores = get_pdm_score_v2(trajectory, pdm_token_paths)   # 调 PDM 算 PDMS 分项
  for metric in self._config.metrics:
      metric_gt = torch.tensor(np.stack([s[metric] for s in sub_scores]))
      metric_gt[metric_gt == 0.5] = 0.0
      metric_loss = F.binary_cross_entropy_with_logits(metric_pred, metric_gt)
      loss_dict[f'{metric}_loss_{self.decoder_idx}'] = metric_loss * metric_loss_weight
  ```
  PDM 打分函数在 `scorer/get_pdm_score_v2.py`，权重 `metric_loss_weight=5.0`。

> **这是 SparseDriveV2 的杀手锏**：它直接拿排行榜的判分函数（PDM）当监督信号，让 metric head 学会"什么轨迹分高"——推理时就能用这些 head 合成 PDMS 式分数来排序，等于"在训练时就把排行榜逻辑学进去"。

## 三、推理：两级评分选最终轨迹

`compute_trajectory` 继承 `AbstractAgent`，真正"选轨迹"逻辑在 `custom_decoder.py:291-317`（推理最后一层）：

```python
# custom_decoder.py:291-317
scores = (
    metric_logit["no_at_fault_collisions"].sigmoid() *
    metric_logit["drivable_area_compliance"].sigmoid() *
    metric_logit["driving_direction_compliance"].sigmoid() *
    metric_logit["traffic_light_compliance"].sigmoid()
) * (
    5 * metric_logit["time_to_collision_within_bound"].sigmoid() +
    5 * metric_logit["ego_progress"].sigmoid() +
    2 * metric_logit["lane_keeping"].sigmoid() +
    2 * metric_logit["history_comfort"].sigmoid()
)
bs_indices  = torch.arange(scores.shape[0], device=scores.device)
mode_indices = scores.argmax(1)                       # 第二级细粒度分的 argmax
trajectory = filter_traj_vocab.flatten(1, 2)[bs_indices, mode_indices]
output["trajectory"] = trajectory
```

流程和 PDMS 公式**同构**：乘性项（NC/DAC/DDC/TLC 的 sigmoid 乘积）× 加权项（TTC/EP/LK/HC 加权求和）。最终 argmax 从粗筛后的 ~200 条里选 1 条。

> **注意**：生成 challenge `submission.pkl` 的脚本（仓库引用的 `run_create_submission_pickle_challenge.py`）在本 fork 内未包含，需依赖官方 navsim devkit；常规 navtest 评测走内联的 `run_pdm_score_navtest_v1_fast.py`。

## 闭环总结

| 阶段 | 入口 | 关键代码 |
|:----:|:------|:----------|
| 架构 | `sparsedrive_model.py` + `cluster_anchor.py` | 冻结 path(1024)×vel(256)=26万 轨迹词汇表 |
| 训练 | navsim `run_training.py` + `agent=sparsedrive_agent` | path/vel/traj 分类 + **PDM score 蒸馏** |
| 推理 | `custom_decoder.py:291` | 两级评分：粗筛 200 条 → PDMS 式分数 argmax |

**一句话记住这个闭环**：离线用 KMeans 建好"path×velocity"的 26 万条轨迹字典，训练时让模型学会给每条打 PDM 式分数（蒸馏自排行榜判分器），推理时先按 path/vel 粗筛到 200 条、再用分数精排取最优——这就是 SparseDriveV2 "检索式规划"的全部秘密。

## 延伸阅读

- 同系列对比：[DiffusionDrive 代码讲解](/posts/code/diffusiondrive代码讲解/)——扩散生成式路线
- 榜单背景：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
