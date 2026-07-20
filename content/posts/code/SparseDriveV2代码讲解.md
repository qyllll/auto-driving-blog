---
title: "代码讲解：SparseDriveV2 因子化轨迹词汇表与两级评分的完整闭环"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["SparseDriveV2", "端到端自动驾驶", "代码讲解", "NAVSIM", "因子化词汇表", "两级评分"]
summary: "逐行拆解 swc-17/SparseDriveV2 的真实源码：从架构图与全局数据流出发，讲清几何路径×速度剖面如何因子化组合成 26 万条轨迹词汇表、两级评分怎么把候选从 26 万粗筛到 200 条再精排、PDM score 蒸馏损失如何用排行榜判分器当老师。一篇把项目挂载方式→数据流→前向传播→训练梯度→推理选轨迹→个人思考讲透的工程向代码讲解。"
---

## 为什么要讲 SparseDriveV2 的代码

SparseDriveV2（swc-17）是 NAVSIM 排行榜上 Scoring-based 路线的代表之一（navtest 官方榜 92.0 PDMS）。它和 DiffusionDrive 思路完全不同：**不扩散、不生成，而是预先建一个巨大的"轨迹词汇表"，再学打分器挑最好的**。它的核心创新是**因子化轨迹词汇表（path × velocity）**和**两级评分（coarse-to-fine）**。这篇直接看真实源码。

> **一句话结论**：SparseDriveV2 = 冻结的 path(1024) × velocity(256) 组合词汇表（26 万条轨迹）+ 两级评分（先按 path/vel 粗筛到 200 条，再用 PDM 蒸馏的 metric head 精排 argmax）。它把"规划"变成了一个"从超大离散候选集里检索最优"的问题。

## 架构总览：先看地图

SparseDriveV2 是**纯规划（planning-only）**模型，没有检测 head。它的推理链路极短：图像过轻量 backbone → 特征送两层 transformer decoder 当"打分器" → 在一个**预先冻结的 26 万条轨迹字典**上打分 → 两级筛选后取最优。下面这张图（SparseDrive 系列论文原图）概括了整体范式：

![SparseDrive 架构总览](/images/sparsedrive/architecture_overview.png)

为了把"代码里数据怎么流"讲清楚，我自己画了一张**数据流图**，把论文图和真实源码对应起来：

```
                输入 (navsim AgentInput)
  ┌───────────────────────────────────────────────┐
  │ 多视角相机 cam_l0/cam_f0/cam_r0 (3×H×W×3)     │
  │ + ego 状态 (vx, vy, yaw, heading, ...)         │
  └───────────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ SparseBackbone (ResNet-34+FPN)   │  冻结? 否（可训）
        │ → 多尺度特征 [B, N_cam, C, h, w] │
        └─────────────────┬───────────────┘
                          │ deformable attention
                          ▼
        ┌─────────────────────────────────┐
        │ CustomTransformerDecoder (2 层)  │  打分器
        │  query = 冻结轨迹词汇表嵌入       │
        └─────────────────┬───────────────┘
                          │
        ┌─────────────────┴───────────────────┐
        │ 第一级(粗)：path/vel 各自 top-k       │  → 筛到 ~200 条
        │ 第二级(细)：组合 traj 用 metric_head  │  → PDMS 式打分
        └─────────────────┬───────────────────┘
                          │ argmax
                          ▼
            输出轨迹 (4s, 8 poses, x/y/θ)
```

> **关键认知**：模型本身**不生成轨迹**，只给一个固定的候选集打分。这与 DiffusionDrive（生成轨迹）形成镜像——SparseDriveV2 把"规划"彻底退化成了一个"检索+排序"问题。这也是它推理极快、且天然契合 NAVSIM 评分逻辑的原因。

## 项目结构：它如何挂在 navsim 上（先厘清边界）

这里要先说清一个容易混淆的点：**SparseDriveV2 是 navsim 的一个 fork，但"挂在 navsim 上"分两层，不要混为一谈**。

- **新增的模型代码**（作者写的部分）：全部在 `navsim/agents/sparsedrive/` 下，作为 navsim 的一个 `Agent` 插件接入。
- **离线的词汇表构建脚本**（一次性工具，不在推理/训练主链路里）：在仓库根目录的 `scripts/cluster/` 下，与 `navsim/` 平级，不是 `sparsedrive/` 的子目录。
- **训练 / 评测入口**：完全复用 navsim 官方的 `run_training.py`、`run_create_submission_pickle.py`，通过 Hydra 的 `agent=sparsedrive_agent` 把自研模型"挂"上去。

```
# 仓库根目录结构（只画与本文相关的部分）
navsim/                              # ← fork 自 autonomousvision/navsim（官方管线，基本不动）
├── agents/
│   └── sparsedrive/                # ★ 作者新增的模型代码（挂在 navsim 上的"插件"）
│       ├── sparsedrive_agent.py      # Agent（继承 AbstractAgent）
│       ├── sparsedrive_model.py      # 模型 + 加载冻结词汇表
│       ├── sparsedrive_backbone.py   # ResNet-34 主干
│       ├── custom_decoder.py         # ★ 两级评分 + 组合 + 损失
│       └── scorer/get_pdm_score_v2.py# score 蒸馏用的 PDM 打分
└── planning/script/run_training.py  # ← 复用官方训练入口（不修改）

scripts/cluster/cluster_anchor.py    # ★ 离线构建 path/velocity 词汇表（与 navsim/ 平级，一次性工具）
```

> **一句话澄清挂载关系**：`run_training.py`（官方）→ 通过配置 `agent=sparsedrive_agent` 实例化 `SparseDriveAgent`（作者）→ agent 内部调用 `SparseDriveModel` + `CustomTransformerDecoder`。词汇表由离线脚本预生成后 `.npz` 加载，不进训练主链路。

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

### 1.2 词汇表构建（离线 KMeans）：这是 26 万条的来源

这一节回答"那 26 万条轨迹到底哪来的"。答案：**离线预聚类，不是模型学的**。

- **path 候选**：对训练集所有轨迹的"形状"（忽略速度，只看几何路径）做 KMeans，`K_PATH = 1024`；
- **velocity 候选**：对所有轨迹的"速度剖面"做 KMeans，`K_VELOCITY = 256`；

```python
# scripts/cluster/cluster_anchor.py
path_cluster = KMeans(n_clusters=K_PATH).fit(paths_flatten).cluster_centers_
path_cluster = path_cluster.reshape(K_PATH, num_pts, 3)
velocity_cluster = KMeans(n_clusters=K_VELOCITY).fit(velocities).cluster_centers_
```

**组合**：遍历 `i in K_PATH × j in K_VELOCITY`，把第 j 个速度剖面沿第 i 条路径的弧长积分插值，得到 `(1024, 256, 8, 3)` 的轨迹张量（8 个 pose，每个 x/y/θ）：

```python
# scripts/cluster/cluster_anchor.py
trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 3))
trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))
# ... interp by cumsum(velocity*DT) along path arc-length ...
np.savez(f"{CKPT_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.npz",
         trajectory=trajectory, trajectory_mask=trajectory_mask)
```

**Vocab 大小 = 1024 × 256 = 262,144 条组合轨迹**（README 所谓"32× denser"，是相对于 Hydra-MDP 的 8K 词表）。运行时加载：

```python
# sparsedrive_model.py:73
self.path_vocab = nn.Parameter(torch.from_numpy(np.load(config.path_anchor)).float(), requires_grad=False)
self.vel_vocab  = nn.Parameter(torch.from_numpy(np.load(config.velocity_anchor)).float(), requires_grad=False)
trajectory_data = np.load(config.trajectory_anchor)
self.traj_vocab = nn.Parameter(torch.from_numpy(trajectory_data["trajectory"]).float(), requires_grad=False)
self.traj_mask  = nn.Parameter(torch.from_numpy(trajectory_data["trajectory_mask"]).float(), requires_grad=False)
```

> **为什么因子化？** 直接枚举 26 万条完整轨迹当固定 query 太浪费参数；拆成 path(往哪走) × velocity(多快走) 两个因子，组合时只需把两个嵌入相加即可指数级覆盖动作空间，vocab 尺寸只是两者之和。这是"在动作空间结构上做归纳偏置"。

### 1.3 两级评分（Two-Level Scoring）：26 万 → 200 → 1

实现于 `custom_decoder.py` 的 `CustomTransformerDecoderLayer`。这一节是全文核心。

**第一级（粗粒度，因子化）**：在 decoder 的每一层，分别对 path 与 velocity 打分，各自 top-k 后同步过滤组合词汇表。比如第 0 层保留 64 条 path、第 1 层保留 20 条 velocity（配置 `path_filter_num=[64, 20]`、`velocity_filter_num=[64, 20]`）：

```python
# custom_decoder.py:167-211
topk_path_scores, topk_path_indices = torch.topk(path_scores, self._config.path_filter_num[self.decoder_idx], dim=1)
filter_traj_vocab = torch.gather(filter_traj_vocab, 1, topk_path_indices[:, :, None, None, None].expand(...))
# velocity 同理
topk_vel_scores, topk_vel_indices = torch.topk(vel_scores, self._config.velocity_filter_num[self.decoder_idx], dim=1)
filter_traj_vocab = torch.gather(filter_traj_vocab, 2, topk_vel_indices[:, :, :, None, None].expand(...))
```

经过两层粗筛，组合候选从 26 万压到 `64 × 20 = 1280`（或接近 200，取决于配置）条。

**组合在 decoder 中的实现**（最后一层）：path 嵌入 ⊕ velocity 嵌入逐元素相加，展平成组合轨迹 query：

```python
# custom_decoder.py:215
traj_emed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
traj_emed = traj_emed.flatten(1, 2)
```

**第二级（细粒度，组合）**：仅最后一层，对粗筛后的组合轨迹用多个 metric head 打分（每个 PDMS 指标一个 head）：

```python
# custom_decoder.py:214, 231
if self.decoder_idx == self.decoder_num_layers - 1:
    traj_scores = self.traj_mlp(...)   # 组合评分（每个 metric 一个 head，见 custom_decoder.py:144-150）
```

### 1.4 主干与多视角图像

Backbone 是 **ResNet-34**（timm）+ FPN，注意**未复用**原 SparseDrive 的 Swin 编码器，这里是轻量 ResNet-34。多视角默认 `cam_l0/cam_f0/cam_r0` 三路，forward 时压平送 backbone 再 reshape 回相机维度，供 deformable attention 按内外参采样：

```python
# sparsedrive_backbone.py:47-69
img = img.flatten(end_dim=1)   # [B, N_cam, 3, H, W] -> [B*N_cam, 3, H, W]
feat = self.backbone(img)
feat = feat.reshape(B, N_cam, C, h, w)
```

## 二、前向传播：一张图对应的代码路径

把上面所有模块串成一次 `forward`。输入是 navsim 的 `AgentInput`，输出是 `(B, 8, 3)` 的轨迹（8 个 pose）。完整时序：

```
compute_trajectory(agent_input)                 # AbstractAgent 接口（推理）
  └─> _build_features(agent_input)              # 拆 camera + ego status
  └─> SparseDriveModel.forward(features)
        ├─ SparseBackbone(img)  → img_feat [B, N_cam, C, h, w]
        ├─ status_encoding(ego) → status_emb    [B, d_model]
        ├─ 冻结 traj_vocab → traj_embed (path/vel 嵌入查表)
        └─ CustomTransformerDecoder(
              query  = traj_embed (26万→粗筛→~200),
              value  = img_feat (deformable attention),
              context= status_emb)
             ├─ 第0层: path/vel 各 top-k 粗筛
             ├─ 第1层: path/vel 各 top-k 粗筛 → 组合 ~200 条
             └─ 末层 : metric_head 对 ~200 条打 PDMS 式分
        └─> 返回 scores + 候选轨迹
```

**输入维度小结**：
- 图像：`[B, 3, H, W]`（三相机已 resize/拼接或分相机进 backbone）
- ego 状态：`[B, 8]`（速度 4 维 + 朝向 2 维 + 位置 2 维，经 `_status_encoding` 映射到 d_model）
- 轨迹词汇表：`traj_vocab [262144, 8, 3]`（冻结）

**输出**：`[B, 1, 8, 3]` 的最优轨迹 + 对应的 metric 分数（推理时用于排序）。

## 三、训练：怎么挂上 navsim + score 蒸馏

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

### 3.1 损失函数（梯度流向）

损失全在 `custom_decoder.py:236-288`，分三族：

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

**梯度流小结**：只有 backbone、`CustomTransformerDecoder` 内的 path/vel/metric head 参数可训；`traj_vocab` 全程冻结（仅作 query 查表）。三族损失加权求和后做反向传播，AdamW 优化。

## 四、推理：两级评分选最终轨迹

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

| 阶段 | 入口 | 关键代码 | 训练/推理差异 |
|:----:|:------|:----------|:--------------|
| 架构 | `sparsedrive_model.py` + `scripts/cluster/cluster_anchor.py` | 冻结 path(1024)×vel(256)=26万 轨迹词汇表 | 词汇表离线生成，两侧共用 |
| 训练 | navsim `run_training.py` + `agent=sparsedrive_agent` | path/vel/traj 分类 + **PDM score 蒸馏** | 额外回传 PDM 算 metric GT |
| 推理 | `custom_decoder.py:291` | 两级评分：粗筛 200 条 → PDMS 式分数 argmax | 无需 PDM，直接 metric head 打分 |

**一句话记住这个闭环**：离线用 KMeans 建好"path×velocity"的 26 万条轨迹字典，训练时让模型学会给每条打 PDM 式分数（蒸馏自排行榜判分器），推理时先按 path/vel 粗筛到 200 条、再用分数精排取最优——这就是 SparseDriveV2 "检索式规划"的全部秘密。

## 个人理解与思考

**1. 它把"规划"问题偷换成了"检索"问题。** 这是 SparseDriveV2 最聪明也最危险的一点。聪明在于：排行榜（PDMS）本身就是在一个固定候选集上打分选优，SparseDriveV2 直接让模型学这个打分函数，训练目标与评测指标**完美对齐**，所以榜上分数高毫不意外。危险在于：如果最优轨迹恰好不在 26 万条字典里（极端场景、字典覆盖盲区），模型**无能为力**——它只能选"字典里最好的"，不能"生成字典外更好的"。这是离散候选集方法的天花板。

**2. 因子化是性价比极高的归纳偏置。** 26 万条组合轨迹只用了 1024+256 两个聚类中心就覆盖，参数几乎不增。相比 Hydra-MDP 直接枚举 8K 固定轨迹，SparseDriveV2 用相近成本拿到了 32× 的密度。这提醒我们：在动作空间有明显"几何 × 速度"结构时，因子化分解通常比扁平枚举好得多。

**3. 两级评分是"精度-算力"的务实妥协。** 直接对 26 万条做细粒度 metric 打分显存爆炸；粗筛先砍到 200 条再精排，把计算量压了三个数量级。这是工程上很典型的"先召回后排序"两阶段架构（和推荐系统如出一辙）。缺点是粗筛用 path/vel 独立打分，可能误杀"path 一般但 vel 绝配"的组合——不过实践中 200 的保留量足够兜底。

**4. 和 DiffusionDrive 的对比给人的启发。** DiffusionDrive 是"生成式"——从 anchor 扩散出轨迹，擅长覆盖字典外的多模态；SparseDriveV2 是"检索式"——在巨字典里选，擅长贴合评测指标且推理极快（无迭代去噪）。**没有谁绝对更好**：要榜单分数 → 检索式更稳；要开放场景泛化 → 生成式更灵活。这也是 NAVSIM 榜单上两条路线并存的底层原因。

## 延伸阅读

- 同系列对比：[DiffusionDrive 代码讲解](/posts/code/diffusiondrive代码讲解/)——扩散生成式路线
- 榜单背景：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
