---
title: "UniAD 代码讲解：端到端自动驾驶的「全栈 Transformer」是怎么拼起来的"
date: 2026-07-20
description: "拆解 OpenDriveLab/UniAD：从 BEV 特征、TrackFormer、MapFormer、MotionFormer 到 Planner 的查询式流水线，以及两阶段训练与 nuScenes 推理"
tags: ["端到端", "BEV", "Transformer", "代码讲解"]
categories: ["代码讲解"]
summary: "「UniAD 把感知（检测/跟踪/建图/预测）和规划塞进一条共享 query 的 Transformer 流水线，本文顺着 mmdet3d_plugin 的代码讲清每个 head 的输入输出与两阶段训练套路」"
---

> 本文是「代码讲解」路线的第 4 篇（前 3 篇：DiffusionDrive、SparseDriveV2、DriveVLA-W0）。UniAD 是 BEV 端到端感知-规划一体化的开山之作（CVPR 2023 Best Paper），代码基于 **OpenDriveLab/UniAD**，构建在 mmdetection3d 之上。

## 0. 先讲清楚几个名词

- **BEV（Bird's Eye View）**：把多视角相机/激光雷达特征拍扁到一张「上帝视角」栅格图，后续所有任务都在这张图上做。
- **Query**：一组可学习向量，代表「要找的东西」（一辆车、一条车道、一个轨迹）。Transformer 用 query 去 BEV 特征里 attend 出对应内容。
- **PETR / 可变形注意力（Deformable Attention）**：query 不在全图做注意力，而是只在预测的几个参考点附近采样，省算力。
- **两阶段训练**：Stage-1 训感知（检测/跟踪/建图/预测），Stage-2 冻结感知、只训规划，避免规划梯度把感知带崩。
- **OccFormer / 占用栅格**：预测每个 BEV 网格未来是否被占据，给规划做「前方有没有障碍」的显式监督。

## 1. 仓库边界：代码都在哪

UniAD 不是从零写的新框架，而是往 **mmdetection3d** 里塞了一个 `mmdet3d_plugin`。真正属于 UniAD 的新代码几乎都在插件目录里：

- `UniAD/`
  - `projects/configs/uniad/`：所有实验配置
    - `uniad_base_e2e.yaml`：端到端（两阶段都训）
    - `uniad_base_track_map.yaml`：只训感知
    - `uniad_base.sh`：启动脚本
  - `mmdet3d_plugin/uniad/`
    - `dense_heads/`
      - `motion_head.py`：MotionFormer，轨迹预测
      - `occ_head.py`：OccFormer，占用预测
      - `plan_head.py`：Planner，规划
    - `tracks/`：TrackFormer，多目标跟踪
    - `backbone/`：BEV 编码器 + 地图头
  - `mmdet3d_plugin/models/detr3d/`：改造版 DETR3D（检测）
  - `mmdet3d_plugin/datasets/`：nuScenes 数据管线
  - `tools/`：`train.py` / `test.py`
  - `ckpts/`：预训练权重放这里

要点：你不会在仓库里找到 `model.py` 这种「总装文件」。**总装发生在 config 里**——yaml 把各个 head 像积木一样拼进 `UniADTrack` / `UniADPlanning` 模型。

## 2. 整体数据流：一条共享 query 的流水线

- 6 路相机图像
  - backbone：EfficientNet / Swin + FPN，得到多视角特征
  - BEVFormer 的 spatial cross-attn 把多视角特征压成 BEV 特征图（H×W×C，例如 200×200×256）
- TrackFormer：输出跟踪后的 agent query（N_agent × C）
- MapFormer：输出地图 query（车道/路口）
- MotionFormer：接收 agent query + map query 相互交互，每个 agent 解码出 K 条未来轨迹 + 置信度
- OccFormer：用 agent 轨迹 + BEV 预测占用栅格
- Planner：用 ego query attend BEV + agent/motion 信息，输出自车未来 T 步坐标

关键设计：下游任务**直接消费上游的 query**，而不是重新检测一遍。TrackFormer 输出的 agent query 直接喂给 MotionFormer，MotionFormer 输出的轨迹直接喂给 Planner。这就是「全栈共享表征」的意思。

## 3. 逐文件拆解

### 3.1 BEV 编码器 backbone

UniAD 复用 **BEVFormer** 的 BEV 特征提取。核心是 `BEVFormerEncoder`，用可变形注意力把多视角相机特征压成 BEV：

```python
# 简化自 BEVFormer：用相机参数把 query 投影到图像平面采样
bev_queries = self.bev_embedding  # (H*W, C)
for lid in range(num_layers):
    bev_queries = self.layers[lid](
        bev_queries,
        key=img_feats,                # 多视角特征
        value=img_feats,
        reference_points=self.reference_points,  # 预定义 BEV 网格
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
bev_feat = bev_queries.view(B, H, W, C).permute(0, 3, 1, 2)
```

`reference_points` 是 BEV 网格对应的真实世界坐标，靠相机内外参投影到图片上做可变形采样。这一步不 UniAD 独有，但它把 BEV 质量直接决定了后面所有任务的 ceiling。

### 3.2 TrackFormer（跟踪）

在检测 head 基础上加 track query：每一帧新检测 + 上一帧的 track query 一起做注意力，输出稳定 ID。

```python
# dense_heads 里 detect + track 融合
det_query = self.embeddings(query)        # 当前帧检测 query
track_query = prev_track_query            # 上一帧带 ID 的 query
all_query = torch.cat([det_query, track_query], dim=0)
hs = self.decoder(all_query, bev_feat)    # 交互
det_out, track_out = split(hs)
# track_out 与 det_out 用相似度匹配，赋予同一 ID
```

### 3.3 MapFormer（矢量化地图）

地图也用 query 表示：每个 query 解码出一条车道线的点序列（lane polyline），而不是语义分割那种稠密栅格。

```python
map_query = self.map_embedding.weight     # (N_map, C)
map_hs = self.map_decoder(map_query, bev_feat)
map_coords = self.map_reg_head(map_hs)    # → (N_map, P, 2) 每条车道 P 个点
```

### 3.4 MotionFormer（运动预测，核心）

每个 agent query 解码出 K 条候选轨迹 + 每条的置信度（softmax 归一化）。注意它**双向交互**：agent↔agent、agent↔map。

```python
# dense_heads/motion_head.py 核心前向（简化）
def forward(self, agent_query, map_query, bev_feat):
    # agent-env 交互：agent 看 BEV
    agent_query = self.agent_bev_cross(agent_query, bev_feat)
    # agent-agent 交互 + agent-map 交互
    for layer in self.motion_layers:
        agent_query = layer(agent_query, map_query)
    # 每个 agent 输出 K 条轨迹
    traj = self.traj_head(agent_query)     # (N_agent, K, T, 2)
    score = self.score_head(agent_query)   # (N_agent, K)  softmax 后置信度
    return traj, score
```

### 3.5 OccFormer（占用预测）

给定 agent 轨迹，把它「未来位置」投影回 BEV，再和 BEV 特征做注意力，预测每个栅格未来是否被占。它给规划提供**显式障碍信号**。

```python
# dense_heads/occ_head.py
def forward(self, bev_feat, traj):
    # 把轨迹采样成 BEV 上的位置，作为 query
    occ_query = sample_traj_to_bev(traj)        # (N_agent*T, C)
    occ = self.occ_decoder(occ_query, bev_feat) # 每个位置是否被占
    return occ
```

### 3.6 Planner（规划，最后一步）

ego 用一个可学习 query，attend BEV + agent 轨迹 + 占用，输出自车轨迹。训练时用两阶段：Stage-2 冻结前面所有 head，只反向传播 planner 和 ego query。

```python
# dense_heads/plan_head.py
def forward(self, ego_query, bev_feat, motion_info):
    ego_query = self.ego_decoder(ego_query, bev_feat, motion_info)
    plan = self.plan_reg_head(ego_query)        # (T, 2) 自车轨迹
    return plan
```

## 4. 训练：两阶段为什么这么设计

UniAD 总损失是各任务之和：

```text
L = L_det + L_track + L_map + L_motion + L_occ + L_plan
```

但规划任务梯度太强、直接端到端会破坏感知，所以：

- **Stage-1**（`uniad_base_track_map.yaml`）：只开感知头，训检测/跟踪/建图/预测，得到稳定 BEV 和 query 表征。
- **Stage-2**（`uniad_base_e2e.yaml`）：freeze 感知，加载 Stage-1 权重，只训 Planner + OccFormer。

```bash
# Stage-1
bash tools/dist_train.sh projects/configs/uniad/uniad_base_track_map.yaml 8
# Stage-2
bash tools/dist_train.sh projects/configs/uniad/uniad_base_e2e.yaml 8 \
    --cfg-options load_from=ckpts/uniad_stage1.pth
```

## 5. 推理：怎么跑起来看轨迹

```bash
# 端到端评测（需要 Stage-2 权重）
bash tools/dist_test.sh projects/configs/uniad/uniad_base_e2e.yaml \
    ckpts/uniad_e2e.pth 8 --eval bbox
```

推理时整条流水线一次性前向：图像 → BEV → 跟踪 → 地图 → 运动 → 占用 → 规划，最后 `plan_head` 输出的 (T,2) 就是自车未来轨迹，直接用于闭环评测（UniAD 配了 `nuscenes` 的 `planning` metric 算 L2 位移和碰撞率）。

## 6. 代码里值得抄的设计

- **query 作为通用接口**：所有任务输入输出都是 query，天然可拼、可冻结、可迁移。这是后续 VAD / SparseDrive 都沿用的思路。
- **显式中间监督**：跟踪、地图、运动、占用每一个都带 loss，等于给规划喂了多个「老师」，缓解端到端难训。
- **代价是重**：6 个 head 全 Transformer，训练和显存都很吃力——这正是后面 VAD（轻量化）、SparseDrive（稀疏化）要解决的问题。

## 7. 和本系列其他文章的关系

- 想看「query 思路」如何被**轻量化** → 见 VAD 代码讲解（去掉占用/分割、用向量化表示）。
- 想看「端到端」如何走向**生成式扩散** → 见 DiffusionDrive。
- 想看「端到端」如何走向 **VLA + 世界模型** → 见 DriveVLA-W0。

---

> 个人思考：UniAD 证明了「一个共享 query 空间里把感知规划串起来」可行且 SOTA，但它重、且规划仍是回归式（直接出轨迹点）。后面两条演进路线——轻量化（VAD）和生成式（DiffusionDrive）——本质上都是在补 UniAD 的两个短板。
