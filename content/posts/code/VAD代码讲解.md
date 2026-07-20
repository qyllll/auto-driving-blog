---
title: "VAD 代码讲解：把场景「矢量化」后，端到端能有多轻"
date: 2026-07-20
description: "拆解 hustvl/VAD：用向量化 agent/地图/运动表示替代稠密占用与分割，看 VAD/ VADv2 如何用更少 head 做端到端规划"
tags: ["端到端", "向量化", "BEV", "代码讲解"]
categories: ["代码讲解"]
summary: "「VAD 在 UniAD 之后走轻量化路线：把 agent、地图、运动全表示成向量（点和线），去掉占用栅格与分割头，VADv2 进一步把规划做成多模态概率分布」"
---

> 本文是「代码讲解」路线的第 5 篇。VAD（Vectorized Scene Representation for Efficient Autonomous Driving，ICCV 2023）来自 hustvl，是 UniAD 之后最常被拿来对比的「轻量端到端」方案。代码在 **hustvl/VAD**。

## 0. 名词速查

- **向量化表示（Vectorized）**：用「点和线」而不是「稠密栅格」描述场景。一辆车 = 一个 bounding box 向量（中心、尺寸、朝向）；一条车道 = 一串点坐标；一条轨迹 = 一串点。好处是信息密度高、不需要占用栅格/分割这种重头。
- **Planning KD（知识蒸馏）**：VAD 用「离线规划的专家轨迹」当 teacher，蒸馏给端到端模型，提升安全率。
- **多模态规划（VADv2）**：不再只出一条轨迹，而是出一堆候选轨迹 + 每条概率，推理时按概率/分数选最优。这是从「回归」走向「生成/选择」的关键一步。

## 1. 仓库结构：和 UniAD 一样是 mmdet3d 插件

- `VAD/`
  - `projects/configs/vad/`：VAD 配置（`vad_base.py`、`vad_nuscenes/`）
  - `mmdet3d_plugin/vad/`
    - `dense_heads/`
      - `vad_head.py`：总装，检测/地图/运动/规划都在这里
      - `vad_track_head.py`：跟踪（可选）
      - `vad_motion_head.py`：运动预测
    - `modules/`
      - `vectorized_map.py`：地图向量化编码器
      - `agent_cross_attn.py`：agent-地图交互
      - `planning_head.py`：规划头（VADv2 多模态）
    - `datasets/`
  - `tools/`

和 UniAD 最大区别：**VAD 没有 OccFormer / 分割头**，所有信息用向量表达，代码量小一圈。

## 2. 整体流水线

- 图像 → BEV 特征（同样用 BEVFormer 类编码器）
- Agent Head：输出 agent 向量（box + 速度/朝向）
- Map Head：输出向量化车道（点序列）
- Motion Head：每个 agent 输出 K 条轨迹向量
- Planning Head：ego query 看 BEV + agent + map + motion，输出
  - 自车轨迹（VAD）/ 多模态轨迹 + 概率（VADv2）

注意 VAD **没有显式占用预测**，agent 和地图用向量直接交互，规划头直接 consume 这些向量。

## 3. 逐模块

### 3.1 Agent Head（检测）

和 DETR3D 类似的 query 解码，但输出是**向量化的 box 参数**而非密集特征。

```python
# dense_heads/vad_head.py 中检测分支（简化）
agent_query = self.agent_embedding.weight
hs = self.agent_decoder(agent_query, bev_feat)
boxes = self.agent_reg_head(hs)     # (N, 10): x,y,z,w,l,h,yaw,vx,vy,cls
```

### 3.2 Map Head（矢量地图）

地图用「点序列 query」表示，每条车道是一组可学习点 + 一个整体 embedding。

```python
# modules/vectorized_map.py
map_query = self.map_embedding.weight     # (N_map, C)
map_hs = self.map_decoder(map_query, bev_feat)
map_points = self.map_reg_head(map_hs)    # (N_map, P, 2) 每条车道 P 个点
```

### 3.3 Motion Head（运动预测）

每个 agent query 解码 K 条轨迹向量，并且让 agent query 和 map query 互相 attend（这样轨迹会顺着车道走）。

```python
# dense_heads/vad_motion_head.py
def forward(self, agent_query, map_query, bev_feat):
    for layer in self.motion_layers:
        agent_query = layer(agent_query, map_query, bev_feat)
    traj = self.traj_head(agent_query)     # (N_agent, K, T, 2)
    return traj
```

### 3.4 Planning Head（核心差异点）

VAD 原始版输出单条规划轨迹；VADv2 改成**多模态候选 + 概率分布**：

```python
# modules/planning_head.py（VADv2 思路，简化）
def forward(self, ego_query, bev_feat, motion_info):
    ego_query = self.ego_decoder(ego_query, bev_feat, motion_info)
    # 多模态：每个 mode 一条轨迹
    trajs = self.mode_traj_head(ego_query)   # (M, T, 2)  M 个候选
    scores = self.mode_score_head(ego_query) # (M,)        softmax 概率
    return trajs, scores
```

推理时，VADv2 用各 mode 的 score 乘上「与 agent 轨迹/地图的碰撞惩罚」做后处理，选最终轨迹。这已经非常接近「生成一堆 + 选一条」的生成式思路，是后面 Diffusion Planner / DriveVLA 的近亲。

## 4. 训练：Planning KD 是关键 trick

VAD 的总损失：

```text
L = L_det + L_map + L_motion + L_plan + L_plan_kd
```

`L_plan_kd` 是用离线规划器（如 Hybrid A* / 专家轨迹）生成的「安全轨迹」做蒸馏。代码里 teacher 轨迹从数据集预生成，训练时直接算 KL / Smooth-L1。

```bash
# 训练 VAD
bash tools/dist_train.sh projects/configs/vad/vad_base.py 8
# 训练 VADv2（多模态）
bash tools/dist_train.sh projects/configs/vad/vad_v2.py 8
```

## 5. 推理

```bash
bash tools/dist_test.sh projects/configs/vad/vad_base.py ckpts/vad.pth 8
```

输出自车轨迹后，用官方 `planning` metric（L2 误差 + 碰撞率）评测。

## 6. VAD vs UniAD：代码层面的取舍

- UniAD 有 6 个 head（含占用/分割），重但监督多；VAD 砍到 4 个，靠向量化表示省掉稠密头。
- VAD 没有 OccFormer，规划头直接吃 agent/map 向量，信息更紧凑。
- VADv2 把规划变成概率多模态，是「回归式端到端 → 生成式选择」的重要过渡。
- 代价：向量化地图依赖离线地图提取质量；稠密占用带来的「显式障碍栅格」安全感没了，要靠 KD 补。

## 7. 在本系列的位置

- 轻量化端到端 → VAD（本文）。
- 共享 query 全栈 → UniAD。
- 生成式扩散规划 → DiffusionDrive / Diffusion Planner。
- 多模态概率规划 → VADv2 是雏形，Diffusion Planner 把它做成了扩散采样。

---

> 个人思考：VAD 的「向量化」本质是**用结构化先验换算力**。它证明了端到端不一定非要 UniAD 那么重的感知栈，但也暴露了回归式规划的局限性——所以 VADv2 才走向多模态概率。这一步，正好接上后面 Diffusion 系列「生成多条再选」的范式。
