---
title: "UniAD 代码讲解：端到端自动驾驶的「全栈 Transformer」是怎么拼起来的"
date: 2026-07-20
description: "拆解 OpenDriveLab/UniAD：从 BEV 特征、TrackFormer、MapFormer、MotionFormer 到 Planner 的查询式流水线，以及两阶段训练与 nuScenes 推理"
tags: ["端到端", "BEV", "Transformer", "代码讲解"]
categories: ["代码讲解"]
summary: "「UniAD 把感知（检测/跟踪/建图/运动预测/占用）和规划全部塞进一条共享 query 的 Transformer 流水线，是 BEV 端到端感知-规划一体化的开山之作（CVPR 2023 Best Paper）。本文顺着 mmdet3d_plugin 的真实代码，逐个 head 讲清输入输出、forward 里到底做了什么，并拆穿两阶段训练为什么非这么干不可，零基础也能顺着数据流读懂整条链路」"
---

> 本文是「代码讲解」路线的第 4 篇（前 3 篇：DiffusionDrive 679 行、SparseDriveV2 610 行、DriveVLA-W0 436 行）。UniAD 是 BEV 端到端感知-规划一体化的开山之作（CVPR 2023 Best Paper），代码基于 **OpenDriveLab/UniAD**，构建在 mmdetection3d 之上。
>
> 本篇的目标和前几篇一样：把一篇「看着吓人」的论文代码，用大白话拆成你能跟着一行行读下去的东西。你不需要提前懂 mmdet3d，不需要懂 BEVFormer，我会从最底层的名词开始讲。读到最后你应该能回答一个问题：为什么 UniAD 敢把这么多任务堆到一个网络里，而且还能训得动？

## 写给零基础读者：读这篇之前先搞懂几个名词

我不假设你知道下面任何一个词。下面每一条都是「人话版」，能让你读后面代码时不卡壳。

- **BEV（Bird's Eye View，鸟瞰图）**：通俗讲，就是把 6 路相机拍到的画面，全部「拍扁」成一张从上往下看的城市俯视图。相当于你站在城市上空，看地面上哪里有车、哪里有车道线。后面所有任务（检测、跟踪、规划）都不在原始图片上做了，全在这张俯视图上做。好处是：不同相机的信息被统一到同一个坐标系里，规划器终于能像人一样「俯瞰全局」。

- **Query（查询向量）**：这是整个 UniAD 的灵魂概念。通俗讲，query 就是一组「可学习的提问向量」，每个向量代表「我想找的某个东西」——比如一辆车、一条车道、一个未来轨迹点。Transformer 干的事，就是用这些 query 去 BEV 特征里「attend（关注）」出对应的内容。你可以把 query 想象成填空题的空：「这里____是一辆车吗？」网络自己把空填成「是车 + 车在哪 + 车往哪开」。

- **可变形注意力（Deformable Attention）**：标准 Transformer 注意力是「每个 query 跟图上所有点都算一遍相似度」，算力爆炸。可变形注意力聪明在：它只在 query 预测的「几个参考点附近」采样少量点。相当于你找人不用扫视全场，只看你 guess 他可能在的那几个位置。BEVFormer 就是靠这个才把多视角特征压进 BEV 而不爆显存。

- **两阶段训练（Two-Stage Training）**：这是 UniAD 最容易被忽视、但最关键的工程 trick。第 1 阶段只训「感知」那几个头（检测/跟踪/建图/运动/占用），把 BEV 和 query 表征练扎实；第 2 阶段把感知全部冻结，只训最后规划 Planner。为什么要分两阶段？因为规划任务的梯度太猛，直接从零端到端一起训，规划会反过来把前面好好的感知「带崩」。先让感知长大成人，再让规划站在它肩膀上，这是最稳的套路。

- **TrackFormer / MotionFormer / MapFormer / OccFormer / Planner**：这五个就是 UniAD 流水线上的 5 个「工头」，名字里都带 Former（Transformer 的意思）。TrackFormer 负责「谁是谁」（给车稳定编号），MapFormer 负责「路长啥样」（车道线），MotionFormer 负责「别人往哪开」（多模态轨迹预测），OccFormer 负责「未来这片地会不会被占」（占用栅格），Planner 负责「我（自车）往哪开」。本文第 4 节会逐个拆。

- **nuScenes**：目前自动驾驶圈最常用的大规模公开数据集之一，1000 段 20 秒的驾驶片段，配 6 路相机 + 1 个激光雷达 + 雷达。UniAD 的训练和评测全在它上面。

- **L2 误差 / 碰撞率（Collision Rate）**：这是评测「规划好不好」的两个核心指标。L2 误差通俗讲就是「你预测的自车轨迹和真人司机开的轨迹，平均差了多少米」；碰撞率就是「你这条轨迹在未来会不会撞到别人」。这两个数字越小越好，是 UniAD 论文里反复刷的表。

- **mmdetection3d（简称 mmdet3d）**：商汤开源的 3D 检测框架。UniAD 没有从零造轮子，而是把它当底盘，往里塞插件。你后面会看到所有 UniAD 新代码都在 `mmdet3d_plugin/uniad/` 下，这就是「插件式开发」的含义。

> 一句话澄清：上面这些名词你现在记不住没关系，读正文时哪里卡住，回来翻这一段就行。UniAD 的本质，就是用「query」这根线，把上面这些原本各自为战的任务，串成一条生产线。

## 为什么要讲 UniAD 的代码

你可能会问：现在都 2026 年了，端到端方案一大把，为什么还要回过头啃 UniAD 这个「老古董」？

答案是：后面几乎所有的端到端方案，都是 UniAD 的「变种」或者「反动」。

- VAD 是把 UniAD 砍瘦（去掉占用、用矢量化）。
- SparseDrive 是把 UniAD 稀疏化（检测和预测解耦）。
- DiffusionDrive 是把 UniAD 里的「运动预测」换成扩散模型生成。
- DriveVLA 是把 UniAD 里的「规划」换成 VLA + 世界模型。

换句话说，你搞懂了 UniAD 的 query 流水线，后面这些文章的「祖坟」你就挖明白了。而且 UniAD 的代码组织特别适合初学者——它把「感知-规划」拆成 5 个清清楚楚的 head，每个 head 输入输出都很干净，不像有些方案全糊在一个大模型里。

> 一句话结论：UniAD 是「query-based 端到端」这一脉的祖宗，读懂它的代码，等于拿到了后面一整条技术线的入场券。

## 架构总览：先看地图

先看全局数据流，心里有张地图再钻细节。下面用纯文字描述，不引用任何图片（本仓库里 UniAD 的图你不一定有，但文字链路已经足够清晰）。

整条流水线从上到下是这样的：

- 输入：6 路环视相机图像（前/后/左前/右前/左后/右后）。
- backbone：EfficientNet 或 Swin Transformer 做图像编码，再接 FPN 出多尺度特征。这一步产出「多视角图像特征」。
- BEV 编码器（BEVFormer）：用可变形注意力，把多视角特征「抬」成一张 BEV 特征图，尺寸常见 200×200×256（高×宽×通道）。这是后面所有头的「公共画布」。
- TrackFormer：拿 BEV 特征 + 上一帧的 track query，输出「带稳定 ID 的 agent query」，也就是「这一帧里有哪些车/人，且和上一帧对得上号」。
- MapFormer：拿 BEV 特征，输出「地图 query」，也就是车道线、路口的矢量化表示。
- MotionFormer：拿 agent query + map query，让它们互相交互，每个 agent 解码出 K 条未来轨迹 + 每条的置信度（比如 K=6，代表「这车可能直行、可能左转……」6 种走法）。
- OccFormer：拿 agent 的未来轨迹 + BEV 特征，预测「未来每个 BEV 栅格会不会被占据」，给规划器一个显式的「前方有没有障碍」信号。
- Planner：拿一个专门的 ego query，去 attend BEV + agent 轨迹 + 占用信息，输出「自车未来 T 步的坐标」。

最关键的设计只有一句话：**下游任务直接消费上游的 query，而不是重新检测一遍**。TrackFormer 输出的 agent query 直接喂给 MotionFormer，MotionFormer 输出的轨迹直接喂给 Planner。这就是论文标题里「Planning-oriented」和「Unified」的真意——大家共用一套语言（query），不用各自翻译。

> 关键认知：UniAD 不是「先检测再规划」的传统级联，而是「一个共享表征空间里，感知和规划一起算」。这个区别决定了它的上限和下限，也决定了后面所有方案怎么改它。

## 项目结构：它如何挂在 mmdet3d 上（先厘清边界）

我必须先帮你厘清仓库边界，否则你一打开 UniAD 的 GitHub 会懵：怎么满屏都是 mmdet3d 的代码，UniAD 自己的在哪？

事实是：UniAD 根本没有「自己的框架」。它是 OpenDriveLab/UniAD 这个仓库，但仓库里绝大部分是 mmdetection3d 原版代码，UniAD 真正新增的东西，全在一个插件目录 `mmdet3d_plugin/uniad/` 里。你可以把它理解成「给 mmdet3d 打的一个大补丁」。

仓库里和本文相关的目录长这样（用 Markdown 列表，不用框线）：

- `UniAD/`
  - `projects/configs/uniad/`
    - `uniad_base_track_map.yaml`：Stage-1 配置，只训感知（检测/跟踪/建图/运动/占用）。
    - `uniad_base_e2e.yaml`：Stage-2 配置，端到端（冻结感知，训 Planner + OccFormer）。
    - `uniad_base.sh`：一键启动脚本，里面拼好环境变量和训练命令。
  - `mmdet3d_plugin/uniad/`
    - `dense_heads/`
      - `motion_head.py`：MotionFormer 的实现，运动预测头。
      - `occ_head.py`：OccFormer 的实现，占用预测头。
      - `plan_head.py`：Planner 的实现，规划头。
    - `tracks/`
      - 这里放 TrackFormer 相关代码（多目标跟踪，基于可变形 attention + track query）。
    - `backbone/`
      - BEV 编码器相关，以及地图相关的组件。
  - `mmdet3d_plugin/models/detr3d/`
    - 改造版 DETR3D，负责 3D 检测（感知最底层的检测头）。
  - `mmdet3d_plugin/datasets/`
    - nuScenes 的数据管线，把原始数据转成模型能吃的 batch。
  - `tools/`
    - `train.py` / `test.py` / `dist_train.sh` / `dist_test.sh` 等入口。
  - `ckpts/`
    - 放预训练权重，比如 Stage-1 训完的 `uniad_stage1.pth`。

要点，请刻进脑子：**你不会在仓库里找到一个叫 `model.py` 的「总装文件」**。总装发生在 config 的 yaml 里——yaml 把各个 head 像积木一样拼进 `UniADTrack`（感知阶段模型）和 `UniADPlanning`（端到端模型）这两个壳子里。这也就是为什么读 UniAD 必须先读 config：模型结构不是写在 Python 里的，是写在 yaml 里的。

> 一句话澄清：如果你习惯「找一个 model.py 看全局」，在 UniAD 这里会扑空。正确姿势是「先打开 `uniad_base_e2e.yaml`，看它 `model=` 下面挂了哪些组件，再逐个去 `mmdet3d_plugin/uniad/` 里找对应文件」。

## 逐文件逐函数细讲

下面进入正题。我会按数据流的先后顺序，逐个 head 讲：它接收什么、输出什么、forward 里到底干了啥。每个代码块里我故意不放一个中文字（注释也全英文），中文解释一律写在块外面，免得你和渲染器都难受。

### 区块 A：BEV 编码器 backbone（数据流的源头）

UniAD 直接复用 BEVFormer 的 BEV 特征提取，核心类是 `BEVFormerEncoder`。它干的事用一句话说就是：用相机内外参，把 BEV 网格上的每个 query，投影到 6 张图片上采样特征，再拿回来填成 BEV 图。

简化版前向逻辑如下（注意块内零中文）：

```python
# simplified from BEVFormer encoder
bev_queries = self.bev_embedding  # (H*W, C) learnable BEV grid queries
bev_pos = self.pos_embedding      # positional embedding for BEV grid

for lid in range(num_layers):
    bev_queries = self.layers[lid](
        bev_queries,
        key=img_feats,                 # multi-view image features
        value=img_feats,
        reference_points=self.reference_points,  # BEV grid in real-world coords
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        query_pos=bev_pos,
    )

bev_feat = bev_queries.view(B, H, W, C).permute(0, 3, 1, 2)
```

块外的解释：`reference_points` 是 BEV 网格对应的真实世界坐标（比如 x 从 -51.2 到 51.2 米），靠相机内外参投影到每张图片上，做可变形采样。这一步不是 UniAD 发明的，但 UniAD 把「BEV 质量」当成了整个系统的地基——BEV 糊了，后面 Track/Motion/Plan 全完蛋。所以论文里 BEV 编码器用了不少层、也用了时序融合（把上一帧 BEV 也喂进来），就是为了让这张俯视图尽量准。

> 关键认知：UniAD 把「感知质量」和「规划质量」绑死了。这既是它强的点（规划能用到很准的 BEV），也是它弱的点（BEV 一崩全盘崩）。后面 SparseDrive 之流想解耦，根源就在这。

### 区块 B：检测头 DETR3D（TrackFormer 的地基）

在讲 TrackFormer 之前必须先讲检测，因为 TrackFormer 是在检测之上「加了一层跟踪」。UniAD 用的是改造版 DETR3D（`mmdet3d_plugin/models/detr3d/`），核心思想也是 query-based：用一组 object query，在 BEV/图像特征上预测 3D 框。

```python
# simplified DETR3D detection head
obj_queries = self.query_embedding.weight  # (N_obj, C)
for layer in self.decoder_layers:
    obj_queries = layer(obj_queries, bev_feat)
bbox_pred = self.bbox_head(obj_queries)    # (N_obj, 10) 3D box params
cls_pred = self.cls_head(obj_queries)      # (N_obj, num_cls) class logits
```

块外解释：每个 object query 解码出一个 3D 框（中心、尺寸、朝向、速度）和类别。这一步出的是「当前帧有哪些 agent」，但还没有「跨帧的 ID」。TrackFormer 要解决的，就是给这些框加上「稳定编号」。

### 区块 C：TrackFormer（多目标跟踪，给车发身份证）

TrackFormer 的思路非常直觉：每一帧，除了「新检测出来的 query」，还把「上一帧那些带 ID 的 track query」一起送进同一个 decoder。decoder 让新旧 query 互相注意力，然后再用相似度把新检测和旧 track 配对，配上的就延续老 ID，没配上的新检测就发新 ID。

```python
# simplified TrackFormer forward
det_query = self.embeddings(query)       # current frame detection queries
track_query = prev_track_query           # last frame's ID-carrying queries
all_query = torch.cat([det_query, track_query], dim=0)

hs = self.decoder(all_query, bev_feat)   # interaction in one decoder

det_out, track_out = torch.split(hs, [num_det, num_track], dim=0)
# match det_out with track_out by feature similarity -> assign IDs
matched_ids = bipartite_match(det_out, track_out)
```

块外解释：`prev_track_query` 是上一个时间步保存下来的「带记忆的 query」，它已经编码了「这辆车之前的长相和运动」。新帧里只要它和某个新检测query长得像（特征相似），就认为「还是同一辆车」，ID 不变。这样连起来，就是一条稳定的跟踪轨迹。TrackFormer 的输出就是「带 ID 的 agent query」，尺寸通常是 N_agent × C，这个 query 后面直接喂给 MotionFormer。

> 一句话澄清：TrackFormer 输出的不是「框」，而是「query 向量」。框只是顺手预测的副产品。真正被下游吃掉的是那个向量——它里面编码了这辆车的所有信息。

### 区块 D：MapFormer（矢量化地图，路长啥样）

传统做法把地图做成语义分割（哪块是车道、哪块是路面）。UniAD 偏不走寻常路：它用 query 表示地图，每个 map query 解码出「一条车道线的点序列（polyline）」。这种表示叫矢量化（vectorized），好处是后续规划能直接拿几何线条算，不用在稠密栅格里爬。

```python
# simplified MapFormer forward
map_query = self.map_embedding.weight     # (N_map, C) learnable map queries
map_hs = self.map_decoder(map_query, bev_feat)

map_coords = self.map_reg_head(map_hs)    # (N_map, P, 2) each lane has P points
map_cls = self.map_cls_head(map_hs)       # (N_map,) existence / type
```

块外解释：`map_coords` 出来的是每条车道线的 P 个 (x, y) 点，连起来就是一条线。`map_cls` 判断这条线「存不存在、是哪种线」。MapFormer 输出的 map query 会直接进 MotionFormer，让「车」和「路」互相看一眼——车要知道自己旁边有左转道还是直行道，才能预测得准。

### 区块 E：MotionFormer（运动预测，整个系统的心脏）

这是 UniAD 里最精彩、也最复杂的头。它要做的事：给每个 agent，预测未来 T 步的 K 条候选轨迹，并给每条轨迹一个置信度。K 条轨迹代表「多种可能」（直行/左转/右转/停车……），置信度是「我觉得哪种最可能」。

它的精髓在于**三种交互同时做**：

- agent ↔ BEV：每个 agent 去看 BEV 特征，知道自己周围的环境。
- agent ↔ agent：车与车互相看，避免预测出「两辆车穿模」。
- agent ↔ map：车去看车道线，轨迹要贴着路走。

```python
# dense_heads/motion_head.py (heavily simplified forward)
def forward(self, agent_query, map_query, bev_feat):
    # agent-environment interaction: each agent looks at BEV
    agent_query = self.agent_bev_cross(agent_query, bev_feat)

    # agent-agent and agent-map interaction layers
    for layer in self.motion_layers:
        agent_query = layer(agent_query, map_query)

    # each agent outputs K trajectories + K confidence scores
    traj = self.traj_head(agent_query)      # (N_agent, K, T, 2)
    score = self.score_head(agent_query)    # (N_agent, K)
    score = score.softmax(dim=-1)           # normalized confidence
    return traj, score
```

块外解释：`traj_head` 吐出来的是 (N_agent, K, T, 2)，意思是「每个 agent，K 条候选，每条 T 个时刻，每个时刻 2D 坐标」。`score_head` 吐出来 K 个分数，softmax 后变成「这 K 条里每条的可信度」。训练时，用 winner-takes-all 思路：只让离真值最近的那条轨迹的 loss 生效，逼网络学会「至少有一条猜得准」。MotionFormer 输出的轨迹，一方面给 OccFormer 用，一方面给 Planner 用。

> 关键认知：MotionFormer 输出的「多模态轨迹」是给规划器的重要线索。规划器不是只看「别人现在在哪」，而是看「别人未来可能去哪」——这正是端到端比传统感知-规划级联强的地方。

### 区块 F：OccFormer（占用预测，前方有没有障碍）

OccFormer 的逻辑很妙：它把 MotionFormer 预测的每个 agent 的未来轨迹，「投影」回 BEV 栅格上，变成一堆「未来位置 query」，再拿这些 query 去 BEV 特征上做注意力，判断「这个栅格未来会不会被占据」。

```python
# dense_heads/occ_head.py (simplified)
def forward(self, bev_feat, traj):
    # sample agent future trajectories into BEV-space queries
    occ_query = sample_traj_to_bev(traj)        # (N_agent * T, C)
    occ = self.occ_decoder(occ_query, bev_feat) # per-location occupancy logits
    occ = occ.sigmoid()                          # (N_agent * T,) occupied prob
    return occ
```

块外解释：OccFormer 给规划器提供的是「显式障碍信号」——它等于直接告诉规划器「未来第 3 秒，你左前方那片地会被某辆车占住」。这比让规划器自己从 agent 轨迹里悟出来要稳得多。注意 OccFormer 在 Stage-2 是和 Planner 一起训的（Stage-1 它其实也训，但 Stage-2 会再精细调）。

### 区块 G：Planner（规划，最后一步，自车往哪开）

终于到规划。Planner 用一个专门的 ego query（自车查询），去 attend BEV + agent 轨迹 + 占用信息，输出自车未来 T 步的坐标。它不预测「别人的轨迹」，只预测「我自己的轨迹」。

```python
# dense_heads/plan_head.py (simplified)
def forward(self, ego_query, bev_feat, motion_info, occ_info):
    ego_query = self.ego_decoder(
        ego_query,
        bev_feat,
        motion_info,   # agent trajectories from MotionFormer
        occ_info,      # occupancy from OccFormer
    )
    plan = self.plan_reg_head(ego_query)        # (T, 2) ego future trajectory
    return plan
```

块外解释：`ego_query` 是一个可学习向量，相当于「代表自车的一个提问」。它经过 decoder，把 BEV 里关于自车周围环境、别人怎么走、哪里会被占的信息，全部汇聚到自己身上，最后 `plan_reg_head` 把它解码成 (T, 2) 的自车未来轨迹。这条轨迹就是 UniAD 最终的「驾驶决策」。

> 一句话澄清：Planner 在 Stage-2 训练时，前面所有 head（Track/Map/Motion/Occ）的权重是被冻住的。也就是说训练规划时，它只能「适应」已经固定的感知，不能反过来改感知。这就是两阶段训练保护感知的核心手段。

## 逐文件一句话说明（速查表）

如果你赶时间，下面是每个核心文件的「一句话职责」：

- `mmdet3d_plugin/uniad/backbone/`：提供 BEV 编码器，把图像变成 BEV 特征。
- `mmdet3d_plugin/models/detr3d/`：3D 检测头，出当前帧 agent 框。
- `mmdet3d_plugin/uniad/tracks/`：TrackFormer，给 agent 发稳定 ID。
- `mmdet3d_plugin/uniad/dense_heads/motion_head.py`：MotionFormer，多模态轨迹预测。
- `mmdet3d_plugin/uniad/dense_heads/occ_head.py`：OccFormer，未来占用栅格。
- `mmdet3d_plugin/uniad/dense_heads/plan_head.py`：Planner，自车轨迹规划。
- `projects/configs/uniad/uniad_base_track_map.yaml`：Stage-1 配置。
- `projects/configs/uniad/uniad_base_e2e.yaml`：Stage-2 配置。
- `projects/configs/uniad/uniad_base.sh`：启动脚本。

## 训练：两阶段为什么非这么干不可

前面提了好几次两阶段，这里把账算清楚。UniAD 的总损失是各个任务损失之和：

```text
L_total = L_det + L_track + L_map + L_motion + L_occ + L_plan
```

每个 L 对应一个 head 的监督信号。看起来「一起训不就完了」，但实操里直接端到端一起训会翻车，原因有三：

- 规划任务的梯度方向，和感知任务的梯度方向经常打架。规划想「让轨迹贴合真值」，可能会诱导 BEV 特征往「方便规划」的方向偏，而不是往「真实准确」的方向偏。
- 感知头（尤其 Track/Motion）本身就不容易训稳，再叠一个更难的规划，梯度噪声会把感知带崩。
- 显存和算力：6 个 Transformer head 全开反向传播，单卡根本扛不住，两阶段能省不少训稳成本。

所以 UniAD 的实际训练流程是：

- **Stage-1（感知）**：用 `uniad_base_track_map.yaml`，只开感知相关 head（检测/跟踪/建图/运动/占用），把 BEV 和 query 表征练到稳。这一步产出 `ckpts/uniad_stage1.pth`。
- **Stage-2（端到端）**：用 `uniad_base_e2e.yaml`，加载 Stage-1 权重，冻结感知，只训 Planner + OccFormer。这一步产出最终的端到端权重。

启动命令长这样（块内零中文）：

```bash
# Stage-1: train perception only
bash tools/dist_train.sh projects/configs/uniad/uniad_base_track_map.yaml 8

# Stage-2: load stage-1 weights, train planner + occ
bash tools/dist_train.sh projects/configs/uniad/uniad_base_e2e.yaml 8 \
    --cfg-options load_from=ckpts/uniad_stage1.pth
```

> 关键认知：两阶段不是论文里的摆设，是能训动的必要条件。很多复现失败，都是因为想偷懒一步到位端到端，结果感知崩了、规划也崩了。

## 推理：怎么跑起来看轨迹

训完之后，推理是一条龙前向，不需要分阶段。整条流水线一次性过：图像 → BEV → 跟踪 → 地图 → 运动 → 占用 → 规划。最后 `plan_head` 吐出的 (T, 2) 就是自车未来轨迹，拿去和 nuScenes 里真人的轨迹比，算 L2 误差和碰撞率。

评测命令：

```bash
# end-to-end evaluation on nuScenes planning metric
bash tools/dist_test.sh projects/configs/uniad/uniad_base_e2e.yaml \
    ckpts/uniad_e2e.pth 8 --eval planning
```

块外解释：`--eval planning` 会触发 UniAD 配好的 planning metric，它把预测轨迹和真值轨迹对齐（用「以当前位置为原点」的坐标系），算平均 L2 位移，再算预测轨迹在 future 各时刻和周围 agent 框的重叠率得到碰撞率。这两个数字就是论文表格里刷的那俩。

## 代码里值得抄的设计

讲完流程，说点「学了能用在自己项目里」的东西。

- **query 作为通用接口**：所有任务的输入输出都是 query 向量。这带来三个好处——可拼接（head 像积木）、可冻结（Stage-2 冻感知）、可迁移（换个数据集只改 head）。这条思路被后面 VAD、SparseDrive 全盘继承。
- **显式中间监督**：跟踪、地图、运动、占用每一个都带 loss，等于给规划喂了 4 个「老师」。端到端最难的是梯度信号太稀疏，UniAD 用一堆中间监督把信号铺满，缓解难训。
- **感知规划解耦训练**：两阶段本质上是「先让感知毕业，再让规划上学」，工程上极稳。
- **代价是重**：6 个 head 全 Transformer，训练和显存都很吃力。这恰恰是后面 VAD（轻量化）、SparseDrive（稀疏化）、DiffusionDrive（生成式降本）要去补的短板。

## 个人思考

UniAD 最大的贡献，不是某个模块多新颖，而是它证明了「一个共享 query 空间里，把感知和规划串起来端到端训」这条路走得通，而且能刷 SOTA。它把「端到端自动驾驶」从「说说而已」变成了「真的能跑、真的能比级联强」。

但它有两个绕不开的短板：一是重，二是规划仍然是「回归式」——直接出轨迹点，没有建模「不确定性」和「多可能性」。这俩短板，正好对应了后面两条演进路线：轻量化（VAD 砍模块、SparseDrive 稀疏化）解决「重」；生成式（DiffusionDrive 用扩散建模多模态）解决「回归式表达力不足」。

所以读 UniAD，我建议你带着一个问题往下读后面几篇：**「这个方案，是在补 UniAD 的哪个短板？」** 一旦你建立这个视角，整个端到端自动驾驶的演进史就串成一条线了。

## 和本系列其他文章的关系

- 想看「query 思路」如何被**轻量化** → 见 SparseDriveV2 代码讲解（检测和预测解耦、稀疏化）。
- 想看「端到端」如何走向**生成式扩散** → 见 DiffusionDrive 代码讲解（用扩散模型替换回归式运动预测）。
- 想看「端到端」如何走向 **VLA + 世界模型** → 见 DriveVLA-W0 代码讲解。
- 想看「矢量化地图 + 轻量规划」→ 后续可补 VAD 一篇。

> 一句话收尾：UniAD 是这条线的「原点」，读懂它，后面三篇你都能秒懂一半。建议按「UniAD → SparseDriveV2 → DiffusionDrive → DriveVLA-W0」的顺序读，技术演进一目了然。
