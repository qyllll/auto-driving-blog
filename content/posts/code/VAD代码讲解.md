---
title: "VAD 代码讲解：把场景「矢量化」后，端到端能有多轻"
date: 2026-07-20
description: "拆解 hustvl/VAD：用向量化 agent/地图/运动表示替代稠密占用与分割，看 VAD/ VADv2 如何用更少 head 做端到端规划"
tags: ["端到端", "向量化", "BEV", "代码讲解"]
categories: ["代码讲解"]
summary: "「VAD 在 UniAD 之后走轻量化路线：把 agent、地图、运动全表示成向量（点和线），去掉占用栅格与分割头，VADv2 进一步把规划做成多模态概率分布，是回归式端到端走向生成式选择的关键过渡」"
---

> 本文是「自动驾驶代码讲解」系列的第 5 篇。前面我们已经讲了 DiffusionDrive（679 行）、SparseDriveV2（610 行）、DriveVLA-W0（436 行）。这一篇我们回到「轻量端到端」的鼻祖之一：**VAD（Vectorized Scene Representation for Efficient Autonomous Driving，ICCV 2023）**，代码来自 **hustvl/VAD**。
>
> 之所以选它，是因为它是 UniAD 之后最常被拿来对比的方案，而且它把「端到端不一定非要很重的感知栈」这件事讲得最清楚。读完这篇，你对后面 Diffusion 系列「生成多条再选一条」的思路会理解得更快。

## 写给零基础读者：读这篇之前先搞懂几个名词

下面这些名词，如果你第一次见，先别慌，我用大白话给你翻译一遍。这节是整个系列里最啰嗦也最有用的一节，建议你慢点看。

**向量化表示（Vectorized）**。这是 VAD 最招牌的概念。在自动驾驶里，我们要描述「周围有什么」：车、行人、车道线、自车轨迹……大多数早期方案用「稠密栅格（occupancy grid）」来描述，也就是把场景切成一张小格子地图，每个格子标上是空还是障碍。VAD 不这么干，它用「点和线」来描述：一辆车 = 一个 bounding box 向量（中心坐标、长宽高、朝向、速度）；一条车道 = 一串点坐标连成的折线；一条轨迹 = 一串点。好处是信息密度高、不需要占用栅格、不需要分割头，模型更轻。你可以理解为：别人用「照片」记场景，VAD 用「坐标列表」记场景。

**端到端（End-to-End）**。传统方案是流水线：先检测，再跟踪，再预测，再规划，每一步单独训、单独调。端到端是说，我让一个模型从「图像」直接出「自车要走的轨迹」，中间环节（检测、地图、运动预测）仍然有，但它们和最终的规划共享同一套特征、一起被训练，目标是让「最后规划得好」反过来带动前面都学好。

**Planning KD（知识蒸馏，Knowledge Distillation）**。KD 的意思是用一个「老师」教「学生」。在 VAD 里，老师是一个离线规划器（比如 Hybrid A* 这种传统方法）预先算出来的「安全轨迹」，学生是端到端模型。训练时除了正常的监督损失，还加一项「让学生的轨迹靠近老师的轨迹」的损失，相当于用传统方法的经验给端到端模型兜底，提升安全率。这是 VAD 一个很关键的 trick。

**多模态规划（VADv2）**。VAD 原始版只出「一条」轨迹，属于回归式。VADv2 改成出「一堆候选轨迹 + 每条的概率」，推理时按概率和碰撞惩罚选最优。这就是从「回归」走向「生成 / 选择」的关键一步，也是后面 Diffusion 系列的直接思想来源。

**BEV（Bird's Eye View，鸟瞰图）**。把多路相机图像「拍扁」到一张从上往下看的俯视图特征上。所有后续任务（检测、地图、运动、规划）都在这张俯视图上做。你可以把它想象成：六张照片被压成一张「上帝视角」的特征地图。

**query**。这是 DETR 系列（包括 BEVFormer、VAD）的核心词。query 就是一组「可学习的小向量」，模型让它去特征图里「找东西」。比如检测 query 去找车，地图 query 去找车道，ego query 去找自车该怎么走。每个 query 解码完就变成一个输出（一个 box、一条车道、一条轨迹）。

**box 参数（x, y, z, w, l, h, yaw, vx, vy, cls）**。这是检测头输出的 10 个量，记下来后面会反复用到：x/y/z 是物体中心的三维坐标；w/l/h 是宽、长、高；yaw 是朝向角（绕竖直轴的旋转）；vx/vy 是物体在 x/y 方向的速度；cls 是类别（车、行人等）。VAD 的 agent head 输出的就是这种向量化 box。

**Agent / Map / Motion / Planning 四个 Head**。这是 VAD 的四大组件，本文会逐个拆：Agent Head 出车辆向量 box；Map Head 出矢量地图；Motion Head 出每个 agent 的未来轨迹；Planning Head 出自车轨迹（VADv2 出多条）。

## 0. 为什么要讲 VAD 的代码，以及一句话结论

为什么要讲它？三个理由：

第一，它是「轻量端到端」的标杆，和 UniAD 形成鲜明对比——同样做端到端，VAD 砍掉了占用和分割头，代码量小一圈，但规划指标不差。搞清楚它怎么做到的，你对端到端架构的「取舍空间」会有实感。

第二，它的 Planning KD 和多模态规划，是后面 DiffusionDrive / Diffusion Planner / DriveVLA 这些「生成式规划」方法的思想源头。理解 VAD，等于拿到了通往后面文章的钥匙。

第三，它的代码挂在 mmdet3d 上，结构清晰，非常适合当「端到端代码入门读物」。

> 一句话结论：VAD 的本质是「用结构化先验（向量表示）换算力」，它证明了端到端不一定要堆很重的感知栈；而 VADv2 把规划从「回归一条」变成「生成多条再选」，这一步正好接上后面所有生成式规划方法。

## 1. 架构总览

在钻进代码之前，先把整条流水线在脑子里过一遍。VAD 的推理流程是一个「层层递进、共享 BEV」的过程：

- 输入：6 路环视相机图像（nuScenes 配置）。
- 编码器：用 BEVFormer 类（temporal + spatial attention）把图像编码成 BEV 特征图。这一步和 UniAD 基本同款，VAD 的重点不在编码器，在后面的 head。
- Agent Head：在 BEV 上用检测 query 解码出周围 agent 的向量 box（10 个参数），包括位置和速度朝向。
- Map Head：用地图 query 解码出矢量化的车道（每条车道是一串点），并且和 agent 做交叉注意力交互，让车道表示带上「和车有关」的信息。
- Motion Head：对每个 agent query，解码出 K 条未来轨迹向量，并让 agent 和 map 互相 attend，使轨迹顺着车道走。
- Planning Head：用 ego query 去看 BEV + agent 向量 + map 向量 + motion 向量，输出自车轨迹。VAD 原始版出单条；VADv2 出 M 条候选 + softmax 概率。
- 训练时额外加 Planning KD：用离线专家轨迹蒸馏规划头。

注意一个关键点：**VAD 没有显式的占用预测（occupancy）**。它不像 UniAD 有 OccFormer 和分割头，所有信息都用「向量」表达，规划头直接 consume 这些向量。这就是它「轻」的来源。

## 2. 项目结构：它如何挂在 mmdet3d 上

VAD 是一个 **mmdetection3d 的插件式仓库**，也就是说，它不重写检测框架，而是在 mmdet3d 外面套一层 `mmdet3d_plugin`，通过 `projects/` 下的配置注册新模块。这种「插件 + 配置」的挂法，是 OpenMMLab 系列（mmdet / mmdet3d）的标准玩法。

仓库的长相大致是这样（用列表表示目录树，避免框线字符）：

- `VAD/`
  - `projects/`
    - `configs/`
      - `vad/`
        - `vad_base.py`：VAD 的基础配置（模型、数据、损失、训练策略的总入口）
        - `vad_nuscenes/`：nuScenes 数据集的具体配置（通常继承 vad_base 再改数据路径和评测）
  - `mmdet3d_plugin/`
    - `vad/`
      - `dense_heads/`
        - `vad_head.py`：总装 head，把检测 / 地图 / 运动 / 规划四个分支组合起来，也是 loss 汇总的地方
        - `vad_track_head.py`：可选的跟踪头（tracking），论文主线上不是重点
        - `vad_motion_head.py`：运动预测头，出每个 agent 的 K 条轨迹
      - `modules/`
        - `vectorized_map.py`：矢量地图编码器，把车道表示成点序列 query
        - `agent_cross_attn.py`：agent 与地图之间的交叉注意力模块
        - `planning_head.py`：规划头，VAD 单轨迹 / VADv2 多模态就在这里
      - `datasets/`：nuScenes 的定制数据管线（pipeline、evaluator 等）
  - `tools/`
    - `dist_train.sh`：分布式训练启动脚本
    - `dist_test.sh`：分布式评测启动脚本

和 UniAD 的边界（这是面试和读代码都常考的点）：

- VAD 同样基于 mmdet3d 插件，同样用 BEVFormer 类编码器，这和 UniAD 一致。
- VAD **没有 OccFormer，没有分割头**。UniAD 有 6 个 head（含占用预测、分割），VAD 砍到 4 个（agent / map / motion / planning），靠向量化表示省掉了所有稠密头。
- VAD 的规划头直接吃「agent 向量 + map 向量 + motion 向量」，信息链路更短、更紧凑。
- VAD 引入了 Planning KD，而 UniAD 没有显式的规划蒸馏。

> 一句话澄清：VAD 不是「另起炉灶的新框架」，它是在 mmdet3d + BEVFormer 这套成熟骨架上，把「感知表征」从稠密换成向量，从而瘦身成功的。理解这一点，你就不会被它「轻量」的标签误导——它的重活（BEV 编码）其实和 UniAD 一样没少。

## 3. 逐文件逐函数细讲

下面我们按「从总装到分支」的顺序，把每个关键文件拆开揉碎。每个代码块里都只有英文和符号，中文解释一律放在块外面，方便你直接复制运行思路。

### 3.1 vad_head.py：四个头的总装与 loss 汇总

`vad_head.py` 是整个 VAD 的「总调度」。它继承自 mmdet3d 的 `Base3DDenseHead`（或类似基类），在 `forward` 里依次调用 agent / map / motion / planning 四个子模块，并在 `loss` 里把它们各自的损失加权求和。

先看它大致怎么把四个分支组织起来（简化伪代码，只保留结构）：

```python
# mmdet3d_plugin/vad/dense_heads/vad_head.py (simplified structure)

class VADHead(Base3DDenseHead):
    def __init__(self,
                 agent_head=None,
                 map_head=None,
                 motion_head=None,
                 planning_head=None,
                 plan_kd_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.agent_head = build_head(agent_head)
        self.map_head = build_head(map_head)
        self.motion_head = build_head(motion_head)
        self.planning_head = build_head(planning_head)
        self.plan_kd_weight = plan_kd_weight

    def forward(self, bev_feat, img_metas=None):
        # 1) agent: (N_agent, 10) boxes
        agent_out = self.agent_head(bev_feat, img_metas)
        # 2) map: list of point sequences
        map_out = self.map_head(bev_feat, img_metas)
        # 3) motion: (N_agent, K, T, 2) trajectories
        motion_out = self.motion_head(agent_out['query'],
                                      map_out['query'],
                                      bev_feat)
        # 4) planning: ego trajectory
        plan_out = self.planning_head(bev_feat,
                                      agent_out['query'],
                                      map_out['query'],
                                      motion_out)
        return agent_out, map_out, motion_out, plan_out

    def loss(self, outs, targets):
        agent_loss = self.agent_head.loss(outs[0], targets['agent'])
        map_loss = self.map_head.loss(outs[1], targets['map'])
        motion_loss = self.motion_head.loss(outs[2], targets['motion'])
        plan_loss = self.planning_head.loss(outs[3], targets['plan'])
        kd_loss = self.planning_head.kd_loss(outs[3], targets['expert_plan'])
        total = (agent_loss + map_loss + motion_loss
                 + plan_loss + self.plan_kd_weight * kd_loss)
        return dict(loss=total,
                    agent_loss=agent_loss,
                    map_loss=map_loss,
                    motion_loss=motion_loss,
                    plan_loss=plan_loss,
                    kd_loss=kd_loss)
```

上面这个 `forward` 的顺序就是前面「架构总览」里那条流水线的代码映射：agent 先出，map 第二，motion 拿 agent+map 的 query 出轨迹，planning 拿全部信息出自车轨迹。注意 motion head 和 planning head 吃的不是「原始 box 坐标」，而是 agent / map 解码出来的 **query 向量**——这保证了信息在向量空间里流动，而不是在坐标空间里拼接。

`loss` 函数把四块损失加起来，再额外加一份 `kd_loss`（Planning KD）。这也是 VAD 训练里最容易被忽略、但其实最重要的那一项。

> 关键认知：`vad_head.py` 本身几乎不含「算法创新」，它只是一个「把四个头粘起来 + 把五个损失加起来」的壳。真正的创新分散在 `vad_motion_head.py`、`vectorized_map.py`、`planning_head.py` 里。读 VAD 代码，重点永远在后三个文件。

### 3.2 Agent Head：向量化的检测

Agent Head 本质上是一个 DETR3D 风格的检测解码器，但它的输出是「向量化的 box 参数」而不是一堆密集特征。它用一组可学习的 agent query，在 BEV 特征上做多层 self/cross attention，最后过一个回归头出 10 维向量。

简化伪代码：

```python
# mmdet3d_plugin/vad/dense_heads/vad_head.py (agent branch, simplified)

class VADAgentHead(nn.Module):
    def __init__(self, num_query=200, embed_dims=256, num_reg=10):
        super().__init__()
        self.agent_embedding = nn.Embedding(num_query, embed_dims)
        self.decoder = nn.ModuleList([
            DetrTransformerDecoderLayer(embed_dims) for _ in range(6)
        ])
        self.reg_head = nn.Linear(embed_dims, num_reg)   # 10 = x,y,z,w,l,h,yaw,vx,vy,cls
        self.cls_head = nn.Linear(embed_dims, 1)

    def forward(self, bev_feat, img_metas):
        query = self.agent_embedding.weight          # (N, C)
        for layer in self.decoder:
            query = layer(query, bev_feat)           # cross + self attn
        boxes = self.reg_head(query)                 # (N, 10)
        scores = self.cls_head(query)                # (N, 1)
        return dict(query=query, boxes=boxes, scores=scores)
```

这里 `reg_head` 输出的 10 维就是前面名词表里说的 `x,y,z,w,l,h,yaw,vx,vy,cls`。`query` 会被原样返回，因为它后面要给 Motion Head 当输入——这就是「向量化表示」带来的连锁好处：检测出的车，它的「身份向量」直接就能拿去预测它的运动，不需要再去做一遍匹配。

> 一句话澄清：Agent Head 输出的 `query` 不是 box 坐标，而是「这辆车的语义向量」。box 坐标是 `reg_head` 从 query 里解出来的副产品。真正在 VAD 内部流转的「车」是 query，不是坐标。

### 3.3 Map Head：把车道变成点序列

Map Head 是 VAD 「向量化地图」的核心。传统方案用分割头在 BEV 上画车道掩码（稠密），VAD 改用「点序列 query」：每条车道由一组可学习点 + 一个整体 embedding 表示。解码器让地图 query 在 BEV 上 attend，最后回归出每条车道的点坐标。

简化伪代码：

```python
# mmdet3d_plugin/vad/modules/vectorized_map.py (simplified)

class VectorizedMapHead(nn.Module):
    def __init__(self, num_map=50, num_pts=20, embed_dims=256):
        super().__init__()
        self.map_embedding = nn.Embedding(num_map, embed_dims)
        self.point_embedding = nn.Embedding(num_pts, embed_dims)
        self.decoder = nn.ModuleList([
            DetrTransformerDecoderLayer(embed_dims) for _ in range(6)
        ])
        self.map_reg_head = nn.Linear(embed_dims, 2)   # each point (x, y)

    def forward(self, bev_feat, img_metas):
        map_query = self.map_embedding.weight              # (N_map, C)
        point_query = self.point_embedding.weight         # (P, C)
        query = map_query.unsqueeze(1) + point_query      # (N_map, P, C)
        query = query.flatten(0, 1)                       # (N_map*P, C)
        for layer in self.decoder:
            query = layer(query, bev_feat)
        points = self.map_reg_head(query)                 # (N_map*P, 2)
        points = points.view(-1, self.num_pts, 2)         # (N_map, P, 2)
        return dict(query=map_query, points=points)
```

这里有个小技巧：每条车道用一个 `map_query` 表示「这条车道是谁」，再叠加每个点的 `point_query` 表示「这个点在哪」。解码后只回归每个点的 `(x, y)`，于是每条车道就成了一串点——这就是「矢量化地图」：一条车道 = 一组有序点，而不是一张掩码图。

`map_query` 同样被返回，后面 Motion Head 要用它做 agent-地图交互。

### 3.4 agent_cross_attn.py：让车和车道互相看一眼

为了让「车的轨迹会顺着车道走」，VAD 在 agent 和 map 之间加了一个交叉注意力模块。名字就叫 `agent_cross_attn`。它的作用是：让 agent query 去 attend map query，从而把「车道形状」这个结构化先验注入到 agent 的运动预测里。

简化伪代码：

```python
# mmdet3d_plugin/vad/modules/agent_cross_attn.py (simplified)

class AgentCrossAttn(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, agent_query, map_query, bev_feat):
        # agent looks at map
        out, _ = self.attn(query=agent_query,
                           key=map_query,
                           value=map_query)
        agent_query = self.norm(agent_query + out)
        return agent_query
```

这块代码很短，但它是 VAD 「向量化交互」思想的精华：车和车道都是向量，它们之间的交互就是一次注意力，不需要把车道栅格化再卷积。这种交互在 Motion Head 的每一层都会发生。

### 3.5 Motion Head：每个 agent 出 K 条轨迹

Motion Head 负责「别人家的车会怎么走」。它对每个 agent query 解码出 K 条未来轨迹（每条轨迹是 T 个时间步的 `(x, y)`），并且在每一层都让 agent query 和 map query 互相 attend（通过上面那个 `agent_cross_attn`），让轨迹贴合车道。

简化伪代码：

```python
# mmdet3d_plugin/vad/dense_heads/vad_motion_head.py (simplified)

class VADMotionHead(nn.Module):
    def __init__(self, num_modes=6, traj_len=12, embed_dims=256):
        super().__init__()
        self.num_modes = num_modes
        self.traj_len = traj_len
        self.motion_layers = nn.ModuleList([
            MotionDecoderLayer(embed_dims) for _ in range(6)
        ])
        self.mode_embedding = nn.Embedding(num_modes, embed_dims)
        self.traj_head = nn.Linear(embed_dims, traj_len * 2)

    def forward(self, agent_query, map_query, bev_feat):
        # agent_query: (N_agent, C), expand to modes
        N = agent_query.shape[0]
        query = agent_query.unsqueeze(1) + self.mode_embedding.weight  # (N, K, C)
        query = query.flatten(0, 1)                                   # (N*K, C)
        for layer in self.motion_layers:
            query = layer(query, map_query, bev_feat)                 # cross attn to map
        traj = self.traj_head(query)                                  # (N*K, T*2)
        traj = traj.view(N, self.num_modes, self.traj_len, 2)         # (N, K, T, 2)
        return traj
```

注意 `mode_embedding`：它给每个 agent 的每条候选轨迹一个「模式编号」向量，这样同一个 agent 的 K 条轨迹在解码时就有不同的「起点倾向」，能自然分化出左转 / 直行 / 右转等不同模式。这是「多模态运动预测」最朴素也最有效的做法之一。

> 关键认知：Motion Head 的 K 条轨迹是「per-agent 多模态」，Planning Head 的 M 条轨迹是「ego 多模态」，两者概念同源但对象不同。前者预测别人，后者规划自己。VADv2 的规划多模态，正是 Motion Head 这套思想的「搬到自车上」。

### 3.6 Planning Head：VAD 单轨迹 vs VADv2 多模态

Planning Head 是整个 VAD 的「最后一公里」，也是 VAD 和 VADv2 差异最大的地方。

**VAD 原始版（单轨迹回归）**：用一个 ego query 去看 BEV + agent + map + motion 的向量，最后回归出一条自车未来轨迹。

简化伪代码：

```python
# mmdet3d_plugin/vad/modules/planning_head.py (VAD v1, simplified)

class VADPlanningHead(nn.Module):
    def __init__(self, traj_len=12, embed_dims=256):
        super().__init__()
        self.ego_embedding = nn.Embedding(1, embed_dims)
        self.ego_decoder = PlanningDecoderLayer(embed_dims)
        self.traj_head = nn.Linear(embed_dims, traj_len * 2)

    def forward(self, bev_feat, agent_query, map_query, motion_traj):
        ego_query = self.ego_embedding.weight            # (1, C)
        ego_query = self.ego_decoder(ego_query,
                                     bev_feat,
                                     agent_query,
                                     map_query,
                                     motion_traj)
        traj = self.traj_head(ego_query)                 # (1, T*2)
        traj = traj.view(1, self.traj_len, 2)            # (1, T, 2)
        return traj
```

**VADv2（多模态概率分布）**：把规划从「回归一条」改成「生成 M 条候选 + 每条 softmax 概率」。这一步让 VADv2 从「回归式端到端」跨进了「生成式选择」的门槛。

简化伪代码：

```python
# mmdet3d_plugin/vad/modules/planning_head.py (VADv2, simplified)

class VADv2PlanningHead(nn.Module):
    def __init__(self, num_modes=10, traj_len=12, embed_dims=256):
        super().__init__()
        self.num_modes = num_modes
        self.mode_embedding = nn.Embedding(num_modes, embed_dims)
        self.ego_decoder = PlanningDecoderLayer(embed_dims)
        self.mode_traj_head = nn.Linear(embed_dims, traj_len * 2)
        self.mode_score_head = nn.Linear(embed_dims, 1)

    def forward(self, bev_feat, agent_query, map_query, motion_traj):
        ego_query = self.ego_embedding.weight                 # (1, C)
        ego_query = self.ego_decoder(ego_query,
                                     bev_feat,
                                     agent_query,
                                     map_query,
                                     motion_traj)
        # expand to M modes
        query = ego_query + self.mode_embedding.weight        # (M, C)
        trajs = self.mode_traj_head(query)                    # (M, T*2)
        trajs = trajs.view(self.num_modes, self.traj_len, 2)  # (M, T, 2)
        scores = self.mode_score_head(query)                  # (M, 1)
        probs = scores.softmax(dim=0)                         # (M,)
        return dict(trajs=trajs, probs=probs)
```

推理时，VADv2 不会直接拿概率最大的那条就走，而是用各 mode 的 `probs` 乘上「与 agent 轨迹 / 地图的碰撞惩罚」做后处理，再选最终轨迹。这已经非常接近「生成一堆 + 选一条」的生成式思路，是后面 Diffusion Planner / DriveVLA 的近亲。

> 一句话澄清：VADv2 的「多模态」不是靠扩散模型采样的，它是用 M 个 mode embedding 一次性并行回归出 M 条轨迹，再 softmax 出概率。所以它快、简单，但也受限于「M 条是固定的、且互相独立」。真正「无限生成」的多模态要靠 Diffusion，那就是本系列后面文章的事了。

### 3.7 Planning KD：用离线专家兜底安全

VAD 在规划上还有一个隐藏大招：Planning KD。它的想法很朴素——端到端模型容易规划出「指标好看但不安全」的轨迹，于是找一个离线传统规划器（如 Hybrid A* 或已标注的安全轨迹）当 teacher，让学生的轨迹往 teacher 靠。

在代码里，teacher 轨迹通常是**数据集预生成**的，训练时直接读进来算损失，不会在训练循环里实时跑规划器（那样太慢）。损失可以用 Smooth-L1（回归坐标）或 KL（对齐概率分布，VADv2 用）。

简化伪代码：

```python
# inside VADPlanningHead / VADHead.loss (simplified)

def kd_loss(self, plan_out, expert_traj):
    if isinstance(plan_out, dict):                 # VADv2
        pred = plan_out['trajs']                   # (M, T, 2)
        # align to expert by best mode
        dist = ((pred - expert_traj) ** 2).sum(-1).mean(-1)   # (M,)
        best = dist.argmin()
        return F.smooth_l1_loss(pred[best], expert_traj)
    else:                                          # VAD v1
        return F.smooth_l1_loss(plan_out.squeeze(0), expert_traj)
```

这段伪代码说明了 KD 在两种版本下的不同处理：VADv2 因为有 M 条，要先选一条「离专家最近」的 mode 再算损失；VAD v1 只有一条，直接算。无论哪种，核心都是「让模型学出一条安全轨迹作为保底」。

> 关键认知：Planning KD 是 VAD 「敢砍掉占用头」的底气之一。没有显式障碍栅格，模型对「什么不能走」的感知变弱，KD 用专家轨迹把这部分安全知识补回来。换句话说，VAD 把「稠密感知的安全感」换成了「蒸馏来的安全知识」。

## 4. 训练与推理：怎么把 VAD 跑起来

代码层面讲完，再补一下工程入口，方便你想动手时知道从哪敲命令。VAD 沿用了 mmdet3d 的 `tools/dist_train.sh` 和 `tools/dist_test.sh`。

训练 VAD 基础版：

```bash
bash tools/dist_train.sh projects/configs/vad/vad_base.py 8
```

训练 VADv2（多模态规划版本，配置通常叫 `vad_v2.py` 之类的，具体以仓库为准）：

```bash
bash tools/dist_train.sh projects/configs/vad/vad_v2.py 8
```

评测（用官方 nuScenes planning metric，即 L2 误差 + 碰撞率）：

```bash
bash tools/dist_test.sh projects/configs/vad/vad_base.py ckpts/vad.pth 8
```

关于配置 `vad_base.py`，它里面主要干四件事：定义 backbone / BEV 编码器、注册四个 VAD head、列出五个损失及其权重、指定数据集和 evaluator。想改 VAD 的行为（比如把 K 调到 8、把 num_modes 调到 12），基本都在这里和对应 head 的 `__init__` 里改。

> 一句话澄清：VAD 的「轻量」主要体现在模型结构（少两个 head），但训练成本并不比 UniAD 低太多，因为 BEV 编码器那一坨重计算它一字未减。所以「轻」指的是「参数和中间表征更省」，不是「训练更快」。

## 5. VAD vs UniAD：代码层面的取舍（再总结一遍）

- UniAD 有 6 个 head（含占用 / 分割），重但监督多；VAD 砍到 4 个（agent / map / motion / planning），靠向量化表示省掉稠密头。
- VAD 没有 OccFormer，规划头直接吃 agent / map / motion 向量，信息链路更短、更紧凑。
- VAD 引入 Planning KD，UniAD 没有显式规划蒸馏。
- VADv2 把规划变成概率多模态，是「回归式端到端 → 生成式选择」的重要过渡。
- 代价：向量化地图依赖离线地图提取质量；稠密占用带来的「显式障碍栅格」安全感没了，要靠 KD 补。

## 5.5 数据管线与矢量地图的离线提取

前面反复提到「向量化地图」，但有个问题没讲透：Map Head 训练时用的「真值车道点序列」从哪来？答案是**离线提取**。nuScenes 本身给的是矢量地图（lane polyline），VAD 的 `datasets/` 里会把官方地图按帧裁剪、采样成固定数量的点序列，作为 Map Head 的监督。也就是说，地图这一支的「真值」不是模型现猜的，而是拿现成高清地图离线处理好的。

简化伪代码（数据侧，非模型侧）：

```python
# mmdet3d_plugin/vad/datasets/ (simplified idea)

def get_map_targets(self, img_metas):
    targets = []
    for meta in img_metas:
        vector_map = meta['vector_map']            # list of polylines
        sampled = []
        for poly in vector_map:
            pts = resample_polyline(poly, num_pts=20)   # fixed length
            sampled.append(pts)
        # pad / truncate to fixed number of lanes
        sampled = sampled[:50]
        targets.append(torch.tensor(sampled))
    return targets                                 # (B, N_map, P, 2)
```

这点很关键，因为它揭示了 VAD 「轻」的又一个前提：它把「地图理解」的难题，一部分外包给了离线高清地图和预处理。在线推理时，如果这些地图缺失或过时，Map Head 的表现会受影响。所以 VAD 的轻量是有条件的——前提是有一份靠谱的矢量地图兜底。

> 一句话澄清：VAD 的 Map Head 不是在 BEV 上「从零分割车道」，而是在「已经有矢量地图真值」的前提下，学一个从 BEV 特征到已知地图结构的对齐/预测。别把它误解成完全端到端地从图像生成地图。

## 6. 个人思考

VAD 的「向量化」本质是用结构化先验换算力。它证明了端到端不一定非要 UniAD 那么重的感知栈，但也暴露了回归式规划的局限性——单条轨迹无论如何也表达不了「前方路口既可左转也可直行」的不确定性。所以 VADv2 才走向多模态概率。

但 VADv2 的 M 条是「固定数量、彼此独立回归」的，本质上还是把多模态当成了「M 个独立回归头」，并没有真正建模「轨迹的空间分布」。这一步，正好接上后面 Diffusion 系列「用扩散模型从噪声里采样出多样且合理的轨迹」的范式。可以说，VADv2 是把门推开了一条缝，DiffusionDrive / Diffusion Planner 才是真正走进去的人。

另外一个值得玩味的点：VAD 的轻量很大程度上是把「安全责任」转移给了 Planning KD。这提醒我们，端到端模型「看起来轻」，未必真的轻——它可能只是把复杂度藏到了数据预处理和蒸馏 teacher 里。读代码时，别只盯着模型 forward，训练流水线里的专家轨迹生成同样重要。

## 7. 和本系列其他文章的关系

- 轻量化端到端 → VAD（本文）。
- 共享 query 全栈、带占用与分割 → UniAD（系列第 1~2 篇方向）。
- 生成式扩散规划 → DiffusionDrive（679 行）、Diffusion Planner。
- 多模态概率规划 → VADv2 是雏形，Diffusion Planner 把它做成了扩散采样；DriveVLA-W0（436 行）则是用 VLM 来做规划决策。

如果你是按顺序读到这的，下一站建议直接看 DiffusionDrive：你会惊喜地发现，它把 VADv2 那种「生成多条再选」的思路，用扩散模型重新实现了一遍，而且因为去除了稠密感知，反而更贴近 VAD 的「轻」哲学。两条线（向量化轻量 + 生成式多模态）在 Diffusion 系列汇合，这就是端到端规划最近一年最清晰的演进脉络。

---

> 写在最后：VAD 的代码量不大，但它每一个 head 都是「向量化表示」这一理念的一次落地。读懂它，你就拿到了理解后续所有生成式规划方法的钥匙。下一篇见。
