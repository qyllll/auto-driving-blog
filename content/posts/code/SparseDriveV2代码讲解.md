---
title: "代码讲解：SparseDriveV2 因子化轨迹词汇表与两级评分的完整闭环"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["SparseDriveV2", "端到端自动驾驶", "代码讲解", "NAVSIM", "因子化词汇表", "两级评分"]
summary: "逐行拆解 swc-17/SparseDriveV2 的真实源码：从架构图与全局数据流出发，用最通俗的大白话讲清几何路径乘以速度剖面如何因子化组合成 26 万条轨迹词汇表、两级评分怎么把候选从 26 万粗筛到 200 条再精排、PDM score 蒸馏损失如何用排行榜判分器当老师。一篇把项目挂载方式、数据流、前向传播、训练梯度、推理选轨迹、个人思考全部讲透、面向零基础读者的工程向代码讲解。"
---

## 写给零基础读者：读这篇之前先搞懂几个名词

在看代码之前，先把几个反复出现的「黑话」翻译成人话，后面就不会晕了：

- **端到端自动驾驶（End-to-End Driving）**：传统自动驾驶是把「看路 → 识别车/人 → 预测 → 规划路线 → 控制方向盘」拆成很多个小模块分别写程序。端到端的意思是「给一张相机拍的画面，直接从画面算出车该怎么开（往哪走、开多快）」，中间不需要人手写一堆规则。SparseDriveV2 就是一个端到端模型。
- **轨迹（Trajectory）**：车在未来几秒内要走的路线，通常表示成未来若干个时刻车的「位置 + 朝向」。比如论文里用 8 个时刻（pose），每个时刻记成 `(x, y, θ)` 三个数——x/y 是平面坐标，θ 是车头朝向角度。一串 `(x, y, θ)` 连起来就是一条轨迹。
- **anchor / 词汇表（Vocabulary）**：可以理解成「提前准备好的一张标准动作菜单」。比如「左转」「右转」「直行」「加速」「减速」这些基本动作。模型不是现场发明动作，而是从这张菜单里挑一个最合适的。菜单里的每一项就叫一个 anchor（锚点），整张菜单就叫词汇表。
- **检索式规划（Retrieval-based Planning）**：不自己「创造」新轨迹，而是从一张现成的、巨大的候选轨迹菜单里「翻」出最好的一条。就像你点外卖不是自己下厨，而是从商家菜单里选。SparseDriveV2 就是这种思路，它和「生成式」相对。
- **生成式规划（Generative Planning）**：相反，模型自己「画」出一条可能菜单里根本没有的新轨迹，比如 DiffusionDrive 用扩散模型一步步「画」出轨迹。
- **cross-attention（交叉注意力）**：深度学习里一种「让一组查询去另一组内容里找信息」的机制。通俗讲：模型拿着「候选动作」这一组问题，去「相机图像」这一组答案里，找跟每个动作最相关的图像区域特征。SparseDriveV2 用 deformable attention（可变形注意力）让候选轨迹去图像上按相机内外参采样特征。
- **top-k 筛选**：从一堆分数里只保留分数最高的 k 个，其余丢掉。比如从 1024 条路径里只留分数最高的 64 条。
- **PDM 仿真器（PDM Simulator）**：NAVSIM 这个评测榜单自带的「虚拟考官」。给它一条轨迹和当前场景，它会模拟车按这条轨迹开下去会不会撞车、有没有压线、有没有按车道方向开、进度好不好，并给出一个 0~1 的综合分数（叫 PDMS）。SparseDriveV2 训练时就把这个「考官」当老师，学它的打分逻辑。
- **waypoint（航点）**：轨迹上一个个具体的「路点」，就是上面说的每个时刻的 `(x, y, θ)`。8 个 waypoint 串起来就是一条 8 点轨迹。
- **冻结（frozen / requires_grad=False）**：意思是这部分参数在训练时「锁死」，不参与梯度更新，永远不变。SparseDriveV2 的 26 万条词汇表就是冻结的。
- **KMeans 聚类**：一种把一堆数据自动分成若干类的算法。比如把训练集里成千上万条真实轨迹的「形状」自动归成 1024 类代表形状。它只是个离线统计工具，不涉及神经网络训练。

## 为什么要讲 SparseDriveV2 的代码

SparseDriveV2（swc-17）是 NAVSIM 排行榜上 Scoring-based（基于打分）路线的代表之一（navtest 官方榜 92.0 PDMS）。它和 DiffusionDrive 思路完全不同：**不扩散、不生成，而是预先建一个巨大的「轨迹词汇表」，再学一个打分器挑最好的**。它的核心创新是**因子化轨迹词汇表（path × velocity）**和**两级评分（coarse-to-fine，粗到细）**。这篇直接看真实源码，把每一步都拆碎了讲。

> **一句话结论**：SparseDriveV2 = 冻结的 path(1024) × velocity(256) 组合词汇表（26 万条轨迹）+ 两级评分（先按 path/vel 粗筛到 200 条，再用 PDM 蒸馏的 metric head 精排 argmax）。它把「规划」变成了一个「从超大离散候选集里检索最优」的问题。

## 架构总览：先看地图

SparseDriveV2 是**纯规划（planning-only）**模型，没有检测 head（不负责识别车/行人）。它的推理链路极短：图像过轻量 backbone（特征提取网络）→ 特征送两层 transformer decoder 当「打分器」→ 在一个**预先冻结的 26 万条轨迹字典**上打分 → 两级筛选后取最优。下面这张图（SparseDrive 系列论文原图）概括了整体范式：

![SparseDrive 架构总览](/images/sparsedrive/architecture_overview.png)

为了把「代码里数据怎么流」讲清楚，我自己画了一张**数据流图**，把论文图和真实源码对应起来：

![SparseDriveV2 推理数据流：navsim输入 → SparseBackbone → 冻结26万词表作query → 两级筛选(粗64×20→细PDMS式) → argmax输出最优轨迹](/images/sparsedrive/dataflow.svg)

> **关键认知**：模型本身**不生成轨迹**，只给一个固定的候选集打分。这与 DiffusionDrive（生成轨迹）形成镜像——SparseDriveV2 把「规划」彻底退化成了一个「检索+排序」问题。这也是它推理极快、且天然契合 NAVSIM 评分逻辑的原因。

## 项目结构：它如何挂在 navsim 上（先厘清边界）

先泼一盆冷水，把最容易看晕的地方讲透：**SparseDriveV2 确实 fork 了 autonomousvision/navsim，但这「fork」里只改了 `navsim/agents/sparsedrive/` 这一个角落**。其他所有东西——训练入口、评测入口、数据缓存、PDM 打分器——全部是 navsim 原装的，一行没动。作者自己写的代码就那一小块「插件」，通过 navsim 的 `Agent` 插件机制挂进去。

还有一个**绝对不要混淆**的点：离线构建词表的脚本 `scripts/cluster/cluster_anchor.py` **不在 `navsim/` 目录下，也不在 `sparsedrive/` 子目录里**，它是仓库根目录下的独立工具，跑一次就产出 `.npz` 词表文件，之后训练/推理都只加载这个文件、再也不碰脚本。下面把「仓库里到底有哪些东西、各归谁管」切成两块看。

**区块 A：fork 自 navsim 的官方管线（作者基本没动，直接复用）**

```
navsim/
|-- agents/
|   `-- sparsedrive/                # <- 唯一作者新增的"插件"目录
|       |-- sparsedrive_agent.py      # Agent（继承 AbstractAgent）入口，navsim 每个场景调它
|       |-- sparsedrive_model.py      # 模型定义 + 加载冻结词汇表（.npz）
|       |-- sparsedrive_backbone.py   # ResNet-34 主干 + FPN 多尺度特征
|       |-- custom_decoder.py         # 两级评分 + 组合 + 损失（全文核心）
|       `-- scorer/
|           `-- get_pdm_score_v2.py   # PDM 打分（复用 navsim 仿真器）
`-- planning/script/run_training.py  # 官方训练入口，直接调用，不改
```

**区块 B：仓库根目录下的作者自有工具（与 navsim/ 平级）**

```
scripts/
|-- cluster/
|   `-- cluster_anchor.py            # 离线 KMeans 建 path/velocity 词表 -> .npz
`-- training/
    `-- sparsedrive_navsimv1.sh      # 只是个调 run_training.py 的启动脚本
```

逐文件一句话说明它干嘛：

- `sparsedrive_agent.py`：Agent 插件入口，navsim 评测时对每个场景调 `compute_trajectory(agent_input)`，它负责把输入整理好送进模型、再把模型输出包成 navsim 要求的格式。相当于「对外接口」。
- `sparsedrive_model.py`：真正的模型类 `SparseDriveModel`，里面定义 backbone、ego 状态编码、加载冻结的 path/velocity/trajectory 词汇表，并把它们组装成一次 forward。相当于「主机箱」。
- `sparsedrive_backbone.py`：图像特征提取器 `SparseBackbone`，用 ResNet-34 + FPN 把三相机图像变成多尺度特征图。相当于「眼睛」。
- `custom_decoder.py`：最核心的 `CustomTransformerDecoder`，实现两级评分、path⊕vel 组合、训练时的损失计算。相当于「大脑里的打分员」。
- `scorer/get_pdm_score_v2.py`：封装 navsim 的 PDM 仿真器，给一条轨迹算 PDMS 各项分数，训练时当「老师」提供监督标签。

> **一句话澄清挂载关系（训练时）**：你执行 `scripts/training/sparsedrive_navsimv1.sh` → 它内部调官方的 `run_training.py` → 通过 Hydra 配置 `agent=sparsedrive_agent` → 实例化 `SparseDriveAgent`（作者插件）→ agent 内部建 `SparseDriveModel` + `CustomTransformerDecoder`。词表由区块 B 的离线脚本**提前**生成好、以 `.npz` 加载，所以训练主链路里根本不存在 `scripts/cluster/` 这个目录。

> **一句话澄清挂载关系（推理时）**：navsim 的评测脚本对每个场景调 `agent.compute_trajectory(agent_input)` → `SparseDriveAgent` 把输入送进 `SparseDriveModel` → 在冻结词表上打分、两级筛选、argmax。全程不依赖任何「生成」，只做「检索+排序」。

Agent 定义：`navsim/agents/sparsedrive/sparsedrive_agent.py:22` `class SparseDriveAgent(AbstractAgent)`。

## 一、架构：因子化轨迹词汇表

### 1.1 它不是「instance query 做检测+规划」

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

`TrajectoryHead` 加载**冻结的** path/velocity/trajectory 词汇表（`requires_grad=False`），作为 query 送入 `CustomTransformerDecoder`。这里要记住：模型查表得到的「轨迹嵌入」是固定的，模型只是拿它们当问题去图像里找答案，并不会改这些轨迹本身。

### 1.2 词汇表构建（离线 KMeans）：这是 26 万条的来源

这一节回答「那 26 万条轨迹到底哪来的」。答案：**离线预聚类，不是模型学的**。

什么叫「因子化（factorized）」？一句话：把一条完整轨迹拆成两个相互独立的部分——**往哪走（path，几何路径）** 和 **开多快（velocity，速度剖面）**。单独对这两部分各聚一类，再两两组合，就能用很少的聚类中心覆盖极多的动作。

- **path 候选**：对训练集所有轨迹的「形状」（忽略速度，只看几何路径）做 KMeans，`K_PATH = 1024`；
- **velocity 候选**：对所有轨迹的「速度剖面」做 KMeans，`K_VELOCITY = 256`；

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

**Vocab 大小 = 1024 × 256 = 262,144 条组合轨迹**（README 所谓「32× denser」，是相对于 Hydra-MDP 的 8K 词表）。运行时加载：

```python
# sparsedrive_model.py:73
self.path_vocab = nn.Parameter(torch.from_numpy(np.load(config.path_anchor)).float(), requires_grad=False)
self.vel_vocab  = nn.Parameter(torch.from_numpy(np.load(config.velocity_anchor)).float(), requires_grad=False)
trajectory_data = np.load(config.trajectory_anchor)
self.traj_vocab = nn.Parameter(torch.from_numpy(trajectory_data["trajectory"]).float(), requires_grad=False)
self.traj_mask  = nn.Parameter(torch.from_numpy(trajectory_data["trajectory_mask"]).float(), requires_grad=False)
```

> **为什么因子化？** 直接枚举 26 万条完整轨迹当固定 query 太浪费参数；拆成 path(往哪走) × velocity(多快走) 两个因子，组合时只需把两个嵌入相加即可指数级覆盖动作空间，vocab 尺寸只是两者之和。这是「在动作空间结构上做归纳偏置」。

**补充一点数学直觉**：如果不因子化，要覆盖 26 万种动作就得存 26 万个向量；因子化后只存 1024 + 256 = 1280 个向量，组合时相乘得到 26 万。参数量从 $O(262144)$ 降到 $O(1024+256)$。代价是假设「路径」和「速度」可以独立拆分——对正常驾驶基本成立，对极端场景可能略有偏差，但工程上极其划算。

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

这里 `⊕` 就是逐元素相加（广播加法）。`filter_path_embed` 形状是 `[B, 64, d]`，`unsqueeze(2)` 后变成 `[B, 64, 1, d]`；`filter_vel_embed` 形状是 `[B, 20, d]`，`unsqueeze(1)` 后变成 `[B, 1, 20, d]`；两者相加会广播成 `[B, 64, 20, d]`，再 `flatten(1,2)` 展平成 `[B, 1280, d]`。这就是「加法组合」——把每条路径和每种速度配对，得到 1280 条组合轨迹的嵌入。注意这只是嵌入层面的组合，对应的真实轨迹坐标由冻结的 `filter_traj_vocab` 同步筛选得到（见下文前向传播第 ⑦ 步）。

**第二级（细粒度，组合）**：仅最后一层，对粗筛后的组合轨迹用多个 metric head 打分（每个 PDMS 指标一个 head）：

```python
# custom_decoder.py:214, 231
if self.decoder_idx == self.decoder_num_layers - 1:
    traj_scores = self.traj_mlp(...)   # 组合评分（每个 metric 一个 head，见 custom_decoder.py:144-150）
```

### 1.4 主干与多视角图像

Backbone 是 **ResNet-34**（timm）+ FPN，注意**未复用**原 SparseDrive 的 Swin 编码器，这里是轻量 ResNet-34。多视角默认 `cam_l0/cam_f0/cam_r0` 三路（左、前、右三个相机），forward 时压平送 backbone 再 reshape 回相机维度，供 deformable attention 按内外参采样：

```python
# sparsedrive_backbone.py:47-69
img = img.flatten(end_dim=1)   # [B, N_cam, 3, H, W] -> [B*N_cam, 3, H, W]
feat = self.backbone(img)
feat = feat.reshape(B, N_cam, C, h, w)
```

**什么叫 deformable attention（可变形注意力）？** 通俗讲：普通注意力让每个查询去看图像的所有像素，计算量巨大；可变形注意力让每个查询只去「它最该看的那几个位置」采样（位置由相机内外参和当前查询推测），省算力又贴合几何。SparseDriveV2 让候选轨迹嵌入作为查询，去三相机特征图上「按相机参数找对应的图像区域」，从而知道「这条路前面有没有车/路口」。

## 二、前向传播：一张图对应的代码路径（重点）

把上面所有模块串成一次 `forward`。输入是 navsim 的 `AgentInput`，输出是 `(B, 8, 3)` 的轨迹（8 个 pose）。下面**逐行跟一遍**推理时的 forward，并标注每一步张量形状变化，让你闭眼也能复现这条链路。

```
① compute_trajectory(agent_input)              # navsim 的 AbstractAgent 接口，每个评测场景调一次
|
`--> ② _build_features(agent_input)             # 把 AgentInput 拆成图像张量 + ego 状态向量
|       · img      : [B, 3, H, W]              # 三相机 cam_l0/f0/r0 已 resize/拼接
|       · ego     : [B, 8]                      # 速度4 + 朝向2 + 位置2
|
`--> ③ SparseDriveModel.forward(features)
     |
     |-- ④ SparseBackbone(img)                    # ResNet-34 + FPN
     |       -> img_feat : [B, N_cam, C, h, w]    # 多尺度特征，N_cam=3
     |
     |-- ⑤ status_encoding(ego)                   # nn.Linear(8, d_model)
     |       -> status_emb : [B, d_model]          # ego 状态投影到模型维度
     |
     |-- ⑥ 冻结 traj_vocab 查表 -> traj_embed       # 不走网络，直接索引 .npz 加载的嵌入
     |       path_embed : [B, 1024, d]            # 1024 条 path 原型嵌入
     |       vel_embed  : [B, 256, d]             # 256 条 velocity 原型嵌入
     |       （注意：这 262144 条"轨迹"不会真的全展开，模型用 path/vel 两个因子分别处理）
     |
     `-- ⑦ CustomTransformerDecoder(              # 2 层，这是"打分器"
             query  = path_embed / vel_embed,     # 初始 query = 冻结因子嵌入
             value  = img_feat,                   # deformable attention 去图像上采特征
             context= status_emb)
            |
            |-- 第 0 层 :
            |     · 对 path_embed 算 path_scores   -> top-k(64)  筛掉 1024->64 条 path
            |     · 对 vel_embed  算 vel_scores    -> top-k(20)  筛掉 256->20 条 vel
            |     · 此刻候选 = 64 x 20 = 1280 条组合轨迹
            |
            |-- 第 1 层 :
            |     · 再各 top-k 一次（配置 path_filter_num/velocity_filter_num）
            |     · 把筛后的 path_embed 与 vel_embed 逐元素相加 -> 组合轨迹嵌入
            |     · traj_embed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
            |     · traj_embed : [B, ~200, d]      # 粗筛后的 ~200 条组合候选
            |
            `-- 末层 (decoder_idx == num_layers-1) :
                  · 对 ~200 条候选各过一个 metric_head（每个 PDMS 指标一个 MLP）
                  · 输出 metric_logit : [B, ~200, num_metrics]
                  · 按 PDMS 同构公式合成最终分 scores : [B, ~200]
     |
     `--> ⑧ argmax(scores) -> 选 1 条
             trajectory = filter_traj_vocab.flatten(1,2)[batch_idx, mode_idx]
             -> 输出 : [B, 1, 8, 3]                 # 最终选中的最优轨迹
```

下面把每一步用**大白话 + 伪代码**拆开讲，并且解释「为什么这样设计」。

**第 ② 步：整理输入（把 navsim 的原始数据变成张量）**

navsim 给模型的原始输入 `AgentInput` 是个复杂对象，里面塞了相机图像、自车状态、地图等。这一步把它拆成模型真正要的两样东西：

```
img : [B, 3, H, W]      # B 是 batch 里场景数，3 是 RGB 三通道，H/W 是图像高宽
ego : [B, 8]            # 8 个数：速度相关 4 个 + 朝向相关 2 个 + 位置相关 2 个
```

为什么要拆？因为图像要送进视觉网络，自车状态要送进一个简单的小网络，二者处理方式完全不同。

**第 ④ 步：图像变特征（SparseBackbone）**

```
img        : [B, 3, H, W]
img_flat   = img.flatten(end_dim=1)            # [B*3, 3, H, W] 把 3 个相机压成一维
feat       = ResNet34_FPN(img_flat)            # 提特征
img_feat   = feat.reshape(B, 3, C, h, w)       # 还原回 [B, N_cam=3, C, h, w]
```

为什么用 ResNet-34 这么"老"的网络？因为 SparseDriveV2 主打"快"和"轻"，推理要实时，用大模型反而拖慢。FPN（特征金字塔）是把不同尺度的特征都保留下来，近处细节和远处轮廓都能看到。

**第 ⑤ 步：自车状态编码（status_encoding）**

```
ego       : [B, 8]
status_emb = Linear(8 -> d_model)(ego)         # [B, d_model]
```

为什么要把"我现在开多快、朝哪、在哪"也编码进去？因为规划不能脱离"当前处境"。比如你现在已经在高速最左道且时速 100，那"猛打右转"这种动作就该被压低分。这个向量作为 decoder 的全局上下文，相当于告诉模型"我现在的处境"。

**第 ⑥ 步：冻结词汇表查表（这是 SparseDriveV2 的灵魂）**

```
path_embed : [B, 1024, d]     # 1024 条路径原型，每条用 d 维向量表示
vel_embed  : [B, 256,  d]     # 256 条速度原型
```

注意：这些嵌入是直接从 `.npz` 文件加载的，**不经过任何网络计算，也不更新**。这步就是"查字典"——把预先聚类好的 path 和 velocity 原型取出来当查询。

**为什么叫"检索式"而不叫"生成式"？** 这是全文最该懂的一句话：DiffusionDrive 那种生成式模型，推理时是"从噪声出发，一步步画出一条可能字典里根本没有的新轨迹"；而 SparseDriveV2 在推理时，候选轨迹**早就全部躺在冻结词汇表里了**，模型唯一能做的只是"给这些现成候选打分、挑最好的一条"。它永远不可能输出字典之外的新轨迹。就像点外卖：生成式是"现场给你炒一盘新菜"，检索式是"从固定菜单里选一个"。所以 SparseDriveV2 把规划退化成了"检索+排序"问题。

**第 ⑦ 步：两级打分器（CustomTransformerDecoder，核心中的核心）**

这是全文最难也最重要的一步，拆成三层来看。

第 0 层（粗筛 path 和 vel 两个因子）：

```
# path 打分：让 1024 条 path 嵌入去图像+自车状态里"找相关性"，得到每条 path 的分数
path_scores = decoder_layer_0(path_embed, img_feat, status_emb)   # [B, 1024]
topk_path   = topk(path_scores, k=64)                              # 只留 64 条最好的 path
filter_path_embed = gather(path_embed, topk_path)                 # [B, 64, d]

# velocity 同理
vel_scores  = decoder_layer_0(vel_embed, img_feat, status_emb)    # [B, 256]
topk_vel    = topk(vel_scores, k=20)
filter_vel_embed = gather(vel_embed, topk_vel)                   # [B, 20, d]

# 此刻组合候选 = 64 x 20 = 1280 条
```

为什么要先粗筛？因为 26 万条全做细粒度打分，显存直接爆。先各砍一刀，把"明显不靠谱"的 path（比如当前该直行却选了急左转）和 vel（比如该慢却选了全速）砍掉，剩下的才值得精算。这是推荐系统里经典的"先召回后排序"两阶段思路。

第 1 层（再做一次粗筛 + 加法组合）：

```
# 对筛后的 64 条 path、20 条 vel 再各打一次分、再各 top-k 一次（配置决定具体数字）
filter_path_embed = topk_path_again(filter_path_embed)            # 比如 [B, 64, d]
filter_vel_embed  = topk_vel_again(filter_vel_embed)              # 比如 [B, 20, d]

# 加法组合：把每条 path 和每种 vel 配对
traj_embed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
# 形状: [B, 64, 1, d] + [B, 1, 20, d] -> 广播成 [B, 64, 20, d] -> flatten -> [B, 1280, d]
```

**path⊕vel 加法组合的实现细节（务必搞懂）**：`unsqueeze` 是在指定维度插一个大小为 1 的维度。path 是 `[B, 64, d]`，在第 2 维插 1 变成 `[B, 64, 1, d]`；vel 是 `[B, 20, d]`，在第 1 维插 1 变成 `[B, 1, 20, d]`。PyTorch 的广播机制会把这两个张量自动扩成 `[B, 64, 20, d]` 再做逐元素加法——结果就是"第 i 条 path 和第 j 种 vel 的嵌入相加"，得到第 (i,j) 个组合轨迹的嵌入。最后 `flatten(1,2)` 把 64×20 压成一维 1280，方便后面统一处理。这就是"因子化"在代码里的落地：用一次加法代替存 26 万个独立向量。

末层（细粒度精排，只在这一层做）：

```
if decoder_idx == num_layers - 1:
    metric_logit = traj_mlp(traj_embed)        # [B, 1280, num_metrics]
    # 每个 PDMS 指标一个 head，比如 no_at_fault_collisions / ego_progress ...
    scores = combine_by_PDMS_formula(metric_logit)   # [B, 1280]
```

为什么只在最后一层做细粒度打分？因为前面已经把候选从 26 万砍到 1280，现在才"值得"对每条都跑一遍多个指标的 MLP 精算。这就是"粗到细（coarse-to-fine）"省算力的精髓。

**第 ⑧ 步：argmax 选最优**

```
mode_idx   = scores.argmax(1)                  # 在 1280 条里选分数最高的那条的下标
trajectory = filter_traj_vocab.flatten(1,2)[batch_idx, mode_idx]   # 取对应真实坐标
output     : [B, 1, 8, 3]                      # 最终选中的最优轨迹（8 个 waypoint，每个 x/y/θ）
```

注意这里取的是**冻结词汇表里对应的真实轨迹坐标**，而不是模型"生成"的坐标。模型只负责打分，轨迹本身永远是字典里的原样。

**每一步在干啥（大白话版）**：
- **②④**：和所有视觉模型一样，把相机图变成"懂驾驶的多尺度特征"。
- **⑤**：把自车"我现在开多快、朝哪、在哪"压成一个向量，作为 decoder 的全局上下文（告诉模型"我现在的处境"）。
- **⑥**：这一步是 SparseDriveV2 的灵魂——它**不生成**轨迹，而是把 26 万条"标准驾驶动作"当成现成的 query 嵌入直接拿来用（冻结，不更新）。相当于给模型一张"标准动作菜单"。
- **⑦**：decoder 做的事不是"创造"，而是"翻菜单"——先看图像 + ego 状态，觉得哪类 path（左转/直行/…）和哪类 vel（快/慢/…）靠谱，用 top-k 把菜单从 26 万砍到 ~200；最后给这 ~200 条逐一打分（分数来自训练时蒸馏的 PDM 判分器）。
- **⑧**：直接选分数最高的那条，完事。

**输入 / 输出维度小结**：
- 输入图像：`[B, 3, H, W]`（三相机）；输入 ego 状态：`[B, 8]` → 投影到 `[B, d_model]`。
- 冻结轨迹词汇表：`traj_vocab [262144, 8, 3]`（始终 `requires_grad=False`）。
- 中间候选：26 万（因子形式）→ 第0层后 1280 → 末层前 ~200。
- 输出：`[B, 1, 8, 3]` 最优轨迹 + 对应的 metric 分数 `[B, ~200]`（推理时用于 argmax 排序）。

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

### 3.1 词汇表怎么预计算（k-means 在训练集上聚类）

回顾 1.2 节：词表不是训练出来的，而是**离线**用 KMeans 在训练集所有真实轨迹上聚类得到的。具体做法：

1. 把训练集里每条专家轨迹拆成「几何路径 path」（只看 x/y/θ 形状，忽略速度）和「速度剖面 velocity」（每个时刻的速度大小）。
2. 对所有 path 做 KMeans，聚成 1024 类，每类的中心就是一条代表路径。
3. 对所有 velocity 做 KMeans，聚成 256 类，每类的中心就是一种代表速度剖面。
4. 两两组合 + 沿弧长插值，生成 `(1024, 256, 8, 3)` 的完整轨迹张量，存成 `.npz`。

这一步**只在准备数据时跑一次**（`scripts/cluster/cluster_anchor.py`），之后训练和推理都只 `np.load` 这个 `.npz`，再也不碰聚类脚本。所以词表是"经验统计得到的先验"，不是神经网络参数。

### 3.2 损失函数（梯度流向）

损失全在 `custom_decoder.py:236-288`，分三族：

- **Path 分类损失**（soft target，对最近 GT path 做 softmax 距离）：
  ```python
  # custom_decoder.py:242-252
  dist = (path_vocab - target_path[:, None])[..., :2].pow(2).sum(-1) * mask
  dist = dist.sum(-1) / valid_cnt * path_sigmas * len_path
  path_loss = F.cross_entropy(path_scores, (-dist).softmax(1))
  ```
  大白话：Ground Truth（专家真实轨迹）对应的 path 应该分数最高。这里不是硬 label，而是用"到各 path 中心的距离"做 soft 标签——离 GT 越近的 path 越该高分。

- **Velocity 分类损失**、**组合轨迹分类损失（IMI imitation）** 同理，都是让模型给"像专家"的候选打高分。

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

  **什么叫"蒸馏（distillation）"？** 通俗讲：有一个很强但不便部署的"老师"（这里就是 NAVSIM 的 PDM 仿真器，它能精确模拟车开下去的后果并打分），让它对每条候选轨迹给出各项分数，然后训练一个轻量的"学生"网络（metric head）去模仿老师的分数。这样推理时学生就可以自己打分，不用再跑笨重的 PDM 仿真器——又快又贴榜。

> **这是 SparseDriveV2 的杀手锏**：它直接拿排行榜的判分函数（PDM）当监督信号，让 metric head 学会"什么轨迹分高"——推理时就能用这些 head 合成 PDMS 式分数来排序，等于"在训练时就把排行榜逻辑学进去"。

**梯度流小结**：只有 backbone、`CustomTransformerDecoder` 内的 path/vel/metric head 参数可训；`traj_vocab` 全程冻结（仅作 query 查表）。三族损失加权求和后做反向传播，AdamW 优化。

**哪些部分冻结不训**：`path_vocab`、`vel_vocab`、`traj_vocab`、`traj_mask` 全部 `requires_grad=False`。也就是说"动作菜单"是固定不变的，模型只学"怎么给菜单里的动作打分"和"怎么从图像里提取特征"。

## 四、推理：两级评分选最终轨迹

`compute_trajectory` 继承 `AbstractAgent`，真正「选轨迹」逻辑在 `custom_decoder.py:291-317`（推理最后一层）：

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

**推理和训练的差异**：
- 训练时，模型需要 PDM 仿真器来提供"老师分数"做蒸馏损失；推理时**完全不需要 PDM**，直接用训练好的 metric head 自己算分数。
- 训练时算三族损失（path/vel/traj 分类 + 蒸馏）；推理时只跑前向、做两级筛选和 argmax，不反向传播。
- 训练时候选可能还要算完整 26 万的粗筛路径；推理时同样走两级筛选，但最后只从 ~200 条里选 1 条。

流程和 PDMS 公式**同构**：乘性项（NC/DAC/DDC/TLC 的 sigmoid 乘积）× 加权项（TTC/EP/LK/HC 加权求和）。最终 argmax 从粗筛后的 ~200 条里选 1 条。

**PDM 分数怎么用在推理？** 关键点：推理时并没有真的调用 PDM 仿真器，而是用训练时蒸馏好的 metric head 直接输出每个指标的 logit，再按 PDMS 的同构公式合成 `scores`。换句话说，PDM 的"打分逻辑"已经被压缩进了这些 head 的权重里——这就是蒸馏的价值：把慢但准的仿真器，变成快且贴榜的小网络。

> **注意**：生成 challenge `submission.pkl` 的脚本（仓库引用的 `run_create_submission_pickle_challenge.py`）在本 fork 内未包含，需依赖官方 navsim devkit；常规 navtest 评测走内联的 `run_pdm_score_navtest_v1_fast.py`。

## 闭环总结

| 阶段 | 入口 | 关键代码 | 训练/推理差异 |
|:----:|:------|:----------|:--------------|
| 架构 | `sparsedrive_model.py` + `scripts/cluster/cluster_anchor.py` | 冻结 path(1024)×vel(256)=26万 轨迹词汇表 | 词汇表离线生成，两侧共用 |
| 训练 | navsim `run_training.py` + `agent=sparsedrive_agent` | path/vel/traj 分类 + **PDM score 蒸馏** | 额外回传 PDM 算 metric GT |
| 推理 | `custom_decoder.py:291` | 两级评分：粗筛 200 条 → PDMS 式分数 argmax | 无需 PDM，直接 metric head 打分 |

**一句话记住这个闭环**：离线用 KMeans 建好「path×velocity」的 26 万条轨迹字典，训练时让模型学会给每条打 PDM 式分数（蒸馏自排行榜判分器），推理时先按 path/vel 粗筛到 200 条、再用分数精排取最优——这就是 SparseDriveV2 「检索式规划」的全部秘密。

## 个人理解与思考

**1. 它把「规划」问题偷换成了「检索」问题。** 这是 SparseDriveV2 最聪明也最危险的一点。聪明在于：排行榜（PDMS）本身就是在一个固定候选集上打分选优，SparseDriveV2 直接让模型学这个打分函数，训练目标与评测指标**完美对齐**，所以榜上分数高毫不意外。危险在于：如果最优轨迹恰好不在 26 万条字典里（极端场景、字典覆盖盲区），模型**无能为力**——它只能选「字典里最好的」，不能「生成字典外更好的」。这是离散候选集方法的天花板。

**2. 因子化是性价比极高的归纳偏置。** 26 万条组合轨迹只用了 1024+256 两个聚类中心就覆盖，参数几乎不增。相比 Hydra-MDP 直接枚举 8K 固定轨迹，SparseDriveV2 用相近成本拿到了 32× 的密度。这提醒我们：在动作空间有明显「几何 × 速度」结构时，因子化分解通常比扁平枚举好得多。从数学上看，扁平枚举要存 $O(M)$ 个向量，因子化只需 $O(\sqrt{M})$ 量级的两个因子，覆盖密度却是指数级的。

**3. 两级评分是「精度-算力」的务实妥协。** 直接对 26 万条做细粒度 metric 打分显存爆炸；粗筛先砍到 200 条再精排，把计算量压了三个数量级。这是工程上很典型的「先召回后排序」两阶段架构（和推荐系统如出一辙）。缺点是粗筛用 path/vel 独立打分，可能误杀「path 一般但 vel 绝配」的组合——不过实践中 200 的保留量足够兜底。

**4. 和 DiffusionDrive 的对比给人的启发。** DiffusionDrive 是「生成式」——从 anchor 扩散出轨迹，擅长覆盖字典外的多模态；SparseDriveV2 是「检索式」——在巨字典里选，擅长贴合评测指标且推理极快（无迭代去噪）。**没有谁绝对更好**：要榜单分数 → 检索式更稳；要开放场景泛化 → 生成式更灵活。这也是 NAVSIM 榜单上两条路线并存的底层原因。

**5. 对工程落地的启示。** SparseDriveV2 把"规划"做成"查表+打分"，推理时没有迭代去噪、没有自回归生成，延迟极低，非常适合车端实时部署。但它的天花板也写在架构里：字典一旦冻结，能力上限就锁死了。如果后续想覆盖新场景，只能重新跑 KMeans 扩字典——这比"让生成模型多学点"要麻烦。所以选路线时要想清楚：你要的是"在已知分布里做到极致快和准"，还是"在未知分布里也能灵活应对"。

## 延伸阅读

- 同系列对比：[DiffusionDrive 代码讲解](/posts/code/diffusiondrive代码讲解/)——扩散生成式路线
- 榜单背景：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
