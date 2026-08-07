---
title: "NAVSIM 详解：自动驾驶规划的事实标准评测基准"
date: 2026-07-19
draft: false
categories: ["知识点拆解"]
tags: ["📊 NAVSIM", "🏆 评测基准", "📐 规划", "🚗 自动驾驶"]
summary: "NAVSIM 是当前自动驾驶规划领域最重要的开环评测基准，几乎所有端到端新工作都在上面比拼 PDMS 分数。它基于 OpenScene 数据设计了非反应式仿真框架，以 PDMS/EPDMS 为核心指标抛弃传统 L2 误差思路。是端到端规划从学术研究走向工业落地的关键度量工具。"
weight: 6
---

## 🎯 NAVSIM 是什么？

> **NAVSIM = Non-reactive Autonomous Vehicle Simulation，一个"非反应式"的自动驾驶开环评测框架与基准。**

如果你这两年读端到端自动驾驶的论文，几乎每一篇都会报一个叫 **PDMS** 的分数，这个分数就来自 NAVSIM。它由图宾根大学、NVIDIA Research、OpenDriveLab 联合提出（Daniel Dauner 等，NeurIPS 2024），目前已经是端到端规划领域**事实上的标准评测**，也是智驾挑战赛（AGC）的官方赛道。

![NAVSIM 核心理念：传统 L2 指标忽视驾驶的多模态性，NAVSIM 用 PDMS 直接衡量"开得好不好"](/images/navsim/navsim_concept.png)

NAVSIM 的几个关键事实：

| 维度 | 说明 |
|------|------|
| **数据来源** | **OpenScene** 数据集，本质上是 **nuPlan** 日志降采样到 **2 Hz** 的子集 |
| **任务** | 给定历史观测（多相机/LiDAR），预测自车未来 **4 秒**轨迹 |
| **仿真方式** | **非反应式（non-reactive）**：背景车按录制轨迹走，自车用 LQR 控制器执行预测轨迹 |
| **核心指标** | **PDMS**（v1）/ **EPDMS**（v2） |
| **发表** | NeurIPS 2024 Datasets & Benchmarks |

理解 NAVSIM 的关键在于"**非反应式**"三个字：自车虽然会把轨迹"开出来"（4 秒前向仿真），但周围车辆、行人**不会因为自车的动作而改变行为**——它们只是回放历史轨迹。这是一种"半开环"设计：既比纯位移误差更接近真实（轨迹要经过物理执行和碰撞检查），又避免了闭环仿真的高昂成本。

一句话定位：**NAVSIM 用开环的代价，换取尽可能接近闭环的信号。**

---

## 📏 核心指标 PDMS 怎么算？

**PDMS（Predictive Driver Model Score，预测驾驶模型得分）** 是 NAVSIM v1 的核心分数，它最大的贡献是**抛弃了与人类轨迹比 L2 误差的思路**，转而直接衡量规划的"质量"。

### 传统 L2 指标的致命缺陷

在 nuScenes 上，规划指标是 **L2 位移误差 / 碰撞率**。但驾驶本质是**多模态**的——前方有障碍时，左变道和右变道都合理。如果你预测了"绕行"，而人类 GT 是"刹车"，L2 会判你错，可你其实开得很好。更糟的是，**"原地不动"几乎永远 L2 最低**，于是模型学会了"躺平"刷分。

### PDMS 的设计哲学

PDMS 的思路是：**不跟人比像不像，而是看这条轨迹本身安不安全、舒不舒服、有没有往前走。** 它由两部分组成——**乘性惩罚（multiplier）** 和 **加权子分（weighted）**：

$$\text{PDMS} = \underbrace{\left(\prod_{m \in \{NC,\, DAC\}} m(\text{agent})\right)}_{\text{乘性惩罚：撞了/越界直接清零}} \cdot \underbrace{\left(\frac{\sum_{m \in \{TTC,\, EP,\, C\}} w_m \cdot m(\text{agent})}{\sum w_m}\right)}_{\text{加权平均：行驶质量}}$$

- **乘性部分**：一旦发生无责碰撞或驶出可驾驶区域，整条轨迹**直接得 0 分**——这是"安全一票否决"。
- **加权部分**：在安全的前提下，再综合"是否前进、是否留有安全距离、是否舒适"给出质量分。

这样设计的好处非常明显：**"原地不动"虽然不撞，但 EP（前进量）极低，拿不到高分；合理的绕行即便偏离人类 GT，只要安全舒适、有进展，照样得高分。**

### 一次评测到底发生了什么？

具体到每条样本，NAVSIM 的评测流水线是这样的：取一帧初始场景 → 让被测模型预测未来 **4 秒**（共 8 个 0.5 秒的时间点）的自车轨迹 → 用一个 **LQR 横向控制器**把这条几何轨迹"翻译"成实际可执行的车辆运动（生成真实的位置、速度、加速度序列）→ 与同样回放 4 秒的背景交通流做**碰撞检测与合规检查** → 算出各个子分并按上式合成 PDMS。整个过程**无需模型在线推理、无需交互**，所以可以像开环一样一次性并行算完整个测试集，这也是它比闭环快一个数量级的根本原因。

---

## 🔍 子指标详解

PDMS 由若干子分组合而成。理解这些子分，你才能看懂论文里的 ablation 在说什么。

| 子指标 | 全称 | 类型 | 取值 | 权重 | 衡量什么 |
|--------|------|------|------|------|---------|
| **NC** | No at-fault Collisions | 乘性 | {0, ½, 1} | — | 是否与动态/静态障碍发生**有责碰撞** |
| **DAC** | Drivable Area Compliance | 乘性 | {0, 1} | — | 是否始终在**可驾驶区域**（车道/路面）内 |
| **EP** | Ego Progress | 加权 | [0, 1] | 5 | **前进量**：实际行驶距离与参考的比值 |
| **TTC** | Time-to-Collision | 加权 | {0, 1} | 5 | 是否始终保留足够**碰撞时间**余量 |
| **C** | Comfort | 加权 | {0, 1} | 2 | 加速度/急动度（jerk）是否在**舒适阈值**内 |

几个要点深入解读：

- **NC 的"无责"很关键**：NAVSIM 区分"责任"。如果碰撞是背景车突然冲出来造成的（自车无法避免），不算自车的错。这避免了把不可抗力的碰撞算到规划器头上。
- **EP 是反"躺平"的利器**：它正比于自车沿参考线的前进距离。原地不动 EP→0，直接压低加权平均分。这也是 NAVSIM 区别于 L2 的核心。
- **TTC 是前瞻安全**：不只看当前这一步撞没撞，而是检查未来每个时间点的碰撞时间是否高于阈值，鼓励"留余地"的规划。
- **C 关注体感**：对纵向/横向加速度和 jerk 设硬阈值，超过即不舒适。NAVSIM v2 进一步把它升级成 **HC（History Comfort）**，还新增 **EC（Extended Comfort）**，检查相邻帧输出轨迹的动态一致性，惩罚"忽左忽右"的抖动。

### NAVSIM v2 的 EPDMS 扩展

2025 年的 v2 版本把指标升级为 **EPDMS（Extended PDMS）**，新增两个乘性项和三个加权项：

| 新增子指标 | 类型 | 含义 |
|-----------|------|------|
| **DDC** Driving Direction Compliance | 乘性 | 是否**逆行/走错方向** |
| **TLC** Traffic Light Compliance | 乘性 | 是否**闯红灯** |
| **LK** Lane Keeping | 加权(2) | 是否长时间**偏离车道中心** |
| **HC** History Comfort | 加权(2) | 含历史运动一致性的**舒适度** |
| **EC** Extended Comfort | 加权(2) | 相邻帧输出的**动态一致性** |

此外，v2 引入 **误报惩罚过滤**：当人类驾驶员本身在该场景也会违规时（如借对向车道绕障），不再惩罚规划器，避免"按规矩开车反而被扣分"。还引入了**伪闭环（pseudo closed-loop）两阶段聚合**：第一阶段在初始场景评测；第二阶段额外准备了一组**预滚动的 follow-up 场景**（每个对应一条不同的 4 秒规划），再根据被测模型第一阶段实际终点与各 follow-up 起点的**高斯核距离**加权汇总，最后两阶段相乘。这样无需真正交互，也能近似"偏离之后会怎样"，进一步缩小开环与闭环的差距。

---

## 📂 navtrain / navtest / navhard 数据划分

NAVSIM 的评测公平性，很大程度上来自它精心设计的 **split（数据划分）**。这些 split 都是对 OpenScene（nuPlan 2Hz）做**场景过滤**得到的。

| Split | 来源 | 用途 | 特点 |
|-------|------|------|------|
| **navtrain** | trainval | **训练** | 过滤掉无聊场景，只保留**有挑战性的非平凡驾驶**；传感器数据 445GB（带历史） |
| **navtest** | test | **NAVSIM v1 标准测试** | 同样过滤为有挑战场景，是 v1 排行榜（PDMS）的评测集 |
| **navhard_two_stage** | test + 合成帧 | **NAVSIM v2 标准测试** | 含**合成观测**的伪闭环两阶段评测，对应 EPDMS 排行榜 |
| **warmup / private_test** | — | 挑战赛 | 热身赛与私有测试集，**禁止用于训练** |

### 为什么要"过滤"？

nuPlan 原始日志里，绝大多数场景是**直道匀速、无交互**的"无聊"片段。如果直接拿来评测，模型只要输出"恒速直行"就能拿高分。NAVSIM 的 **scene filter** 专门挑选那些**"恒速恒向"会失败**的场景——比如路口转弯、变道、减速避让、应对行人。这样刷出来的分数才有区分度。

### navtrain vs navtest 的本质区别

- **navtrain** 是给模型**学习**用的，场景丰富但分布相对平缓；
- **navtest** 是**只评不训**的盲测集，分布更偏**安全关键与长尾**。
- 两者都来自过滤，但**完全独立、无重叠**，navtest 禁止用于训练，否则成绩无效。

一句话：**navtrain 教模型"怎么开"，navtest 考模型"关键时刻开不开得对"。**

值得一提的是三者之间的**演进逻辑**：navtest 是 v1 时代的"一锤定音"；navhard 则是 v2 为治理"开环刷榜"而引入的升级版——它不仅包含真实帧，还**掺入合成观测**，并要求两阶段伪闭环评测，专门用来暴露纯开环看不到的规划脆弱性。所以当你看到论文报分时，一定要分清是 **PDMS（navtest，v1）** 还是 **EPDMS（navhard，v2）**，后者通常比前者低 **3–10 分**，含金量更高。

---

## 🗃️ NAVSIM 数据到底长什么样？从源码看一个 scene

上面说了切分，但真正的难点在于**一个"样本"（scene）具体包含哪些数据、模型训练时又能看到什么**。这一节全部依据 NAVSIM 官方源码 `navsim/common/dataclasses.py`、`navsim/common/dataloader.py` 和 `navsim/planning/dataset` 逐行还原。

### 数据的三层组织：log → frame → scene

数据不是整卷倒给模型的，而是按**三层组织**：

```
OpenScene（=nuPlan 降采样到 2Hz）
   └── log     一段完整行车记录（约几百到上千帧 @2Hz）
          └── frame  一帧（2Hz 采样，间隔 NAVSIM_INTERVAL_LENGTH=0.5s）
              └── scene  以一个「带路由的帧」为中心截取的时序窗口 = 基本训练/评测样本
```

关键点：

- **log 是下载/切分的最小单位**：`train_logs`（13,180 段）、`val_logs`（2,730 段）都按整段 log 分，**不按帧切**。这样同一段行车记录不会泄漏到训练和验证两边。
- **scene 是训练/评测的最小单位**：训练 batch 里装的是一个个 scene。一个 scene 由 `SceneMetadata`（log 名、scene token、地图名、初始帧 token、历史/未来帧数）+ `map_api`（地图）+ **一列 `Frame`** 组成。
- **scene 由滑窗采样生成**：`split_list(input_list, num_frames, frame_interval)` 会从 log 里每 `frame_interval` 帧取一个长度为 `num_frames` 的窗口。默认 `frame_interval=1`、`num_frames = 4 + 10 = 14`（4 历史 + 10 未来），所以**重叠的帧可以出现在多个 scene**，这正是 103,288 帧比 log 数高很多的原因。

![NAVSIM 数据加工链：OpenScene → scene 采样 → 过滤 → 各数据集](/images/navsim/navsim_data_chain.svg)

### 一个 frame 里的五个内容块

每个 `Frame`（`dataclasses.py` 的 `Frame`）包含五类字段，可用 `Scene.from_scene_dict_list()` 从 log 装载：

| 内容块 | 类型 | 作用（是观测还是特权） |
|--------|------|----------------------|
| `token` / `timestamp` | 基础 | 唯一标识（即该帧 LiDAR 点数 token） |
| `roadblock_ids` | `List[str]` | 该帧自车所在车道，后续导出**路由**（特权） |
| `traffic_lights` | `List[(lane_connector, bool)]` | 路口信号灯状态，`True`=红（特权） |
| `annotations` | `Annotations` | **包围框真值标定**（特权） |
| `ego_status` / `lidar` / `cameras` | 观测 | 模型可见输入 |

### 标定信息（Annotations）——"人类专家标注了什么"

这就是你问的"**人类专家轨迹/标定**"落点。`Annotations` 类（`dataclasses.py:260`）在一个场景里携带五段等长数组，**逐对象对齐**：

- `boxes`：3D 包围框，尺寸/朝向。姿态用 `BoundingBoxIndex` 索引：`(x, y, z, len, width, height, heading)`。
- `names`：**类别标签** `List[str]`，包含 vehicle、pedestrian、bicycle、traffic_cone、barrier、czone、general 等。
- `velocity_3d`：3D 速度向量。
- `instance_tokens` / `track_tokens`：跨帧计数与追踪的恒定 ID，供时序关联。

评测/训练用这些真值框前后对齐成**跟踪轨迹**。但注意 **NAVSIM v1/v2 评测不直接要求模型做感知**：打分时碰撞、车道内等用这些真值框离线算，模型自己只需输出未来的自车几何轨迹。

### 观测数据（AgentInput）——模型真正拿到的东西

`Scene.get_agent_input()` 从 scene 里抽出模型可见的观测，组成一个 `AgentInput`，**不包含任何特权信息**：

- **EgoStatus**（当前帧 + 历史帧）：自车 `ego_pose`（3 维）、`ego_velocity` 与 `ego_acceleration`（各 2 维），都转到局部坐标系。
- **Cameras × 8**：`cam_f0 / cam_l0 / cam_l1 / cam_l2 / cam_r0 / cam_r1 / cam_r2 / cam_b0`，每个含 image + `sensor2lidar` 外参 + 内参 + 畸变。
- **LiDAR 合并点云**：5 个 LiDAR 融合为一次 `(6, n)` 数组 `+x,y,z,intensity,ring,lidar_id`，频率对标 2Hz。
- **driving_command 离散意图**：左变道 / 直行 / 右变道（由**期望路由**推出，`EgoStatus` 装载），另有第 4 档"未知"用于训练时过滤歧义样本。

NAVSIM 的观测设计很克制：**相机+LiDAR 只覆盖 2 秒过去 / 2Hz**（每帧 4 帧历史观测）；并刻意用一张图说明了：排列在图里的这些**只有测试帧才释放**——**地图、tracks、occupancy 等你在训练可以用，但排行榜提交时拿不到**，防止做题式作弊。

![一个 NAVSIM scene 里：观测 vs 特权 vs 训练标签](/images/navsim/navsim_frame_content.svg)

### 训练标签：人类专家 GT 轨迹

训练时由 **TrajectoryTargetBuilder** 通过 `Scene.get_future_trajectory()` 取出**人类驾驶员真车未来轨迹**作为回归目标：把未来 5s（10 帧）自车 GEOM 姿态转换到当前后轴为原点的局部坐标。这正是 EgoStatusMLP、TransFuser 等 baseline 做行为克隆（behavior cloning）时监督的标签——"人类专家开成了什么样"。

---

## 🏭 NAVSIM 训练流水线：从 scene 到 cache 到 batch

NAVSIM 提供一套轻量训练接口。它会先把原始 scene 预处理成**特征/标签缓存（feature/target cache）**，训练时直接从缓存读张量，避免每轮重复解析日志与大传感器。

### 1. Feature builder 与 Target builder

学习型 Agent 需实现两个接口（`docs/agents.md` 与 `planning/training`）：

- **`get_feature_builders()`**：把 `AgentInput`（观测）变成特征张量。内置两个示例——
  - `EgoStatusFeatureBuilder`：输出 `[velocity(2), acceleration(2), driving_command(1)]` 拼接的 5 维向量；
  - `TransfuserFeatureBuilder`：输出前相机图、LiDAR BEV map、ego status 三块。
- **`get_target_builders()`**：把 `Scene`（含特权 GT）变成监督标签，`TrajectoryTargetBuilder` → `trajectory` 目标张量 `[T, 3]`（x,y,heading）。

### 2. CacheOnlyDataset：无日志直接训练

`navsim/planning/training/dataset.py` 的 **`CacheOnlyDataset`** 是训练的主力。它从一个 **cache 目录**构造 dataset：目录按 `log_name / token / <builder名>.gz` 组织，每个 `.gz` 是 **gzip + pickle** 压缩的特征或目标字典（`compresslevel=1`）。`__getitem__` 依次加载每个 builder 的文件，合并得到 `(features, targets)` 字典——**训练过程中根本不再碰原始 sensor 数据**。

`Dataset`（非 pure-cache 版）则支持两种模式：
- 给了 cache 路径：与 CacheOnlyDataset 类似，若非全缓存则先跑一段**离线/在线 caching**（`run_dataset_caching.py` 可并行预热大数据集）；
- 不给 cache：每个 batch 实时 `get_scene_from_token()` 解析 scene。

### 3. 一个完整的 EgoStatusMLP 示例

最简 baseline 把 Agent 当"盲司机"：忽略全部传感器，只用当前 `[速度, 加速度, driving_command]` 预测未来轨迹。源码可见其 MLP 输入是 **8 维**（`Linear(8, hidden)`），输出 `num_poses×3`：

```text
feature = [ego_velocity(2), ego_acceleration(2), driving_command(1)]
                                    │
                                    ▼
                    Linear(8→hidden) →ReLU →Linear →ReLU →Linear
                                    │
                                    ▼
              reshape → [num_poses, 3]  (x, y, heading 未来轨迹)
label  ← get_future_trajectory()  （人类专家 GT)
```

它虽然"盲"，但常被当作**kinematic 上限**：只外推状态就能拿到的 PDMS，衡量"不看四周能开多好"。

### 4. 评测不再走训练缓存：MetricCache

评测和训练分开。评测前先做 `metric_cache`（`nav/planning/metric_caching/metric_cache.py`），把每条样本的**静态计算量** lzma 压缩落地，评测时直接读入，避免重复算地图、中心线：

| MetricCache 字段 | 内容 |
|------------------|------|
| `trajectory` | 参考轨迹（PCM 决策器提出的提案） |
| `human_trajectory` | 人类专家 GT 轨迹 |
| `past_human_trajectory` | 人类历史轨迹 |
| `observation` / `centerline` / `route_lane_ids` / `drivable_area_map` | 观测 + 参考中线 + 路由车道 + 可驾驶区域 |
| `past/current/future_tracked_objects` | 过去(2Hz)/当前/未来(10Hz)目标轨迹 |
| `ego_state` / `map_parameters` | 自车状态 + 地图参数 |

这样一个 cache 就够跑 PDM 打分：模型轨迹 + cache 里的参考轨迹送入 `PDMSimulator` 控制器，再配合 `traffic_agents_policy` 的背景交通流完成碰撞与合规检查。

### 预计算为何重要？

评测时我们要对**整个 navtest(12k 帧) 或 navhard(真实+合成)** 逐帧算 PDMS。地图解析、中心线抽取、可驾驶区域这些与模型无关的昂贵计算，多跑一次只会浪费算力。NAVSIM 改成**一次 cache、多次评测复用**，这是它评测高效、可大规模并行的工程基石。

---

## 🆚 为什么 NAVSIM 比 nuScenes 的 L2 更好？

这要从 L2 误差的根本问题说起。

| 对比维度 | nuScenes L2 误差 | NAVSIM PDMS |
|---------|-----------------|-------------|
| **衡量对象** | 与人类 GT 的**位移差** | 轨迹本身的**安全/舒适/进展** |
| **多模态** | ❌ 把合理替代方案判为错 | ✅ 只要合理就给分 |
| **"躺平"问题** | ❌ 原地不动 L2 最低，最"优" | ✅ EP 惩罚，不动拿低分 |
| **分布偏移** | 严重：自车轨迹几乎总在 GT 流形上 | 缓解：不要求贴合 GT |
| **与闭环相关性** | **几乎无相关** | **有正相关** |

![开环指标与闭环得分的相关性：PDMS（绿）与闭环分数有正相关，而传统 OLS/L2（橙）几乎无关](/images/navsim/navsim_closedloop_alignment.png)

NAVSIM 论文里一个核心实验（见上图）：他们把多个规划器放进真正的闭环仿真，发现**传统 L2/OLS 指标和闭环分数几乎零相关**——一个 nuScenes 上 L2 最低的模型，闭环里可能是个马路杀手；而 **PDMS 与闭环分数呈现明显的正相关**。这正是 NAVSIM 迅速取代 nuScenes、成为端到端规划标配的原因：**它在开环代价下，给出了一个真正"预言"驾驶能力的信号。**

> 💡 一句话：**nuScenes 的 L2 衡量"像不像人开"，NAVSIM 的 PDMS 衡量"开得好不好"——后者才是我们真正关心的。**

---

## 🏆 当前 NAVSIM 排行榜 Top（2025–2026）

下面是 NAVSIM 上有代表性的 SOTA 工作（分数为公开报告值，含 v1 的 PDMS 与 v2 的 EPDMS）。注意榜单迭代极快，半年就有新王者。

| 排名 | 模型 | PDMS(v1) ↑ | EPDMS(v2) ↑ | 关键技术 |
|------|------|-----------|------------|---------|
| 1 | **CLOVER** | **94.5** | **90.4** | 候选生成+打分器闭环自蒸馏 |
| 2 | **CLEAR** | 93.7 | — | VAE 单步漂移 + Drive-JEPA + 小 LLM |
| 3 | **DriveVLA-W0** | 93.0 | 86.1 | 世界模型预训练 + AR 动作头（单目） |
| 4 | **LWDrive** | 92.0 | 89.6 | 世界模型引导的层级化规划 |
| 5 | **ASSCG** | 91.4 | — | 快慢系统自适应 Query 门控 |
| 6 | **D³-MoE** | 91.3* | 87.5* | 解耦扩散 MoE + 风格控制 |
| 7 | **HiST-VLA** | — | 88.6 | 层次化时空 VLA |
| 8 | **WoTE** | 88.3 | — | 面向思考的端到端 |
| 9 | **DiffusionDrive** | 88.1 | 84.5 | 扩散头多模态轨迹生成 |
| 10 | **S-squared-VLA** | 87.1 | — | 语义/空间解耦 VLA |
| — | **OneDrive** | 86.8 | — | 统一多范式 VLA |
| — | **UniAD** | 83.4 | — | 经典基线（2023） |
| 参考 | **人类驾驶** | ~94.8 | — | 性能天花板 |

\* D³-MoE 为 Best-of-Three 集成分数。

几点趋势观察：

- **天花板逼近人类**：头部工作（CLOVER 94.5）已非常接近人类参考分（~94.8），navtest 上的 PDMS 几乎**快要刷满**。
- **战场转移到 v2/navhard**：因为 v1 接近饱和，2025 年起竞争焦点转向 **EPDMS** 和**伪闭环 navhard**（更难、含分布外场景），顶尖模型的 EPDMS 仍在 90 上下挣扎。
- **方法论收敛**：榜单前列几乎全是 **"生成式多模态候选 + 打分挑选" + 大模型/世界模型"** 的组合，纯确定性回归已被淘汰。
- **安全类子分饱和**：NC、TTC、Comfort 在 SOTA 模型上普遍接近满分，真正拉开差距的是 **EP（前进量）**——敢不敢开得快、开得远。

---

## ⚠️ NAVSIM 的局限性

NAVSIM 再好用，也**不是闭环**。认清它的边界，才能正确解读分数。

| 局限 | 说明 | 后果 |
|------|------|------|
| **非反应式** | 背景车不回应自车 | 自车激进加塞也不会被撞回来，**高估安全性** |
| **无误差恢复** | 只仿真 4 秒，不递归 | 看不到"小偏差滚雪球成大事故"的**复合误差** |
| **分布偏移** | 评测仍在数据集分布内 | OOD（真正长尾）场景的失败**完全暴露不出来** |
| **传感器固定** | 无天气/光照/对抗扰动 | 感知鲁棒性测不到 |
| **开环打分本质** | 仍是非交互式 | "刷榜技巧"（针对指标过拟合）依然可能 |

最典型的反例：一个在 NAVSIM 上拿 90+ 分的模型，闭环里可能因为**第一次轻微偏离车道后，场景就跑出训练分布**而彻底崩溃——这就是所谓的 **compounding error（误差累积）** 雪球效应。NAVSIM 的非反应式设计**根本触发不了这种失败模式**。

> ⚠️ **记牢：NAVSIM 高分是"能规划"的必要条件，远非"能开车"的充分条件。**

---

## 🔗 与 Bench2Drive 等闭环评测的互补关系

正因为 NAVSIM 有上述局限，社区形成了"**开环 + 闭环**"双轨评测的共识，而 **Bench2Drive** 正是闭环侧的代表。

| 维度 | NAVSIM（开环/半开环） | Bench2Drive（闭环） |
|------|----------------------|---------------------|
| **底层** | OpenScene / nuPlan 真实日志 | **CARLA** 仿真器 |
| **交互** | 非反应式 | **完全反应式**，交通流实时响应 |
| **指标** | PDMS / EPDMS | Driving Score（完成率×合规） |
| **成本** | 低，提交预测即可 | 高，需模型在线推理 rollout |
| **覆盖** | 真实数据分布 | 44 条预定义路线、9 类能力的**场景库** |
| **强项** | 真实感、大规模、快速迭代 | **真能跑通**、能暴露累积误差与交互失败 |

两者的互补关系可以这样理解：

- **NAVSIM 负责"快速筛选"**：成本低、迭代快，适合算法研发的日常打磨。一个新想法先在 NAVSIM 上验证有没有前途。
- **Bench2Drive 负责"终极考核"**：只有在闭环里真正跑通 44 条路线、低碰撞高完成率，才算"能开车"。

有意思的是，2026 年的一项跨基准研究（arXiv:2605.00066）系统比较了两者，得出三个关键结论：

1. **PDMS 与闭环 Driving Score 强正相关但不单调**——存在明显的排名反转；
2. **EP（前进量）是闭环成功最强的单项预测器**，甚至比碰撞指标 NC 更能预言闭环表现；
3. **TTC 和 Comfort 在 SOTA 上已接近饱和**，对区分模型贡献很小。

这恰好印证了"开环负责精度与质量、闭环负责行为与交互"的分工。

### 理想的评测组合拳

```
NAVSIM（开环，PDMS）  →  快速验证规划质量、刷算法迭代
Bench2Drive（闭环，DS） →  暴露累积误差、验证真能开
实路路测（影子模式）   →  真正长尾、天气、对抗场景
```

三者缺一不可：**开环看精度 + 闭环看行为 + 实路看长尾**。NAVSIM 不是终点，而是这条评测链上最关键、最高效的第一环。

---

## ✅ 小结

抓住这三点，你就抓住了 NAVSIM 的精髓：

1. **本质** = 基于 nuPlan/OpenScene 的**非反应式开环评测**，用 4 秒前向仿真让轨迹"开出来"再做碰撞/合规检查，成本远低于闭环。
2. **指标** = **PDMS / EPDMS**，由"乘性安全惩罚（NC、DAC）"和"加权质量分（EP、TTC、Comfort）"组成，**抛弃 L2 误差**，直接衡量"开得好不好"，并与闭环分数正相关。
3. **定位** = 当前端到端规划的**事实标准**，navtest 接近饱和、战场转向 v2 的 EPDMS 与 navhard；但它**仍非闭环**，必须与 Bench2Drive、实路测试互补。

一句话总结：**NAVSIM 让"规划好不好"第一次变得可量化、可复现、可比拼——它是端到端驾驶走向科学的基石。**

---

*💡 这是「知识点拆解」系列的第 6 篇。结合前面讲过的端到端演进、DiffusionDrive、Flow Matching，你就能完整理解一篇端到端论文里"PDMS 88→92"到底意味着什么。下期我们继续拆解闭环评测与强化学习在规划里的落地。*
