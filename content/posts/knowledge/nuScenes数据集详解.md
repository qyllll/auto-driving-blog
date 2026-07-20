---
title: "nuScenes数据集详解"
date: 2026-07-19
draft: false
author: "AutoDriving Blog"
tags: ["数据集", "nuScenes", "评测", "自动驾驶"]
categories: ["知识点拆解"]
---

## 一、概述

nuScenes 是由 Aptiv（现 Motional）与 nuTonomy 团队于 2019 年发布的大规模自动驾驶多模态数据集，也是最早提供完整传感器套件标注的全场景数据集之一。它的发布标志着自动驾驶感知研究从单一传感器（仅 LiDAR 或仅 Camera）向多传感器融合的重要转折，极大地推动了 3D 检测、跟踪、预测和规划等多个方向的发展。

nuScenes 包含 1000 个驾驶场景（scene），每个场景时长约 20 秒，步长 20Hz 录制，关键帧以 2Hz 采样标注。数据采集覆盖波士顿和新加坡两个城市——包括城市主干道、居民区、十字路口、环形交叉口、隧道和高速公路等多种路况，以及白天、夜晚、黄昏、雨天等不同光照和天气条件。这种多样性使得基于 nuScenes 训练的模型对光照和天气变化具有更好的鲁棒性。

该数据集的核心价值在于其多模态特性：每辆数据采集车在同一时间戳下同时提供 6 个环视摄像头图像（覆盖 360° 视野）、5 个毫米波雷达点云和 1 个 32 线 Velodyne 激光雷达点云。研究者可以基于这些数据开发真正利用多模态传感器融合的感知算法，也可以分别验证不同传感器配置下算法的性能上限。

## 二、传感器配置

每一帧数据中，nuScenes 的传感器配置规格与部署位置如下：

### 2.1 传感器规格

- **摄像头（Camera）**：6 个环视摄像头，型号为 Basler ace acA1600-60gc（160 万像素），分别安装于车顶前部（CAM_FRONT）、左前（CAM_FRONT_LEFT）、右前（CAM_FRONT_RIGHT）、左侧（CAM_BACK_LEFT）、右侧（CAM_BACK_RIGHT）和后部（CAM_BACK）。每个摄像头输出 1600×900 像素 RGB 图像，帧率 12Hz，通过硬件同步实现同一时刻的全局快门采集。所有摄像头的水平视野之和覆盖 360°。

- **激光雷达（LiDAR）**：1 个 Velodyne HDL-32E 32 线旋转式激光雷达，安装于车顶中心位置，垂直视野范围 -30° 至 +10°（共 40°），水平视野 360°，帧率 20Hz。每帧返回约 80 万个点，波长 905nm，最大探测距离约 100m。点云密度随距离增加而减小，近距离（< 20m）密度最高。

- **毫米波雷达（Radar）**：5 个 Continental ARS 408 长距毫米波雷达，分别安装在车前保险杠中央、车尾中央以及四个角落位置。工作频率 76-77GHz，帧率 13Hz，最大探测距离约 250m，水平视野 ±60°，速度测量范围 -100 至 +200 km/h。每个雷达返回约 200 个检测目标，包含相对位置、径向速度、雷达截面（RCS）等信息。

- **GPS/IMU**：采用 Applanix POSLV 220 导航系统，提供厘米级定位和姿态信息。输出包括全局坐标（UTM 坐标系）、航向角、横滚角、俯仰角、速度矢量、加速度和角速度，更新频率 100Hz。

### 2.2 传感器同步

所有传感器通过硬件触发线实现时间同步。激光雷达作为主时钟源（20Hz），摄像头和雷达通过外部触发信号与 LiDAR 对齐。数据后处理阶段使用插值方法将所有传感器的时间戳对齐到最近的关键帧。研究使用时应特别注意：不同传感器的实际采集时刻存在微小的亚毫秒级偏移，高速运动场景中这个偏移可能导致明显的配准误差。

## 三、数据格式与结构

nuScenes 采用关系型数据库结构设计，使用 JSON 格式来组织存储元数据。所有标注文件以表（table）的形式组织，每个表包含多个条目（entry），表之间通过 token 建立关联关系。

### 3.1 sample

sample 表是数据集的核心连接点，代表一个"关键帧"。每秒标注 2 个关键帧，因此每个约 20 秒的场景包含约 40 个关键帧。每个 sample 记录的是一个特定时刻的快照，包含时间戳（timestamp）和指向各类传感器数据的索引。

sample 的字段主要包括：
- **token**：唯一标识符（32 位十六进制字符串）
- **timestamp**：Unix 时间戳（微秒级精度）
- **scene_token**：所属场景的唯一标识，用于与 scene 表关联
- **prev** 和 **next**：前后关键帧的 token，构成双向链表，方便按时间顺序遍历
- **data**：字典结构，包含每个传感器通道对应的 sample_data token

### 3.2 sample_data

sample_data 表记录了每个传感器在 sample 对应时刻采集的具体数据信息。每个条目包含：

- **filename**：数据文件路径（图像为 .jpg，点云为 .pcd 文件）
- **fileformat**：文件格式类型
- **sensor_modality**：传感器模态（camera、lidar、radar）
- **channel**：传感器通道名（如 CAM_FRONT、LIDAR_TOP）
- **timestamp**：实际采集时间（微秒），可能与 sample 的 timestamp 存在微小偏移
- **ego_pose_token**：该时刻 EGO 的全局位姿
- **calibrated_sensor_token**：传感器的标定参数（内参和外参）
- **sample_token**：所属 sample
- **prev** 和 **next**：同一传感器通道的上一帧和下一帧数据

### 3.3 sample_annotation

sample_annotation 是数据集中最核心的标注表，记录了每个关键帧中所有被标注物体的 3D 信息。每个标注条目包含：

- **token**：唯一标识符
- **sample_token**：所属 sample
- **instance_token**：实例 ID，用于跨帧追踪同一物体
- **category_token**：所属类别
- **visibility_token**：可见性（0-100%，分为 4 个等级：0-40%、40-60%、60-80%、80-100%）
- **translation**：物体在全局坐标系下的中心位置 [x, y, z]（单位：米）
- **rotation**：物体朝向的四元数表示 [w, x, y, z]
- **size**：3D 边界框的尺寸 [width, length, height]（单位：米）
- **num_lidar_pts**：落在该物体框内的激光雷达点数
- **num_radar_pts**：落在该物体框内的毫米波雷达目标数
- **prev** 和 **next**：同一实例在前后帧的标注

### 3.4 scene

scene 表定义了连续的驾驶片段，每个 scene 包含一系列按时间排序的 sample。scene 的字段包括：

- **token**：唯一标识
- **name**：场景名称，格式如 "scene-0001"
- **description**：场景描述（"Night rain, intersection, heavy traffic" 等）
- **log_token**：所属 log 记录，包含采集时间、城市等元信息
- **nbr_samples**：场景包含的关键帧数
- **first_sample_token**：该场景的第一个 sample 的 token
- **last_sample_token**：最后一个 sample 的 token
- **duration**：场景时长（秒）

### 3.5 category

category 表定义了分类体系，采用两级层次结构。一级类别包括 vehicle、human、animal、movable_object、static_object 等，二级类别做进一步细分化：

- **vehicle**：car（乘用车）、truck（卡车）、bus（巴士）、trailer（拖车）、motorcycle（摩托车）、bicycle（自行车）
- **human**：pedestrian（行人）、stroller（婴儿推车）、wheelchair（轮椅）
- **movable_object**：traffic_cone（交通锥）、barrier（路障）
- **static_object**：不会移动的道路设施

数据集中共有 23 个标注类别，但大多数评估任务仅关注其中 10 个主要类别（car、truck、bus、trailer、motorcycle、bicycle、pedestrian、traffic_cone、barrier）。

### 3.6 辅助表

除以上核心表外，nuScenes 还包含多个辅助表：

- **ego_pose**：EGO 车辆在每个 sample 时刻的全局位姿（translation + rotation）和速度
- **calibrated_sensor**：每个传感器的内参（相机内参矩阵、LiDAR 外参等）和外参（相对于车辆中心）
- **log**：记录日志元信息，包括城市、车辆 ID、日期等
- **instance**：所有被标注物体的实例列表，记录每个物体首次和最后一次出现的关键帧
- **visibility**：可见性等级定义表

## 四、数据划分

nuScenes 官方数据划分严格遵循场景级（scene-level）划分，确保同一场景的数据不会同时出现在训练集和验证集中：

| 划分 | 场景数 | 关键帧数 | 用途 |
|------|--------|----------|------|
| 训练集 train | 700 | 约 28,000 | 模型训练 |
| 验证集 val | 150 | 约 6,000 | 超参调优与模型选择 |
| 测试集 test | 150 | 约 6,000 | 最终评估（标注不公开） |

此外官方提供 mini 版本：mini-train（8 场景，约 320 关键帧）和 mini-val（2 场景，约 80 关键帧）。建议在 mini 版本上调试代码逻辑，在完整训练集上训练最终模型。测试集评估须通过官网提交结果（检测结果 JSON 文件 + 方法说明 + 使用的传感器模态），官方服务器使用统一的评估脚本计算指标。

## 五、感知任务

### 5.1 3D 目标检测

3D 目标检测是 nuScenes 最核心的评测任务。评估使用两个综合指标——NDS 和 mAP。

**NDS（nuScenes Detection Score）**：将检测精度和质量量化为 0-1 的综合分数：

```
NDS = (mAP + mATE + mASE + mAOE + mAVE + mAAE) / 6
```

其中 mAP 为平均精度，其余五项为 True Positive 指标：

- **mAP**：基于中心距离匹配的平均精度，匹配阈值取 0.5m、1m、2m、4m 四个距离阈值并取平均。与 COCO 的 IoU 匹配不同，nuScenes 使用中心距离匹配更适用于自动驾驶场景——远处的物体即使 3D IoU 很低，其中心位置可能仍是准确的。

- **ATE（Average Translation Error）**：匹配的预测框与真值框中心点之间的欧氏距离（单位：米）。ATE 越小越好，反映了模型的定位能力。

- **ASE（Average Scale Error）**：1 - 预测框与真值框之间的 3D IoU，反映了尺寸估计的准确度。

- **AOE（Average Orientation Error）**：预测朝向与真值朝向之间的最小角度差（单位：弧度）。AOE 的范围为 [0, π]，值越小越好。注意朝向估计的误差对下游规划任务影响很大——即使检测框位置准确，朝向错误可能导致碰撞风险评估完全错误。

- **AVE（Average Velocity Error）**：预测速度与真值速度之间的 L2 误差（单位：m/s）。速度估计的准确性对轨迹预测任务至关重要。

- **AAE（Average Attribute Error）**：属性分类的 1 - 准确率。属性包括车辆是否停靠、行人是否行走等。

**官方排行榜**中当前最先进方法的 NDS 已经超过 0.80，mAP 超过 0.70。后融合方法（如 BEVFusion、TransFusion）通常优于前融合方法。

### 5.2 多目标跟踪

跟踪任务要求模型持续检测并关联帧间目标。主要指标为 **AMOTA**：

```
AMOTA = (1 / L) × Σ_{l∈L} MOTA(l)
```

其中 MOTA(l) 在不同召回率阈值下分别计算。AMOTA 综合了检测质量和 ID 一致性，比传统的 MOTA 更适合自动驾驶场景中的追踪评估。

辅助指标包括：
- **AMOTP（Average Multi-Object Tracking Precision）**：匹配的预测轨迹与真值轨迹之间的平均位置误差
- **sAMOTA（scaled AMOTA）**：将 AMOTA 缩放到 0-1 区间，缓解不同方法间性能差距过大的问题

### 5.3 语义分割

nuScenes 提供约 40,000 张精细标注的环视图像用于像素级语义分割。标注包含 32 个语义类别（道路、人行道、建筑、车辆、行人等），采用 panoptic 格式同时提供语义标签和实例 ID。

评估采用 mIoU（mean Intersection over Union）：
```
mIoU = (1/C) × Σ_{c∈C} (TP_c / (TP_c + FP_c + FN_c))
```

其中 C 为类别数，TP、FP、FN 分别为真阳性、假阳性和假阴性像素数。该任务通常使用全卷积网络或视觉 Transformer 架构。

### 5.4 轨迹预测

轨迹预测任务要求模型根据目标的历史轨迹（过去 2 秒）预测其未来位置（未来 6 秒）。每个目标需输出多个候选轨迹（通常 K=5 或 K=10）及其置信度。

评估指标包括：
- **minADE（Minimum Average Displacement Error）**：K 个候选轨迹中与真值误差最小的轨迹的平均位移误差
- **minFDE（Minimum Final Displacement Error）**：K 个候选轨迹中最优轨迹的终点位移误差
- **MR（Miss Rate）**：所有候选轨迹与真值的 final displacement 均超过 2m 的比例
- **brier_minFDE**：结合置信度校准的 minFDE，增加了对置信度估计的惩罚

## 六、规划 Benchmark

nuScenes Planning Challenge 是 nuScenes 针对端到端规划任务推出的评测基准。该挑战赛于 2021 年首次设立，每年定期更新评估标准和排行榜，是端到端规划方法最常用的开环评估平台之一。

### 6.1 评估设置

规划任务将 EGO 车辆的历史传感器数据（2 秒历史）映射到未来 8 秒的轨迹。模型需要输出一系列路径点或控制指令。评估采用开环（open-loop）方式——模型的预测不影响后续输入。

### 6.2 开环 L2 指标

评测指标包括：
- **L2 位移误差**：在 1.0s、2.0s、3.0s 时域上预测位置与真值的欧氏距离
- **ADE**：整个预测时域上的平均位移误差
- **FDE**：最终时刻（8s）的终点位移误差
- **碰撞率**：预测轨迹中 EGO 与其他物体的碰撞比例

### 6.3 开环评估的根本缺陷

NAVSIM 论文的实证研究表明：

1. **相关性极低**：开环 L2（3s）与闭环 PDMS 的 Pearson 相关系数 r ≈ 0.3。某些开环 L2 为 0.5m 的模型在闭环中频繁碰撞，而开环 L2 为 1.2m 的模型在闭环中表现良好。

2. **分布偏移效应**：开环中的模型始终看到的是接近真值轨迹的观测（因为输入来自 log），因此模型学会了在"理想路线附近"进行预测。但在闭环中，模型自身的误差会将观测分布拉离训练分布，导致性能崩溃。

3. **补偿学习**：开环评测鼓励模型学习"平均轨迹"——在一些情况下，预测"保持直行"比预测"即将转弯"能得到更低的 L2 误差（因为任何转弯预测的微小偏差都会被惩罚）。但这种保守策略在真实驾驶中完全不可用。

4. **碰撞率不可靠**：在开环中，碰撞检测基于预测轨迹而非实际执行轨迹。模型可以通过预测"急刹车"（突然减速的轨迹）来人为降低碰撞率，但这样的轨迹在实际执行中可能导致追尾。

### 6.4 nuScenes2NavSim 转换

为弥补 nuScenes 缺乏闭环仿真的不足，NAVSIM 团队开发了 nuScenes2NavSim 转换工具。转换过程：

1. 从 nuScenes log 中提取场景地图数据
2. 将标注的 3D 边界框转化为 NAVSIM 的 agent 格式
3. 配置非反应式交通参与者（non-reactive agents）
4. 提供 EGO 初始位置和目标导航路线

转换后的场景可以直接在 NAVSIM 平台中运行闭环仿真评估，使研究者获得 nuScenes 场景下的 PDMS、碰撞率和驾驶分数等闭环指标。

### 6.5 nuScenes-GR-20K

nuScenes-GR-20K 是从原始 nuScenes 中筛选出的困难场景子集，约 20,000 个规划片段，专门用于评估端到端驾驶模型的泛化鲁棒性（Generalization Robustness）。与原始数据相比，GR-20K 聚焦于：

- 高交互密度的场景（多车交织、行人横穿）
- 低频率但高风险的边缘场景（无保护转弯、遮挡恢复）
- 天气和光照条件剧烈变化的场景

该数据集常用于 GRPO（Group Relative Policy Optimization）等强化学习方法的训练与评估，能够区分在常规场景上性能饱和的模型。

## 七、常见误区与改进方向

常见误区包括：仅使用 LiDAR 忽视多模态融合（BEVFusion 等能显著提升远距离检测精度）；过度关注 NDS 总值而忽略分解指标差异（ATE 和 AOE 对规划影响远大于 AVE）；在验证集上反复调参导致性能虚高。改进方向：开发多模态融合方法，利用时序标注做时序融合，结合 nuScenes2NavSim 扩展至闭环规划评估，引入场景级难度分层。

## 八、总结

nuScenes 以其 1000 个场景、多模态传感器配置（6 相机 + 5 雷达 + 1 LiDAR）和标准化评测体系（NDS、mAP、AMOTA、mIoU），是自动驾驶感知领域使用最广泛的基准数据集之一。研究者需清醒认识其局限性——开环与闭环的鸿沟、两个城市的数据偏差以及传感器配置与量产车的差异——将 nuScenes 与 NAVSIM、nuPlan 等闭环仿真平台结合使用，是实现从感知到规划完整验证的关键路径。
