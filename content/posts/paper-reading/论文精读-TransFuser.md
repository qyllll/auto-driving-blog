---
title: '论文精读｜TransFuser：用 Transformer 做传感器融合——让相机和激光雷达"商量着开车"'
date: 2026-07-30
draft: false
categories: ["论文精读"]
tags: ["🔗 端到端", "🤖 传感器融合", "🧠 Transformer", "🚗 自动驾驶", "👁️ 视觉", "📡 激光雷达"]
summary: "TransFuser 在 CARLA 上首次验证了 Transformer 做多模态传感器融合的可行性——用自注意力让图像和 LiDAR 特征在多个分辨率上密集交互，取代简单的几何融合。在 CARLA Leaderboard 和 Longest6 基准上以显著优势领先此前所有方法，将碰撞率降低 48%。"
weight: 38
---

## 论文信息

- **标题**：*TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving*
- **作者机构**：**图宾根大学（University of Tübingen）** × **MPI for Intelligent Systems**（Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger）
- **arXiv**：[2205.15997](https://arxiv.org/abs/2205.15997)（NeurIPS 2022，CVPR 2021 的期刊扩展版）
- **代码**：[github.com/autonomousvision/transfuser](https://github.com/autonomousvision/transfuser)
- **一句话总结**：用 Transformer 把**相机图像**和**LiDAR 鸟瞰图**的特征在多个分辨率上做密集融合，让两个模态"商量"着理解场景，再通过 GRU 自回归预测轨迹。在 CARLA Town05 比之前最好方法 DS 高 48%，碰撞降低 48%。

---

## 要解决什么问题：传感器融合不是"拼图"，是"对话"

端到端自动驾驶的一个核心工程问题：车上装了**多个传感器**（相机、LiDAR、雷达），怎么把它们的特征真正融合起来？

之前的做法主要有两种：

| 融合方式 | 做法 | 问题 |
|---------|------|------|
| **Late Fusion（后融合）** | 每个模态独立处理，最后把结果拼一起 | 错过模态间的细粒度交互信息 |
| **Geometric Fusion（几何融合）** | 把图像特征投影到 BEV 空间（基于已知的相机-LiDAR 外参）再拼接 | 依赖精确标定，在密集交通下无法捕捉复杂的交互场景 |

这两种方式的共同问题：**融合是静态的、一次性的**——图像和 LiDAR 的特征没有真正"交流"过。比如 LiDAR 看到了远处一辆车的点云但不知道是啥类型，相机能识别出"那是一辆卡车"但不知道它的准确距离——这两种信息应该在多个层次上反复交互，而不是到了最后一层才拼接。

![TransFuser 核心动机：在密集交通场景下，几何融合和后融合远不如注意力融合](/images/transfuser/fig1_motivation.jpg)

**图 1：TransFuser 的核心动机。** 在高密度动态交通场景下（多车、多行人、复杂路口），基于几何投影或简单拼接的融合方式会让模型漏掉关键的交互信息。TransFuser 用 Transformer 的自注意力机制让图像和 BEV 特征在**多个分辨率层次**上反复交互，最终学到融合了全局 3D 场景上下文的紧凑特征。

---

## 核心思想：在多个分辨率上用自注意力做密集融合

TransFuser 的核心主张很简单但有效：

> **在 CNN 特征提取器的每个分辨率层级上，用 Transformer 的自注意力让图像特征和 LiDAR BEV 特征互相"看"对方一眼，把融合嵌入到特征提取的过程中，而不是放在最后。**

![TransFuser 整体架构](/images/transfuser/fig2_architecture.png)

**图 2：TransFuser 整体架构。** RGB 图像和 LiDAR BEV 分别经过独立的 CNN 特征提取器。在 4 个分辨率层级上，每个分支的特征被展平为 token 序列，经过 Transformer 做自注意力融合（让图像 token 和 LiDAR token 互相注意力），融合后的特征再 reshape 回原分辨率回到各自分支继续前向传播。最终两个分支的 512 维特征向量相加，经 MLP 降维后作为 GRU 的初始隐状态，自回归预测 4 个差分 waypoint。

整条数据流拆成四大步：

1. **输入表示**：RGB 图像（400×300） + LiDAR BEV（256×256，2 通道：高度 + 密度）
2. **多分辨率 Transformer 融合**：在 4 个分辨率层级上分别做自注意力，让图像和 LiDAR 特征密集交互
3. **全局特征编码**：融合后的特征 → AvgPool → FC → 512 维全局场景向量
4. **自回归 waypoint 预测**：512 维向量 → MLP(256, 128) → 64 维 → GRU 初始隐状态 → 逐 step 预测 4 个差分 waypoint → PID 控制器

下面逐块详细拆解。

---

## 方法详解

### 一、输入表示（Input Representation）

TransFuser 使用两种互补的传感器模态：

**1. RGB 图像（相机）**
- 单目前视相机，分辨率 400×300
- 提供丰富的语义信息（物体类别、颜色、纹理、交通灯状态、路标）
- **缺点**：没有深度信息，单目测距不准

**2. LiDAR BEV（激光雷达鸟瞰图）**
- 将 LiDAR 点云投射到 256×256 的 BEV 栅格（每个像素对应 0.125m，覆盖 32m × 32m 区域）
- 每个栅格的**2 个通道**：
  - **高度通道**：该栅格内点云的最大高度（归一化到 [0,1]）
  - **密度通道**：该栅格内点云的数量（对数归一化到 [0,1]）
- 提供精确的 3D 几何信息（物体位置、距离、形状）
- **缺点**：没有语义信息（不知道"这是一个行人"）

**互补关系**：相机知道"那是什么"，LiDAR 知道"它在哪"。关键是让两者在特征层面交互。

### 二、多分辨率 Transformer 融合（核心创新）

这是 TransFuser 最核心的模块。做法如下：

#### 2.1 为什么在多个分辨率上融合？

CNN 的不同层编码了不同层次的信息：
- **浅层**（高分辨率）：边缘、纹理、局部几何
- **深层**（低分辨率）：语义类别、全局场景结构

TransFuser 选择在 **4 个分辨率层级**上都做融合（而不是只在最后一层），让信息在不同抽象层次间充分交换。

#### 2.2 单层融合的具体操作（以其中一个分辨率为例）

```text
图像分支特征图：尺寸 H₁ × W₁ × C₁
BEV 分支特征图：  尺寸 H₂ × W₂ × C₂

Step 1：展平为 token 序列
  图像 token: (H₁×W₁) 个，每个维度 C₁
  BEV token:  (H₂×W₂) 个，每个维度 C₂

Step 2：拼接所有 token
  总 token 数 = H₁×W₁ + H₂×W₂，统一维度 C（用 1×1 卷积对齐）

Step 3：加上可学习位置编码（让网络知道每个 token 的原始空间位置）

Step 4：送入 Transformer
  Q = Fin @ Mq, K = Fin @ Mk, V = Fin @ Mv
  A = softmax(QK^T / √D) @ V     ← 自注意力：每个 token 关注所有其他 token
  Fout = MLP(A) + Fin              ← 残差连接

Step 5：reshape 回原始分辨率
  融合后的特征拆分为图像部分和 BEV 部分，分别 reshape 回 H₁×W₁×C₁ 和 H₂×W₂×C₂

Step 6：元素相加回到原分支
  融合后的特征图 + 原分支特征图（残差连接）
```

**关键理解**：自注意力让**图像 token 可以看到 LiDAR token 的信息**，反之亦然。一个图像中的"红色圆形" token 可以通过注意力机制从 LiDAR 的某个 token 那里了解到"那个圆形在 15 米外"。

#### 2.3 多尺度处理

由于高分辨率特征图 token 数量太大（计算量 O(N²)），Transformer 无法直接处理。TransFuser 的做法：

- 将高分辨率的特征图用 **AvgPool** 下采样到与低分辨率相同的尺寸
- Transformer 输出后用**双线性插值**上采样回原始分辨率
- 再与原始分支特征图元素相加

```text
分辨率层级：
  Level 1: Image (176×40→22×5)  + BEV (64×64→8×8)
  Level 2: Image (88×20→22×5)   + BEV (32×32→8×8)
  Level 3: Image (44×10→22×5)   + BEV (16×16→8×8)
  Level 4: Image (22×5)          + BEV (8×8)
  （→ 后的尺寸是 AvgPool 下采样后的 token 数量）
```

每个 TransFuser 模块有 **L=4 层**，每层 4 个注意力头。

#### 2.4 全局特征

经过 4 层融合后，两个分支各自输出的特征图经过 **AvgPool → FC** 压缩为 512 维向量，然后**元素相加**得到最终的全局场景特征。

> **关键认知**：这 512 维向量不是简单的"图像特征 + BEV 特征"，而是经过了多层次自注意力交互后的**联合嵌入**——它既包含了图像的语义信息，也包含了 LiDAR 的几何信息，更重要的是，它编码了**两个模态之间的对应关系**（比如"这个像素对应的那个点云"）。

### 三、Waypoint 预测网络

全局特征向量不直接预测方向盘/油门，而是预测**轨迹 waypoint**，再由 PID 控制器转为控制信号。

#### 3.1 自回归 GRU 解码器

```text
输入：512 维全局特征
→ MLP(512→256→128→64)
→ 64 维向量作为 GRU 初始隐状态 h₀

for t = 1 to 4:
    输入：当前点 w_{t-1}（第一个点为 (0,0)）+ 目标位置 G
    输出：差分 waypoint δw_t
    最终 waypoint：w_t = w_{t-1} + δw_t
```

每个 step 预测**差分** waypoint（相对于上一步的位移），而不是绝对坐标。这样更稳定——模型只需预测"下一步往哪走"，而不是直接猜"终点在哪"。

**目标位置 G**（GPS 坐标，已转换到自车坐标系）作为额外输入，确保轨迹朝着导航方向前进。

#### 3.2 PID 控制器

预测的 4 个 waypoint → 两个 PID 控制器：
- **纵向 PID**：根据 waypoint 间的距离计算目标速度 → 油门 / 刹车
- **横向 PID**：根据 waypoint 间的方向计算目标转角 → 方向盘

此外还有一个**蠕行机制（Creeping）**：如果车停了超过 55 秒（红灯排队等情况），就设定目标速度 4 m/s 蠕动 1.5 秒，防止模型在"停车"状态陷入死循环。蠕行时用 LiDAR 检测前方是否有障碍物，有就取消蠕行（安全启发式）。

### 四、辅助任务（Auxiliary Tasks）

只靠 4 个 waypoint 的 L1 损失监督是不够的——场景太复杂，梯度信号太稀疏。TransFuser 加了 4 个辅助任务，让中间特征学到更丰富的场景表示：

![辅助任务：图像分支预测深度和语义分割，BEV 分支预测 HD 地图和检测框](/images/transfuser/fig3_auxiliary.jpg)

**图 3：辅助损失。** 除了 waypoint L1 损失，还有 4 个辅助任务。

| 辅助任务 | 分支 | 解码器 | 损失 | 作用 |
|---------|------|--------|------|------|
| **深度估计** | 图像 | Conv Decoder | L1 | 让图像特征学到距离信息，弥补 LiDAR 缺失时的深度线索 |
| **语义分割** | 图像 | Conv Decoder | 交叉熵（7 类） | 让图像特征识别道路、车辆、行人、红绿灯等语义 |
| **HD 地图预测** | BEV | Conv Decoder | 交叉熵（3 类） | 让 BEV 特征编码可通行区域 |
| **车辆检测** | BEV | CenterNet Decoder | Focal + CE + L1 | 让 BEV 特征感知周围车辆位置和朝向 |

**为什么辅助任务重要**：它们提供了强力的**中间监督**，确保中间特征图不仅仅服务于 waypoint 预测，还编码了场景的各种结构化信息。这相当于把"暗盒"的特征提取变成了"带语义标签"的特征提取。

### 五、损失函数

总的训练损失：

\[
\mathcal{L} = \mathcal{L}_{\text{waypoint}} + \lambda_1 \mathcal{L}_{\text{depth}} + \lambda_2 \mathcal{L}_{\text{semantic}} + \lambda_3 \mathcal{L}_{\text{HDmap}} + \lambda_4 \mathcal{L}_{\text{detection}}
\]

其中 waypoint 损失是 L1：

\[
\mathcal{L}_{\text{waypoint}} = \sum_{t=1}^{T} ||w_t - w_t^{gt}||_1
\]

所有损失权重 λ 都设为 1（简单均匀加权）。

---

## 训练：Expert 数据生成

### 数据如何获取

TransFuser 的训练数据通过一个**规则专家**在 CARLA 中采集：
- **228k 帧**数据，来自 8 个 CARLA 城镇
- 包含约 2500 条城区路口路线（平均 100m）+ 1000 条高速公路路线（平均 400m）
- 每 0.5 秒存一帧（2 FPS）

### 专家策略（Expert Policy）

专家使用**特权信息**（仿真器提供的 ground truth）来生成完美轨迹：

1. **A\* 全局规划器**：生成从起点到终点的粗粒度路径
2. **横向 PID**：沿着 A\* 路径行驶，最小化与路径的角度偏差
3. **纵向 MPC**：分 3 档目标速度
   - 正常：4.0 m/s
   - 路口内：3.0 m/s
   - 预测到碰撞/闯红灯：0.0 m/s（停车）
4. **碰撞预测**：用预训练的**自行车模型（Bicycle Model）** 预测未来 4 秒所有车辆的运动轨迹（路口场景）或 1 秒（其他场景），检测是否会碰撞

![专家左转场景：等待对向车流通过后完成左转](/images/transfuser/fig4_expert_a.jpg)

**图 4a：专家在路口等待。** 自行车模型预测如果此时左转会撞到对向车辆，所以专家选择停车等待。

![对向车流通过后，专家完成左转](/images/transfuser/fig4_expert_b.jpg)

**图 4b：对向车流通过后，专家完成左转。** 碰撞预测显示安全后，专家加速通过路口。

---

## 实验与结果

### Longest6 基准

为了能在本地高效评估，作者提出了 **Longest6 基准**：
- 从 CARLA 官方的 76 条路线中选出每个城镇最长的 6 条路线（共 36 条）
- 平均路线长度 **1.5 km**（接近官方 Leaderboard 的 1.7 km）
- 最高密度交通 + 6 种天气 × 6 种光照组合
- 包含 NHTSA 预碰撞场景（急刹、变道冲突等）

### 主要结果

| 方法 | DS ↑ | RC ↑ | IS ↑ | Veh ↓ | Collisions/km 对比 |
|------|:----:|:----:|:----:|:-----:|:-----------------:|
| Late Fusion (LF) | 22.47 | 83.30 | 0.27 | 4.63 | - |
| Geometric Fusion (GF) | 27.32 | 91.13 | 0.30 | 4.64 | - |
| LAV | 32.74 | 70.36 | 0.51 | 0.83 | - |
| **Latent TransFuser** (image only) | 37.31 | 95.18 | 0.38 | 3.66 | - |
| **TransFuser** (image + LiDAR) | **47.30** | 93.38 | **0.50** | **2.45** | **比 GF 降 48%** |
| Expert (upper bound) | 76.91 | 88.67 | 0.86 | 0.28 | - |

**核心发现**：
- TransFuser 的 DS 比最好的几何融合（GF）高 **20 分（+73%）**
- 每公里碰撞次数从 4.64（GF）降到 2.45（TransFuser），降低 **48%**
- 即使只用相机的 Latent TransFuser 也比 LAV 等完整方法高

### CARLA Leaderboard

在官方的 **100 条秘密路线**上，TransFuser 提交时在所有已发布方法中排名第一。

### 推理速度

| 方法 | 单模型 | 集成 3 个模型 |
|------|:-----:|:------------:|
| Late Fusion | 23.5 ms | 46.7 ms |
| Geometric Fusion | 43.5 ms | 69.1 ms |
| **TransFuser** | **27.6 ms** | **59.6 ms** |

TransFuser 在单 RTX 3090 上 **>36 FPS**，完全可以实时运行。

### 注意力可视化

![注意力图可视化](/images/transfuser/fig6_attention1.png)

**图 5：注意力可视化。** 对于红色标记的 query token（上图是图像 token，下图是 LiDAR token），绿色标记了 top-5 高注意力权重的 token（来自另一个模态）。可以看到 TransFuser 的注意力高度集中在车辆、行人、交通灯等关键目标周围——说明模型学会了跨模态关注"什么重要"。

---

## 关键设计分析

### 1. 为什么在多个分辨率上做融合？

| 融合策略 | 做法 | 表现（DS） |
|---------|------|:---------:|
| 仅在最高分辨率融合 | 只在前 2 层做 | 35.2 |
| 仅在最低分辨率融合 | 只在后 2 层做 | 39.8 |
| **全部 4 层融合** | 所有分辨率都做 | **47.3** |

浅层特征包含位置/边缘信息，深层特征包含语义信息——两者都需要跨模态交互。

### 2. 为什么用自注意力（self-attention）而不是交叉注意力？

自注意力让**所有 token 互相看**，包括同模态内部和跨模态。这样：
- 图像内部的 token 可以互相交互（全局上下文）
- LiDAR 内部的 token 可以互相交互
- 图像 token 可以关注 LiDAR token（跨模态融合）
- LiDAR token 可以关注图像 token

统一在一个注意力矩阵里完成，不需要设计复杂的跨模态路由。

### 3. 为什么用 waypoint 而不是直接控制信号？

waypoint 比控制信号更**平滑、更可解释**。方式：
- 预测 4 个差分 waypoint（L1 损失）
- PID 将 waypoint 转为控制信号
- PID 控制器本身是固定的、物理可解释的（比例 P → 误差大小，积分 I → 历史累积误差，微分 D → 误差变化趋势）

---

## 局限与反思

### TransFuser 的局限

**第一**，碰撞仍然偏多（每公里 2.45 次 vs Expert 的 0.28 次）。主要失败场景是**密集交通下的变道**和**无保护左转**——模型在需要精确时机判断的场景中表现不佳。

![变道失败场景](/images/transfuser/fig5_lanechange.jpg)

**图 6：变道失败。** 在密集交通中变道时，TransFuser 容易连续碰撞。这是所有模仿学习方法在高密度交互场景下的共同难题。

**第二**，依赖 LiDAR 输入。Latent TransFuser 虽然比之前的方法好，但 DS 37.31 仍然远低于 47.30——说明 LiDAR 的精确几何信息对端到端驾驶来说仍然是不可或缺的。

**第三**，辅助任务的损失权重都是简单设置为 1，没有学习自适应权重。后续工作可以用不确定性加权或 GradNorm 来优化。

### 和本系列其他论文的关系

TransFuser 在"端到端驾驶规划"这条线上是一个**早期的奠基作品**：

- **对比 UniAD（规划中心派）**：UniAD 把"以规划为中心"的方法系统化，而 TransFuser 更关注"怎么把传感器特征融合好"。UniAD 的 TrackFormer/MapFormer 其实继承了这个思路——在不同层让不同模态交互。但 UniAD 是纯视觉的（只用相机），而 TransFuser 验证了相机 + LiDAR 的优势。

- **对比 VAD / VADv2（向量化派）**：VADv2 用向量化场景 token（几百个稀疏 token）替代了 TransFuser 的稠密 BEV 特征图（64×64），效率更高。但 VADv2 的核心创新是概率规划，而 TransFuser 的传感器融合思路在 VAD 系列中被继承——BEVFormer 本质上也是在 BEV 空间用注意力做多视图图像融合。

- **对比 NEAT（注意力 BEV 变换派）**：NEAT 也是用注意力从图像生成 BEV 表示，但 NEAT 只做图像→BEV 的单向投影，而 TransFuser 做的是**双向密集融合**——图像和 BEV 互相投影。

- **对比后续工作**：TransFuser 后来被扩展到 AIM、TCP、ThinkTwice、DriveAdapter 等方法中，它的多分辨率融合设计成为了端到端驾驶的"标准组件"。可以说，TransFuser 是第一个让 Transformer 在端到端驾驶中"work"的工作。

---

## 个人思考

TransFuser 最值得学习的不是它刷了多少分，而是它对问题的理解方式：

**在 2021 年，所有人都在用几何方法做传感器融合（投影拼接）时，TransFuser 提出"让模态之间对话"——这个思路在今天的多模态学习中已经成为常识，但在当时是开创性的。**

自动驾驶领域有一个长期存在的倾向：**盲目堆更复杂的模型**。但 TransFuser 让我们看到一个更重要的趋势——**不是模型越复杂越好，而是信息交互越充分越好**。用 Transformer 的简单自注意力，加上"多个分辨率上反复融合"这个设计原则，就实现了对当时所有复杂方法的全面超越。

另外值得一提的是，TransFuser 是**端到端学习可复现性的标杆**——它开源了完整的训练代码、评测代码和数据集，让后来的研究者可以在统一的框架下做公平对比。这在当时的 CARLA 研究中是稀缺的。

> **一句话总结**：TransFuser = 双分支 CNN（图像 + LiDAR BEV）+ 4 层多分辨率自注意力融合 + GRU 自回归 waypoint 预测 + 4 个辅助任务（深度/语义/地图/检测）。它最核心的贡献是：**证明了"在特征提取过程中做多分辨率跨模态注意力"可以取代"几何投影后融合"——这个设计原则影响了后续几乎所有端到端驾驶方法。**

---

## 参考资料

- 论文：[arXiv:2205.15997](https://arxiv.org/abs/2205.15997)
- 代码：[github.com/autonomousvision/transfuser](https://github.com/autonomousvision/transfuser)
- 相关论文：NEAT [arXiv:2104.09224](https://arxiv.org/abs/2104.09224), CILRS [arXiv:1910.13690](https://arxiv.org/abs/1910.13690)
