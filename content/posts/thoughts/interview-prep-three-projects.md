---
title: "自动驾驶面试深度复习：RiskField-VLA + TrustDrive-WAM + JEPA-DRIVE 三大项目全拆解"
date: 2026-09-23
draft: false
categories: ["个人思考"]
summary: "三大项目 + 比亚迪 Cosmos-3 MOT 世界模型面试复习长文（3000+ 行）：术语定义与公式；Flow Matching/Flow-GRPO/风险场 VLA/WAM/JEPA/MoT 双塔完整伪代码；「你负责什么 / FM / VLA / 世界动作模型 / JEPA / MOT」多份第一人称口述稿（含算法代码与追问链）；40 道压力面问答。"
tags: ["面试", "自动驾驶", "Flow Matching", "GRPO", "VLA", "WAM", "JEPA", "NAVSIM"]
math: true
weight: 98
---

# 使用说明

本文档分三大部分，建议复习顺序：

1. **第一部分：知识点深度复盘** —— 所有概念、专有名词的定义、公式、代码示例。每个术语都按「定义 / 为什么需要 / 代码怎么体现 / 面试怎么答」四个角度展开。
2. **第二部分：核心算法伪代码** —— 从概念到可运行伪代码，逐段注释。覆盖 Flow Matching、Flow-GRPO、风险场 VLA、WAM 联合流、JEPA 全流程。
3. **第二部分补章：项目口述话术** —— Flow Matching / VLA / 世界动作模型 / JEPA / **比亚迪 Cosmos-3 MOT** / 「你负责什么」总述，每份含算法+代码的 3-5 分钟第一人称讲法。
4. **第三部分：压力面问题与参考回答** —— 40 道题 + 追问链，含标准回答与加分回答。

---

# 第一部分：知识点深度复盘

> 本部分覆盖三大项目涉及的所有概念。每个小节先给「一句话定义」，再展开「为什么需要」「数学/代码细节」「面试答题模板」。

---

## 1. 自动驾驶大背景：为什么需要端到端？

### 1.1 专有名词表（本节）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| E2E | End-to-End | 从传感器输入到轨迹/控制输出，中间不依赖人工设计接口的单一可学习系统 |
| 模块化 | Modular Pipeline | 感知、预测、规划、控制分开训练，用固定格式接口串联 |
| Corner Case | - | 长尾危险场景，出现频率低但一旦出事后果严重 |
| 误差累积 | Error Propagation | 上游模块的小误差被下游放大（感知偏 0.1 米，规划可能偏 1 米） |
| 有损传递 | Lossy Interface | 模块间只传结构化结果（框、轨迹），丢掉原始特征里的语义 |

### 1.2 传统自动驾驶 vs 端到端

**传统方案（模块化）的流水线**：

```text
传感器输入
  -> 感知 Perception: 检测(检测框) / 跟踪(目标ID) / 预测(未来轨迹)
  -> 规划 Planning: 规则引擎 或 优化求解器 生成自车轨迹
  -> 控制 Control: PID / MPC 把轨迹变成方向盘/油门/刹车
```

每个模块独立训练，中间传递的是「人工设计的接口」（bounding box、lane id、规则配置……）。

模块化三大问题（面试高频）：

1. **信息有损**：感知只输出 3D 框，框里丢掉了外观纹理、姿态语义、遮挡关系；下游规划拿不到这些。
2. **误差累积**：感知 3% 的漏检率，到规划层可能变成「对静止障碍物完全无感知」；误差不是线性传递而是被放大。
3. **规则写不完**：无保护左转、加塞博弈、施工区借道……corner case 数量无限，if-else 永远覆盖不全。

**端到端方案**：

```text
传感器输入
  -> 单一神经网络(感知特征 + 语义理解 + 轨迹生成一体)
  -> 输出: 未来 T 步的自车轨迹 [[x0,y0], [x1,y1], ...]
```

端到端优势：

1. **无损传递**：中间用特征向量（feature map / embedding），不是人工接口，下游能「看到」上游的全部信息。
2. **数据驱动**：给什么分布的数据学什么分布，上限取决于数据和算力，不取决于规则工程师数量。
3. **联合优化**：感知和规划的 loss 可以一起反传，感知会学「对规划有用」的特征，而不是「检测框 IoU 高」的特征。

**端到端的代价**（面试必被追问）：

- 黑盒，难调试、难验证安全。
- 需要大量高质量轨迹数据（或奖励信号）。
- 分布偏移（distribution shift）：开环训练、闭环执行时误差会累积，自己造成的误差成为新的输入（DAgger 论文经典问题）。

### 1.3 NAVSIM 评测基准与 PDMS 详解

**NAVSIM** 是当前端到端自动驾驶的主流开环（open-loop）评测基准，不用真车，用高精地图 + 仿真器回放，速度快、可复现。

**PDMS（Pilot-Driving Metric Score）公式化理解**：

```text
PDMS = NC * DAC * Comfort * (0.5 * TTC + 0.5 * EP_comfort_adj) * ... 
（具体权重随 NAVSIM 版本略有差异，核心是"乘积"结构）
```

子指标逐个解释：

| 缩写 | 全称 | 中文 | 评价什么 | 为 0 的情况 |
|------|------|------|----------|-------------|
| NC | No at-fault Collision | 无责碰撞 | 自车是否发生有责碰撞 | 撞了就是 0 |
| DAC | Drivable Area Compliance | 可行驶区域合规 | 是否压实体线、出路面 | 出路就是 0 |
| TTC | Time-to-Collision | 碰撞时间 | 与前车/障碍物的最小碰撞时间 | 足够安全才给分 |
| Comfort | - | 舒适度 | 加速度、jerk（加加速度）是否平稳 | 急刹急转扣分 |
| EP | Ego Progress | 自车进度 | 相对合理路线前进了多少 | 站着不动接近 0 |
| SLC | Speed Limit Compliance | 限速合规 | 是否超速 | 超速扣分/归零 |

**为什么是乘积不是加权和**（面试经典追问）：

```text
加权和: 0.0*0.4 + 1.0*0.6 = 0.6   # 碰撞了还能拿 0.6 分，不合理
乘积:   0.0 * 1.0 * ... = 0.0     # 一票否决，碰撞直接归零
```

乘积结构表达了安全的「不可交易性」：不能用舒适度去补偿碰撞。这也是为什么 21% 的长尾场景分数趋近于 0——只要 NC 或 DAC 挂了，整条轨迹的分数就塌了。

**NC / DAC 等指标在代码层面怎么算**（概念级伪代码）：

```python
def pdms_single_trajectory(traj, scenario):
    # traj: (T, 2) 或 (T, 3) 自车未来轨迹
    # scenario: 含障碍物轨迹、车道多边形、限速的场景包

    nc = 1.0
    for obs in scenario.obstacles:
        if check_collision(traj, obs.future_bbox):  # 有责碰撞检测
            nc = 0.0
            break

    dac = 1.0
    if not inside_drivable_area(traj, scenario.drivable_polygon):
        dac = 0.0

    ttc = compute_ttc(traj, scenario.nearest_lead)   # 越大越安全，clip 后映射 [0,1]
    comfort = comfort_score(traj)                     # 基于 a, jerk 的分段函数
    ep = progress_score(traj, scenario.route)         # 沿路线前进比例，惩罚磨蹭
    slc = speed_limit_score(traj, scenario.speed_limit)

    # 乘积结构：任何一项为 0，总分为 0
    pdms = nc * dac * comfort * ttc * ep * slc
    return pdms
```

**我们三个项目的分数对照**：

| 项目 | 核心技术 | PDMS |
|------|----------|------|
| RiskField-VLA | 风险场 + Flow Matching + Flow-GRPO | 0.8713 |
| TrustDrive-WAM | 世界动作模型 + 可信路由 | 0.8012 |
| JEPA-DRIVE | JEPA 隐空间世界模型（无标注） | 0.7285 |

**NAVSIM v1 vs v2**：v2 增加更多长尾交互、更严的评测协议（更多失败模式一票否决），基线分数普遍比 v1 低。

**面试答题模板（NAVSIM）**：

> NAVSIM 是开环评测基准，核心指标 PDMS 是子指标的乘积——碰撞或压线直接归零，所以优化重点是把长尾场景从 0 分拉回正数，而不是在常规场景刷小数点。我们的方案分别达到 0.87 / 0.80 / 0.73，提升主要来自 corner case 的修复。

### 1.4 更多背景名词

**开环 vs 闭环**：

```text
开环 Open-loop: 播放录制场景，模型只输出轨迹，不真正驱动仿真车。
  优点: 快、可复现、无 compounding error
  缺点: 不能评价"我走了这步之后世界怎么变"

闭环 Closed-loop: 模型输出控制仿真车前进，场景实时演化。
  优点: 真实反映部署问题
  缺点: 慢、方差大、训练时难用（需要可微仿真或 RL）
```

**规控（Planning & Control）**：规划出轨迹（一串时空点），控制把轨迹变成执行器指令。

**高精地图 HD Map**：厘米级车道线、拓扑、限速信息；端到端趋势是少依赖甚至无图（mapless）。

**多模态融合 Multi-modal Fusion**：摄像头 + LiDAR + Radar 特征对齐到同一空间（常见 BEV）再联合推理。

---

## 2. VLA（Vision-Language-Action）模型

### 2.1 专有名词表（本节）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| VLA | Vision-Language-Action | 视觉 + 语言指令作为输入，动作/轨迹作为输出的大模型 |
| LLM | Large Language Model | 以 token 序列建模的大语言模型，可作 VLA 的「大脑」 |
| Action Chunk | - | 一次输出的一段未来动作序列（不是单步），减少误差累积 |
| Token | - | 模型处理的最小离散单元；视觉 patch、文字词、动作码都可以是 token |
| Cross-Attention | 交叉注意力 | Query 来自一模态，Key/Value 来自另一模态，实现信息查询 |
| Self-Attention | 自注意力 | Q/K/V 都来自同一序列，建模序列内部依赖 |
| Embedding | 嵌入 | 把离散符号映射为连续向量 |
| Prompt | - | 给模型的输入指令/上下文（语言 prompt、视觉 prompt） |

### 2.2 什么是 VLA？

**一句话定义**：VLA 把「看见的图像」「听到/读到的语言指令」融合后，直接输出「动作」（轨迹、关节角、底盘指令）的模型。

```text
输入:  image(s) + language instruction
输出:  action / trajectory

例:  图像(前方路口) + "在下一个路口左转"  ->  左转轨迹
```

**代表工作对比**（面试常问「你看过哪些 VLA」）：

| 工作 | 动作表示 | 生成方式 | 特点 |
|------|----------|----------|------|
| OpenVLA | 离散动作 token | LLM 自回归 softmax | 简单，但离散化损失精度 |
| RDT | 连续动作 | 扩散模型 DDPM | 多模态动作分布，采样步数多 |
| RDT-2 | RVQ 码 + 连续 | Stage1 离散 CE，推理连续 Flow | 训练用离散监督，推理不用离散 |
| π₀ | 连续动作 | Flow Matching + 动作专家 | VLM KV cache + 300M 动作专家 cross-attn |
| π₀-FAST | FAST 编码动作 | 自回归 | π₀ 的变体，不是 π₀ 原版 |

**重要纠偏（面试防坑）**：π₀ 原版没有 FAST；FAST 是单独变体。π₀ 原版 = Flow Matching + 独立动作专家通过 cross-attention 读 VLM 的 KV cache。RDT-2 的 RVQ 只在训练 Stage 1 用（VLM 学 CE），推理是连续 Flow Matching，不跑 RVQ 解码。

### 2.3 Attention 机制：从公式到代码

**Self-Attention**：

```python
import torch
import torch.nn.functional as F

def self_attention(x, num_heads=8):
    """
    x: (B, L, D)  序列特征, L=token 数, D=维度
    return: (B, L, D)
    """
    B, L, D = x.shape
    head_dim = D // num_heads

    # 线性投影得到 Q, K, V  (每个头独立投影)
    q = x.view(B, L, num_heads, head_dim).transpose(1, 2)  # (B, H, L, hd)
    k = x.view(B, L, num_heads, head_dim).transpose(1, 2)
    v = x.view(B, L, num_heads, head_dim).transpose(1, 2)

    # 缩放点积注意力
    # scores[i,j] = q_i · k_j / sqrt(hd)  表示位置 i 对位置 j 的关注程度
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, L, L)

    # softmax 把分数变成概率分布（每行和为 1）
    attn = F.softmax(scores, dim=-1)                          # (B, H, L, L)

    # 用注意力权重加权求和 V
    out = attn @ v                                            # (B, H, L, hd)
    out = out.transpose(1, 2).reshape(B, L, D)
    return out
```

**Cross-Attention（VLA 中图像条件、动作专家读 VLM 时都用它）**：

```python
def cross_attention(query, context):
    """
    query:  (B, Lq, D)  例如 128 个 object query / 动作 token
    context:(B, Lk, D)  例如 图像 patch 特征 / VLM KV
    return: (B, Lq, D)  query 从 context 中"查"到的信息
    """
    # Q 来自 query 侧, K/V 来自 context 侧 —— 这是和 self-attn 的唯一区别
    q = proj_q(query)        # (B, Lq, D)
    k = proj_k(context)      # (B, Lk, D)
    v = proj_v(context)      # (B, Lk, D)

    scores = q @ k.transpose(-2, -1) / (D ** 0.5)   # (B, Lq, Lk)
    attn = F.softmax(scores, dim=-1)                  # 每个 query 对 context 的权重
    out = attn @ v                                    # (B, Lq, D)
    return out
```

**面试一句话**：self-attn 是「自己序列内互相看」，cross-attn 是「拿我的问题去查你的字典」；VLA 里 object query 查图像特征、动作专家查 VLM KV，都是 cross-attn。

### 2.4 Object Query 详解

**定义**：一组可学习的参数向量，每个 query 负责「查询场景中一个目标/一个槽位的特征」。

```python
class ObjectQueryDecoder(nn.Module):
    def __init__(self, num_queries=128, embed_dim=256, num_layers=3):
        super().__init__()
        # 核心: 128 个可学习向量, 随机初始化, 训练中更新
        self.object_queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # 若干层 decoder: 每层 = self-attn(query 之间交互) + cross-attn(query 查图像)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])

    def forward(self, image_feats):
        """
        image_feats: (B, L_img, D)  展平后的图像 patch 特征
        return: (B, 128, D)  128 个目标的特征
        """
        b = image_feats.shape[0]
        q = self.object_queries.unsqueeze(0).expand(b, -1, -1)  # (B, 128, D)

        for layer in self.layers:
            q = layer(q, image_feats)  # query 之间 self-attn + 查 image_feats
        return q  # (B, 128, D)
```

**为什么是 128 不是 64/512**：

- 太少：同场景交互目标（车、人、骑行者、静态障碍）经常超过 60，query 不够会「几个目标挤一个槽位」。
- 太多：冗余 query 学不到东西，attention 矩阵 512xL 图像特征算力翻倍。
- 128 还能天然构成 128 节点交互图（邻接矩阵 128x128），显式建模两两博弈。

**Query 初始化方式**（追问点）：随机初始化 / 可学习位置编码 / 从 anchor 框初始化（DETR3D 风格）。我们用可学习参数随机初始化，端到端训。

### 2.5 BEV（鸟瞰图）与相机到 BEV 的投影

**定义**：把多路环视相机的透视图像特征，投影到统一的俯视平面网格上，得到「上帝视角」特征。

**为什么要 BEV**：

- 规划在平面世界做（车道坐标系），BEV 和轨迹同构。
- 多相机特征在 BEV 下对齐，才能做跨相机的目标关联。
- 时序 BEV 可以直接 stack 出速度估计（同一格子前后帧位移）。

**LSS 风格的深度软投影（概念伪代码）**：

```python
def camera_to_bev(image_feats, intrinsics, extrinsics, depth_bins=64):
    """
    image_feats: (B, Ncam, C, H, W)  N 路相机特征
    intrinsics:  (B, Ncam, 3, 3)     内参 K
    extrinsics:  (B, Ncam, 4, 4)     外参 (camera->ego)
    depth_bins:  离散深度采样数
    return: bev (B, C, X, Y)
    """
    # 1) 每个像素预测一个深度分布 (软化, 可微)
    depth_dist = softmax(depth_head(image_feats), dim=1)  # (B,Ncam,Dbins,H,W)

    # 2) 构造每个 (u,v,depth) 的 3D 点 (相机系), 再用外参转到自车系
    # pts_ego = extrinsics @ inv(K) @ [u,v,1]*depth

    # 3) 把 3D 点按 xy 落到 BEV 网格, 深度维做 sum/softmax 聚合
    #    双线性插值保证可微
    bev = scatter_to_bev_grid(pts_ego, image_feats, depth_dist)
    return bev
```

**BEVFormer 风格的 attention BEV query（另一种主流）**：

```python
# 1) 可学习 BEV query 网格 (e.g. 50x50)
# 2) 每个 BEV 格子沿高度采几个 reference point, 投影回每路相机取特征
# 3) cross-attn: BEV query 查相机特征
# 4) 时序: BEV query 还会先和上一帧 BEV 做 deformable attention
```

**面试一句话**：BEV 是把「各相机像素」变成「统一俯视网格特征」的中间表征；实现上要么深度软投影（LSS），要么可学习 BEV query 去 cross-attn 图像（BEVFormer）。我们的风险场 32x32 网格就是在 BEV 坐标系上定义的。

### 2.6 时序特征与运动状态

**定义**：不仅看当前帧，还聚合过去 K 帧的 BEV/目标特征，使模型能估计速度、加速度、意图（打灯、减速）。

```python
def temporal_fusion(bev_current, bev_past_list, ego_motion):
    """
    bev_current: (B, C, X, Y) 当前帧 BEV
    bev_past_list: 上一帧、上上帧 ...
    ego_motion: 自车位姿变化 (用于把历史 BEV 对齐到当前自车坐标)
    """
    # 关键: 先用自车位姿把历史 BEV warp 到当前坐标系, 否则「车动了」会被当成「目标动了」
    aligned = [warp_bev(b, pose) for b, pose in zip(bev_past_list, ego_motion)]
    aligned.append(bev_current)
    stacked = torch.stack(aligned, dim=0)          # (K+1, B, C, X, Y)

    # 沿时间维做 attention 或 GRU 聚合
    fused = temporal_attention(stacked)            # (B, C, X, Y)
    return fused
```

**面试常见追问**：为什么不直接 stack？—— 不对齐自车运动的话，历史帧相对当前是「平移+旋转」过的，直接 stack 会糊掉；必须先 ego-motion compensation（运动补偿）。

### 2.7 导航指令与语言条件

自动驾驶 VLA 的「语言」模态通常是结构化导航文本：

```text
"keep lane", "merge left onto highway", "turn left at next intersection"
```

编码方式：直接喂给 VLM/文本塔 → 池化成 vector 作为生成条件；或 token 化后进 cross-attn。我们项目里导航指令与 BEV 特征拼接后作为 Flow Matching 的条件 c。

---

## 3. 世界模型（World Model）与 WAM

### 3.1 专有名词表（本节）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| World Model | 世界模型 | 给定状态与动作，预测下一状态（或状态分布）的模型 p(s'|s,a) |
| WAM | World Action Model | 同时建模动作与后果的联合世界模型 |
| OOD | Out-of-Distribution | 输入落在训练分布之外 |
| ID | In-Distribution | 输入在训练分布内 |
| Support Domain | 支持域 | 训练数据覆盖的输入区域，域内预测相对可信 |
| Trust Region | 可信域/信任域 | 策略优化中限制更新步长的区域；或世界模型中「可信输入区域」——语境需区分 |
| Rollout | - | 从某状态连续执行策略/模型，生成一段未来 |
| Counterfactual | 反事实 | 「如果当时换一个动作会怎样」的假设推演 |
| Compounding Error | 误差累积误差 | 模型自推（自己的预测当下一步输入）时误差指数放大 |
| DAgger | - | 用「自己状态下的专家标签」缓解分布偏移的经典算法 |
| Reward Hacking | 奖励作弊 | 策略找到奖励函数漏洞而非真正完成任务 |

**注意**：PPO 论文里的 trust region 是「限制新旧策略 KL」的优化概念；我们项目说的可信域/支持域是「世界模型输入是否被训练数据覆盖」的概念。面试被问到时先确认语境，再作答——这是很好的加分点。

### 3.2 世界模型：从定义到公式

**核心定义**：

```text
学习动力学:  p(s_{t+1} | s_t, a_t)

s_t: 系统状态 (所有交通参与者位姿、速度、信号灯...)
a_t: 自车动作 (轨迹段 / 加速度 / 转向角)
```

**类比**：老司机打方向前，脑内已经「放电影」——后车会不会挤、行人会不会走。这就是生物版世界模型（也与 LeCun 的「System 1/System 2」叙述相关，可作开放题素材）。

**分类（面试常考对比）**：

| 类型 | 预测目标 | 例子 | 算力 | 是否需要画出未来 |
|------|----------|------|------|------------------|
| 像素世界模型 | 未来帧像素 | DriveDreamer, GAIA-1, Genie | 极高 | 是 |
| 表征世界模型 | 未来隐向量 | JEPA, Dreamer, IRIS | 中 | 否 |
| 几何世界模型 | 未来点云/占据 | 部分 occupancy 预测 | 中 | 部分 |
| 概率轨迹预测 | 他车未来分布 | VectorNet, LaneGCN | 低 | 否 |

**为什么「预测像素」对规划是过度需求**：规划不需要知道未来树叶怎么反光，只需要知道「空不空、危不危险」。像素 loss 会被大量与决策无关的自由度淹没。

### 3.3 WAM：联合动作-后果建模

**传统割裂流水线**：

```text
轨迹生成器 -> 若干候选轨迹
评分器(独立模块) -> 对每条轨迹事后打分
选择器 -> 取最高分

问题: 生成时不知道后果, 评分时轨迹已固定; 两个模块的表征不对齐
```

**WAM 联合建模**：

```text
在同一个流/扩散过程中, 同步演化:
  z_traj        轨迹隐状态
  z_conseq      后果隐状态 (碰撞? 舒适? 进度?)
  z_risk        风险隐状态
  z_support     是否落在支持域 (可信度)

四者共享参数与时间步 t, 互相作为条件
```

**优势**：

1. 轨迹与后果表征对齐（同一个 t 的联合状态）。
2. 生成中途就能「感知」后果，早期路径就向低风险区漂移（相当于引导采样）。
3. 支持域信号与生成同步，方便在线可信路由。

**联合流的数学形式**（衔接第二部分代码）：

```text
联合状态:  z = [traj; conseq; risk; support]  ∈ R^d

流匹配条件路径:  z_t = (1-t)*ε + t*z_1,   ε ~ N(0, I)
网络预测联合速度:  v_θ(z_t, t, c) ∈ R^d
损失:  || v_θ(z_t,t,c) - (z_1 - ε) ||²
```

注意各分量可以共享一个 backbone，输出头拆成 4 段；也可以 4 个专家 + 门控（我们的实现更接近后者 + 联合注意力）。

### 3.4 OOD：定义、检测、为什么致命

**定义**：训练分布 P_train 之外的输入。自动驾驶典型 OOD：

```text
天气: 训练无暴雪, 测试遇暴雪
车辆: 训练无超高货车 / 异形工程车
交互: 训练无「公交出站连续加塞」
地图: 临时施工改道, 车道拓扑变化
传感器: 镜头污损、逆光、夜间眩光
```

**为什么危险**：神经网络对 OOD 仍会输出「自信」的预测（softmax 不会自动变谦虚），规划器把错误预测当事实 → 碰撞。

**OOD 检测常用手段**（面试加分）：

```python
def ood_score(feats, train_stats):
    # 1) Mahalanobis 距离: 到训练类中心的马氏距离
    # 2) 能量分数: E(x) = -logsumexp(logits)
    # 3) 重构/预测误差: JEPA 里 predictor 误差大 -> 可疑
    # 4) 支持域打分: 小分类器/密度估计 P_train(x)
    return score
```

我们的可信路由把「支持域分数」作为第一道门，低分直接 fallback。

### 3.5 可信域 / 支持域 / 可信路由

**支持域 Support Domain**：训练样本在输入空间覆盖的区域（更严谨：数据流形及其邻域）。

**可信路由双层评估**（我们项目的表述）：

```text
Layer 1  可靠性信任: 预测是否在支持域内?  support > tau_s ?
Layer 2  决策信任:    该预测带来的效用是否优于基线? U(traj) > U(base) ?

都通过 -> 使用世界模型输出
否则   -> fallback 传统规划 (规则/优化)
```

**代码直觉**（完整版见第二部分算法 4）：

```python
trusted = (support_score > tau_s) and (risk_score < tau_r) and (utility > u_base)
next_planner = world_model_planner if trusted else classical_planner
```

### 3.6 逆一致性约束 Inverse Consistency

**问题动机**：若只约束 forward `s,a -> s'`，网络可能忽略 `a`（很多场景忽略动作也能拟合平均运动），动作信息泄漏不到表征里。

**约束**：

```text
forward:   s' = f(s, a)
inverse:   s_hat = g(s', a')     (常用 a' = a 或学一个逆动作)
要求:      s_hat ≈ s
```

**与 CycleGAN cycle-consistency 的关系**：同构思想——双向闭环防止映射「作弊」。面试可主动类比，显示知识面。

**防作弊解读**：

```text
如果 f 完全不看 a:  f(s,a)=f(s), 那么 g 可能也只需 s' 就还原 s 的平均值
但真实世界不同 a 导致不同 s', 数据驱动下 g(s') 分布变宽, 还原误差上升
加入 cycle loss 后, f 必须用 a 区分分支, 才能让 g 精确还原
```

### 3.7 分布偏移与闭环训练

**开环训练的盲区**：数据集轨迹是人类司机的，模型执行自己的轨迹后，状态分布偏离数据集（steering bias, compounding error）。

**缓解方法**：

1. DAgger：在「模型自己的状态」上收集专家标签再训。
2. 闭环仿真 + RL（我们 Flow-GRPO 属于奖励驱动，接近这一类的离线/在线混合）。
3. 数据增强：对轨迹做扰动并标注（suffix supervision, “what if I drift”）。

**NAVSIM 是开环的**——面试若被问「0.87 离上车有多远」，标准回答里必须主动承认开环-闭环 gap，并给出可信路由/兜底策略作为安全论证的一部分。

---

## 4. JEPA：联合嵌入预测架构

### 4.1 专有名词表（本节）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| JEPA | Joint Embedding Predictive Architecture | 在嵌入空间预测被遮/未来目标的抽象表征，不重建像素 |
| I-JEPA | Image JEPA | 图像版：补全被 mask 块的表征（预测现在） |
| V-JEPA | Video JEPA | 视频版：预测下一时刻表征（预测未来，世界模型属性） |
| MAE | Masked Autoencoder | 掩码后在像素空间重建，解码器很重 |
| MLM | Masked Language Modeling | BERT 完形填空，JEPA 的 NLP 精神祖先 |
| EMA | Exponential Moving Average | 指数滑动平均，目标网络缓慢追随在线网络 |
| stop-gradient | 梯度截断 | 前向算 loss 时对某分支不反传（sg(.) 或 detach） |
| Collapse | 表征坍缩 | 所有输入映射到同一向量，loss 很低但信息为 0 |
| VICReg | Variance-Invariance-Covariance Regularization | 用方差/协方差/不变性正则防坍缩 |
| BYOL / SimSiam | - | 不负样本的对比变体，同样依赖 predictor+stopgrad 防坍缩 |
| Linear Probing | 线性探测 | 冻结主干只训线性层，评估表征是否线性可分 |
| Fine-tune | 微调 | 解冻全部/部分参数在下游任务上训练 |
| Zero-shot | 零样本 | 不见过任务标签直接推理 |
| Cycle Consistency | 循环一致性 | A->B->A 应回到 A |
| Patch | - | ViT 把图像切成的小方块，一个 patch = 一个 token |

### 4.2 JEPA vs 像素重建：为什么要换范式

```text
MAE/Diffusion 路径:  mask -> encoder -> decoder -> 重建像素 -> L2/噪声明确似然
JEPA 路径:          mask -> encoder -> predictor -> 预测目标块的 embedding -> L2 in feature space
```

**三个理由（面试标准三点）**：

1. **算力**：像素 decoder + 高分辨率 loss 很贵；表征空间只回归一个向量。
2. **语义聚焦**：纹理、光照、背景噪声在像素 loss 里权重高但对决策无用；JEPA 只逼模型学「结构与语义」。
3. **多模态未来**：路口未来可直行可左转，像素回归平均会得到「双影鬼车」；表征空间可保留多峰（配合条件/流模型）。

**代价（平衡回答）**：表征空间没有显式似然，更依赖 EMA/predictor/正则防坍缩；下游需要额外评分头才能对齐 PDMS 这种任务指标。

### 4.3 I-JEPA vs V-JEPA 对比（必考）

| 维度 | I-JEPA | V-JEPA |
|------|--------|--------|
| 输入 | 单图，挖几个 block | 视频片段的可见帧 |
| 预测目标 | 被挖块的表征（补全空间） | 未来帧的表征（预测时间） |
| 是否世界模型 | 否，更像特征学习 | 是，学习 p(z_future|z_past) |
| 我们项目 | - | 用视频帧间预测做驾驶场景预训练 |

**V-JEPA 世界模型公式**：

```text
给定帧 x_1..x_t 可见, 预测 x_{t+1} 的表征:
  z_ctx = Enc(x_1..x_t)
  z_hat = Predictor(z_ctx, pos=time_emb)
  loss  = || z_hat - sg( Enc_ema(x_{t+1}) ) ||²
```

### 4.4 JEPA 三组件与训练循环（概念伪代码）

```python
class JEPA(nn.Module):
    def __init__(self):
        self.ctx_enc  = ViT()      # 可训练, 有梯度
        self.tgt_enc  = ViT()      # EMA 更新, 无梯度
        self.predictor= Predictor()# 可训练

    def loss(self, x, mask):
        # 1. 切 patch, 分上下文/目标
        ctx_patches, tgt_patches = split_patches(x, mask)

        # 2. 上下文编码 (有梯度)
        z_ctx = self.ctx_enc(ctx_patches)

        # 3. 目标编码 (stop-grad + EMA)
        with torch.no_grad():
            z_tgt = self.tgt_enc(tgt_patches)

        # 4. 预测目标表征 (需要知道目标在哪 -> 位置条件)
        z_hat = self.predictor(z_ctx, position_of_masked_blocks)

        # 5. 特征空间 MSE
        return mse(z_hat, z_tgt)

    @torch.no_grad()
    def ema_update(self):
        for pt, ps in zip(self.tgt_enc.parameters(), self.ctx_enc.parameters()):
            pt.mul_(self.momentum).add_(ps, alpha=1 - self.momentum)
```

### 4.5 表征坍缩：机理与三道防线

**机理**：若 ctx/tgt 一起用同一 loss 优化，最优解之一是「所有向量输出常数 c」，`||c-c||²=0`。

**三道防线**：

1. **stop-gradient**：loss 对 tgt 分支不反传，tgt 不会主动「配合」ctx 变简单。
2. **EMA**：tgt 只缓慢追随 ctx，瞬间坍缩被时间常数拖住；ctx 必须预测一个「昨天的自己」，迫使保留信息。
3. **显式正则（VICReg / 冗余减除）**：方差下限 + 协方差去相关，从几何上禁止所有点重合。

**面试一句话**：EMA + stopgrad 是「让老师慢半拍且不许对答案」，VICReg 是「直接规定点云不能塌成一个点」。我们项目三道全上（EMA 0.996 + stopgrad + VICReg）。

### 4.6 VICReg 三项损失逐项拆解

```python
def vicreg_loss(z, lambda_=25.0, mu_=25.0, nu_=1.0):
    """
    z: (B, D) 批量表征
    目标: 方差足大, 协方差近 0, 同图不同增广保持不变
    """
    # --- invariance: 同图两增广 a,b 的表征应接近 ---
    # loss_inv = mse(z_a, z_b)

    # --- variance: 每一维 std >= 1, 防止退化到常数 ---
    std = torch.sqrt(z.var(dim=0) + 1e-4)          # (D,)
    loss_var = F.relu(1.0 - std).mean()            # std<1 就罚

    # --- covariance: 非对角协方差 -> 0, 防止维度共线/塌缩到低维流形 ---
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (B - 1)                    # (D, D)
    off = cov - torch.diag(torch.diag(cov))
    loss_cov = (off ** 2).sum() / z.shape[1]

    return loss_inv + lambda_ * loss_var + mu_ * loss_cov
```

**各项物理意义**：variance 防「所有点重合」；covariance 防「维度之间复制」（信息秩不够）；invariance 保持语义稳定性。

### 4.7 线性探测 vs 微调：怎么证明表征好

```text
Linear Probing:  冻结 Enc, 只训 Linear: R^D -> num_classes
  分数高 => 表征已经线性可分 => 预训练学到了语义

Full Fine-tune:  解冻全部, 小学习率
  分数高 => 表征好, 且还有可挖掘的适配空间

两者都高: 最理想
Probe 高 FT 低: 有过拟合/破坏表征风险 (少见)
Probe 低 FT 高: 表征原始但可塑 (常见于弱预训练)
```

我们 JEPA-DRIVE 阶段 2 的「冻结世界模型 + 训练评分头」本质就是大规模线性/浅层 probing 的工程化版本。

### 4.8 其他常见名词（防「没听过」）

| 名词 | 解释 |
|------|------|
| Predictive Coding | 神经科学传统：大脑不断预测下一时刻输入，只编码意外（prediction error） |
| Free Energy Principle | Friston 理论，LeCun 叙述时常引用：系统最小化预测误差界 |
| Bidirectional (BiD) training | MAE 早期变体，编码器也能看到 mask token；最终 MAE 证明 unidirectional 够用 |
| Target Network | RL 里延迟更新的网络，与 EMA target 同思想（DQN、BYOL） |
| Barlow Twins / redundancy reduction | 维度去相关思想源头，VICReg 的近亲 |
| Mask ratio | 被遮 patch 比例，MAE 常用 75%，JEPA 视任务 40%-80% |

---

## 5. Flow Matching（流匹配）

### 5.1 专有名词表（本节）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| Flow Matching | - | 监督学习回归条件概率流的速度场，训练连续生成模型 |
| ODE | Ordinary Differential Equation | 确定性微分方程：给定初值，轨迹唯一 |
| SDE | Stochastic Differential Equation | 随机微分方程：转移带噪声，可定义条件概率密度 |
| Vector Field | 速度场/向量场 | 每个空间点上给一个「往哪走、走多快」的向量 |
| Euler Integration | 欧拉积分 | x_{t+1} = x_t + v*dt，一阶数值 ODE 解法 |
| Probability Path | 概率路径 | 从噪声分布到数据分布的一族插值分布 p_t |
| Conditional Flow Matching | 条件流匹配 | 对单个 (x0,x1) 对的路径做回归，边缘化后等于 FM |
| Rectifying Flow | 整流流 | 直线插值路径的流匹配，与我们 SDE 推导同族 |
| CFG | Classifier-Free Guidance | 推理时放大条件与无条件差值，提高指令服从度 |
| Sampler | 采样器 | 推理时数值求解 ODE/SDE 的循环（步数、调度） |
| sigma_min / sigma_max | - | 噪声调度端点，控制路径两端噪声量 |
| Logit | - | softmax 前的原始分数 |

### 5.2 核心思想：把生成变成「学一条直线的速度」

```text
训练对: x0 ~ N(0,I)  (噪声),  x1 ~ p_data  (真实轨迹/图)

直线插值:  x_t = (1-t)*x0 + t*x1,   t in [0,1]

对 t 求导:  dx/dt = x1 - x0   (常数速度!)

学习目标:  网络 v_theta(x_t, t) 拟合 (x1 - x0)
```

网络在任意点 `(x_t, t)` 上如果都能答对「当前该往哪走」，推理时从噪声出发反复查询网络，就能走到数据。

### 5.3 与 DDPM 的对照表（高频考题）

| 维度 | DDPM / 扩散 | Flow Matching |
|------|-------------|---------------|
| 路径形状 | 弯曲（方差调度 + 噪声） | 直线（线性插值） |
| 网络学什么 | 噪声 ε 或 x0 | 速度场 v = x1-x0 |
| Loss | E[‖ε - ε_θ‖²] | E[‖v - v_θ‖²] |
| 推理 | 反复去噪，常用 20-50 步 | ODE 积分，常用 5-32 步 |
| 概率视角 | SDE + score | ODE + vector field |
| 数学工具 | Fokker-Planck, score matching | Continuous Normalizing Flow (CNF), ODE |
| 工程生态 | Stable Diffusion 系 | SD3, FLUX, π₀, 很多新工作 |

**为什么直线更快（直觉）**：弯路要不断「纠偏曲率」，直线每一步方向几乎不变，大步长也不太偏。

### 5.4 训练 Loss 逐步推导（代码级）

```python
def fm_loss(model, batch, t_eps=1e-5):
    """
    batch: 真实数据 (B, L, D), 例如一条轨迹展平
    """
    B = batch.shape[0]
    x1 = batch

    # Step1: 重参数化采噪声
    x0 = torch.randn_like(x1)

    # Step2: 采 t, 常用 U(0,1); 也可 importance sampling 靠近两端
    t = torch.rand(B, device=batch.device)

    # Step3: 沿直线走到中间点 (广播 t)
    t_b = t.view(B, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * x1

    # Step4: 网络看 (x_t, t) 预测速度
    v = model(x_t, t)                     # (B, L, D)

    # Step5: 目标速度是常数场 x1-x0
    target = x1 - x0

    # Step6: 均方误差, 对所有元素平均
    return F.mse_loss(v, target)
```

**变体**：

- **OT-CFM**：对 (x0,x1) 做最优传输配对，减少路径交叉（样本效率更高）。
- **条件 FM（CFM）**：`p_t(x|x1)` 单条路径回归，数学上边缘化后梯度无偏。
- **与 score 的关系**：直线高斯路径下，score 与速度有闭式关系 `v = ... score ...`，SDE 推导时要用。

### 5.5 推理：Euler 积分与步数权衡

```python
@torch.no_grad()
def fm_sample(model, shape, steps=20, cfg_scale=1.0, cond=None):
    x = torch.randn(shape)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / steps
        t = torch.full((shape[0],), t_val, device=x.device)

        # 可选 CFG: 放大条件方向
        if cfg_scale != 1.0 and cond is not None:
            v_c = model(x, t, cond)
            v_u = model(x, t, None)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = model(x, t, cond)

        x = x + v * dt          # Euler: 一步
        # 高阶: Heun = 先 Euler 试走, 再用端点速度平均校正
    return x
```

**步数权衡**：

```text
steps 少: 快, 截断误差大 (轨迹终点偏)
steps 多: 准, 慢, RL 采样成本线性涨
常用:  SFT 生成 8-16 步; Flow-GRPO 采样 32 步保 log_prob 精度
蒸馏:  consistency / step-distill 压到 1-4 步
```

### 5.6 条件生成：轨迹任务里的 c 是什么

```text
c = concat(或 cross-attn 注入):
  - BEV / 感知特征
  - 导航指令 embedding
  - z_risk 风险先验 (Risk-Init-Flow)
  - 行为模式 token (TrustDrive-WAM 的 16 组)
  - ego 状态 (速度、档位)
```

条件进网络的三种方式：加法注入（adaLN/FiLM）、拼接进 token 序列、cross-attention。我们 DiT 风格用 adaLN + cross-attn 混合。

### 5.7 手推：为什么 Euler 几步就能走对

因为目标速度场训练成 `x1-x0` 常数方向，理想情况下：

```text
v(x_t, t) = x1 - x0  对所有 t 成立

x0 + (x1-x0)*1 = x1   一步精确到位

实际网络有误差, 且不同样本速度在同一点平均 -> 场不完全恒定
所以用多步 + 可选 CFG 校正
```

面试若被问「理论上不是一步？」——答：理想场一步；真实场是边缘化后的平均场，路径会弯，需要多步数值积分，这也是为什么 Flow-GRPO 仍用 32 步循环。

---

## 6. Flow-GRPO：流匹配上的强化学习后训练

### 6.1 专有名词表（本节）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| GRPO | Group Relative Policy Optimization | 同 prompt 采一组样本，组内归一化 advantage 的策略梯度算法 |
| PPO | Proximal Policy Optimization | 用 clip 或 KL 限制更新步长的 actor-critic 算法 |
| Actor-Critic | - | actor 出动作，critic 估价值 |
| Advantage | 优势函数 | A = Q - V，动作比「平均水平」好多少 |
| Baseline | 基线 | 减小梯度方差的参照值 |
| Policy π_θ | 策略 | 参数化动作分布 |
| log_prob | 对数概率 | log π(a|s)，PPO ratio 的原料 |
| Ratio | 重要性采样比 | π_new/π_old = exp(logp_new - logp_old) |
| Clip | 裁剪 | 把 ratio 限制在 [1-ε, 1+ε] 的悲观下界 |
| KL Divergence | KL 散度 | 两个分布差异的非对称度量 |
| Reward Model | 奖励模型/打分器 | 把轨迹映射为标量分数的函数（我们用 PDM） |
| LoRA | Low-Rank Adaptation | 冻结 W0, 只训低秩增量 BA |
| Checkpoint | - | 训练存档点 |
| Constrained MDP | 约束马尔可夫决策过程 | 在期望约束下优化回报（安全 RL 常用框架） |
| λ-return / GAE | 广义优势估计 | actor-critic 里估 A 的自举方法（GRPO 不用 critic 时可不提） |

### 6.2 GRPO：从 LLM 到组内竞争

**DeepSeek-R1 风格 GRPO 核心**（LLM 离散情形）：

```text
1. 对问题 q 采 G 条回答: y1..yG ~ π_old(.|q)
2. 每条得分: r_i = reward(y_i, q)   (规则验证/裁判模型)
3. 组内归一化: A_i = (r_i - mean(r)) / (std(r)+eps)
4. 每个 token 的 PPO clip loss, 用 A_i 当全程标签
5. 没有 critic, baseline 来自组内均值
```

**为什么不用 critic**：LLM 的 value 头难训、显存翻倍；组内样本天然提供 baseline。方差 vs 便宜的经典折中。

**advantage 归一化代码**：

```python
def compute_group_advantages(rewards, group_size, clamp=2.0):
    """
    rewards: (num_groups * G,) 展平的组分数
    """
    r = rewards.view(-1, group_size)          # (Ng, G)
    mean = r.mean(dim=1, keepdim=True)
    std  = r.std(dim=1, keepdim=True) + 1e-4
    adv  = (r - mean) / std                   # 组内标准化
    adv  = torch.clamp(adv, -clamp, clamp)    # 压 outlier, 防单个异常分数主导
    return adv.reshape(-1)
```

**直觉**：一条轨迹不再和「绝对满分」比，而是和「同场景其它 23 条」比——比同伴好就提高概率。

### 6.3 连续动作的障碍：为什么不能直接 softmax log_prob

```text
LLM:  π(token=k) = softmax(logits)[k]  ∈ (0,1),  sum=1
      log_prob = log softmax, 离散可枚举

Flow: 转移 x -> x' 是确定性 ODE:  x' = x + v*dt
      确定性映射下, 条件分布是 Dirac delta
      log p(x'|x) = -inf (除 x'=f(x)) -> ratio 无意义
```

**解法路线对比**：

| 路线 | 思路 | 问题 |
|------|------|------|
| 高斯化近似 | 把每步转移当 N(mean, σ²I) | σ 要仔细设 |
| 概率流 ODE + Hutchinson score 估 | 用 score 估 log 密度 | 工程复杂，噪声大 |
| **引入 SDE**（我们/论文） | 显式加噪，转移真有高斯密度 | 与 FM 训练一致性需保持边际不变 |

### 6.4 SDE 化：四步推导（面试白板级）

**步骤 1：原 Flow 是 ODE**

```text
dx = v_theta(x,t) dt,   x_0 ~ N(0,I) -> x_1 ~ p_data
```

**步骤 2：改写成 SDE，要求边际分布不变**

```text
dx = [ f(x,t) ] dt + sigma_t dw

f(x,t) = v_theta(x,t) + (1/2) g(t)^2 * score p_t(x)
```

（漂移里加 score 项，是「加噪声还保持同一条 p_t」的标准 Fokker-Planck 补偿。）

**步骤 3：直线路径下 score 与速度的闭式关系**

对 `x_t = (1-t)x0 + t x1` 且 `x0~N(0,I)` 的边缘高斯：

```text
score p_t(x) = ( (1-t)*x1_expect - x ) / (t^2 * noise_scale)
# 实现里用可算的 v_theta 与调度参数替换, 得到只依赖网络输出的 drift
```

**步骤 4：离散化 + 写出转移高斯**

```text
Euler-Maruyama:
  mean = x_t + f_theta(x_t,t)*dt
  x_{t+1} = mean + sqrt(|dt|)*sigma_noise*eps

因此:
  p(x_{t+1}|x_t) = N(mean, (sigma_noise^2 * |dt|) * I)
  log p = -||x_{t+1}-mean||^2 / (2 s^2) - D/2*log(2*pi*s^2)
```

**实现中的均值公式**（对应第二部分代码）：

```text
mean = x_t * (1 + sn^2/(2t) * dt)
     + v_theta * (1 + sn^2*(1-t)/(2t)) * dt
sn = sqrt(t/(1-t)) * noise_level
```

（系数来自调度展开，不同论文符号略异，面试手推到「高斯转移 + score 补偿」即可，不必背到每个符号。）

### 6.5 PPO-Clip 逐项解释

```python
def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)     # = pi_new/pi_old
    surr1 = ratio * adv                        # 未裁剪
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * adv  # 裁剪
    # max 取的是"对 agent 更悲观"的下界 (adv 可正可负时的下确界)
    loss = -torch.min(surr1, surr2).mean()
    return loss
```

**分情况**：

```text
adv > 0 (轨迹好):
  希望 ratio 上升 -> 但 ratio > 1+eps 后 surr2 停止增长, 梯度截断
adv < 0 (轨迹差):
  希望 ratio 下降 -> ratio < 1-eps 后不再更狠地压
```

**为什么悲观取 min**：对最大化目标取两个估计的更小者，防止高估收益——PPO 的保守策略改进。

### 6.6 KL 惩罚与 reference model

```text
L_total = L_clip + beta * KL(pi_new || pi_ref)

pi_ref: 加载同一 checkpoint 但关闭 LoRA adapter 的原始模型
```

**作用**：

1. 防灾难性遗忘（Flow Matching 预训练生成能力）。
2. 防 reward hacking 沿奇怪方向漂移。
3. 与 clip 互补：clip 管单步，KL 管全局锚点。

**beta 调度**：太小→KL 爆炸不稳定；太大→训不动。常见 0.01-0.5 或 adaptive（目标 KL 触发）。

### 6.7 LoRA 数学与「只训 0.5%」

```text
原始: W  ∈ R^{d_out × d_in}    (冻结)
LoRA: W' = W + (alpha/r) * B A
      A ∈ R^{r × d_in},  B ∈ R^{d_out × r},  r << min(d_in,d_out)
      只训练 A, B;  初始化 A~N(0,σ), B=0 保证开始时 W'=W
```

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)   # 冻结原权重
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        # 原路径 + 低秩旁路
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
```

**为什么 RL 后训特别适合 LoRA**：

- Reward 引导的改动集中在少数方向 → 低秩假设成立。
- 显存：优化器状态只覆盖 A,B。
- 可插拔：任务切换 = 换 adapter；ref model = 同权重关 adapter。

**追问「rank 取多少」**：r=8~64 常见；我们轨迹任务维度低、数据少，偏小 r 防过拟合，总参数约 0.5%（~60M 级别相对 12B 主干）。

### 6.8 在线难例挖掘（Hard Example Mining）

```text
问题: 80% 常规场景对 GRPO 几乎无梯度 (组内分数都接近满分, adv≈0)
      真正拉开差距的是 21% corner case

做法:
  1. 按历史奖励方差/失败率给场景打 priority
  2. 采样 batch 时按 priority 加权, 提高高难场景概率
  3. 或过滤掉"全对/全错"的组 (adv 全 0 或饱和)
```

**与课程学习关系**：easy-to-hard 是主动课程；难例挖掘是被动重加权。我们 SFT+GRPO 两阶段都用了。

### 6.9 奖励设计细节（PDM + Agent 级归因）

```python
def reward_fn(traj, scenario):
    # 1) 官方 PDMS 主项
    pdms = pdms_single_trajectory(traj, scenario)   # 见 1.3

    # 2) Agent 级归因: 碰撞时定位"谁的责任/离谁最近"
    #    把稀疏的 0/1 碰撞拆成"离危险源的裕度" -> 稠密 shaping
    margin = min_over_agents(clearance(traj, agent) - safe_buffer)

    # 3) 归因加权: 对高风险交互场景, 该 agent 相关项权重放大
    w = risk_weight[scenario.hardest_agent_id]
    return pdms + w * shaped_margin
```

**为什么要 Agent 级**：全局 PDMS 有时「侥幸没碰但贴脸」得 0 缓慢改进信号；按最近威胁源做 shaping，梯度更指向真正风险。防 hacking：shaping 权重上限 + 主项仍是官方 PDMS。

### 6.10 Flow-GRPO 完整训练数据流（文字流程，不用 ASCII 图）

```text
for each train step:
  1. 取一个 batch 场景 prompts
  2. 每个场景: SDE 采样 G 条轨迹, 记录逐步 log_prob_old
  3. PDM 打分器给每条轨迹 r; 组内归一化 -> advantage
  4. 若干 inner epoch:
       对每条旧轨迹重算当前模型 log_prob_new
       ratio = exp(logp_new - logp_old)
       loss = -min(ratio*A, clip(ratio)*A) + beta*KL
       backward 只落在 LoRA 参数
  5. 可选: 更新难例权重; 记录 KL、分数、PDMS 到日志
```

---

## 7. 风险场 Risk Field

### 7.1 专有名词表（本节）

| 术语 | 一句话定义 |
|------|------------|
| Risk Field | 把所有交通参与者影响建成连续时空风险密度场 |
| Occupancy | 占据：格子是否被占用（几何，偏 0/1 或占用概率） |
| Intention | 意图：变道、让行、抢行等离散/连续行为倾向 |
| α-gating | 用可学习权重 alpha 对多源特征做软选择性聚合 |
| Softmax 门控 | 把门控 logits 变成和为 1 的权重 |
| Entropy | 熵：分布不确定性；均匀分布熵最大，one-hot 熵为 0 |
| z_risk | 风险场压缩后的隐向量，作为全局风险先验 |
| Interaction Graph | 交互图：agent 为节点、交互强度为边 |
| Adjacency Matrix | 邻接矩阵：图的 NxN 边权存储 |
| Clearance | 间距：自车与他目标的最小距离/时间裕度 |
| Risk Aversion | 风险厌恶：同等期望下偏好更低方差结果 |

### 7.2 为什么需要风险场（对比 occupancy）

```text
Occupancy 回答:  "这里有没有东西?"
Risk Field 回答: "这里的危险程度是多少, 随时间怎么变, 受谁影响?"
```

场景：旁车侵入车道但还没压线——occupancy 可能仍算「车道内空闲」，risk field 因其速度矢量和 TTA（time-to-agreement）已把该格点亮。

**时空维度**：输出 32 x 32 x 8。

```text
32 x 32: BEV 网格分辨率 (每格物理尺寸由场景范围 / 32 决定)
8:       未来 8 个时间片的风险演化 (不是单帧快照)
每个值 ∈ [0,1] 或 logit: 该时空点的风险强度
```

### 7.3 构建流程伪代码与逐行解释

```python
class RiskFieldNet(nn.Module):
    def __init__(self, n_agents=128, d=256, grid=(32, 32, 8)):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_agents, d))
        self.gate = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, 1))
        self.to_grid = nn.Linear(d, grid[0]*grid[1]*grid[2])
        self.grid = grid

    def forward(self, bev, agent_state=None):
        # (1) object query 查 BEV: 谁在哪
        #     cross-attn(Q=queries, KV=flatten(bev))
        agents = cross_attn(self.queries, flatten(bev))       # (B,128,d)

        # (2) 可选: 融入速度/类别等状态
        if agent_state is not None:
            agents = agents + agent_state_embed(agent_state)

        # (3) 交互图: 双线性 form Q W K^T 得到 128x128 边权
        #     高分边 = 强交互 (跟车、并线博弈)
        # agents = agents + graph_propagate(agents)

        # (4) alpha 软门控: 每个 agent 一个标量权重, softmax 归一
        logits = self.gate(agents).squeeze(-1)                # (B,128)
        alpha = torch.softmax(logits, dim=-1)                 # sum=1

        # (5) 加权池化 -> 解码到网格
        pooled = (alpha.unsqueeze(-1) * agents).sum(dim=1)    # (B,d)
        field = self.to_grid(pooled)                          # (B, 32*32*8)
        field = field.view(-1, *self.grid)                    # (B,32,32,8)

        # (6) 风险隐编码: 池化向量本身或再过一层
        z_risk = pooled                                       # (B,d)
        return field, z_risk, alpha
```

**alpha 为什么 softmax**：保证可加权、权重和为 1（贡献可解释、尺度稳定）；也可 sigmoid 独立门控，但会失去「相对重要性」竞争。

### 7.4 z_risk 的三个下游用途

```text
1. Risk-Init-Flow:  初始噪声 = f(z_risk) + eps   -> 生成从更合理区域出发
2. 专家门控:        熵 = H(field); g = softmax(MLP(H)) -> Base vs LTE
3. 评分器/RL:       score_input = concat(traj_feat, z_risk) -> 裕度评估
```

```python
def risk_entropy(field, eps=1e-8):
    # field 先归一化成伪概率 (softmax over spatial or sigmoid+norm)
    p = field.flatten(1)
    p = p / (p.sum(-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(-1)   # (B,) 熵
```

### 7.5 面试对比题模板：风险场 vs 预测后处理

> 传统：先各自预测轨迹，再用碰撞检测后处理筛选。交互信息在「预测」阶段是解耦的，博弈关系（他让我还是我让他）要到检测时才出现。风险场把交互前移到表征阶段，用 α 门控显式聚合，生成阶段通过 z_risk 条件化，属于「predict-then-check」到「joint risk representation」的升级。

---

## 8. 双专家 Base + LTE

### 8.1 名词

| 术语 | 定义 |
|------|------|
| LTE | Long-Tail Expert，长尾专家 |
| Mixture of Experts | 专家混合：多个子网络 + 路由 |
| Hard Routing | one-hot 选一个专家（Switch Transformer） |
| Soft Routing | 加权混合多个专家输出 |
| Curriculum Learning | 课程学习：由易到难组织数据 |
| Online Hard Mining | 在线难例挖掘：按损失/失败率重采样 |
| Bimodal Skill | 双峰技能：常规稳健 vs 长尾激进 |

### 8.2 为什么双专家（数据分布视角）

```text
数据分布:  P常规 >> P长尾
单模型 + 平均 loss  ->  梯度被常规场景主导 ->  逼近保守平均策略
双专家:   Base 拟合 P常规, LTE 在 P长尾上加权训练 -> 分布不再单一
```

**门控输入为什么用风险熵而不是 hard 标签**：

```text
hard 标签:  需要人标"这是长尾", 边界主观, 错标代价高
风险熵:     从模型自己的风险场连续导出, 可微, 天然 soft
熵高 -> 场景复杂 -> LTE 权重升
熵低 -> 常规 -> Base 权重升
```

```python
def dual_expert_forward(base, lte, risk_field):
    h = risk_entropy(risk_field)                      # (B,)
    w = torch.softmax(self.gate(h.unsqueeze(-1)), -1) # (B,2)
    v = w[:, 0:1, None] * base + w[:, 1:2, None] * lte
    return v, w
```

### 8.3 LTE 初始化与训练细节

```text
LTE 不从零初始化:  load_state_dict(Base) 后再在长尾子集 finetune
否则早期 LTE 输出噪声大, 门控学不敢用它

长尾子集来源:
  - 碰撞/压线失败回放
  - 风险熵 top-k 场景
  - 合成插入 (cut-in, 突然鬼探)
```

**防过拟合**：LTE 数据少 → 强 EMA、小 lr、早停；门控加 dropout；始终与 Base 软混合（不完全替换）。

### 8.4 Risk-Init-Flow（风险初始化流）

```text
标准 FM:  x0 ~ N(0, I)
我们:     x0 = MLP(z_risk) + sigma_small * N(0,I)

含义:  噪声分布不再全局各向同性, 而是条件于当前场景风险
       -> 初始点已偏向"风险场允许的区域", 积分路径更短、更少违例
```

**数学注意**：训练时也必须用同一 x0 分布（条件 FM），否则 train/inference mismatch。若训练仍用纯噪声、推理用风险初始化，会有分布偏移——面试若被追问，答「初始化作为条件的一部分进入训练」。

**多样性**：保留 `sigma_small * noise` 避免同场景 K 条候选完全相同（组内 advantage 需要多样性，否则 std->0 不稳定）。

---

## 9. 关键指标与缩写总表

| 缩写 | 全称 | 含义 |
|------|------|------|
| E2E | End-to-End | 端到端 |
| BEV | Bird's Eye View | 鸟瞰图 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| WAM | World Action Model | 世界动作模型 |
| JEPA | Joint Embedding Predictive Architecture | 联合嵌入预测架构 |
| FM | Flow Matching | 流匹配 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| PPO | Proximal Policy Optimization | 近端策略优化 |
| ODE | Ordinary Differential Equation | 常微分方程 |
| SDE | Stochastic Differential Equation | 随机微分方程 |
| OOD / ID | Out/In of Distribution | 分布外/内 |
| PDMS | Pilot-Driving Metric Score | NAVSIM 核心乘积指标 |
| NC | No at-fault Collision | 无责碰撞 |
| DAC | Drivable Area Compliance | 可行驶区域合规 |
| TTC | Time-to-Collision | 碰撞时间 |
| EP | Ego Progress | 自车进度 |
| SLC | Speed Limit Compliance | 限速合规 |
| LoRA | Low-Rank Adaptation | 低秩适配 |
| EMA | Exponential Moving Average | 指数移动平均 |
| ViT | Vision Transformer | 视觉 Transformer |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| MAE | Masked Autoencoder | 掩码自编码器 |
| VICReg | Variance-Invariance-Covariance Reg. | 防坍缩正则 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| RL | Reinforcement Learning | 强化学习 |
| CFG | Classifier-Free Guidance | 无分类器引导 |
| LSS | Lift-Splat-Shoot | 深度提升投影 BEV 方法名 |
| GQA / MQA / MHA | Group/Multi-Query/Multi-Head Attention | 注意力变体（K/V 头数不同） |
| RVQ | Residual Vector Quantization | 残差向量量化（RDT-2 训练用） |
| CE | Cross-Entropy | 交叉熵 |
| MSE | Mean Squared Error | 均方误差 |
| PDM | Predictive Driver Model | NAVSIM 打分器家族名 |
| FPS | Frames Per Second | 帧率 |
| TTFT / Latency | - | 首包延迟 / 端到端时延 |
| FLOPs | Floating Point Ops | 计算量 |
| KV Cache | Key-Value Cache | 自回归推理缓存 |
| adaLN | adaptive LayerNorm | 用条件调制 LN 的 scale/shift |
| FiLM | Feature-wise Linear Modulation | 条件线性调制 |
| MLP-Mixer | - | 用 MLP 替代 attention 的视觉骨干 |
| Deformable Attn | 可变形注意力 | 只在参考点附近采样的高效 attention |
| IoU | Intersection over Union | 框重叠度 |
| NMS | Non-Max Suppression | 非极大值抑制去重框 |
| TTA | Time-To-Agree / analysis | 语境：碰撞相关时间裕度 |
| MDP | Markov Decision Process | 马尔可夫决策过程 |
| HJ | Hamilton-Jacobi | 可达性分析常用 PDE（安全验证） |

---

# 第二部分：核心算法伪代码实现

> 每个算法按「它是什么（30 秒版）」->「数学/直觉」->「完整伪代码」->「逐段注释」->「面试易错点」组织。代码为教学伪代码，张量 shape 全部标注。

---

## 算法 1：Flow Matching 完整实现

### 1.1 它是什么（30 秒版）

在噪声和数据之间拉一条直线，训练网络在直线任意点回答「现在往哪走」；推理时从纯噪声出发，按网络指示的方向多走几步，到达数据分布。

### 1.2 数学（一分钟版）

```text
路径:  x_t = (1-t) x0 + t x1,  x0~N(0,I), x1~p_data, t~U[0,1]
速度:  u_t = d x_t / dt = x1 - x0   (常数场)
学习:  min_theta  E || v_theta(x_t, t) - (x1 - x0) ||^2
推理:  dx/dt = v_theta(x,t),  x(0)~N(0,I), 数值积分到 t=1
```

### 1.3 模型骨架：DiT 风格速度场网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """
    把标量时间步 t∈[0,1] 变成 sinusoidal 向量 (类比 Transformer 位置编码)
    t: (B,)
    return: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (10000.0 / half))
    # t 通常很小 (0~1), 乘一个大 scale 防止 sin 近似常数
    args = t[:, None].float() * 1000.0 * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    """一层 Transformer: adaLN 调制 + self-attn + adaLN + MLP"""
    def __init__(self, d_model=256, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        # 时间步 -> 4 个调制向量 (shift, scale, gate_attn, gate_mlp)
        self.ada = nn.Linear(d_model, 4 * d_model)

    def forward(self, x, t_emb):
        """
        x:     (B, L, D)  当前状态 token (轨迹点/patch)
        t_emb: (B, D)     时间嵌入
        return:(B, L, D)
        """
        shift1, scale1, gate1, gate2 = self.ada(t_emb).chunk(4, dim=-1)
        # adaLN: 用条件调制归一化输出  h = norm(x)*(1+scale)+shift
        h = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + gate1[:, None, :] * self.attn(h, h, h, need_weights=False)[0]
        x = x + gate2[:, None, :] * self.mlp(self.norm2(x))
        return x


class VelocityFieldDiT(nn.Module):
    """Flow Matching 速度场网络 v_theta(x_t, t, cond)"""
    def __init__(self, traj_dim=2, d_model=256, n_layers=6, nhead=8):
        super().__init__()
        self.in_proj  = nn.Linear(traj_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_proj = nn.Linear(256, d_model)   # 场景条件 (BEV/z_risk)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, traj_dim)

    def forward(self, x_t, t, cond):
        """
        x_t:  (B, L, 2)  带噪轨迹, L=未来步数 (如 30)
        t:    (B,)       时间步
        cond: (B, 256)   场景条件向量
        return:(B, L, 2) 速度场
        """
        h = self.in_proj(x_t)                 # (B, L, D)
        t_emb = self.time_mlp(timestep_embedding(t, 256))
        c = self.cond_proj(cond)              # (B, D)
        t_emb = t_emb + c                     # 条件并入时间条件 (也可用 cross-attn)

        # 每个 token 都加上同样的条件嵌入
        h = h + t_emb[:, None, :]
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(h)               # (B, L, 2)
```

### 1.4 训练一步（完整可读版）

```python
def flow_matching_train_step(model, optimizer, real_traj, cond):
    """
    real_traj: (B, L, 2) 真实未来轨迹 (已归一化到 ego 系)
    cond:      (B, 256)  感知+导航条件
    """
    model.train()
    B = real_traj.shape[0]
    optimizer.zero_grad()

    # [1] 采噪声端点 x0 和数据端点 x1
    x0 = torch.randn_like(real_traj)
    x1 = real_traj

    # [2] 采连续时间步, 训练时必须覆盖 (0,1) 全程, 否则推理某段没学过
    t = torch.rand(B, device=real_traj.device)

    # [3] 线性插值构造路径上的点 (这就是 "conditional path sample")
    t_b = t.view(B, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * x1

    # [4] 网络预测速度
    v_pred = model(x_t, t, cond)             # (B, L, 2)

    # [5] 监督目标: 直线路径的速度恒为 x1-x0
    v_target = x1 - x0                       # (B, L, 2)

    # [6] 对 batch、序列、维度全部平均, 标量 loss
    loss = F.mse_loss(v_pred, v_target)

    # [7] 标准反传; LoRA 场景下只有 adapter 参数 .grad 非 None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()
```

**逐段对照表**：

| 代码步骤 | 数学对象 | 面试一句话 |
|----------|----------|------------|
| x0, x1 | 路径两端 | 噪声端 + 真轨迹端 |
| t ~ U[0,1] | 随机时刻 | 让网络学会路径上每一点的速度 |
| x_t 插值 | 条件路径样本 | 这就是 Flow Matching 的 "Flow" |
| v_target=x1-x0 | 常数速度场 | 直线路径求导得到 |
| MSE | 回归 loss | 和 DDPM 的 eps 预测同构，只是目标不同 |

### 1.5 推理：多步 Euler 采样器

```python
@torch.no_grad()
def flow_matching_sample(model, cond, num_steps=16, shape=None):
    """
    从噪声生成一条轨迹
    shape: (B, L, 2) 若已知
    return: (B, L, 2)
    """
    model.eval()
    B = cond.shape[0]
    if shape is None:
        shape = (B, 30, 2)

    x = torch.randn(shape, device=cond.device)   # t=0 端: 纯噪声
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i / dt if False else i / num_steps   # 0, dt, 2dt, ...
        t = torch.full((B,), t_val, device=cond.device)
        v = model(x, t, cond)                # 网络说: 这里往哪走
        x = x + v * dt                       # Euler 走一小步

    # 循环结束 t≈1, x 即生成轨迹 (再反归一化到米制)
    return x
```

**Heun（二阶）变体**（追问「更高精度」时答）：

```text
k1 = v(x, t)
x_pred = x + k1*dt
k2 = v(x_pred, t+dt)
x_new = x + 0.5*(k1+k2)*dt      # 两次前向, 精度 O(dt^2)
```

### 1.6 面试易错点

1. 训练目标是 **速度** 不是「去噪后的 x1」——写成 `mse(model(x_t,t), x1)` 是错的（那是 x0-prediction 变体，FM 标准是 velocity）。
2. t 广播必须 `(B,1,1)`，写成 `(B,)` 会 broadcast 错轴。
3. 推理必须 `torch.no_grad()`，且 **不要** `model.train()`（dropout/BN 行为会变）。
4. 归一化空间要一致：训练在归一化轨迹上，推理也要在归一化空间走，最后一步再反变换。

---

## 算法 2：Flow-GRPO 完整实现

### 2.1 它是什么（30 秒版）

对 Flow Matching 模型做 GRPO 后训练：同场景采多条轨迹 -> PDM 打分 -> 组内比出好坏 -> PPO-clip 更新 LoRA。难点是连续流没有 softmax log_prob，用 **SDE 化** 每步转移写成高斯，从而 log_prob 可算。

### 2.2 SDE 单步 + log_prob（带完整公式注释）

```python
def sde_step_with_logprob(model, x_t, t, dt, cond, noise_level=0.7):
    """
    一步随机流匹配采样, 同时返回 log p(x_next|x_t)

    x_t: (B, L, 2)  当前状态
    t:   (B,)       当前时间
    dt:  float      步长 > 0
    cond:(B, 256)
    return:
      x_next:    (B, L, 2)
      log_prob:  (B,)     该步转移的对数密度 (对所有元素乘积取 log -> sum)
      mean:      (B, L, 2) 高斯均值 (KL 用)
    """
    # (1) 当前网络速度场
    v = model(x_t, t, cond)                              # (B, L, 2)

    # (2) 当前噪声水平调度  sigma(t), 这里用 t 本身
    sigma = t.clamp(1e-5, 1.0 - 1e-5)                   # (B,)
    # 离散化后的步噪声 std, 与调度匹配
    sigma_noise = torch.sqrt(sigma / (1 - sigma + 1e-8)) * noise_level

    # (3) 均值: 漂移 = 确定性流速度 + score 补偿项 (保证边际分布仍是 p_t)
    #     实现采用的显式系数 (与论文同族, 符号以实现为准):
    #     mean = x * (1 + sn^2/(2*sigma)*dt) + v * (1 + sn^2*(1-sigma)/(2*sigma)) * dt
    sn2_over_2s = (sigma_noise ** 2) / (2 * sigma + 1e-8)
    mean_x = 1.0 + sn2_over_2s * dt
    mean_v = 1.0 + (sigma_noise ** 2) * (1 - sigma) / (2 * sigma + 1e-8) * dt
    mean = x_t * mean_x[:, None, None] + v * mean_v[:, None, None]

    # (4) 采样下一状态 (Euler-Maruyama)
    std = (sigma_noise * torch.sqrt(torch.abs(dt))).clamp_min(1e-6)
    # std 形状广播到 (B,1,1) 沿元素维
    eps = torch.randn_like(x_t)
    x_next = mean + std[:, None, None] * eps

    # (5) 高斯 log-prob:
    #     log N(x; mean, s^2) = -0.5 * sum_i ((x_i-mean_i)/s)^2 - D*log(s) - 0.5*D*log(2pi)
    #     注意 detach(mean) 不必须 (我们要梯度进 mean -> 进 v -> 进 LoRA),
    #     但 x_next 在 PPO 里来自旧轨迹时应对 x_next.detach() 避免把梯度灌进采样图
    D = x_t[0].numel()
    diff = x_next.detach() - mean                          # (B, L, 2)
    log_prob = -0.5 * (diff ** 2).sum(dim=(1, 2)) / (std ** 2)
    log_prob = log_prob - D * torch.log(std) - 0.5 * D * torch.log(
        torch.tensor(2.0 * 3.141592653589793, device=x_t.device)
    )
    # 上式 D*log(s) 里 s 是标量 per-batch; 若 std 是 (B,) 则:
    #   log_prob -= D * torch.log(std)
    return x_next, log_prob, mean
```

**注意**：教学代码里系数与具体调度绑定；面试时说清「Fokker-Planck 补偿使边际不变 + 离散化得高斯转移」比背系数更重要。

### 2.3 采样一组轨迹并缓存 log_prob_old

```python
@torch.no_grad()
def sample_group(model, cond, G=24, num_steps=32, dt=None):
    """
    同一场景采 G 条轨迹
    cond: (1, 256) 或 (B,256), 下面按单场景 B=1 演示再 batch 化
    return:
      trajs:     (G, L, 2)
      logps:     (G, num_steps)  每步的 log_prob, 便于逐步 PPO
    """
    if dt is None:
        dt = 1.0 / num_steps
    trajs, logps = [], []
    for g in range(G):
        x = torch.randn(1, 30, 2, device=cond.device)
        step_logs = []
        for i in range(num_steps):
            t = torch.full((1,), i / num_steps, device=cond.device)
            x, lp, _ = sde_step_with_logprob(model, x, t, dt, cond)
            step_logs.append(lp)           # (1,)
        trajs.append(x.squeeze(0))
        logps.append(torch.stack(step_logs, dim=1).squeeze(0))  # (num_steps,)
    return torch.stack(trajs), torch.stack(logps)  # (G,L,2), (G,T)
```

### 2.4 组内优势 + PPO 主循环

```python
def flow_grpo_update(model, ref_model, trajs, logps_old, rewards, 
                     cond, opt, cfg=None):
    """
    trajs:     (B*G, L, 2)  旧轨迹 (不再重新采样)
    logps_old: (B*G, T)
    rewards:   (B*G,)
    """
    G = cfg.group_size
    eps = cfg.clip_range          # 0.2
    beta = cfg.kl_beta            # 0.05
    dt = 1.0 / cfg.num_steps

    # ---- A. 组内相对优势 ----
    r = rewards.view(-1, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    adv = adv.clamp(-2, 2).reshape(-1)            # (B*G,)

    # ---- B. 多个 inner epoch 重用旧轨迹 (off-policy) ----
    for _ in range(cfg.num_inner_epochs):
        for i in range(trajs.shape[0]):
            traj_i = trajs[i:i+1]                 # (1,L,2) 旧轨迹
            # 逐步重算当前策略下的 log_prob_new
            logp_new_seq = recompute_logprob_current(
                model, traj_i, cond[i:i+1], dt, cfg.num_steps
            )                                      # (1, T)
            logp_old = logps_old[i:i+1]            # (1, T)

            # ratio_t = exp(logp_new_t - logp_old_t)
            ratio = torch.exp(logp_new_seq - logp_old)
            a = adv[i].view(1, 1)

            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * a
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL: 当前 mean vs ref mean (同一 x_t,t)
            kl = gaussian_kl_mean(
                mean_cur=model_mean(model, traj_i, cond[i:i+1], dt),
                mean_ref=model_mean(ref_model, traj_i, cond[i:i+1], dt),
                std=cfg.std,
            )
            loss = policy_loss + beta * kl

            opt.zero_grad()
            loss.backward()
            # 只裁 LoRA: 其它参数 grad 本应为 None; 若全参则手动 mask
            opt.step()
    return adv.mean().item(), rewards.mean().item()
```

### 2.5 recompute_logprob_current：PPO 的关键「重算」

```python
def recompute_logprob_current(model, old_traj, cond, dt, T):
    """
    关键思想:
      - 轨迹状态序列取自旧轨迹 (x_0, x_1, ..., x_T)  固定
      - 但每个 (x_t -> x_{t+1}) 的高斯均值用【当前】模型算
      - 于是 logp_new 随参数变化, 可以反传; 轨迹本身不重新采样
    """
    L = old_traj.shape[1]
    # 构造路径点: 把旧轨迹看作 SDE 网格上的状态
    # 教学简化: 直接把轨迹点当作 x_{t_j}; 真实实现还依赖流匹配的 x_t 构造
    logps = []
    for j in range(T):
        # x_t 近似: 带噪状态 = 插值或直接用轨迹点+噪声, 与采样时一致
        x_t = old_traj  # 简化示意; 实现需与 sde 采样时状态定义完全一致
        t = torch.full((old_traj.shape[0],), j / T, device=old_traj.device)
        _, lp, _ = sde_step_with_logprob(model, x_t, t, dt, cond)
        logps.append(lp)
    return torch.stack(logps, dim=1)   # (1, T)
```

**面试必须说清的三点**：

1. 轨迹 **不重新采样**（重要性采样框架，靠 ratio 纠正新旧策略差）。
2. `x_next` 在算 loss 时要用旧轨迹的 `x_{t+1}`，均值用新模型——写成「新模型重新 roll out 再算」会让策略梯度变成 on-policy 发散。
3. log_prob 是 **逐步** 的，PPO 可以逐步 clip，也可以把整条轨迹 logp 当 token 序列 sum——LLM GRPO 是 per-token，我们 per-step。

### 2.6 计算图（文字描述，避免 ASCII 图错位）

```text
标量 loss
  <- min(ratio*A, clip(ratio)*A)
  <- ratio = exp(logp_new - logp_old)
  <- logp_new = GaussianLogProb(x_next_old, mean_theta, std)
  <- mean_theta = drift(x_t, v_theta, t)     (含 score 补偿)
  <- v_theta = VelocityFieldDiT(x_t, t, cond)
  <- theta 中只有 requires_grad=True 的 LoRA A,B 获得 .grad
  -> optimizer.step() 只改 adapter; base 权重字节级不变
```

### 2.7 面试易错点

1. **advantage 归一化必须按组 view(-1,G)**，不能全局归一化（组间难度不同）。
2. **ratio 数值稳定**：用 `exp(logp_new - logp_old)` 不要 `exp(new)/exp(old)`。
3. **ref model 必须 eval + no_grad**，且关闭 adapter（同一 base 权重）。
4. 采样用 SDE、训练 loss 与采样同一调度，否则 train/test mismatch。
5. GRPO 组内 std 过小（分数全一样）时 advantage 全 0，该 batch 无梯度——难例挖掘的动机。

---

## 算法 3：VLA 轨迹生成（RiskField-VLA）

### 3.1 它是什么（30 秒版）

用 128 个 object query 从 BEV 中提取 agent 特征，聚合成时空风险场，用风险熵门控 Base/LTE 双专家，风险编码优化 Flow Matching 初始噪声，生成多条候选轨迹供打分选择。

### 3.2 从相机到 agent 特征

```python
class PerceptionHead(nn.Module):
    def __init__(self, num_queries=128, d=256):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_queries, d))
        self.layers = nn.ModuleList([DecoderLayer(d) for _ in range(3)])

    def forward(self, cam_feats):
        """
        cam_feats: (B, Ncam, C, H, W)  N=6~8 路环视
        通常先展平+pos_emb: (B, Ncam*H*W, C)
        return: (B, 128, d)
        """
        kv = flatten_with_pos(cam_feats)       # (B, Lkv, C)
        x = self.q[None].expand(cam_feats.size(0), -1, -1)
        for lyr in self.layers:
            # DecoderLayer = self-attn(128 个 query 互看) + cross-attn(查 kv)
            x = lyr(x, kv)
        return x                               # 每个 query 聚焦一类目标
```

### 3.3 交互图传播（可选模块）

```python
def graph_propagate(nodes, topk=16):
    """
    nodes: (B, N, D) N=128
    边权 = softmax_k( QK^T / sqrt(d) ) 只保留每行 top-k 避免 O(N^2) 噪声
    """
    Q = nodes
    K = nodes
    logits = Q @ K.transpose(-2, -1) / (nodes.size(-1) ** 0.5)  # (B,N,N)

    # 可选 mask: 用欧氏距离/类别剔除不可能交互对
    topk_val, topk_idx = logits.topk(topk, dim=-1)
    attn = F.softmax(topk_val, dim=-1)                          # (B,N,k)

    # gather 邻居值并加权
    neigh = torch.gather(
        nodes.unsqueeze(2).expand(-1, -1, topk, -1),  # (B,N,k,D)
        dim=2,
        index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, nodes.size(-1)),
    )
    out = (attn.unsqueeze(-1) * neigh).sum(dim=2)                # (B,N,D)
    return nodes + out   # residual
```

### 3.4 风险场 + z_risk（完整前向）

```python
class RiskFieldBuilder(nn.Module):
    def __init__(self, n=128, d=256, grid=(32, 32, 8)):
        super().__init__()
        self.grid = grid
        self.gate = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
        self.decod = nn.Linear(d, grid[0] * grid[1] * grid[2])
        self.norm = nn.LayerNorm(d)

    def forward(self, agents):
        """
        agents: (B,128,D)  来自 PerceptionHead (+ 可选 graph)
        return: field (B,32,32,8), z_risk (B,D), alpha (B,128)
        """
        alpha = torch.softmax(self.gate(agents).squeeze(-1), dim=-1)  # (B,128)
        pooled = (alpha.unsqueeze(-1) * agents).sum(1)                 # (B,D)
        pooled = self.norm(pooled)
        field = self.decod(pooled).view(-1, *self.grid)                # (B,32,32,8)
        # 风险强度用 sigmoid 压到 (0,1) 可解释
        field = torch.sigmoid(field)
        return field, pooled, alpha
```

### 3.5 Risk-Init-Flow 双专家生成

```python
class RiskFlowPlanner(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.base = VelocityFieldDiT(d_model=d)
        self.lte  = VelocityFieldDiT(d_model=d)
        self.noise_head = nn.Sequential(
            nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 30 * 2)
        )
        self.gate = nn.Linear(1, 2)

    def init_noise(self, z_risk, B):
        """Risk-Init-Flow: 主方向来自风险, 保留小随机保多样性"""
        main = self.noise_head(z_risk).view(B, 30, 2)
        return main + 0.1 * torch.randn_like(main)

    def expert_weights(self, field):
        # 风险熵高 -> 场景长尾 -> LTE 权重应升高 (gate 自己学这个映射)
        p = field.flatten(1)
        p = p / (p.sum(-1, keepdim=True) + 1e-8)
        h = -(p * (p + 1e-8).log()).sum(-1, keepdim=True)   # (B,1)
        return torch.softmax(self.gate(h), dim=-1)           # (B,2)

    @torch.no_grad()
    def generate(self, z_risk, field, cond, K=8, steps=32):
        B = z_risk.shape[0]
        w = self.expert_weights(field)                       # (B,2)
        cands = []
        for _ in range(K):                                   # K 条候选
            x = self.init_noise(z_risk, B)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((B,), i / steps, device=x.device)
                vb = self.base(x, t, cond)
                vl = self.lte(x, t, cond)
                # 软混合, 不 hard switch (训练更稳)
                v = w[:, 0:1, None] * vb + w[:, 1:2, None] * vl
                x = x + v * dt
            cands.append(x)
        return torch.stack(cands, dim=1)                     # (B,K,30,2)
```

### 3.6 选择与评分（推理期）

```python
def select_best(cands, scorer, z_risk):
    """
    cands: (B,K,30,2)
    多目标: score = PDMS_proxy - lane_dev - comfort_pen + progress
    """
    scores = scorer(cands, z_risk)   # (B,K)
    idx = scores.argmax(dim=1)       # 每个 batch 取一条
    best = cands[torch.arange(cands.size(0)), idx]
    return best, scores
```

### 3.7 面试易错点

1. 说清 **128 query 是可学习槽位**，不是检测框的替代 NMS 输出——可以输出「无目标」的低权重 query。
2. Risk-Init 的噪声头必须 **参与训练**（和 FM loss 或 aux loss 联合），推理时突然换初始化会 mismatch。
3. 双专家是 **软门控混合**，不是 if-else 硬路由；硬路由会有梯度断点、训练不稳。
4. 生成 K 条是为了组内多样性，K=1 就没有 GRPO 的 std。

---

## 算法 4：WAM 联合动作-后果流（TrustDrive-WAM）

### 4.1 它是什么（30 秒版）

把轨迹、后果、风险、支持域四条隐变量放在 **同一个流** 里联合去噪/积分；生成过程中心跳式更新可信度，由可信路由决定信世界模型还是回退经典规划。

### 4.2 联合状态打包与拆包

```python
# 维度设计示例
TRAJ_DIM = 60        # 30 步 * (x,y)
CONQ_DIM = 32        # 后果向量: 碰撞裕度, 舒适, 进度, 合规 ...
RISK_DIM = 16
SUPP_DIM = 8         # 支持域 logit
JOINT_DIM = TRAJ_DIM + CONQ_DIM + RISK_DIM + SUPP_DIM  # 116

def pack(z_traj, z_conq, z_risk, z_supp):
    return torch.cat([z_traj, z_conq, z_risk, z_supp], dim=-1)

def unpack(z):
    a = z[..., :TRAJ_DIM]
    b = z[..., TRAJ_DIM:TRAJ_DIM+CONQ_DIM]
    c = z[..., TRAJ_DIM+CONQ_DIM:TRAJ_DIM+CONQ_DIM+RISK_DIM]
    d = z[..., TRAJ_DIM+CONQ_DIM+RISK_DIM:]
    return a, b, c, d
```

### 4.3 联合速度场与训练

```python
class JointVelocityNet(nn.Module):
    def __init__(self, d_joint=116, d_model=512, cond_dim=256):
        super().__init__()
        self.inp = nn.Linear(d_joint, d_model)
        self.cond = nn.Linear(cond_dim, d_model)
        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.out = nn.Linear(d_model, d_joint)

    def forward(self, z_t, t, cond):
        h = self.inp(z_t) + self.cond(cond) + timestep_embedding(t, d_model)
        h = self.backbone(h)
        return self.out(h)   # 联合速度, 再 unpack 成 4 段


def joint_fm_loss(model, batch_traj, conq_gt, risk_gt, supp_gt, cond):
    """
    所有分量端点打包成 z1, 与联合噪声线性插值
    """
    z1 = pack(batch_traj, conq_gt, risk_gt, supp_gt)
    z0 = torch.randn_like(z1)
    t = torch.rand(z1.size(0), device=z1.device)
    t_b = t.view(-1, 1)
    z_t = (1 - t_b) * z0 + t_b * z1
    v = model(z_t, t, cond)
    return F.mse_loss(v, z1 - z0)
```

**为什么四条要同一条流**：共享 t 与联合速度，后果分量始终「贴着」轨迹分量演化；若四个独立流各自积分，对应时刻的 (轨迹, 后果) 语义不对齐，评分头会学到错位配对。

### 4.4 行为模式 token 与多峰

```python
class BehaviorTokens(nn.Module):
    def __init__(self, n_modes=16, cond_dim=256):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_modes, cond_dim))
        self.mode_proj = nn.Linear(cond_dim, cond_dim)

    def forward(self, mode_ids):
        """
        mode_ids: (B,) 或 (B,K) 每个候选一个模式 id
        推理: 枚举 0..15 得 16 条不同风格候选
        训练: GT 模式可由驾驶风格标签指定, 或 max 多模态匹配 ( winner-takes-all )
        """
        return self.mode_proj(self.tokens[mode_ids])  # (B, cond_dim)
```

**多峰 loss（WTA）**：

```python
# 16 条预测与 GT 轨迹两两距离, 只回传最近一条 (mode collapse 比全平均好)
d = torch.cdist(pred_modes.view(16, -1), gt.view(1, -1))  # (16,)
loss_wta = d.min()
```

### 4.5 逆一致性网络

```python
class InverseConsistency(nn.Module):
    def __init__(self):
        self.forward_net = Dynamics()   # (s, a) -> s'
        self.inverse_net = Dynamics()   # (s', a') -> s_hat

    def loss(self, s, a):
        s_next = self.forward_net(s, a)
        # 逆向用同一动作 (或学 delta_a); 教学取 a'=a
        s_hat = self.inverse_net(s_next, a)
        return F.mse_loss(s_hat, s)
```

**加入后果后的 cycle**（我们的版本）：

```text
(traj -> conseq) 与  (conseq -> 重建 traj) 对齐
loss_cycle = mse(traj, recon_traj(conseq))
```

### 4.6 可信路由完整代码

```python
class TrustRouter:
    def __init__(self, tau_supp=0.8, tau_risk=0.4, tau_gain=0.0):
        self.tau_supp = tau_supp
        self.tau_risk = tau_risk
        self.tau_gain = tau_gain

    @torch.no_grad()
    def route(self, supp_logit, risk, util_wm, util_base):
        """
        supp_logit: (B,) 支持域打分 (0-1)
        risk:       (B,) 世界模型预测的风险
        util_wm:    (B,) 用 WM 轨迹的效用 (PDMS 线性代理)
        util_base:  (B,) 经典规划器效用
        return: use_wm (B,) bool
        """
        p_supp = torch.sigmoid(supp_logit)
        layer1 = (p_supp > self.tau_supp) & (risk < self.tau_risk)
        layer2 = (util_wm - util_base) > self.tau_gain
        return layer1 & layer2

    def plan(self, wm_trajs, base_traj, **kw):
        use = self.route(**kw)                    # (B,)
        # where 选: True 用 WM 最优, False 用 base
        best_wm = wm_trajs.argmax_over_utility()  # 示意
        out = torch.where(use.view(-1, 1, 1), best_wm, base_traj)
        return out, use.float().mean()            # 返回平均信任率做日志
```

**阈值怎么定**：验证集扫 tau，目标是「信任率-风险曲线」的拐点；部署可保守（高 tau_s）。面试答「验证集标定 + 故障注入测 fallback 召回率」。

### 4.7 面试易错点

1. WAM 不是「先生成再评」——后果是 **并行积分** 出来的，不是后处理算出来的。
2. 可信路由是 **双层**（可靠性 + 效用），只说「加个 if 判断」太浅。
3. 逆一致性 **不能** 去掉 stopgrad 或自由学 a'——否则 trivial 解（forward 恒等、inverse 恒等）也能低 loss。
4. Fallback 基线必须 **始终前向**（或懒计算），否则 `use=False` 路径没有轨迹。

---

## 算法 5：JEPA 全流程（JEPA-DRIVE）

### 5.1 它是什么（30 秋版）

三阶段：1) 视频帧间 JEPA 自监督学场景动力学表征；2) 冻结世界模型，训隐空间评分头对齐 PDMS；3) 感知-世界模型-生成-评分端到端微调。全程不依赖像素重建。

### 5.2 Patch 化与 Mask

```python
def patchify(x, patch=16):
    """x: (B,3,H,W) -> (B, N, patch*patch*3)  ViT 标准切法"""
    B, C, H, W = x.shape
    p = patch
    x = x.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)
    return x


def make_mask(B, N, ratio=0.75, device='cpu'):
    """随机挖掉 ratio 比例的 patch 作为预测目标"""
    n_mask = int(N * ratio)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for i in range(B):
        idx = torch.randperm(N, device=device)[:n_mask]
        mask[i, idx] = True   # True = 目标块 (被挖)
    return mask
```

**可选 block mask**（I-JEPA 原版用大块，学长程结构）：

```python
def block_mask(B, H_grid, W_grid, size=8):
    # 每张图随机选若干 size x size 连续块
    ...
```

### 5.3 JEPA 模块

```python
class JEPA(nn.Module):
    def __init__(self, enc=None, d=768, pred_depth=6, pred_dim=1024):
        super().__init__()
        self.ctx_enc = enc or ViT(patch_size=16, embed_dim=d)
        self.tgt_enc = copy.deepcopy(self.ctx_enc)   # 结构完全一致
        for p in self.tgt_enc.parameters():
            p.requires_grad = False                   # 不进 optimizer

        self.predictor = nn.Sequential(
            nn.Linear(d + 128, pred_dim),            # + 位置/时间条件
            nn.GELU(),
            nn.Linear(pred_dim, pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, d),
        )
        self.momentum = 0.996

    def forward(self, frames, mask):
        """
        frames: (B,3,H,W) 或 (B,T,3,H,W) 时间维展进 batch
        mask:   (B,N) True=目标
        """
        patches = patchify(frames)                          # (B,N,Dp)
        ctx = patches[~mask]                                 # (M, Dp)
        tgt = patches[mask]                                 # (K, Dp)

        z_ctx = self.ctx_enc(ctx)                           # (M, d) 可训
        with torch.no_grad():
            z_tgt = self.tgt_enc(tgt)                       # (K, d) EMA+sg

        # 需要对齐: 每个目标位置的预测 = 上下文汇聚 + 位置嵌入
        # 教学简化: 全局池化预测一个向量, 正式版 per-target token
        ctx_pool = z_ctx.mean(0, keepdim=True).expand(z_tgt.size(0), -1)
        pos = pos_embed_for_masked(mask)
        z_hat = self.predictor(torch.cat([ctx_pool, pos], dim=-1))

        return F.mse_loss(z_hat, z_tgt)

    @torch.no_grad()
    def ema_update(self):
        for pt, ps in zip(self.tgt_enc.parameters(), self.ctx_enc.parameters()):
            pt.data.lerp_(ps.data, 1.0 - self.momentum)  # 等价 EMA
```

**正式版 per-target**：predictor 输入应为 `(z_ctx_tokens, mask_pos_tokens)` 的交叉注意力输出，每个目标 patch 出一个预测——面试可主动说「全局池化是教学简化」。

### 5.4 三阶段训练

```python
# ========== Stage 1: JEPA 自监督 ==========
def stage1(model, loader, steps=100000):
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for step, videos in enumerate(loader):
        # videos: (B, T, 3, H, W) 连续帧
        t = random.randrange(videos.size(1) - 1)
        x_t = videos[:, t]
        # 目标是下一帧 (世界模型属性) 或同帧 mask 块 (I-JEPA 属性)
        x_next = videos[:, t + 1]
        mask = make_mask(B, N, ratio=0.75)

        # 可见上下文来自 x_t, 目标块来自 x_next 的表征 (时序 JEPA)
        loss = model.temporal_loss(x_t, x_next, mask)
        loss.backward(); opt.step()
        model.ema_update()

# ========== Stage 2: 冻结 WM, 训评分头 ==========
def stage2(model, scorer, loader):
    freeze(model)
    for batch in loader:
        with torch.no_grad():
            z = model.ctx_enc(batch['visible_patches'])
        # score head: 预测候选轨迹的 PDMS 或排序
        pred = scorer(z, batch['cand_trajs'])     # (B,K)
        # 排序 loss: 对齐官方指标的相对序
        loss = listwise_rank_loss(pred, batch['pdms'])
        loss.backward(); opt_scorer.step()

# ========== Stage 3: 端到端联合 ==========
def stage3(full_model, loader):
    unfreeze_perception(full_model)
    for batch in loader:
        z_scene = full_model.perception(batch['imgs'])
        z_future = full_model.world_model(z_scene)     # 未来隐表征
        trajs = full_model.generator(z_future)         # 密集轨迹
        scores = full_model.scorer(trajs, z_future)
        loss = (
            rank_loss(scores, batch['pdms'])
            + 0.1 * full_model.jepa_aux(batch)         # 保持动力学
            + 0.05 * vicreg_loss(z_scene)              # 防坍缩
        )
        loss.backward(); opt.step()
```

### 5.5 65536 条密集轨迹的因子化生成

```text
朴素:  直接回归 65536 条 -> 参数爆炸
因子化:  path (空间形状) x speed (时间分配) 独立生成再组合
  path:  256 条几何路径 (曲率/偏移模式)
  speed: 256 种速度剖面 (加速/巡航/减速)
  组合:  256 * 256 = 65536
评分:  隐空间 scorer 批处理所有组合 (不生成像素)
选择:  argmax score, 或 top-m 进入下游控制
```

```python
def dense_candidates(paths, speeds):
    """
    paths: (B, 256, L, 2)
    speeds: (B, 256, L, 1)  时间缩放/速度系数
    return: (B, 65536, L, 2)  广播组合 — 实现上可 chunk 评分别物化全量
    """
    B, P, L, _ = paths.shape
    S = speeds.size(1)
    traj = paths.unsqueeze(2) * speeds.unsqueeze(1)  # (B,P,S,L,1)*(B,P,S,L,2)
    return traj.view(B, P * S, L, 2)
```

**工程点**：65536 不一定物化，scorer 可分块算 top-k 再精排。

### 5.6 Cycle-Energy 双空间自监督

```python
def cycle_energy_loss(enc_ctx, enc_tgt, pred, x_a, x_b):
    """
    正向: A 可见 -> 预测 B 被 mask 处的表征
    逆向: B 可见 -> 预测 A
    能量: 重构误差本身可当不确定性 (能量越大越不可信 -> 支持域信号)
    """
    # forward
    z_a = enc_ctx(observe(x_a, mask_b_side))
    z_b_tgt = enc_tgt(target(x_b, mask_b_side))     # no grad
    e_fwd = F.mse_loss(pred(z_a, pos_b), z_b_tgt)

    # backward
    z_b = enc_ctx(observe(x_b, mask_a_side))
    z_a_tgt = enc_tgt(target(x_a, mask_a_side))
    e_bwd = F.mse_loss(pred(z_b, pos_a), z_a_tgt)

    energy = e_fwd + e_bwd
    return energy, energy.detach()   # loss 用 energy; 能量值给 router
```

**能量 -> 支持域**：训练分布内 energy 低，OOD/长尾 energy 高；用验证集拟合 `support = sigmoid(a - b*energy)` 或分位数归一。

### 5.7 VICReg（完整可运行风格）

```python
def vicreg_loss(za, zb, lam=25.0, mu=25.0, nu=1.0):
    """za, zb: 同图两增广 (B,D)"""
    # invariance
    inv = F.mse_loss(za, zb)
    # variance on centered batch of za
    std = torch.sqrt(za.var(dim=0, unbiased=False) + 1e-4)
    var_term = F.relu(1.0 - std).mean()
    # covariance
    z = za - za.mean(dim=0)
    cov = (z.T @ z) / (za.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_term = (off ** 2).sum() / za.size(1)
    return inv + lam * var_term + mu * cov_term
```

### 5.8 面试易错点

1. Target encoder **不进 optimizer**——用 `requires_grad_(False)` + EMA，不是「有梯度但 lr=0」。
2. `loss.backward()` 时图里不能还有 tgt 分支的可微路径——`torch.no_grad()` 或 `detach()` 必须有。
3. 阶段 2 评分头用 **排序 loss** 不是 MSE——PDMS 是乘积/序数指标，绝对值校准次要。
4. 三阶段的「冻结/解冻」边界要答清楚：S1 冻感知了吗？我们 S1 主要练 WM；S2 全冻 WM；S3 解感知联合调。
5. JEPA-DRIVE 0.7285 低于另两项目，主动解释：基线未做满 RL 后训练 + 无标注设定 + 范式验证阶段。

---

## 算法 6：辅助组件速查（面试手写题高频）

### 6.1 Softmax 与温度

```python
def softmax_with_temperature(logits, tau=1.0):
    # tau -> 0 近似 argmax (更确定); tau 大分布更平 (更多探索)
    return F.softmax(logits / tau, dim=-1)
```

GRPO 采样时 SDE 的 `noise_level` / 策略熵 bonus 扮演类似「温度」角色。

### 6.2 KL 散度（高斯与样本估计）

```python
def kl_gaussian(mu1, logvar1, mu2, logvar2):
    # KL(N1 || N2)
    return (logvar2 - logvar1 + (logvar1.exp() + (mu1-mu2)**2) / logvar2.exp() - 1) * 0.5

def kl_sample_estimate(p_samples, logp, logq):
    # E_p[logp - logq]
    return (logp - logq).mean()
```

### 6.3 GAE（若被问 actor-critic，对比 GRPO 用不上）

```python
def gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv
```

**对比**：GAE 需要 value 网络；GRPO 用组均值当 baseline，省 critic。

### 6.4 温和的 Reward 归一与 winsorize

```python
def robust_normalize(r, group_size, lo=-2, hi=2):
    r = r.view(-1, group_size)
    r = (r - r.mean(1, True)) / (r.std(1, True) + 1e-4)
    return r.clamp(lo, hi).view(-1)
```

### 6.5 学习率与 warmup（工程追问）

```text
SFT: 1e-4 ~ 3e-4, cosine decay
GRPO: 1e-5 ~ 5e-5 (更小, 防 KL 爆)
warmup 3% steps;  grad clip 1.0;  LoRA dropout 0.05
```

---

# 第二部分补章：四大面试必问「项目怎么做的」完整话术

> 面试官最常问的五个问题：**你负责什么？Flow Matching 怎么做的？VLA 怎么做的？世界模型/世界动作模型怎么做的？JEPA 具体怎么做的？**
>
> 本章为每个项目准备一段 **3-5 分钟口述稿**，结构统一为：
> 【我负责什么】->【整体 pipeline】->【核心算法 + 代码怎么写的】->【为什么这样设计】->【效果与追问预案】。
> 口述时按「先说我在哪一环」再展开，避免一上来就堆名词。括号内是**可略过的深水细节**，面试官感兴趣再讲。

---

## 话术 A：Flow Matching —— 「你们 Flow Matching 具体怎么做的？」

### A.0 30 秒电梯版（先接住问题）

> Flow Matching 是我们轨迹生成器的**生成范式**，替代扩散模型的 DDPM。一句话：在高斯噪声和真实轨迹之间构造**直线概率路径**，训练一个 Transformer 网络去回归这条路径上的**速度场**；推理时从噪声出发，用 Euler 积分 16~32 步就能走出一条轨迹。相比 DDPM 的 20~50 步去噪，直线路径更快、更好训，也方便我们后面接 GRPO 做 RL 后训练。

### A.1 我负责什么

> 在这个子模块里我负责三件事：一是**速度场网络的结构选型与实现**（DiT + adaLN 条件注入）；二是**训练目标与路径采样**（conditional flow matching 的 loss）；三是**采样器与后续 RL 的衔接**——因为 Flow-GRPO 要在每一步上算 log-prob，我把确定性 ODE 采样改造成了可算 log-prob 的 SDE 形式。整个生成器输入是感知出来的条件向量 `cond`（BEV 池化 + 导航指令 + 风险先验拼起来），输出是未来 30 步的自车轨迹 `（30, 2）`。

### A.2 整体 pipeline（口述顺序）

> 整条链路是：
> 1. 真实轨迹做**归一化**（以自车为原点、朝向对齐），得到 `x1 ~ (B, 30, 2)`；
> 2. 每个训练步采一个 `x0 ~ N(0, I)` 同 shape 的噪声，再采 `t ~ U(0,1)`；
> 3. 沿直线插值出中间状态 `x_t = (1-t)*x0 + t*x1`；
> 4. 网络吃 `(x_t, t, cond)`，输出速度预测 `v_pred`；
> 5. 回归目标就是常数速度 `x1 - x0`，MSE 一步反传。
>
> 推理时从纯噪声出发，for 循环里每步 `x = x + v*dt`，`dt = 1/steps`，走完 steps 步得到轨迹，再反归一化回米制坐标。

### A.3 核心代码（面试可直接默写的骨架）

```python
# ---- 训练一步: 10 行核心 ----
def fm_train_step(model, x1, cond):
    B = x1.shape[0]
    x0 = torch.randn_like(x1)                 # 噪声端
    t  = torch.rand(B, device=x1.device)      # 随机时间
    tb = t.view(B, 1, 1)                      # 广播到 (B,L,2)
    xt = (1 - tb) * x0 + tb * x1              # 直线插值
    v  = model(xt, t, cond)                   # 网络预测速度
    return F.mse_loss(v, x1 - x0)             # 目标 = 常数速度场

# ---- 推理采样: Euler ----
@torch.no_grad()
def fm_sample(model, cond, steps=20):
    x = torch.randn(cond.size(0), 30, 2, device=cond.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.size(0),), i / steps, device=cond.device)
        v = model(x, t, cond)
        x = x + v * dt                        # 一步积分
    return x                                  # 归一化空间轨迹
```

**速度场网络结构（一句话带过 + 深水细节）**：

> 网络是 DiT 风格：轨迹点先 `Linear` 升到 256 维，时间步做 sinusoidal embedding 过 MLP，再和条件向量相加做 **adaLN** 调制，堆 6 层 Transformer block（每层 self-attn 建模 30 个轨迹点之间的时序依赖），最后 `Linear` 投回 `(30, 2)`。条件注入我们试过拼接和 cross-attn，最终 adaLN + 加法最稳、开销最小。

### A.4 为什么不用 DDPM（必被追问）

| 点 | DDPM | Flow Matching（我们） |
|----|------|----------------------|
| 路径 | 方差调度的弯路 | 直线，速度恒为 `x1-x0` |
| 目标 | 预测噪声 ε | 预测速度 v |
| 步数 | 20-50 | 16-32（RL 还要复用，步数越少采样越便宜） |
| RL 接口 | score / ε 转 log-prob 绕 | SDE 化后每步直接是高斯，log-prob 闭式 |

> 补充一句有深度的：**我们选 FM 不只是快，更是为了 GRPO**。LLM 的 GRPO 靠 `softmax` 拿 log-prob，DDPM 的反向高斯也有，但 FM 的直线路径在「加一点噪声转成 SDE」时 score 和速度有闭式关系，漂移项能写干净，这是我后面做 Flow-GRPO 推导时的前提。

### A.5 效果 + 追问预案

**Q：轨迹多样性和模式坍缩怎么办？**
> 三层：一是采样端保留随机噪声（推理也是从随机 `x0` 出发）；二是条件端挂 16 个可学习行为 token，推理时枚举不同 token 得到保守/激进/变道等不同风格的候选；三是打分器在 top-K 里选，不只取一条。Risk-Init 版本里初始噪声还加了 `0.1*randn` 防止同场景 K 条一模一样——组内 GRPO 需要 std>0。

**Q：steps 能压到 1 吗？**
> 可以做 consistency distillation / step-distill，但我们 RL 阶段仍用 32 步：一是 log-prob 累计更细，二是轨迹只有 30 点、算力可接受，32 步单条采样毫秒级。部署前可以再蒸馏。

**Q：损失为什么不直接 MSE 到 x1？**
> 那是 x0-prediction 或直接回归均值，会把多模态回归到平均（鬼影轨迹）。FM 回归的是**速度场**，分布信息藏在向量场和随机初始里，多模态靠多次采样 `x0` 展开。

---

## 话术 B：VLA —— 「VLA 你们怎么做的？你负责什么？」

### B.0 30 秒电梯版

> 我们的 VLA 是 **RiskField-VLA**：视觉（8 路环视）+ 语言（导航指令）进一个感知大模型，输出不是离散 token，而是**连续轨迹分布**，用 Flow Matching 解码。项目名里的 RiskField 是我这边的核心创新——把 128 个 object query 聚合成 **32×32×8 的时空风险场**，再用风险场做双专家门控和初始噪声条件，最后 Flow-GRPO 做后训练。NAVSIM PDMS **0.8713**。

### B.1 我负责什么（划清边界，面试必问）

> 我负责的是**感知之后的「风险建模 + 轨迹生成 + RL 后训练」全链路**，具体三块：
> 1. **风险场模块**：从 object query 特征到时空风险场、`z_risk` 风险隐编码、风险熵门控；
> 2. **生成器**：双专家（Base/LTE）+ Risk-Init-Flow 的 Flow Matching 轨迹解码，以及候选选择；
> 3. **Flow-GRPO 后训练**：SDE 化采样、组内优势、PPO-clip + KL，LoRA 微调。
>
> 相机前处理、BEV 主干、导航文本编码是组里同学的基础模块，我消费他们的 `cond` 特征。

### B.2 整体 pipeline（按数据流讲，面试官最爱这个）

> 一次前向的数据流：
>
> **第一步，感知。** 8 路相机过共享 CNN/ViT backbone，投影到 BEV；同时 128 个可学习 object query 对 BEV 做 cross-attention，出来 `(B,128,256)`——每个 query 负责盯场景里一个潜在目标（车、人、骑行者，也可能空槽）。这 128 个特征再过 2-3 层「query 之间 self-attn + 查 BEV 的 cross-attn」，self-attn 那一步就是在建**交互图**（谁跟谁有博弈）。
>
> **第二步，风险场。** 对每个 query 过一个小 MLP 得到标量 logit，`softmax` 成 128 维的 α 权重——这就是 **α 软门控**：有的目标对当前风险贡献大（对向来车、加塞车），权重大。加权池化后过 `Linear` 展到 `32×32×8` 再 `sigmoid`，得到时空风险场；池化向量本身（256 维）就是 **`z_risk`**。空间 32×32 是 BEV 网格，时间维 8 是未来 8 个预测步的风险演化。
>
> **第三步，条件拼接。** `cond = [BEV 池化特征, 导航指令 embedding, ego 状态, z_risk]`，维度对齐后进生成器。
>
> **第四步，轨迹生成。** 用风险场算熵：熵高场景复杂，走 LTE 长尾专家；熵低走 Base。两路速度场按门控权重软混合。初始噪声不用纯 `N(0,I)`，而是 `noise_head(z_risk) + 0.1*randn`——**Risk-Init-Flow**，让采样起点就偏向当前风险允许的区域。然后 Euler 走 32 步，出 8 条候选。
>
> **第五步，打分与后训练。** 推理用 PDM 代理分选最优；训练后期上 Flow-GRPO，同场景采 24 条组内比优势，PPO-clip 更新 LoRA。

### B.3 核心算法 + 代码（风险场这段必细讲）

```python
# ========= 风险场: query -> alpha -> field -> z_risk =========
class RiskFieldNet(nn.Module):
    def __init__(self, n_agents=128, d=256, grid=(32, 32, 8)):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1)
        )                                    # 每个 agent -> 一个贡献 logit
        self.decode = nn.Linear(d, 32 * 32 * 8)
        self.grid = grid

    def forward(self, agents):               # agents: (B, 128, D)
        # 1) alpha 软门控: softmax 保证权重和为 1, 可解释为"谁占风险预算"
        alpha = torch.softmax(self.gate(agents).squeeze(-1), dim=-1)  # (B,128)

        # 2) 加权聚合 128 个目标 -> 全局风险向量
        pooled = (alpha.unsqueeze(-1) * agents).sum(dim=1)            # (B,D)

        # 3) 解码到时空网格, sigmoid 到 (0,1) 当风险强度
        field = self.decode(pooled).view(-1, 32, 32, 8)
        field = torch.sigmoid(field)                                   # (B,32,32,8)

        z_risk = pooled                                                # (B,D)
        return field, z_risk, alpha
```

```python
# ========= 双专家门控 + Risk-Init 初始噪声 =========
class DualExpertFlow(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.base = VelocityDiT(d)     # 常规专家
        self.lte  = VelocityDiT(d)     # Long-Tail Expert, 从 Base 热启动
        self.noise_head = nn.Linear(d, 30 * 2)   # z_risk -> 主噪声方向
        self.gate = nn.Linear(1, 2)              # 风险熵 -> 两专家权重

    def expert_mix(self, field):
        # 风险场熵: 分布越均匀(复杂)熵越大 -> 更信 LTE
        p = field.flatten(1)
        p = p / (p.sum(-1, keepdim=True) + 1e-8)
        h = -(p * (p + 1e-8).log()).sum(-1, keepdim=True)   # (B,1)
        w = torch.softmax(self.gate(h), dim=-1)              # (B,2)
        return w

    def sample_x0(self, z_risk):
        main = self.noise_head(z_risk).view(-1, 30, 2)
        return main + 0.1 * torch.randn_like(main)  # Risk-Init + 多样性

    @torch.no_grad()
    def generate(self, field, z_risk, cond, K=8, steps=32):
        w = self.expert_mix(field)
        outs = []
        for _ in range(K):
            x = self.sample_x0(z_risk)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((x.size(0),), i / steps, device=x.device)
                v = w[:, 0:1, None] * self.base(x, t, cond) \
                  + w[:, 1:2, None] * self.lte(x, t, cond)   # 软混合
                x = x + v * dt
            outs.append(x)
        return torch.stack(outs, dim=1)          # (B, K, 30, 2)
```

### B.4 为什么这样设计（三个「为什么」）

1. **为什么风险场而不是 occupancy？**
   Occupancy 只回答「有没有东西」，是几何；风险场回答「这里多危险、随时间怎么变、受谁影响」。旁车还没压线但速度矢量指向我们，occupancy 还是空的，风险场已经点亮——交互信息被前移到表征阶段，而不是生成完再碰运气。

2. **为什么双专家？**
   数据上常规 : 长尾 ≈ 8 : 2，单模型 MSE 被常规主导，梯度拉向保守平均。Base 拟合常规，LTE 在难例子集上加权，门控用**风险熵**（连续、可微）而不是人工 hard 标签，避免标错边界。

3. **为什么 Risk-Init？**
   标准 FM 起点在全空间各向同性噪声，积分要自己「学着绕开」高风险区。用 `z_risk` 映射初始分布，等于把先验放进 `p_0`，路径更短、早期违例更少。关键是训练时必须用**同一初始化分布**（条件 FM），否则 train/inference mismatch——这点面试可以主动点出来，显示踩过坑。

### B.5 效果与追问预案

**Q：0.8713 相对基线涨在哪？**
> 拆 PDMS 看，涨的几乎全在 **NC/DAC 的长尾子集**：基线在约 21% 高危交互上因碰撞/压线被乘积归零，风险场让模型「看见」这些交互并生成避让轨迹，把这批从 ~0 拉回 0.5+；常规 79% 我们没有掉点（comfort/EP 持平）。所以不是刷平均分，是修一票否决项。

**Q：128 query 够吗？怎么确定的？**
> 同屏强交互目标很少超过 50；128 是网格搜索 64/128/256 后的点——64 在密集路口漏目标，256 算力涨、收益平。128 还刚好构成 128 节点交互图，邻接矩阵形状整齐，方便做 top-k 稀疏 attention。

**Q：你和做感知的同学接口是什么？**
> 我拿三样：`(1) BEV 特征图 or 池化向量`、`(2) query 输出的 128 个 agent 特征`、`(3) 导航指令 token`。风险场、生成器、GRPO 都在我这边；感知的 loss 不反传进我的采样器（RL 阶段感知冻结，只训 LoRA 挂在生成器和风险头上）。

**Q：Flow-GRPO 为什么只 LoRA？**
> 三点：(1) 预训练速度场是通用生成先验，全参会灾难性遗忘，KL 也拉不回来；(2) RL 信号是标量奖励，信息量小于 SFT 文本，低秩假设（改动集中在少数方向）成立；(3) 0.5% 参数 ≈ 优化器状态小一个数量级，组大小 24×32 步的采样已经很贵，训练侧要省。`W' = W + (α/r)BA`，`B=0, A~N(0,σ)` 初始化保证起点就是原模型。

---

## 话术 C：世界模型 / 世界动作模型（TrustDrive-WAM）——「你们世界模型怎么做的？WAM 又是什么？」

### C.0 30 秒电梯版（区分两个词，很多人栽在这）

> 先分清：**世界模型**是经典定义——给定状态和动作预测下一状态 `p(s'|s,a)`，让我们能「想象」走了这条轨迹世界会怎样。**WAM（World Action Model）** 是我们项目的具体形态：不把「生成动作」和「预测后果」拆成两个模块，而是在**同一个流匹配过程**里联合演化轨迹、后果、风险、支持域四类隐变量——所以叫**联合动作-后果流**。外面再套一层**可信路由**：世界模型的预测不当真理、当「证据」，支持域分数不够就回退经典规划。NAVSIM **0.8012**。

### C.1 我负责什么

> 我负责 WAM 核心与安全闭环三件事：
> 1. **联合流**：把四类隐变量打包成一个向量做 Flow Matching，设计各分量的监督信号（轨迹用 L2，后果用 PDMS 分量，支持域用能量分数）；
> 2. **逆一致性与行为 token**：防止模型忽略动作输入；16 个行为模式 token 覆盖多峰驾驶风格；
> 3. **可信路由与 fallback**：双层信任评估（支持域 + 效用），在线决定信模型还是信经典规划器。
>
> 世界模型的主干时序编码（视频/轨迹 encoder）是共用的，我在它之上建「动作-后果联合解码头」。

### C.2 整体 pipeline（先讲清和传统方案的差别）

> **传统割裂做法**：先跑一个轨迹生成器吐 K 条，再跑一个独立评分网络事后打分，最后 argmax。问题是生成时不知道后果，评分和生成的表征不对齐，属于 predict-then-check。
>
> **我们的 WAM**：
> 1. 联合状态打包 `z = [traj(60) | conseq(32) | risk(16) | support(8)]`，共 116 维；
> 2. 和噪声 `z0` 直线插值得 `z_t`，一个联合速度场网络 `v_θ(z_t, t, c)` 输出 116 维速度，**同一时间步 t 上四条一起积分**；
> 3. 条件 `c` = 感知上下文 + 行为 token（枚举 16 个得 16 条不同风格候选）；
> 4. 积分中途每个中间点都能读出当前的 `support` logit 和 `risk`——**心跳式**给路由器用，不用等全程走完；
> 5. 路由：`layer1` 看支持域和风险（可靠性信任），`layer2` 看 WM 轨迹效用是否优于经典基线（决策信任），都过才用 WM，否则 `torch.where` 切到经典轨迹。
>
> 这样轨迹和后果的表征天然对齐（同 t），评分头不用跨模块对齐；代价是要设计好四条的 loss 权重和防坍缩。

### C.3 核心算法 + 代码

```python
# ========= 维度与打包 =========
TRAJ, CONQ, RISK, SUPP = 60, 32, 16, 8      # 60=30步*xy
JOINT = TRAJ + CONQ + RISK + SUPP           # 116

def pack(traj, conq, risk, supp):
    return torch.cat([traj, conq, risk, supp], dim=-1)

def unpack(z):
    return (z[..., :TRAJ],
            z[..., TRAJ:TRAJ+CONQ],
            z[..., TRAJ+CONQ:TRAJ+CONQ+RISK],
            z[..., TRAJ+CONQ+RISK:])

# ========= 联合速度场 =========
class JointVelocityNet(nn.Module):
    def __init__(self, d=512, cond_dim=256):
        super().__init__()
        self.inp  = nn.Linear(JOINT, d)
        self.cond = nn.Linear(cond_dim, d)
        self.mlp  = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.GELU()
        )
        self.out = nn.Linear(d, JOINT)

    def forward(self, z_t, t, cond, mode_tok):
        h = self.inp(z_t) + self.cond(cond + mode_tok) \
            + timestep_embedding(t, d)
        return self.out(self.mlp(h))         # 一次出 116 维速度

# ========= 联合 FM loss =========
def joint_fm_loss(net, traj1, conq1, risk1, supp1, cond, mode):
    z1 = pack(traj1, conq1, risk1, supp1)
    z0 = torch.randn_like(z1)
    t  = torch.rand(z1.size(0), device=z1.device)
    zt = (1 - t[:, None]) * z0 + t[:, None] * z1
    v  = net(zt, t, cond, mode)
    # 各分量可加权: 轨迹主任务权重大, 后果/风险中等, 支持域用能量蒸馏
    vz, vc, vr, vs = unpack(v - z1 + z0)     # 目标速度 z1-z0 也拆开
    loss = (w_t * (vz ** 2).mean()
          + w_c * (vc ** 2).mean()
          + w_r * (vr ** 2).mean()
          + w_s * (vs ** 2).mean())
    return loss
```

```python
# ========= 可信路由 =========
class TrustRouter:
    def __init__(self, tau_s=0.8, tau_r=0.4, tau_u=0.0):
        self.tau_s, self.tau_r, self.tau_u = tau_s, tau_r, tau_u

    @torch.no_grad()
    def route(self, supp_logit, risk, u_wm, u_base):
        p_s = torch.sigmoid(supp_logit)
        reliable = (p_s > self.tau_s) & (risk < self.tau_r)  # 层1 可靠性
        useful   = (u_wm - u_base) > self.tau_u               # 层2 决策信任
        return reliable & useful

    def select(self, wm_traj, base_traj, supp, risk, u_wm, u_base):
        use = self.route(supp, risk, u_wm, u_base)            # (B,)
        out = torch.where(use.view(-1, 1, 1), wm_traj, base_traj)
        return out, use.float().mean()   # 信任率, 直接当监控指标
```

```python
# ========= 逆一致性: 防止忽略动作 =========
def inverse_consistency(forward_dyn, inverse_dyn, s, a):
    s_next = forward_dyn(s, a)            # 前向: s,a -> s'
    s_hat  = inverse_dyn(s_next, a)       # 逆向: s',a -> s
    return F.mse_loss(s_hat, s)           # 必须闭环, 否则可忽略 a
```

**四条为什么必须同一条流**（面试金句）：
> 如果轨迹和后果各跑各的 ODE，同一「时刻」的 `(traj, conseq)` 语义对不齐，评分头会学到错位配对；共享 t 和联合速度场，代价分量始终贴着轨迹分量走，表征空间天然配对。

### C.4 「世界模型」vs「世界动作模型」——必答题的标准答案

> 世界模型是**类**，WAM 是**种**。类的经典形式只建模环境动力学 `p(s'|s,a)`，动作是外部输入；WAM 把动作的**生成**和动作的**后果**放进同一个联合分布/同一条流里学——动作不再只是条件，后果也不再是事后插件。对外表现有三点不同：
> 1. 生成中途就能读出后果和风险（可早停、可引导）；
> 2. 支持域和动作同步演化，路由不用等独立「验证网络」；
> 3. 逆一致性直接约束「动作信息确实进了表征」。
>
> 如果面试官问「和 Dreamer 那类世界模型比」：Dreamer 在 latent 里 roll 出 reward/value 再学策略，是 model-based RL 范式；我们是**开环评测 + 后训练**范式，WAM 更像「带后果头的条件生成器 + 安全路由器」，最终指标直接对齐 NAVSIM PDMS，不训 value 函数。

### C.5 效果与追问预案

**Q：可信阈值怎么定？上线怎么监控？**
> 验证集扫 `tau_s`，画「信任率 vs 碰撞率」曲线取拐点，部署取保守侧（高 `tau_s`）。线上监控三个数：信任率、fallback 触发率、fallback 段的 PDMS。故障注入（遮挡镜头、注入 OOD 车辆）看 fallback 召回率是否掉——这题答出来就是安全工程分。

**Q：如果 WM 和经典规划器都错呢？**
> 路由只保证「不确定时不信 WM」，不保证经典永远对。兜底再往下是安全层：RSS/可控域约束、最小风险策略（MRC）。我们项目做到「OOD 不盲信」，完整安全论证要接 HJ 可达性或 RSS——我会主动说这条边界，不吹「绝对安全」。

**Q：逆一致性会不会学成恒等映射？**
> 如果 forward 和 inverse 都学恒等，cycle loss 也低——所以 forward 必须受动力学监督（下一状态预测 loss），inverse 只作为正则；且 `a` 用的是真实执行动作不是自由变量。另外加 stopgrad，防止梯度把 forward 压成可逆玩具映射。

**Q：为什么 PDMS 比 RiskField-VLA 低 7 个点？**
> 三个原因：(1) 联合四头共享容量，轨迹专用容量相对少；(2) 可信路由保守，早期 `tau` 高导致常走 fallback，EP 被基线拖累；(3) 多峰 16 token 还在调，偶发选错风格。消融上单轨迹头能到 0.83+，说明瓶颈在联合与路由调参，不在范式。

---

## 话术 D：JEPA —— 「JEPA 你们具体怎么做的？」

### D.0 30 秒电梯版

> JEPA 是 **联合嵌入预测架构**：不重建像素，在**表征空间**预测被遮住/未来时刻的抽象向量。我们做的 **JEPA-DRIVE** 是驾驶场景的世界模型路线——三阶段：**Stage1** 用视频帧间 JEPA 自监督学「给定当前场景，下一刻表征长什么样」；**Stage2** 冻结世界模型，训一个隐空间评分头去对齐 NAVSIM 的 PDMS 排序；**Stage3** 解冻感知，感知-世界模型-生成器-评分器端到端调。全程**不需要轨迹人工标注之外的像素级监督**，基线 PDMS **0.7285**。

### D.1 我负责什么

> 我负责 **JEPA 主干与三阶段训练框架**：mask 策略与帧间目标构造、Target Encoder 的 EMA 更新、predictor 结构、VICReg/Cycle-Energy 防坍缩与能量到支持域的映射、以及 Stage2/3 的评分头与联合 loss 权重。65536 条密集轨迹的**因子化生成**（path×speed）和排序选择也是我写的；感知 backbone 冻结/解冻策略和组内对齐。

### D.2 整体 pipeline（三阶段一条线讲完）

> **Stage 1 —— 自监督预训练（学世界模型）。**
> 取连续视频帧 `x_t` 和 `x_{t+1}`。对 `x_t` 挖 75% 的 patch 做可见上下文，**目标不是重建 x_t 的像素**，而是 `x_{t+1}` 被 mask 区域的 **Target Encoder 输出**。Context Encoder 有梯度，Target Encoder 是 Context 的 EMA 副本 + `no_grad`。Predictor 吃上下文表征 + 位置/时间条件，预测目标表征，loss 是特征空间 MSE。这一步学到的是「场景动力学的抽象」——车流怎么演化、遮挡关系怎么变，而不是树叶纹理。
>
> **Stage 2 —— 冻结 WM，训评分头（对齐任务）。**
> 把 JEPA 冻死，特征出来喂评分头，输入是候选轨迹的特征拼场景特征，输出 PDMS 分数；loss 用 **listwise 排序 loss** 不是 MSE——因为 PDMS 是乘积指标，官方也只 care 排序，回归绝对值会把尾部碰撞样本的权重带偏。
>
> **Stage 3 —— 端到端联合。**
> 解冻感知分支，全链 `感知 -> 世界模型 -> 轨迹生成 -> 评分` 一起调，loss = 排序 loss + `0.1 * JEPA 辅助`（别丢掉动力学）+ `0.05 * VICReg`（防联合训练时坍缩）。生成端用 path×speed 因子化一次出 65536 条，评分头在**隐空间**批量打，不需要生成像素。

### D.3 核心算法 + 代码

```python
# ========= JEPA 前向 =========
class JEPA(nn.Module):
    def __init__(self, d=768, m=0.996):
        super().__init__()
        self.ctx = ViT(patch=16, dim=d)          # 可训练
        self.tgt = copy.deepcopy(self.ctx)       # EMA, requires_grad=False
        for p in self.tgt.parameters():
            p.requires_grad = False
        self.pred = nn.Sequential(               # 上下文+位置 -> 目标表征
            nn.Linear(d + 128, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, d),
        )
        self.m = m

    def forward(self, x_ctx_patches, x_tgt_patches, pos):
        z_ctx = self.ctx(x_ctx_patches)                 # 梯度到 ctx
        with torch.no_grad():                           # 关键: stop-grad
            z_tgt = self.tgt(x_tgt_patches)             # 只 EMA 更新
        z_pool = z_ctx.mean(dim=0, keepdim=True) \
                     .expand(z_tgt.size(0), -1)
        z_hat = self.pred(torch.cat([z_pool, pos], -1))
        return F.mse_loss(z_hat, z_tgt)                 # 特征空间 MSE

    @torch.no_grad()
    def ema_update(self):
        for pt, ps in zip(self.tgt.parameters(), self.ctx.parameters()):
            pt.data.lerp_(ps.data, 1.0 - self.m)        # EMA
```

```python
# ========= Stage 1: 帧间世界模型 JEPA =========
def stage1_train(jepa, loader, opt):
    for videos in loader:                               # (B,T,3,H,W)
        t = torch.randint(0, videos.size(1) - 1, (1,)).item()
        x_now, x_next = videos[:, t], videos[:, t + 1]
        mask = make_block_mask(B, N, ratio=0.75)        # True=目标块

        # 上下文来自当前帧可见块, 目标来自下一帧被遮块 (预测未来)
        loss = jepa.temporal_loss(x_now, x_next, mask)
        loss.backward(); opt.step()
        jepa.ema_update()

# ========= Stage 2: 冻结 + 排序评分头 =========
def stage2_train(jepa, scorer, loader, opt_s):
    freeze(jepa)
    for batch in loader:
        with torch.no_grad():
            z = jepa.ctx(batch['ctx_patches'])          # (B,D)
        pred = scorer(z, batch['cand_trajs'])           # (B,K) 分数
        loss = listwise_rank_loss(pred, batch['pdms'])  # 对齐官方序
        loss.backward(); opt_s.step()

def listwise_rank_loss(pred, gt):
    """软排序: 高分轨迹应有更高 gt 的概率 (softmax 对齐)"""
    p = F.softmax(pred / 0.1, dim=-1)
    q = F.softmax(gt  / 0.1, dim=-1)
    return F.kl_div(p.log(), q, reduction='batchmean')
```

```python
# ========= Cycle-Energy: 双向 + 能量给支持域 =========
def cycle_energy(jepa, xa, xb, mask_b, mask_a):
    e_fwd = jepa(xa, xb, mask_b)     # A 看可见, 预测 B 的 mask 处
    e_bwd = jepa(xb, xa, mask_a)     # 反向
    energy = e_fwd + e_bwd           # 样本能量
    support = torch.sigmoid(2.0 - 4.0 * energy)  # 能量低->支持域内(示意标定)
    return energy, support
```

```python
# ========= Stage 3: 端到端 =========
def stage3_train(full, loader, opt):
    unfreeze(full.perception)
    for batch in loader:
        z = full.perception(batch['imgs'])
        z_fut = full.world_model(z)                   # 未来隐表征
        trajs = full.generator(z_fut)                 # 或 path*speed 因子化
        scores = full.scorer(trajs, z_fut)
        loss = (listwise_rank_loss(scores, batch['pdms'])
                + 0.1 * full.jepa_aux(batch)
                + 0.05 * vicreg_loss(z))
        loss.backward(); opt.step()
```

```python
# ========= 65536 密集轨迹: path x speed 因子化 =========
def dense_paths_speeds(paths, speeds):
    """
    paths: (B,256,L,2)  空间形状
    speeds:(B,256,L,1)  时间分配/速度系数
    -> (B, 65536, L, 2)  256*256 组合
    注意: 实现里 scorer 可 chunk 评分别物化全量显存
    """
    return (paths.unsqueeze(2) * speeds.unsqueeze(1)) \
            .flatten(1, 2)
```

### D.4 为什么 JEPA（必考对比）

**vs MAE/像素重建：**
> MAE 要把遮住的每个像素画回来，算力花在纹理、光照这些对决策无用的自由度上；JEPA 只要求「预测对抽象」，一个向量回归搞定。用考试类比：MAE 是把被挡的图重画一遍，JEPA 是回答「被挡的是什么」。

**vs 扩散式像素世界模型（DriveDreamer 那类）：**
> 三点：算力（不生成像素）、物理一致性（像素生成会画出穿模，表征可注入几何先验）、接口（表征直接进规划头，不用先解码再读语义）。代价是表征没有显似然，更依赖 EMA+stopgrad+VICReg 三道防坍缩，以及 Stage2 这种任务对齐头。

**I-JEPA vs V-JEPA：**
> I-JEPA 单图挖块，补的是**空间上的现在**；V-JEPA 看前几帧预测下一帧表征，补的是**时间上的未来**——只有 V-JEPA 算世界模型。我们 Stage1 是帧间目标，属于 V-JEPA 路线。

### D.5 防坍缩（ JEPA 必问）

> 表征坍缩 = 所有输入映到同一点，loss=0 但信息为 0。机理是 Context 和 Target 一起被同一个 loss 拉，可以「串通」输出常数。三道防线：
> 1. **stop-gradient**：Target 分支 `no_grad`，不对答案；
> 2. **EMA（0.996）**：Target 慢半拍，瞬时坍缩被时间常数拖住，Context 必须预测「昨天的自己」才准；
> 3. **VICReg**：方差项罚 `std<1`，协方差项罚维度间相关——几何上禁止点云塌缩。
>
> 类比：EMA+stopgrad 是「老师慢半拍且不许对答案」，VICReg 是「直接规定不许缩成一个点」。我们三个都开了，Stage3 还保留 0.05 的 VICReg 权重防联合训练时被排序 loss 带崩。

### D.6 效果与追问预案

**Q：0.7285 比另两个低，JEPA 是不是不行？**
> 先对齐口径：这是**无标注自监督 + 未做满 GRPO** 的基线，另两项目是监督/后训练打满的。0.7285 证明范式闭环可行；消融里去掉 Stage2 排序对齐只有 0.68，说明瓶颈在任务对齐不在 JEPA 表征本身。计划里 Stage2 换 listwise + 引入 Flow-GRPO，内部复现预期能过 0.78。

**Q：评分头为什么排序 loss 不用 MSE？**
> PDMS 是乘积且官方评测是相对排序；MSE 会被「一堆 0.9 分的常规样本」主导梯度，长尾 0 分样本之间怎么排学不到。listwise KL 让整组分布对齐，和 GRPO 组内思想同源——面试可以把这两个项目串起来说，显示方法论统一。

**Q：mask 比例多少？为什么 75%？**
> 75% 是 MAE/JEPA 常用甜点：遮太少，任务太简单（局部插值就能骗过 loss）；遮太多，上下文没信息，predictor 学到先验平均。驾驶视频我还试过 block mask——遮整块车道比随机 patch 难，更逼出「车流结构」而不是「纹理补全」。

**Q：Target Encoder 要进 optimizer 吗？**
> 绝对不进。只通过 EMA 从 Context 拷贝权重。如果 Target 也有 Adam 状态，等于两边一起被 loss 拽，坍缩防线失效。代码里 `requires_grad_(False)` 且不 add 到 optimizer param group——这是代码审查常抓的点。

**Q：和 LeCun 说的 JEPA 是一回事吗？**
> 是同一条线：预测嵌入空间、用 stopgrad/EMA 防坍缩、I-JEPA 补空间 V-JEPA 补时间。我们加的是**驾驶特有的东西**：帧间目标当世界模型、能量当支持域接可信路由、排序头对齐 PDMS。可以说「架构是 LeCun 的，任务封装和安全接口是我们的」。

---

## 话术 D+：比亚迪 Cosmos-3 MOT 世界模型 —— 「你在比亚迪的 MOT 世界模型具体怎么做的？」

> 面试官在比亚迪这段经历上最爱追问的就是 **MOT（Mixture-of-Transformers）世界模型架构**。本节给一份可直接口述的第一人称完整讲法：先 30 秒接住问题，再讲我负责什么、整条 pipeline、核心算法与代码、为什么这样设计，最后是追问预案。括号内是可略过的深水细节，对方感兴趣再展开。

### D+.0 30 秒电梯版（先接住问题）

> 在比亚迪我做的是基于 **NVIDIA Cosmos-3 双塔 MoT** 改造的**车载世界模型**。一句话说清楚架构：**MoT 把 token 分成理解（Reasoner / und）和生成（Generator / gen）两条路径——同一层 Transformer 里两套独立的 QKV/FFN 权重，靠 `moe_gen_mask` 分发；理解侧做因果自回归读场景和指令，生成侧做双向注意力 + Rectified Flow 出连续轨迹/视频 latent；gen 每层还能 cross-attend und 的语义，且 `detach` 防止生成噪声污染理解塔。** 我在这条链路上负责 **MoT 双路径注意力与序列打包的工程落地、DCAE 视频 token 化接口、以及 Generator 侧连续轨迹 Flow Head 的接入与训练**，目标是让车端能「读懂场景 + 想象下一步 + 出连续轨迹」在一个模型里闭环。

### D+.1 我负责什么（划清 ownership，面试必问）

> 我负责的是 **Cosmos-3 骨干之上的「MOT 落地 + 生成侧动作接口」**，具体三块：
>
> 1. **MoT 双塔前向与序列打包**：实现/改通 `PackedAttentionMoT` / `MoTDecoderLayer` 的 und/gen 分发——`moe_gen_mask` 怎么从 packing 阶段生成、causal vs full attention 怎么在同一条 packed 序列上共存、gen→und 的 cross-attn 梯度要不要 `detach`；对齐 NVIDIA `cosmos-framework` 源码语义并做车载数据管线适配（多相机 token、自车状态 token 拼进 und 侧）。
> 2. **DCAE / tokenizer 接口**：把环视多路视频压进 DCAE latent（时间 /4、空间 /32 的因果 VAE），保证编码器**只看过去不偷看未来**；打通「视频 → latent → MoT → latent → 解码/评」的数据通路，以及驾驶场景下 latent 通道与分辨率的配置选型。
> 3. **Generator 侧连续轨迹 Flow Head**：在 gen token hidden 上挂 Flow Head，用 Rectified Flow（就是直线速度场那套）一次生成完整轨迹块 `A_t = {a_{t+1},...,a_{t+T}}`；后面可选接 Flow-GRPO 做组内相对优势微调。Reasoner 侧通常冻结或极小 lr，避免 RL/动作监督把多模态理解打坏。
>
> 骨干权重、Tokenizer 预训练、集群调度不是我这边的 ownership——我消费他们交付的 checkpoint 和 latent 接口。

### D+.2 整体 pipeline（按数据流讲，面试官最爱听这个）

> 一次「多模态输入 → 世界模型 → 轨迹/未来」的完整数据流，我按顺序讲：
>
> **第一步，输入 token 化。** 四类输入：视觉观测（环视/BEV 相关帧）、语言指令（导航/任务）、自车状态、场景目标。文本走 BPE；图像/视频走 DCAE 空间/时空因果 VAE；状态走连续 embedding（低维连续值先 Fourier/MLP 升维再进序列——直接丢 (x,y,yaw) 表达能力不够）。全部映射到同一 hidden space，但**按模态保留各自统计特性**——这是「同一表征空间、不同编码策略」。
>
> **第二步，序列打包（Sequence Packing）。** 把 und token 和 gen token 拼成**一条**序列，并打上布尔 mask：哪些位置是理解路径，哪些是生成路径。und 段在前（或按 packing 策略交错），gen 段（噪声 latent / 动作 token）在后。`moe_gen_mask` 形状 `(B*L,)`，**整个前向不变**——这条 mask 是 MoT 能否跑起来的工程关键，pack 错了会静默训崩。
>
> **第三步，MoT 双塔前向（架构心脏）。** 每一层 `MoTDecoderLayer`：
> - 用 mask 把 hidden 拆成 `und_h` 和 `gen_h`；
> - **各走各的 LayerNorm**：`input_layernorm` vs `input_layernorm_moe_gen`，`torch.where(mask, gen_normed, und_normed)` 选出来；
> - **各走各的 QKV/O**：理解侧 `q/k/v/o_proj`，生成侧 `q/k/v/o_proj_moe_gen`，权重独立、初始化独立、优化器里也是两组参数；
> - **注意力模式不同**：und = **causal**（只能看前面，适合 next-token 理解）；gen = **full bidirectional**（噪声轨迹/视频 token 互相全看，适合流匹配/扩散去噪）；
> - **gen→und cross-attention**：生成侧用 gen 的 Q 去查 und 的 K/V，相当于「动作专家回头看语义方案」；实现里对 `und_h.detach()`，**梯度不反传进理解塔**（和 π₀ 的 `no_grad` 同思想，粒度更细——只断 cross 分支，und 自己的 CE 梯度还在）；
> - `gen_out = self_attn(gen) + cross_attn(gen, und)` 再过 gen 侧 FFN；und 过 und 侧 FFN；残差汇回同一个 `hidden_states` 张量。
>
> **第四步，双头出口。** 理解 token 出 `lm_head` 做 next-token CE；生成 token 出 **diffusion/Flow head** 预测向量场（或轨迹速度）。训练 loss = `CE(und) + λ * MSE(gen_rectified_flow)`。
>
> **第五步，生成与闭环推理。** gen 侧从 `A0 ~ N(0, I)` 出发，Rectified Flow 多步积分（或 UniPC 调度）得到轨迹块/视频 latent；**只执行前 K 步**，拿新观测重新打包、重新 Reasoner、重新生成——滚动闭环，而不是一次吐全剧本站死。
>
> **第六步（可选后训练）。** Flow-GRPO：同场景采 G 条轨迹，组内 advantage，PPO-clip 更新 **Flow Head + Generator Router（若挂了 MoE）+ 被激活 gen experts**；Reasoner 冻结或保守微调。

### D+.3 核心算法 + 代码（MOT 这段面试可直接对着讲）

```python
# ========= MoT 核心: 一份输入, 两套权重, mask 分发 =========
class PackedAttentionMoT(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads):
        super().__init__()
        # --- und 路径 (Reasoner, causal) ---
        self.q_proj = nn.Linear(hidden, n_heads * head_dim)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim)  # GQA
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # --- gen 路径 (Generator, full attn) 权重完全独立 ---
        self.q_proj_gen = nn.Linear(hidden, n_heads * head_dim)
        self.k_proj_gen = nn.Linear(hidden, n_kv_heads * head_dim)
        self.v_proj_gen = nn.Linear(hidden, n_kv_heads * head_dim)
        self.o_proj_gen = nn.Linear(n_heads * head_dim, hidden)
        self.q_norm_gen = RMSNorm(head_dim)
        self.k_norm_gen = RMSNorm(head_dim)
        # 专供 gen->und cross-attn 的 K norm (统计量可能和 und 自用不同)
        self.k_norm_und_for_gen = RMSNorm(head_dim)

    def forward(self, h, moe_gen_mask):
        """
        h: (B, L, D)  打包后的整段序列
        moe_gen_mask: (B, L) True = gen 路径, False = und 路径
        """
        und_mask = ~moe_gen_mask
        und_h = h[und_mask]          # (N_und, D)
        gen_h = h[moe_gen_mask]      # (N_gen, D)

        # 1) und: causal self-attn  (只看左边, 理解/自回归)
        uq = self.q_norm(self.q_proj(und_h))
        uk = self.k_norm(self.k_proj(und_h))
        uv = self.v_proj(und_h)
        und_out = sdpa(uq, uk, uv, is_causal=True)

        # 2) gen: full self-attn  (全双向, 流/扩散去噪要互相看)
        gq = self.q_norm_gen(self.q_proj_gen(gen_h))
        gk = self.k_norm_gen(self.k_proj_gen(gen_h))
        gv = self.v_proj_gen(gen_h)
        gen_self = sdpa(gq, gk, gv, is_causal=False)

        # 3) gen -> und cross-attn: 动作/视频查语义
        #    detach: 生成噪声的梯度不污染 Reasoner
        uk_for_gen = self.k_norm_und_for_gen(self.k_proj(und_h.detach()))
        uv_for_gen = uv.detach()
        gen_cross = sdpa(gq, uk_for_gen, uv_for_gen, is_causal=False)

        gen_out = self.o_proj_gen(gen_self + gen_cross)
        und_out = self.o_proj(und_out)

        out = torch.zeros_like(h)
        out[und_mask] = und_out
        out[moe_gen_mask] = gen_out
        return out


class MoTDecoderLayer(nn.Module):
    """一层 = 双 LN + 双路径 attn + 双 LN + 双 FFN, 残差共享同一 hidden"""
    def forward(self, h, moe_gen_mask):
        resid = h
        h_n = torch.where(
            moe_gen_mask.unsqueeze(-1),
            self.ln_gen(h),          # gen 专属 norm
            self.ln_und(h),          # und 专属 norm
        )
        h = resid + self.attn(h_n, moe_gen_mask)

        resid = h
        h_n = torch.where(
            moe_gen_mask.unsqueeze(-1),
            self.post_ln_gen(h),
            self.post_ln_und(h),
        )
        ff = torch.where(
            moe_gen_mask.unsqueeze(-1),
            self.mlp_gen(h_n),       # 生成侧 FFN
            self.mlp_und(h_n),       # 理解侧 FFN
        )
        return resid + ff
```

**双头 + 双 loss（总装层概念）：**

```python
class OmniMoTForCausalLM(nn.Module):
    def forward(self, input_ids, moe_gen_mask, labels=None, flow_target=None):
        hidden = self.backbone(input_ids, moe_gen_mask)   # (B,L,D)

        # 理解头: next-token CE  (Reasoner)
        und_logits = self.lm_head(hidden[~moe_gen_mask])
        loss_und = cross_entropy(und_logits, labels)

        # 生成头: Rectified Flow 速度场 MSE  (Generator)
        # flow_target = x1 - x0, 网络在 x_t=(1-t)x0+t*x1 上回归
        gen_h = hidden[moe_gen_mask]
        v_pred = self.flow_head(gen_h, t)                 # 向量场
        loss_gen = F.mse_loss(v_pred, flow_target)

        return loss_und + self.lambda_gen * loss_gen
```

**序列打包与 mask（工程最常问）：**

```python
def pack_und_gen(und_tokens, gen_tokens):
    """
    und: 文本/状态/指令  -> False
    gen: 噪声 latent / 动作 token -> True
    拼成一条序列, 返回 ids + moe_gen_mask + attention 布局
    """
    input_ids = torch.cat([und_tokens, gen_tokens], dim=1)          # (B, Lu+Lg)
    moe_gen_mask = torch.cat([
        torch.zeros_like(und_tokens, dtype=torch.bool),
        torch.ones_like(gen_tokens, dtype=torch.bool),
    ], dim=1)                                                       # (B, Lu+Lg)
    # und 段: causal mask; gen 段: full mask; 另有 gen 可看 und 的 cross 布局
    attn_mask = build_mot_attention_mask(
        len_und=und_tokens.size(1),
        len_gen=gen_tokens.size(1),
        und_causal=True,
        gen_bidirectional=True,
        gen_attend_und=True,
    )
    return input_ids, moe_gen_mask, attn_mask
```

**DCAE 角口（为什么先压缩再进 MoT）：**

```python
# 视频 (B,T,H,W,3) -> latent (B, C, T//4, H//32, W//32)
# 因果: 编码第 t 帧时 padding 只允许用 <=t 的帧, 不许偷看未来
# 否则世界模型「预测未来」时 tokenizer 已经泄题, 评测全虚高
latent = dc_ae_encoder.encode(video, causal=True)   # 训练/推理同构
# MoT gen 路径吃的就是这个 latent 展平后的 token
```

**Rectified Flow 与前面话术 A 的关系（主动串起来，显示体系化）：**

```text
x0 ~ N(0,I),  x1 = 真实轨迹/latent
x_t = (1-t) x0 + t x1
v_target = x1 - x0
loss_gen = || flow_head(gen_hidden, t) - v_target ||^2
推理: 从噪声 Euler/UniPC 积分到 t=1 -> 轨迹块或视频
```

> 我会主动说：**话术 A 里的 Flow Matching 就是我们 Generator 头的生成范式**；Cosmos-3 里叫 Rectified Flow，数学上是同一族直线流。这样面试官会感觉你把「比亚迪 MOT」和「后面项目 FM/GRPO」串成一条技术线，而不是三段无关经历。

### D+.4 为什么这样设计（四个「为什么」，比 MoE/MHA/双塔对比）

**Q：为什么 MoT 双塔，而不是「一个大稠密 Transformer 硬吃所有 token」？**

> 五模态统计特性差太远：语言离散稠密语义、视频连续强空间冗余、动作连续低维强因果。硬塞进一套 QKV/FFN 会互相稀释——语言梯度把感知带跑，或动作 token 被视频 token 淹没。MoT 的哲学是 **该分开的分开（每层两套投影 + 两套 FFN + 两套 LN），该共享的共享（同一个 hidden 张量、残差、层深、打包序列）**。和稠密单塔比：模态冲突低、可按需只激活一塔、加新模态不用重训整个骨架。

**Q：MoT 和 MoE 什么区别？（必被问，很多人混）**

> **粒度不同，是正交的两层路由：**
> - **MoT**：决定 token 走 **Reasoner 还是 Generator**——模态/角色级，两条路径，mask 在 packing 时就定了，训练中基本不变；
> - **MoE**：决定 token 进了某一塔之后，**由哪些专家 FFN 处理**——expert 级，router 每 step 学 top-k，共享专家 + 稀疏路由专家（DeepSeek-style）。
>
> 用公司打比方：MoT 是「进研发部还是产品部」，MoE 是「进了研发部后进哪几个项目组」。我们在方案里可以 **MoT 底座 + 每塔内部再加 Sparse MoE**——Reasoner 专家池偏空间关系/语言指令/对象交互，Generator 专家池偏轨迹结构/时间一致性/连续运动。参数总量上去，每个 token 激活量不上去。

**Q：为什么 und causal、gen bidirectional？**

> und 是理解/自回归任务——读指令、读历史观测，本质是 next-token，causal 才和 LM 预训练目标对齐，也才方便 KV cache 推理。gen 是流匹配/扩散——去噪时整段轨迹/整个 latent 需要互相一致（第 10 步刹车和第 1 步方向盘要同一驾驶意图），必须 full attention。硬统一成一种 mask，要么理解泄漏未来（作弊），要么生成互相看不见（轨迹前后打架）。

**Q：为什么 gen→und cross-attn 还要 `detach`？**

> 两个原因：一是 **保护 Reasoner**——轨迹/视频噪声很大，若梯度自由打回理解塔，CE 预训练能力会被生成任务的杂梯度冲垮（π₀ 冻 VLM 同理）；二是 **训练信号纯度**——und 的梯度只来自 next-token，gen 的梯度只来自 Flow MSE，边界清晰、日志好查。若要端到端联调，可以给 cross 分支一个带 scaler 的小梯度，但默认车载方案是 detach + Reasoner 冻结。

**Q：DCAE 为什么必须因果？**

> 世界模型的卖点是**预测未来**。若 encoder 编第 t 帧时用了第 t+1 帧，等于 tokenizer 先看了答案，后面 Flow 再「预测」就是泄题——开环分数虚高，上车必崩。因果卷积 padding 是把「不许偷看未来」焊进结构里，而不只是靠数据管线自律。

**Q：为什么轨迹不自回归吐 token，而用 Flow 一次出块？**

> 自回归离散动作：步间误差累积、时序连续性靠 token 化硬拼、难保持整条轨迹平滑。Flow 一次生成 `T` 步轨迹块：块内联合建模连续性，天然多峰（换初始噪声），部署时 **execute first K steps → 新观测再生成**（receding horizon），比「生成全部再执行」闭环得多。和 π₀ / Diffusion Planner 的动机一致。

### D+.5 和面试官可能的深挖链（第一人称短答）

**链 1：MoT 细节**
1. mask 从哪来？→ packing 阶段烧进 `moe_gen_mask`，前向只读不改。
2. 两套权重怎么优化？→ 两个 param group；可 und/gen 不同 lr（und 更小）。
3. GQA 为什么 KV head 少？→ 长序列 KV cache 省显存，车载多相机序列长，收益大。
4. QK RMSNorm 干嘛？→ 稳 attention logits，防爆——DeepSeek 验证过的技巧。

**链 2：和 π₀ / 单塔 VLA 对比**
1. 和 π₀ 差在哪？→ π₀ 是「冻结 VLM + 独立 300M 动作专家 cross-attn 读 KV cache」，动作专家是外挂子网络；MoT 是**每层双投影双 FFN 的一体骨架**，und/gen 深度交织，不是末端挂头。
2. 和「单塔多模态 LLM 直接回归动作」比？→ 单塔无专用生成注意力/无流头，连续动作质量和多峰性差；MoT 把生成注意力模式和 LM 理解模式真正分开。

**链 3：训练与部署**
1. Stage 怎么排？→ (1) 继承 Cosmos-3 预训练；(2) 若加 MoE：dense FFN 换 shared+routed，router 预热；(3) SFT 式 Flow 轨迹监督；(4) 可选 Flow-GRPO。
2. 车端跑得动吗？→ Nano 级 backbone（十几 B）+ 只激活 gen 头出轨迹时可以；完整视频世界模型在云端做数据/仿真，车端可蒸馏到更小 Flow 策略——分工是「云端想象、车端执行」。
3. 推理几条流步？→ 轨迹块 8~32 步量级；视频生成侧可用 UniPC 加速。轨迹短、维低，步数可压。

**链 4：你在比亚迪具体碰过哪些坑？（务必准备 2 个真细节）**

> 准备稿 A——**pack 与 attention mask 不一致**：`moe_gen_mask` 和 causal/full 布局差一个位置，und 看到了 gen 噪声，表现为 CE 不降、Flow 却在降。排查手段：打印每层 und 位置的 attention 熵、可视化 mask 矩阵、单 batch overfit 回归。
>
> 准备稿 B——**cross-attn 忘了 detach（或误 detach 了 und 自己的 CE 路径）**：要么 Reasoner 梯度爆炸被 Flow loss 主导，要么 und 完全学不动。修复：只在 `k_proj(und_h.detach())` / `v.detach()` 的 cross 分支切断；用 `grad_norm` 分桶监控两塔。
>
> 准备稿 C——**DCAE 非因果/训练推理增强不一致**：开环视频指标好、闭环一跑就漂。修复：encoder 全因果 padding；训练/推理同一套 temporal jitter；用「遮未来帧」的单元测试断言 latent 不依赖未来。

### D+.6 效果与边界（诚实版，避免吹穿帮）

> **我这样收尾（若被问指标/结果）：**
> 架构侧我交付的是 MOT 前向、pack、Flow Head 接入和可复现的训练配置；指标上关注三类：理解侧任务 CE/下游理解分不掉点（证明没伤 Reasoner）、生成侧轨迹/仿真指标相对单塔或外挂专家基线的增益、以及推理延迟与显存（双路径分发的 overhead）。完整量产指标属于部门口径，我讲我模块内的消融：**有无 gen→und cross、有无双套 LN/FFN、detach 开关** 对生成质量与理解分的 trade-off。
>
> **主动划边界：** Cosmos-3 是我们的底座（NVIDIA 开源权重/框架），我们的贡献是**车载场景下的 MOT 工程化、多传感器 token 接入、连续轨迹生成头与训练闭环**——不是从零发明 MoT。面试里说清「底座 vs 我的改造」比含糊地说「我设计了 Cosmos」更可信，也防住「那你讲讲原论文细节」的偷袭。

### D+.7 90 秒连贯口述稿（可直接背）

> 「比亚迪这段我做 Cosmos-3 世界模型的 MOT 落地。MoT 的核心是：同一层 Transformer 里准备 und/gen 两套 QKV、O、FFN、LayerNorm，用序列打包阶段生成的 `moe_gen_mask` 把 token 分发到两条路径。Reasoner 走 causal attention，做 next-token 理解；Generator 走 full attention，做 Rectified Flow 连续生成。每层 gen 还会 cross-attend und 的语义，这条分支上对 und 做 detach，避免生成梯度打坏预训练理解。视频先用因果 DCAE 压到 latent，时间除 4、空间除 32，保证编码不偷看未来。我负责 pack/mask 与双路径前向的正确性、多传感器 token 接入，以及 gen 头上的轨迹 Flow Head——输入场景和指令，一次生成未来 T 步轨迹块，执行前 K 步再滚动重规划。和后面项目的联系是：Flow Head 用的就是直线流匹配；若上 RL，就是同一套 Flow-GRPO。和 π₀ 比，我们不是外挂动作专家，而是每层双塔交织。我踩过的坑主要是 pack 和 attention mask 不一致、以及 cross-attn 的 detach 边界——都有单测和分桶 grad_norm 监控兜住。」

## 话术 E：「你负责什么？」总述版（开场 1 分钟，三项目通用模板）

> 先补充一段比亚迪经历（很多面试官会先问这段）：
>
> **在比亚迪我做基于 Cosmos-3 的 MOT 双塔世界模型落地**——Reasoner/Generator 每层两套投影，理解用因果 LM，生成用 Rectified Flow 出连续轨迹，视频走因果 DCAE，gen cross-attn 读语义且 detach。这段让我把「多模态理解 + 连续生成」的底座架构打穿了，后面三个项目都是这条线上的深挖。
>
> 然后是我主责的三个递进项目，都围绕**端到端自动驾驶的「感知-预测-生成-评估」闭环**：
>
> **第一个项目 RiskField-VLA**，我负责风险建模和轨迹生成后训练——从 128 个 object query 聚出时空风险场，条件化 Flow Matching 双专家生成轨迹，再用 Flow-GRPO（SDE 化 + PPO-clip + LoRA）对齐 NAVSIM PDMS，最后 **0.8713**。
>
> **第二个项目 TrustDrive-WAM**，我负责把「生成动作」和「预测后果」合成一条联合流，外加可信路由——世界模型预测当证据不当真理，支持域不够就回退经典规划，**0.8012**。
>
> **第三个项目 JEPA-DRIVE**，我负责 JEPA 世界模型的三阶段训练——自监督帧间表征预测、冻结后排序评分头、端到端联合，走无像素重建路线，**0.7285**。
>
> 横向看，我擅长的是**连续生成模型（Flow Matching）+ 强化学习后训练（GRPO）+ 安全可信接口（风险场/支持域）** 这三块的落地，而不是只会调检测框。

**使用提示**：面试官问「你负责什么」时，先 1 分钟总述，再让对方挑一个项目深入——然后切到对应话术 A/B/C/D。切忌三个项目同时展开讲细节。

---

## 话术 F：Flow Matching 与 Flow-GRPO 如何串讲（被要求「讲一个你最有深度的算法」时）

> 我讲 Flow-GRPO 吧，因为它跨了生成模型和 RL 两边，坑最多。
>
> **问题设定**：我们已有 Flow Matching 轨迹生成器，SFT 后 PDMS 卡住，想用 RL 拉长尾分数。GRPO 的框架是现成的——同场景采一组，组内归一化 advantage，PPO-clip 更新。
>
> **第一个坎：log-prob 算不了。** LLM 里 `log π(a)` 直接 `log_softmax`。Flow 推理是确定性 ODE，`x_{t+1}` 是 `x_t` 的函数，条件分布是 Dirac，`log p = -inf`，ratio 爆炸。解法是**把采样改成 SDE**：漂移里加 score 补偿项保证边际分布和原 FM 一致，再 Euler-Maruyama 离散化，每步转移变成 `N(mean, σ²I)`，于是
> `log p = -||x'-mean||²/(2σ²) - D/2 log(2πσ²)` 闭式可算。score 又可以用速度场和直线路径的恒等式换成 `v_θ`，整条链可微。
>
> **第二个坎：PPO 的 off-policy。** 采样时记录 `log_prob_old`，训练时不重新 rollout（贵且方差大），而是拿旧轨迹的状态序列，用**新模型**重算每步 `log_prob_new`，`ratio = exp(new - old)`，clip 到 `[0.8, 1.2]`，再加对 ref model（关 LoRA 的同一 base）的 KL。梯度从 ratio 经高斯 log 概率进 mean，再进 `v_θ`，只落在 LoRA 的 A/B 上——`W' = W + (α/r)BA`，0.5% 参数。
>
> **第三个坑：组内 std≈0。** 常规场景 24 条分都接近满分，advantage 全 0，白采样。所以加**难例挖掘**：按历史失败率重采样长尾场景，保证每组都有区分度。
>
> **结果**：长尾子集 NC/DAC 从归零拉回正区间，总 PDMS 到 0.8713；训练监控看三个数——组内 reward std、policy KL、LoRA 输出范数。KL 突然掉、reward 单边涨，基本是 reward hacking，要查打分器漏洞。
>
> （如果对方点头，再展开 SDE 四步推导或 PPO-clip 代码；如果对方打断，停在「确定性 ODE 没有 log-prob，所以 SDE 化」这句核心上。）

---

# 第三部分：面试压力面问题与参考回答

> 覆盖概念、原理、项目细节、开放题。每题给「标准回答」和「加分回答」，部分附「追问链」。

---

## A. Flow Matching 相关

**Q1：Flow Matching 和 DDPM 有什么区别？为什么你们选 Flow Matching？**

> **标准回答**：DDPM 是弯路去噪，前向方差调度形成弯曲路径，反向常要 20-50 步；Flow Matching 在噪声和数据间拉直线，网络学速度场，推理 5-32 步。我们选 FM 一是轨迹任务要在 RL 里反复采样（每场景 24 条×32 步），采样器快 5 倍 RL 就快 5 倍；二是直线路径下 score 与速度有闭式关系，后面做 Flow-GRPO 的 SDE 推导时干净。
>
> **加分回答**：再补数学——DDPM 目标 `E‖ε-ε_θ‖²`，FM 目标 `E‖v-(x1-x0)‖²`；DDPM 概率视角是 SDE+score，FM 是 CNF+向量场。工程上 FM 生态（SD3/FLUX/π₀）也在替我们踩坑。

**Q2：为什么推理用 Euler 不用高阶积分器？**
> **标准**：直线路径速度近似常数，Euler 一阶误差在 16-32 步内可接受；RK4 每步 4 次前向，轨迹只有 30 点，省下的精度不如省算力。
> **加分**：Heun 两次前向可作折中；1 步蒸馏场景才明显需要高阶或 consistency。我们 RL 用 32 步是为 log-prob 累计精度，不是积分精度不够。

**Q3：多模态坍缩（只会出平均轨迹）怎么解？**
> **标准**：(1) 从随机 `x0` 多次采样展开多模态；(2) 行为 token 条件化 16 种风格；(3) 不用单点 MSE 直接回归 x1（那才是必然坍缩）。
> **加分**：Risk-Init 加 `0.1*noise` 保组内多样性，否则 GRPO std→0；可提 winner-takes-all 或 mixture 头作备选。

**Q4：训练时 t 的分布为什么均匀采？**
> **标准**：覆盖路径全程，否则某段速度场没学过，积分到那里会拐。
> **加分**：可 importance sampling 偏向两端（曲率大处）或 SNR 加权；我们简单 U(0,1)+少量 t-embedding 缩放够用。

**Q5：如果一步 Euler 误差大，你会改 loss 吗？**
> **标准**：先加步数/Heun 验证是否真是积分误差；若是网络欠拟合，加容量或按 t 加权 loss。
> **加分**：velocity 目标理论上一步到位，实际多步是因为**边缘化后平均场会弯**——可提 OT-CFM 减路径交叉。

---

## B. Flow-GRPO 相关

**Q6：GRPO 是 LLM 的，你怎么迁到连续流上？**
> **标准**：核心差在 log-prob。离散 token 用 softmax；确定性 ODE 没有转移密度。我们引入 SDE，漂移加 score 项保边际，离散化后每步是高斯，log-prob 闭式，PPO ratio 就能算。
> **加分**：四步：Fokker-Planck 反解漂移；直线路径用速度替换 score；Euler-Maruyama；高斯 log 概率。轨迹不重采样，只重算 log_prob_new。

**Q7：PPO clip 取多少？clip 和 KL 各管什么？**
> **标准**：ε=0.2 经典值。clip 限制**单步**新旧比偏离；KL 拉住**整体**不远离 ref。
> **加分**：只有 clip 长期仍会漂；只有 KL 单步可能过大。监控三元组：clip 裁剪比例、KL、有效样本数——裁剪比例>40% 说明步子太大或数据太 off-policy。

**Q8：为什么只 LoRA？全参不好吗？**
> **标准**：0.5% 参数，保 Flow 预训练、省显存、可插拔 ref=同权重关 adapter。
> **加分**：RL 标量奖励信息量低于 SFT，低秩假设成立；全参易沿 reward 奇异方向 hack。

**Q9：奖励怎么设计？怎么防 hack？**
> **标准**：主项官方 PDMS（NC/DAC/TTC/Comfort/EP/SLC 乘积），加 agent 级最近威胁裕度做 shaping。
> **加分**：防 hack 三件套：EP 防「停着不动刷安全」；shaping 权重有上界且主项仍是官方分；组内相对优势看方向不看绝对值，打分器整体漂移不改梯度方向。reward 突涨 KL 突降要人工查漏洞。

**Q10：advantage 归一化细节？**
> **标准**：`view(-1, G)` 组内 `(r-mean)/std`，clamp ±2。
> **加分**：不能全局归一（组间难度不同）；std+ε 防除零；全同分组丢弃或降权——连着讲难例挖掘动机。

**Q11：和 PPO 比 GRPO 去掉 critic 的代价？**
> **标准**：方差略升（baseline 质量不如 value），省一个网络的显存和训练不稳。
> **加分**：轨迹任务 horizon 短、组内对比信号够；若未来上闭环多步决策，value 可能回来。

---

## C. VLA 与风险场相关

**Q12：object query 128 怎么定？**
> **标准**：64/128/256 网格搜索，128 平衡漏检与算力；强交互目标很少超 50。
> **加分**：128 节点交互图 shape 好看，方便 top-k sparse attention；空 query 学「无目标」是 DETR 传统。

**Q13：风险场 vs occupancy？**
> **标准**：occupancy 管几何有没有；风险场管危险程度、时间演化、交互来源。
> **加分**：旁车未压线但速度指向我们，occ 空、risk 亮；α 门控把交互前移到表征，z_risk 给生成/门控/评分三处复用。

**Q14：双专家数据不均衡怎么训？**
> **标准**：Base 全量，LTE 难例子集+热启动，门控用风险熵软混合。
> **加分**：LTE 不从零训防早期噪声；难例来自失败回放+熵 top-k+合成 cut-in；软路由有梯度、硬路由断。

**Q15：Risk-Init 和训练一致吗？**
> **标准**：必须一致——训练时 `x0=head(z_risk)+ε`，推理同式，否则 mismatch。
> **加分**：这是条件 FM 的 `p_0(·|c)`；纯推理换初始化是常见 bug，可以主动说踩过。

**Q16：导航语言怎么进网络？**
> **标准**：文本塔 embedding 池化进 cond，和 BEV/z_risk 拼接。
> **加分**：可 cross-attn token 级；指令换向（left/right）要做 counterfactual 测试防语言不敏感。

---

## D. 世界模型与 WAM 相关

**Q17：「世界模型预测天然可信」为什么是缺陷？**
> **标准**：OOD 时预测偏，规划器不知道偏，盲信就撞。应把 WM 输出当证据不当裁决。
> **加分**：类比「不知道自己不知道」；我们双层路由=知道边界；监控 fallback 率。

**Q18：逆一致性具体怎么做？防什么？**
> **标准**：`s,a->s'` 再 `s',a->s` 闭环 MSE；防模型忽略 a。
> **加分**：类比 CycleGAN cycle；forward 要有动力学监督防恒等双射作弊；stopgrad 防钻空子。

**Q19：可信路由实时性？**
> **标准**：路由本身几个 MLP，微秒级；WM 前向才是开销。
> **加分**：先能量/support 轻量门控，OOD 直接 fallback 不跑完整 WM——条件计算。

**Q20：四条隐变量为什么同一条流？**
> **标准**：共享 t，同刻 (traj,conseq) 对齐；分头各积分会错位。
> **加分**：loss 分量加权；中途可读 support 做早停；和「predict-then-check」对比。

**Q21：阈值怎么定？故障注入测什么？**
> **标准**：验证集扫 τ，信任率-碰撞率曲线拐点取保守侧。
> **加分**：故障注入（镜头遮挡、OOD 车）看 fallback 召回；线上盯信任率、fallback PDMS、KL。

---

## E. JEPA 相关

**Q22：JEPA vs MAE？**
> **标准**：MAE 像素重建，JEPA 表征预测；算力与语义聚焦差异。
> **加分**：多模态未来下像素回归出鬼影；JEPA 代价=无显似然，靠 EMA/正则/任务头。

**Q23：坍缩三道防线分工？**
> **标准**：stopgrad 不对答案；EMA 慢半拍；VICReg 几何禁塌。
> **加分**：只 stopgrad 无 EMA 仍可能慢慢漂移坍缩；VICReg 三项方差/协方差/不变性各防什么——方差防重合、协方差防维度复制、不变性防语义抖。

**Q24：I-JEPA 和 V-JEPA？哪个是世界模型？**
> **标准**：I 补空间现在，V 预测时间未来；V 是世界模型。
> **加分**：我们 Stage1 帧间目标=V-JEPA 路线，时序条件还有 t→t+1 的时间嵌入。

**Q25：三阶段为什么先冻 WM？**
> **标准**：防评分头梯度把预训练表征打歪；表征稳了再联合。
> **加分**：Stage2 本质大号 linear probing；Stage3 才解感知；类比 LLM：先训 LM 再对齐。

**Q26：65536 怎么选？排序 loss 为什么不用 MSE？**
> **标准**：path×speed 256×256，隐空间批量分，chunk 防显存。
> **加分**：listwise 对齐官方序，和 GRPO 组内同思想；MSE 被高分常规样本主导、长尾序学不到。

**Q27：0.7285 是不是范式失败？**
> **标准**：无标注+未打满 RL 的基线；消融去 Stage2 更低，瓶颈在对齐。
> **加分**：预期接 GRPO 后 0.78+；主动给数字拆解，不回避。

---

## F. 综合与开放题

**Q28：三个项目什么关系？**
> **标准**：VLA=生成更好的轨迹；WAM=联合评估后果+可信路由；JEPA=高效无像素世界模型。感知-预测-生成-评估闭环。
> **加分**：优化目标递进——生成质量 → 后果准确 → 预测效率与物理一致；方法论横线=Flow Matching 与组内相对信号。

**Q29：重做会改什么？**
> **标准**：风险场端到端学；阈值元学习；JEPA 三阶段更紧联合。
> **加分**：加闭环指标；奖励从规则 PDM 走向学习型；三个项目统一到一套 adapter 热切换的代码库。

**Q30：端到端未来怎么看？**
> **标准**：趋势是 E2E+世界模型+RL 三位一体。
> **加分**：E2E 给上限，世界模型给想象与安全验证，RL 给对齐；可信域是上车前提。

**Q31（压）：0.8713 但 21% 场景趋近 0，提升在哪？**
> **标准**：乘积指标下修复的是从 0 到正的子集，常规 79% 不掉。
> **加分**：给拆解数字（长尾 0→0.5+），说明涨分结构=修一票否决项。

**Q32（压）：LoRA 是不是不够强？**
> **标准**：选择不是妥协；RL 低秩假设成立。
> **加分**：全参在标量奖励下更易 hack；ref=关 adapter 工程也依赖 LoRA 结构。

**Q33（压）：Flow-GRPO 是腾讯的，你做了什么？**
> **标准**：迁移到轨迹：ODE→SDE、PPO-clip+KL、PDM+agent 归因、难例挖掘。
> **加分**：文生图 vs 轨迹的差异列表（低维、乘积奖励、长尾分布），点明直接套会挂的三处。

**Q34（压）：JEPA 分低是不是 JEPA 不行？**
> 见 Q27。

**Q35（压）：上车最大挑战？**
> **标准**：实时性、安全兜底、分布 gap。
> **加分**：最核心是**验证**——开环 0.87 不等于闭环安全；可信路由是安全论证一块拼图，完整还要 RSS/HJ/最小风险策略；体现边界感。

**Q36（压）：你简历里「独立完成」具体指什么？有没有水分？**
> **标准**：按模块列 ownership——风险场/生成器/GRPO 是我；BEV 主干是共用。
> **加分**：能当场写出哪几个文件、loss 权重为什么取 0.1/0.05、默认超参；被问测试用例能举「关 adapter 复现 ref 分数」这类自测。

**Q37（压）：如果 GRPO 训崩了你怎么 debug？**
> **标准**：看三曲线——reward、KL、组内 std；再看 clip 比例。
> **加分**：顺序：1) reward 涨 KL 爆→提 beta/降 lr；2) reward 不动 std≈0→难例/加多样性；3) reward 突跳→查打分器 hack；4) grad nan→查 log std 下溢、clamp σ；5) 先 overfit 一个场景组过拟合通路再 scale。

**Q38（压）：世界模型和轨迹预测网络区别？**
> **标准**：预测网络输出他车分布；世界模型含自车动作条件、能反事实 rollout、可带后果/支持域头。
> **加分**：WAM 还联合动作生成，不只是条件预测；接路由就从「预测模块」升「决策证据模块」。

**Q39：最近论文里和你工作最相关的一篇？**
> **准备稿**：π₀（FM+动作专家）、Flow-GRPO（我们后训练直系）、JEPA/V-JEPA（第三项目）、NAVSIM（评测）、RDT-2（离散训练连续推理——防被 FAST/RVQ 细节问穿）。每篇准备 30 秒「解决了什么/和我差异」。

**Q40：手推一下 FM 的一步更新 / 高斯 log-prob（白板）**
> **标准板书**：
> `x_t=(1-t)x0+t x1`；`∂x/∂t=x1-x0`；loss=MSE。
> `log N(x;μ,σ²)=-½Σ((x-μ)/σ)² - D log σ - (D/2)log 2π`。
> **加分**：再写 ratio=`exp(lp_new-lp_old)`，clip 区间，指出对 μ 的梯度链到 v_θ。

---

## G. 快速自测清单（面试前 30 分钟扫一遍）

- [ ] Flow Matching 训练 loss 目标是速度还是噪声？
- [ ] Euler 推理伪代码能默写吗？dt 和 t 的广播 shape？
- [ ] ODE 为什么没有 log_prob？SDE 化保什么不变？
- [ ] 高斯 log-prob 公式？ratio 怎么用 exp 差值算？
- [ ] PPO clip 方向：adv>0 时 clip 哪边？
- [ ] KL 的 ref model 怎么构造（关 LoRA）？
- [ ] 组内 advantage 的 view(-1,G) 为什么不能全局？
- [ ] 128 query、32x32x8 风险场每个数字的含义？
- [ ] alpha softmax 之后 sum=1 代表什么？
- [ ] Base/LTE 门控输入为什么是熵不是标签？
- [ ] Risk-Init 训练和推理必须一致的哪一行？
- [ ] WAM 四条隐变量维度 60/32/16/8 怎么来的？
- [ ] 可信路由两层各判什么？fallback 用什么轨迹？
- [ ] 逆一致性防什么？怎么防恒等作弊？
- [ ] JEPA 三阶段各自冻结/解冻谁？
- [ ] Target encoder 为什么不能进 optimizer？
- [ ] VICReg 三项 loss 各防什么？
- [ ] 排序 loss 和 MSE 在 PDMS 上的差别？
- [ ] path×speed 怎么得到 65536？
- [ ] 三个项目的 PDMS：0.8713 / 0.8012 / 0.7285
- [ ] 每个项目「我负责的三件事」各一句话
- [ ] 开环 NAVSIM 和闭环上车的 gap 怎么答？
- [ ] GRPO 训崩的 debug 顺序？
- [ ] π₀ 原版有没有 FAST？RDT-2 推理走不走 RVQ？
- [ ] MoT vs MoE 路由粒度各是什么？
- [ ] und 为什么 causal、gen 为什么 bidirectional？
- [ ] gen→und cross-attn 为何 detach？只断 cross 还是断整个 und？
- [ ] moe_gen_mask 何时生成、前向中能否改？
- [ ] DCAE 压缩比与为何必须因果？
- [ ] 比亚迪 MOT 90 秒口述稿能否顺下来？
- [ ] 你在 MOT 落地踩过的 2 个坑（pack/mask、detach 边界）？

---

## H. 附录：常见追问链预演（面试官连环挖坑）

**链 1：Flow Matching -> SDE -> log_prob -> LoRA**
1. FM 是确定性的吧？→ 是，ODE。
2. 那 GRPO 的 log π 从哪来？→ 采样改 SDE，高斯转移。
3. 改 SDE 还能学原来的速度场吗？→ 漂移加 score 补偿保边际，训练目标不变。
4. 梯度怎么到大模型？→ log-prob→mean→v_θ→只 LoRA A/B。
5. 为什么不全参？→ 见 Q8。

**链 2：风险场 -> 熵门控 -> 双专家 -> RL**
1. 为什么不用 occ？→ 交互与时间维。
2. 熵高为什么用 LTE？→ 长尾分布在 LTE。
3. 门控可微吗？→ softmax(熵) 可微，端到端。
4. RL 阶段动不动门控？→ 动，同样挂 LoRA；监控两专家激活比例漂移。

**链 3：世界模型 -> 支持域 -> 路由 -> 安全论证**
1. 凭什么敢信 WM？→ 不全信，双层路由。
2. 支持域怎么算？→ 训练分布能量/密度，JEPA 项目里是 cycle energy。
3. 阈值哪来？→ 验证集曲线拐点+故障注入。
4. 最终安全谁兜底？→ RSS/HJ/最小风险，项目边界说清楚。

**链 4：JEPA -> 坍缩 -> EMA -> 评分头**
1. 会不会全塌了？→ 三道防线。
2. EMA 系数多少？0.996，太小跟太快可能共谋，太大欠拟合。
3. 为什么排序不回归？→ 官方乘积+序数评测。
4. 分低怪谁？→ 给消融数字，瓶颈在对齐与 RL 未打满。

---

*复习路径建议：先读第一部分把名词打成常识 -> 对着第二部分手写一遍 FM 和 GRPO 的训练步 -> 用第二补章的四份话术各录 3 分钟自述 -> 第三部分自问自答并过一遍追问链。*
