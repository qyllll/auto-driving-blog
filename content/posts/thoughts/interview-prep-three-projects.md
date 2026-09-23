---
title: "自动驾驶面试深度复习：RiskField-VLA + TrustDrive-WAM + JEPA-DRIVE 三大项目全拆解"
date: 2026-09-23
draft: false
categories: ["个人思考"]
summary: "VLA/WM/WAM + 比亚迪MOT + 四项目面试全书（5300+ 行）：原理与伪代码；第一人称口述；预训练后训练；具身/VLA岗位指南；ML八股/概率/手写题/论文；按真实被问原题整理的实战包（自我介绍、手撕FM、yaw、RL数据单位、离职与反问、自驾vs具身迁移）。"
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
5. **第四部分：预训练与后训练全解** —— 自监督目标、数据配方、SFT→RL 流水线、DPO/PPO/GRPO 对比、工程清单与 Q&A。
6. **第五部分：具身智能 / VLA 工程师岗位指南** —— JD 黑话翻译、知识树、动作表示、从 0 设计 VLA、八股与手写题、2-4 周冲刺计划。
7. **第六部分：VLA / WM / WAM 原理精讲（第一人称）** —— 统一建模四层、世界模型定义与三形态、世界动作模型四支柱、对比表、口述稿与追问链。
8. **第七部分：基础补强** —— ML 八股（反传/LN/优化器/过拟合）、数学概率（高斯/KL/MLE/重要性采样）、手写题（FM 必背 + LRU/滑窗等）、π₀/OpenVLA/Diffusion Policy 论文 60 秒、行业与基模话术。
9. **第八部分：真实面试实战包** —— 自我介绍、比亚迪 ownership 与手撕 Flow Matching、yaw 角、RL 数据计量、数据处理、MOT+FM+自回归+锚点 query、离职原因、自驾 vs 具身迁移、反问与 7 天计划。

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

---

# 第四部分：预训练与后训练全解（面试高频）

> 大厂/创业公司问「你做过预训练吗」「SFT 和 RL 怎么接」「为什么先 SFT 再 GRPO」几乎是标配。本部分把**预训练 → SFT → RL 后训练**整条链讲透，并和我们三个项目 + 比亚迪 MOT 对齐。

---

## 1. 名词总表（预训练/后训练）

| 术语 | 全称 | 一句话定义 |
|------|------|------------|
| Pretraining | 预训练 | 在大规模通用数据上以自监督目标学通用表征/能力，不绑定单一下游任务 |
| Post-training | 后训练 | 预训练之后，用更窄但更「对齐」的数据（指令、偏好、奖励）把模型调到可用/安全 |
| SFT | Supervised Fine-Tuning | 监督微调：用「输入-标准输出」配对做交叉熵/回归，教格式与任务行为 |
| RLHF | RL from Human Feedback | 人类偏好训练奖励模型，再 PPO 优化策略（InstructGPT 路线） |
| RLAIF | RL from AI Feedback | 用 AI 裁判代替人类标注偏好 |
| DPO | Direct Preference Optimization | 跳过显式 RM，用偏好对直接改策略的闭式 loss |
| GRPO | Group Relative Policy Optimization | 同 prompt 采一组，组内相对优势，无 critic |
| PPO | Proximal Policy Optimization | critic + clip/KL 的经典策略梯度 |
| Reward Model | 奖励模型 | 把输出映射为标量分数的模型 |
| Preference Pair | 偏好对 | (win, lose) 一对，DPO/RLHF 的原子样本 |
| Rejection Sampling | 拒绝采样 | 采多条，按规则/RM 只留最好的当 SFT 数据 |
| CoT | Chain-of-Thought | 思维链：先推理再给答案 |
| Distillation | 蒸馏 | 大模型（teacher）的分布/logits 教小模型（student） |
| Curriculum | 课程学习 | 数据由易到难组织 |
| Hard Negative Mining | 难负例挖掘 | 挑模型易错样本加权训练 |
| EMA | Exponential Moving Average | 权重滑动平均，评测/稳定性常用 |
| Checkpoint Averaging | 权重平均 | 多 checkpoint 平均，常提泛化 |
| Data Mixing | 数据配比 | 多源数据比例（文本/视觉/动作/仿真） |
| Dedup | 去重 | 近重复数据剔除，防记忆化与过拟合 |
| Safety Alignment | 安全对齐 | 有害内容拒答、价值观约束 |
| Red-teaming | 红队测试 | 专门找漏洞/越狱的对抗测试 |
| Scaling Law | 缩放定律 | loss 随参数/数据/算力的幂律关系 |
| Token Budget | token 预算 | 序列长度与 batch 的算力规划 |
| MoE Router | 专家路由器 | 决定 token 激活哪些专家 |
| Load Balancing Loss | 负载均衡 loss | 防 router 坍缩到少数专家 |
| Zero/One/Few-shot | 0/1/少样本 | 不给/给一/给几个示例就推理 |
| In-context Learning | 上下文学习 | 不更新权重，靠 prompt 示例学会任务 |
| Catastrophic Forgetting | 灾难性遗忘 | 微调后丢掉预训练通用能力 |
| Elastic Weight Consolidation 等正则 | 权重锚定 | 用 KL/锚点参数防遗忘 |
| Offline RL | 离线 RL | 只在固定数据集上优化，不在线交互 |
| Online RL | 在线 RL | 策略实时与环境交互采样 |
| Off-policy / On-policy | 同/异策略 | 数据来自旧策略 vs 当前策略 |
| Importance Sampling | 重要性采样 | 用 ratio 纠正新旧策略分布差 |
| World Model Rollout | 世界模型推演 | 用 WM 替代真实环境做虚拟交互 |
| Sim-to-Real | 仿真到现实 | 仿真训练、实车/真机部署的迁移 |
| Domain Randomization | 域随机化 | 随机化纹理/光照/物理参数提升鲁棒 |
| Action Chunking | 动作块 | 一次预测多步动作，降频、稳 |
| Teleoperation | 遥操作 | 人遥控机器人收集演示 |
| DAgger | - | 在策略状态上收集专家标签 |
| Human-in-the-loop | 人在环 | 关键节点人工审核/纠正 |
| RLAIF Safety | - | AI 裁判做安全维度 reward |

---

## 2. 预训练 Pretraining：到底在学什么

### 2.1 三种主流自监督目标

```text
(1) 自回归语言建模 (Causal LM)
    目标:  P(x_t | x_{<t})   交叉熵
    学到:  语法、世界知识、指令跟随的底子
    例子:  GPT 系、Reasoner 塔的 und 路径

(2) 掩码重建 (MLM / MAE / BERT 风)
    目标:  被 mask 的单元 (词/patch) 重建
    学到:  双向上下文表征
    例子:  BERT, MAE

(3) 对比 / 联合嵌入预测 (SimCLR, JEPA)
    目标:  同增广对齐, 或 表征空间预测未来/被遮块
    学到:  不变性 + 时序动力学
    例子:  CLBYOL, I-JEPA, V-JEPA, 我们的 JEPA-DRIVE Stage1
```

**代码级对照（自回归 CE vs JEPA MSE vs Flow MSE）：**

```python
# A. 自回归 LM (Reasoner / und)
logits = lm_head(hidden[:, :-1])              # 预测下一个 token
loss_lm = F.cross_entropy(
    logits.reshape(-1, vocab),
    labels[:, 1:].reshape(-1),
    ignore_index=pad_id,
)

# B. JEPA (表征预测)
with torch.no_grad():
    z_tgt = target_encoder(masked_patches)
z_pred = predictor(context_encoder(visible), pos)
loss_jepa = F.mse_loss(z_pred, z_tgt)

# C. Rectified Flow / FM (Generator / gen)
x_t = (1 - t) * x0 + t * x1
v   = flow_head(gen_hidden, t)
loss_fm = F.mse_loss(v, x1 - x0)

total = loss_lm + w_j * loss_jepa + w_f * loss_fm  # 多任务预训练常见加权
```

### 2.2 驾驶/具身预训练的特有维度

| 维度 | 文本 LLM | 驾驶/具身预训练 |
|------|----------|-----------------|
| Token | BPE 词 | 视频 patch、BEV 格、轨迹点、状态向量、指令 |
| 时间 | 一维序 | 三维 mRoPE（T,H,W）或因果时序 |
| 目标 | 下一词 | 下一帧表征 / 速度场 / 动作块 / 检测 |
| 数据 | 网页文 | 多相机日志、仿真、遥操作、Omniverse 合成 |
| 对齐难题 | 指令-回答 | 视觉-语言-动作三模态时间对齐 |
| 安全 | 有害话术 | 碰撞、压线、失效降级 |

### 2.3 数据工程（预训练成败一半在数据）

```text
1. 采集: 实车日志 / 遥操作演示 / 仿真合成 (Cosmos+Omniverse 风格)
2. 清洗: 时间戳对齐、传感器标定、烂帧丢弃、车道级轨迹质量过滤
3. 去重: 感知哈希 + embedding 近邻去重, 防「10 万帧都是同一路口」
4. 打标: 可自动的 (碰撞检测、PDMS、红绿灯) + 人工抽检
5. 配比: 真实:仿真:合成 = ? ; 常规:长尾 = ?  (我们项目约 8:2 再难例上采样)
6. 增广: 光照、天气、镜头污损、轨迹扰动 (给「偏了还能救」的标签)
7. 混合课程: 先短 horizon 简单路, 后长 horizon 交互
```

**面试金句**：
> 预训练不是「把数据丢进 dataloader」——配比和去重对驾驶模型的收益经常大于改结构。长尾不足就用仿真和难例挖掘，而不是无脑堆常规高速片段。

### 2.4 Scaling 与结构选择（会被问「为什么这个规模」）

```text
小模型 (1-4B):  车端实时, 容量受序列长度和模态数限制
中模型 (10-20B): Cosmos Nano 级, 云端/域控, 我们 MOT 常讨论的档
大模型 (60B+):  云端世界模型/教师, 蒸馏给车端

选择原则:
  - 输入 token 数 (多相机视频 >> 文本)
  - 推理频率 (规划 10Hz vs 视频评 1Hz)
  - 是否要在线 RL (采样贵, 模型不能过大)
  - 是否当 teacher (大) 还是 deploy (小)
```

### 2.5 预训练常见坑（第一人称可背）

1. **模态坍缩到语言**：文本 token 多，共享骨干被 CE 主导——塔间/模态均衡 loss 或分塔隔离（MoT 动机之一）。
2. **视频 encoder 偷看未来**：DCAE 非因果 → 预测任务泄题。
3. **位置编码错**：T/H/W 混用 1D RoPE，模型分不清帧序和空间序。
4. **pad 泄漏**：attention 没屏蔽 pad，学到「永远看最后一个」。
5. **LR 过大伤预训练权重**：后训练阶段应用小 lr + 只训 adapter。

---

## 3. 后训练 Post-training：SFT → RL 的完整地图

### 3.1 后训练为什么必要

```text
预训练学到的是「像训练分布」，不是「按你的意图和安全约束行动」。
后训练三件事:
  A. 格式与能力:  按 schema 出轨迹/JSON, 会 CoT, 会拒答
  B. 价值与安全:  不编造、不危险、不越权
  C. 任务性能:    PDMS/成功率/延迟 等硬指标拉齐
```

### 3.2 典型流水线（自驾/具身版）

```text
Stage 0  预训练 backbone
         语言 CE + 视觉/视频 JEPA 或对比 + (可选) Flow 初训

Stage 1  SFT / 行为克隆 BC
         数据: 人类演示 / 高分轨迹 / 高质量指令-响应
         目标: 会按任务出合法轨迹; 会听导航指令
         形式: Flow MSE 到演示轨迹, 或 LM CE 到动作 token

Stage 2  Rejection Sampling / DPO (可选)
         采多条 -> 规则过滤 -> 留下的当正样本
         或偏好对: (安全轨迹胜, 危险轨迹负) -> DPO

Stage 3  RL 后训练 (GRPO / PPO / Flow-GRPO)
         奖励: PDM / 成功率 / 舒适 / 进度
         目标: 冲长尾、修 SFT 分布外的失败

Stage 4  蒸馏与部署压缩
         大 teacher -> 小 student; 量化; 步数蒸馏 (32->4)
```

**我们项目的对应关系（面试串讲）：**

| Stage | RiskField-VLA | TrustDrive-WAM | JEPA-DRIVE | 比亚迪 MOT |
|-------|---------------|----------------|------------|------------|
| 预训练 | VLM/BEV 主干 | 世界模型时序编码 | Stage1 JEPA | Cosmos-3 双塔 |
| SFT/BC | 轨迹 Flow SFT | 联合流轨迹分量 | Stage2/3 评分头 | 轨迹 Flow Head SFT |
| RL | Flow-GRPO 0.8713 | 可信路由+可选 GRPO | 计划中 | 可选 Flow-GRPO |
| 压缩 | LoRA 已很小 | fallback 降级 | 隐空间评分省算力 | Nano 车端 |

### 3.3 SFT 细节（常问「你的 SFT 数据怎么来的」）

```python
# 轨迹任务的 SFT = Flow Matching 到演示分布
def sft_step(batch):
    x1 = batch['expert_traj']          # 人类/过滤后的演示
    x0 = torch.randn_like(x1)
    t  = torch.rand(B)
    xt = (1-t)*x0 + t*x1
    v  = model(xt, t, batch['cond'])
    return F.mse_loss(v, x1 - x0)

# 拒绝采样版: 同场景采 8 条 -> PDM 过滤 -> 只留 top-1 再 SFT
def rejection_sft(model, scenes):
    for sc in scenes:
        cands = model.sample(sc, K=8)
        scores = pdms(cands, sc)
        best = cands[scores.argmax()]
        if scores.max() > thresh:
            buffer.append(best)
    train_on(buffer)
```

**SFT 数据质量 > 数量**：1 万条高质量、低碰撞、舒适轨迹，好过 100 万条平庸刹车点头数据。

**SFT 常见失败**：

- 只学会「平均保守」→ 长尾全靠 RL 修
- 指令不敏感（left/right 说啥都直行）→ 对比指令增广
- 动作抖动（jerk 大）→ 舒适度加权或对控制量导数惩罚

### 3.4 DPO vs PPO/GRPO（高频对比题）

| 维度 | DPO | PPO | GRPO / Flow-GRPO |
|------|-----|-----|------------------|
| 需要 RM？ | 不显式要，偏好对内化 | 要 value 或 reward | 要标量 reward |
| 在线采样？ | 否，离线偏好 | 是 | 是（组采样） |
| critic | 无 | 有，贵 | 无 |
| 主要风险 | 偏好分布偏、过拟合 pair | 不稳定、reward hacking | 组内方差、KL 漂移 |
| 适合 | 格式/风格/安全偏好 | 通用 | 有明确可计算奖励（PDMS） |

```python
# DPO 核心 loss (对比参考, 我们主用 GRPO)
def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    # w=win, l=lose
    ratio = beta * (
        (pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l)
    )
    return -F.logsigmoid(ratio).mean()
```

**为什么我们轨迹用 GRPO 不用 DPO**：PDMS 是**可计算的环境奖励**，不需要人类标注偏好对；组内相对优势直接利用「同场景 24 条谁更好」，比构造 win/lose 对更省标注、信号更密。

### 3.5 RL 后训练工程清单（被问「你怎么训的 GRPO」按这个扫）

```text
1. 采样:  组大小 G, 步数 T, 温度/噪声水平, 种子固定可复现
2. 打分:  奖励函数版本号, 单位, clip 范围
3. 优势:  组内归一, clamp, 剔除 std=0 组
4. 更新:  inner epochs, clip ε, KL β, lr, warmup, grad clip
5. 参考:  ref = 同 base 关 adapter, 定期对拍 logits
6. 参数:  LoRA rank/α/dropout; 哪些模块可训 (Flow head/gen router)
7. 监控:  reward, KL, clip比, entropy/std, grad_norm, 各 PDMS 子项
8. 防hack: EP 防不动, 打分器红队, 人工看轨迹视频
9. 评测:  留出场景集 + 长尾子集 + 开销 (延迟/显存)
10. 回滚:  每 N step 存 ckpt, 指标回退即回滚
```

### 3.6 预训练 vs 后训练：面试标准对比话术

> **预训练**解决「模型懂不懂世界」——大规模自监督，目标是通用表征（语言 CE、视频 JEPA、Flow 初训），数据脏一点也能学，容量和配比是关键。
>
> **后训练**解决「模型听不听话、强不强」——SFT 给格式和基线能力，RL/DPO 对齐奖励与安全。后训练数据少而精，对 KL/clip/参考策略更敏感。
>
> 我们三个项目和 MOT 都是这条流水线的不同切面：MOT/JEPA 偏预训练底座，VLA/WAM 的 Flow SFT 是 BC，Flow-GRPO 是 RL 后训练。面试官问任何一段，我都能指回这条链上的位置。

### 3.7 压力面：预训练/后训练 Q&A 补充

**Q：你做过预训练吗？从零还是继续训？**
> **标准**：按事实答——若主要是加载 Cosmos/VLM/视频骨干做领域适配，就说「预训练权重 + 领域继续预训练/中期训练 + 任务后训练」，不要吹成从零 pretrain trillion token。
> **加分**：能说出继续预训练时的配比、lr 量级（通常比从零低 1-2 个数量级）、以及如何用小规模 proxy 验证数据配方。

**Q：SFT 和 RL 先后顺序能反过来吗？**
> **标准**：一般 SFT→RL：SFT 先把输出拉进合理空间，RL 在此基础上探索；直接 RL 从随机策略探索驾驶轨迹效率极低。
> **加分**：例外是已有强 BC 策略时可跳过大量 SFT；以及 rejection sampling 其实是「用 RL 思想产 SFT 数据」的中间态。

**Q：灾难性遗忘怎么防？**
> **标准**：小 lr、少 epoch、LoRA/Adapter、KL 锚 ref、混一部分预训练数据回放（replay）。
> **加分**：监控 held-out 通用任务分；MOT 场景就是「Reasoner CE 分不掉、只提升生成头」的分模块遗忘监控。

**Q：奖励稀疏怎么办？**
> **标准**：过程奖励 shaping（离障碍裕度）、分层奖励（子目标）、难例保证有信号、组内相对化把绝对稀疏变相对稠密。
> **加分**：警惕 shaping 引入 hacking——主项仍是终局 PDMS。

**Q：数据不足怎么后训练？**
> **标准**：仿真增广、拒绝采样自我提升、蒸馏大模型/规则专家、DPO 用少量偏好、迁移其他城市/车型。
> **加分**：先查是不是数据质量问题而不是量的问题；长尾 21% 上采样好过继续堆 79%。

**Q：在线 RL 和世界模型 rollout 怎么选？**
> **标准**：真仿真贵 → 用 WM/开环打分器当廉价环境；WM 不可信时 fallback 或只在支持域内 rollout。
> **加分**：我们 WAM 的可信路由正是「WM 只在支持域当环境」的安全版 model-based RL；NAVSIM 开环打分也是低成本 reward oracle。

**Q：LoRA 和全参在后训练各自何时用？**
> **标准**：探索/对齐/多任务切换 → LoRA；数据极多、要榨干容量、最终一版定妆 → 可全参小 lr。
> **加分**：后训练常用「生成头全参 + 主干 LoRA/冻结」的混合，和 MOT 的「Flow Head 训、Reasoner 冻」一致。

**Q：为什么 GRPO 不用 critic？你的 reward 是学习的还是规则的？**
> **标准**：组均值当 baseline，省 critic；我们主 reward 是规则 PDM/PDMS，可复现、难 hack 方向明确。
> **加分**：若以后上偏好/主观舒适，可加学习 RM，需与规则项加权并做红队。

---

# 第五部分：具身智能算法工程师 / VLA 工程师 岗位准备指南

> 面向 JD 高频出现的 **具身智能算法工程师**、**VLA 算法工程师**（有时叫「机器人学习」「端到端大模型」）。分五块：岗位到底要什么、知识树、项目怎么讲、代码/八股清单、面试轮次与准备计划。

---

## 1. 岗位拆解：JD 里的黑话翻译

### 1.1 常见 JD 条目 → 实际在考什么

| JD 原话 | 面试真正在考 |
|---------|--------------|
| 熟悉 Transformer/多模态大模型 | Attention 手推、VLM 结构、训练目标 |
| 熟悉扩散/Flow 生成模型 | DDPM vs FM、采样步数、log-prob |
| 熟悉 VLA（π₀/OpenVLA/RDT） | 动作表示、离散 vs 连续、动作专家 |
| 熟悉强化学习（PPO/GRPO） | advantage、clip、KL、reward 设计 |
| 熟悉世界模型 | p(s'|s,a)、WM 类型、与预测网络区别 |
| 具身数据采集与遥操作 | 演示质量、DAgger、sim2real |
| 仿真环境（Isaac/MuJoCo/Genesis） | 物理、域随机、闭环评测 |
| 车端部署/量化 | 延迟、INT8、算子、传感器同步 |
| 论文复现/开源贡献 | 动手能力、读代码、消融设计 |
| 与硬件/标定/控制联调 | 坐标系、时间同步、接口 |

### 1.2 两类岗位的微妙差别

**VLA 算法工程师（偏模型）**

```text
核心产出:  VLA 模型结构、动作解码、指令跟随、多任务
高频技术:  VLM backbone, Flow/Diffusion action head,
           action chunk, cross-attn, 多相机 token
评测:      指令成功率, 操作任务成功率, 导航到达率
我方优势:  FM/Flow Head、MoT、风险条件化、GRPO 后训练
```

**具身智能算法工程师（偏系统+学习）**

```text
核心产出:  感知-决策-控制闭环, 数据闭环, 仿真到真机
高频技术:  策略学习(BC/RL), 世界模型, 状态估计,
           遥操作数据, 域随机, 安全约束
评测:      任务成功率, 鲁棒性, 实时性, 失败恢复
我方优势:  三项目闭环叙事 + 可信路由/安全 + NAVSIM 经验
```

**面试一句话定位（建议背）**：
> 「我更偏 **VLA/世界模型的模型侧**，但有完整自驾闭环和后训练经验——从多模态理解（MOT/JEPA）到连续动作生成（Flow）到奖励优化（GRPO）到安全接口（可信域）。如果岗位需要更多真机/仿真，我的迁移路径是：轨迹=低维 action chunk，PDMS=task reward，开环评测=仿真 rollout。」

---

## 2. 知识树：VLA/具身工程师要准备的模块

### 2.1 模块一：VLM 与多模态基础（必会）

```text
- ViT patchify, CLIP 双塔, 纯 VLM (LLaVA 式: 视觉 token 进 LLM)
- Cross-attention 注入 vs 拼接注入
- 位置编码: 1D RoPE vs 2D/sRoPE vs mRoPE(T,H,W)
- 指令微调: LLaVA-NeXT / Qwen-VL 级结构直觉
- KV cache, GQA, 前向延迟从哪来
必会代码:  写一个 mini LLaVA 前向 (vision tower -> projector -> LLM)
```

### 2.2 模块二：动作表示与解码（VLA 核心差异点）

| 方案 | 表示 | 解码 | 代表 | 面试考点 |
|------|------|------|------|----------|
| 离散 token | 动作量化成词表 | 自回归 softmax | OpenVLA | 码本大小、精度损失 |
| 回归头 | 直接 μ | 一步 MLP | BC-Z 风格 | 多峰坍缩 |
| 扩散 | 噪声->动作 | DDPM 20-50 步 | RDT | 步数、条件 |
| 流匹配 | 噪声->动作 | FM 5-32 步 | π₀, 我们 | 速度场、SDE/GRPO |
| 混合 | RVQ 离散辅助 | 训练 CE 推理连续 | RDT-2 | 训推不一致 |
| FAST | 动作->文本码 | AR | π₀-FAST | 与原版 π₀ 区分 |

**动作块 Action Chunk 细节**：

```text
输出 T=16/32 步, 不是 1 步:
  - 降低决策频率依赖, 时间上更平滑
  - 错误会「锁」T 步 -> 需要 replan 策略 (执行前 K 步)
  - 与 Flow 一次生成整块天然契合
维度例:  机械臂 (x,y,z,rx,ry,rz,gripper) 或 车 (x,y,yaw) / (steer,accel)
```

### 2.3 模块三：生成模型（和我们 FM 部分打通）

```text
必须能白板:
  1. DDPM 前向反向, 为何多步
  2. FM 直线路径, loss, Euler
  3. 两者 log-prob / score 关系 (为 GRPO 铺垫)
  4. CFG:  v = v_uncond + s*(v_cond - v_uncond)
  5. 步数蒸馏直觉
可选加分:
  - Rectified Flow / SD3 / FLUX 训练细节
  - 一致性模型
  - 在 1D 轨迹 vs 2D 图像上的差异 (轨迹维低, 步数可少)
```

### 2.4 模块四：强化学习与后训练（岗位 JD 高频）

按第四部分复习，另补具身特有：

```text
- 奖励设计: 稀疏成功 0/1 vs 中间 shaped
- 安全 RL: 约束 MDP, 惩罚项, 可信域/后备策略 (我们的 TrustRouter)
- SimRL: 在 Isaac/MuJoCo rollout, 域随机
- 离线到在线: 先 BC 再 PPO/GRPO (Decision Transformer 知道即可)
- 探索: 噪声动作、熵奖励、Curiosity (能聊即可)
```

### 2.5 模块五：世界模型与仿真

```text
- 定义 p(s'|s,a); 像素 vs 表征 vs 几何
- 用途: (a) 想象 rollout 做 RL  (b) 合成数据  (c) 评测
- 与我方: JEPA 表征 WM, WAM 联合动作后果, MOT 全模态 WM
- 仿真栈: Isaac Sim/Lab, MuJoCo, Genesis, CARLA, NAVSIM
- 关键: 物理真实性, 渲染 gap, 随机化
```

### 2.6 模块六：数据与遥操作（具身 JD 几乎必问）

```text
- 遥操作设备: 力反馈手柄, 外骨骼, 手套, 主从臂
- 演示质量: 成功率、多样性、纠正率、段长
- 标注: 自动切段、失败标记、语言指令生成 (VLM 可自动生成指令)
- DAgger: 在机器人自己的失误状态问专家「这时该怎样」
- 数据效率: 10 条好演示 vs 100 条烂演示; 视觉增广
- Sim 数据:  procedural 域随机, 真机 fine-tune
我方可迁移故事: NAVSIM 长尾筛选 ≈ 具身失败案例挖掘
```

### 2.7 模块七：部署与实时（工程向加分）

```text
- 延迟预算: 感知 30ms + 策略 20ms @ 20-50Hz 控制
- FP16/INT8, TensorRT, 量化对 Flow 步数敏感点
- 多传感器时间同步、外参、自车运动补偿
- 失败模式: 传感器掉线 -> 模态 dropout 预训练 (MOT/Cosmos 也讲这个)
- 安全监控: 输出限幅、可行性检查、看门狗
```

### 2.8 必读论文/开源（VLA 面试弹药，各 30 秒能讲）

| 类别 | 材料 | 你的一句话 |
|------|------|------------|
| VLA | OpenVLA | 离散动作 + 开源基线 |
| VLA | π₀ / π₀-FAST | FM 动作专家；FAST 是另一变体 |
| VLA | RDT / RDT-2 | 扩散/混合动作，RVQ 训推差异 |
| 生成 | Flow Matching 论文 | 直线路径与 CFM |
| RL | PPO, GRPO/DeepSeek-R1 | clip；组相对优势 |
| 世界模型 | Dreamer 系, V-JEPA | latent rollout；表征预测 |
| 端到端 | UniAD/VAD, NAVSIM | 模块端到端；开环 PDMS |
| 具身 | ACT, Diffusion Policy | 动作块；扩散策略 |
| 架构 | MoT/Cosmos-3, DeepSeekMoE | 双塔；稀疏专家 |

**Diffusion Policy / ACT 一定要会**（具身面试比自驾更爱问）：

```text
Diffusion Policy:  观察历史 obs -> 条件扩散出 action chunk
                    与图像扩散同构, 条件是 obs
ACT (Aloha):      CVAE + Transformer, 预测动作块, 低延迟真机常用
对比:  Diffusion 多峰强; ACT 快; 我们 FM 折中且可接 GRPO
```

---

## 3. 项目怎么讲（具身/VLA 面试版 60 秒 x N）

### 3.1 通用公式

```text
[问题] 场景里什么不行 -> [我] 负责哪三块 ->
[方法] 结构/训练/奖励一句黑话 ->
[结果] 数字或消融 -> [迁移] 对贵司具身/VLA 的价值
```

### 3.2 比亚迪 MOT 60 秒（偏 VLA 岗）

> 「我做 Cosmos-3 MoT 车载世界模型：每层 und/gen 双投影，理解因果、生成双向流，视频走因果 DCAE，gen cross-attn 读语义并 detach。我负责 pack/mask、传感器 token 接入和轨迹 Flow Head。这和 VLA 岗位的同构是：**VLM=Tower und，action head=gen+Flow**，π₀ 是外挂专家，我们是一体双塔。」

### 3.3 RiskField-VLA 60 秒（偏 VLA 岗）

> 「多相机+导航 VLA，128 query 出时空风险场，风险熵门控 Base/LTE，Risk-Init 的 FM 出 8 条候选，Flow-GRPO 后训练，NAVSIM 0.8713。对操作类 VLA：风险场=安全性条件，action chunk=我们的轨迹块，PDMS=你们的任务 reward+安全项。」

### 3.4 TrustDrive-WAM 60 秒（偏具身/系统岗）

> 「动作和后果同一联合流生成，双层可信路由，OOD 就回退经典规划。具身里对应：世界模型想象 rollout 的可信边界、安全监控和 fallback——这是真机上比刷成功率更硬的工程能力。」

### 3.5 JEPA-DRIVE 60 秒（偏研究/世界模型岗）

> 「无像素重建的世界模型三阶段：JEPA 帧间预训练、冻结后排序评分头、端到端。防坍缩 EMA+stopgrad+VICReg。对具身就是廉价 imagination 和 demo 增广的基础表征。」

---

## 4. 代码与八股清单（VLA/具身版）

### 4.1 必须能手写（白板/共享屏幕）

```text
1. Multi-head / cross-attention  (10 行级)
2. Flow Matching train step + Euler sample
3. PPO-clip loss + ratio
4. GRPO 组内 advantage
5. LoRA 前向
6. JEPA EMA update + stopgrad loss
7. Action chunk 拼接与执行前 K 步 replan 伪代码
8. 简单 BC: obs->action MLP/MSE
```

### 4.2 必背八股（分层）

**第一层（答得干脆）**
- Transformer 复杂度、为什么 LN、为什么残差
- GQA/MHA 区别
- ReLU/GELU、学习率 warmup
- 过拟合怎么查（train/val gap、数据污染）

**第二层（拉开差距）**
- FM vs DDPM、DPO vs GRPO、MoT vs MoE
- 扩散多峰 vs 回归均值
- 世界模型 vs 预测网络
- 灾难性遗忘、reward hacking 案例

**第三层（显示深度）**
- 确定性 ODE 无 log_prob，SDE 化保边际
- JEPA 坍缩机理
- PDMS 乘积与长尾
- 支持域与可信路由、开环闭环 gap

### 4.3 系统/实验素养（研究岗+工程岗都加分）

```text
- 消融表怎么设计: 一次只动一个变量
- 随机种子: 至少 3 seed 报 mean±std
- 复现: 固定 cudnn deterministic, 记录 commit+config
- 调试顺序: overfit 一个 batch -> 小数据 -> 全量
- 日志: loss 分项、grad norm 分桶、定性轨迹视频
```

---

## 5. 面试轮次与 2-4 周准备计划

### 5.1 典型轮次

```text
轮 1  简历深挖 + 项目 30 分钟     -> 话术 A-D+, E, F
轮 2  基础: ML/深度学习/数据结构  -> 八股第一层 + 手写 attention
轮 3  专业: VLA/生成/RL/世界模型  -> 话术 + 第三部分 Q&A + 第四部分
轮 4  系统设计: 从0设计一个VLA     -> 见 5.2 模板
轮 5  HR/主管: 动机、论文、反问    -> 见 5.3
```

### 5.2 「从 0 设计一个 VLA 系统」答题模板（高概率）

```text
1. 任务与接口:  输入(多相机+语言+本体), 输出(action chunk / 轨迹), 频率, 延迟预算
2. 模态编码:    ViT/VLM, 时间窗, 位置编码, 同步
3. 融合:        拼接 / cross-attn / BEV; 语言如何注入
4. 动作头:      FM vs 扩散 vs 离散; chunk 长度; 多峰
5. 训练数据:    演示来源, 配比, 长尾, 质量过滤
6. 训练流程:    预训练 -> SFT/BC -> (可选) GRPO; 冻结策略
7. 安全:        输出限幅, fallback, 支持域/OOD 检测, 看门狗
8. 评测:        仿真成功率, 关键指标, 回归测试集
9. 部署:        量化, 步数, 监控, 日志回流闭环
10. 迭代:       哪里最可能是瓶颈(数据/奖励/结构), 如何消融
```

**收尾金句**：
> 「我会把 **安全路由和评测集** 和模型第一周一起建，而不是上线前才补——自驾和具身的失败都长在长尾和分布外。」

### 5.3 反问面试官（准备 3 个）

```text
1. 团队当前瓶颈是数据、奖励设计还是真机/仿真吞吐？
2. VLA 是自研骨干还是站在开源 π₀/OpenVLA 上改？后训练用 RL 吗？
3. 成功率之外如何定义安全与失败恢复？有没有闭环评测？
```

### 5.4 2-4 周冲刺表（可直接执行）

**Week 1 — 项目焊死**
- [ ] 话术 A/B/C/C+/D/D+/E/F 各录 3 遍，卡壳处改稿
- [ ] 默写 FM、GRPO、attention、JEPA 四段代码
- [ ] 准备 2 个 MOT 踩坑 + 1 个 GRPO debug 故事

**Week 2 — 专业深度**
- [ ] 第三部分 40 题过两遍（自问自答出声）
- [ ] 第四部分预训练/后训练表全部能讲
- [ ] 对比表：DPO/PPO/GRPO、DDPM/FM、MoT/MoE、I-JEPA/V-JEPA

**Week 3 — 岗位对齐**
- [ ] 精读目标公司 3 个 JD，把黑话映射到知识树
- [ ] 读 2 篇 VLA 论文（π₀ + Diffusion Policy 或 ACT）各写 30 秒
- [ ] 做一次「从 0 设计 VLA」模拟 20 分钟

**Week 4 — 模拟与查漏**
- [ ] 找人 mock 2 次：一次深挖简历，一次八股+手写
- [ ] 手写题限时：attention / FM step / PPO clip
- [ ] 准备反问、薪资、到岗时间
- [ ] 通读本文自测清单，勾选为 0 的项突击

---

## 6. 给「VLA 工程师」的差异化加分项（在众多只会背论文的人里出头）

1. **训推一致意识**：RDT-2 RVQ 训推差异、DCAE 因果、Risk-Init 必须进训练——说明你踩过「评测虚高」的坑。
2. **奖励可计算**：PDMS 规则奖励 + shaping + 防 hack，不是空谈 RLHF。
3. **安全接口**：可信域、fallback、模态 dropout——真机/量产最缺的人不是会调扩散的，是会让模型「知道自己不知道」的。
4. **效率**：Flow 步数、LoRA、隐空间评分 65536 轨迹、推理只激活需要的头——讲得出 ms 和显存的人更可信。
5. **数据直觉**：长尾 21%、难例挖掘、仿真配比——具身团队 50% 时间在数据上。
6. **完整链路**：MOT/JEPA（表征）+ FM（生成）+ GRPO（优化）+ 路由（安全）四个词能串成一句职业故事。

---

## 7. 岗位向压力面试题补充（具身/VLA）

**Q：Diffusion Policy 和你们 Flow 轨迹有什么异同？**
> 同：条件生成 action chunk，多峰，迭代采样。异：目标函数（ε/v）、步数调度、你们条件里多了风险场/导航/可信信号，且后接 GRPO；Diffusion Policy 原版多是纯 BC。

**Q：ACT 和 Diffusion Policy 怎么选？**
> 看延迟与多峰：真机高频控制优先 ACT/流少步；强多峰、多模演示优先扩散/FM 多采样。可以蒸馏扩散到少步给部署。

**Q：真机 RL 你怎么做？环境交互太贵怎么办？**
> 先 BC 起步；仿真大规模 RL + 域随机；真机只做短程 fine-tune 和安全约束校验；世界模型 rollout 做廉价 candidate 筛选——和我们 WAM 路由一致。

**Q：没有人类偏好标注，怎么做对齐？**
> 规则奖励（成功、安全、时间）+ AI judge + 拒绝采样；驾驶里 PDMS 就是天然 RM。主观「自然度」才更需要人类。

**Q：指令有歧义/冲突（「加速」但前方有人）优先级？**
> 安全约束硬屏蔽 > 任务指令；结构上可用安全头/fallback 覆盖，而不是指望 LM 自己领悟——对应可信路由分层。

**Q：多任务 VLA 怎么防任务混淆？**
> 任务 token/语言显式条件、数据按任务配比、per-task adapter 或专家、评测分任务回归。

**Q：你们的模型怎么上真车/真机的延迟？**
> 给预算表：编码 xx ms + MoT/策略 xx ms + Flow K 步 xx ms；再讲量化与步数蒸馏、异步感知、执行前 K 步。

**Q：如果让你重新做，数据会怎么设计？**
> 长尾导向的主动采集/仿真、失败案例回放池、指令多样性、多车多城去重、自动质量分——从「有多少采多少」变成「缺什么造什么」。

---

## 8. 本部分自测清单

- [ ] 预训练三大自监督目标能各写一行 loss？
- [ ] 后训练 Stage0-4 流水线和三项目对照表？
- [ ] SFT 数据从哪来？拒绝采样流程？
- [ ] DPO/PPO/GRPO 对比表能否不看稿说全？
- [ ] 为什么轨迹选 GRPO 不选 DPO？
- [ ] RL 工程 10 条清单（采样-奖励-优势-更新-监控）？
- [ ] 灾难性遗忘 / reward hacking 各一个具体解法？
- [ ] VLA 动作表示 6 种方案表？
- [ ] Action chunk 为什么、执行前 K 步？
- [ ] Diffusion Policy vs ACT vs FM 一句话差异？
- [ ] MoT vs MoE；π₀ vs 我们的 MOT？
- [ ] 从 0 设计 VLA 的 10 步模板？
- [ ] 数据/遥操作/DAgger 能聊 2 分钟？
- [ ] 具身 vs VLA 岗位定位一句话？
- [ ] 2-4 周计划是否写进日历？

---

*全文复习建议顺序：第一部分名词 -> 第二部分手写 FM/GRPO -> 补章话术（含比亚迪 MOT）出声背 -> 第四部分预训练/后训练 -> 第五部分岗位对齐与 mock -> 第三部分 40 题查漏。*

---

# 第六部分：三大原理重点精讲（第一人称）—— VLA 统一建模 / 世界模型 / 世界动作模型

> 面试里最容易被要求「你把这三者的原理给我讲透」的，就是 **VLA、WM、WAM**。本部分按「先串关系 → 再逐个原理 → 再横向对比 → 再口述稿 → 再追问」组织，全部按**第一人称**写，可直接当讲稿背，也可按层次深浅现场裁剪。

---

## 0. 开场 60 秒：三者是什么关系（先把地图给面试官）

> 「我先用一分钟把 VLA、世界模型、WAM 三者的关系说清楚，这样后面细节不会散。
>
> **VLA 回答的是『我该怎么动』**——看到什么、听到什么指令，直接吐出动作或轨迹，是 **policy（策略）**。
>
> **世界模型 WM 回答的是『我动了之后世界会怎样』**——给定状态和动作，预测下一状态 `p(s'|s,a)`，是 **model（环境模型）**，不负责做决策，只负责想象后果。
>
> **WAM 把这两件事拧成一股**——在同一个模型、同一条生成流里，**一边生成动作，一边同步演化后果、风险和可信度**，所以叫 World **Action** Model：动作和世界不再割裂成『先规划、再预测』两个模块。
>
> 用开车类比：
> - **VLA** 是司机的手：看路打方向；
> - **WM** 是司机脑内小电影：『我若现在变道，旁车会怎样』；
> - **WAM** 是司机本人：打方向的**同时**脑子里小电影已经在放，而且他知道自己**什么时候该信这个想象、什么时候该回退到保守开法**。
>
> 我做过的链路是：比安迪 MOT 是『理解+生成』的统一底座；RiskField-VLA 是带风险条件的 **VLA policy**；JEPA 是**表征空间 WM**；TrustDrive-WAM 是 **动作-后果联合的 WAM + 可信路由**。三者我都能落到代码和训练目标上讲。」

**一句话索引（黑板可写）：**

```text
VLA:  p(a | o, lang)                 策略: 观测+指令 -> 动作
WM :  p(s' | s, a)                   模型: 状态+动作 -> 下一状态(或表征)
WAM:  p(a, conseq, risk, trust | c)  联合: 条件下动作与后果同时生成
关系:  VLA 只出动作; WM 只预测; WAM = 动作生成 ⊗ 后果演化 ⊗ 可信度
用法:  WM 可给 VLA 做想象/奖励/筛选; WAM 把这条边焊进生成过程内部
```

---

## 1. VLA 统一建模原理（第一人称精讲）

### 1.1 我怎么给 VLA 下定义（30 秒）

> 「VLA，Vision-Language-Action，本质是**把三种模态统一进同一个条件生成问题**：
> **输入**是视觉观测 `o`（多相机/深度/BEV）和语言指令 `l`（导航或任务）；
> **输出**是动作 `a`——可以是离散 token，也可以是连续轨迹块 `a_{t+1..t+T}`。
> 形式化就是：**`p_θ(a | o, l, s_ego)`**。
> 所谓『统一建模』，我理解有四层含义，面试我会按这四层展开，而不是只背一个名字。」

### 1.2 「统一」的四层原理（核心，必背骨架）

#### 第一层：模态统一 —— 都变成 token，进同一个 Transformer

> 「第一层统一是**表征层的统一**。视觉变成 patch token，语言变成 BPE token，状态/动作变成连续 embedding 或量化 token。它们投影到**同一个 hidden space**，用同一套 attention 交互。
>
> 这不是简单拼接：拼接只保证『能进同一个网络』，统一建模要求网络**学会模态间的对齐关系**——『前方红灯』这句话要能点亮图像里那个红灯区域的语义，『左转』要能约束轨迹头往左弯。
>
> 实现上我见过/用过三种注入方式，我会主动对比：
> 1. **早融合拼接**：所有 token 进一条序列（LLaVA 风格）——简单，序列长；
> 2. **cross-attention 注入**：动作头用 Q 查视觉/语言 K/V（π₀ 动作专家、我们 gen→und）——解耦，保护 backbone；
> 3. **双塔/分路径**：MoT 那种 und/causal 与 gen/bidirectional 分权——模态统计特性差太大时用。
>
> 我在比亚迪 MOT 里就是第 3 种 + 第 2 种的组合：**Reasoner 统一理解视觉和语言，Generator 统一生成，层间 cross-attn 对齐**。」

```python
# 统一到一条序列 (概念)
def unify_modalities(vision_patch, text_token, state_vec):
    v = vision_proj(vision_patch)          # (B, Lv, D)
    t = text_proj(text_token)              # (B, Lt, D)
    s = state_mlp(state_vec)               # (B, 1, D)  连续->D 维
    s = s.unsqueeze(1) if s.dim()==2 else s
    # 拼成一条, 用 token type / mask 标明来源
    seq = torch.cat([t, v, s], dim=1)      # (B, Lt+Lv+1, D)
    type_emb = cat([type_text, type_vision, type_state])
    return seq + type_emb                  # 统一 hidden 空间
```

#### 第二层：任务统一 —— 感知、理解、决策是一个生成问题，不是三个网络

> 「第二层是**任务层的统一**。传统栈是检测网 + 预测网 + 规划优化器，接口是框和规则。VLA 把它们收成**一次条件生成**：看见的和听懂的都是条件 `c=(o,l)`，规划输出只是条件下的一个样本 `a ~ p(a|c)`。
>
> 好处我会讲三点，面试官很吃这种结构化回答：
> 1. **信息无损**：中间不压成『检测框列表』，纹理、遮挡、意图梯度能进 loss；
> 2. **联合梯度**：动作 loss 可以反传到视觉编码器，迫使视觉学『对决策有用』的特征，而不是只刷 IoU；
> 3. **一个评测闭环**：最终指标就是任务成功率/PDMS，不存在『检测很准规划很烂』的模块扯皮。
>
> 代价我也主动说：黑盒、数据饥渴、开环-闭环 gap——所以后面才需要世界模型和可信域补安全论证。」

#### 第三层：动作统一 —— 用「块」和「分布」而不是单步点估计

> 「第三层是**动作表示的统一**。VLA 业界已经从『每步吐一个点』收敛到：
> **输出一个动作块（action chunk）或轨迹块** `A_t = {a_{t+1},...,a_{t+T}}`，
> 并且输出的是**分布/可采样生成器**，不是回归均值。
>
> 为什么必须是分布：驾驶和操作都天然多模态——同一场景可以『让’也可以『抢』，回归到均值会得到两头不靠的鬼影轨迹。所以动作头要么离散自回归（OpenVLA），要么扩散/RDT，要么**流匹配**（π₀、我们的项目）。
>
> 我统一用的说法是：
> **策略 = 条件生成模型；动作 = 从噪声到演示分布的一次流式解码。**
>
> 部署上再加 receding horizon：一次生成 T 步，**只执行前 K 步**，拿新观测重新生成——闭环修正误差累积。」

```python
# 第三层原理的最小代码骨架
def vla_policy_forward(obs, lang, state):
    # 1) 模态统一编码
    c = unify_and_fuse(obs, lang, state)     # 条件向量/序列
    # 2) 动作块生成: FM / diffusion / AR 三选一
    x = torch.randn(B, T, act_dim)           # 噪声动作块
    for i in range(steps):                   # 流/扩散迭代
        t = full(B, i/steps)
        v = action_head(x, t, c)             # 速度场
        x = x + v * dt
    # 3) 执行前 K 步, 不是全部执行完再反应
    return x[:, :K]                          # receding horizon
```

#### 第四层：训练统一 —— 预训练懂世界，SFT 会任务，RL 对齐奖励

> 「第四层是**训练目标的统一**。VLA 不是单个 loss 喂大的，我按流水线讲：
> 1. **预训练**：VLM 的语言 CE + 视觉对齐，让模型『懂场景』；可选视频/JEPA 学动力学；
> 2. **SFT/BC**：演示轨迹上做 Flow MSE 或动作 token CE，让模型『会出合法动作』；
> 3. **后训练**：GRPO/DPO 用任务奖励拉长尾——『出的动作要高分，不只是像人类』。
>
> 这三层 loss 形式不同，但**参数空间和条件接口是同一个**——这就是训练维度的统一。我做过的 Flow-GRPO 正是第三层：不改感知语义，只在动作生成头上用组内相对优势做 PPO-clip。」

### 1.3 VLA 统一建模的「不统一」在哪（主动暴露边界，加分）

> 「我也会主动说清楚什么**没有**被统一掉：
> - 时间尺度：视觉可以 10Hz，安全监控要 100Hz，不能全塞进一次生成；
> - 概率 vs 规则：碰撞检查、 RSS 这类硬约束不适合纯靠采样，需要输出后验算/fallback；
> - 计算：视频生成头和车端延迟预算冲突，所以量产常是『理解+小策略头在车端，大世界模型在云端』。
> 所以我的立场是：**统一建模统一的是表征和主生成路径，安全与实时仍要外层工程兜住**——这也是我做可信路由的原因。」

### 1.4 VLA 原理第一人称完整口述稿（90 秒）

> 「VLA 的统一建模，我拆成四层讲。
>
> **第一层模态统一**：视觉 patch、语言 token、状态向量投到同一 hidden space，用同构 attention 交互——可以早融合拼接、动作头 cross-attn 查视觉，或 MoT 这种双路径分权；关键不是拼得进去，而是让『左转』指令能约束到轨迹、『红灯』能约束到速度。
>
> **第二层任务统一**：感知和规划不再是两个网络加接口，而是一次条件生成 `p(a|o,l)`，动作 loss 能反传改视觉，评测只看最终任务分。
>
> **第三层动作统一**：输出是**动作块的分布**不是单步点；多模态用生成模型解，回归均值会出鬼影；部署只执行前 K 步再 replan。
>
> **第四层训练统一**：预训练学世界、SFT 学任务格式、GRPO/DPO 对齐奖励，三阶段同一套参数和条件口。
>
> 我的项目都能挂上去：MOT 是第一层的双塔实现；RiskField-VLA 是第二、三层——风险场进条件、FM 出 8 条轨迹块；Flow-GRPO 是第四层。」

---

## 2. 世界模型 WM 原理（第一人称精讲）

### 2.1 定义（30 秒，先给公式）

> 「世界模型，一句话：**一个学出来的环境动力学**——
> `p_φ(s_{t+1} | s_t, a_t)`，
> 输入当前状态和动作，输出下一状态（或其分布/表征）。
> 它**不产生策略**，不告诉你『该不该变道』，只回答『你若变道，下一刻世界长什么样』。
> 有 WM 之后，智能体可以先在脑子里 rollout，再决定执行哪条——这是它和 VLA 的本质分界。」

### 2.2 为什么需要 WM（三个用途，面试按这个答）

> 「我会先讲用途再讲结构，这样不空：
>
> **用途一：想象/规划（imagination）**。在 WM 里虚拟执行多条动作，看谁的后果好——model-based planning。没有 WM 就只能真去环境里试错，真车/真机试不起。
>
> **用途二：更便宜的奖励与数据**。WM rollout 可以当 RL 的廉价环境；也可以反向筛数据、合成难例。我们 NAVSIM 开环打分本质上是一个**非学习的、规则的伪世界模型+评分器**；JEPA 是把『预测未来』学成表征，给下游当先验。
>
> **用途三：表征与泛化**。学 `p(z'|z,a)` 会逼模型抓**物理因果**（谁推谁、遮挡谁消失），而不是背景纹理——这是 JEPA 路线相对像素重建的核心卖点。
>
> 反过来说，**WM 单独不能开环上路**：它没有偏好、没有任务目标，只预测不决策。所以工业界永远是 WM + policy 组合。」

### 2.3 WM 的三种形态（原理分型，必背）

| 形态 | 预测什么 | 优点 | 缺点 | 代表 |
|------|----------|------|------|------|
| 像素 WM | 未来帧 RGB | 直观、可人眼看 | 算力爆炸、学无关纹理、易穿模 | DriveDreamer, GAIA, Sora 类 |
| 几何/占据 WM | 未来点云、占据格 | 贴近规划接口 | 模态少、丢语义 | occupancy 预测 |
| **表征 WM** | 未来隐向量 `z'` | 快、聚焦语义、可接任何头 | 无显似然、需防坍缩、要任务头对齐 | **JEPA / Dreamer latent** |

> 「我主推第三种，原理上一句话：
> **在嵌入空间预测未来，把『画出来』的算力换成『理解会怎样』的算力。**
> 多模态未来（路口可左可右）在像素回归里平均成鬼影，在表征空间却可以保留多个模式的条件分支。」

### 2.4 表征世界模型的数学骨架（JEPA/我做的）

```text
编码:   z_t     = Enc(x_t)                  可见上下文/当前帧
目标:   z_{t+1}* = sg( EMA_Enc(x_{t+1}) )   目标编码, stop-grad + 慢更新
预测:   ẑ_{t+1} = Predictor(z_t, pos/time)   只预测表征, 不重建像素
损失:   L = || ẑ_{t+1} - z_{t+1}* ||²       特征空间 MSE

防坍缩三件套:
  stopgrad:  目标端不接收梯度, 不能「配合」输出常数
  EMA:       目标端慢半拍, 必须「被昨天的自己」预测
  VICReg:    方差下限+协方差去相关, 几何上禁止点云塌缩
```

```python
def world_model_step(enc, ema_enc, pred, x_now, x_next, mask):
    z_ctx = enc(observe(x_now, mask))           # 当前可见 -> 上下文
    with torch.no_grad():
        z_tgt = ema_enc(x_next)                 # 未来帧 -> 目标表征
    z_hat = pred(z_ctx, time_or_pos_cond)       # 预测未来表征
    return F.mse_loss(z_hat, z_tgt)
```

### 2.5 WM 与「轨迹预测网络」的区别（高频陷阱题）

> 「很多面试官会拿这个考基础，我会主动切开：
>
> - **他车轨迹预测**：条件是感知历史，输出是**其他 agent** 的轨迹分布，通常不闭环到自车动作对世界的影响；
> - **世界模型**：条件含**自车/自体动作 a**，输出是**整个系统状态**（含自己）的演化，支持反事实『若我动作不同』，可以多步 rollout。
>
> 关键判据就一条：**输入里有没有动作、输出能不能被动作改变、能不能连续推多步**。只预测别人、不建模自己动作的，严格说是 prediction module，不是完整 WM。」

### 2.6 WM 为什么危险：OOD 与误差累积（原理 + 安全）

> 「WM 原理上还有两个必须主动讲的缺陷，否则显得只会吹：
>
> **1) 分布外不可信**。神经 WM 对没见过的天气/车型仍输出自信预测，policy 盲信就出事。原理层解法是给 WM 配**支持域/能量分数**——训练分布内能量低，OOD 能量高，超出阈值不许当真。
>
> **2) 多步 rollout 误差指数累积**。训练是单步 `p(s'|s,a)`，推理却把预测当下一步输入——小偏差在 10 步后放大到不可用（compounding error）。缓解：短 horizon、闭环校正、或用真实观测周期性重置（我们执行前 K 步就隐含了对 WM/policy 的周期重置）。」

### 2.7 WM 原理第一人称口述稿（90 秒）

> 「世界模型我讲四点：定义、用途、分型、缺陷。
>
> **定义**是学出来的动力学 `p(s'|s,a)`——给动作预测世界，不给决策。
> **用途**三块：想象 rollout 做规划；当廉价环境出奖励和合成数据；逼表征学物理因果。
> **分型**上像素 WM 太贵还易穿模，几何 WM 接口窄，我更认**表征 WM**：JEPA 式在嵌入空间预测未来，loss 是特征 MSE，靠 EMA+stopgrad+VICReg 防坍缩——这就是我 JEPA-DRIVE 的 Stage1。
> **缺陷**必须说：OOD 不可信、多步误差累积。所以 WM 不能直接当安全依据，要支持域门控 + 短闭环。
> 和轨迹预测的区别：WM 输入必须含动作、输出是含自己的完整状态、能反事实多步推演。
> 我和它的接口：JEPA 出 `z'`，WAM 把动作和后果放进同一条流，NVSIM 规则打分其实是非学习的伪 WM。」

---

## 3. WAM 世界动作模型原理（第一人称精讲）

### 3.1 定义（30 秒，和 WM 划清）

> 「WAM，World **Action** Model。如果说 WM 是『只预测、不负责出招』，WAM 就是**把出招和算招焊死在同一个生成过程里**。
>
> 形式上我不再只建 `p(s'|s,a)`，而是建：
> `p_θ(a, s', r, τ | c)`——
> 在场景条件 `c` 下，**动作块 `a`、后果 `s'/后果特征`、风险 `r`、可信域指示 `τ` 联合生成**。
>
> 所以它相对 WM 多出来的，不是多接了一个分类头，而是**联合（joint）**：四个东西共享同一条流的时间 `t`，同一时刻的『动作与后果』天然配对。」

### 3.2 为什么必须「联合」——原理论证（面试最爱追问的一步）

> 「如果拆成两步：先 VLA 出轨迹，再 WM 打分——会出三个结构性问题，我逐个说：
>
> **问题 A：表征错位**。轨迹生成器和后果预测器各自训练，第 t 步的『动作特征』和第 t 步的『后果特征』不在同一语义层，打分头要重新对齐，信息在接口处丢失。
>
> **问题 B：生成时瞎**。生成阶段不知道后果，只能事后 filter；高风险方向已经浪费了采样与容量，长尾上等于『先犯错再检讨』。
>
> **问题 C：可信度没有位置**。『我有多确定』是另一个模型事后估的，和生成动力学脱节，路由阈值难标定。
>
> WAM 的原理级回答是：**把动作、后果、风险、可信域放进同一个向量场联合积分**——
> `z = [a | conseq | risk | trust]`，
> `z_t = (1-t)ε + t z_1`，
> `v_θ(z_t, t, c)` 一次预测整段速度。
> 这样：(1) 同 `t` 配对，无接口错位；(2) 生成中途就能读到风险，可早停可引导；(3) 可信域与动力学同演化，路由读同一状态。
>
> 这就是我认为 WAM 相对『VLA+外挂 WM』的本质增量——**不是多了个模块，是换了联合分布。**」

```python
# WAM 原理的最小实现形状
TRAJ, CONQ, RISK, TRUST = 60, 32, 16, 8

def pack(a, c, r, t):
    return torch.cat([a, c, r, t], dim=-1)     # 联合状态 z

def wam_train(z1, cond):
    z0 = torch.randn_like(z1)
    t  = torch.rand(B, 1, device=z1.device)
    zt = (1 - t) * z0 + t * z1                 # 联合直线路径
    v  = joint_velocity_net(zt, t, cond)       # 一个向量场
    return F.mse_loss(v, z1 - z0)              # 整段一起回归

def unpack(z):
    return z[..., :60], z[..., 60:92], z[..., 92:108], z[..., 108:]
```

### 3.3 WAM 原理的四条支柱（展开讲的目录）

> 「我会把 WAM 拆成四条支柱，面试官如果只让讲两分钟，我讲前两条；让深挖，四条全上。

**支柱一：联合动作-后果流（Joint Flow）**
> 动作与后果共享 ODE/SDE 时间轴；损失按分量加权但**同一个 `t`**。生成 = 同时『开出一条轨迹』和『写下这条轨迹的代价卡片』。

**支柱二：反事实与逆一致性（Counterfactual + Inverse Consistency）**
> WM 要支持『换动作会怎样』，所以动作信息必须真的进表征。若只监督 `s,a→s'`，网络可以忽略 `a` 也能拟合平均运动。加逆一致性：`s,a→s'` 再 `s',a→s_hat`，要求闭环还原——逼着模型使用 `a`。原理上这和 CycleGAN cycle loss 同构。

**支柱三：风险与支持域同演化（Risk + Support Co-evolution）**
> 风险不是事后算的碰撞布尔，而是与轨迹同流生成的连续量；支持域/能量告诉我们『这一步预测是否在训练分布内』。二者都是 `z` 的分量，**路由读的是生成状态，不是另一个网络的黑盒输出**。

**支柱四：可信路由（Trust-aware Control Barrier）**
> 双层：`可靠性 = support>τ_s and risk<τ_r`；`决策增益 = U_wm > U_base`。
> 通过则用 WAM 轨迹，否则回退经典规划。原理定位：**WM/WAM 的输出是证据不是裁决**——这是把『不完美的世界模型』放进行驶回路的安全哲学。

### 3.4 WAM vs WM vs VLA 对比表（黑板级）

| 维度 | VLA | 传统 WM | **WAM** |
|------|-----|---------|---------|
| 核心问题 | 我怎么动 | 动了世界怎样 | **动与后果一起长出来** |
| 输出 | 动作 `a` | 下一状态 `s'` | **`a, 后果, 风险, 可信` 联合** |
| 是否决策 | 是 | 否 | **是（动作侧）** |
| 是否预测 | 否（或隐式） | 是 | **是（后果侧）** |
| 训练 loss | 演示 CE/FM | 动力学 MSE/像素 | **联合 FM + 逆一致 + 排序** |
| 安全角色 | 策略本身 | 预测先验 | **策略 + 自评可信 + 可回退** |
| 模块形态 | 单策略 | 单模型 | **联合分布 + 路由** |
| 我的对应 | RiskField-VLA / MOT Flow Head | JEPA-DRIVE | **TrustDrive-WAM** |

### 3.5 一个具象例子（把原理讲活，面试加分）

> 「旁车贴着我并线这个场景，我会用三个模型各走一遍，证明我懂差别：
>
> - **纯 VLA**：看见加塞，条件里有风险的话出一条避让轨迹——它**不知道**避让后自己会不会压实线，除非训练里见过；
> - **外挂 WM**：先出 8 条轨迹，再逐条预测『压实线概率』，filter——**两阶段接口**，若 WM 与轨迹头特征不对齐会误杀好轨迹；
> - **WAM**：同一条流里，避让轨迹和『线外风险上升、舒适下降、支持域仍在』同时长出来；当 risk 与 trust 分量显示『这次预测偏 OOD』，路由直接回退保守跟车。
>
> 同一个场景，WAM 的增量是**把『后果卡片』和『方向盘』写在同一张草稿纸上，而且草稿纸边上印着『这页可信度』**。」

### 3.6 WAM 原理第一人称口述稿（90 秒）

> 「WAM 我按『一句定义、一条增量、四条支柱』讲。
>
> **定义**：World Action Model——联合分布 `p(a, 后果, 风险, 可信 | 条件)`，动作和世界演化在同一个生成流里。
>
> **相对 WM 的增量不是加头，是改分布**。拆开的 VLA+WM 有表征错位、生成时瞎、可信度无位置三个问题；联合流让同一时间 `t` 的动作和后果天然配对，生成中途就能读风险和信任分量。
>
> **四条支柱**：一是联合流，分量共享向量场；二是动作必须进表征——逆一致性闭环防『忽略 a』；三是风险和支持域与轨迹同演化；四是双层可信路由——support 和风险过门才用 WAM，否则回退经典规划，哲学上 **WAM 输出是证据不是裁决**。
>
> 对比：VLA 只出招，WM 只算招，WAM 是边出招边算招还标可信度。我 TrustDrive-WAM 的 PDMS 0.8012，消融掉路由或掉逆一致都有明显回退——原理和实验是对得上的。」

---

## 4. 三者串讲：从原理到我的项目地图（60–120 秒）

> 「最后我用 30 秒把三者收进一条工程线，方便您定位我的经验：
>
> **比亚迪 MOT** 是统一底座：Reasoner 理解视觉语言，Generator 用流生成——这是 VLA『模态统一 + 任务统一』的架构实现；
> **RiskField-VLA** 是策略层：风险场进条件，FM 出轨迹块，Flow-GRPO 后训练——VLA 的动作统一与训练统一；
> **JEPA-DRIVE** 是世界模型层：表征空间预测未来，防坍缩三件套——纯 WM，不碰决策；
> **TrustDrive-WAM** 是耦合层：动作-后果联合流 + 逆一致 + 可信路由——WAM 原理的完整落地。
>
> 所以如果问『你更懂 VLA 还是世界模型』，我的答案是：**VLA 和 WM 我都单独做过，WAM 是我把两者在原理上焊在一起的那一段**——这也是我认为具身/VLA 岗位下一步真正要补的安全与反事实能力。」

**黑板最终版（可画可写）：**

```text
        统一建模 (VLA)                 环境想象 (WM)
  o,l ------------------> a      s,a ------------------> s'
  模态/任务/动作/训练统一           像素/几何/表征 三型
                 \                   /
                  \                 /
                   v               v
              WAM 联合:  p(a, 后果, 风险, 可信 | c)
                   |
                   +--> 可信路由: 支持域内用 WAM, 否则 fallback
                   
我的对应:  MOT/VLA=上排 | JEPA=WM | TrustDrive-WAM=下排
```

---

## 5. 追问链预演（VLA / WM / WAM 专版）

**链 A：VLA 统一**
1. 统一是不是就是 all-in-one 网络？→ 不是，是模态/任务/动作/训练四层，外加承认非实时硬约束不硬塞。
2. 语言到底怎么进动作头？→ 拼接 embedding / cross-attn / 作为 Flow 条件 `c` 的一项；可测 left/right 敏感度。
3. 为什么 chunk 不是越长越好？→ 误差锁定更久，replan 延迟与平滑 trade-off。
4. 离散 token 和连续流怎么选？→ 码本精度与实现 vs 多峰与低步数；RDT-2 是训练离散、推理连续的折中。
5. 和检测头多任务会不会互相撕？→ 分 loss 权重、分阶段解冻、或 MoT 分路径隔离梯度。

**链 B：WM**
1. 为什么不用像素 WM？→ 算力、鬼影、接口——表征 WM 更贴决策。
2. 怎么证明学到物理而不是纹理？→ linear probing、换背景/纹理增广、计数与深度辅助探针、反事实动作敏感度。
3. 多步 rollout 崩了怎么办？→ 短 horizon、周期用真观测重置、能量高就停止想象。
4. 和 Diffusion 世界模型关系？→ 生成骨干可同构，差别在输出空间（像素 vs z）与是否要解码。
5. 训练要动作标签吗？→ 要 `a_t` 与 `s_{t+1}` 对齐；日志里就是控制/轨迹字段。

**链 C：WAM**
1. 不就是多任务学习吗？→ 多任务是并行头，WAM 强调**共享流时间与中间状态可读、可路由**。
2. 逆一致会不会学成恒等？→ forward 另有动力学监督 + stopgrad，不允许双端自由拟合。
3. 路由阈值谁定？→ 验证集信任率-风险曲线拐点 + 故障注入。
4. 推理变慢多少？→ 联合向量维度从 60 到 116，相对 backbone 可忽略；真正开销在采样步数。
5. 和 model-based RL 的 Dreamer 差在哪？→ Dreamer 在 latent 里训 value/策略；我们是条件生成 + 规则奖励后训练 + 安全路由，对齐开环 PDMS，不维护 value 网络。

---

## 6. 本部分 60 秒自测（不看稿能否答出）

- [ ] VLA 统一建模的四层各是什么？
- [ ] `p(a|o,l)` 和 `p(s'|s,a)` 写错没有？
- [ ] WM 三大用途 + 三形态？
- [ ] 表征 WM 防坍缩三件套？
- [ ] WM vs 轨迹预测的判据（动作在不在输入）？
- [ ] WAM 为什么必须联合（三个结构性问题）？
- [ ] 四条支柱名字能否脱口而出？
- [ ] VLA / WM / WAM 对比表能否 30 秒扫完？
- [ ] 加塞例子能否 30 秒讲完三个模型差异？
- [ ] 我四个项目分别挂在哪一层？

---

*本部分与第二补章话术互补：话术管「怎么开口」，本部分管「原理讲多深」。面试若只给 1 分钟，用各节 30/90 秒稿；若给 3 分钟以上，按四层/三用途/四支柱展开，并主动抛对比表和加塞例子。*

---

# 第七部分：基础补强 —— ML 八股 / 数学概率 / 手写题 / 论文深挖

> 按「1–2 轮几乎必问」到「看目标」排序。自驾/具身技术面很少纯 LeetCode Hard，但 **ML 基础 + 概率** 几乎每场都有；手写题准备**高频中等**即可。

---

## 1. ML 基础八股（1–2 轮几乎必问）

### 1.1 反向传播（必考，建议能口述链式法则）

**一句话**：前向算 loss，反向用链式法则把 `∂L/∂θ` 一路乘回去，再用优化器更新 `θ`。

```text
例: y = Wx + b, L = (y - t)^2
前向:  z = Wx + b,  L = (z-t)^2
反向:
  dL/dz = 2(z-t)
  dL/dW = dL/dz * x          # 矩阵形状: 外积
  dL/db = dL/dz
  dL/dx = dL/dz * W          # 继续传给上一层
更新:  W = W - lr * dL/dW
```

**面试常追问：**
- 为什么需要激活函数？→ 不加则多层线性仍等价一层线性，没有非线性拟合能力。
- 梯度消失/爆炸？→ 深层连乘：激活导数 <1 或权重过大时指数衰减/放大；解法：ReLU 系、残差、LN、合理初始化、梯度裁剪。
- 梯度裁剪为什么常用 `clip_grad_norm` 而不是 value？→ 按范数整体缩放，保持方向，对 Transformer/RL 更稳。

### 1.2 归一化：BN vs LN vs RMSNorm（VLM/Transformer 必问）

| | BatchNorm | LayerNorm | RMSNorm |
|---|-----------|-----------|---------|
| 沿哪维归一 | batch 维 (N,H,W…) | 特征维 (C 或 D) | 特征维 |
| 依赖 batch？ | **是**（推理用 running 统计） | 否 | 否 |
| 适合 | CNN、大 batch | RNN/Transformer | LLaMA 系、省算力 |
| 推理训练一致？ | 需 eval 模式 | 一致 | 一致 |

```python
# LayerNorm: 对每个样本的 D 维做标准化再仿射
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

# RMSNorm: 去掉均值中心化，只除 RMS，更快
def rms_norm(x, gamma, eps=1e-6):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * gamma
```

**为什么 Transformer 用 LN 不用 BN？** 序列长度不齐、batch 间 token 分布不稳；BN 会把不同样本的同一位置混在一起统计，破坏序列独立性。LN/RMSNorm 只看当前 token 自己的特征维。

**Pre-LN vs Post-LN**：Pre-LN（先 LN 再 attn/mlp，残差更干净）训练更稳、是现代默认；Post-LN 原始 Transformer 需要 warmup。MOT 的 `MoTDecoderLayer` 是 Pre-LN 风格。

### 1.3 优化器（SGD → Adam → AdamW）

```text
SGD:   θ = θ - lr * g
Momentum:  v = β v + g ;  θ = θ - lr * v     # 冲量, 抗噪声
Adam:  同时估一阶动量 m 和二阶动量 v_t
       m = β1 m + (1-β1) g
       v = β2 v + (1-β2) g^2
       m̂ = m/(1-β1^t), v̂ = v/(1-β2^t)        # 偏差修正
       θ = θ - lr * m̂ / (sqrt(v̂)+eps)
AdamW: 权重衰减从梯度里拆出来直接作用在 θ 上
       θ = θ - lr * (m̂/(sqrt(v̂)+eps) + λθ)  # 解耦衰减, 比 L2 更对
```

| 超参 | 常见值 | 作用 |
|------|--------|------|
| lr | pretrain 1e-4~3e-4; SFT 稍小; RL 更小 1e-5 级 | 步长 |
| β1, β2 | 0.9, 0.95 或 0.999 | 动量时间尺度 |
| weight decay | 0.01~0.1 | 防过拟合 |
| warmup | 1%~3% steps | 防早期大梯度打飞 |
| grad clip | 1.0 | 防爆 |

**为什么 RL 后训练 lr 要更小、常要 warmup？** 组内采样 already  off-policy + KL 约束，大 lr 容易 KL 爆、clip 比例飙升。

### 1.4 过拟合 / 欠拟合 / 正则（几乎必问）

```text
症状:
  train 高 val 低  -> 过拟合
  train val 都低   -> 欠拟合 (容量不够/没训够/特征差)

过拟合手段 (按性价比):
  1. 加数据 / 数据增强 / 去重防记答案
  2. 早停 (early stopping), 监控 val 曲线
  3. Dropout (Transformer 里常 0.0~0.1, LLM 预训练常 0)
  4. Weight decay / label smoothing
  5. 减容量: 少层/窄 hidden/低 rank (LoRA r 调小反了是更小容量)
  6. 数据配比: 防止单源刷爆
  7. EMA 权重做评测

欠拟合手段:
  加容量、加训练步数、调 lr、检查 label 是否对齐、检查 mask/pad 泄漏
```

**诊断三件套**（面试可背）：画 train/val 分项 loss；固定 seed 过拟合 1 个 batch 看能否到 0（管线通不通）；查数据（标签错位、重复、pad）。

### 1.5 损失函数速查

| 损失 | 公式直觉 | 用在哪 |
|------|----------|--------|
| MSE | `(pred-gt)^2` | 轨迹回归、Flow 速度场、JEPA |
| MAE/L1 | 绝对值 | 对 outlier 更稳 |
| CE | `-log p(correct)` | 语言 token、分类 |
| Focal | `(1-p)^γ * CE` | 极端正负样本不均 |
| Contrastive | 拉近正、推远负 | SimCLR/CLIP |
| Huber | 平滑 L1/L2 | 轨迹含噪声时 |
| Ranking/Listwise | 对齐序而非绝对值 | 我们 PDMS 评分头 |
| PPO clip | `-min(ratio A, clip A)` | GRPO 后训练 |

**为什么轨迹有时用 Huber 不用 MSE？** 少数离群标注（打滑、定位跳）在 MSE 下梯度过大，Huber 尾部线性更稳。

### 1.6 Dropout / Label Smoothing / 激活

```text
Dropout:  训练随机置零, 集成多种子网络; 推理关闭并 scale
Label smoothing: 把 one-hot -> (1-ε) + ε/K, 防对错过度自信, LLM 常 0.1
ReLU: max(0,x), 会死神经元;  GELU/SiLU 平滑版, 现代 LLM/DiT 常用
SwiGLU: LLaMA FFN 用, 门控, 比 GELU 略强但参数结构不同
```

### 1.7 Dropout 和「训练/推理不一致」检查清单

```text
[ ] model.train() / model.eval() 切对了吗
[ ] BN 的 running stats 更新了吗
[ ] 扩散/流采样步数 train 和 eval 一致吗
[ ] 数据增强 eval 是否误开
[ ] dropout 在 RL inner epoch 是否造成 log_prob 噪声过大
```

### 1.8 梯度检查点 / 混合精度 / 显存（工程加分）

```text
AMP/FP16/BF16:  前向反向低精度, master 权重 FP32 更新
BF16 比 FP16 少溢出问题, A100/H100 上 LLM 常用
Gradient checkpointing: 前向只存边界激活, 反传重算, 省显存换时间
梯度累积:  小 batch 模拟大 batch
梯度同步:  DDP 每 step all-reduce; FSDP 分片参数
```

---

## 2. 数学 / 概率（研究岗 + 大厂基础轮）

### 2.1 期望方差与协方差

```text
E[X] = sum p(x) x          离散;  积分连续
Var(X) = E[(X-μ)^2] = E[X^2] - μ^2
Std = sqrt(Var)
Cov(X,Y) = E[(X-μx)(Y-μy)]
Corr = Cov/(σx σy)  in [-1,1]

和的期望: E[X+Y]=E[X]+E[Y]
独立时和的方差: Var(X+Y)=Var(X)+Var(Y)
线性: E[aX+b]=aE[X]+b;  Var(aX+b)=a^2 Var(X)
```

**和我们的联系**：GRPO 组内 advantage 用均值方差归一化——本质是把 reward 变成组内标准化随机变量，期望 0 方差 1，稳定不同场景量纲。

### 2.2 高斯分布（Flow/GRPO 的地基）

```text
一元:  N(μ,σ²),  pdf = 1/sqrt(2πσ²) * exp(-(x-μ)²/(2σ²))
log pdf = -0.5*log(2π) - log σ - (x-μ)²/(2σ²)

多元对角:  N(μ, Σ),  Σ=diag(σ²)
log N(x) = -0.5 * sum_i [(x_i-μ_i)²/σ_i² + log σ_i²] + const

采样:  x = μ + σ ⊙ ε,  ε~N(0,I)   (重参数化, 可反传)
N(0,I) + 线性变换 = 高斯
两个高斯乘积仍是高斯 (共轭), 条件高斯有闭式
```

**手写题预警**：面试可能让你写高斯 log-prob 或采样——见 Flow-GRPO 代码段。

### 2.3 MLE / MAP / CE 的关系（研究岗爱问）

```text
MLE:  θ* = argmax log p(D|θ)
     等价于最小化负对数似然
分类 softmax+CE = 多项分布的 MLE
MSE  = 高斯噪声假设下的 MLE (方差固定)

MAP:  argmax log p(D|θ)+log p(θ)
     先验项 -> 等价正则 (L2 <-> 高斯先验)

生成模型两大类:
  似然法:  自回归、VAE、流 (可算 bound/似然)
  非似然/得分法:  扩散/对比, 学 score 或速度场
Flow Matching 属于学向量场连接两端分布, 直观上是"把概率质量搬运的最优路径回归"
```

**一句话**：「CE 是分类的 MLE；MSE 是高斯噪声下的 MLE；我们 Flow 的 MSE 是速度场回归，不是直接对数据似然最大化。」

### 2.4 KL 散度 / 交叉熵 / 总变差

```text
KL(p||q) = sum p log(p/q) = E_p[log p - log q]
  非负, 不对称;  KL=0 iff p=q
  上界相关: KL >= 0,  Pinsker: TV² <= KL/2

交叉熵 H(p,q) = -sum p log q = H(p) + KL(p||q)
  最小化 CE 等价最小化 KL (p 固定时)

应用:
  RL 的 KL 惩罚:  防 π 远离 π_ref
  JEPA/VICReg: 不直接用 KL 但思想是约束分布形状
  DPO: 隐式 reward = β log π/π_ref
```

**和 L2 的区别**：L2 比数值差；KL 比分布形状，对「把概率放到 p 几乎为 0 的地方」惩罚更狠——RL 防乱飘用 KL 更有理论感。

### 2.5 梯度与链式法则（口述版）

```text
方向导数最大方向 = 梯度方向
SGD 沿负梯度
多元链式:  ∂L/∂x = ∂L/∂y * ∂y/∂x
雅可比矩阵 J = ∂y/∂x, 反传是 J^T 乘上游梯度
Hessian 与曲率: 牛顿法用, 深度学习一般不用全 Hessian
```

### 2.6 概率不等式/估计（加分）

```text
大数定律:  样本均值 -> 期望
中心极限:  独立同分布和趋向正态
蒙特卡洛:  用采样估计期望 (RL reward 平均、FM loss 都是 MC)
重要性采样:  E_p[f] = E_q[(p/q) f]  -> PPO ratio 的来源
偏差-方差分解:  泛化误差 = 偏差² + 方差 + 噪声
```

**重要性采样和 PPO 串讲**（高价值答案）：
> 「PPO 的 ratio 就是重要性采样权重：数据来自 π_old，却要优化 π_new 的期望，乘 `π_new/π_old` 校正分布；clip 是方差控制，防止比值极端样本主导梯度。」

### 2.7 信息论与正则（一句带过）

```text
互信息 I(X;Y)=H(X)-H(X|Y)
最大互信息 -> 学有判别力的表征 (对比学习与之呼应)
MDL/压缩视角: 好表征短描述长度 -> VICReg 防冗余维度
```

### 2.8 线性代数快问快答

```text
矩阵乘法结合律有, 交换律一般没有
特征分解 A v = λ v;  对称阵可正交对角化
SVD:  A = U Σ V^T;  低秩近似 = 只留大奇异值  -> LoRA 低秩动机
范数:  Frobenius / L2 权重范数与 weight decay
数值稳定:  softmax 先减 max;  log-sum-exp 技巧
```

---

## 3. 手写题 / LeetCode 中高频（看公司，准备 10 道级别）

> 自驾/具身公司算法轮常见：**数组字符串、链表、二叉树、栈队列、哈希、双指针、滑动窗口、LRU、简单 DP**。Hard 图论/并发题较少；**更常见的是手写模型片段**（见下）。

### 3.1 模型手写（比 LC 更可能！优先准备）

```text
1. scaled dot-product attention + 多头拆分
2. Flow Matching 训练 step + Euler 推理   [你已被考过]
3. PPO-clip loss
4. GRPO 组内 advantage
5. LoRA 前向
6. 高斯 log-prob
7. softmax / LN 手写数值稳定版
8. BEV 简单投影或 IoU 计算
```

**Flow Matching 训练+推理（你被手撕过的标准答案，请背到肌肉记忆）：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityNet(nn.Module):
    def __init__(self, dim=2, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, dim),
        )

    def forward(self, x, t):
        # x: (B, L, 2), t: (B,)
        h = torch.cat([x, t.view(-1, 1, 1).expand_as(x[..., :1])], dim=-1)
        return self.net(h)

def fm_train_step(model, opt, x1):
    """
    x1: (B, L, 2) 真实轨迹
    """
    model.train()
    B = x1.shape[0]
    x0 = torch.randn_like(x1)                 # 噪声端
    t = torch.rand(B, device=x1.device)       # 时间
    tb = t.view(B, 1, 1)
    xt = (1 - tb) * x0 + tb * x1              # 直线插值
    v = model(xt, t)                          # 预测速度
    target = x1 - x0                          # 目标速度
    loss = F.mse_loss(v, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def fm_sample(model, B, L=30, steps=20):
    model.eval()
    x = torch.randn(B, L, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i / steps, device=x.device)
        v = model(x, t)
        x = x + v * dt                        # Euler
    return x                                  # 生成轨迹
```

**追问准备**：为何 target 是 `x1-x0`？（直线求导）；steps 为何 20？（精度 vs 延迟）；如何加条件？（cat/cross-attn 条件进 net）；多模态怎么保？（随机 x0 + 行为 token）。

### 3.2 LC 高频清单（各备 15 分钟思路即可）

**滑动窗口 —— 无重复最长子串**
```python
def length_of_longest_substr(s: str) -> int:
    seen, left, ans = set(), 0, 0
    for right, ch in enumerate(s):
        while ch in seen:
            seen.remove(s[left]); left += 1
        seen.add(ch)
        ans = max(ans, right - left + 1)
    return ans
```

**双指针 —— 两数之和（有序数组）/ 移除元素**
```python
def two_sum_sorted(nums, target):
    i, j = 0, len(nums) - 1
    while i < j:
        s = nums[i] + nums[j]
        if s == target: return [i, j]
        if s < target: i += 1
        else: j -= 1
    return []
```

**哈希 —— 两数之和（无序）**
```python
def two_sum(nums, target):
    pos = {}
    for i, x in enumerate(nums):
        if target - x in pos: return [pos[target - x], i]
        pos[x] = i
    return []
```

**链表 —— 反转 / 环 / 合并两个有序链表**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def reverse_list(head):
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast: return True
    return False
```

**LRU（必考级）**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d = OrderedDict()   # 保插入顺序

    def get(self, key):
        if key not in self.d: return -1
        self.d.move_to_end(key)  # 访问移到队尾
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)  # 淘汰队首
```

**手写 OrderedDict 原理追问**：哈希表 O(1) 查找 + 双向链表维护新旧；Python 标准库 `OrderedDict` 近似；面试可主动说「自己实现就是 dict + doubly linked list」。

**栈 —— 有效括号 / 最小栈**
```python
def is_valid_brackets(s):
    mp = {')': '(', ']': '[', '}': '{'}
    st = []
    for ch in s:
        if ch in '([{': st.append(ch)
        else:
            if not st or st[-1] != mp[ch]: return False
            st.pop()
    return not st
```

**二叉树 —— 层序遍历 / 最大深度**
```python
from collections import deque

def level_order(root):
    if not root: return []
    q, out = deque([root]), []
    while q:
        level = []
        for _ in range(len(q)):
            n = q.popleft(); level.append(n.val)
            if n.left: q.append(n.left)
            if n.right: q.append(n.right)
        out.append(level)
    return out

def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**简单 DP —— 爬楼梯 / 最长递增子序列 O(n log n)**
```python
def climb(n):
    a, b = 1, 1   # dp0, dp1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

import bisect
def lis(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)
```

**区间/贪心 —— 合并区间**
```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out
```

### 3.3 手写题考场策略

```text
1. 先复述+问边界 (空、重复、规模、有序否)
2. 先说暴力, 再优化到 O(n)/O(n log n)
3. 边写边说不变量
4. 用例: 空、单元素、全相同、负数、极长
5. 如果卡住: 降级讲思路/复杂度, 别硬憋
6. 模型题: 先说训练目标和张量 shape 再写
```

### 3.4 时间预算

```text
算法轮 60 分钟典型:
  5min 自我介绍/项目钩子
  10min 一道中等 LC (或直接模型题)
  20min 项目深挖 (MOT/WAM/FM)
  10min ML 八股
  10min 反问
模型手写题通常直接替掉 LC。
```

---

## 4. 论文深挖（π₀ / OpenVLA / Diffusion Policy 等，各 60 秒 + 追问）

### 4.1 π₀（必准备，你项目直系亲属）

**60 秒：**
> 「π₀ 是 Physical Intelligence 的 VLA：冻结/半冻结 VLM 做多模态理解，动作不走 LLM 自回归 token，而是 **Flow Matching 动作专家**——一个小一些的 transformer 通过 **cross-attention 读 VLM 的 KV cache**，从噪声迭代出连续动作块。设计动机是：动作要连续、低延迟、多峰；LLM 离散 token 精度和实时性不够。**π₀ 原版没有 FAST**；FAST 是另一条离散化动作的变体。」

**结构要点：**
```text
输入: 图像 + 语言 + 机器人状态
VLM:  编码成 KV
动作专家: 初始噪声动作块, T 步 FM, cross-attn 条件在 KV 上
输出: action chunk (多关节连续值)
训练: 演示上 Flow MSE; 可再 RL (π₀ 家族后续工作)
```

**追问：**
1. 和我们比亚迪 MOT 差在哪？→ π₀ 动作专家是**外挂子网 + 末端 cross-attn**；MOT 是**每层双套 QKV/FFN 分路径**，交织更深；我们 gen 还双向自注意 + 可接风险/可信条件。
2. 为什么冻结 VLM？→ 保护语言/视觉能力，动作梯度杂；我们 Reasoner detach 同理。
3. 步数多少？→ 动作块短，个位数到几十步可调；延迟用步数和蒸馏权衡。
4. 和 Diffusion Policy 关系？→ 同属条件动作生成；π₀ 强在大规模 VLM 条件与语言泛化。

### 4.2 OpenVLA

**60 秒：**
> 「OpenVLA 是开源 VLA 基线：视觉编码器 + LLM，把**动作量化成离散 token**，用 next-token prediction 自回归吐动作。优点是结构简单、复用 LLM 生态、开源权重大；缺点是量化损失、逐 token 延迟、多峰靠采样温度硬撑。我一般拿它当『离散动作路线』的对照物。」

**追问：** 动作怎么量化？→ 各维均匀 bins 或学习码本；bin 太少精度差，太多词表大。和 π₀ 选型：要连续丝滑选 FM/扩散，要极致复用 LLM 训练设施可选离散。

### 4.3 RDT / RDT-2（你简历里有，防深挖）

**RDT**：双臂/多任务大扩散动作模型，条件含语言和本体，动作用扩散生成多峰。

**RDT-2**：训练上引入 **RVQ 残差量化**，让 VLM 用 CE 学离散动作（监督稳）；**推理不再走 RVQ 解码**，而是连续动作头/流——训练离散、推理连续。面试若问「推理用不用 RVQ」答：**不用**，RVQ 主要服务训练阶段的语义对齐。

### 4.4 Diffusion Policy（具身岗高频）

**60 秒：**
> 「Diffusion Policy 把视觉条件下的机器人动作当成扩散生成问题：观测历史编码成条件，用 DDPM/一致性方式从噪声生成 **action chunk**。核心卖点是**多峰动作分布**（擦桌子可以左绕右绕）和比单纯回归更稳的成功率；代价是采样步数。ACT 则是 CVAE+Transformer，更快偏单峰；我项目用的 Flow Matching 可看成这条线上更直线、更易接 RL 的点。」

**追问：**
1. 为什么 chunk？→ 平滑、降决策频率、和扩散多模一次生成匹配。
2. 观测怎么进？→ CNN/ViT 编码历史帧 + 机器人状态，作为空间/时间条件（FiLM 或拼接）。
3. 推理步数？→ 原版可到 100；实际 10–50 或蒸馏到 1–4。
4. 和 π₀ 比？→ DP 专注操作、条件较小；π₀ 挂大 VLM 语言泛化强。

### 4.5 ACT（Aloha 系，具身常问）

**60 秒：**
> 「ACT = Action Transformer with CVAE：编码观测，隐变量 z 吸收多模态，解码动作块；训练时 KL 约束 z，推理可 z=0 或采样。特点是**快、适合真机高频**，在 Aloha 双臂上很成功。和扩散比：单步快、多峰靠 CVAE 不如扩散强，但部署友好。」

### 4.6 其他扫盲名单（被点名能接住即可）

| 论文/系统 | 一句话 |
|-----------|--------|
| CLIP | 图文对比对齐，零样本分类 |
| LLaVA | 视觉特征投影进 LLM 做指令微调 |
| V-JEPA | 视频表征预测未来，世界模型向 |
| Dreamer | latent 世界模型 + 想象中训策略 |
| Decision Transformer | 用 RL �Return 当条件的序列建模 |
| Diffusion Transformer (DiT) | patch token + 时空注意力的扩散骨干 |
| FLUX/SD3 | 图像 FM 主流，工程参考 |
| DeepSeekMoE | 共享专家 + 细粒度路由专家 |
| DAgger | 在策略状态收集专家标签 |
| ACT/Aloha | 真机低延迟动作块 |

### 4.7 「基模大家都怎么做」+「自动驾驶到什么程度」（你被问过，标准话术）

**基模（通用模型）三条路线：**
```text
1. 单一大模型吃所有模态 (稠密):  一个 Transformer 硬融合 -> 简单但模态互相稀释
2. 混合专家 MoE / MoT:            参数分片, 激活稀疏 -> Cosmos-3 MoT, DeepSeekMoE
3. 模块基座 + 统一接口:            各域最强骨干 + 对齐层/动作头 (很多车厂现实选择)

我合作的比亚迪侧: 路线 2+3 —— Cosmos-3 MoT 双塔当底座,
Reasoner 继续多模态理解, Generator 挂 Flow 动作专家 (FM + 自回归 + 锚点 query 工程变体)
```

**自动驾驶现在什么程度（客观叙事，防吹防黑）：**
```text
能力:
  - 高速/城市 NOA 已量产, 头部能处理大部分常规交互
  - 端到端 + 世界模型成为主流研发方向, 数据闭环比模型结构更像护城河
  - 开环榜 (NAVSIM 等) 分数内卷, 与真闭环仍有 gap
局限:
  - 长尾 corner case、施工、恶劣天气、博弈仍要大量兜底规则
  - 传感器+算力成本、法规责任、接管率是量产瓶颈
  - 严格 L4 运营设计域仍窄

我的判断 (面试可说):
  下一阶段胜负手是 [数据闭环 + 世界模型想象 + 安全可信接口],
  不是单一网络结构; 这也是我从自驾往具身迁的逻辑 —— 技能同构, 数据形态不同
```

**为何比亚迪自驾不如华为/理想（高危题，答法=只讲结构差异不贬低）：**
```text
不要说: 组织/谁不行 (显得幼稚且有风险)
要讲客观结构因素:
  1. 战略重心: 车型矩阵宽、电动化三电投入极重, 智驾资源被摊薄 vs 全栈软件公司
  2. 全栈自研深度: 华为有芯片+车云+ADS 全链, 理想重仓智驾研发与数据闭环时间早
  3. 数据与量产闭环: 头部城市 NOA 量、接管数据回流规模存在先发积累
  4. 生态: 昇腾算力/工具链自研 vs 采购适配的迭代速度差

高情商收尾:
  「差距更多在全栈投入节奏和数据闭环规模, 不是某一层算法不可逾越;
   我们用开放底座 (Cosmos 等) + 场景化 WAM/安全层, 是成本结构下的理性路线;
   算法点上 FM/联合流这些我是一线做过的, 迁移速度不差。」
```

---

# 第八部分：真实面试实战包（按你被问过的原题准备）

> 依据你回忆的现场：自我介绍、比亚迪负责什么、手撕 FM、yaw 角、RL 数据单位、数据怎么处理、MOT+FM+自回归+锚点 query、离华为理想差距、自驾 VLA/WM vs 具身、迁移、行业阶段、基模、反问、离职原因。**全部第一人称可背。**

---

## 1. 自我介绍（1 分钟 / 2 分钟两版）

### 1.1 一分钟版（技术面开场）

> 「您好，我叫 XXX，目前在比亚迪（清华背景合作公司）做**自动驾驶世界模型与端到端策略**，主要和比亚迪侧联合攻关 **MOT 架构的世界模型 + Flow Matching 动作专家**。
>
> 我的技术主线是三块：
> **1) 统一建模底座**——Cosmos-3 双塔 MoT，Reasoner 理解、Generator 生成，序列打包和双路径注意力是我落地的；
> **2) 连续动作生成**——动作专家用 Flow Matching，训过训练步和 Euler 推理，也接过自回归与锚点 query 的工程变体；
> **3) 世界动作联合**——WAM：动作与后果同一条流，加可信路由。
>
> 在此之外我独立推进过风险场 VLA + Flow-GRPO（NAVSIM 0.87）、JEPA 表征世界模型等项目。我接下来想投**具身智能/VLA**，因为自驾的多模态条件生成、动作块、后训练和安全接口，和机械臂/移动机器人是同构问题，我希望把这套闭环搬到真机上。」

### 1.2 两分钟版（可插数字与公司合作点）

在 1 分钟版基础上加：
> 「和比亚迪的合作形态是**联合项目**：我这边出模型结构与训练框架（MOT 前向、Flow 动作专家、WAM 联合损失），车端数据与场景由业务侧提供。交付上我负责关键代码路径和消融，例如动/静锚点 query 对长尾切入的成功率、动作专家 FM vs 自回归的延迟对比。我更擅长把论文级结构落到可训练可监控的代码上，而不只是调包。」

### 1.3 自我介绍公式（记不住长稿就用这个）

```text
身份一句话 -> 主项目(MOT+FM 合作) -> 技能三条(底座/动作/后训练或安全) ->
一个数字 -> 迁移动机(具身) -> 停, 等对方接话
切忌: 流水账教育经历、与岗位无关的课设
```

---

## 2. 「你在比亚迪主要负责什么」+ 手撕 Flow Matching 预案

### 2.1 标准回答（先 ownership 再细节）

> 「我在比亚迪合作项目里，主线是 **MOT 架构下的动作专家与世界模型训练**，具体三块：
>
> **第一，Flow Matching 动作专家。** 在 Generator 侧/动作头上实现条件流匹配：轨迹/关节动作块从噪声直线路径回归速度场，我写过并能默写**训练 step 和 Euler 推理**；也做过与**自回归动作头**的对比（离散 token 稳、好复用 LM 设施；FM 连续、多峰、延迟可控），以及**锚点 query** 驱动的动作槽位初始化。
>
> **第二，MOT 双塔前向。** und/gen 两套投影、causal vs bidirectional、gen→und cross-attn 与 detach 边界、序列打包 mask。
>
> **第三，WAM 联合与训练闭环。** 动作与后果同一联合流，逆一致、风险/支持域分量、可信路由；后训练侧接组相对策略优化。
>
> 数据侧我参与多相机 token 对齐、演示/仿真轨迹清洗和质量过滤。手撕的话我从 FM 训练和推理写起都可以。」

### 2.2 手撕 FM：考场 5 分钟脚本（你说被要求写 train + sample）

**Step0 黑板公式（30 秒）**  
`x_t=(1-t)x0+t x1`，`v*=x1-x0`，`loss=MSE(v_θ,v*)`，推理 `x←x+vΔt`。

**Step1 写模块**（VelocityNet，shape 注释清楚）  
**Step2 写 train_step**（采 x0、采 t、插值、预测、MSE、backward）  
**Step3 写 sample**（randn、for steps、Euler、no_grad）  
**Step4 主动补 3 个 bullet 加分**：
- 条件怎么进：`cond` 拼进特征或 cross-attn；
- 步数与 Heun 二阶；
- 若接 GRPO：确定性 ODE 无 log_prob，采样改 SDE。

**易错点（写之前默念）：**
```text
[ ] t 广播 (B,1,1) 不是 (B,)
[ ] target 是 x1-x0 不是 x1
[ ] sample 里 model.eval + no_grad
[ ] 训练/推理同一归一化空间
[ ] dt = 1/steps, t = i/steps 对齐
```

**若被追问「自回归和 FM 你们怎么选」：**
```text
自回归:  动作离散化, 复用 CE/KV cache, 精度受 bin, 逐步误差累积
FM:      连续块一次出, 多峰好, 步数换延迟, 易接 GRPO
锚点 query: 给动作头一组可学习槽位, 从场景里 cross-attn 取目标位置
          再在锚点附近流式细化 —— 相当于 DETR anchor 对连续动作的类比,
          收敛更快、多目标/多假设好挂
最终选择看: 延迟预算、演示是否多峰、是否要 RL
```

### 2.3 若突然改考「写 attention / PPO」快速入口

同第七部分清单；PPO 一定写 `ratio=exp(new-old)` + `min` + `clip`。

---

## 3. 高频场景题（你被问过的原题）

### 3.1 「为什么自动驾驶要 yaw 角？不是输出 x,y 就行吗？」

> 「因为车是**非完整约束（non-horizon）的刚体**，不是质点。
>
> 1. **状态可定义**：只给 (x,y) 无法区分车头朝东还是朝北，同一位置不同朝向，下一步可行域完全不同；
> 2. **运动学**：自行车模型 `ẋ=v cosψ, ẏ=v sinψ, ψ̇=v/L tanδ`——yaw ψ 是状态转移核心变量，缺了无法积分轨迹；
> 3. **几何占据**：碰撞检测是**有向包围盒**，长度宽度沿 yaw 展开；两个车中心重合判断不了，必须朝向；
> 4. **控制接口**：横摆角速度、航向跟踪、曲率 κ≈Δψ/Δs 都直接用 yaw；
> 5. **多模态表示**：换道/掉头在 (x,y) 下轨迹交叉纠缠，在 (x,y,ψ) 下方向语义清晰。
>
> 所以我们轨迹常见输出是 **(x, y, yaw)** 三维（或再加 v、曲率、加速度）；若输出世界系速度矢量，也要能等价反出 yaw。2D 只存在于把车当质点的极简规划玩具里。」

**加分一句**：「SE(2) 位姿 = 平移 2 维 + 旋转 1 维 = yaw 正是那个旋转自由度。」

### 3.2 「强化学习的数据集计量单位是什么？」

> 「我一般按**三层单位**讲，避免只答一个词：
>
> **1) 单条数据 —— transition / step**：`(s_t, a_t, r_t, s_{t+1})` 一条时序步，有的库叫 `timestep` 或 `t`；
> **2) 一段数据 —— episode / clip / trajectory**：从初始到终止（或时长上限）的一段；机器人/视频 RL 常用 **clip**（固定秒数或固定帧的片段），自驾日志常用 **scenario / log segment**；
> **3) 数据集规模 —— 常见四种并报**：
> - `num_steps`（总 transition 数，百万/千万 step）；
> - `num_episodes`（多少段）；
> - `num_envs × num_steps`（并行环境步）；
> - 若是演示/离线数据集：**hours / hours of teleop** 或 **clips × fps × 时长**。
>
> 例：『我们离线池约 2e6 steps / 5 万 clips；GRPO 每更新采样 G=24 条轨迹 × T=32 步 = 768 个 RL step/场景，再组内归一。』
>
> 若对方指的是**监督视频/VLA 预训练**，则常用：**frames、clips、hours、episodes、keyframes、camera-hours**（多相机还要乘 N_cam）。我们 MOT 训练说的 **clips + frames + 相机数** 就是这个口径。」

**你上次说 T、clips 是对的**——这次补上 transition/step 和「总步数 vs 段数 vs 小时数」三件套即可显得更专业。

### 3.3 「数据怎么处理？」（训练数据管线，通用+我们的）

```text
自驾/MOT/WAM/VLA 通用七步:
1. 采集与同步:  多传感器时间对齐, 外参标定, 丢包检测
2. 抽段:        按场景切 clip (变道/无保护左转/加塞...), 固定时长或事件驱动
3. 清洗:
   - 轨迹质量: 定位跳变、急刹尖峰、人为接管段剔除或降权
   - 重复: 感知哈希 + embedding 去重
   - 错误标: 规则碰撞误报、车道线错
4. 标签/条件:  自车轨迹重采样到统一 T, 归一化到 ego 系;
              导航指令、红绿灯、他车未来 (预测用), 风险标签可自动算
5. 编码:        图像-> DCAE/BEV token; 文本-> BPE; 动作-> 归一化数值
              (归一化均值方差或按轴 max scale, 否则 FM 各维尺度失衡)
6. 配比与增广:  常规:长尾≈8:2 再难例上采样; 天气/光照/噪声增广;
              模态 dropout (掉一路相机仍能训)
7. 打包:        变长 padding 或 sequence packing; 烧 moe_gen_mask
              训练/验证按场景切分防泄漏 (同 clip 不能跨集合)
```

**面试金句**：「处理数据我会先问三件事：**对齐单位、归一化空间、长尾怎么进 batch**——这三件错了，结构再花也白训。」

### 3.4 「你们的 VLA / 世界动作模型具体怎么做的？」（合并总答）

> **VLA（RiskField-VLA / MOT 动作侧）：**
> 输入多相机 + 导航 + 自车状态；MOT/VLM 统一编码；条件里拼**时空风险场**；动作头 **Flow Matching** 出 T 步轨迹块，双专家 Base/LTE 软门控，Risk-Init 初始化噪声；推理产 K 条用 PDM 类分数选；后训练 **Flow-GRPO**（SDE 采样 log_prob + PPO clip + LoRA）。
>
> **WAM：**
> 不把『出招』和『算招’拆开。联合向量 `z=[轨迹|后果|风险|可信]` 同一条直线流积分；加**逆一致**逼着用动作；推理读支持域与风险做**双层可信路由**，不可信就回退经典规划。这样生成中途就知道代价，而不是事后 filter。
>
> **和纯 WM 差别**：纯 JEPA 只预测 `z_{t+1}` 不决策；WAM 是 `p(a,后果,风险,可信|c)`。

### 3.5 「MOT + Flow Matching + 自回归 + 锚点 query」关系（合作项目核心题）

```text
我按「谁负责什么」讲:
1. MOT:        底座. 每层 und/gen 双路径, 理解用因果+CE, 生成用双向+流/扩散
2. 动作专家:    Generator 侧条件生成 action chunk
   - FM 路线:     连续多峰, 我们主力, 能手撕 train/sample
   - 自回归路线:   动作离散 token, 复用 LM, 便于与语言指令交织
   - 二者常做 A/B: 延迟、成功率、多峰多样性
3. 锚点 query:  可学习/场景生成的动作槽位 (DETR anchor 类比)
                先 cross-attn 初定位/初朝向, 再 FM 在锚点邻域细化
                收敛快, 多假设 (直行/变道) 可挂多锚点
4. 训练:        SFT 演示 + 可选 GRPO; Reasoner 冻结或 detach
一句话: MOT 是双塔公路, 锚点 query 是匝道口, FM/自回归是两种发动机, 动作 chunk 是货
```

### 3.6 「自驾的 VLA/世界模型 vs 具身的有什么区别？怎么迁移？」

**60 秒标准答案：**

> 「**本质同构，差在五处维度**——我会先讲同构建立信心，再讲差异显示懂行。
>
> **同构（可迁移的 80%）：**
> 1. 都是 `p(动作块|视觉,语言,本体状态)` 的条件生成；
> 2. 都用 Flow/扩散做多峰动作，chunk + 执行前 K 步 replan；
> 3. 都要预训练-BC-SFT-RL 后训练，奖励设计、组内优势、LoRA 逻辑相同；
> 4. 都要世界模型做想象与安全，支持域/OOD 思想相同；
> 5. 多模态 token 化、注意力、EMA 防坍缩同一套。
>
> **差异（迁移要改的 20%）：**
>
> | 维度 | 自动驾驶 | 具身操作/移动 |
> |------|----------|----------------|
> | 输出维 | 低维 SE(2)：x,y,yaw,(v) | 机械臂 7-DoF×双臂 + 夹爪，维更高 |
> | 时域 | 3–8s 轨迹，2–10Hz 规划 | 动作块 0.5–2s，控制 10–50Hz+ |
> | 状态 | 开环/弱闭环车路 | **强接触物理**，力/柔顺/打滑 |
> | 传感器 | 环视+定位，外参稳 | 手眼相机近距遮挡，频繁动 |
> | 环境 | 公共道路，法规约束 | 桌面/室内，任务奖励更稀疏或人工 |
> | 数据 | 车队日志 clip 小时 | 遥操作 episode 短、采集贵 |
> | 失效 | 碰撞/接管，可 fallback 慢 | 摔机损坏，安全项更硬 |
> | 指标 | PDMS/接管率 | 成功率、完成时间、鲁棒性 |
>
> **迁移路径我会直接说：**
> 1. **动作头重参**：轨迹(x,y,yaw) → 关节流形，锚点 query 换成末端位姿/关节 anchor；
> 2. **条件替换**：风险场→任务与接触风险；导航语言→操作指令；
> 3. **奖励替换**：PDMS 乘积 → 成功率+时间+力约束，仍可用组相对 RL；
> 4. **世界模型换模态**：DCAE 视频/表征 WM 保留，rollout 加接触物理可用仿真补；
> 5. **频率与安全**：缩短 horizon、提高 replan，fallback 从『靠边停车』改成『冻结/回初始位』。
>
> 所以我不是跨行从零，是**把条件生成 + 后训练 + 可信接口这套方法论换 payload**。」

**相似之处可再补 5 条（有时间就抛）：**
```text
- 都有多相机/多视角 token 与时序对齐
- 都有『指令敏感性』问题 (左转 vs 加速 / 拧开 vs 关上)
- 都有长尾数据问题 -> 拒绝采样/仿真合成
- 都有开环评测虚高、闭环才见真章
- 都需要模态 dropout 防传感器单点故障 (一路相机/一个关节编码器)
```

### 3.7 「离职原因」（你在比亚迪，这题必准备，勿说前东家坏话）

> **推荐结构（真诚 + 向前看 + 岗位匹配）：**
>
> 「我在比亚迪这段收获很大——**MOT 底座、Flow 动作专家、WAM 联合建模**都是能进到生产级训练管线的工作，公司对世界模型方向也很支持。
>
> 我看新机会主要是三个原因：
> 1. **方向聚焦**：我想更纯粹地做 **VLA/具身智能**，从『车上低维轨迹』走向『真机高维动作+接触物理』，贵司这个岗位和我的技术债几乎 1:1 对齐；
> 2. **闭环深度**：我希望更多在**真机/仿真闭环**里打转，而不仅是开环日志评测——这是我想补的能力面；
> 3. **成长曲线**：在现有合作里我已把 FM/MOT/WAM 主路径打通，下一步需要更大规模的数据闭环和更强的基座去碰撞，我判断贵司这边平台更合适。
>
> 所以不是对现状不满，而是**下一阶段的技能栈和贵司岗位更匹配**。」

**避雷：**
```text
不要: 加班、领导、工资低、比亚迪不如华为、学不到东西
可以微调: 「联合项目协作模式上，我更希望在产品-算法更紧的团队里端到端负责」
若追问具体矛盾: 只谈工作方式/技术路线偏好, 不谈人事
```

**变体 —— 「为什么从自动驾驶转具身」：**
> 「不是转赛道，是同一技术栈换载体：条件生成、动作块、RL 后训练、世界模型、安全路由全部复用；具身的强闭环和任务奖励反而更能验证我这些方法。」

### 3.8 「职业规划」（HR/主管）

```text
1 年:  在贵司 VLA/具身把真机成功率与数据闭环打穿, 独立扛一条任务线
3 年:  能定义动作基座(生成+后训练+安全)标准, 带小方向
长期:  世界模型驱动的通用操作/移动智能体 —— 和我现在 WAM 路线连续
避免: 「转管理」「创业」「读博再说」与岗位冲突的回答
```

### 3.9 优缺点（技术人安全版）

```text
优点:  能把论文结构落到可训练代码 (MOT/FM 一线); 习惯分模块消融与监控
缺点:  有时过早抠实现细节; 正在练习先对齐目标再深潜 —— 举一次被 mentor 拉回来的例子
```

---

## 4. 反问清单（你被要求反问，准备 6 选 3）

### 4.1 技术深挖型（显水平）

```text
1. 团队 VLA 是自研基座还是站在 π₀/OpenVLA/自研 VLM 上改? 动作头更偏 FM 还是自回归?
2. 后训练用 GRPO/PPO/DPO 吗? 奖励是规则任务成功还是学习 RM? 防 hack 怎么做?
3. 真机与仿真比例? 域随机和 sim2real 瓶颈卡在渲染还是动力学?
4. 世界模型是服务想象 RL、数据增广还是评测? 和策略是分开训还是一体?
5. 安全上除了成功率, 有没有支持域/OOD、fallback、看门狗这类硬约束?
```

### 4.2 团队与成长型（HR 轮）

```text
6. 这个岗位前三个月最希望我解决的具体问题是什么?
7. 团队现在最大瓶颈是数据吞吐、奖励设计还是真机时间?
8. 算法从论文到上机的决策链是怎样的? 消融文化如何?
```

### 4.3 业务型（终面/主管）

```text
9. 产品侧第一阶段的成功率/延迟/成本红线分别是?
10. 对「自驾经验迁移具身」的候选人, 你们最担心哪一环? 我可以怎么证明?
```

**模板**：技术面反问 1+2+5；HR 反问 6+7；终面 9+10。

---

## 5. 现场快答卡（30 秒级，防突然袭击）

| 问题 | 30 秒骨架 |
|------|-----------|
| yaw 为什么需要 | 非完整约束 + 运动学 + 有向框碰撞 + 控制曲率 |
| RL 数据单位 | transition/step；episode/clip；总 steps 或 hours 一起报 |
| 数据怎么处理 | 同步→抽段→清洗→归一→配比→增广→packing |
| FM 怎么训练 | x0,t,插值,MSE 到 x1-x0 |
| FM 怎么推理 | randn + Euler T 步 |
| FM vs AR | 连续多峰低延迟 vs 离散复用 LM；锚点 query 做初值 |
| VLA 是什么 | `p(a|o,l)` 条件动作块生成 |
| WM 是什么 | `p(s'|s,a)` 只预测不决策 |
| WAM 是什么 | `p(a,后果,风险,可信\|c)` 联合流+路由 |
| MOT vs MoE | 塔/路径级两条 vs FFN 专家级 top-k |
| 比亚迪合作 | MOT 底座 + FM 动作专家 + WAM 联合，我负结构与训练 |
| 为何离职 | 方向聚焦真机闭环 + 岗位技术债匹配，不贬低现司 |
| 自驾 vs 具身 | 同构条件生成/后训练；差在动作维、频率、接触、奖励、数据 |
| 自驾到啥程度 | NOA 量产成熟，胜负手转数据+WM+安全，开环闭环仍有 gap |
| 基模怎么做 | 稠密统一 / MoE-MoT / 模块基座；我们是 MoT+动作头 |

---

## 6. 本部分自测清单

- [ ] 1 分钟自我介绍能顺完不超时？
- [ ] 比亚迪 ownership 三块 + 手撕 FM train/sample 8 分钟内写完？
- [ ] FM vs 自回归 vs 锚点 query 关系能否 1 分钟讲清？
- [ ] yaw 角四点理由？
- [ ] RL 数据三层单位 + 一个带数字例句？
- [ ] 数据处理七步？
- [ ] 离职原因不踩雷版本？
- [ ] 自驾 vs 具身对比表 + 迁移五步？
- [ ] 离华为理想差距的高情商结构？
- [ ] 行业阶段 + 基模三路线？
- [ ] 反问 3 个已选好？
- [ ] LRU / 滑动窗口 / 树层序 / 高斯 log prob 能限时写？
- [ ] π₀、Diffusion Policy、ACT 各 60 秒？
- [ ] KL vs CE、MLE、重要性采样与 PPO 能串讲？

---

## 7. 全文最终复习优先级（如果只剩 7 天）

```text
Day1  自我介绍 + 比亚迪 ownership + 手撕 FM (抄 5 遍默 2 遍)
Day2  话术: MOT / VLA / WAM / JEPA 出声录屏回听
Day3  第六部分 VLA/WM/WAM 原理 + 第三部分 40 题前 20
Day4  yaw / RL 数据单位 / 数据处理 / FM vs AR vs 锚点 (本部分 3)
Day5  ML 八股 LN-Adam-过拟合 + 概率 KL-MLE-高斯
Day6  离职/规划/反问 + 自驾vs具身 + 行业与基模
Day7  LRU+滑窗+attention 手写 + π₀/DP/ACT 60 秒 + 模拟一轮 60min
```

---

*到此，本文覆盖：领域原理与四项目代码、第一人称话术、预训练后训练、岗位指南、VLA/WM/WAM 精讲、ML/数学/手写/论文基础、以及按你真实被问原题整理的实战包。吃透后优先保证「说得出、写得出手撕、答得体面离职与反问」——这三件比再堆冷门论文分高。*
