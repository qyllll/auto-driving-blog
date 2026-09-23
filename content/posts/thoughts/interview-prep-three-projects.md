---
title: "自动驾驶面试深度复习：RiskField-VLA + TrustDrive-WAM + JEPA-DRIVE 三大项目全拆解"
date: 2026-09-23
draft: false
categories: ["个人思考"]
summary: "覆盖三大项目的完整面试复习文档：第一部分深度复盘所有知识点和专有名词，第二部分给出 Flow Matching、Flow-GRPO、VLA、WAM、JEPA 等核心算法的伪代码实现，第三部分附面试官压力面问题与参考回答。一篇搞定所有面试准备。"
tags: ["面试", "自动驾驶", "Flow Matching", "GRPO", "VLA", "WAM", "JEPA", "NAVSIM"]
math: true
weight: 98
---

# 第一部分：知识点深度复盘

> 这一部分覆盖三大项目涉及的所有概念、算法、专有名词。每个概念从"是什么 → 为什么 → 怎么用"三个角度讲清楚，小白也能看懂。

---

## 1. 自动驾驶大背景：为什么需要端到端？

### 1.1 传统自动驾驶 vs 端到端

**传统方案（模块化）**：

```
摄像头/LiDAR → 感知(检测/跟踪/预测) → 规划(规则/优化) → 控制
    每个模块独立训练，中间传递的是"人工设计的接口"
```

问题：
- 信息在模块间传递时**有损**（感知输出的是框，丢了纹理、语义）
- 模块间**误差累积**（感知错一点，规划就错很多）
- 规则-based 规划在长尾场景（corner case）**写不完规则**

**端到端方案（E2E）**：

```
摄像头/LiDAR → 单一神经网络 → 轨迹/动作
    从传感器到轨迹，一个网络搞定
```

优势：
- 信息**无损传递**（中间用特征向量，不是人工接口）
- **数据驱动**：给什么数据学什么，上限更高
- **泛化能力强**：见过的场景模式能迁移

### 1.2 NAVSIM 评测基准

**NAVSIM** 是目前自动驾驶端到端算法的主流评测基准。

**核心指标 PDMS（Pilot-Driving Metric Score）**：

```
PDMS = 各项子指标的加权乘积

子指标包括：
- NC (No at-fault Collision)：无责碰撞
- DAC (Drivable Area Compliance)：可行驶区域合规
- TTC (Time-to-Collision)：碰撞时间（越安全越高）
- Comfort：舒适度（加速度、jerk）
- EP (Ego Progress)：自车进度（别停下来不动）
- Speed Limit Compliance：限速合规
```

**为什么 PDMS 重要**：它是**乘积**，不是加权和。任何一项为 0，总分就为 0。这就是为什么长尾高危场景（碰撞、越界）会让分数趋近于 0——论文里说的"约 21% 长尾高危交互 Corner-Case 极易发生碰撞、违规、分数趋近于 0"就是这个意思。

**NAVSIM v1 vs v2**：v1 是基础版，v2 增加了更多长尾场景和更严格的评测协议。我们的项目分别达到 0.8713（RiskField-VLA）、0.8012（TrustDrive-WAM）、0.7285（JEPA-DRIVE）。

---

## 2. VLA（Vision-Language-Action）模型

### 2.1 什么是 VLA？

VLA = 视觉 + 语言 + 动作 三模态融合模型。

```
输入：图像（视觉）+ 自然语言指令（语言）
输出：机器人/车辆的动作（轨迹、关节角度等）

类比：你看到路况 + 听到"前方左转" → 你打出方向盘
```

**代表工作**：
- OpenVLA：开源 VLA，用 LLM 输出离散动作 token
- RDT：用扩散模型生成连续动作
- π₀：用 Flow Matching 生成连续动作

### 2.2 VLA 在自动驾驶中的角色

在 RiskField-VLA 项目中，VLA 负责：
1. **理解场景语义**：从图像中识别交通参与者、车道线、交通标志
2. **理解风险**：识别哪些交互是高危的
3. **生成轨迹**：输出自车应该走的轨迹

### 2.3 关键组件

**Object Query**：
```
可学习的查询向量，每个 query 负责"关注"场景中的一个目标。
128 组 object query = 128 个"注意力槽位"，从特征图中提取 128 个目标的特征。
类比：128 个侦察兵，每人负责盯一个方向/目标。
```

**BEV（Bird's Eye View，鸟瞰图）**：
```
把多路相机的图像投影到俯视平面，得到"上帝视角"的特征图。
好处：统一的空间坐标系，方便做规划。
```

**时序特征**：
```
不仅看当前帧，还看前几帧，得到"运动状态"（速度、加速度方向）。
类比：你看前几帧知道那辆车在加速，而不是只看到它此刻的位置。
```

---

## 3. 世界模型（World Model）与 WAM（World Action Model）

### 3.1 世界模型是什么？

**核心定义**：给定当前状态和动作，预测下一个状态会怎样。

```
p(s_{t+1} | s_t, a_t)

s_t = 当前状态（所有交通参与者的位置、速度）
a_t = 动作（自车的转向、油门）
s_{t+1} = 下一个状态
```

**类比**：你在踩刹车之前，脑子里已经"想象"了车会减速、后车会跟上来——这就是世界模型。

### 3.2 像素世界模型 vs 表征世界模型

| | 像素世界模型 | 表征世界模型 |
|---|---|---|
| 预测什么 | 未来帧的像素 | 未来帧的抽象表征 |
| 需要画出来？ | 是（生成视频） | 否（只输出向量） |
| 计算量 | 巨大 | 小 |
| 代表 | DriveDreamer, Sora | JEPA, Dreamer |

### 3.3 WAM（World Action Model）是什么？

WAM 是世界模型的一个变种：**不仅预测世界状态，还联合建模动作的影响**。

```
传统：先生成轨迹 → 后接一个评分器打分（割裂）
WAM：生成轨迹的同时，同步演化"后果、风险、支持域"（联合）

优势：轨迹和后果是"同时生成"的，不是事后预测的
```

在 TrustDrive-WAM 项目中，WAM 的核心创新是**联合动作-后果流**：在 Flow Matching 生成轨迹的过程中，同步演化四类隐表征——轨迹、后果、风险、支持域。

### 3.4 OOD（Out-of-Distribution，分布外）

```
训练数据的分布 = 模型"见过"的数据分布
OOD = 落在训练分布之外的输入

自动驾驶中的 OOD：
  - 训练时没遇到过的天气（暴雪、大雾）
  - 训练时没见过的车辆类型（异形车）
  - 训练时没遇到过的交互（突然窜出的行人）

为什么 OOD 危险：模型对 OOD 输入的预测不可靠，
但规划器可能"盲目相信"这些预测 → 灾难性后果
```

### 3.5 可信域 / 支持域（Trust Region / Support Domain）

```
支持域 = 模型"有把握"的输入区域
类比：一个学生只学过加减法，你问他乘法——他还在"支持域"之外

在 TrustDrive-WAM 中：
  可靠性信任校验 = 检查当前预测是否落在训练数据的支持域内
  如果是 → 信任世界模型的预测
  如果不是 → 切回传统规划策略
```

### 3.6 逆一致性约束（Inverse Consistency）

```
正向：给定状态 s 和动作 a → 预测下一状态 s'
逆向：给定 s' 和动作 a → 能否恢复出 s？

逆一致性 = 正向预测和逆向恢复要一致
作用：防止模型"忽略"动作信息
  （如果模型不看动作，正向逆向当然一致，但没有意义）
  只有真正"用了"动作信息，正逆才一致
```

---

## 4. JEPA（Joint Embedding Predictive Architecture）

### 4.1 JEPA 是什么？

JEPA = 联合嵌入预测架构，是 Yann LeCun 提出的自监督学习框架。

**核心思想**：在**嵌入空间（表征空间）做预测**，而不是重建像素。

```
传统方法（MAE/Diffusion）：预测被遮住部分的像素 → 花大量算力在无关细节上
JEPA：预测被遮住部分的抽象表征 → 只关注语义，忽略纹理

类比：
  像素预测 = 考试时要把被遮住的每个像素都画出来
  表征预测 = 只回答"被遮住的是什么动物"
```

### 4.2 为什么 JEPA 适合自动驾驶？

```
1. 像素冗余太多：背景渐变、光照变化对决策没用
2. 多模态未来：前方路口可能直行、左转、右转，像素空间回归到均值会得到"鬼影"
3. 生成式不等于理解：能画出逼真图像不等于理解物理因果
```

### 4.3 I-JEPA vs V-JEPA

| | I-JEPA（图像版） | V-JEPA（视频版） |
|---|---|---|
| 输入 | 一张图片，挖掉一块 | 一段视频的前几帧 |
| 预测什么 | 被挖掉那块的表征（**补全现在**） | 下一帧的表征（**预测未来**） |
| 是世界模型吗？ | 不是 | 是 |

**V-JEPA 才是世界模型**：看到前 4 帧，预测第 5 帧的表征——这才是预测未来。

### 4.4 JEPA 的三个组件

```
1. Context Encoder（上下文编码器）：编码可见部分 → 上下文表征 s_x
2. Target Encoder（目标编码器）：编码目标部分 → 目标表征 T(y)（用 EMA 更新）
3. Predictor（预测器）：输入 s_x + 位置条件 → 预测目标表征 ŝ_y

Loss = ||ŝ_y - sg(T(y))||²  （sg = stop-gradient）
```

### 4.5 表征坍缩（Representation Collapse）

```
问题：如果两个编码器一起训练，它们可以"串通"——都输出常向量 [0,0,0,...]，loss=0 但什么都没学到

解法：EMA + stop-gradient
  Target Encoder 不靠梯度更新，而是缓慢追随 Context Encoder
  算 loss 时梯度不传给 Target Encoder
  → Context Encoder 必须"真正理解"才能预测 Target 的输出
```

### 4.6 专有名词

**ViT（Vision Transformer）**：把图片切成 patch，每个 patch 当作一个 token，用 Transformer 处理。

**自监督学习**：不需要人工标注，从数据本身构造"伪答案"。如 BERT 的完形填空。

**预训练 + 微调**：先在大数据上学通用能力（预训练），再在目标任务上微调。

**线性探测（Linear Probing）**：冻结主干，只训练一个线性分类器，评估表征质量。

---

## 5. Flow Matching（流匹配）

### 5.1 什么是 Flow Matching？

Flow Matching 是一种训练**连续生成模型**的方法。核心思想：

```
在噪声 x₀ 和数据 x₁ 之间拉一条直线：
  x_t = (1-t)·x₀ + t·x₁，t ∈ [0,1]

训练网络预测"速度场" v_θ(x_t, t) = 从 x_t 指向 x₁ 的方向
```

**对比扩散模型（DDPM）**：

| | DDPM | Flow Matching |
|---|---|---|
| 路径 | 弯路（逐步加噪再去噪） | **直线**（一步到位） |
| 学什么 | 预测噪声 ε | 预测速度场 v |
| 推理步数 | 20-50 步 | **5-10 步** |
| 数学基础 | SDE（随机微分方程） | ODE（常微分方程） |

### 5.2 为什么 Flow Matching 更快？

```
DDPM = 蒙眼走迷宫，每步只能摸到附近，需要慢慢试
Flow Matching = 拿着地图走直线，直接朝目的地走
```

### 5.3 Flow Matching 训练 Loss

```
L = E[||v_θ(x_t, t) - (x₁ - x₀)||²]

1. 采样数据对：x₀=噪声，x₁=真实数据
2. 随机采时间步 t
3. 插值得到中间状态 x_t = (1-t)x₀ + tx₁
4. 网络预测 v_θ(x_t, t)
5. 正确答案是 x₁ - x₀（从噪声指向数据的方向）
6. 算 MSE loss，反向传播
```

### 5.4 Flow Matching 推理（ODE 积分）

```
x₀ ~ N(0, I)  （纯噪声）
for t in [0, 0.1, 0.2, ..., 0.9]:
    v = v_θ(x_t, t)        # 网络预测速度
    x_{t+1} = x_t + v·Δt   # Euler 积分，走一步
x₁ = 最终生成的数据
```

---

## 6. Flow-GRPO（流匹配强化学习后训练）

### 6.1 GRPO 是什么？

GRPO（Group Relative Policy Optimization）= DeepSeek-R1 提出的强化学习算法，核心是**组内竞争**。

```
1. 同一个 prompt 生成一组样本（如 24 条轨迹）
2. 用奖励模型打分
3. 组内归一化：advantage_i = (r_i - mean) / std
4. PPO 风格 loss 更新：比平均好的概率上升，差的下降

优势：不需要 critic 价值网络，用组内统计做 baseline
```

### 6.2 Flow-GRPO 的核心创新

**问题**：LLM 的 GRPO 中 log_prob 直接从 softmax 取 log（离散 token）。但 Flow Matching 是连续的，不能直接取 softmax。

**解法**：引入 SDE（随机微分方程），把确定性 ODE 变成有随机性的 SDE：

```
ODE（确定性）：dx = v_θ dt → log_prob 无法直接算
SDE（随机性）：dx = f dt + σ dw → 转移概率是高斯分布 → log_prob 直接可算

log p(x_{t+1}|x_t) = -||x_{t+1}-mean||²/(2σ²) - log(σ) - log(√(2π))
```

### 6.3 Flow-GRPO 完整流程

```
采样：同一场景生成 K 条轨迹（SDE 采样，记录 log_prob_old）
打分：奖励模型给每条轨迹打分（安全、舒适、进度）
Advantage：组内归一化
训练：PPO loss = -min(ratio·A, clip(ratio, 1±ε)·A)
  其中 ratio = exp(log_prob_new - log_prob_old)
梯度：只更新 LoRA 参数（0.5%），冻结主干
```

### 6.4 PPO-Clip 机制

```
ratio = π_new / π_old = exp(log_prob_new - log_prob_old)

loss = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)

当 A > 0（好轨迹）：希望 ratio 上升，但 clip 到 1+ε 防止更新太猛
当 A < 0（差轨迹）：希望 ratio 下降，但 clip 到 1-ε 防止降太猛
作用：限制每步更新幅度，训练稳定
```

### 6.5 KL 惩罚

```
L = L_policy + β·KL(π_new || π_ref)

π_ref = reference model（禁用 LoRA 的原始模型）
作用：防止新模型偏离原始模型太远，避免灾难性遗忘
```

---

## 7. 风险场（Risk Field）

### 7.1 什么是风险场？

传统方案对每个交通参与者单独预测，**没有显式建模交互关系**。风险场把所有参与者的影响建模为一个**连续的空间风险分布**。

```
类比：磁场——每个磁铁在空间中产生磁场，叠加起来决定铁屑的分布
风险场——每个交通参与者在空间中产生"风险"，叠加起来决定哪里危险
```

### 7.2 时空风险场

```
输出：32 × 32 × 8 的连续风险场
  32 × 32 = 空间分辨率（BEV 网格）
  8 = 时间维度（未来 8 步的风险演化）

每个网格值 = 该位置在该时刻的碰撞风险
```

### 7.3 α 软门控路由聚合

```
128 个 agent 的特征 → α 软门控 → 加权聚合 → 32×32×8 风险场

α 软门控 = learnable 的权重，决定每个 agent 对风险场的贡献
类比：不是所有车都一样危险，门控网络学习"谁更重要"
```

### 7.4 z_risk 风险隐编码

```
风险场 → 编码器 → z_risk（一个紧凑的向量）
作用：作为统一的风险先验，传递给下游模块
  - 轨迹生成：用 z_risk 优化初始噪声分布
  - 专家门控：用 z_risk 决定用哪个专家
  - 评分器：用 z_risk 评估轨迹风险
```

---

## 8. 双专家架构（Base + LTE）

### 8.1 为什么需要双专家？

```
常规场景（80%）：需要稳定、保守的轨迹
长尾高危场景（20%）：需要激进但安全的轨迹（如紧急避让）

单模型：两种场景混在一起学，互相妥协
双专家：Base 专家学常规，LTE（Long-Tail-Expert）学长尾
```

### 8.2 风险场熵自动门控

```
风险场熵 = 风险分布的不确定性
  熵高 → 场景复杂/长尾 → 倾向用 LTE
  熵低 → 场景常规 → 倾向用 Base

门控 = softmax(熵) → 权重 → 加权混合两个专家的输出
```

### 8.3 Risk-Init-Flow（风险初始化流）

```
标准 Flow Matching：噪声 ~ N(0, I)（纯随机）
Risk-Init-Flow：噪声 = f(z_risk)（用风险编码优化初始噪声分布）

作用：让轨迹生成"从更合理的起点出发"，减少盲目性
类比：不是从零开始找路，而是先大致知道危险在哪，从安全区域开始搜
```

---

## 9. 关键指标与缩写速查表

| 缩写 | 全称 | 含义 |
|------|------|------|
| E2E | End-to-End | 端到端，从传感器直接到轨迹 |
| BEV | Bird's Eye View | 鸟瞰图视角 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| WAM | World Action Model | 世界动作模型 |
| JEPA | Joint Embedding Predictive Architecture | 联合嵌入预测架构 |
| FM | Flow Matching | 流匹配 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| PPO | Proximal Policy Optimization | 近端策略优化 |
| ODE | Ordinary Differential Equation | 常微分方程 |
| SDE | Stochastic Differential Equation | 随机微分方程 |
| OOD | Out-of-Distribution | 分布外 |
| PDMS | Pilot-Driving Metric Score | NAVSIM 核心指标 |
| NC | No at-fault Collision | 无责碰撞 |
| DAC | Drivable Area Compliance | 可行驶区域合规 |
| TTC | Time-to-Collision | 碰撞时间 |
| EP | Ego Progress | 自车进度 |
| LoRA | Low-Rank Adaptation | 低秩适配微调 |
| EMA | Exponential Moving Average | 指数移动平均 |
| ViT | Vision Transformer | 视觉 Transformer |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| MMAE | Masked Autoencoder | 掩码自编码器 |
| VICReg | Variance-Invariance-Covariance Regularization | 表征坍缩正则 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| RL | Reinforcement Learning | 强化学习 |

---

# 第二部分：核心算法伪代码实现

> 这一部分从概念到伪代码，逐步实现每个核心算法。先讲"是什么"，再讲"怎么写代码"。

---

## 算法 1：Flow Matching 完整实现

### 1.1 概念回顾

Flow Matching 训练一个网络预测"从噪声到数据的方向"（速度场），推理时用 Euler 积分沿速度场走多步生成数据。

### 1.2 训练伪代码

```python
import torch
import torch.nn as nn

# ========== 模型定义 ==========
class DiT(nn.Module):
    """Diffusion Transformer：输入带噪状态+时间步，输出速度场"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.time_embed = nn.Linear(1, hidden_dim)  # 时间步编码
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)  # 输出速度
    
    def forward(self, x_t, t):
        """
        x_t: (B, seq_len, input_dim) 当前带噪状态
        t:   (B,) 时间步
        return: v_theta, shape (B, seq_len, input_dim) 速度场
        """
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden_dim)
        h = x_t + t_emb.unsqueeze(1)  # 时间步注入
        h = self.transformer(h)       # Transformer 处理
        v_theta = self.output_proj(h) # 输出速度场
        return v_theta

# ========== Flow Matching 训练 ==========
def flow_matching_train_step(model, real_data, noise):
    """
    训练一步 Flow Matching
    real_data: (B, seq_len, dim) 真实数据（轨迹/图片）
    noise:     (B, seq_len, dim) 随机噪声
    """
    B = real_data.shape[0]
    
    # 第 1 步：随机采时间步 t ~ U[0, 1]
    t = torch.rand(B, device=real_data.device)
    
    # 第 2 步：线性插值得到中间状态
    # x_t = (1-t) * noise + t * real_data
    t_expand = t.view(-1, 1, 1)
    x_t = (1 - t_expand) * noise + t_expand * real_data
    
    # 第 3 步：网络预测速度场
    v_pred = model(x_t, t)  # (B, seq_len, dim)
    
    # 第 4 步：正确答案 = 从噪声指向数据的方向
    v_target = real_data - noise  # x1 - x0，恒定速度
    
    # 第 5 步：MSE Loss
    loss = nn.MSELoss()(v_pred, v_target)
    
    return loss

# ========== Flow Matching 推理（ODE 积分）==========
@torch.no_grad()
def flow_matching_sample(model, shape, num_steps=10):
    """
    从噪声生成数据
    返回最终生成的轨迹
    """
    # 从纯噪声开始
    x = torch.randn(shape)  # x_0 ~ N(0, I)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i * dt
        t_batch = torch.full((shape[0],), t, device=x.device)
        
        # 网络预测速度
        v = model(x, t_batch)
        
        # Euler 积分：沿速度方向走一步
        x = x + v * dt
    
    return x  # x_1 = 生成的数据
```

### 1.3 逐行解读

| 代码行 | 对应概念 |
|--------|---------|
| `t = torch.rand(B)` | 随机采时间步，让模型学会在任意时刻预测速度 |
| `x_t = (1-t)*noise + t*real_data` | 线性插值，构造"半噪声半数据"的中间状态 |
| `v_target = real_data - noise` | 正确答案：从噪声指向数据的方向 |
| `loss = MSE(v_pred, v_target)` | 让网络预测的方向和真实方向一致 |
| `x = x + v * dt` | Euler 积分：每步沿预测的速度方向走一小步 |

---

## 算法 2：Flow-GRPO 完整实现

### 2.1 概念回顾

Flow-GRPO = 用 GRPO 强化学习微调 Flow Matching 模型。核心：采样多条轨迹 → 打分 → 组内比较 → PPO loss 更新。

### 2.2 SDE 采样（带 log_prob）

```python
def sde_step_with_logprob(model, x_t, t, dt, noise_level=0.7):
    """
    Flow Matching 的 SDE 采样一步，同时计算 log_prob
    关键：引入随机性，使转移概率可算
    """
    # 第 1 步：网络预测速度场
    v_theta = model(x_t, t)  # (B, seq_len, dim)
    
    # 第 2 步：计算 SDE 均值（带 score 修正的逆时 SDE）
    # mean = x_t * (1 + σ_noise²/(2σ) * dt) 
    #      + v_theta * (1 + σ_noise²*(1-σ)/(2σ)) * dt
    sigma = t  # 当前噪声水平
    sigma_noise = torch.sqrt(sigma / (1 - sigma + 1e-8)) * noise_level
    
    mean_coeff_x = 1 + (sigma_noise**2) / (2 * sigma + 1e-8) * dt
    mean_coeff_v = 1 + (sigma_noise**2 * (1 - sigma)) / (2 * sigma + 1e-8) * dt
    mean = x_t * mean_coeff_x + v_theta * mean_coeff_v * dt
    
    # 第 3 步：采样下一步（加随机噪声）
    std = sigma_noise * torch.sqrt(torch.abs(dt))
    epsilon = torch.randn_like(x_t)
    x_next = mean + std * epsilon
    
    # 第 4 步：计算 log_prob（高斯分布的 log 概率）
    # log p(x_next | x_t) = -||x_next - mean||² / (2σ²) - log(σ) - log(√(2π))
    log_prob = -((x_next.detach() - mean)**2).sum(dim=(1,2,3)) / (2 * std**2)
    log_prob = log_prob - torch.log(std + 1e-8) - 0.5 * torch.log(2 * torch.pi)
    
    return x_next, log_prob, mean
```

### 2.3 GRPO 训练主循环

```python
def flow_grpo_train_step(model, ref_model, reward_fn, prompts, config):
    """
    Flow-GRPO 一个训练步
    """
    G = config.group_size       # 每组轨迹数，如 24
    eps = config.clip_range     # PPO clip range，如 0.2
    beta = config.kl_beta       # KL 惩罚系数
    
    # ========== 第 1 步：采样一组轨迹 ==========
    all_trajectories = []
    all_log_probs_old = []
    all_rewards = []
    
    for prompt in prompts:  # 遍历每个场景
        trajectories = []
        log_probs_old_list = []
        
        for g in range(G):  # 每个场景采 G 条轨迹
            # SDE 采样，记录每步的 log_prob
            x = torch.randn(batch_shape)  # 初始噪声
            step_log_probs = []
            
            for step in range(num_steps):
                t = torch.full((B,), step/num_steps)
                x, log_prob, _ = sde_step_with_logprob(
                    model, x, t, dt
                )
                step_log_probs.append(log_prob)
            
            trajectories.append(x)  # 生成的轨迹
            log_probs_old_list.append(torch.stack(step_log_probs, dim=1))
        
        # 打分
        rewards = reward_fn(trajectories, prompt)
        
        all_trajectories.extend(trajectories)
        all_log_probs_old.extend(log_probs_old_list)
        all_rewards.extend(rewards)
    
    # ========== 第 2 步：组内归一化算 advantage ==========
    rewards_tensor = torch.tensor(all_rewards)
    mean_r = rewards_tensor.mean()
    std_r = rewards_tensor.std() + 1e-4
    advantages = (rewards_tensor - mean_r) / std_r
    advantages = torch.clamp(advantages, -2.0, 2.0)  # 裁剪 outlier
    
    # ========== 第 3 步：PPO 更新 ==========
    for inner_epoch in range(config.num_inner_epochs):
        for traj, log_prob_old, adv in zip(
            all_trajectories, all_log_probs_old, advantages
        ):
            for j in range(len(log_prob_old)):  # 遍历每一步
                # 重新计算当前模型下的 log_prob
                # 关键：用旧轨迹的 x_j+1，但用当前模型的 v_theta
                _, log_prob_new, _ = compute_log_prob_current(
                    model, traj, j
                )
                
                # PPO ratio
                ratio = torch.exp(log_prob_new - log_prob_old[j])
                
                # PPO-clip loss
                unclipped_loss = -adv * ratio
                clipped_loss = -adv * torch.clamp(
                    ratio, 1.0 - eps, 1.0 + eps
                )
                policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
                
                # KL 惩罚
                if beta > 0:
                    with torch.no_grad():
                        _, _, mean_ref = sde_step_with_logprob(
                            ref_model, traj[j], t, dt
                        )
                        _, _, mean_cur = sde_step_with_logprob(
                            model, traj[j], t, dt
                        )
                    kl_loss = ((mean_cur - mean_ref)**2).mean() / (2 * std**2)
                    loss = policy_loss + beta * kl_loss
                else:
                    loss = policy_loss
                
                # 反向传播（只更新 LoRA）
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    return advantages.mean(), rewards_tensor.mean()
```

### 2.4 计算图与梯度流

```
loss (PPO)
  → log_prob_new（高斯 log_prob 对 mean 求导）
    → mean（SDE 均值公式，对 v_theta 线性）
      → v_theta（DiT 前向输出）
        → LoRA 参数（梯度只落在 LoRA A/B 矩阵上）

关键：v_theta 是 DiT 的输出，mean 是 v_theta 的线性函数，
所以 d(log_prob)/d(v_theta) 可以手推闭式解。
```

---

## 算法 3：VLA 轨迹生成（RiskField-VLA）

### 3.1 概念回顾

VLA = 视觉 + 语言 → 动作轨迹。在 RiskField-VLA 中，额外引入风险场来优化轨迹生成。

### 3.2 Agent 级感知与风险场构建

```python
class RiskFieldBuilder(nn.Module):
    """构建时空风险场"""
    def __init__(self, num_queries=128, num_agents=128, 
                 risk_grid=(32, 32, 8)):
        super().__init__()
        self.num_queries = num_queries
        self.grid_h, self.grid_w, self.grid_t = risk_grid
        
        # Object Query：128 个可学习的查询向量
        self.object_queries = nn.Parameter(
            torch.randn(num_queries, 256)
        )
        
        # α 软门控网络：决定每个 agent 对风险场的贡献
        self.alpha_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 风险场解码器
        self.risk_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.grid_h * self.grid_w * self.grid_t)
        )
    
    def forward(self, camera_features):
        """
        camera_features: (B, num_cameras, H, W, C) 8 路相机特征
        return: 
          agent_features: (B, 128, 256) 每个 agent 的特征
          risk_field: (B, 32, 32, 8) 连续风险场
          z_risk: (B, 256) 风险隐编码
        """
        # 1. 用 object query 从相机特征中提取 agent 特征
        # cross-attention: query 关注 feature map
        agent_features = cross_attention(
            self.object_queries,  # (128, 256)
            camera_features       # (B, ..., C)
        )  # (B, 128, 256)
        
        # 2. 构建交互图：128 个节点，两两计算关系
        interaction_graph = build_interaction_graph(agent_features)
        # (B, 128, 128) 邻接矩阵，表示交互强度
        
        # 3. α 软门控聚合：每个 agent 的贡献权重
        alpha = self.alpha_gate(agent_features)  # (B, 128, 1)
        alpha = torch.softmax(alpha, dim=1)       # 归一化
        
        # 4. 加权聚合生成风险场
        weighted_features = (alpha * agent_features).sum(dim=1)  # (B, 256)
        risk_field = self.risk_decoder(weighted_features)        # (B, 32*32*8)
        risk_field = risk_field.view(-1, 32, 32, 8)             # (B, 32, 32, 8)
        
        # 5. 风险隐编码
        z_risk = weighted_features  # (B, 256)
        
        return agent_features, risk_field, z_risk
```

### 3.3 Risk-Init-Flow 轨迹生成

```python
class RiskInitFlowGenerator(nn.Module):
    """带风险初始化的 Flow Matching 轨迹生成器"""
    def __init__(self, traj_dim=2, hidden_dim=256):
        super().__init__()
        # 标准 Flow Matching 网络
        self.velocity_net = DiT(input_dim=traj_dim, hidden_dim=hidden_dim)
        # 风险编码 → 初始噪声的映射
        self.risk_to_noise = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, traj_dim * traj_seq_len)
        )
        # 双专家
        self.base_expert = self.velocity_net
        self.lte_expert = copy.deepcopy(self.velocity_net)  # Long-Tail-Expert
        # 风险熵 → 门控
        self.gate = nn.Linear(1, 2)  # 输入熵，输出两个专家的权重
    
    def forward(self, scene_context, z_risk, risk_field, num_candidates=8):
        """
        生成多条候选轨迹
        scene_context: 场景上下文（感知特征+导航指令）
        z_risk: 风险隐编码
        risk_field: 风险场
        return: (B, K, traj_seq_len, 2) K 条候选轨迹
        """
        B = scene_context.shape[0]
        
        # === Risk-Init-Flow：用风险编码优化初始噪声 ===
        # 标准 FM: noise = torch.randn(...)  纯随机
        # 我们:   noise = risk_to_noise(z_risk) + 少量随机
        risk_init = self.risk_to_noise(z_risk)  # (B, traj_seq_len * traj_dim)
        risk_init = risk_init.view(B, traj_seq_len, traj_dim)
        noise = risk_init + 0.1 * torch.randn_like(risk_init)  # 加少量随机保持多样性
        
        # === 风险熵门控 ===
        risk_entropy = -risk_field.flatten(1) * torch.log(
            risk_field.flatten(1) + 1e-8
        ).sum(dim=1, keepdim=True)  # (B, 1) 风险场熵
        gate_weights = torch.softmax(self.gate(risk_entropy), dim=-1)  # (B, 2)
        
        # === ODE 积分生成 K 条轨迹 ===
        candidates = []
        for k in range(num_candidates):
            x = noise.clone()
            for step in range(num_steps):  # 32 步 Euler
                t = torch.full((B,), step / num_steps, device=x.device)
                
                # 双专家混合
                v_base = self.base_expert(x, t, scene_context)
                v_lte = self.lte_expert(x, t, scene_context)
                
                # 门控混合
                v = gate_weights[:, 0:1, 0] * v_base + gate_weights[:, 1:2, 0] * v_lte
                
                # Euler 积分
                x = x + v * dt
            
            candidates.append(x)
        
        return torch.stack(candidates, dim=1)  # (B, K, seq, 2)
```

---

## 算法 4：WAM 联合动作-后果流（TrustDrive-WAM）

### 4.1 概念回顾

传统：先生成轨迹 → 后接评分器（割裂）
WAM：生成轨迹的同时，同步演化"后果、风险、支持域"四类隐表征（联合）

### 4.2 联合动作-后果流伪代码

```python
class JointActionConsequenceFlow(nn.Module):
    """联合动作-后果流：轨迹和后果同步生成"""
    def __init__(self, traj_dim=2, consequence_dim=64):
        super().__init__()
        # 统一的速度场网络：同时输出轨迹速度和后果速度
        self.joint_velocity_net = nn.Sequential(
            nn.Linear(traj_dim + consequence_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, traj_dim + consequence_dim)
        )
        # 四类隐表征：轨迹、后果、风险、支持域
        self.traj_embed = nn.Linear(traj_dim, consequence_dim)
        self.consequence_embed = nn.Linear(consequence_dim, consequence_dim)
        self.risk_embed = nn.Linear(consequence_dim, consequence_dim)
        self.support_embed = nn.Linear(consequence_dim, consequence_dim)
        
        # 行为模式 token（16 组）
        self.behavior_tokens = nn.Parameter(torch.randn(16, consequence_dim))
    
    def forward(self, scene_context, behavior_token_idx):
        """
        同步生成轨迹和后果
        """
        # 初始状态：轨迹=噪声，后果=行为token
        traj = torch.randn(B, traj_seq_len, traj_dim)
        consequence = self.behavior_tokens[behavior_token_idx]  # (B, consequence_dim)
        risk = torch.zeros(B, consequence_dim)
        support = torch.ones(B, consequence_dim)  # 初始假设在支持域内
        
        # 联合 ODE 积分：四类表征同步演化
        for step in range(num_steps):
            t = step / num_steps
            
            # 拼接所有状态
            joint_state = torch.cat([
                traj.flatten(1), consequence, risk, support
            ], dim=-1)
            
            # 统一网络预测速度
            v = self.joint_velocity_net(
                torch.cat([joint_state, t], dim=-1)
            )
            
            # 拆分并更新四类表征
            v_traj = v[:, :traj_dim*traj_seq_len].view(B, traj_seq_len, traj_dim)
            v_consequence = v[:, traj_dim*traj_seq_len:traj_dim*traj_seq_len+consequence_dim]
            
            traj = traj + v_traj * dt
            consequence = consequence + v_consequence * dt
            # risk 和 support 同步演化...
        
        # 逆一致性约束：正向预测后果 → 逆向恢复轨迹 → 应该一致
        recovered_traj = self.inverse_predict(consequence)
        consistency_loss = F.mse_loss(traj, recovered_traj.detach())
        
        return {
            'traj': traj,
            'consequence': consequence,
            'risk': risk,
            'support_domain': support,
            'consistency_loss': consistency_loss
        }
```

### 4.3 可信路由（Trust Router）

```python
class RiskBoundedTrustRouter:
    """风险约束可信路由：决定是否信任世界模型的预测"""
    def __init__(self, support_threshold=0.8, risk_threshold=0.5):
        self.support_threshold = support_threshold
        self.risk_threshold = risk_threshold
    
    def route(self, candidate_trajs, consequences, support_scores, risk_scores):
        """
        双层信任评估：
        1. 可靠性信任：预测是否落在支持域内
        2. 决策信任：预测是否带来性能增益
        """
        decisions = []
        for traj, cons, support, risk in zip(
            candidate_trajs, consequences, support_scores, risk_scores
        ):
            # 第一层：可靠性信任校验
            is_reliable = support > self.support_threshold
            
            # 第二层：决策信任评估
            utility = compute_utility(traj)  # 效用
            is_beneficial = utility > baseline_utility
            
            # 联合打分
            trust_score = (
                0.4 * support +      # 支持域
                0.3 * (1 - risk) +   # 碰撞风险（越低越好）
                0.3 * utility        # 效用
            )
            
            if is_reliable and is_beneficial:
                decisions.append(('USE_WM', traj))  # 信任世界模型
            else:
                decisions.append(('FALLBACK', baseline_traj))  # 切回传统规划
        
        return decisions
```

---

## 算法 5：JEPA 完整实现（JEPA-DRIVE）

### 5.1 概念回顾

JEPA = 在表征空间预测，不重建像素。三个组件：Context Encoder、Target Encoder（EMA）、Predictor。

### 5.2 JEPA 训练伪代码

```python
import torch
import torch.nn as nn

class JEPA(nn.Module):
    """JEPA 联合嵌入预测架构"""
    def __init__(self, encoder_dim=768, predictor_dim=1024):
        super().__init__()
        # Context Encoder：可训练，有梯度
        self.context_encoder = ViT(patch_size=16, embed_dim=encoder_dim)
        
        # Target Encoder：EMA 更新，无梯度
        self.target_encoder = ViT(patch_size=16, embed_dim=encoder_dim)
        
        # Predictor：输入上下文表征+位置条件，预测目标表征
        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim + pos_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, encoder_dim)
        )
        
        # EMA momentum
        self.ema_momentum = 0.996
    
    def forward(self, images, mask):
        """
        images: (B, 3, H, W) 输入图片
        mask:   (B, num_patches) bool，True = 被挖掉的目标块
        """
        B, C, H, W = images.shape
        num_patches = (H // 16) * (W // 16)
        
        # 第 1 步：切 patch
        patches = patchify(images, patch_size=16)  # (B, num_patches, dim)
        
        # 第 2 步：分离上下文和目标
        context_patches = patches[~mask]   # 可见 patch
        target_patches = patches[mask]      # 被遮住的 patch
        
        # 第 3 步：Context Encoder（有梯度）
        context_repr = self.context_encoder(context_patches)  # (B, num_ctx, dim)
        
        # 第 4 步：Target Encoder（无梯度，EMA 更新）
        with torch.no_grad():  # stop-gradient！
            target_repr = self.target_encoder(target_patches)  # (B, num_tgt, dim)
            target_repr_pooled = target_repr.mean(dim=1)        # (B, dim)
        
        # 第 5 步：Predictor 预测目标表征
        context_pooled = context_repr.mean(dim=1)  # (B, dim)
        pos_tokens = get_position_embeddings(mask)  # 位置条件
        pred_input = torch.cat([context_pooled, pos_tokens], dim=-1)
        pred_repr = self.predictor(pred_input)  # (B, dim)
        
        # 第 6 步：算 Loss（表征空间 MSE）
        loss = F.mse_loss(pred_repr, target_repr_pooled)
        
        return loss
    
    def update_ema(self):
        """EMA 更新 Target Encoder"""
        for param_t, param_s in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            param_t.data = (
                self.ema_momentum * param_t.data + 
                (1 - self.ema_momentum) * param_s.data
            )
```

### 5.3 JEPA-DRIVE 的三阶段训练

```python
# ============ 阶段 1：JEPA 自监督预训练 ============
def stage1_jepa_pretrain(jepa_model, driving_videos):
    """学习时序场景未来隐表征预测，规避像素级重建"""
    for video_batch in driving_videos:
        frames = video_batch['frames']  # (B, T, 3, H, W)
        
        # 帧间预测：用第 t 帧预测第 t+1 帧的表征
        for t in range(T - 1):
            frame_t = frames[:, t]
            frame_next = frames[:, t + 1]
            
            # 掩码当前帧的一部分
            mask = random_mask(num_patches, ratio=0.75)
            
            # JEPA 预测
            loss = jepa_model(frame_t, mask)  # 预测被遮住部分的表征
            
            loss.backward()
            optimizer.step()
        
        # EMA 更新
        jepa_model.update_ema()

# ============ 阶段 2：冻结世界模型，训练隐空间评分头 ============
def stage2_train_scorer(jepa_model, scorer_head, navsim_data):
    """输出与 PDMS 安全指标对齐"""
    # 冻结世界模型
    for param in jepa_model.parameters():
        param.requires_grad = False
    
    for batch in navsim_data:
        with torch.no_grad():
            scene_repr = jepa_model.context_encoder(batch['images'])
        
        # 评分头：预测每条轨迹的 PDMS 分数
        pred_scores = scorer_head(scene_repr, batch['candidate_trajs'])
        gt_scores = batch['pdms_scores']
        
        # 排序学习 loss（对齐 NAVSIM 官方指标）
        loss = ranking_loss(pred_scores, gt_scores)
        loss.backward()
        optimizer_scorer.step()

# ============ 阶段 3：端到端联合调优 ============
def stage3_end2end_finetune(full_model, navsim_data):
    """释放感知分支，感知-世界模型-生成器-评分器联合调优"""
    # 解冻感知分支
    for param in full_model.perception.parameters():
        param.requires_grad = True
    
    for batch in navsim_data:
        # 完整前向：感知 → 世界模型 → 轨迹生成 → 评分
        scene_repr = full_model.perception(batch['images'])
        future_repr = full_model.world_model(scene_repr)  # 预测未来表征
        trajs = full_model.generator(future_repr)         # 生成密集轨迹
        scores = full_model.scorer(trajs)                  # 打分
        
        # 多任务 loss
        loss = (
            F.mse_loss(scores, batch['pdms_scores']) +  # 评分对齐
            0.1 * jepa_loss +                           # JEPA 正则
            0.05 * vicreg_loss                          # 防坍缩
        )
        loss.backward()
        optimizer.step()
```

### 5.4 VICReg 正则（防表征坍缩）

```python
def vicreg_loss(representations):
    """
    VICReg: Variance-Invariance-Covariance Regularization
    三项约束防止表征坍缩
    """
    # 1. 方差项：每个维度的方差要足够大（不能是常数）
    var = representations.var(dim=0)
    variance_loss = F.relu(1.0 - var).mean()  # 方差 < 1 时惩罚
    
    # 2. 协方差项：不同维度之间要去相关
    cov = torch.cov(representations.T)
    off_diagonal = cov - torch.diag(torch.diag(cov))
    covariance_loss = (off_diagonal**2).sum() / representations.shape[1]
    
    # 3. 不变项：同一图片的不同增强应该有相似的表征
    # （用 VICReg 的 invariance 项）
    
    return variance_loss + 0.04 * covariance_loss
```

### 5.5 Cycle-Energy 双空间自监督损失

```python
def cycle_energy_loss(jepa_model, frame_a, frame_b):
    """
    Cycle-Energy: 双空间自监督
    正向：A 的表征 → 预测 B 的表征
    逆向：B 的表征 → 预测 A 的表征
    双向一致 = 好的表征
    """
    # 正向
    repr_a = jepa_model.context_encoder(frame_a)
    pred_b = jepa_model.predictor(repr_a)
    target_b = jepa_model.target_encoder(frame_b)  # EMA, no grad
    
    # 逆向
    repr_b = jepa_model.context_encoder(frame_b)
    pred_a = jepa_model.predictor(repr_b)
    target_a = jepa_model.target_encoder(frame_a)
    
    # 双向 cycle loss
    forward_loss = F.mse_loss(pred_b, target_b.detach())
    backward_loss = F.mse_loss(pred_a, target_a.detach())
    
    cycle_loss = forward_loss + backward_loss
    return cycle_loss
```

---

# 第三部分：面试压力面问题与参考回答

> 这一部分模拟真实面试场景，覆盖概念、原理、项目细节、开放性问题。每个问题给出"标准回答"和"加分回答"。

---

## A. Flow Matching 相关

**Q1：Flow Matching 和 DDPM 有什么区别？为什么你选 Flow Matching？**

> **标准回答**：DDPM 是弯路去噪，前向过程逐步加噪形成随机路径，反向需要 20-50 步逐步退回。Flow Matching 是直线插值，在噪声和数据之间拉一条直线，网络学习这条线上的速度场，推理只需 5-10 步。我们选 Flow Matching 是因为 GRPO 需要反复采样（每个场景 24 条轨迹），采样效率差 5 倍意味着 RL 训练也慢 5 倍。
>
> **加分回答**：补充数学区别——DDPM 训练目标是预测噪声 ||ε_θ - ε||²，Flow Matching 是预测速度 ||v_θ - (x₁-x₀)||²。DDPM 基于 SDE，FM 基于 ODE。另外 FM 的速度场几乎恒定（= x₁-x₀），学起来更容易收敛。

**Q2：Flow Matching 推理时为什么用 Euler 积分而不是更高阶的积分器？**

> **标准回答**：Euler 积分实现简单、速度快，而且 Flow Matching 的路径是直线，速度场几乎恒定，低阶积分器的精度损失很小。更高阶的积分器（如 RK4）每步需要多次网络前向，计算量翻倍但收益不大。
>
> **加分回答**：实际工程中会根据步数权衡——步数多时 Euler 就够了，步数极少时（如 1 步蒸馏）可能需要更精确的积分。我们的方案是 32 步 Euler，平衡了精度和速度。

**Q3：如果 Flow Matching 生成的轨迹很"平庸"（多模态坍缩），怎么解决？**

> **标准回答**：多模态坍缩是指模型只生成"平均"轨迹，丢掉了多种可能性。解决方案有：1）增加采样的随机性（SDE 而不是 ODE）；2）条件化——给不同的行为 token（我们的项目用了 16 组可学习行为模式 token）；3）对抗训练或 mode-seeking loss。
>
> **加分回答**：在 TrustDrive-WAM 中，我们引入 16 组可学习行为模式 token 作为生成条件，引导模型输出覆盖保守跟车、加速通行、侧向偏移等不同驾驶假设的 16 条多样化候选轨迹，从条件端解决模式坍缩。

---

## B. Flow-GRPO 相关

**Q4：GRPO 最早是给 LLM 设计的，你怎么把它迁移到 Flow Matching 上？**

> **标准回答**：核心挑战是 log_prob 的计算方式不同。LLM 中 token 是离散的，log_prob 直接从 softmax 取 log。但 Flow Matching 是连续的 ODE，确定性系统没有转移概率。我们的解法是引入 SDE，把确定性 ODE 变成有随机性的 SDE，这样转移概率变成高斯分布，log_prob 就是高斯 log_prob，直接可算。
>
> **加分回答**：具体推导分四步：1）用 Fokker-Planck 方程反解 SDE 漂移项，保证边际分布不变；2）在整流流下用速度场替换 score（闭式恒等式）；3）欧拉-马鲁亚马离散化；4）得到带 score 修正的均值公式和高斯 log_prob。这样 PPO 的 ratio 就能算了。

**Q5：PPO-clip 的 clip range 设多大？为什么？**

> **标准回答**：我们设 0.2，这是 PPO 的经典默认值。clip 的作用是限制新旧策略的偏离不超过 ±20%。太大容易训练不稳定，太小学习太慢。
>
> **加分回答**：在 Flow-GRPO 中，clip 和 KL 惩罚是双重保护。clip 限制单步更新幅度，KL 惩罚限制整体偏离 reference model 的程度。两者缺一不可——只有 clip 没有 KL，长时间训练还是会漂移；只有 KL 没有 clip，单步可能更新太猛导致崩溃。

**Q6：为什么只微调 LoRA？全参微调不行吗？**

> **标准回答**：LoRA 只更新 0.5% 的参数（约 60M），原始权重全部冻结。好处是：1）训练快、显存省；2）不破坏预训练知识；3）可以多个任务各挂一个 LoRA。
>
> **加分回答**：全参微调 12B 模型需要大量显存和数据，而且容易灾难性遗忘——模型会忘记 Flow Matching 预训练学到的生成能力。LoRA 的低秩分解 W = W₀ + (α/r)·BA 本质上是在已有的速度场上叠加一个低秩"微调偏移"，GRPO 的梯度告诉这个偏移往 reward 高的方向挪，原始权重不动。

**Q7：奖励模型怎么设计的？会不会被 hack？**

> **标准回答**：奖励模型基于 NAVSIM 官方 PDM 打分器，包括碰撞、可行驶区域、TTC、舒适度等子指标。hack 的风险是存在的——模型可能找到"钻空子"的方式（比如永远不动车来避免碰撞）。
>
> **加分回答**：防 hack 的手段有三个：1）EP（Ego Progress）指标要求自车必须前进，防止"不动"策略；2）clip + KL 双重约束限制偏离；3）组内归一化——只看相对好坏，不看绝对分数，即使奖励模型整体偏移也不影响训练方向。

---

## C. VLA 与风险场相关

**Q8：为什么用 128 个 object query？怎么确定的？**

> **标准回答**：128 是通过实验确定的。太少（如 32）不够覆盖场景中的所有交通参与者，太多（如 512）计算量大且很多 query 是冗余的。实际场景中同时交互的目标通常不超过 50-100 个。
>
> **加分回答**：128 个 query 对应 128 节点交互图，可以显式建模两两之间的博弈关系（邻接矩阵 128×128）。这比稀疏 token 方案更能捕捉"这辆车让我"还是"那辆车抢行"这种交互语义。

**Q9：风险场和传统的 occupancy prediction 有什么区别？**

> **标准回答**：Occupancy prediction 预测每个格子是否被占用（0/1），是几何层面的。风险场预测每个格子的碰撞风险（连续值），是**交互层面**的——考虑了其他交通参与者的意图和博弈关系。
>
> **加分回答**：风险场多了三个维度的信息：1）**时间演化**（32×32×8，未来 8 步的风险变化）；2）**交互关系**（通过 α 软门控聚合所有 agent 的影响）；3）**风险语义**（z_risk 编码可以直接给下游用，不只是占用/空闲）。

**Q10：Base 和 LTE 双专家怎么训练？数据不均衡怎么办？**

> **标准回答**：Base 专家在全量数据上训练，LTE 只在长尾高危样本上训练。门控网络用风险场熵作为输入，学习"什么时候该用哪个专家"。
>
> **加分回答**：长尾数据天然稀少（21%），直接训练 LTE 会过拟合。我们的做法是：1）SFT 阶段引入在线难例挖掘，自动筛选高难度样本；2）LTE 初始化自 Base 的权重（不是从零开始）；3）门控用熵而非硬标签，允许两个专家的输出平滑混合。

---

## D. WAM 与可信域相关

**Q11：为什么说"世界模型预测天然可信"是一个缺陷？**

> **标准回答**：传统方案默认世界模型的预测是准确的，直接拿来做规划。但在 OOD 输入下（训练时没遇到过的场景），世界模型的预测可能偏差很大，而规划器不知道这个预测不可靠，会"盲目相信"它，导致碰撞或越界。
>
> **加分回答**：类比一个人做判断——如果你不知道自己的知识边界，就会在不懂的领域自信地给出错误答案。我们方案的核心是把世界模型预测视为"决策证据"而非"裁决依据"，增加一个可信度评估层，知道自己什么时候不知道。

**Q12：逆一致性约束具体怎么实现的？**

> **标准回答**：正向：给定状态 s 和动作 a，预测下一状态 s'。逆向：给定 s' 和动作 a'，尝试恢复 s。约束要求正向和逆向的结果一致。
>
> **加分回答**：如果不加这个约束，模型可能完全忽略动作输入——因为忽略动作也能让正向 loss 很小（只依赖当前状态）。逆一致性强迫模型"真正使用"动作信息，因为只有用了动作，正逆才能闭环。这在数学上类似 cycle consistency loss（CycleGAN 的思想）。

**Q13：可信路由在实际部署中怎么保证实时性？**

> **标准回答**：可信路由本身是一个轻量级的打分操作（几个 MLP 层），计算量可以忽略。主要开销在世界模型的前向推理，但我们可以做条件计算——只有可信的场景才完整推理世界模型，不可信的直接走传统规划。
>
> **加分回答**：实际上做的是"先评估再推理"：1）先用支持域检测（轻量）判断是否在训练分布内；2）在分布内才激活世界模型；3）不在分布内直接 fallback。这样大部分常规场景走快速路径，只有需要时才走完整路径。

---

## E. JEPA 相关

**Q14：JEPA 和 MAE（Masked Autoencoder）有什么区别？**

> **标准回答**：MAE 在像素空间重建被遮住的部分，JEPA 在表征空间预测被遮住部分的抽象表征。MAE 会花大量算力在无关细节（纹理、光照）上，JEPA 只关注语义。
>
> **加分回答**：用考试类比——MAE 要求你把被遮住的每个像素都画出来（难且大部分没用），JEPA 只要求你回答"被遮住的是什么"（抓核心语义）。实测在 linear probing 上 JEPA 逼近对比学习，但在计数、深度预测等低层任务上反超 MAE。

**Q15：表征坍缩怎么防止？EMA 和 stop-gradient 分别起什么作用？**

> **标准回答**：表征坍缩是所有输出都变成常向量，loss=0 但没学到东西。EMA 让 Target Encoder 变化很慢，stop-gradient 阻止梯度传给 Target Encoder。这样 Context Encoder 必须"真正理解"才能预测 Target 的输出。
>
> **加分回答**：类比师生游戏——如果老师配合学生（一起被训练），老师会悄悄把谜底放简单，学生什么都没学到。EMA + stop-grad 让老师不动，学生只能真正去猜。我们项目还额外用了 VICReg 正则（方差+协方差约束）作为第三道防线。

**Q16：为什么用 JEPA 而不是直接用扩散模型做世界模型？**

> **标准回答**：三个原因：1）**效率**——不生成像素，只预测一个向量，快得多；2）**物理一致性**——像素级生成可能画出穿墙的车，表征空间可以注入几何先验；3）**可解释性**——表征可以直接喂给规划头，不需要先解码成像素再读语义。
>
> **加分回答**：从历史脉络看——扩散模型兴起后，人们用它生成视频做世界模型（如 DriveDreamer），但计算太贵。JEPA 提出"表征空间预测"替代方案，保留了预测能力但丢掉了像素重建。这是 LeCun 一直倡导的方向：预测抽象状态，不预测无关细节。

**Q17：65536 条密集轨迹是怎么生成的？选哪条？**

> **标准回答**：用 Path/Velocity 因子化解耦生成器——把轨迹分解为"路径"（空间形状）和"速度"（时间分配）两个独立维度，分别生成后组合。65536 = 256 条路径 × 256 种速度模式。选择用 JEPA 隐空间评分器，融合双 Cycle-Energy 信号，用排序学习对齐 NAVSIM PDMS 指标。
>
> **加分回答**：因子化解耦的好处是组合爆炸——路径和速度独立生成，可以覆盖远超朴素枚举的轨迹空间。评分在隐空间完成（不生成像素），所以即使 65536 条也很快。排序学习比回归更鲁棒，因为 PDMS 是乘积指标，绝对数值不重要，相对排序才重要。

---

## F. 综合与开放性问题

**Q18：你的三个项目之间有什么关系？**

> **标准回答**：三个项目是一条递进的技术路线：
> - **RiskField-VLA**：用风险场增强 VLA 的感知，用 Flow Matching 生成轨迹，用 Flow-GRPO 做后训练——解决"生成什么轨迹"的问题
> - **TrustDrive-WAM**：加入世界模型，联合建模动作和后果，增加可信路由——解决"轨迹后果如何评估"的问题
> - **JEPA-DRIVE**：用 JEPA 替代像素世界模型，在表征空间完成全部预测——解决"世界模型如何高效且物理可信"的问题
>
> **加分回答**：从优化目标看——VLA 关注"生成更好的轨迹"，WAM 关注"更准确地评估后果"，JEPA 关注"更高效地预测未来"。三者合起来就是"感知-预测-生成-评估"的完整闭环。

**Q19：如果让你重新做这三个项目，你会改什么？**

> **标准回答**：1）RiskField-VLA 的风险场是手工设计的特征，我会尝试端到端学习风险表征；2）TrustDrive-WAM 的可信路由阈值是手动调的，我会用元学习自动学；3）JEPA-DRIVE 的三阶段训练比较复杂，我会尝试更紧密的联合训练。
>
> **加分回答**：更宏观地说——三个项目都在 NAVSIM 这一个 benchmark 上，我会增加真实路测的闭环评测。另外，三个项目的奖励/评分函数都是基于规则的（PDM 打分器），长期看应该用学习的奖励模型（RLHF 式）。

**Q20：你怎么看端到端自动驾驶的未来？**

> **标准回答**：端到端是趋势，因为它数据驱动、上限高。但纯端到端缺乏可解释性和安全保障，所以需要世界模型提供"想象"能力，需要可信域提供安全边界。
>
> **加分回答**：我认为未来是"端到端 + 世界模型 + 强化学习"的三位一体——端到端做感知和生成，世界模型做后果预测和安全验证，强化学习做持续优化。我的三个项目正好覆盖了这三个方向，这也是我认为最有前途的技术路线。

**Q21：（压力面）你的 PDMS 0.8713，但你说 21% corner case 分数趋近于 0，那你的提升到底在哪？**

> **标准回答**：0.8713 是整体 PDMS，corner case 那 21% 单独看确实是 0 分，但我们的方案把这部分从 0 提升到了一个正数，所以整体分数从基线的约 0.75 提升到 0.87。常规场景的 79% 我们没有损失性能。
>
> **加分回答**：更准确地说——基线在 corner case 上得分接近 0（因为乘积指标，碰撞就归零），我们的风险场方案让模型"看到"了这些风险，生成了避让轨迹，把这部分从 0 拉到了 0.5-0.6 左右。整体提升的贡献主要来自这 21% 的修复，而不是常规场景的微调。

**Q22：（压力面）你用 LoRA 微调，是不是说明你的方法不够强？全参微调是不是更好？**

> **标准回答**：LoRA 不是"妥协"，而是"选择"。全参微调确实可能更强，但会破坏预训练知识、需要更多数据、显存开销大。在 RL 后训练场景下，我们只微调 0.5% 参数就达到了显著提升，说明方法本身是有效的。
>
> **加分回答**：而且 RL 后训练和 SFT 不同——SFT 需要大量数据来覆盖任务分布，RL 只需要 reward 信号。LoRA 的低秩假设（任务改动是低秩的）在 RL 场景下更成立，因为 reward 引导的改动往往集中在少数关键方向。如果全参微调，反而容易 reward hacking。

**Q23：（压力面）Flow-GRPO 论文是腾讯 ARC Lab 的，你只是复现，有什么创新？**

> **标准回答**：我们不是简单复现——我们把 Flow-GRPO 从图像生成迁移到了自动驾驶轨迹生成，做了三个适配：1）ODE 转 SDE 的轨迹采样求解 log-prob；2）PPO-clip + KL 约束的精细化策略优化；3）PDM + Agent 级归因打分器的双闭环监督。还引入了在线难例挖掘。
>
> **加分回答**：论文的 Flow-GRPO 是在 FLUX 上做的文生图，我们的场景完全不同——轨迹是低维的（2D），奖励是乘积指标（PDMS），场景有长尾分布。直接套用会失败，需要针对这些特点做算法适配。这就是我们的贡献。

**Q24：（压力面）JEPA-DRIVE 的 PDMS 0.7285 比另外两个项目低很多，是不是 JEPA 不行？**

> **标准回答**：0.7285 是基线版本，还没有做完整的 Flow-GRPO 后训练。而且 JEPA-DRIVE 是全自监督训练、无需人工标注的，对比的公平性不同。另外三个项目的目标不同——JEPA-DRIVE 重点验证的是"表征空间预测范式是否可行"，不是追求最高分。
>
> **加分回答**：坦率说，JEPA 路线目前确实不如像素生成成熟，主要瓶颈是：1）隐空间评分器的对齐精度；2）65536 条轨迹的选择效率；3）跨模态对齐的稳定性。但它的优势——效率、物理一致性、无标注——是另外两个项目不具备的。0.7285 证明了范式可行，后续加 RL 后训练有明确的提升空间。

**Q25：（压力面）如果让你在真实车上部署，你觉得最大的挑战是什么？**

> **标准回答**：三个挑战：1）**实时性**——Flow Matching 推理 32 步 + GRPO 的 SDE 采样，在车端算力上可能不够；2）**安全性**——任何生成模型都可能出错，需要可信路由做兜底；3）**数据分布**——训练数据和真实路况的 gap，特别是 OOD 场景。
>
> **加分回答**：我认为最核心的挑战是**验证**——如何证明系统在所有场景下都安全？NAVSIM 是开环评测，真实是闭环的。我们的可信域方案是应对这个挑战的一种思路（知道自己什么时候不确定），但离完整的安全论证还有距离。这也是为什么行业还在 L2+ 徘徊，真正的 L4 需要更严格的安全框架。

---

## G. 快速自测清单

面试前快速过一遍：

- [ ] Flow Matching 的训练 loss 和推理过程？
- [ ] DDPM vs Flow Matching 的三个核心区别？
- [ ] GRPO 的 advantage 怎么算？为什么不用 critic？
- [ ] Flow-GRPO 如何解决连续空间 log_prob 不可算的问题？
- [ ] PPO-clip 的作用和原理？
- [ ] LoRA 的数学公式和为什么只更新 0.5% 参数？
- [ ] VLA 的三个模态分别是什么？
- [ ] Object Query 的作用和 128 个怎么用？
- [ ] 风险场和 occupancy 的区别？
- [ ] Base + LTE 双专家的门控机制？
- [ ] WAM 的"联合动作-后果流"是什么意思？
- [ ] 可信域/支持域的定义和作用？
- [ ] 逆一致性约束防止什么问题？
- [ ] JEPA 和 MAE 的核心区别？
- [ ] EMA + stop-gradient 如何防止表征坍缩？
- [ ] I-JEPA vs V-JEPA 的区别？
- [ ] 三个项目的技术路线和相互关系？
- [ ] PDMS 的乘积指标意味着什么？
- [ ] OOD 场景下世界模型为什么危险？
- [ ] 你的三个项目各自的 PDMS 分数？

---

*这篇文档覆盖了三大项目的全部核心知识点。建议按"第一部分通读 → 第二部分手写伪代码 → 第三部分自问自答"的顺序复习。*
