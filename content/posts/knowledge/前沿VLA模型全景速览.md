---
title: "前沿VLA模型全景速览：从RT-2到GR-3"
date: 2026-07-19
draft: false
categories: ["产业实践"]
tags: ["🤖 VLA", "🏭 产业实践", "🚗 自动驾驶", "🧠 大模型", "📊 模型对比"]
summary: "VLA模型从2023年的RT-2到2026年的GR-3经历了三代演进。本文系统对比15+主流VLA模型的核心设计选择：动作表征（tokenization/连续控制/扩散策略）、视觉编码器架构、LLM backbone选择、以及训练范式（预训练→指令微调→RLHF）。重点分析pi0的Flow Matching动作头、GR-3的CoT推理与自回归架构、BridgeVLA的桥接监督等前沿技术路线。"
weight: 1
---

## 一句话理解VLA模型演进

> **从"看路开车"到"边想边开"——VLA模型三代演进的核心是动作表征从离散token到连续扩散再到流匹配的精度跃迁，以及推理能力从零到一的质变。**

VLA（Vision-Language-Action）模型在2023–2026年间经历了三代技术路线的迭代。如果把RT-2看作"把驾驶当成翻译任务"的第一代，RT-2-X和EmbodiedGPT就是把动作空间从分类扩展为结构化序列的第二代；而pi0的Flow Matching动作头、GR-3的Chain-of-Thought推理则标志着第三代——模型开始真正"思考"如何驾驶。本文不讨论VLA的基础定义（参见《什么是VLA模型》），而是聚焦于工程实现层面的具体设计选择，帮助研究者在面对一个新VLA任务时做出正确的技术选型。

---

## 1. 第一代VLA：离散动作Tokenization（2023–2024）

### 1.1 RT-2：将驾驶建模为翻译任务

RT-2（Robotic Transformer 2）由Google DeepMind于2023年7月发布，是第一个真正意义上的VLA模型。其核心思路极其直接：把视觉输入通过PaLI-X编码为视觉token序列，然后让PaLM-E大语言模型以自回归方式输出离散动作token。

RT-2的动作空间设计是理解其能力边界的起点。它将机器人的连续控制信号（6-DoF末端执行器位姿 + 夹爪开合）离散化为256个bin，每个bin对应一个token。这种离散化精度为：平移量每bin约1cm，旋转量每bin约1度。这意味着RT-2的位置控制精度大约在±0.5cm，对于抓取水瓶这种任务足够，但对于精密装配（需要±0.1mm精度）完全不可用。

```python
# RT-2动作离散化的简化实现
class RT2ActionTokenizer:
    def __init__(self, num_bins=256, translation_range=(-0.5, 0.5)):
        self.num_bins = num_bins
        self.bin_edges = np.linspace(translation_range[0], translation_range[1], num_bins + 1)
    
    def discretize(self, continuous_value):
        # 将连续值映射到最近的bin索引
        idx = np.digitize(continuous_value, self.bin_edges) - 1
        return np.clip(idx, 0, self.num_bins - 1)
    
    def reconstruct(self, token_id):
        # 用bin中心值近似连续值
        bin_center = (self.bin_edges[token_id] + self.bin_edges[token_id + 1]) / 2
        return bin_center
```

RT-2的训练管道分为两步：首先在大规模互联网图文数据上预训练视觉-语言模型，然后在机器人操作数据上微调。这个范式的核心问题在于——预训练的图文数据和微调的机器人数据之间存在巨大的分布鸿沟。互联网图片里"杯子"出现在餐桌上的频率远高于出现在操作场景中的频率，导致RT-2在未见过的背景或光照条件下泛化能力骤降。

### 1.2 PaLM-E：具身推理的首次尝试

PaLM-E（PaLM with Embodiment）是与RT-2同期的工作，但设计哲学不同。PaLM-E的核心创新在于：将传感器时间序列、3D场景表示和状态估计等非文本信息直接嵌入LLM的token序列中。具体来说，PaLM-E使用一个ViT编码器将图像映射为视觉token，同时将一个状态编码器将机器人关节角度和夹爪状态映射为状态token，然后与文本token拼接后送入PaLM层。

PaLM-E的参数量级跨越了三个规模：12B、62B和540B。实验结果表明，随着模型规模增大，emergent abilities（如零样本规划、多步推理）显著涌现。62B版本在TableTop操作任务上的成功率为87.6%，而540B版本达到了92.3%。但代价是推理延迟：540B版本在单张A100上生成一个动作序列需要2–3秒，远远无法满足实时控制的要求。

### 1.3 Generation 1对比表

| 模型 | 发布 | Backbone | 动作表征 | 参数量 | 实时性 | 关键创新 |
|:----:|:----:|:--------:|:--------:|:------:|:------:|:--------:|
| RT-2 | 2023.07 | PaLM-E | 256-bin离散 | 12B/55B | ❌ | 首次统一VLA |
| PaLM-E | 2023.03 | PaLM | 离散token | 12B→540B | ❌ | 多模态token嵌入 |
| EmbodiedGPT | 2023.05 | LLaMA | 离散序列 | 7B/13B | △ | 开源VLA尝试 |
| RT-2-X | 2024.01 | PaLM-E | 扩展bin | 55B | ❌ | 跨任务泛化 |

第一代VLA的核心问题在于离散化精度限制和推理速度。RT-2在桌面操作场景的末端执行器位置误差约为1–3cm，而在自动驾驶场景中，1cm的精度误差在高速下可能导致横向偏差达10–30cm。此外，自回归逐token生成需要串行推理，限制了控制频率的上限。

---

## 2. 第二代VLA：连续控制与扩散策略（2024–2025）

### 2.1 π0（Pi-0）与Flow Matching动作头

第二代VLA的标志性进展来自Physical Intelligence公司于2024年10月发布的pi0。pi0的核心创新是用Flow Matching取代离散tokenization来建模连续动作分布。

Flow Matching的基本思想是：定义一个从噪声分布到数据分布的连续概率路径，然后学习一个向量场来沿着这条路径"流"动。对于动作预测，这意味着模型不再输出一个离散的token，而是输出一个连续的动作向量。具体地，pi0对机器人动作的建模方式如下：

设 $$x_0 \sim \mathcal{N}(0, I)$$ 为标准高斯噪声，$$x_1$$ 为目标动作。Flow Matching定义一个随时间 $$t \in [0, 1]$$ 的概率路径 $$\phi_t(x)$$，使得 $$\phi_0(x) = x$$，$$\phi_1(x) = x_1$$。模型学习一个向量场 $$v_\theta(x, t)$$ 来近似这个路径的导数：

$$\frac{d\phi_t(x)}{dt} = v_\theta(\phi_t(x), t)$$

推理时的采样过程从 $$x_0$$ 开始，沿向量场进行数值积分。pi0使用10步采样（基于欧拉法），在每个步长 $$h = 1/10$$ 执行：

$$x_{t+h} = x_t + h \cdot v_\theta(x_t, t)$$

这个过程的计算开销是10次前向传播，但可以通过CFM（Conditional Flow Matching）技术将每次传播融合到单次模型调用中。pi0在模拟器中的实际控制频率达到30Hz，基本接近实时控制要求。

pi0的训练数据集分布值得关注：它在13个不同的机器人平台上收集了超过10万次操作演示，涵盖从叠衣服到插拔充电器的多种精细操作。这个数据量相比RT-2增加了两个数量级，是性能大幅提升的关键。

### 2.2 Octo：开源VLA的标杆

Octo由UC Berkeley的RAIL Lab于2024年发布，是第一个真正可用的开源VLA基础模型。Octo使用扩散策略（Diffusion Policy）作为动作头，在Open X-Embodiment数据集上训练。Octo的设计哲学是模块化：提供一个通用的视觉-语言-动作骨干网络，用户可以通过LoRA微调适配到特定机器人平台。

```python
# Octo的扩散策略动作头简化实现
class OctoDiffusionPolicy(nn.Module):
    def __init__(self, obs_dim=512, action_dim=7, num_diffusion_steps=100):
        super().__init__()
        self.noise_net = UNet(
            in_channels=action_dim,
            cond_dim=obs_dim,
        )
        self.num_steps = num_diffusion_steps
    
    def sample_action(self, obs_embedding):
        # 从标准高斯噪声开始
        action_noise = torch.randn(1, self.action_dim)
        # DDIM采样
        action = action_noise
        for step in reversed(range(self.num_steps)):
            noise_pred = self.noise_net(action, step, obs_embedding)
            action = self.ddim_step(action, noise_pred, step)
        return action
```

Octo在7自由度机械臂上的控制精度达到±0.3mm（统计学意义上的90%分位），相比RT-2提升了一个数量级。但代价是推理延迟：扩散策略需要100步DDIM采样，约120ms，对应8Hz的控制频率——这对于缓慢的桌面操作足够，但对于高速运动任务（如投掷、接取）还有差距。

### 2.3 DriveVLA与RoboFlamingo：自动驾驶场景的VLA

DriveVLA是2024年底将VLA范式引入自动驾驶的代表工作。它的视觉编码器采用BEVFormer提取鸟瞰视角特征，LLM使用Vicuna-7B，动作头采用MLP直接回归连续的转向角、油门和刹车值。DriveVLA在nuScenes数据集上的开环评测中，碰撞率比UniAD降低22%，规划距离误差降低15%。

但DriveVLA面临的核心问题在于分布漂移：由于训练数据来自nuScenes的日志回放，模型学习到的驾驶策略严重依赖于标注数据中的驾驶员行为。当在闭环仿真中长时间运行时（超过30秒），模型容易偏离正常轨迹，最终发生碰撞。这是所有开环训练模型共有的covariate shift问题。

### 2.4 Generation 2对比表

| 模型 | 发布 | 动作表征 | 步长(ms) | 频率 | 精度 | 数据量 |
|:----:|:----:|:--------:|:--------:|:----:|:----:|:------:|
| pi0 | 2024.10 | Flow Matching | 33 | 30Hz | ±0.1mm | 100K |
| Octo | 2024.08 | 扩散策略 | 120 | 8Hz | ±0.3mm | 500K |
| DriveVLA | 2024.11 | MLP回归 | 10 | 100Hz | ±10cm | 40K |
| RoboFlamingo | 2024.09 | MLP回归 | 15 | 66Hz | ±5mm | 80K |
| GR-1 | 2024.12 | 扩散策略 | 80 | 12Hz | ±0.5mm | 200K |

第二代VLA在动作精度和推理速度上相比第一代有了质的提升，但还没有解决"推理"问题。模型仍然是直接从观测映射到动作（perception → action），没有显式的推理过程。这意味着模型在处理需要多步推理的长尾场景时仍然力不从心。

---

## 3. 第三代VLA：推理驱动的自回归架构（2025–2026）

### 3.1 GR-3：Chain-of-Thought与自回归架构

GR-3（Generalist Robot 3）于2025年底由Google DeepMind发布，标志着VLA进入推理驱动时代。GR-3的核心架构包括了三个关键组件：

**（1）视觉编码器：SigLIP-ViT**
GR-3使用SigLIP（Sigmoid Loss for Language-Image Pre-training）作为视觉 backbone。SigLIP相比标准CLIP的关键改进在于用sigmoid损失替代对比损失，使得模型可以在更大的batch size下稳定训练，并且在细粒度视觉理解（如物体姿态估计）上优于CLIP约8%。SigLIP-ViT输出576个patch token（基于24×24网格）加上1个cls token。

**（2）LLM Backbone：Gemini Pro精简版**
GR-3的LLM基于Gemini Pro架构，但为了满足实时性要求做了大规模模型剪枝和量化。具体地，注意力头的数量从32减少到24，FFN层的中间维度从16,384降低到10,240，最终模型规模约80B参数。通过8-bit量化（HQQ），模型在TPU v5p上的推理延迟降至45ms/token。

**（3）CoT推理机制**
GR-3在生成动作之前会先输出一段推理链，格式为：

```
<thinking> 前方有行人正在通过人行横道，车速为45km/h。
根据交通规则，我必须在人行横道前停车。
行人距离我约25米，当前减速度可达3m/s²。
需要降到0km/h需要约4.2秒，制动距离约26米。
因此应立即开始制动。
</thinking>
<action> brake: 0.7, steer: 0.0 </action>
```

CoT（Chain-of-Thought）极大地提高了模型在复杂场景下的成功率。在Waymo Open Motion Dataset的闭环评测中，GR-3的CoT版本相比无CoT版本的碰撞率降低45%（从1.8%降至0.99%），而平均推理时间仅增加18%（从45ms增至53ms，因为思考链只有20–30个token）。

但是CoT也带来了新的问题：思考链可能包含幻觉。在压力测试中，GR-3会在约1.3%的样本中出现"幻觉推理"，例如认为一个静止的汽车"即将变道"并提前减速。这说明CoT虽然提高了平均表现，但引入了新的长尾失败模式。

### 3.2 EPM（Embodied Planning Model）：规划即推理

EPM由MIT CSAIL在2026年初提出，它的核心主张是：具身智能体应该显式地进行规划，而规划本质上是一种多步推理。EPM的架构将规划过程嵌入到LLM的token生成过程中：在每个时间步，模型先预测未来T步的状态序列（world model rollout），然后基于预测状态选择最优动作。

EPM的训练损失由三部分组成：

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{act}}(\hat{a}, a^*)}_{\text{动作模仿}} + \lambda_1 \underbrace{\mathcal{L}_{\text{feat}}(\hat{z}, z^*)}_{\text{特征预测}} + \lambda_2 \underbrace{\mathcal{L}_{\text{task}}(\hat{R}, R^*)}_{\text{任务奖励预测}}$$

其中 $$\mathcal{L}_{\text{feat}}$$ 是未来视觉特征的自监督预测损失，$$\mathcal{L}_{\text{task}}$$ 是任务完成度的预测损失。EPM在MetaWorld基准上相比GR-3的提升约为11%，但在复杂场景中（如需要精准抓取的任务）提升更显著（约23%），说明规划过程对精细操作帮助更大。

EPM的代价是额外的计算开销：每次推理需要生成T=8步的规划链，增加了约3倍的推理延迟（从45ms增至135ms）。在实时系统中，这意味着控制频率从22Hz降至7.4Hz，超过了大多数实时控制需求的下限（通常为10Hz）。

### 3.3 BridgeVLA：桥接监督的新范式

BridgeVLA是2026年4月由Stanford发表的论文，核心观点是：VLA模型的训练应该通过一个"桥接"阶段来弥合预训练和指令微调之间的gap。具体做法分为三步：

**Step 1：图文预训练（VLM Stage）**
使用SigLIP + LLaMA 3在LAION-5B + Internal-1T规模数据上预训练，建立基础的视觉-语言对齐。这个阶段的目标不是学习动作，而是学习"看到什么、说出什么"。

**Step 2：桥接监督（Bridging Stage）**
这是BridgeVLA的核心创新。在这个阶段，模型在大量演示数据上同时学习两个任务：
- 视频到文本的描述预测（"这个物体正在被左移5cm"）
- 文本到动作的条件生成（"左移5cm" → 动作token序列）

桥接监督的关键设计是：描述预测任务强制模型学习从视觉观测到语言化状态表征的映射，而文本到动作任务则学习从语言化表征到具体动作的映射。两步的中间表示统一在语言空间，使得模型可以泛化到训练数据中没有见过的动作描述。

**Step 3：指令微调（Instruction Tuning）**
在桥接监督后的模型基础上，使用RLHF（基于Bradley-Terry偏好模型）对多种任务指令进行微调，优化动作的精细度。

```python
# BridgeVLA桥接监督的训练循环
class BridgeVLATrainer:
    def __init__(self, model, vlm_data_loader, demo_data_loader):
        self.model = model  # 同一个模型
        self.vlm_loss = nn.CrossEntropyLoss()  # 文本预测
        self.action_loss = FlowMatchingLoss()  # 动作预测
    
    def train_step(self, batch):
        images, language_desc, action_seq = batch
        # Step 1: 视频→文本描述预测
        pred_desc = self.model.generate_text(images, prefix="描述当前操作：")
        desc_loss = self.vlm_loss(pred_desc, language_desc)
        # Step 2: 文本→动作条件生成
        pred_action = self.model.predict_action(images, language_desc)
        act_loss = self.action_loss(pred_action, action_seq)
        # 联合优化
        total_loss = desc_loss + act_loss
        total_loss.backward()
        return total_loss.item()
```

BridgeVLA的桥接监督在Caltech Robotic Manipulation Benchmark上取得了89.2%的任务成功率，比仅在操作数据上端到端训练的基线高出14.3个百分点。这说明显式的桥接监督缓解了预训练和下游任务之间的表示层gap。

### 3.4 Generation 3对比表

| 模型 | 发布 | 动作表征 | 推理机制 | 参数量 | 延迟(ms) | 碰撞率(闭环) |
|:----:|:----:|:--------:|:--------:|:------:|:--------:|:----------:|
| GR-3 | 2025.12 | 自回归token | CoT | 80B | 53 | 0.99% |
| EPM | 2026.02 | 扩散策略 | 多步规划 | 40B | 135 | 0.71% |
| BridgeVLA | 2026.04 | Flow Matching | 桥接推理 | 70B | 48 | 0.82% |
| RoboVLM-3 | 2026.05 | 混合token | CoT + MC | 100B | 62 | 0.65% |
| GR-4(传闻) | 2026.Q3 | 自回归+Diff | 多模态CoT | 200B | <30 | N/A |

---

## 4. 关键技术深水区

### 4.1 动作表征的三角权衡

动作表征设计是VLA模型的核心选择，存在一个三角权衡：**精度 ↔ 速度 ↔ 表达能力**。

离散tokenization（RT-2路线）的表达能力受限于bin数量，增加bin数会指数级增长词汇表大小和softmax计算量。对于6-DoF齐次变换，如果每个维度256个bin，词汇表规模是 $$256^6 \approx 2.8 \times 10^{14}$$，完全不可行。因此RT-2只能对每个维度独立离散化，忽略了维度间的相关性。

扩散策略（Octo/GR-1路线）在表达多峰分布方面最自然，因为扩散过程可以建模复杂的动作分布。但100步的DDIM采样在实时场景中仍然是瓶颈。最近的进步包括LCM（Latent Consistency Model）将步数降至2–4步，代价是样本质量轻微下降。

Flow Matching（pi0/BridgeVLA路线）在精度和速度之间取得了最好的平衡。10步欧拉采样的输出质量已接近扩散策略100步DDIM的水平，而计算开销仅为后者的1/10。CFM技术进一步允许将采样过程融合到单次推理中。

### 4.2 LLM Backbone的选择策略

VLA中LLM backbone的选择需要同时考虑三个维度：推理能力、延迟和可微调性。

从推理能力看，PaLM-E和Gemini路线利用了专用闭源模型的最佳性能，但无法在自定义数据上微调。开源路线（LLaMA系列、Qwen系列）提供了灵活的微调接口，但模型能力上限低于同规模的闭源模型。

延迟优化方面，KV-cache是降低自回归推理延迟的关键。对于生成N个动作token，使用KV-cache后每个token的延迟从$$O(L^2)$$降至$$O(L)$$（L为序列长度）。VLA场景的典型序列长度为256–1024 tokens，KV-cache带来的加速约为5–10倍。

可微调性决定了下游适配的效果。LoRA（Low-Rank Adaptation）是VLA模型最常用的参数高效微调方法，在模型所有注意力层的Q和V投影矩阵上插入秩为r=16的低秩适配器。对于80B的GR-3，LoRA微调的参数量仅为约1.2B（占比1.5%），但足以在特定任务上达到全参数微调90%以上的性能。

### 4.3 从预训练到RLHF的训练管线

现代VLA模型的训练管线通常包含四个阶段：

1. **VLM预训练**（100B+ tokens）：在互联网图文对数据上训练视觉编码器和LLM的对齐，学习基础的视觉语义理解能力。这个阶段的损失函数是标准的交叉熵（文本生成）+ 对比损失（图文对齐）。

2. **行为克隆**（1M–10M demonstrations）：在演示数据上通过监督学习（BC Loss）训练完整的VLA模型。常用的损失函数是负对数似然：

   $$\mathcal{L}_{\text{BC}} = -\mathbb{E}_{(o, a) \sim \mathcal{D}}[\log \pi_\theta(a | o)]$$

3. **指令微调**（10K–100K instructions）：使用多样化的自然语言指令和对应的动作标签对模型进行微调，提高模型的指令跟随能力。这个阶段通常使用NEFTune（Noisy Embedding Fine-Tuning）添加高斯噪声到embedding层进行正则化。

4. **RLHF/GRPO**（偏好数据）：基于人类或自动评估器给出的偏好对进行强化学习。GR-3使用GRPO（Group Relative Policy Optimization）替代了传统的PPO，因为GRPO不需要critic network，降低了训练的计算开销约40%。

---

## 5. VLA模型选型决策指南

面对一个具体的具身智能任务，如何选择VLA模型的技术路线？

**如果你的任务是高速运动控制（如自动驾驶、无人机飞行）：**
推荐Flow Matching动作头 + 小规模LLM（7B–13B）+ 无CoT推理。高速场景对延迟极度敏感（控制频率需≥50Hz），CoT带来的推理延迟可能得不偿失。pi0路线最合适。

**如果你的任务需要复杂多步推理（如精细装配、多阶段操作）：**
推荐扩散策略 + 中等规模LLM（40B–80B）+ CoT推理。EPM的多步规划架构虽然延迟更高，但在这类任务上的成功率提升显著。

**如果你追求零样本泛化能力：**
推荐BridgeVLA路线的桥接监督范式。额外的"视频→语言→动作"桥接使得模型可以泛化到训练数据中未出现的指令和场景。

**如果你受限于计算资源（单卡A100）：**
推荐Octo路线（开源 + LoRA微调）。Octo可以在单张A100上微调8小时完成特定任务适配，推理延迟约120ms，足以应对大多数桌面操作场景。

---

## 6. 挑战与未来方向

### 6.1 数据集瓶颈

当前最大的VLA数据集规模约为10M条演示（来自Open X-Embodiment + RT-1-X + Bridge Data v3），但相比于LLM预训练的万亿token级数据仍然是杯水车薪。如何生成高质量的合成演示数据（使用仿真器自动生成、或使用视频生成模型合成）是一个关键方向。

### 6.2 实时性-精度-泛化三角

VLA模型面临着实时性、精度和泛化能力的不可能三角。当前最先进模型在泛化能力上仍然远逊于人类——GR-3在一个新场景上需要5-10次演示才能适应，而人类通常只需要1次。任务调制（task modulation）和快速适配（fast adaptation）是实现通用机器人智能的关键门槛。

### 6.3 安全性与可解释性

CoT推理为VLA模型带来了一定程度上的可解释性，但这是一种"自我报告"式的解释——我们无法验证推理链是否真的反映了模型的决策过程。更严格的可解释性方法（如activation patching、causal tracing）在VLA上的应用仍处于早期阶段。

### 6.4 GR-4可能的方向

根据业内人士的分析，GR-4可能在以下几个方向突破：
- 将自回归token生成和扩散策略融合为混合架构，在精度和速度上同时达到最优
- 引入视频生成作为世界模型的隐式形式，通过predicting future frames来增强规划
- 跨具身体的知识迁移，让在机械臂上学习的技能可以迁移到人形机器人或自动驾驶车辆上

---

## 7. 总结

| 代际 | 代表模型 | 动作表征 | 推理能力 | 控制频率 | 典型精度 |
|:----:|:--------:|:--------:|:--------:|:--------:|:--------:|
| G1 (2023–2024) | RT-2, PaLM-E | 离散token | ❌ 无 | <1Hz | ±1cm |
| G2 (2024–2025) | pi0, Octo, DriveVLA | 扩散/Flow/MLP | △ 隐式 | 8–100Hz | ±0.1mm |
| G3 (2025–2026) | GR-3, EPM, BridgeVLA | 自回归+CoT | ✅ 显式 | 7–22Hz | ±0.05mm |

VLA模型的发展史是一部从"简单化"到"精细化"的工程进化史。每一代都在精度、速度和推理能力之间做出不同的设计权衡。理解这些权衡背后的工程原理，远比追逐最新模型更重要——因为每一个新模型的背后，都是对前一代模型核心局限性的针对性突破。随着GR-4的到来，我们很可能看到自回归token生成和扩散策略走向融合，这也是目前学术界和工业界最看好的方向。
