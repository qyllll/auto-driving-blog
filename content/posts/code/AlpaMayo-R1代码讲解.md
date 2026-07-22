---
title: "AlpaMayo-R1 代码讲解：VLM 说推理链、Expert 雕轨迹、一个模型两套头"
date: 2026-07-22
draft: false
categories: ["代码讲解"]
tags: ["VLA", "扩散模型", "因果推理", "GRPO", "代码讲解", "NVIDIA"]
summary: "零基础逐文件拆解 NVIDIA AlpaMayo-R1 推理源码：从 VLM/Expert/Diffusion 三组件架构到单轮动力学动作空间，从 Flow Matching 去噪到 Fourier 特征编码，用伪代码 + 大白话讲清这个 10B 参数的 VLA 模型是怎么从传感器数据一路「推理」到轨迹输出的。"
---

## 写给零基础读者：读这篇之前先搞懂几个名词

AlpaMayo-R1 的代码不长，但它把好几个「听起来很唬人」的概念揉在了一起。你要是直接冲进源码，多半会被一堆缩写劝退。所以我们先花点时间，把会反复出现的黑话全部翻译成大白话。你现在不用全记住，有个印象，等下看代码时回头对照就行。

- **VLA（Vision-Language-Action，视觉语言动作模型）**：给模型一张图（有时再加一句话指令），它直接吐出「车该怎么开」。它的核心思路是：**把开车这件事，当成一个「语言建模」问题来做**——图像当输入，动作当输出，中间用一个已经很会「说话」的大模型（VLM）串起来。把里面那根大模型底座叫做「VLM 骨干」。

- **VLM（Vision-Language Model，视觉语言模型）**：一个「能看图说话」的多模态大模型。它本来的看家本领是「看一张图，用自然语言回答关于这张图的问题」，比如 Qwen2.5-VL、GPT-4V。AlpaMayo-R1 在它上面做改造，让它不光会回答问题，还会「开车」。

- **token（离散词）**：大模型眼里的世界不是连续的，而是一个个「词」。它把一句话拆成一串编号（比如「你」=101，「好」=102），每个编号就是一个 token。模型的工作说白了就是「猜下一个 token 是几号」。理解这个对 AlpaMayo-R1 尤其重要：它不光把文字拆成 token，还把历史轨迹、未来动作全拆成 token——**开车就是说话，轨迹就是文字**。

- **CoC（Chain of Causation，因果推理链）**：一种结构化的推理格式。普通 CoT（思维链）是自由写的，比如「前面有车，我要减速」。CoC 更严格，必须包含三要素：当前时刻的**驾驶决策**（比如「减速让行」）、影响这个决策的**关键因素**（比如「前方 10 米处有行人横穿」）、以及把两者串起来的推理过程。CoC 确保了推理链和驾驶动作之间有清晰的因果对应。

- **GRPO（Group Relative Policy Optimization，组相对策略优化）**：一种不需要价值网络的强化学习算法。对同一个场景采样多个候选轨迹，用组内奖励的均值和标准差做基线，比平均好的给正信号、比平均差的给负信号。省掉了 PPO 里那个和策略一样大的 critic 网络。DeepSeek-R1 就用它。

- **Flow Matching（流匹配）**：一种生成模型。和扩散模型（DDPM）类似，Flow Matching 也是从纯噪声逐步变成数据。但 DDPM 是「弯曲路径」（随机微分方程 SDE），Flow Matching 是「直线路径」（常微分方程 ODE）。直线意味着更少的生成步数——AlpaMayo-R1 只用了 10 步。

- **DDPM（Denoising Diffusion Probabilistic Model，去噪扩散概率模型）**：另一种生成模型。通过给数据慢慢加噪声、再训练网络去噪来生成新样本。步骤多（100~1000 步），但效果好。

- **Expert（动作专家模块）**：AlpaMayo-R1 里独立于 VLM 的一个轻量 Transformer 网络，专门负责处理轨迹 token。VLM 生成 CoC 推理链，Expert 接收 VLM 的上下文（KV cache）并迭代去噪生成轨迹。两者分工明确：VLM 管「想」，Expert 管「动」。

- **单轮动力学模型（Unicycle Model）**：一个最简化的车辆运动模型。它把车的运动分解为两个控制量：**加速度**（踩油门/刹车）和**曲率**（打方向盘）。给定这两个量，用简单的物理公式就能积分出车的位置 (x, y) 和朝向。好处是模型不需要直接预测位置，而是预测更底层、更平滑的控制量。

- **Fourier 编码（傅里叶特征编码）**：一种把连续数值映射到高频空间的技术。直觉上，神经网络很难直接感受「x=0.5」这种小数的小变化，但如果把它变成 `[sin(ω₁x), cos(ω₁x), sin(ω₂x), cos(ω₂x), ...]` 这么一组信号，网络就能通过不同频率的响应来捕捉细微变化。这个技巧在 NeRF（神经辐射场）里很出名。

- **KV cache（键值缓存）**：Transformer 生成时，每次只生成一个新 token，但需要「回顾」所有之前生成的 token。KV cache 就是把之前所有 token 的注意力键和值存起来，避免每步都从头算一遍。AlpaMayo-R1 里 VLM 生成的 KV cache 被直接传给 Expert 复用。

- **Euler 积分**：一种最朴素的数值积分方法。给定一个速度 v 和时间步 dt，新位置 = 旧位置 + dt × v。在 Flow Matching 里，每一步去噪就是 x = x + dt × v，像沿着一条直线向前迈一小步。

- **action space（动作空间）**：模型输出的是轨迹吗？其实不是。AlpaMayo-R1 模型输出的不是 (x,y) 坐标，而是**加速度和曲率**这对控制量。然后通过单轮动力学模型积分出轨迹。这么做的原因是：控制量比坐标更平滑、更物理、更容易学。

- **mode collapse（模式坍缩）**：扩散模型的一个常见问题——不同随机噪声经过扩散模型的去噪后，往往收敛到相似的轨迹。比如让模型生成 10 条不同轨迹做候选，结果 10 条都差不多。这会严重降低规划的多样性。

如果这些词你都有个模糊印象了，下面读代码会顺畅很多。我们开始。

## 为什么要讲 AlpaMayo-R1 的代码

这是「自动驾驶代码讲解」系列的第 10 篇。前面我们已经走过了 UniAD / VAD（端到端回归范式）、DiffusionDrive（扩散轨迹生成）、SparseDriveV2（稀疏检索式规划）、AutoVLA（动作 token 化 + GRPO）等。

AlpaMayo-R1（NVIDIA，2025，arXiv:2511.00088）把这条线推到了一个新的高度。它是**第一个把「结构化因果推理」和「GRPO 强化学习后训练」整合进 VLA 的开源模型**。10B 参数，真车路测延迟仅 99ms，在 NAVSIM 长尾场景上规划准确率 +12%、近距离冲突率 -35%。

更值得讲的是它的代码设计——不是把所有逻辑塞进一个大模型里，而是拆成了三个清晰独立、可插拔的组件：**VLM（负责看和想）+ Expert（负责迭代去噪）+ Diffusion（负责采样调度）**。这种模块化设计让它很容易被复用和扩展。

这篇不推公式，直接顺着仓库把关键实现讲清楚。

> **一句话结论**：AlpaMayo-R1 = VLM（Qwen3-VL-8B 背板，自回归生成 CoC 推理链）+ Expert（轻量 2B Transformer，在 VLM 的 KV cache 上迭代去噪）+ Diffusion（Flow Matching，10 步 Euler 积分从噪声生成动作）+ 单轮动力学（把加速度/曲率解码成轨迹点）。

### AlpaMayo-R1 框架总览

这是论文中的端到端架构图：

![AlpaMayo-R1 端到端架构](/images/alpamayo-r1/architecture.png)

从上图可以看到完整的数据流：
- **输入**：多相机多帧图像 + 历史自车轨迹
- **VLM 编码**：图像 → ViT 编码（支持 Triplane/Flex 压缩）→ 与文本 token 一起送入 Qwen3-VL
- **推理**：VLM 自回归生成 CoC 推理链（慢）→ 遇到 `<traj_future_start>` 停止
- **扩散生成**：Expert 模块利用 VLM 的 KV cache，配合 Flow Matching 迭代去噪
- **动作解码**：单轮动力学把 (accel, kappa) 转成 (x, y, yaw) 轨迹点

## 架构总览：先看地图

![AlpaMayo-R1 代码结构总览](/images/alpamayo-r1/code_structure.svg)

在钻进任何一个文件之前，先把整条链路在脑子里画一遍。AlpaMayo-R1 的一次「决策」大致是这样走的：

- **输入端**：把 4 个相机 × 4 帧的图像 + 历史 1.6s 自车轨迹一起读进来。图像通过 VLM 的视觉编码器变成视觉 token，历史轨迹通过 DeltaTokenizer 编码成离散 token。
- **VLM 自回归生成 CoC**：模型的 VLM 骨干（Qwen3-VL-8B）开始逐 token 生成——先产生一段文本推理链（CoC），比如「前方有行人横穿，因此我选择减速让行」。生成到 `<traj_future_start>` 这个特殊 token 时停止。
- **Expert + Diffusion 去噪**：VLM 的 KV cache（包含图像特征 + CoC 文本上下文）被传给 Expert 模块。Expert 是一个轻量 Transformer（约 2B 参数），它和 Flow Matching 采样器配合，从随机噪声出发，用 10 步 Euler 积分生成动作。
- **动作解码**：生成的 (accel, kappa) 动作通过单轮动力学积分成 (x, y, yaw) 轨迹点，完成从「推理」到「开动」的全流程。

![AlpaMayo-R1 推理流程](/images/alpamayo-r1/inference_flow.svg)

训练分三步走（但本次开源的推理代码只包含前两步的训练后模型）：

- **Stage-1（动作模态注入）**：让 VLM 学会输出驾驶动作 token。用标准的 next-token prediction 训练。
- **Stage-2（推理能力激发）**：用 CoC 数据集做 SFT，让模型学会输出推理链 + 轨迹 token。
- **Stage-3（RL 后训练）**：用 GRPO 做强化学习，同时优化推理质量 + 推理-动作一致性 + 轨迹质量。**（该阶段权重未开源）**

> **关键认知**：AlpaMayo-R1 里没有一个「专门的规划器」。规划、推理、决策，全部被分配给了两个清晰分工的模块——VLM 负责「想明白为什么这么开」，Expert 负责「把它开出来」。这种解耦是它最优雅的设计。

## 项目结构：先厘清边界

读任何开源代码，第一件事是搞清楚「哪些是作者自己写的核心、哪些是白嫖的外部库」。AlpaMayo-R1 的边界很清楚：

- **基座（外部）**：Qwen3-VL-8B（VLM 骨干，来自 HuggingFace transformers），Cosmos-Reason 微调版。
- **AlpaMayo-R1 自己写的核心**：
  1. **VLA 模型组装**：怎么在 Qwen3-VL 上挂 Expert 和 Diffusion 模块（`alpamayo_r1.py`、`base_model.py`）
  2. **动作空间**：怎么用单轮动力学把轨迹编成 (accel, kappa) 再解码回来（`action_space/`）
  3. **Flow Matching**：直线路径去噪采样器（`diffusion/`）
  4. **动作投影**：Fourier 编码 + MLP 把动作映射成 token（`action_in_proj.py`）
  5. **推理入口和数据加载**（`test_inference.py`、`load_physical_aiavdataset.py`）

用普通 Markdown 列一下仓库结构：

- alpamayo/
  - src/alpamayo_r1/
    - config.py （配置类：AlpamayoR1Config，容纳 diffusion/action_space/expert 等全部配置）
    - helper.py （构造对话模板 + 初始化处理器）
    - test_inference.py （端到端推理入口，从 clip_id 到轨迹，不到 60 行）
    - load_physical_aiavdataset.py （从 Physical AI AV Dataset 加载多摄图像和自车轨迹）
    - models/
      - alpamayo_r1.py （主角：AlpamayoR1，整合 VLM + Expert + Diffusion + Action Space）
      - base_model.py （基类 ReasoningVLA + TrajectoryFusionMixin + 配置 + tokenizer）
      - action_in_proj.py （PerWaypointActionInProjV2：Fourier 编码 + MLP 投影）
      - token_utils.py （StopAfterEOS 停止条件 + 文本提取 + 填充替换）
      - delta_tokenizer.py （DeltaTrajectoryTokenizer：历史/未来轨迹的离散 token 编码）
    - action_space/
      - action_space.py （抽象基类 ActionSpace）
      - unicycle_accel_curvature.py （单轮动力学：accel + kappa ↔ traj 的双向映射）
      - discrete_action_space.py （离散轨迹 tokenizer：均匀量化 + 查表解码）
    - diffusion/
      - base.py （BaseDiffusion + StepFn Protocol）
      - flow_matching.py （FlowMatching：Euler 积分采样 + 训练数据构造）
    - geometry/ （旋转矩阵、坐标变换工具）
    - common/ （日志工具）

逐个用一句话说明它干嘛：

- `config.py`：把所有组件的配置统一管理。`diffusion_cfg` 配 Flow Matching，`action_space_cfg` 配单轮动力学，`expert_cfg` 配 Expert 结构，`action_in/out_proj_cfg` 配投影模块。全部用 `hydra.utils.instantiate` 从 dict 注入。
- `helper.py`：构建 VLM 的对话模板——把图像和文本占位 token 拼成 `[system, user, assistant]` 三段式消息。同时负责初始化处理器（processor）。
- `test_inference.py`：最简推理入口。加载数据 → 构造消息 → 调 processor → 调 model.sample_trajectories → 打印 CoC + 算 minADE。
- `load_physical_aiavdataset.py`：按 clip_id 从 NVIDIA 的 Physical AI AV Dataset 加载一个场景。包含坐标变换（世界系→自车 t0 局部系）。
- `alpamayo_r1.py`：主角文件。定义 `AlpamayoR1` 类——继承 `ReasoningVLA`，挂接 Expert、ActionSpace、Diffusion、action_in/out_proj。核心方法是 `sample_trajectories_from_data_with_vlm_rollout`，包含 VLM 自回归生成 + Expert 扩散去噪的全流程。
- `base_model.py`：基类 `ReasoningVLA`。负责 VLM 加载、tokenizer 初始化（加入轨迹特殊 token）、权重注册（`AutoConfig.register` + `AutoModel.register` 让模型可从 HuggingFace 一键加载）。
- `action_in_proj.py`：FourierEncoderV2 + MLPEncoder 做动作投影。输入是噪动作 (accel, kappa) + 时间步 t，输出是 Expert 能理解的 embedding。
- `token_utils.py`：StopAfterEOS（遇到 `<traj_future_start>` 停止生成）+ extract_text_tokens（从输出 token 中提取 CoC 文本）+ replace_padding_after_eos（清理填充）。
- `delta_tokenizer.py`：把轨迹 (x,y,yaw) 编码成离散 token——对 Δxyz 做均匀量化（num_bins=1000），支持往返编码/解码。
- `unicycle_accel_curvature.py`：ActionSpace 的具体实现。traj_to_action（轨迹 → 加速度+曲率，训练用）和 action_to_traj（accel+kappa → 轨迹点，推理用）的双向映射。核心运动学方程基于单轮车模型。
- `discrete_action_space.py`：DiscreteTrajectoryTokenizer，用均匀量化把连续动作变成离散 token，供 VLM 做 next-token prediction。
- `flow_matching.py`：Flow Matching 采样器。推理用 Euler 积分（x = x + dt × v），训练用 construct_training_data 构造加噪样本 + compute_loss_from_pred 算 MSE。
- `geometry/` 和 `common/`：辅助工具——旋转矩阵、角度化简、日志分层。

> **边界提醒**：下面所有代码都是**简化后的示意伪代码**，用来讲清楚思路，不是逐字符照抄仓库。真实代码会有更多工程细节，但主干逻辑就是这些。代码块里我用纯英文（变量名、符号），中文解释都放在块外。

## 一、推理入口：不到 60 行跑通全流程

### 1.1 test_inference.py：从 clip_id 到轨迹

AlpaMayo-R1 的推理入口简单到令人惊讶——全部逻辑只有约 50 行有效代码：

```python
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
messages = helper.create_message(data["image_frames"].flatten(0, 1))

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(messages, tokenize=True, ...)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True,
    )
print("CoC:\n", extra["cot"][0])
min_ade = eval_minADE(pred_xyz, data["ego_future_xyz"])
print("minADE:", min_ade, "meters")
```

逐行说人话：

- `clip_id`：指定要加载的数据片段 ID，从 `vla_golden.parquet` 这类索引文件里可以查到。
- `load_physical_aiavdataset(clip_id, t0_us=5_100_000)`：加载这个片段在 **第 5.1 秒** 的那一帧。历史 16 帧（1.6s 历史）和未来 64 帧（6.4s 预测）都以这一帧为锚点。为什么是 5.1 秒？因为片段的前几秒通常是车辆静止/起步阶段，5.1s 后才有丰富的驾驶行为。
- `helper.create_message(data["image_frames"])`：把 4 个相机 × 4 帧的图像拼成一个对话模板——告诉模型「你是驾驶助手，请根据场景输出推理和轨迹」。
- `AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B")`：从 HuggingFace 一键加载 22GB 的模型权重。能这么直接用，是因为代码里把自定义模型注册进了 transformers 的 AutoModel 体系（`AutoConfig.register` + `AutoModel.register`）。
- `processor.apply_chat_template`：把对话模板转成模型能吃的 token 张量——图像变成 ViT patch embedding，文本变成 token ID。
- `sample_trajectories_from_data_with_vlm_rollout`：核心推理方法。先让 VLM 自回归生成 CoC 推理链，再用 Expert + Diffusion 去噪生成轨迹。
- `extra["cot"]`：模型输出的 CoC 推理链文本——不是人类标注的，是模型自己生成的。
- `minADE`：最小平均位移误差——衡量生成的轨迹里最好那条和真值轨迹有多接近。

| 参数 | 作用 | 典型值 |
|:---|:---|:---|
| `t0_us` | 采样锚点时刻（微秒） | 5.1M µs（第 5.1 秒） |
| `num_traj_samples` | 每个场景采样多少条轨迹 | 1（推理用，调高可看多样性） |
| `max_generation_length` | VLM 最多生成多少 token | 256 |
| `top_p` / `temperature` | 文本生成的随机性控制 | 0.98 / 0.6 |

### 1.2 数据加载：世界坐标系→自车坐标系

`load_physical_aiavdataset.py` 的核心工作不是「读取数据」，而是**坐标系变换**。它把车在世界坐标系中的位置，变换到「以当前时刻自车为原点」的局部坐标系：

```python
# 核心：世界系 → 自车 t0 局部系
t0_xyz = ego_history_xyz[-1]                    # t0 时刻世界坐标
t0_rot = spt.Rotation.from_quat(ego_history_quat[-1])  # t0 时刻朝向
t0_rot_inv = t0_rot.inv()

# 历史轨迹：平移 + 旋转
ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
# 未来轨迹：平移 + 旋转
ego_future_xyz_local  = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
```

**为什么一定要转到自车局部坐标系？** 想象一下，如果保持世界坐标系，同一个「左转弯」动作，在十字路口 A 和十字路口 B 对应的 (x,y) 数值完全不同。模型必须额外学一个「当前位置在哪」的隐式编码。而在自车局部坐标系里，所有「左转弯」的轨迹都大致是「x 正方向 + y 正方向」，大大简化了学习难度。

数据加载的输出：

| 张量 | 形状 | 含义 |
|:---|:---|:---|
| `image_frames` | (4, 4, 3, H, W) | 4 个相机 × 4 帧（t-0.3s ~ t0） |
| `ego_history_xyz` | (1, 1, 16, 3) | 历史轨迹 1.6s @10Hz，局部坐标 |
| `ego_future_xyz` | (1, 1, 64, 3) | 未来轨迹 6.4s @10Hz，局部坐标 |
| `ego_history_rot` | (1, 1, 16, 3, 3) | 历史朝向（旋转矩阵） |
| `camera_indices` | (4,) | 相机索引 [0,1,2,6] 按固定排序 |

4 个相机按固定顺序排列：`[cross_left, front_wide, cross_right, front_tele]`。这保证了模型每次看到的数据顺序一致，不需要额外处理。

### 1.3 helper.py：消息模板构造

```python
def create_message(frames):
    num_traj_token = 48  # 历史轨迹占位符数量
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    return [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": frame} for frame in frames  # 16 张图（4cam×4帧）
        ] + [
            {"type": "text", "text": f"{hist_traj_placeholder} output the chain-of-thought reasoning, then output the future trajectory."}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<|cot_start|>"}  # 预告：推理从这里开始
        ]},
    ]
```

说人话：

- 模板就是三段对话：system 告诉模型「你是司机」、user 给模型看图 + 放历史轨迹占位符 + 让模型输出推理和轨迹、assistant 用一个 `<cot_start>` token 预告「推理即将开始」。
- 那 48 个 `<traj_history>` 占位符后面会被 `fuse_traj_tokens()` 替换为真实的历史轨迹 token。为什么是 48 个？因为历史轨迹 16 个点 × 3 维（Δx, Δy, Δyaw）= 48 个离散 token。
- 图像有 16 张（4 相机 × 4 帧），每张都被 VLM 的视觉编码器压缩成 patch token。

**processor** 要做的事就是把这些「对话模板」（里面有文字、有图像）转成模型能吃的 token 张量。这一步由 HuggingFace 的 `AutoProcessor` 完成——它把图像 resize 成固定大小、切成 patch、投影成 token embedding，最后和文本 token 拼成一个长序列。

---

## 二、核心模型：VLM + Expert + Diffusion 三组件

### 2.1 模型初始化：三组件组装

`alpamayo_r1.py` 里定义的 `AlpamayoR1` 类，其 `__init__` 做的就是**组装三组件**：

```python
class AlpamayoR1(ReasoningVLA):
    def __init__(self, config, pretrained_modules=None, original_vocab_size=None):
        super().__init__(config, ...)  # 加载 VLM 骨干 + tokenizer

        # 组件1：Expert——轻量 Transformer，只处理动作 token
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        self.expert = AutoModel.from_config(expert_config)
        del self.expert.embed_tokens  # Expert 不自己做 embedding

        # 组件2：Action Space——单轮动力学 (accel, kappa)
        self.action_space = hyu.instantiate(config.action_space_cfg)

        # 组件3：Diffusion——Flow Matching 采样器
        self.diffusion = hyu.instantiate(config.diffusion_cfg,
                                          x_dims=self.action_space.get_action_space_dims())

        # 动作投影模块
        self.action_in_proj = hyu.instantiate(config.action_in_proj_cfg, ...)
        self.action_out_proj = hyu.instantiate(config.action_out_proj_cfg, ...)
```

三个组件各自的职责：

| 组件 | 用什么实现的 | 输入 | 输出 | 参数量 | 管什么 |
|:---|:---|:---|:---|:---:|:---|
| **VLM**（骨干） | Qwen3VLForConditionalGeneration | 图像 + 文本 | CoC 文本 + KV cache | ~8B | 理解场景、生成推理 |
| **Expert**（动作专家） | AutoModel.from_config(text_config) | 动作 token embedding + KV cache | 预测的速度场 v | ~2B | 迭代去噪、精修轨迹 |
| **Diffusion**（采样器） | FlowMatching | 随机噪声 + step_fn | 去噪后的动作 | 0（无参数） | 调度去噪步数 |

**Expert 为什么可以独立出来？** 这是 AlpaMayo-R1 最巧妙的设计。Expert 直接从 VLM 的 `text_config` 实例化——它的结构和你 Qwen3-VL 里那个语言模型一模一样，但它不要自己的 embedding 层（删掉了 `embed_tokens`），因为它的输入不是文字 ID，而是 `action_in_proj` 投影好的向量。这就意味着 Expert 可以「听懂」VLM 的上下文（通过 KV cache）——VLM 说「前面有行人，应该减速」，Expert 在这个语义背景下做去噪，生成符合这个推理的动作。

### 2.2 参数总量

整个模型 10B 参数，分布：

- **VLM 骨干**（Qwen3-VL-8B）：约 8B 参数，负责视觉编码 + 文本生成
- **Expert**（轻量 Transformer）：约 2B 参数，负责动作去噪

可训练的参数只有 **62.8M**（指 SFT 阶段），因为 VLM 骨干也参与了训练。在推理阶段，全部 10B 参数都被使用。

### 2.3 推理采样：两阶段流水线

`sample_trajectories_from_data_with_vlm_rollout` 是整个推理管线的灵魂。它把整个过程分为清晰的两个阶段：

```python
def sample_trajectories_from_data_with_vlm_rollout(self, data, ...):
    # ========== 阶段1：VLM 自回归生成 CoC 推理链 ==========
    input_ids = data["tokenized_data"]["input_ids"]
    traj_data_vlm = {"ego_history_xyz": ..., "ego_history_rot": ...}
    input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)  # 历史轨迹→token

    vlm_outputs = self.vlm.generate(
        input_ids=input_ids,
        generation_config={"top_p": 0.98, "temperature": 0.6, ...},
        stopping_criteria=StopAfterEOS(eos_token_id=traj_future_start_id),
        logits_processor=ExpertLogitsProcessor(...),  # 屏蔽离散轨迹 token！
    )
    # 此时 vlm_outputs 包含：CoC 文本 + KV cache

    # ========== 阶段2：Expert + Diffusion 去噪生成轨迹 ==========
    def step_fn(x, t):
        # 把噪声动作投影成 Expert 能吃的 embedding
        future_token_embeds = self.action_in_proj(x, t)
        # Expert 在 VLM 的 KV cache 上做前向
        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            past_key_values=prompt_cache,  # 复用 VLM 的 KV cache
            ...
        )
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        pred = self.action_out_proj(last_hidden)
        return pred  # 预测的速度场 v

    sampled_action = self.diffusion.sample(
        batch_size=total_batch, step_fn=step_fn, ...
    )
    # 解码：动作 → 轨迹
    pred_xyz, pred_rot = self.action_space.action_to_traj(sampled_action, ...)
    return pred_xyz, pred_rot
```

**阶段1——VLM 自回归生成**，详细来看每一小步：

- `fuse_traj_tokens`：把输入 token 序列里那 48 个 `<traj_history>` 占位符替换成真的历史轨迹 token——这些是 `DeltaTrajectoryTokenizer` 把历史轨迹 (x,y,yaw) 编码成的一串离散编号。
- `vlm.generate(...)`：标准的 HuggingFace 自回归生成。模型看图 + 看历史轨迹 → 逐 token 输出 CoC 推理链。
- `ExpertLogitsProcessor`：一个非常重要的细节——它把离散轨迹 token（`<i0>` ~ `<i767>`）的 logit 全部 mask 成 `-inf`。这意味着 VLM 在生成时**只能生成文本 token**，不能生成轨迹 token。轨迹 token 留给 Expert 去处理。这是「VLM 管推理、Expert 管动作」分工的结构保障。
- `StopAfterEOS`：遇到 `<traj_future_start>` 这个特殊 token 就停止生成。这个 token 是「推理完成，接下来该生成轨迹了」的信号。
- 生成完成后，VLM 的 `past_key_values`（KV cache）——里面包含了图像 token、CoC 文本 token 的注意力键值——被**直接传给 Expert**。

**阶段2——Expert + Diffusion 去噪**，每一小步：

- `action_in_proj(x, t)`：把当前噪声动作 x（形状 [B, 64, 2]，64 个 waypoint × (accel, kappa)）和时间步 t 投影成 Expert 的 hidden_size 维 embedding。方法是用 Fourier 编码 + MLP（下一节详细讲）。
- `self.expert(inputs_embeds=future_token_embeds, past_key_values=prompt_cache)`：Expert 在 VLM 的 KV cache 上做前向。Expert 的输入不是 token ID，而是 action_in_proj 输出的向量。它的作用是「在当前场景上下文下，预测这个噪声动作需要往哪个方向修正」。
- `self.action_out_proj(last_hidden)`：Expert 最后一层的隐藏状态投影回动作维度 (64, 2)，得到预测的速度场 v。
- `self.diffusion.sample(...)`：Flow Matching 的 Euler 积分。从随机噪声出发，step_fn 每步给出速度场 v，x = x + dt × v，10 步后得到干净的动作。
- `self.action_space.action_to_traj(...)`：把 (accel, kappa) 解码成 (x, y, yaw) 轨迹点。

**为什么 VLM 和 Expert 要分工？为什么不直接让 VLM 自己生成轨迹？** 因为 VLM 是自回归的——逐 token 生成，每个 token 依赖之前所有的 token。这对文本没问题（文本就是逐字生成的），但对轨迹来说太慢且不够灵活。Expert 的并行去噪可以在 10 步内同时处理 64 个 waypoint，而且可以通过多步迭代逐步精修，比一步到位的自回归更稳定。

### 2.4 TrajectoryFusionMixin：历史轨迹注入

在 `base_model.py` 里定义了一个混入类 `TrajectoryFusionMixin`，核心方法就是 `fuse_traj_tokens`：

```python
def fuse_traj_tokens(self, input_ids, traj_data):
    if traj_data is None: return input_ids

    # 把历史轨迹 (xyz, rot) 编码为离散 token
    hist_idx = tokenize_history_trajectory(
        self.hist_traj_tokenizer, traj_data,
        self.hist_token_start_idx
    )
    # 替换占位符
    input_ids = replace_pad_token(
        input_ids, hist_idx,
        self.config.traj_token_ids["history"]
    )
    return input_ids
```

说人话：VLM 看到的 token 序列里，本来有 48 个 `<traj_history>` 占位符。现在把它们替换成真正的轨迹 token。`HistTrajectoryTokenizer` 用 `DeltaTrajectoryTokenizer` 把历史 16 帧的 (x,y,yaw) 编码为 48 个离散 token（每帧 3 个 Δ 量）。这样 VLM 在生成推理时，就能看到「这辆车过去 1.6s 是怎么开的」。

### 2.5 模型注册：一键加载的魔法

为什么能 `AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B")`？因为代码在文件末尾做了注册：

```python
class AlpamayoR1(ReasoningVLA):
    config_class = AlpamayoR1Config

AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
```

这两行代码把自定义的模型类和配置类注册进了 HuggingFace transformers 的 `AutoModel` 体系。从此 transformers 就知道：当你看到 `"nvidia/Alpamayo-R1-10B"` 这个模型名，它的 config 是 `AlpamayoR1Config`，模型结构是 `AlpamayoR1`。这是那 22GB 权重能一键加载的技术基础。

---

## 三、动作空间：为什么用加速度和曲率

### 3.1 动作空间 vs 轨迹空间

AlpaMayo-R1 的模型**不直接输出轨迹点 (x,y)**，而是输出一对控制量 (accel, kappa)——即**加速度和曲率**。然后通过单轮动力学模型积分出轨迹。

| 表示方式 | 模型输出 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **原始 waypoint (x, y)** | 直接回归坐标 | 直观 | 对噪声敏感、轨迹可能不物理 |
| **加速度+曲率 (accel, kappa)** | 控制量 | 物理合理、平滑、数值稳定 | 需要额外解码步骤 |
| **航向角+速度 (θ, v)** | 控制量 | 直观 | 积分漂移问题 |

`UnicycleAccelCurvatureActionSpace` 实现了这个双向映射。我们先看它的核心——**正向**和**反向**两个方向。

### 3.2 正向：动作→轨迹（推理用）

推理时，模型预测出 (accel, kappa)，需要转成 (x, y, θ) 才能用：

```python
def action_to_traj(self, action, traj_history_xyz, traj_history_rot, t0_states=None):
    accel, kappa = action[..., 0], action[..., 1]

    # Step 1：反标准化（模型输出是标准化后的，需要转回物理值）
    accel = accel * accel_std + accel_mean      # 范围 ≈ [-9.8, +9.8] m/s²
    kappa = kappa * kappa_std + kappa_mean      # 范围 ≈ [-0.2, +0.2] m⁻¹

    # Step 2：从历史状态估计 t0 时刻速度
    if t0_states is None:
        t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
    v0 = t0_states["v"]  # t0 时刻的初速度

    # Step 3：单轮动力学积分（核心）
    dt = 0.1  # 时间步长 0.1s

    velocity = v0 + cumsum(accel * dt)        # v(t) = v0 + ∫a·dt
    theta    = cumsum(kappa * velocity * dt)   # θ(t) = ∫κ·v·dt
    x        = cumsum(velocity * cos(θ) * dt)  # x(t) = ∫v·cos(θ)·dt
    y        = cumsum(velocity * sin(θ) * dt)  # y(t) = ∫v·sin(θ)·dt

    return traj_future_xyz, traj_future_rot
```

说人话——这条积分链在干什么：

- **v(t) = v0 + ∫a·dt**：速度 = 初速度 + 加速度的累加。踩油门（a > 0）速度增加，踩刹车（a < 0）速度减小。
- **θ(t) = ∫κ·v·dt**：朝向角 = 曲率 × 速度的累加。曲率 κ 表示「方向盘的转动程度」——κ 大意味着急转弯，κ 小意味着直行。
- **(x,y) = ∫v·cos(θ)·dt, ∫v·sin(θ)·dt**：位置 = 速度在 x/y 方向上的投影积分。车头的方向（θ）决定了速度是往左还是往右。

这三个方程组就是经典的单轮车运动学模型。它的假设是：**车辆像一辆自行车，后轮驱动前轮转向**。虽然简单，但对 6.4s 的规划来说是够用的。

### 3.3 反向：轨迹→动作（训练用）

训练时，我们有真值轨迹 (x,y,θ)，需要转成 (accel, kappa) 做监督信号：

```python
def traj_to_action(self, traj_history_xyz, traj_history_rot,
                   traj_future_xyz, traj_future_rot):
    full_xy = concat(history_xyz[-1:], future_xyz)  # 拼起来
    dxy = diff(full_xy, dim=-2)                      # 位置差分→速度
    theta = theta_smooth(future_rot, ...)            # 平滑化航向角
    v = dxy_theta_to_v(dxy, theta, v0, ...)          # 从位置差分估计速度
    accel = finite_diff(v)                           # 速度差分→加速度
    kappa = dtheta / (v * dt)                        # 航向变化率→曲率
    return normalize(accel, kappa)                   # 标准化到 0 均值 1 方差
```

说人话：训练时，我们有「人类司机怎么开」的真值轨迹，我们要反过来求出对应的 (accel, kappa)，作为模型需要学习的目标。这个反算使用了**正则化最小二乘**（`solve_xs_eq_y`，即解一个带正则化的线性方程组 `Ax = y`），来保证求出的速度是平滑合理的。

### 3.4 输入标准化

`UnicycleAccelCurvatureActionSpace` 有一个非常重要的设计：**模型的输入输出都是标准化的**（均值 0，方差 1）。

```python
accel = (accel - accel_mean) / accel_std  # 标准化
kappa = (kappa - kappa_mean) / kappa_std  # 标准化
```

这意味着：
- **模型预测**的 accel/kappa 范围大约在 [-3, +3] 之间（而不是物理值 ±9.8 和 ±0.2）
- **进入 `action_to_traj` 时**，要先反标准化（乘 std 加 mean）才能得到物理值
- **进入 `traj_to_action` 时**，先算物理值，再标准化后才作为模型的学习目标

为什么要标准化？因为神经网络的激活函数（比如 SiLU）在 [-3, +3] 范围之外会饱和。如果你让模型直接预测 [-9.8, +9.8] 这么大的数值，梯度信号会很差，模型几乎学不动。

### 3.5 离散轨迹 Tokenizer

除了连续动作空间，AlpaMayo-R1 还有一个离散版本的 tokenizer，供 VLM 做 next-token prediction（在 Stage-1 训练中使用）：

```python
class DiscreteTrajectoryTokenizer:
    def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
        action = self.action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
        # 均匀量化到 num_bins=1000 个离散值
        action = (action - dims_min) / (dims_max - dims_min) * (num_bins - 1)
        return action.round().long().reshape(batch_size, -1)

    def decode(self, hist_xyz, hist_rot, tokens):
        action = tokens.float() / (num_bins - 1)
        action = action * (dims_max - dims_min) + dims_min
        return self.action_space.action_to_traj(action, hist_xyz, hist_rot)
```

一条轨迹有 64 步 × 2 维 = **128 个离散 token**，每一步的 accel 和 kappa 分别被量化为 0~999 之间的整数。这些离散 token 和文本 token 一起，在 VLM 的自回归生成过程中被预测（但注意推理时这些离散 token 被 ExpertLogitsProcessor mask 了，实际由 Expert 处理连续版本）。

---

## 四、Flow Matching 扩散解码器

### 4.1 为什么是 Flow Matching 而不是 DDPM

Flow Matching 和 DDPM 都是生成模型，但关键区别在于**生成路径**。

| 维度 | DDPM | Flow Matching |
|:---|:---|:---|
| 生成路径 | 弯曲的随机微分方程 SDE | **直线**的常微分方程 ODE |
| 步数需求 | 20~1000 步 | **10 步即可** |
| 采样方式 | Langevin 动力学 | **Euler 积分**（x += dt × v） |
| 训练目标 | 预测噪声 ε | 预测速度场 v |

Flow Matching 的「直线路径」意味着：从纯噪声到干净数据的路径是一条直线，你只要沿着斜率（速度场 v）走固定步数就能到达。DDPM 的路径是弯曲的，每一步的噪声分布在变化，需要更小的步长。

### 4.2 FlowMatching 采样

```python
class FlowMatching(BaseDiffusion):
    def sample(self, batch_size, step_fn, device, inference_step=10):
        # 起点：高斯噪声
        x = torch.randn(batch_size, *self.x_dims, device=device)
        # 时间步：0 → 1 的线性插值
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)

        for i in range(inference_step):
            dt = time_steps[i+1] - time_steps[i]                # 步长
            v = step_fn(x=x, t=time_steps[i])                    # 模型预测速度场
            x = x + dt * v                                       # Euler 积分

        return x  # 去噪后的动作 (accel, kappa)
```

步数 `inference_step=10` ——注意这里 10 不是训练 epoch 数，而是**一条轨迹的生成步数**。10 步从噪声到数据，比 DDPM 的几百步快了数十倍。

### 4.3 训练数据构造

训练时，Flow Matching 的损失是这样构造的：

```python
def construct_training_data(self, x):
    # Beta 分布采样时间步（偏重于高噪声端）
    t = self.beta_dist.sample((batch_size,))      # Beta(1.5, 1.0)
    t = self.beta_scale_constant - t * self.beta_scale_constant

    noise = torch.randn_like(x)
    noisy_x = t * x + (1 - t) * noise  # 线性插值：t=0→纯噪声，t=1→干净数据
    return {"noisy_x": noisy_x, "timesteps": t, "noise": noise, "x": x}

def compute_loss_from_pred(self, training_data, pred):
    x = training_data["x"]
    noise = training_data["noise"]
    target = x - noise  # 速度场方向 = 数据 - 噪声
    return F.mse_loss(target, pred)
```

说人话：

- `noisy_x = t * x + (1-t) * noise`：这是 Flow Matching 的核心——不像 DDPM 那样逐步加噪，而是直接做**线性插值**。t 接近 0 时，noisy_x ≈ 纯噪声；t 接近 1 时，noisy_x ≈ 干净数据。
- `target = x - noise`：速度场的方向就是从噪声指向数据的向量。模型要学的是这个方向。
- `Beta(1.5, 1.0)` 时间步采样：偏重于 t 接近 1 的区域（即更靠近干净数据的那一侧），让模型更关注「精修」阶段的预测。

---

## 五、动作投影模块：Fourier 编码 + MLP

### 5.1 PerWaypointActionInProjV2

`action_in_proj.py` 实现的 `PerWaypointActionInProjV2` 模块，负责把「当前噪声动作 (accel, kappa) + 时间步 t」映射成 Expert 能理解的 embedding：

```python
class PerWaypointActionInProjV2(nn.Module):
    def __init__(self, in_dims, out_dim, ...):
        # 2 个 Fourier 编码器，分别对 accel 和 kappa 编码
        self.sinus = nn.ModuleList([
            FourierEncoderV2(dim=20, max_freq=100.0) for _ in range(2)
        ])
        self.timestep_fourier_encoder = FourierEncoderV2(dim=20, max_freq=100.0)
        self.encoder = MLPEncoder(num_input_feats=..., out_dim=out_dim)

    def forward(self, x, timesteps):
        # 对 accel 和 kappa 分别做 Fourier 编码
        accel_feat = self.sinus[0](x[:, :, 0])  # accel → 20 维高频信号
        kappa_feat = self.sinus[1](x[:, :, 1])  # kappa → 20 维高频信号

        # 时间步也做 Fourier 编码
        t_feat = self.timestep_fourier_encoder(timesteps)

        # 拼接 + MLP + LayerNorm
        x = torch.cat([accel_feat, kappa_feat, t_feat], dim=-1)  # ~60 维
        return self.norm(self.encoder(x))  # → Expert hidden_size
```

### 5.2 FourierEncoderV2 的原理

```python
class FourierEncoderV2(nn.Module):
    def __init__(self, dim, max_freq=100.0):
        half = dim // 2
        # 对数间隔的频率：ω = [10^0, 10^0.2, ..., 10^log10(max_freq)]
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.freqs = freqs[None, :]

    def forward(self, x):
        arg = x[..., None] * self.freqs * 2 * torch.pi
        # 输出 = [sin(ω₁x), ..., sin(ω_half x), cos(ω₁x), ..., cos(ω_half x)]
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
```

**为什么需要 Fourier 编码？** 直觉上，神经网络很难区分 x=0.45 和 x=0.46 这种微小差异——在 32 位浮点数里，这两个数只差不到 0.1%。但如果用 Fourier 编码，0.45 和 0.46 在 100Hz 频率上的响应（sin(100×0.45×2π) vs sin(100×0.46×2π)）会产生显著不同的数值。这相当于给了网络一把「显微镜」，让它可以感知到连续数值的细微变化。这个技巧在 NeRF（神经辐射场）中第一次被广泛使用。

### 5.3 反向投影

`action_out_proj` 就更简单了——就是一个 `nn.Linear`：

```python
self.action_out_proj = nn.Linear(expert_config.hidden_size, 2)
# 输入：Expert 最后一层隐藏状态 (B, 64, hidden_size)
# 输出：速度场 v (B, 64, 2) 即每个 waypoint 的 (Δaccel, Δkappa)
```

---

## 六、Token 工具函数

`token_utils.py` 里提供了几个关键工具，虽然代码很少但很重要：

### 6.1 StopAfterEOS

```python
class StopAfterEOS(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        self.eos_found = None

    def __call__(self, input_ids, scores, **kwargs):
        batch_size = input_ids.shape[0]
        if self.eos_found is None:
            self.eos_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        if self.eos_found.all():
            return True  # 全部生成完了
        last_tokens = input_ids[:, -1]
        current_has_eos = last_tokens == self.eos_token_id
        self.eos_found = self.eos_found | current_has_eos
        return False  # 还要继续
```

这个停止条件的特殊之处在于：它不是在**遇到** EOS token 时就停止，而是在**遇到后多生成一个 token** 再停。因为 VLM 的 KV cache 是在生成下一个 token 后才更新的，如果遇到 EOS 立刻停，KV cache 里就没有包含 EOS token 自身的信息。

### 6.2 ExpertLogitsProcessor

```python
class ExpertLogitsProcessor(LogitsProcessor):
    def __init__(self, traj_token_offset, traj_vocab_size):
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids, scores):
        # 把离散轨迹 token 的 logit 设为 -inf
        scores[:, self.traj_token_offset:self.traj_token_offset + self.traj_vocab_size] = float("-inf")
        return scores
```

这个 processor 在 VLM 自回归时，把编号 `<i0>` ~ `<i767>` 对应的 logit 全部设成 `-inf`，让 VLM 「不会输出」这些离散轨迹 token。VLM 只能输出文本 token。那些被 mask 掉的轨迹 token 留给 Expert 模块去处理——这是分工的结构保障。

### 6.3 extract_text_tokens

```python
def extract_text_tokens(tokenizer, output_tokens):
    decoded_batch = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    extracted_text = {}
    for token in ["cot", "meta_action", "answer"]:
        extracted_text[token] = extract_between_special_tokens(decoded_batch, token)
    return extracted_text
```

从 VLM 的完整输出中，提取 `<cot_start>...<cot_end>` 之间的 CoC 推理链文本。这些文本就是模型「认为」自己为什么这么开车的理由。

---

## 七、把整条链路串起来

到这里所有零件都讲完了，我们把一次完整的推理串成一条线：

**准备阶段（一次性）**：

- 从 HuggingFace 加载 22GB 模型权重 → `AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B")`
- 指定要评测的场景 → `clip_id` + `t0_us=5.1s`

**推理阶段（对每个场景）**：

- `load_physical_aiavdataset(clip_id, t0_us)` → 加载 4 相机 ×4 帧图像 + 历史/未来轨迹，全部转到自车 t0 局部坐标系
- `helper.create_message(image_frames)` → 构造对话模板：告诉模型「你是司机，请看图输出推理和轨迹」
- `processor.apply_chat_template(messages)` → 图像变 patch embedding，文本变 token ID，拼成一个大序列
- `fuse_traj_tokens(input_ids, traj_data)` → 把 48 个 `<traj_history>` 占位符替换成真实历史轨迹的离散 token
- **阶段 1—VLM 自回归生成**：
  - `vlm.generate()` 逐 token 生成
  - `ExpertLogitsProcessor` 屏蔽轨迹 token，只能输出文本
  - `StopAfterEOS` 遇到 `<traj_future_start>` 后多生成 1 个 token 停止
  - 输出：CoC 推理链文本 + KV cache
- **阶段 2—Expert + Diffusion 去噪**：
  - 初始化高斯噪声 x `(B, 64, 2)`
  - `action_in_proj(x, t)`：Fourier 编码 + MLP 投影成 Expert embedding
  - `expert(inputs_embeds, past_key_values=KV_cache)`：Expert 在 VLM 上下文上预测速度场 v
  - `action_out_proj(hidden)`：投影回动作维度
  - `x = x + dt * v`：Euler 积分（×10 步）
  - 输出：干净动作 `(accel, kappa)`
- `action_space.action_to_traj(动作)`：单轮动力学积分 → (x, y, yaw) 轨迹点 `(B, 64, 3)`
- 输出：pred_xyz（64 个 waypoint）+ extra["cot"]（CoC 推理链）

用一张对照表钉死每个环节的关键信息：

| 环节 | 文件 | 输入 | 输出 | 一句话 |
|:---|:---|:---|:---|:---|
| 数据加载 | `load_physical_aiavdataset.py` | clip_id | 图像+轨迹张量 | 加载并转到局部坐标 |
| 对话构造 | `helper.py` | 图像帧 | 对话模板 | 告诉模型「你是司机」 |
| VLM 生成 | `alpamayo_r1.py` | token + KV cache | CoC 文本 | 自回归吐推理链 |
| Expert 去噪 | `alpamayo_r1.py` | 噪声动作 + KV cache | 速度场 v | 在 VLM 上下文上修正轨迹 |
| Diffusion 采样 | `flow_matching.py` | 随机噪声 | 干净动作 | 10 步 Euler 积分 |
| 动作解码 | `unicycle_accel_curvature.py` | (accel, kappa) | (x, y, yaw) | 单轮动力学积分 |
| 投影模块 | `action_in_proj.py` | 动作+时间步 | Expert embedding | Fourier 编码 + MLP |

## 和本系列其他文章的关系

AlpaMayo-R1 不是孤立的，它在端到端自动驾驶的演化线上有明确的坐标：

- **对比 UniAD / VAD（端到端回归）**：那两条是「网络直接回归轨迹坐标」的路线，没有大模型、没有生成式模型。AlpaMayo-R1 则引入了 10B 的 VLM 骨干和扩散生成式解码器。这是从「回归范式」到「生成式 VLA」的跨越。

- **对比 DiffusionDrive（扩散轨迹生成）**：DiffusionDrive 用**截断 DDPM** 从 anchor 出发 2 步去噪生成轨迹，依赖预定义 anchor。AlpaMayo-R1 用 **Flow Matching** 从纯噪声出发 10 步去噪，不依赖 anchor。且 AlpaMayo-R1 多了一个 CoC 推理链——它不只是「生成轨迹」，而是「先想明白为什么，再生成轨迹」。

- **对比 AutoVLA（动作 token + GRPO）**：两者都用 GRPO 做后训练，都是 VLA 架构。但核心差异在于动作表示——AutoVLA 把动作**离散化**成 codebook token 让 VLM 直接「说」出来；AlpaMayo-R1 把动作表示为**连续的**(accel, kappa)，用一个独立的 Expert 模块做扩散生成。一个走「离散 token + 统一生成」，一个走「连续动作 + 分模块生成」。这两篇是理解「VLA 动作表示」两条路线的关键对照。

- **通向 pi0**：AlpaMayo-R1 告诉你「推理链 + 分模块 Expert + Flow Matching」是一种可行的 VLA 架构。而 pi0 会告诉你另一种可能——用 Flow Matching 做统一的动作生成，不需要独立 Expert，也不需要推理链。两篇合起来，正好是 VLA 架构设计空间的两个极端。

## 个人思考

**1. AlpaMayo-R1 最让我欣赏的设计是「VLM 管推理，Expert 管动作」的解耦哲学。** 这看起来只是一个工程的模块划分，但它的影响很深远：VLM 的推理链可以单独被评估和调试（它输出的是自然语言），Expert 的去噪过程也可以单独优化（调整步数、换采样器）。如果有天出现了更好的扩散采样器，换掉 `flow_matching.py` 就行，VLM 完全不用动。这种「不像一个大黑盒，而像三个螺丝拼接的精密仪器」的架构，才是能持续演进的系统。

**2. 单轮动力学动作表示是一个被低估的工程细节。** 很多端到端规划工作直接回归 (x,y) 坐标，然后花大量精力在轨迹平滑上。AlpaMayo-R1 选择从物理模型出发——用 accel 和 kappa 做中间表示，自然地保证了轨迹的平滑性和运动学可行性。这个做法的启发是：**与其在输出后加一大堆后处理让轨迹看起来平滑，不如让模型生来就在「平滑的空间」里工作**。这就像写文章时用语法正确的语言写，而不是写完了再用自动纠错改一遍。

**3. KV cache 共享是「推理链约束轨迹生成」的实现基础。** 推理链不是一句空话——Expert 做去噪时，它的注意力可以「看到」VLM 生成的 CoC 文本。这意味着 Expert 知道「模型刚说了要减速让行」，所以它生成的轨迹也会偏向减速。这个因果链是训练出来的（通过 GRPO 优化推理-动作一致性），但 KV cache 的共享是它的物理基础——没有这个共享，推理链和轨迹就分别活在两个孤立的模块里，所谓「因果」就只剩文本层面的装饰了。

**4. 10B 参数推理只用了 10 步扩散，这种「大模型 + 少步生成」的搭配很值得研究。** 直觉是：模型越大，每步去噪的质量越高，所以需要的步数越少。AlpaMayo-R1 用 10B 骨干换来了 10 步去噪，而小模型可能需要 20 步以上才能达到同样质量。这个 trade-off（参数量 vs 推理步数）对未来模型设计有指导意义——也许「大模型 + 少步共识」比「小模型 + 多步精修」在延迟和效果上都更优。

**5. 一点反思：CoC 推理链到底「帮助」了规划多少？** 论文报告了 +12% 的规划准确率提升，但这个提升有多少来自于推理链本身、多少来自于更大的模型和更多的训练数据？如果能做一个消融实验——同样的模型架构、同样的训练数据，只是把推理链从生成序列中移除——看指标下降多少，会更令人信服。不过话说回来，从产品角度看，推理链即使对指标提升贡献有限，它提供的可解释性也足够有价值——一个能解释「为什么在这个路口减速让行」的自动驾驶系统，比一个开得更好但说不清理由的系统，在安全审计和用户信任上都有不可替代的优势。

---

*📖 代码讲解系列第 10 篇。本文基于 [github.com/NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) commit 17 个（main 分支）的代码分析。SFT 和 RL 后训练代码在 [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes)。*
