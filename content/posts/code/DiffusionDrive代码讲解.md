---
title: "代码讲解：DiffusionDrive 从架构到训练推理的完整闭环"
date: 2026-07-20
draft: false
categories: ["代码讲解"]
tags: ["DiffusionDrive", "扩散规划", "端到端自动驾驶", "代码讲解", "NAVSIM", "Truncated Diffusion"]
summary: "逐行拆解 hustvl/DiffusionDrive 的真实源码，面向零基础读者：从架构图与全局数据流出发，用大白话讲清什么是 anchor、什么是去噪、什么是 cross-attention、什么是 BEV、什么是截断扩散，再逐文件讲清它如何作为 navsim 的 Agent 插件挂载，最后把一次前向传播拆成 7 到 8 步、把训练和推理的差异讲透。一篇把项目挂载方式→数据流→前向传播→训练梯度→推理选轨迹→个人思考全部讲明白的极详细工程向代码讲解。"
---

## 写给完全没基础的同学：先补几个最关键的名词

在正式读代码之前，我们先把这篇文章会反复出现的几个「黑话」用一句话翻译成人话。你不用现在就完全懂，只要有个印象，后面看到它们就不会慌。

- **端到端自动驾驶（End-to-End Autonomous Driving）**：传统自动驾驶要把任务拆成「先识别车道线、再检测车辆、再规划路径、再控制方向盘」好几步，每一步都是独立模型。端到端就是「给模型一张图（和激光雷达数据），模型直接吐出方向盘怎么打、油门怎么踩、未来几秒车往哪走」，中间环节全部交给一个神经网络自己搞定。本文讲的 DiffusionDrive 就是端到端里的「规划」那一步：输入是传感器数据，输出是未来一小段时间的行驶轨迹。

- **轨迹（Trajectory）**：就是「车未来要走的路线」。本文里一条轨迹用 8 个点表示，每个点包含横向位置 x、纵向位置 y、朝向角 θ（theta，表示车头朝哪个方向），所以一条轨迹的形状是 `[8, 3]`，也就是 8 个时刻、每个时刻 3 个数。

- **anchor（锚点 / 锚轨迹）**：字面意思是「船锚」，用来固定位置。在本文里，anchor 不是点，而是一条「模板轨迹」。作者事先在训练集里用聚类算法找出 20 种最常见的驾驶模式（比如直行、左转、跟车减速……），每种模式存一条代表轨迹，这 20 条就是 20 个 anchor。模型不是从零开始想轨迹，而是从这 20 个模板出发去「微调」，所以 anchor 可以理解为「20 个候选驾驶姿势的草稿」。

- **扩散模型（Diffusion Model）**：一种生成式 AI。它的训练过程是「往一张清晰图上一点点加噪声，直到变成纯噪声；再训练一个网络学会把噪声一步步去掉还原清晰图」。推理时，从纯噪声出发，让网络反复「去噪」，最后「雕」出一张新图。DiffusionDrive 把这套思路用在「生成行驶轨迹」上，不是生成图片。

- **去噪（Denoising）**：把「带噪声的模糊东西」一点点变清晰的过程。比如你给 anchor 轨迹加一点随机扰动（噪声），它就变成一条「歪歪扭扭的模糊轨迹」，去噪网络的工作就是预测「刚才加的扰动是多少」，把它减掉，轨迹就恢复（或修正）成合理形状。

- **标准扩散 vs 截断扩散**：标准扩散从「纯高斯噪声」（完全随机、毫无意义的一团数）出发，要迭代几十步才能雕出像样的结果；DiffusionDrive 的截断扩散是从「anchor 模板 + 一点点噪声」出发，起点已经很接近答案了，所以只要 2 步去噪就能出结果。这就是它快的原因。

- **BEV（Bird's Eye View，鸟瞰图）**：从「天上往下看」的视角。想象你变成一只鸟，垂直俯视这辆车和周围，看到的就是 BEV。自动驾驶里常把相机和激光雷达数据统一投影到「以自车为中心、从上往下看」的坐标系里，方便网络理解「哪有车、哪有车道」。

- **ResNet（残差网络）**：一种很经典、很常用的图像特征提取神经网络（"残差"指它用"捷径连接"让很深的网络也能训得动）。你可以把它理解成「图像→一串有意义的数字特征」的转换器。本文里相机图像和激光雷达 BEV 图都各自过一个 ResNet-34（34 层的 ResNet）来提取特征。

- **cross-attention（交叉注意力）**：Transformer 里的一种机制，作用是「让一组查询（query）去另一组内容（key/value）里找相关信息」。打个比方：你（query）拿着问题去翻一本书（key/value），注意力机制决定你该重点看书的哪一页。本文里，anchor 轨迹点（query）会去「场景特征图」上找自己附近有什么障碍物、车道，从而被修正得更合理。

- **query（查询）**：在 Transformer 语境下，query 就是「我想从别处获取信息的一方」。本文里 20 个 anchor 轨迹会被编码成 20 组 query，它们去和场景特征做交叉注意力，相当于「20 个草稿分别去场景里核对：我这条路线前面有没有车、会不会出界」。

- **FiLM 调制**：一种「用一个数（比如当前去噪步数 timestep）去缩放和偏移特征」的小技巧。可以理解为「告诉网络：现在是第几步去噪，请你据此调整一下你的处理方式」。

- **argmax（取最大值的下标）**：比如 20 个模式各自有一个置信度分数 `[0.1, 0.8, 0.05, ...]`，argmax 就是「找出最大的那个，返回它的位置（这里是第 2 个）」，于是最终选第 2 个模式对应的轨迹。

- **NAVSIM**：一个自动驾驶规划评测基准（benchmark）。它提供传感器数据、标准评测指标（PDMS），以及一套官方训练/评测代码。DiffusionDrive 的代码就是「挂」在 NAVSIM 这套官方代码之上的插件。

- **Hydra**：Facebook 出品的配置管理库。用它可以像「`agent=diffusiondrive_agent`」这样在命令行切换配置，不用改代码。

- **PyTorch-Lightning**：一个把训练流程（训练循环、分布式、日志）封装好的 PyTorch 上层框架，作者用它来跑训练。

- **DDIM / DDPM**：两种扩散模型的「去噪采样算法」。DDPM 是原始慢版本（几十步），DDIM 是加速版本（可跳步）。DiffusionDrive 用 DDIMScheduler 来实现可控步数的去噪。

- **k-means（聚类）**：一种「把一堆数据自动分成 K 类」的无监督算法。本文用它把训练集里成千上万条真实轨迹聚成 20 类，每类的中心就是一条 anchor。

- **Focal Loss / L1 Loss**：两种损失函数。Focal Loss 擅长「在正负样本不平衡时聚焦难样本」，用来做分类；L1 Loss 是「预测值和真实值差的绝对值之和」，用来做回归（让轨迹更准）。

如果你上面这些名词都大致有印象了，下面读代码会轻松很多。我们开始。

## 为什么要讲 DiffusionDrive 的代码

DiffusionDrive（CVPR 2025，作者单位 hustvl）是端到端规划里**扩散策略**路线的开创者，也是 NAVSIM 排行榜上扩散类方法的基线。它最被人津津乐道的工程 trick 是：**把标准扩散从几十步去噪压到 2~8 步，还能天然支持多模态轨迹**（也就是一次能给出多种可能的开法，最后挑最好的）。

这篇不讲公式推导，直接看源码——仓库 `hustvl/DiffusionDrive` 把整套 navsim 内联（fork 进来）了，DiffusionDrive 自身代码全部在 `navsim/agents/diffusiondrive/` 下。

> **一句话结论**：DiffusionDrive 的本质 = ResNet 主干 + 20 个 k-means anchor 轨迹 + 从 anchor 加「截断噪声」的轻量扩散解码器；训练从 anchor 加噪、推理从 anchor 去噪 2 步，最后用分类头 argmax 选一条轨迹。

## 架构总览：先看地图

DiffusionDrive 是**生成式规划**：它不从固定候选集里选，而是**从 anchor 出发、用扩散逐步「雕」出一条轨迹**。下面这张图（DiffusionDrive 论文原图）给出了整体范式：

![DiffusionDrive 架构总览](/images/diffusiondrive/architecture.png)

为把源码里的数据流向讲清楚，我画了一张**数据流图**：

![DiffusionDrive 推理数据流：navsim输入 → TransFuser融合主干 → 冻结20 anchor作query → 截断扩散2步去噪 → cls argmax输出最优轨迹](/images/diffusiondrive/dataflow.svg)

> **关键认知**：和 SparseDriveV2（检索式）相反，DiffusionDrive 是**生成式**——它能在 20 个 anchor 附近「生成」字典外的轨迹，泛化更灵活；代价是推理要走去噪循环（虽已压到 2 步）。两篇对照读，能看清 NAVSIM 榜单上「生成 vs 检索」两条路线的分野。

把整条链路用一句话串起来就是：

**传感器数据（相机 + 激光雷达）→ 拼成全景图 + BEV 特征 → 过 ResNet 主干并融合 → 得到「场景特征」→ 20 个冻结 anchor 作为起点 → 加一点截断噪声 → 2 步去噪（每步用场景特征做交叉注意力修正）→ 得到 20 条候选轨迹 + 20 个打分 → argmax 选 1 条 → 输出 `[B, 8, 3]` 轨迹。**

## 项目结构：它如何挂在 navsim 上（先厘清边界）

和 SparseDriveV2 一样，DiffusionDrive 也是 navsim 的一个 fork（分叉副本），但**作者只动了 `navsim/agents/diffusiondrive/` 这一个角落**，其余训练/评测管线全是 navsim 原装。挂载关系只有一层（没有 SparseDriveV2 那种「仓库根目录独立脚本」），所以反而更干净：

- **新增的模型代码**（作者写的部分）：全部在 `navsim/agents/diffusiondrive/` 下，作为 navsim 的一个 `Agent` 插件接入。
- **训练 / 评测入口**：完全复用 navsim 官方的 `run_training.py`、`run_create_submission_pickle.py`，通过 Hydra 的 `agent=diffusiondrive_agent` 把自研模型「挂」上去。
- **官方管线文件**（fork 内联，基本不动）：`navsim/planning/script/run_training.py` 等。

下面用普通 Markdown 列表（不是代码块）展示目录结构，避免代码块内中文被逐字符换行的问题：

- navsim/
  - agents/diffusiondrive/ （作者新增的模型代码，挂在 navsim 上的插件）
    - transfuser_agent.py
    - transfuser_model_v2.py
    - transfuser_backbone.py
    - transfuser_features.py
    - transfuser_loss.py
    - transfuser_config.py
    - modules/
      - multimodal_loss.py
      - conditional_unet1d.py
      - scheduler.py
  - planning/script/run_training.py （复用官方训练入口，不修改）

逐个文件用一句话说明它干嘛：

- `transfuser_agent.py`：Agent 接口层。它继承 navsim 的 `AbstractAgent`，对外暴露 `compute_trajectory(agent_input)`（给输入、吐轨迹）和 `get_sensor_config` 等标准方法；内部把原始 `AgentInput` 交给 `transfuser_features.py` 做特征构建，再送给 `V2TransfuserModel` 前向，最后把模型输出整理成 navsim 要的轨迹格式。
- `transfuser_model_v2.py`：核心模型文件。定义了 `V2TransfuserModel`（主干 + TrajectoryHead）以及 `TrajectoryHead`（截断扩散解码器，含 denoiser、分类头 plan_cls、回归头 plan_reg）。这是你读源码最该盯紧的文件。
- `transfuser_backbone.py`：实现 TransFuser 的「图像 + LiDAR 融合主干」。包含两个 ResNet-34 编码器（一个吃图像、一个吃 BEV），以及把两者特征融合的交叉注意力层，输出统一的「场景 token 特征」。
- `transfuser_features.py`：负责把 navsim 的 `AgentInput`（原始相机图、激光雷达点云、地图、自车状态）转成模型能吃的张量。三相机横向拼接成全景图就在这里做。
- `transfuser_loss.py`：顶层 loss 聚合处。把轨迹损失、agent 检测损失、BEV 语义损失等按权重相加，得到总 loss 返回给 PyTorch-Lightning。
- `transfuser_config.py`：所有超参数的「集中营」。图像/LiDAR 用的网络结构、anchor 文件路径、各项 loss 权重、模式数 `ego_fut_mode=20` 等都在这里定义。
- `modules/multimodal_loss.py`：实现多模态监督的核心。它用「真实轨迹离哪个 anchor 最近」来定分类标签，再算 focal 分类损失 + 对该模式的 L1 回归损失。
- `modules/conditional_unet1d.py`：条件 1D UNet 去噪网络的实现（论文原版设计）。注意本项目实际推理用的是 `TrajectoryHead` 里内联的 `CustomTransformerDecoder`，这个文件更多是保留的对照实现。
- `modules/scheduler.py`：对 `diffusers.DDIMScheduler` 的轻量封装，集中管理「加噪 / 去噪步数 / 时间步选取」逻辑。

> **注意**：文件名沿用 `transfuser_*`，但类 `V2TransfuserModel` / `TransfuserAgent` 实际实现的是 DiffusionDrive——这是历史遗留命名（从 TransFuser 改过来的），读源码时别被名字误导。挂载关系一句话：**`run_training.py`（官方）→ `agent=diffusiondrive_agent` → `TransfuserAgent` → `V2TransfuserModel`（内含扩散 TrajectoryHead）**。

## 一、架构：截断扩散到底截在哪

这一节我们搞懂两件事：(1) 主干怎么处理图像和激光雷达；(2) 截断扩散 TrajectoryHead 到底改了标准扩散的哪一步。

### 1.1 主干与环视图像处理（先把"看"这件事讲透）

模型首先要「看懂」周围。它有两路输入：相机拍的图、激光雷达扫的点云。两者都先用 ResNet-34 提取特征。配置里写得很直白：



- image_architecture: str = "resnet34"
- lidar_architecture: str = "resnet34"


**什么是 ResNet-34？** 简单说，它是一个「把图片压缩成一串有意义数字」的成熟神经网络，有 34 层深。给它一张 `[3, 256, 1024]` 的图（3 是 RGB 三通道，256 是高，1024 是宽），它输出一串「这张图里有什么」的特征向量。图像用 ResNet-34，激光雷达的 BEV 图也用 ResNet-34，两路各提各的。

**关键在「多视角环视的处理方式」**——不是逐相机分别编码，而是把左/前/右三路裁剪后横向拼成一张 4:1 全景图再送 encoder。为什么这么做？因为自动驾驶车一般装多个相机（左、前、右），如果分别过 backbone 再融合，要写一套跨相机融合逻辑，麻烦。作者图省事（也是一种工程取舍），直接把三张图拼成一张超宽全景图，一次 backbone 搞定：



- cameras = agent_input.cameras[-1]

- l0 = cameras.cam_l0.image[28:-28, 416:-416]
- f0 = cameras.cam_f0.image[28:-28]
- r0 = cameras.cam_r0.image[28:-28, 416:-416]

- stitched_image = np.concatenate([l0, f0, r0], axis=1)
- resized_image = cv2.resize(stitched_image, (1024, 256))
- tensor_image = transforms.ToTensor()(resized_image)


> **工程要点**：把多相机拼成单张全景图，省掉了逐相机独立 backbone + 跨相机融合的复杂度，代价是相机间几何关系被「压扁」进 2D 拼接，依赖后续 transformer 自己学回来。在 NAVSIM 这种以前视为主的场景里够用；若要做到全向感知，这种拼接会损失侧视信息。

**LiDAR 这边怎么处理？** 激光雷达点云先被投影成一张「从上往下看」的 BEV 图（每个像素代表地面某个位置有没有障碍物、反射强度多少），再送进另一个 ResNet-34。两张图（图像特征、BEV 特征）随后在 backbone 里通过交叉注意力融合——让「图像里看到的车」和「激光里扫到的车」互相印证，得到更靠谱的场景理解。

### 1.2 截断扩散策略（TrajectoryHead）—— 本文核心

实现类 `TrajectoryHead`（`transfuser_model_v2.py:382-558`），使用 `diffusers` 的 `DDIMScheduler`。

**anchor 定义**：20 个 k-means 聚类锚轨迹（预生成 `kmeans_navsim_traj_20.npy`），作为不可训练参数（冻结）。这 20 个 anchor 对应 20 种驾驶模式（直行/左转/跟车……）。它们怎么来的？作者在训练集的全部真实轨迹上跑 k-means，聚成 20 类，每类的中心轨迹存成一个 `.npy` 文件。加载时把它变成 `nn.Parameter` 但 `requires_grad=False`，意思是「这是常量，训练时别改它」：



- self.diffusion_scheduler = DDIMScheduler(
-     num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample",
- )
- plan_anchor = np.load(plan_anchor_path)
- self.plan_anchor = nn.Parameter(
-     torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False,
- )


**训练：从 anchor 加「截断噪声」**。标准扩散从标准高斯 `N(0,1)`（完全随机的噪声）出发、timestep 取满 1000；DiffusionDrive 改成从 anchor 出发、timestep **截断到 50**（也就是说，训练时只模拟「噪声加到中等程度」的情况，不模拟「完全变成纯噪声」的极端情况，因为推理时也不会走到那）：




- plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
- odo_info_fut = self.norm_odo(plan_anchor)
- timesteps = torch.randint(0, 50, (bs,), device=device)
- noise = torch.randn(odo_info_fut.shape, device=device)
- noisy_traj_points = self.diffusion_scheduler.add_noise(
-     original_samples=odo_info_fut, noise=noise, timesteps=timesteps,
- ).float()
- noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
- noisy_traj_points = self.denorm_odo(noisy_traj_points)


**推理：只去噪 2 步**，初始样本 = anchor 加固定截断时刻 `t=8` 的噪声（而非纯高斯）。也就是说推理时根本不让轨迹变成纯噪声，只「轻扰」一下：



- def forward_test(...):
-     step_num = 2
-     self.diffusion_scheduler.set_timesteps(1000, device)
-     step_ratio = 20 / step_num
-     roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1] ...

-     plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
-     img = self.norm_odo(plan_anchor)
-     noise = torch.randn(img.shape, device=device)
-     trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
-     img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)


**为什么「从 anchor 出发 + 截断 timestep」能大幅提速？** 因为 anchor 已经把轨迹拉到了「合理驾驶模式」附近，扩散只需要做小幅修正，不需要从纯噪声一步步生成——所以 2 步就够。这是 DiffusionDrive 相比标准 DDPM 扩散（几十步）的**核心加速来源**。

> **一句话记住「截断扩散」**：标准扩散像「给一张白纸从头画一幅画」（慢，要几十步）；截断扩散像「给一张已经画了 80% 的草稿，只补两笔」（快，2 步）。anchor 就是那张 80% 的草稿。

**去噪网络是 `CustomTransformerDecoder`**（2 层，非标准 UNet1D），每层做：轨迹点在 BEV 上 grid_sample 交叉注意力 → 与 agent/ego query 交叉注意力 → 时间步 FiLM 调制 → 回归 offset（偏移量）。这里的「offset」是「应该对当前轨迹点加多少修正」：



- poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points


注意这里用「**残差回归**」：网络不直接预测最终轨迹，而是预测「相对当前 noisy 轨迹的修正量」，加到当前轨迹上。这比直接预测绝对轨迹更容易学，因为修正量通常很小。

### 1.3 多模态轨迹表示（20 条候选怎么来的）

20 个 anchor 一一对应 20 个模式（`ego_fut_mode = 20`），每个模式回归 `num_poses×3 (x,y,θ)`（8 个时刻，每时刻 x、y、朝向 θ 共 3 个数）：



- plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)


**没有独立 scorer 模块**；选择由分类头 `plan_cls`（每模式一个置信度）承担，取 argmax 即最终轨迹：



- mode_idx = poses_cls.argmax(dim=-1)
- mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
- best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)


> **多模态是什么意思？** 简单说，同一个场景可能有多种合理开法（比如「直接超车」或「先跟车再超」）。DiffusionDrive 一次给出 20 条不同风格的候选（对应 20 个 anchor 模式），最后用分类头挑一条最靠谱的。这就是「多模态输出」——比只输出一条死板轨迹更灵活。

## 二、前向传播：一张图对应的代码路径（本文重点）

把上面所有模块串成一次 `forward`。输入是 navsim 的 `AgentInput`，输出是 `(B, 8, 3)` 的轨迹（B 是 batch 里场景个数，8 个 pose，3 是 x/y/θ）。下面**逐行跟一遍**推理时的 forward，并标注每一步张量形状——注意它和 SparseDriveV2 的「检索式」完全相反，是「生成式」。

为了清晰，下面用「纯英文代码块 + 代码块外中文解说」的方式来串这 7~8 步。代码块里只放英文/数字/符号（不含任何中文，避免中英文混排对齐错位），每一步的中文含义紧跟在代码块下方的列表里。

- `compute_trajectory(agent_input)`
  - `build_features(agent_input)`
    - `img = [B, 3, 256, 1024]`
    - `lidar = [B, C, H, W]`
    - `ego_map = vector tokens`
  - `V2TransfuserModel.forward(features)`
    - `TransFuser_backbone(img, lidar)` → `fused_feat = [B, N_token, d]`
    - `build_20_anchor_queries(frozen)` → `anchor_embed = [B, 20, 8, 2]`
    - `TrajectoryHead.forward_test(anchor_embed)`
      - `noisy = add_noise(anchor, noise, t=8)` → `noisy = [B, 20, 8, 2]`
      - `for step in range(2): offset = denoiser(noisy, fused_feat, timestep); poses_reg = poses_reg + offset`
      - `plan_reg = [B, 20, 8, 3]`
      - `plan_cls = [B, 20]`
    - `mode_idx = plan_cls.argmax(dim=-1)`
    - `best_reg = plan_reg.gather(1, mode_idx)`
    - `output = [B, 1, 8, 3]`

上面每一步对应的中文含义（逐行对照）：

- `compute_trajectory(agent_input)`：navsim 的 AbstractAgent 接口，每个评测场景调一次，是整套推理的入口。
- `build_features`：把传感器变成张量。
  - `img = [B, 3, 256, 1024]`：三相机 cam_l0/f0/r0 拼成的全景图。
  - `lidar = [B, C, H, W]`：激光雷达压成的 BEV 特征（可选）。
  - `ego_map = vector tokens`：自车状态（速度/朝向/位置）和地图信息编码成的向量 token。
- `V2TransfuserModel.forward`：主模型前向。
  - `TransFuser_backbone`：图像 ResNet-34 + LiDAR ResNet-34 融合，输出 `fused_feat = [B, N_token, d]`，即统一的多模态场景特征。
  - `build_20_anchor_queries`：构造 20 个冻结的 anchor query（plan_anchor [20,8,2] 经 norm 后作初始样本），输出 `anchor_embed = [B, 20, 8, 2]`，代表 20 种驾驶模式原型（直行/左转/跟车等）。
  - `TrajectoryHead.forward_test`：截断扩散解码（推理专用）。
    - `add_noise(anchor, noise, t=8)`：给 anchor 加截断噪声（t=8 固定时刻，不是纯高斯，起点已接近合理轨迹），得 `noisy = [B, 20, 8, 2]`。
    - 循环 2 步去噪：`denoiser` 用 deformable/cross-attention 融合 noisy 与 fused_feat、用 timestep 作 FiLM 调制，预测 offset，残差累加；2 步后得到 `plan_reg = [B, 20, 8, 3]`（20 条已"雕"好的轨迹）。
    - `plan_cls = [B, 20]`：分类头对 20 条模式各打一个置信度。
  - `argmax(plan_cls)`：选置信度最高的模式，`gather` 取出对应轨迹，最终 `output = [B, 1, 8, 3]` 即选中的最优轨迹。

下面把上面每一步拆开，**每一步都给一段伪代码（纯 ASCII、标注张量形状）+ 一段大白话解释**。

---

#### 步骤 ②：构建输入特征（把传感器变成张量）

伪代码（用行内代码表示，避免代码块竖排）：

输入 `agent_input`（navsim 原始输入，含相机图、LiDAR、地图、自车状态）
- `img = stitch_3_cameras(agent_input.cameras)` ：三相机拼成全景图
- `lidar_bev = project_lidar_to_bev(agent_input.lidar)` ：激光雷达压成 BEV
- `ego_token, map_token = encode_ego_map(agent_input)` ：自车与地图编码成向量
- 输出：`features = {img, lidar_bev, ego_token, map_token}`

**大白话**：这一步是「翻译」。navsim 给的原始数据是人类友好的（图片文件、点云数组），模型只吃「张量（一堆数字）」。所以这里把三相机拼成全景图、把激光雷达压成俯视图、把自车速度和地图信息编码成向量。输出就是一堆规整的数字块，准备喂给网络。

---

#### 步骤 ④：TransFuser 融合主干（让模型「看懂」场景）

伪代码：


- img_feat  = ResNet34(img)
- bev_feat  = ResNet34(lidar_bev)

- fused_feat = cross_attention(img_feat, bev_feat)

- fused_feat = fused_feat + ego_token + map_token
- 输出: fused_feat


**大白话**：这一步是「理解场景」。想象你同时看照片（相机）和雷达（激光），脑子里把两者对上：「照片里那团白色，雷达也说那里有东西，那应该是一辆车」。ResNet 负责各自提取特征，交叉注意力负责「图像和雷达互相确认」，最后加上「我现在以多少速度在开、前方地图长啥样」这种全局信息。输出的 `fused_feat` 就是模型对「当前这一帧场景」的整体理解，后面去噪修正轨迹时，anchor 就靠它来「看路」。

---

#### 步骤 ⑤：构造 20 个 anchor query（准备 20 张草稿）

伪代码：


- plan_anchor = self.plan_anchor
- anchor = plan_anchor.unsqueeze(0).repeat(B,1,1,1)
- anchor = norm_odo(anchor)
- 输出: anchor_embed = anchor


**大白话**：这一步是「拿出 20 张驾驶草稿」。那 20 条 k-means 聚类出来的模板轨迹，被复制成 batch 里每个场景都有一份（因为不同场景只是「场景不同」，但都从同一套驾驶模式草稿出发）。注意此时每条草稿还是「模板原样」，还没结合具体场景——它只是「直行模板」「左转模板」这种通用姿势。

---

#### 步骤 ⑥-a：给 anchor 加截断噪声（把草稿轻微弄糊）

伪代码：


- noise = randn_like(anchor_embed)
- t = 8  (固定截断时间步，对所有样本一样)
- noisy = diffusion_scheduler.add_noise(anchor_embed, noise, t)
- 输出: noisy


**大白话**：这一步是「故意把草稿弄糊一点点」。为什么？因为去噪网络需要「从模糊到清晰」这个过程来发挥作用。但注意，这里只加到 `t=8`（扩散时间步很小），意味着只加了一丁点噪声，草稿还是「直行模板略歪」这种状态，远没到「纯随机噪声」。这就是「截断」的真意：**起点离答案极近，所以后面只要修两笔**。

对比一下：标准扩散会把草稿加到 `t≈500~1000`，变成纯噪声，然后要从头画，所以要几十步。DiffusionDrive 只加到 `t=8`，相当于只把草稿抖了一下，2 步就修回来了。

---

#### 步骤 ⑥-b：循环去噪 2 步（核心生成循环）

伪代码：


- current = noisy
- for step in range(2):

-     offset = denoiser(
-         traj=current,
-         scene=fused_feat,
-         t=current_timestep,
-     )
-     current = current + offset

- plan_reg = concat([current, theta_head(current)], dim=-1)
- 输出: plan_reg


**大白话**：这一步是「反复看路、反复修」。循环只跑 2 次（step_num=2）。每次循环里，去噪网络（2 层 transformer）做三件事：(1) 让 20 条轨迹点去场景特征上「做交叉注意力」，相当于每条轨迹问场景「我前面有车吗、我出车道了吗」；(2) 用当前时间步 `t` 通过 FiLM 告诉网络「这是第几步去噪」；(3) 输出一个「offset 修正量」，加到当前轨迹上。两次循环后，20 条草稿就从「略歪的模板」变成「贴合当前场景的具体轨迹」了。最后补上朝向角，得到 20 条完整轨迹 `plan_reg`。

> **为什么 2 步够？** 因为起点（anchor + t=8 噪声）已经非常接近合理轨迹，网络只需要做两次小幅修正。如果起点是纯噪声，两次修正远远不够，那就得几十步。所以「好起点」换来了「少步数」。

---

#### 步骤 ⑥-c：分类头打分（给 20 条候选各打一个分）

伪代码：


- plan_cls = cls_head(plan_reg, fused_feat)
- 输出: plan_cls


**大白话**：这一步是「给 20 条候选打分」。每条轨迹经过一个小的分类头，结合场景特征，输出一个 0~1 之间的置信度，表示「在当前场景下，这条开法有多合理」。注意分类头和去噪网络共享场景理解，所以它打分是「因地制宜」的——同一个「左转模板」，在「前方是左转路口」时分数高，在「直道」时分数低。

---

#### 步骤 ⑦：argmax 选 1 条（最终拍板）

伪代码：


- mode_idx = plan_cls.argmax(dim=-1)
- best_reg = plan_reg.gather(1, mode_idx)
- 输出: best_reg


**大白话**：这一步是「拍板」。20 个分数里取最大的那个（`argmax`），然后去 `plan_reg` 里把对应那条轨迹拿出来。比如第 3 个模式分数最高，就输出第 3 条轨迹。最终形状 `[B, 8, 3]`——batch 里每个场景一条 8 点 3 属性的轨迹，交给 navsim 评测。

---

**每一步在干啥（大白话版小结）**：
- **②④**：把相机+LiDAR 融成一套「懂场景」的特征，和 SparseDriveV2 的 backbone 角色一样。
- **⑤**：准备好 20 个「驾驶模式模板」（anchor）——这是扩散的**起点**，不是最终答案。
- **⑥-a**：在 anchor 上加一点噪声（截断到 t=8，不是从纯噪声开始），等于「故意把模板弄糊一点，等下再修」。这就是为什么只要 2 步去噪——起点已经很接近答案。
- **⑥-b**：去噪网络（2 层 CustomTransformer）反复「看场景特征 + 看当前糊轨迹」，预测「该往哪修」，逐步把轨迹雕清晰。2 步后得 20 条具体轨迹。
- **⑥-c / ⑦**：分类头判断「这 20 条里哪条最靠谱」，argmax 选出最终 1 条。

**输入 / 输出维度小结**：
- 输入图像：`[B, 3, 256, 1024]`（全景拼接）；LiDAR/BEV：`[B, C, H, W]`（可选）；anchor：`[20, 8, 2]`（冻结）。
- 中间：20 个 anchor 模式 × 2 步去噪 → 20 条候选轨迹。
- 输出：`plan_reg [B, 20, 8, 3]`（20 条候选）+ `plan_cls [B, 20]`（置信度），推理时 argmax 取 1 条。

**和 SparseDriveV2 的对照表**（一眼看懂两条路线差异）：

| 维度 | SparseDriveV2（检索式） | DiffusionDrive（生成式） |
|:---|:---|:---|
| 候选来源 | 冻结 26 万词表查表 | 20 anchor + 扩散生成 |
| decoder 干啥 | 打分 + 筛选 | 去噪 + 生成 |
| 候选数量 | 26万 → 粗筛 ~200 → 1 | 20（生成）→ 1 |
| 选轨迹 | argmax(PDMS式分数) | argmax(plan_cls) |
| 推理是否迭代 | 否（一次前馈） | 是（2 步去噪） |
| 擅长 | 贴合榜单指标、快、确定 | 覆盖词表外多模态 |

## 三、训练：怎么挂上 navsim

DiffusionDrive **不写自己的训练入口**，直接复用 navsim 官方 `run_training.py`（PyTorch-Lightning），只通过 Hydra 配置 `agent=diffusiondrive_agent` 挂载自己的模型：



- python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
-     agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent \
-     train_test_split=navtrain split=trainval trainer.params.max_epochs=100 ...


agent 配置 `navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`：


- _target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent
- config:
-   _target_: navsim.agents.diffusiondrive.transfuser_config.TransfuserConfig
- checkpoint_path: null
- lr: 6e-4


Agent 类继承 `AbstractAgent`（`transfuser_agent.py:32`），优化器用 `AdamW + WarmupCosLR`，图像 encoder 学习率 ×0.5。

### 3.1 anchor 是怎么来的（k-means 聚类）

在讲 loss 之前，必须先讲清 anchor 的来源，因为它直接决定分类标签怎么定。流程是：


- 1. 收集训练集 navtrain 里所有场景的"真值未来轨迹"（每条都是 [8, 2] 或 [8,3]）
- 2. 把所有轨迹的点展平，跑 k-means，聚成 K=20 类
- 3. 每类的中心轨迹 -> 存成 kmeans_navsim_traj_20.npy
- 4. 训练/推理时加载该文件，作为冻结的 plan_anchor [20, 8, 2]


**大白话**：作者先把训练集里人类司机实际怎么开车的几万条轨迹收集起来，用聚类算法自动分成 20 类（比如一类是「平稳直行」、一类是「中等左转」……）。每类的「平均样子」就是一条 anchor。这样 20 个 anchor 天然覆盖了训练集里最常见的 20 种驾驶姿势，作为扩散起点非常合理。

### 3.2 训练时怎么加噪（截断到 50）

回忆前面 1.2 节的代码：训练时不是加到纯噪声，而是 `timesteps = torch.randint(0, 50, ...)`，也就是只在「噪声程度 0~50」之间随机取一个时间步加噪。这是因为推理时最多只走到 t=8 附近，训练时没必要模拟「完全噪声」的极端情况——既省算力，又让训练分布和推理分布更一致（减少 train/inference gap）。

加完噪后，网络要预测「加了多少噪声 / 轨迹该往哪修」，然后用下面的 loss 监督。

### 3.3 损失函数（梯度流向）

顶层聚合在 `transfuser_loss.py:11-53`：


- loss = (
-     config.trajectory_weight * trajectory_loss
-     + config.diff_loss_weight * diffusion_loss
-     + config.agent_class_weight * agent_class_loss
-     + config.agent_box_weight * agent_box_loss
-     + config.bev_semantic_weight * bev_semantic_loss
- )


真正的**多模态监督**在模型内部逐 decoder 层计算（深监督），本体在 `modules/multimodal_loss.py`：用 GT 轨迹到 20 个 anchor 的 L2 距离找最近模式作为分类标签 → focal loss 分类 + 对该模式做 L1 回归：



- dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
- mode_idx = torch.argmin(dist, dim=-1)
- loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(poses_cls, target_classes_onehot, gamma=2.0, alpha=0.25)
- reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)
- ret_loss = loss_cls + reg_loss


**这段用大白话解释**：对于每条训练样本，我们先看「真值轨迹（人类实际怎么开）离 20 个 anchor 里哪一个最近」，最近的那个就被标记为「正确模式」（正样本）。然后：
- 分类损失（focal loss）：让模型学会「这个场景下，正确的驾驶模式应该是那个正样本模式」——相当于教分类头打分打对。
- 回归损失（L1 loss）：只对这个正样本模式对应的轨迹做「让它更接近真值」的监督——相当于教去噪网络把这条轨迹修得更准。

数学上，分类用 focal loss，公式写作：

$$ \mathcal{L}_{cls} = -\alpha (1-p_t)^\gamma \log(p_t) $$

其中 $p_t$ 是模型对正样本模式的预测概率，$\alpha=0.25$、$\gamma=2.0$ 是焦点参数，用来让模型更关注难分样本。回归用 L1 loss：

$$ \mathcal{L}_{reg} = \frac{1}{N}\sum_{i=1}^{N} | \hat{y}_i - y_i | $$

其中 $\hat{y}_i$ 是预测轨迹点，$y_i$ 是真值轨迹点。

> **注意**：仓库里 `diff_loss_weight` 路径当前 `diffusion_loss=0`——即训练时**直接监督 anchor 回归 + 分类，并不反向传播扩散重建损失**。扩散仅作为推理时的轨迹生成/修正机制。这是读源码时容易踩的坑。

**梯度流小结**：只有主干、`TrajectoryHead`（denoiser + cls/reg head）参数可训；`plan_anchor` 全程冻结。多模态 focal+L1 损失逐层加权求和后反向传播，AdamW 优化。

## 四、推理：怎么跑出 submission

### 4.1 复用官方提交脚本

DiffusionDrive 直接复用 navsim 官方 `run_create_submission_pickle.py` 生成 `submission.pkl`（未自定义）。该脚本对每个 token 调 agent 的 `compute_trajectory`：



- agent.initialize()
- trajectory = agent.compute_trajectory(agent_input)


`compute_trajectory` 本身继承 `AbstractAgent`（`abstract_agent.py:62`），内部 build features → `forward` → 取 `predictions["trajectory"]`。真正的「采样几步 + 选轨迹」落在 `TrajectoryHead.forward_test`：

- 去噪 **2 步**（`step_num=2`），从 anchor + 截断噪声（`t=8`）出发；
- 每步 DDIM `step` 更新；
- 按分类分数 argmax 从 20 个模式选一条输出。

**推理去噪循环的具体迭代过程**（补全 1.2 节的简写），用伪代码展开：



- anchor = plan_anchor.repeat(B,1,1,1)
- x = add_noise(anchor, noise, t=8)

- timesteps = [8, 4]   (示意，实际由 step_ratio 算)
- for t in timesteps:
-     noise_pred = denoiser(x, fused_feat, t)
-     x = ddim_step(x, noise_pred, t, t_prev)

- plan_reg = concat([x, theta_head(x)], -1)
- best = plan_reg.gather(1, plan_cls.argmax(-1))


**大白话**：推理时不再随机加噪（训练才随机），而是固定从 `t=8` 的轻微噪声起点出发，然后按 DDIM 的跳步规则走 2 步（比如从 t=8 到 t=4 再到 t=0），每步都拿场景特征做交叉注意力修正。2 步结束，轨迹就清晰了，最后分类头挑最好的那条。

### 4.2 本地算分 / 上榜

评测用官方 `run_pdm_score.py`；正式上榜把 `submission.pkl` 传 HuggingFace 官方空间（navtest / navhard 榜）。

**训练和推理的差异总结表**：

| 环节 | 训练 | 推理 |
|:---|:---|:---|
| 加噪时间步 | 随机 0~50 | 固定 t=8 |
| 去噪步数 | 不需要迭代（直接监督回归） | 2 步 DDIM 迭代 |
| 扩散重建损失 | 关闭（diff_loss=0） | 不适用（无监督，只用 cls+reg 头） |
| 目标 | 让 anchor 模式会分类 + 会残差回归 | 从 anchor 出发 2 步雕出并选最优轨迹 |
| anchor 状态 | 冻结 | 冻结（同一份） |

## 闭环总结

| 阶段 | 入口 | 关键代码 | 训练/推理差异 |
|:----:|:------|:----------|:--------------|
| 架构 | `transfuser_model_v2.py` TrajectoryHead | anchor 高斯 + 截断 timestep（训练 50 / 推理 t=8、2 步） | anchor 冻结，两侧共用 |
| 训练 | navsim `run_training.py` + `agent=diffusiondrive_agent` | 多模态 focal+L1 损失（深监督） | 随机 t∈[0,50) 加噪回归 |
| 推理 | navsim `run_create_submission_pickle.py` | `forward_test` 去噪 2 步 + cls argmax 选轨迹 | 固定 t=8 起，2 步去噪 |

**一句话记住这个闭环**：训练时让 20 个 anchor 学会「覆盖所有驾驶模式 + 对每个模式做残差回归」，推理时从 anchor 加一点截断噪声、2 步去噪、分类头挑最好的那条——这就是 DiffusionDrive 又快又多模态的秘密。

## 个人理解与思考

**1. 「截断扩散」是对自动驾驶场景的精准定制。** 标准 DDPM 从纯噪声生成，需要几十步才能收敛；但驾驶轨迹不是「任意图像」，它有强结构（平滑、符合运动学、贴合车道）。DiffusionDrive 用 k-means anchor 把起点从「纯噪声」拉到「合理驾驶模式附近」，于是扩散只需做小幅修正——2 步足矣。这启示我们：**扩散的起点选择比步数更重要**，给模型一个好的先验，能极大压缩生成开销。换句话说，与其苦哈哈地训一个「从噪声到轨迹」的万能生成器，不如先告诉模型「大方向就这 20 种」，让它专注微调。

**2. 训练和推理的目标不一致，是个值得警惕的信号。** 训练时 `diffusion_loss=0`，模型实际是在做「anchor 回归 + 分类」的硬监督，扩散重建损失没参与。这意味着推理时的扩散去噪，本质上是在一个「没被扩散损失调过」的 decoder 上跑——它能 work，靠的是 anchor 先验足够强 + 残差回归 head 已经学好了。但严格说，训练/推理的分布是有 gap 的：训练时网络看到的是「anchor 加 0~50 步噪声」，推理时是「anchor 加固定 t=8 噪声 + 2 步 DDIM」。若想把扩散用得更充分（比如支持更长 horizon、更自由的多模态），应该把扩散重建损失也打开，让训练真正覆盖推理会走到的噪声区间。

**3. 全景图拼接是取舍鲜明的工程选择。** 三相机拼一张 1024×256 全景图，省掉跨相机 fusion 模块，简洁高效；但几何畸变和信息损失不可忽视（侧视相机被压扁）。在 NAVSIM 这种前视为主的评测里够用，放到需要全向感知的开放场景，可能不如真正多相机 token + cross-attention 稳健。我个人倾向认为，这是「为榜单效率妥协」的设计，而非「为实车鲁棒」的设计。

**4. 和 SparseDriveV2 的对比给人的启发。** SparseDriveV2 是「检索式」——在 26 万字典里选，推理无迭代、贴合榜单指标；DiffusionDrive 是「生成式」——从 anchor 扩散出轨迹，擅长覆盖字典外的多模态。两者都挂在 navsim 上、都复用官方管线，但哲学相反：**前者相信「好答案在被枚举的候选里」，后者相信「好答案该被生成出来」**。作为 VLA 方向工程师，我认为实车部署更看重 SparseDriveV2 式的可解释与确定性（输出必在可行集），而算法探索阶段 DiffusionDrive 式的生成灵活性更有想象空间。值得补充的是，DiffusionDrive 的「生成」其实也被 anchor 强烈约束，所以它并非完全自由生成，而是「受限生成」——这恰好是它能在实车场景里保底安全的原因，也模糊了「生成 vs 检索」的界线：它更像「在 20 个检索到的模板附近做生成式精修」。

**5. 一个延伸想法：anchor 数量是不是越多越好？** 本文用 20 个。更多 anchor（比如 50）能覆盖更细的驾驶模式，但推理时每条都要走 2 步去噪，计算量线性增长；更少（比如 6）则多模态表达力下降。20 是作者在「覆盖度 vs 算力」间的折中。未来若上更强去噪网络或更长 horizon，这个超参值得重新扫一遍。

## 延伸阅读

- 同系列对比：[SparseDriveV2 代码讲解](/posts/code/sparsedrivev2代码讲解/)——另一条「词汇表 + 两级评分」路线
- 榜单背景：[NAVSIM 排行榜深度分析](/posts/knowledge/navsim排行榜深度分析/)
