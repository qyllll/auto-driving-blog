---
title: "AutoVLA 代码讲解：把「开车」变成「说动作 token」并用 GRPO 微调"
date: 2026-07-20
description: "拆解 ucla-mobility/AutoVLA：动作 codebook 离散化、快慢思考两阶段训练、SFT + GRPO 拒绝采样微调，看 VLA 如何端到端输出可执行驾驶动作"
tags: ["VLA", "GRPO", "动作token", "代码讲解"]
categories: ["代码讲解"]
summary: "「面向零基础读者逐行拆解 ucla-mobility/AutoVLA（NeurIPS 2025）：从 VLA/动作 codebook/KMeans/快慢思考/SFT/GRPO 等概念讲起，配合论文架构图 + 自绘 SVG（整体架构、KMeans 聚类、两阶段训练流程），逐文件细讲 build_codebook 聚类、modeling_auto_vla 挂 action_head、LoRA 微调策略、SFT 数据模板设计、Stage-2 RFT 拒绝采样精修、推理旁路，最后给出个人思考与系列关系」"
---

## 写给零基础读者：读这篇之前先搞懂几个名词

AutoVLA 的代码不长，但它把好几个「听起来很唬人」的概念揉在了一起。你要是直接冲进源码，多半会被一堆缩写劝退。所以我们先花点时间，把会反复出现的黑话全部翻译成大白话。你现在不用全记住，有个印象，等下看代码时回头对照就行。

- **VLA（Vision-Language-Action，视觉语言动作模型）**：给模型一张图（有时再加一句话指令），它直接吐出「车该怎么开」。它的核心思路是：**把开车这件事，当成一个「语言建模」问题来做**——图像当输入，动作当输出，中间用一个已经很会「说话」的大模型（VLM）串起来。AutoVLA 用的基座就是 Qwen2.5-VL-3B。

- **Qwen2.5-VL**：阿里开源的一个「能看图说话」的多模态大模型（VLM，Vision-Language Model）。它本来的看家本领是「看一张图，用自然语言回答问题」。AutoVLA 就是站在它的肩膀上，把它从「会说话」改造成「会开车」。这里用的是 3B（30 亿参数）的小号版本，方便训练和部署。

- **token（离散词）**：大模型眼里的世界不是连续的，而是一个个「词」。它把一句话拆成一串编号（比如「你」=101，「好」=102），每个编号就是一个 token。模型的工作说白了就是「猜下一个 token 是几号」。这是理解 AutoVLA 最关键的一环：**它想办法把驾驶动作也变成 token**，这样开车就和说话变成了同一件事。

- **动作 codebook（动作词表）**：这是 AutoVLA 的灵魂。真实的驾驶动作是连续的数（比如方向盘转 12.3 度、油门踩 0.4）。可大模型只认「离散的词」。怎么办？把海量真实动作拿去做聚类，聚成 K 个「典型动作」，每个典型动作给一个编号——这张「编号 → 典型动作」的对照表，就叫动作 codebook。有了它，一个连续动作就能被近似成「最像的那个编号」，也就变成了一个 token。

- **KMeans 聚类**：一种最经典的「物以类聚」算法。你给它一堆点和一个数字 K，它就帮你把这些点分成 K 堆，并算出每一堆的「中心点」。AutoVLA 拿它把海量连续驾驶动作聚成 K 个簇，每个簇的中心就是 codebook 里的一个「动作词」。

- **快慢思考（Slow-Fast Thinking）**：人开车也是这样——遇到复杂路口会「慢慢想」（我要不要让行？对面那辆车会不会加塞？），遇到直道就「凭肌肉记忆」直接开。AutoVLA 让模型先用自然语言「慢思考」一遍（把推理理由写出来），再输出「快思考」的动作 token（真正拿去执行的）。妙的是，推理时如果你嫌慢，可以把慢思考那段直接**旁路掉**，只留快思考，速度立刻上来。

- **SFT（Supervised Fine-Tuning，监督微调）**：拿专家开车的数据，一条条喂给模型让它模仿。就像驾校教练手把手教你「这个场景就该这么打方向」。这是 AutoVLA 的第一阶段，目的是让模型先「会开」。

- **GRPO / RFT（Rejection sampling Fine-Tuning，拒绝采样微调）**：SFT 学出来的模型往往「平庸」——它学的是所有专家的平均值，遇到需要果断的场景反而畏手畏脚。第二阶段就用强化学习的思路来「精修」：让模型自己开很多遍，用规则去打分，只挑那些开得好的轨迹留下来，再拿这些好样本继续训。这套「多开几遍 → 打分 → 只留高分 → 再训」的做法，本质就是拒绝采样，也和 GRPO（一种不需要价值网络的强化学习算法）的思想相通。

- **rule reward（规则奖励）**：给模型开的车打分的「评分标准」。但它不是训一个神经网络来打分，而是**写死的规则**——有没有撞车、有没有闯红灯、开得舒不舒服（急刹急打方向就扣分）。规则奖励的好处是简单、可靠、不会被模型「钻空子」。

如果这些词你都有个模糊印象了，下面读代码会顺畅很多。我们开始。

## 为什么要讲 AutoVLA 的代码

这是「自动驾驶代码讲解」系列的第 6 篇。前面我们已经走过了 UniAD / VAD（端到端回归范式，直接拿网络算出轨迹）、DiffusionDrive（用扩散模型生成轨迹）、DriveVLA-W0（VLA 大模型 + 世界模型）。这条路线越走越有意思：从「网络直接回归几个数」，一路演化到「让大模型像说话一样把动作说出来」。

AutoVLA（NeurIPS 2025，加州大学洛杉矶分校 ucla-mobility 团队）就是这条演化线上最纯粹的一个样本。它把「端到端自动驾驶」推到了一个几乎极端的位置：**连驾驶动作都变成了大模型词表里的 token，于是整套 NLP 的训练栈（SFT、强化学习对齐）几乎可以原样搬过来用**。

它反直觉、也最值得讲的一点是：开车这件事，居然真的可以被压缩成「说一串动作词」，而且效果还不错。代码在 `ucla-mobility/AutoVLA`。这篇我们不推公式，直接顺着仓库把关键实现讲清楚。

> **一句话结论**：AutoVLA = Qwen2.5-VL-3B 当基座 + 一张动作 codebook（KMeans 把连续动作聚成 K 个离散 token）+ 快慢思考（先自然语言推理、再吐动作 token，推理可旁路）+ 两阶段训练（Stage-1 SFT 模仿、Stage-2 RFT/GRPO 拒绝采样精修）。开车 = 让大模型「说出正确的动作 token」。

### AutoVLA 框架总览

这是论文中的 AutoVLA 整体框架图：

![AutoVLA Framework (原论文 Figure 1)](/images/autovla/autovla_framework.png)

从上图可以看到完整的数据流：
- **输入**：多视角图像 + 自车状态 + 导航指令 → Vision Encoder (SIGLIP) 编码为视觉 token
- **推理**：VLM Backbone 自回归生成推理 token（慢思考）→ 动作 token（快思考）
- **动作解码**：动作 token id → Codebook 查表 → 连续驾驶动作（方向盘、油门、刹车）
- **训练**：SFT 联合学习推理 + 动作，RFT（GRPO）精修决策

## 架构总览：先看地图

![AutoVLA 完整架构图：输入 → VLM → 快慢思考 → Codebook → 动作](/images/autovla/autovla_overview.svg)

在钻进任何一个文件之前，先把整条链路在脑子里画一遍。AutoVLA 的一次「决策」大致是这样走的：

- **输入端**：把当前的摄像头图像（一张或多张）+ 一段文字提示（prompt，比如「描述场景并给出驾驶决策」）一起塞进 Qwen2.5-VL-3B。
- **理解与慢思考**：Qwen2.5-VL 照它本来的方式，把图像编码成视觉特征，和文字一起做自回归生成——先生成一段自然语言的「推理理由」（这就是慢思考，slow）。
- **输出快思考**：紧接着，模型继续生成一串特殊的「动作 token」（这就是快思考，fast）。这些 token 的编号，对应的就是 codebook 里的某个典型动作。
- **动作还原**：拿到动作 token 的编号后，去 codebook 里一查，把编号翻译回连续的驾驶动作（方向、油门、刹车），交给车去执行。

训练分两步走：

- **Stage-1（SFT 模仿）**：用专家数据，让模型学会「看图 → 说理由 → 输出正确动作 token」这一整套。
- **Stage-2（RFT / GRPO 精修）**：让训好的模型自己多开几遍，用规则奖励挑出开得好的轨迹，再拿这些好样本回炉重训，让它在关键场景更果断、更安全。

推理时，如果追求低延迟，可以把慢思考那段推理**整个跳过**，直接让模型吐动作 token，速度立刻提上来——这就是快慢思考设计的实用价值。

> **关键认知**：AutoVLA 里没有一个「专门的规划器」或「专门的控制器」。规划、决策、动作，全部被塞进了大模型的 token 生成里。它把一个控制问题，硬生生变成了一个「预测下一个 token」的语言问题。这就是为什么它能几乎白嫖整套 NLP 训练技术栈。

## 项目结构：先厘清边界

读任何开源代码，第一件事是搞清楚「哪些是作者自己写的核心、哪些是白嫖的外部库」。AutoVLA 的边界很清楚：

- **基座（外部）**：Qwen2.5-VL-3B。这是阿里的开源 VLM，AutoVLA 不重新训它的主体结构，只在它上面做改造和微调。
- **AutoVLA 自己写的核心**：三件事——
  1. **动作 token 化**：怎么把连续动作聚类成 codebook（`build_codebook.py`）、怎么在模型里挂一个动作头去预测/解码这些 token（`modeling_auto_vla.py` + `action_head.py`）。
  2. **两阶段训练**：Stage-1 的 SFT（`sft.py`）、Stage-2 的 RFT/GRPO（`grpo_trainer.py`）。
  3. **推理入口**：含慢思考旁路的开关（`run_infer.py`）。

用普通 Markdown 嵌套列表画一下仓库结构（避免在代码块里逐字符对齐中文）：

- AutoVLA/
  - auto_vla/model/ （模型定义：Qwen2.5-VL + 动作头）
    - modeling_auto_vla.py （在基座 VLM 上挂 action_head）
    - action_head.py （动作 token 的投影与解码）
  - auto_vla/data/ （数据集与 action codebook 构造）
    - build_codebook.py （KMeans 聚类生成 codebook）
    - action_token_dataset.py （把样本组织成「图 + 慢思考 + 动作 token」）
  - auto_vla/train/ （两阶段训练）
    - sft.py （Stage-1 模仿学习）
    - grpo_trainer.py （Stage-2 RFT / GRPO 拒绝采样微调）
  - auto_vla/infer/ （推理，含慢思考旁路）
    - run_infer.py
  - scripts/ （一键脚本）
    - action_token_cluster.sh （跑聚类生成 codebook）
    - run_sft.sh （跑 Stage-1）
    - run_rft.sh （跑 Stage-2 GRPO / 拒绝采样）
  - configs/ （超参配置）

逐个一句话说明它干嘛：

- `build_codebook.py`：把海量连续驾驶动作用 KMeans 聚成 K 个簇，簇中心存成 codebook。这是「连续 → 离散」的入口。
- `action_token_dataset.py`：把每条训练样本组织成「图像 + 慢思考文字 + 动作 token id」三件套，喂给训练器。
- `modeling_auto_vla.py`：继承 Qwen2.5-VL，在输出侧额外挂一个 `action_head`，让模型除了会说话还会「说动作」。
- `action_head.py`：动作头的具体实现，负责把隐藏状态投影成 codebook 维度的 logits（预测哪个动作 token），以及把 token 反查成连续动作。
- `sft.py`：Stage-1 训练主循环，模仿学习。
- `grpo_trainer.py`：Stage-2 训练主循环，rollout + 规则打分 + 拒绝采样 + 再微调。
- `run_infer.py`：推理入口，`fast_only` 开关决定要不要跑慢思考。

> **边界提醒**：下面所有代码都是**简化后的示意伪代码**，用来讲清楚思路，不是逐字符照抄仓库。真实代码会有更多工程细节（分布式、混合精度、数据并行等），但主干逻辑就是这些。代码块里我一律用英文，中文解释都放在块外，避免对齐错乱。

## 一、动作 codebook：把连续动作变成离散 token

![KMeans 聚类生成动作 Codebook 流程：连续动作 → 聚类 → 离散 token](/images/autovla/codebook_kmeans.svg)

这是整个 AutoVLA 的地基。没有它，大模型根本没法「说」出动作。核心文件是 `data/build_codebook.py`。

### 1.1 为什么要聚类

先想清楚痛点：真实驾驶动作是连续的实数。比如某一时刻的动作可能是「方向盘 -3.7 度、油门 0.42、刹车 0.0」。这是一个连续向量。但大模型的词表是**有限个离散 token**，它没法直接吐出一个任意实数。

解决办法很朴素：**既然连续值有无穷多种，那我就挑出 K 个最有代表性的「典型动作」，以后所有动作都用最接近的那个典型动作来近似**。这 K 个典型动作，就是 codebook。挑「典型动作」的过程，正好就是 KMeans 聚类干的事。

### 1.2 用 KMeans 生成 codebook

```python
from sklearn.cluster import KMeans
import numpy as np

actions = load_all_actions()          # (N, D)
kmeans = KMeans(n_clusters=K, random_state=0).fit(actions)
codebook = kmeans.cluster_centers_    # (K, D)
token_ids = kmeans.predict(actions)   # (N,)
np.save("codebook.npy", codebook)
np.save("token_ids.npy", token_ids)
```

逐行说人话：

- `load_all_actions()`：把训练集里所有时刻的动作都读进来，堆成一个大矩阵，形状 `(N, D)`——N 是样本总数（可能几百万），D 是动作维度（比如方向、油门、刹车就是 3 维，也可能是一段未来轨迹的多个点拼起来）。
- `KMeans(n_clusters=K).fit(actions)`：让 KMeans 把这 N 个动作分成 K 堆。K 就是 codebook 的大小，也就是「动作词表」有多少个词。K 太小，动作被压得太粗糙（转向精度不够）；K 太大，token 太多、学起来难。这是个要调的超参。
- `kmeans.cluster_centers_`：这就是 K 个簇的中心，形状 `(K, D)`。每一行是一个「典型动作」——这正是我们要的 codebook。
- `kmeans.predict(actions)`：把每个原始动作映射到「离它最近的簇编号」，得到 `(N,)` 的 token id 数组。这一步就是把连续动作**离散化**成 token。
- 最后把 codebook 和 token_ids 存盘，训练时直接读。

对应的一键脚本是 `scripts/action_token_cluster.sh`，它内部就是调用上面这段逻辑，把整个数据集扫一遍生成 codebook。

> **关键认知**：codebook 是连接「大模型」和「车」的翻译词典。模型只管吐编号，编号怎么变回车能执行的动作，全靠这本词典查表。所以 codebook 的质量（K 选多少、动作怎么归一化）直接决定上限——词典太粗糙，模型说得再准也开不好。

#### 超参 K 怎么选

K（codebook 大小）是 AutoVLA 最重要的超参数，直接影响性能。论文中的探索显示：

- **K 太小（如 64）**：量化误差大，动作精度不足，模型只能在很粗的粒度上选择，泊车、跟车等精细场景表现差
- **K 适中（如 256–512）**：量化误差可接受，模型能学到有区分度的动作分布，nuPlan 闭环评测分数最高
- **K 太大（如 1024+）**：动作粒度够细了，但分类任务变难（1024 类比 256 类难学得多），并且尾部类别的样本太少，模型学不好那些不常见的动作

论文最终采用的 K 在 256–512 之间，这个范围在「精度」和「可学性」之间取得了最佳平衡。相比之下，DriveVLA-W0 用 256 个均匀分箱（uniform binning），而 AutoVLA 的 KMeans 聚类是**数据驱动**的——高频动作区域被分配更多的 token，低频区域共享较少的 token，在同样的 K 下能达到更低的平均量化误差。

### 1.3 反查：从 token 还原成动作

模型推理时吐出的是编号，得翻译回连续动作才能开车。这一步就是简单的查表：

```python
def token_to_action(token_id, codebook):
    return codebook[token_id]         # (D,)
```

就这么一行。`token_id` 是模型吐出的编号，拿它去 codebook 里取出对应那行，就是连续动作向量。是不是简单到有点朴素？但这正是 VLA 优雅的地方——离散化只是一层薄薄的翻译，两头都干净。

> **一句话澄清**：离散化必然带来「量化误差」。你的真实动作是 -3.7 度，最近的 codebook 词可能是 -4.0 度，这 0.3 度就丢了。K 越大误差越小，但 token 越多越难学。AutoVLA 的做法是在「精度」和「可学性」之间找一个平衡的 K。这和 DriveVLA-W0 用 256 个 bin 均匀分箱是同一类思路，只不过 AutoVLA 用聚类（数据驱动、非均匀），能把 token 更多地分配给「常见动作」区域。

## 二、模型：在 Qwen2.5-VL 上挂一个动作头

有了 codebook，接下来要让大模型能「预测动作 token」。核心文件是 `model/modeling_auto_vla.py`。

### 2.1 继承基座，挂一个 action_head

```python
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

class AutoVLA(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, num_action_tokens=K):
        super().__init__(config)
        self.action_head = nn.Linear(config.hidden_size, num_action_tokens)
```

说人话：

- `class AutoVLA(Qwen2_5_VLForConditionalGeneration)`：直接**继承** Qwen2.5-VL 的现成模型类。这意味着 Qwen 会看图、会编码、会生成文字的所有本事，AutoVLA 全盘继承，一行都不用重写。
- `self.action_head = nn.Linear(hidden_size, num_action_tokens)`：唯一新增的东西。就是一个线性层，把模型内部的隐藏状态（`hidden_size` 维）投影成 `K` 维的分数（logits）。这 K 个分数，就是「模型觉得下一个动作是 codebook 里第几个词」的打分。

一句话：**AutoVLA = 原封不动的 Qwen2.5-VL + 一个额外的线性动作头**。就这么点改动。

### 2.2 LoRA 微调：为什么只改一小部分参数

实际训练时，AutoVLA 不会全量更新 Qwen2.5-VL 的 30 亿参数——那样太贵了。它的做法是用 **LoRA（Low-Rank Adaptation）** 在注意力层插入可训练的低秩矩阵：

```python
# 伪代码：在 attention 的 q_proj, k_proj, v_proj, o_proj 挂 LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,               # 低秩秩大小（越大越强但越贵）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)
model = AutoVLA.from_pretrained("Qwen2.5-VL-3B")
model = get_peft_model(model, lora_config)
model.action_head = ActionHead(...)  # 动作头全量训练（不做 LoRA）
```

这意味着：
- Qwen 主干只通过 LoRA 调一小部分参数（约 0.1%–1%），保持预训练知识不破坏
- `action_head` 是新加层，**全量训练**——因为它是随机初始化的，信息量最多，需要彻底学
- 两阶段训练都用同一个 LoRA 配置，Stage-2 继续微调 Stage-1 的 LoRA 权重

> **关键认知**：这和 DriveVLA-W0 的做法相反——DriveVLA-W0 全量微调了整个 VLM 主干。LoRA 更省显存、更难训飞，但理论上限不如全量。AutoVLA 选 LoRA 是典型的「稳」字优先。

### 2.3 前向传播：文字损失 + 动作损失

```python
def forward(self, pixel_values, input_ids, labels, action_labels=None):
    out = super().forward(
        pixel_values=pixel_values,
        input_ids=input_ids,
        labels=labels,
        output_hidden_states=True,
    )
    last_hidden = out.hidden_states[-1]              # (B, L, H)
    action_logits = self.action_head(last_hidden)    # (B, L, K)
    if action_labels is not None:
        loss_a = F.cross_entropy(
            action_logits.view(-1, K),
            action_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = out.loss + loss_a
        return total_loss, action_logits
    return action_logits
```

逐段说人话：

- `super().forward(...)`：调用 Qwen 原本的前向。它照常算「文字部分」的 next-token 预测损失（存在 `out.loss` 里）——也就是让模型学会生成那段慢思考推理。`output_hidden_states=True` 是为了把中间隐藏状态掏出来给动作头用。
- `last_hidden = out.hidden_states[-1]`：取最后一层的隐藏状态，形状 `(B, L, H)`——B 是批大小，L 是序列长度，H 是隐维度。这里面浓缩了模型「看完图、想完理由」之后的全部理解。
- `action_logits = self.action_head(last_hidden)`：动作头把每个位置的隐藏状态投影成 K 维打分，形状 `(B, L, K)`。
- `F.cross_entropy(...)`：动作部分用交叉熵损失——本质是个 K 分类问题，让模型在「该输出动作的那些位置」预测出正确的 codebook 编号。`ignore_index=-100` 是个常规技巧，把「不该算动作损失的位置」（比如文字部分）标成 -100 跳过。
- `total_loss = out.loss + loss_a`：**文字损失 + 动作损失一起回传**。文字损失逼模型学会说理由，动作损失逼模型学会说动作。两个损失共享同一个主干，于是「理解」和「决策」被绑在一起联合训练。

> **关键认知**：文字部分和动作部分用的是**两套输出头**——文字走 Qwen 原本的语言头（预测词表里的普通 token），动作走新加的 `action_head`（预测 codebook 里的 K 个动作 token）。它们共享 Qwen 主干的理解，但各管各的输出空间。这是「快慢思考」在结构上的落地：慢思考是文字头的活，快思考是动作头的活。

## 三、action_head：动作 token 的投影与解码

上面的动作头只是一个 `nn.Linear`，看起来太简单。真实的 `model/action_head.py` 通常会封装得更完整一点——把「投影成 logits」「从 logits 采样 token」「token 反查成连续动作」都收进一个模块，用起来更顺手。

```python
class ActionHead(nn.Module):
    def __init__(self, hidden_size, codebook):
        super().__init__()
        self.num_tokens = codebook.shape[0]
        self.proj = nn.Linear(hidden_size, self.num_tokens)
        self.register_buffer("codebook", torch.tensor(codebook))

    def logits(self, hidden):
        return self.proj(hidden)                 # (B, L, K)

    def decode(self, action_token_id):
        return self.codebook[action_token_id]    # (..., D)

    def loss(self, hidden, action_labels):
        logits = self.logits(hidden)
        return F.cross_entropy(
            logits.view(-1, self.num_tokens),
            action_labels.view(-1),
            ignore_index=-100,
        )
```

逐块说人话：

- `self.proj`：投影层，隐藏状态 → K 维 logits，和上一节一样。
- `self.register_buffer("codebook", ...)`：把 codebook 作为「不参与训练的常量」挂进模块（buffer 会跟着模型一起搬到 GPU、一起存盘，但梯度不会更新它）。因为 codebook 是聚类算好的固定词典，训练时不动它。
- `logits(hidden)`：算打分，给训练和推理共用。
- `decode(action_token_id)`：**解码**——拿模型吐出的编号去 codebook 查表，得到连续动作。这就是前面 `token_to_action` 的封装版。
- `loss(hidden, action_labels)`：**投影**方向的训练损失，K 分类交叉熵。

所以 action_head 干的就两件对称的事：训练时把隐藏状态**投影**成 logits 去算分类损失（学会预测正确 token）；推理时把预测出的 token id **解码**回连续动作（拿去开车）。一投一解，闭环。

> **一句话澄清**：为什么把 codebook 存成 buffer 而不是可训练参数？因为如果让 codebook 跟着一起训，它就会「漂移」——训练早期模型还没学好，反而把词典带偏了。固定 codebook 让「动作空间」始终稳定，模型只需专心学「在什么场景说哪个词」，训练更稳。

## 四、快慢思考的两阶段训练

![AutoVLA 两阶段训练流程：Stage-1 SFT 模仿学习 → Stage-2 RFT/GRPO 自我改进](/images/autovla/training_pipeline.svg)

模型结构就绪，接下来是怎么训。AutoVLA 分两个阶段，对应两个脚本 `run_sft.sh` 和 `run_rft.sh`。

### 4.1 Stage-1：SFT 模仿（先学会开车）

第一阶段的目标很简单：让模型先「会开」。做法就是拿专家驾驶数据模仿学习。跑起来只要一行：

```bash
bash scripts/run_sft.sh
```

SFT 的数据很讲究：每一条样本不只有「图像 + 正确动作」，还带着一段**慢思考的自然语言推理**。比如：

- 图像：一个有行人的路口。
- 慢思考（文字标注）：「前方右侧有行人正在过马路，应减速让行，保持车道。」
- 动作 token：对应「减速、保持方向」的那个 codebook 编号。

#### 数据模板长什么样

这是理解模型训练最重要的细节。`action_token_dataset.py` 的核心工作就是把原始数据组织成以下模板：

```
<|im_start|>user
<image>\n当前场景多视角图像已提供。请描述场景、推理驾驶决策，然后输出动作 token。<|im_end|>
<|im_start|>assistant
前方有行人正在过马路，右侧车道被施工车辆占用，左侧车道畅通。
因此我选择向左变道并适当加速，安全通过该路口。
<action_42><|im_end|>
```

模型看到的 token 序列是：

```
[img_tokens][text_tokens][cot_tokens][action_token_id][eos]
```

其中：
- `img_tokens`：视觉编码器输出的图像 token（SIGLIP ViT 产出，约 256–1024 个）
- `text_tokens`：系统提示 + 用户指令（固定模板）
- `cot_tokens`：慢思考推理文字（**需要模型自己生成的监督目标**）
- `action_token_id`：一个特殊的 token id，对应 codebook 里的某个动作（**由 action_head 预测**）

对应的标签（labels）设置：
- `cot_tokens` 位置：**文字 loss 生效**，监督模型学会生成推理
- `action_token_id` 位置：**动作 loss 生效**（由 `action_labels` 提供），`ignore_index=-100` 跳过其他位置
- 用户指令部分：`ignore_index=-100` 跳过（不需要预测用户的提问）

为什么 `action_labels` 和 `labels` 要分开？因为动作 token id（如 42）在 NLP 词表里可能对应的是另一个词。如果用同一个 `labels` 去监督，文字头会以为要在那个位置输出词表里的第 42 个词（可能是"to"或"the"），但动作头要在那里输出 codebook 的第 42 个动作——**两套头看的是同一个位置、但监督信号不同**。这就是为什么 forward 里需要单独的 `action_labels` 参数。

这样模型学的是一整条链路：**看到场景 → 说出为什么 → 输出正确动作**。训练主循环长这样：

```python
model = AutoVLA.from_pretrained("Qwen2.5-VL-3B")
optimizer = AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    loss, _ = model(
        pixel_values=batch["images"],
        input_ids=batch["input_ids"],       # includes reason + action template
        labels=batch["text_labels"],        # supervise the slow-thinking text
        action_labels=batch["action_token"],# supervise the fast action token
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

说人话：

- `input_ids` 里是一整套模板：图像占位 + 提示语 + 一段留给慢思考的位置 + 一个留给动作 token 的位置。
- `labels`（文字标签）监督慢思考那段——让模型学会生成合理的推理理由。
- `action_labels`（动作标签）监督动作 token——让模型学会在最后吐出正确的 codebook 编号。
- 两个损失（文字 + 动作）在 `model.forward` 里已经加好了，一次 `backward` 全部回传。

跑完 Stage-1，模型就是一个「会看图、会说理由、会输出动作 token」的基本可用司机了。但它有个通病：学的是所有专家的**平均行为**，遇到需要果断决策的危险场景，往往会「和稀泥」输出一个不痛不痒的动作。这就要靠 Stage-2 来治。

> **关键认知**：SFT 只会「模仿」，不会「判断好坏」。它把专家数据里的所有动作都当成对的照单全收，包括那些平庸甚至矛盾的示范。所以 SFT 的天花板就是「专家的平均水平」。想突破，必须引入「哪个动作更好」的信号——这正是 Stage-2 强化学习要补的东西。

### 4.2 Stage-2：RFT / GRPO 强化（学会开得更好）

第二阶段用强化学习的思路精修。AutoVLA 用的是 **RFT（拒绝采样微调）**，它和 GRPO 是一脉相承的思想。核心文件 `train/grpo_trainer.py`，一键脚本：

```bash
bash scripts/run_rft.sh
```

先讲清楚 RFT 的直觉。SFT 是「照抄专家」，RFT 是「让模型自己练，然后只保留练得好的」。具体四步：

1. **rollout（多开几遍）**：让当前模型对同一个场景生成好几条不同的轨迹（靠采样的随机性产生差异）。
2. **打分**：用规则奖励给每条轨迹打分——撞了车扣大分、闯红灯扣分、开得平稳加分。
3. **拒绝采样（只留高分）**：把分数低的轨迹扔掉，只保留分数高的那些。
4. **再微调**：拿这些「自己产出的高分样本」当新的 SFT 数据，继续训模型。

主循环长这样：

```python
policy = load_sft_model()
optimizer = AdamW(policy.parameters(), lr=5e-6)

for it in range(RFT_ROUNDS):
    batch = sample_scenes(N)
    dataset = []
    for scene in batch:
        rollouts = [policy.generate(scene) for _ in range(M)]   # M samples per scene
        rewards = [rule_reward(scene, traj) for traj in rollouts]
        best = max(zip(rollouts, rewards), key=lambda x: x[1])
        if best[1] > threshold:
            dataset.append((scene, best[0]))                    # keep the good one
    policy = sft_step(policy, dataset)                          # fine-tune on kept samples
```

逐步说人话：

- `rollouts = [policy.generate(scene) for _ in range(M)]`：对每个场景，让模型采样生成 M 条不同轨迹。因为生成时有随机性，这 M 条会有好有坏。
- `rewards = [rule_reward(scene, traj) ...]`：给每条轨迹用规则打分（下一节细讲规则）。
- `best = max(...)`：挑出这一组里分数最高的那条。这就是 GRPO 的核心思想——**组内相对比较**，不需要一个绝对的价值网络来估计「这个状态值多少分」，只要比较同一场景下哪条轨迹更好就行。
- `if best[1] > threshold`：只有当最好那条也确实够好（超过阈值）才保留，否则这个场景这一轮就跳过（拒绝采样的「拒绝」）。
- `sft_step(policy, dataset)`：拿保留下来的高分样本，再做一遍 SFT 式的微调。

看出来了吗？RFT 在工程上其实**退化成了「拒绝采样 + 再监督」**——不需要复杂的策略梯度、不需要价值网络、不需要 KL 约束那一大套。它把强化学习最难落地的部分全砍了，只留下「多生成、挑好的、再学」这个朴素但极稳的循环。

> **一句话澄清**：为什么说它和 GRPO 相通？GRPO（Group Relative Policy Optimization）的精髓就是「对同一个 prompt 采一组答案，用组内相对好坏当优势信号，不用价值网络」。AutoVLA 的 RFT 把这个思想推到极致：连策略梯度都不算了，直接「组内选最好的那条，当正样本重训」。这是一种更粗暴、但工程上更不容易训飞的近似。

### 4.3 规则奖励长什么样

规则奖励是 Stage-2 的评分标准，它决定了「什么样的驾驶算好」。AutoVLA 用的是**写死的规则**，不是学出来的奖励模型：

```python
def rule_reward(scene, traj):
    r = 0.0
    if has_collision(scene, traj):
        r -= 10.0
    if runs_red_light(scene, traj):
        r -= 5.0
    r -= comfort_penalty(traj)      # jerk, hard braking, sharp steering
    r += progress_reward(traj)      # making forward progress
    return r
```

说人话：

- `has_collision`：撞了就重罚(-10)。安全是第一位的。
- `runs_red_light`：闯红灯罚(-5)。遵守交规。
- `comfort_penalty`：舒适性惩罚——急刹、急打方向、忽快忽慢（jerk 大）都扣分。
- `progress_reward`：有效前进给奖励，防止模型学成「原地不动最安全」的躺平策略。

用规则奖励而不是训一个奖励模型，好处是：**简单、透明、不会被模型钻空子**。缺点是它只能覆盖你写得出规则的那些维度（撞车、闯红灯好判断，但「开得像个老司机」就很难用规则量化）。这是个务实的取舍——AutoVLA 选择了「稳」。

> **关键认知**：规则奖励 + 拒绝采样，是「不训奖励模型」的强化学习。它绕开了 RLHF 里最麻烦、最容易出问题的「奖励模型」环节。代价是奖励信号比较粗，但对自动驾驶这种「安全底线明确、可量化」的任务，规则奖励反而比学出来的奖励模型更可靠——毕竟你不希望一个学歪了的奖励模型把「撞车」判成高分。

## 五、推理：慢思考可以旁路

训练讲完，最后看推理。核心文件 `infer/run_infer.py`。这里最有意思的是快慢思考的**旁路开关**。

```python
@torch.no_grad()
def infer(model, image, codebook, fast_only=True):
    if fast_only:
        prompt = "Describe the scene briefly and output the action token."
    else:
        prompt = "Think step by step about the driving decision, then output the action token."

    input_ids = tokenize(prompt)
    output = model.generate(
        pixel_values=image,
        input_ids=input_ids,
        max_new_tokens=8 if fast_only else 128,
    )
    action_id = extract_action_token(output)
    return codebook[action_id]
```

说人话：

- `fast_only=True`（快思考模式）：提示语只让模型「简单描述一下就直接给动作」，`max_new_tokens` 设得很小（比如 8）——因为不需要生成长篇推理，模型几乎是「看一眼就出手」。速度快，适合实时驾驶。
- `fast_only=False`（完整快慢思考）：提示语让模型「一步步想清楚驾驶决策，再给动作」，`max_new_tokens` 设得大（比如 128）——模型会先吐出一大段自然语言推理，再给动作。慢，但可解释、遇到复杂场景更稳。
- `extract_action_token(output)`：从生成结果里把动作 token 揪出来。
- `codebook[action_id]`：查表还原成连续动作，交给车执行。

**这就是快慢思考的实用价值所在**：同一个模型、同一套权重，靠一个开关就能在「快而糙」和「慢而稳」之间切换。直道用快思考省算力，复杂路口切慢思考求稳妥。甚至可以做成自适应——先快思考,遇到模型不确定的场景再触发慢思考。

> **一句话澄清**：为什么慢思考能被旁路却不影响正确性？因为动作 token 是从**隐藏状态**投影出来的，而隐藏状态本身就浓缩了 Qwen2.5-VL 对图像的理解。慢思考那段自然语言，主要是在**训练时**帮模型把「理解」和「决策」的因果链学扎实。一旦学好了，推理时那段推理更多是「给人看的解释」，模型内部其实已经「想明白」了。所以旁路它不影响出手，只是少了一份可解释的说明书。

## 六、把整条链路串起来

到这里所有零件都讲完了，我们把一次完整的「训练 → 推理」串成一条线，帮你在脑子里连成整体。

**准备阶段（一次性）**：

- 跑 `action_token_cluster.sh` → `build_codebook.py` 用 KMeans 把海量连续动作聚成 K 个簇 → 得到 codebook。从此每个连续动作都能映射成一个 token id。

**Stage-1（SFT 模仿）**：

- `run_sft.sh` → `sft.py` 加载 Qwen2.5-VL-3B，挂上 `action_head` → 用「图 + 慢思考文字 + 动作 token」的样本训练 → 文字损失 + 动作损失联合回传 → 模型学会「看图、说理由、输出动作 token」。

**Stage-2（RFT / GRPO 精修）**：

- `run_rft.sh` → `grpo_trainer.py` 加载 SFT 模型 → 对每个场景 rollout 出 M 条轨迹 → `rule_reward` 规则打分 → 只留组内最高分的（拒绝采样）→ 拿高分样本再微调 → 模型在关键场景更果断、更安全。

**推理（部署）**：

- `run_infer.py` → 图像 + 提示语进 Qwen2.5-VL → （可选慢思考）→ 动作头吐出动作 token id → `codebook[id]` 查表还原成连续动作 → 交给车执行。`fast_only` 开关控制快慢。

一句话记住这个闭环：**KMeans 造词典，SFT 教说话，RFT 教说好话，推理时看图说动作词、查词典开车，慢思考随时可旁路提速。**

再用一张对照表把每个环节的「入口文件 / 干什么 / 训练推理差异」钉死，方便你回头查：

| 环节 | 入口文件 | 干什么 | 训练/推理差异 |
|:---|:---|:---|:---|
| 动作离散化 | `build_codebook.py` | KMeans 聚类生成 codebook | 一次性预处理，两阶段都读它 |
| 模型结构 | `modeling_auto_vla.py` | Qwen2.5-VL 挂 action_head | 结构不变，权重两阶段更新 |
| 动作头 | `action_head.py` | 投影出 logits / 解码回动作 | 训练用投影算损失，推理用解码查表 |
| 模仿学习 | `sft.py` | 文字 + 动作联合监督 | 仅训练 |
| 强化精修 | `grpo_trainer.py` | rollout + 规则打分 + 拒绝采样 | 仅训练 |
| 推理入口 | `run_infer.py` | 看图吐 token、查表开车 | 仅推理，含慢思考旁路开关 |

## 和本系列其他文章的关系

AutoVLA 不是孤立的，它在这条「端到端自动驾驶」的演化线上有明确的坐标：

- **对比 UniAD / VAD（端到端回归）**：那两个是「网络直接回归出轨迹坐标」的路线，没有大模型、没有 token 化。AutoVLA 则把动作塞进了 LLM 的 token 空间，于是能白嫖整套语言模型的训练技术。这是「回归范式」到「生成范式」的跨越。

- **对比 DiffusionDrive（扩散生成轨迹）**：DiffusionDrive 用扩散模型在**连续空间**里生成轨迹（anchor + 截断扩散），动作始终是连续的。AutoVLA 反其道而行，先把动作**离散**成 token 再让 LLM 生成。一个走连续生成，一个走离散预测——这是 VLA 领域两条主要的动作表示路线之争。

- **对比 DriveVLA-W0（VLA + 世界模型）**：两者都是「VLA + 动作 token + 强化微调」。但侧重不同：DriveVLA-W0 强调用**世界模型（预测未来帧）**提供稠密监督，逼大模型「想清楚世界会怎么变」；AutoVLA 强调**快慢思考 + 拒绝采样对齐**，走的是「用规则奖励精修决策」的路子。微调上，AutoVLA 用 RFT/GRPO 拒绝采样（无价值网络、组内比较），DriveVLA-W0 用 Flow Matching 风格的连续动作生成。两篇对照着读，能把「VLA 怎么训、动作怎么表示」这两个核心问题看得很透。

- **通向 pi0**：AutoVLA 告诉你「动作可以离散成 token 让 LLM 说」，而 pi0 会告诉你另一种可能——**动作不一定非要离散**，用 Flow Matching 直接对连续动作做生成也很好用，而且更精细。这两篇合起来，正好是「离散 token 派」和「连续流派」的正面对话。

## 个人思考

**1. AutoVLA 最妙的一步是「把控制问题伪装成语言问题」。** 一旦动作变成了 token，SFT、拒绝采样、GRPO 这些从 NLP 搬过来的技术几乎不用改就能用。这是一种极其聪明的「问题重定义」——不是发明新方法，而是把一个难问题翻译成一个已经有成熟工具箱的问题。对做工程的人来说，这种「换个空间就白嫖整套生态」的思路，比任何具体技巧都值钱。

**2. RFT 退化成「拒绝采样 + 再监督」是务实到有点反潮流的选择。** 现在大家都在卷 PPO、GRPO 的各种花式实现，AutoVLA 却把强化学习砍到只剩「多生成、挑好的、再学」。它赌的是：对自动驾驶这种安全敏感、奖励可规则化的任务，**稳定压倒一切**——一个能稳定收敛的粗糙方法，胜过一个可能训飞的精妙方法。我很认同这个判断。车端不是刷榜，训飞一次的代价可能是「学出一个危险的策略」，这是不能接受的。

**3. 规则奖励是把双刃剑。** 它的可靠性来自「透明、不可被钻空子」，但它的天花板也来自这里——你只能奖励你写得出规则的东西。「像老司机一样有预判」「在拥堵中优雅博弈」这些高阶驾驶素养，很难用规则量化。所以 AutoVLA 的规则奖励能保证「安全、合规、不难受」，但要迈向「优秀」，迟早还是得引入更丰富的奖励信号（可能是学出来的、可能是人类偏好的）。这是这条路线未来必须面对的瓶颈。

**4. 快慢思考的旁路设计，暴露了「可解释性」在部署时的尴尬地位。** 慢思考那段自然语言推理，训练时是宝（帮模型学因果），推理时却可以随手扔掉（为了提速）。这说明在当前范式下，「可解释」和「实时」某种程度上是对立的——你想要模型给你讲清楚为什么这么开，就得付出延迟代价。真正优雅的方案，应该是让「解释」不额外增加决策延迟（比如异步生成解释、或者只在需要时触发）。AutoVLA 的旁路开关是个实用的权宜之计，但不是终局。

**5. 离散 token 化的量化误差，是这条路线绕不开的原罪。** 无论 K 调多大，连续动作被压成有限个词，精度损失总是存在的。低速泊车这种需要精细控制的场景，离散化的粗糙感会更明显。这也是为什么 pi0 那样的连续流方案有它的道理。我的判断是：**离散派和连续派会长期共存**——离散派赢在能无缝复用 LLM 生态、训练简单；连续派赢在动作精度。最终可能是混合的——用离散 token 做粗决策、用连续解码器做精修。AutoVLA 把离散这条路走得很扎实，是理解整个 VLA 领域绕不开的一块拼图。
