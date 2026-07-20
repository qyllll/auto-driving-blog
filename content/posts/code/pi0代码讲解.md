---
title: "pi0 代码讲解：通用 VLA 的「VLM 主干 + Flow Matching 动作专家」"
date: 2026-07-20
description: "拆解 Physical-Intelligence/openpi：PaliGemma VLM 当通用感知-语义主干，独立的 Flow Matching action expert 在 10 步 ODE 内生成连续动作，看通用机器人 VLA 如何端到端输出动作"
tags: ["VLA", "Flow Matching", "机器人", "代码讲解"]
categories: ["代码讲解"]
summary: "「pi0 用 Google 的 PaliGemma 视觉语言模型当『懂场景、懂语言』的通用感知-语义主干，再并联挂一个独立的 Flow Matching 动作专家，通过 cross-attention 从 VLM 取特征，在 10 步 ODE Euler 积分内把随机噪声动作流到真实连续动作；本文顺着 Physical-Intelligence 官方 openpi 仓库，逐文件逐函数把 pi0.py 总装两塔、paligemma.py 封装主干、action_expert.py 的 Flow Matching 去噪、trainer.py 的向量场 MSE 损失、runtime.py 的 10 步 ODE 生成、transforms 里的动作 tokenize 与冻结策略讲透，并与 AutoVLA / DriveVLA-W0 做对照，最后把本系列九篇串成一条端到端自动驾驶与通用机器人 VLA 的演进线」"
---

> 本文是「代码讲解」路线的第 9 篇（收官篇）。pi0（Physical Intelligence，2024）是**通用机器人 VLA** 的代表作，但它和前几篇的 AutoVLA / DriveVLA-W0 共享同一套底层思想：VLM 负责理解世界，动作头负责把理解变成可执行动作。代码在官方 **Physical-Intelligence/openpi**。

## 写给零基础读者：读这篇之前先搞懂几个名词

很多同学一看到「Flow Matching」「cross-attention」「两塔并联」就头大。别慌，这一段专门用大白话把后面要反复用到的名词翻译一遍。你可以先扫一眼，后面遇到忘了再回来翻。

- **PaliGemma**：Google 出的一个「轻量版视觉语言模型（VLM）」。它自己又由两小块拼成——一个叫 **SigLIP** 的视觉塔（看图的）和一个叫 **Gemma** 的语言塔（读文字的）。pi0 把它整个借过来当「通用感知-语义主干」，意思是：摄像头画面它看得懂，你说的话它也听得懂，但它本身**不负责伸手去干活**。

- **Action Expert（动作专家）**：pi0 里一个**完全独立的 Transformer 模块**，专门干一件事——把「VLM 看明白的特征 + 一团噪声动作」去噪成「真实的、能执行的动作」。它和 VLM 是**两套不共享的权重**，是并联的两座塔，不是 VLM 里塞个小头。

- **Flow Matching（流匹配）**：这是 pi0 动作生成的核心数学工具。它不像老一代 DDPM 那样「一步步加噪、再一步步去噪」。它直接去学一个**从噪声到真实数据的向量场（一个 ODE 方向场）**，然后沿着这个向量场走几步（积分）就到了真实数据。pi0 实际只用 10~50 步，比 DDPM 快一个数量级。你可以把它理解成「给一团乱麻直接指一条最短下山的路，走十步就到谷底」，而不是「蒙着眼睛一步步试探」。

- **cross-attention 注入**：动作专家自己没有视觉能力，它靠 **cross-attention（交叉注意力）** 这个机制，从 VLM 主干那里「借」语义特征。就像动作专家是个只懂动手的工人，VLM 是个懂全局的工程师，工人每干一步都回头问工程师「现在场景是啥样」，而不是把工程师也拉来一起搬砖。

- **ODE 积分 / Euler 步**：ODE 就是常微分方程，这里指「向量场定义的轨迹」。Euler 步是最朴素的积分法——知道当前点在哪、方向往哪，就朝方向走一小步。重复走 N 步就到了。pi0 推理时大概走 10 步。

- **多机器人通用**：同一套架构，换不同的动作投影头和 tokenizer，就能跑机械臂抓取、叠衣服、收拾桌子等不同机器人、不同任务。这是 pi0 被叫「通用」的原因。

- **tokenizer / transform（动作维度适配）**：不同机器人动作维度不一样（机械臂可能是 7 维，移动底盘可能是 2 维）。`transforms` 模块负责把原始观测和原始动作「打包成模型能吃的 token」，把维度对不齐的问题在入口处解决掉。

- **两塔结构（VLM 主干 + 动作专家并联）**：这是 pi0 的骨架。一座塔（VLM）永远在「理解」，另一座塔（动作专家）永远在「生成动作」，两者权重独立、并行工作，靠 cross-attention 串联。记住这四个字后面会反复出现。

> 一句话澄清：所谓「两塔」，不是指跑在两台机器上，而是指**模型里有两份互相独立、不共享权重的 Transformer 权重**。它们可以、也通常跑在同一张显卡上。

## 0. 为什么要讲 pi0 的代码

前面八篇我们把自动驾驶的端到端方法从 UniAD 讲到 DriveVLA-W0，已经把「感知-预测-规划」和「VLM 理解 + 动作生成」两条线都摸了一遍。但一直有个问题没回答：**动作到底该离散成 token 用 RL 对齐（像 AutoVLA），还是该连续地用生成模型直接吐出来（像 Diffusion Policy）？** pi0 给了一个非常干净、非常工程化、而且被真实机器人验证过的答案：连续 Flow Matching，不用 RL，动作专家独立成塔。

讲 pi0 还有个额外好处：它是**官方开源、代码可读性极高**的仓库（Physical-Intelligence/openpi），不像很多论文只给半截伪代码。顺着它的源码，你能看到一套「通用 VLA 该怎么落地」的工业级范本。

## 1. 一句话结论

> 一句话结论：pi0 = 冻结（或极低 lr）的 **PaliGemma VLM 主干** + 一个独立训练的 **Flow Matching 动作专家**；前者负责「看懂场景和语言」，后者通过 cross-attention 借前者的特征，沿 10 步 ODE 把噪声动作流成真实连续动作；训练时只训动作专家 + 轻量适配层，VLM 主干基本不动。

记住这句话，后面所有代码都是在为它做注脚。

## 2. 架构总览

先用文字把 pi0 的数据流捋一遍，不引用任何图片，纯靠描述你也能在脑子里画出来：

1. **输入**：若干张摄像头图像（多视角）+ 一段自然语言指令（比如「把红色方块放到蓝色碗里」）+ 当前机器人本体状态（关节角、夹爪开合等）。
2. **VLM 主干（PaliGemma）**：图像过 SigLIP 视觉塔，语言过 Gemma 语言塔，两者在 Transformer 里融合，吐出一串**语义特征**（shape 大概是 `(B, L, H)`，B 是 batch，L 是 token 数，H 是隐藏维度）。这一步**不参与动作梯度的反向传播**（冻结或极低 lr）。
3. **动作专家（Action Expert）**：把「当前噪声动作序列 + 时间步 t + VLM 特征」一起喂进去。它在内部先用自注意力（self-attention）让动作 token 之间互相看，再用 cross-attention 去 VLM 特征里「取」和当前动作相关的语义。最后用一个线性头输出**向量场 v**（指向真实动作的方向）。
4. **训练目标（Flow Matching）**：构造一条从噪声到真实动作的线性插值路径 `x_t = (1-t)*noise + t*action`，让网络预测的 `v_pred` 去逼近真值向量场 `v_target = action - noise`，用 MSE 回归。
5. **推理（ODE 积分）**：从纯噪声出发，沿网络给出的向量场走约 10 步 Euler，就得到真实动作序列，直接发给机器人执行。

下面用 Markdown 列表把模块边界再划清楚：

- **理解侧（冻结/低 lr）**：PaliGemma（SigLIP 视觉塔 + Gemma 语言塔）。
- **生成侧（主训练对象）**：Action Expert（自注意力 + 对 VLM 的 cross-attention + 线性输出头）。
- **训练侧**：Flow Matching 损失（向量场 MSE）。
- **推理侧**：10 步 ODE Euler 积分。
- **适配侧**：transforms（观测/动作 tokenize，动作维度对齐）。

> 关键认知：pi0 的「通用」不是靠一个巨大模型硬吞所有任务，而是靠「**主干通用、动作头分身**」——VLM 主干一招鲜吃遍所有机器人，动作专家每家机器人各训各的。这点和自动驾驶里「一套感知 backbone 复用、规划头分场景」的思路异曲同工。

## 3. 项目结构

我们要讲的代码来自官方仓库 **Physical-Intelligence/openpi**。先说边界：这个仓库是 pi0 的**官方实现**，不是第三方复刻，所以命名和论文一一对应。核心边界就一句话——

> pi0 = PaliGemma VLM 主干 + 独立 Flow Matching 动作专家，两塔并联、不共享权重。

用 Markdown 嵌套列表把目录树画出来（注意这里**不用任何框线字符**，纯靠缩进）：

- `openpi/`
  - `src/openpi/`
    - `models/`
      - `pi0.py`：pi0 总装，把 VLM 和 Action Expert 拼到一起
      - `paligemma.py`：PaliGemma 主干封装（加载权重、前向、取特征）
      - `action_expert.py`：Flow Matching 动作专家（自注意 + 对 VLM 交叉注意，输出向量场 v）
    - `training/`
      - `config.py`：训练超参配置
      - `trainer.py`：训练循环 + Flow Matching 损失
    - `inference/`
      - `runtime.py`：推理（10 步 ODE Euler 积分生成动作）
    - `transforms/`
      - 观测 tokenize（图像/语言 → VLM 输入）
      - 动作 tokenize（机器人状态 → `(Ta, action_dim)`）
  - `examples/`
    - `aloha/`：具体机器人（如 ALOHA 机械臂）适配示例
  - `scripts/`
    - `train.py`：启动训练
    - `infer.py`：启动推理

后面我们就按这个顺序，逐文件逐函数拆。

## 4. pi0.py：总装两塔

`pi0.py` 是整个模型的「总装车间」。它不实现任何花哨的数学，只负责把 VLM 主干和动作专家**拼起来、对齐维度、定义前向接口**。

先看它怎么初始化。简化伪代码如下（块内零中文，中文说明在块外）：

```python
# src/openpi/models/pi0.py (simplified)
import torch
import torch.nn as nn
from .paligemma import PaliGemma
from .action_expert import ActionExpert


class PI0(nn.Module):
    def __init__(self, paligemma: PaliGemma, action_dim: int, action_horizon: int, vlm_hidden: int):
        super().__init__()
        self.vlm = paligemma
        self.action_expert = ActionExpert(
            action_dim=action_dim,
            vlm_hidden=vlm_hidden,
            horizon=action_horizon,
        )
        # lightweight adapter that projects vlm features into expert space
        self.vlm_adapter = nn.Linear(vlm_hidden, vlm_hidden)

    def forward(self, images, lang, noisy_actions, t):
        with torch.no_grad():
            vlm_feat = self.vlm(images, lang)          # (B, L, H) semantic features
        vlm_feat = self.vlm_adapter(vlm_feat)
        act_out = self.action_expert(noisy_actions, vlm_feat, t)
        return act_out                                  # predicted vector field v(x, t)
```

中文说明：

- `self.vlm` 就是 PaliGemma 主干。注意 `forward` 里用 `torch.no_grad()` 把它**包起来**——这意味着训练时 VLM 主干不参与反向传播，梯度根本不会流进 SigLIP/Gemma。这就是「冻结主干」的硬实现。
- `self.action_expert` 是真正要训的塔。它接收三个东西：噪声动作、VLM 特征、时间步 t。
- `self.vlm_adapter` 是一个**轻量适配层**（一个线性层），把 VLM 的隐藏维度投到动作专家方便用的空间。训练时这个层是**可训**的，属于「冻结主干 + 训轻量适配层」策略的一部分。

> 一句话澄清：`torch.no_grad()` 包住 VLM 是最省显存的冻结写法——连中间激活都不用存，反向时直接跳过。如果你看到有人用 `requires_grad=False` 也行，但 openpi 这种写法更干净。

`PI0` 类对外只暴露一个 `forward(images, lang, noisy_actions, t)`，返回向量场。它本身不关心「这是训练还是推理」，训练和推理的区别全在**调用方**怎么构造 `noisy_actions` 和 `t`：训练时 `noisy_actions` 是插值出来的，推理时 `noisy_actions` 是纯噪声。这个设计很妙——模型本体无状态，干净。

## 5. paligemma.py：VLM 主干封装

`paligemma.py` 的职责是把 Google 的 PaliGemma 原模型**封一层皮**，让它能吐出中间语义特征，而不是只吐文本。我们看简化版：

```python
# src/openpi/models/paligemma.py (simplified)
from transformers import PaliGemmaForConditionalGeneration


class PaliGemma(nn.Module):
    def __init__(self, pretrained_id: str):
        super().__init__()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained_id)
        # freeze all parameters of the backbone
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images, lang):
        outputs = self.model(pixel_values=images, input_ids=lang, output_hidden_states=True)
        # take the last hidden state of the text+image tokens as semantic features
        feat = outputs.hidden_states[-1]
        return feat
```

中文说明：

- 它直接复用 HuggingFace 的 `PaliGemmaForConditionalGeneration`，省去自己写 SigLIP + Gemma 融合的麻烦。
- 初始化时把所有参数 `requires_grad = False`，**彻底冻结**。这和 pi0.py 里的 `no_grad` 是双保险：即使哪天有人忘了包 `no_grad`，梯度也流不进主干。
- `forward` 里开 `output_hidden_states=True`，把最后一层隐藏状态当语义特征返回。这个 `(B, L, H)` 张量就是动作专家要「借」的东西。

> 关键认知：pi0 没有重新训一个 VLM，而是**白嫖**了 Google 在海量图文数据上预训练好的 PaliGemma。这解释了为什么 pi0 能用相对少的机器人数据训出通用策略——语义理解能力是「借」来的，不是从零学的。

这里顺带提一句 PaliGemma 内部的两塔：SigLIP 视觉塔把图像打成视觉 token，Gemma 语言塔把文字打成文本 token，两者拼成一条序列进 Transformer。pi0 只取融合后的隐藏态，不关心内部怎么融的。

## 6. action_expert.py：Flow Matching 动作专家

这是 pi0 的心脏。`action_expert.py` 定义一个 Transformer，输入是「噪声动作序列 + 时间步 + VLM 特征」，输出是**指向真实动作的向量场 v**。

先看结构：

```python
# src/openpi/models/action_expert.py (simplified)
import torch
import torch.nn as nn


class ExpertBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)   # self-attn over actions
        self.ca = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)   # cross-attn to vlm
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, vlm_feat, t_emb):
        x = self.norm1(x + self.sa(x, x, x)[0])
        x = self.norm2(x + self.ca(x, vlm_feat, vlm_feat)[0])   # query from actions, kv from vlm
        x = self.norm3(x + self.ff(x))
        return x


class ActionExpert(nn.Module):
    def __init__(self, action_dim: int, vlm_hidden: int, horizon: int, n_layer: int = 8, dim: int = 1024):
        super().__init__()
        self.tok_emb = nn.Linear(action_dim + vlm_hidden, dim)
        self.time_emb = nn.Sequential(nn.Linear(1, dim), nn.SiLU())
        self.blocks = nn.ModuleList([ExpertBlock(dim) for _ in range(n_layer)])
        self.head = nn.Linear(dim, action_dim)

    def forward(self, noisy_actions, vlm_feat, t):
        # concat noise action tokens with vlm tokens along feature dim
        x = torch.cat([noisy_actions, vlm_feat[:, : noisy_actions.size(1), :]], dim=-1)
        x = self.tok_emb(x)
        x = x + self.time_emb(t.unsqueeze(-1))
        for blk in self.blocks:
            x = blk(x, vlm_feat, t)
        return self.head(x)                                        # vector field v
```

中文说明，逐点讲：

- **`ExpertBlock`** 是动作专家的基本层，里面三件套：
  - `sa`：自注意力，让「动作序列里不同时间步的动作」互相看（比如第 3 步的动作会参考第 1、2 步）。
  - `ca`：交叉注意力，**query 来自动作，key/value 来自 VLM 特征**。这就是前面说的「动作专家回头问工程师」。动作 token 通过它从 VLM 那里取和自己相关的语义（比如「手要伸向红块」）。
  - `ff`：前馈网络，常规操作。
  - 三个 `norm` + 残差，标准的 Pre-LN Transformer 风格。
- **`ActionExpert`** 把噪声动作和 VLM 特征在特征维拼起来（`torch.cat`），过一个线性嵌入 `tok_emb` 升到模型维度 `dim`。
- **时间步嵌入** `time_emb`：Flow Matching 是连续时间，t 是个标量，过一个小 MLP 变成和隐藏维同形的向量，加回去。这告诉网络「现在在第几步去噪」。
- 最后 `head` 是个线性层，把 `dim` 维压回 `action_dim`，输出**向量场 v**。注意输出维度和输入动作维度一致——因为向量场要描述「动作每维该往哪走」。

> 一句话澄清：动作专家输出的不是「动作本身」，而是「动作该往哪个方向改」的向量场。真正的动作是推理时沿这个场积分走出来的。这点务必分清，否则后面看损失函数会懵。

为什么用 cross-attention 而不是把 VLM 特征直接 concat 进每层？因为 concat 会让动作专家被迫处理变长的 VLM token 序列，且语义和动作混在一起难训。cross-attention 让动作专家**按需取用**语义，更稳、更通用。

## 7. training/trainer.py：Flow Matching 损失

训练侧的核心在 `trainer.py`。它要回答一个问题：怎么让动作专家学会「输出正确的向量场」。答案就是 Flow Matching 的标准损失。

先上公式（KaTeX，公式内全英文，不用 \tag，用 aligned）：

$$
\begin{aligned}
x_t &= (1 - t) \cdot \text{noise} + t \cdot \text{action} \\
v_{\text{target}} &= \text{action} - \text{noise}
\end{aligned}
$$

意思是：在时间 t，把噪声和真实动作做线性插值得到 `x_t`；而这条直线路径上的真值向量场方向，正好就是 `action - noise`（从噪声指向动作）。网络只要学会预测这个方向就行。

看简化训练代码：

```python
# src/openpi/training/trainer.py (simplified)
import torch
import torch.nn.functional as F


def fm_loss(model, images, lang, action):
    noise = torch.randn_like(action)
    t = torch.rand((action.size(0),), device=action.device)        # uniform in [0, 1]
    t_expand = t.view(-1, 1, 1)
    x_t = (1 - t_expand) * noise + t_expand * action               # linear path
    v_target = action - noise                                      # ground-truth field
    v_pred = model(images, lang, x_t, t)                           # predicted field
    return F.mse_loss(v_pred, v_target)


def train_one_step(model, opt, batch):
    images, lang, action = batch
    loss = fm_loss(model, images, lang, action)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()
```

中文说明：

- `noise` 从标准正态采样，和真实动作同形。
- `t` 在 `[0,1]` 均匀采样，每个样本一个时间。
- `x_t` 就是上面公式里的线性插值；`v_target = action - noise` 是真值向量场。
- `v_pred` 是动作专家沿当前 `x_t` 和时间 `t` 预测的向量场。
- 损失就是两者 **MSE**。就这么简单，没有 KL、没有对抗、没有 RL——这也是 pi0 比 AutoVLA 干净的地方。

> 关键认知：Flow Matching 的损失本质是「回归一个方向向量」。它比 DDPM 的噪声预测损失更直接——DDPM 要预测加进去的噪声，Flow Matching 直接预测「朝数据的速度」。数学上等价但数值更稳、步数更少。

注意 `train_one_step` 里优化器 `opt` 通常**只把动作专家和适配层放进去了**，VLM 主干参数根本没注册进优化器（因为前面 `requires_grad=False`）。这就是「动作头只训动作专家 + 轻量适配层，VLM 主干冻结」的工程落地。

## 8. inference/runtime.py：10 步 ODE Euler 积分

推理侧在 `runtime.py`。训练完，动作专家会输出向量场，推理就是**从纯噪声出发，沿向量场走 N 步**。

公式（Euler 积分一步）：

$$
x_{i+1} = x_i + \frac{1}{N} \cdot v(x_i, t_i)
$$

看代码：

```python
# src/openpi/inference/runtime.py (simplified)
import torch


class Runtime:
    def __init__(self, model, n_step: int = 10):
        self.model = model
        self.n_step = n_step

    @torch.no_grad()
    def predict(self, images, lang, action_dim: int, horizon: int):
        vlm_feat = self.model.vlm(images, lang)
        x = torch.randn((images.size(0), horizon, action_dim), device=images.device)
        for i in range(self.n_step):
            t = torch.full((images.size(0),), i / self.n_step, device=images.device)
            v = self.model.action_expert(x, vlm_feat, t)
            x = x + (1.0 / self.n_step) * v          # one Euler step
        return x                                      # real action sequence
```

中文说明：

- 推理同样用 `no_grad`，因为不训了。
- `x` 从纯噪声起手，shape 是 `(B, horizon, action_dim)`——horizon 是动作序列长度（比如未来 16 步动作）。
- 循环 `n_step=10` 次，每步算当前向量场 `v`，朝它走 `1/n_step` 这么长一步。10 步走完，`x` 就从噪声「流」到了真实动作分布。
- 返回的就是**可以直接发给机器人执行的动作序列**。

> 一句话澄清：推理时 t 是从 0 走到接近 1（i/n_step），对应训练时路径从噪声(0)到动作(1)。t 必须和训练分布对齐，否则向量场会指错方向。

为什么只用 10 步？因为 Flow Matching 学的是**直线路径上的恒定方向场**，不像 DDPM 那种弯曲、需要很多步才能拐过来的轨迹。10 步在真实机器人上已经够平滑够准，这也是 pi0 能实时（~10Hz 以上）的关键。Diffusion Policy 类方法通常要 20~100 步，差一个数量级。

## 9. transforms/：观测与动作的 tokenize

不同机器人动作维度天差地别，pi0 用 `transforms` 模块在入口处把一切对齐。简化逻辑：

```python
# src/openpi/transforms/ (simplified)
def transform_obs(raw_images, lang_prompt, tokenizer):
    images = preprocess_images(raw_images)         # resize / normalize -> VLM input
    lang_ids = tokenizer(lang_prompt)["input_ids"]
    return images, lang_ids


def transform_action(robot_state, action_horizon, action_dim):
    # pad / clip robot state into fixed (horizon, action_dim) tensor
    action = pad_to_shape(robot_state, (action_horizon, action_dim))
    return action
```

中文说明：

- `transform_obs`：把原始多视角图像预处理成 PaliGemma 要的尺寸/归一化，把语言指令 tokenize 成 input_ids。产出 VLM 的输入。
- `transform_action`：把不同机器人的原始状态（维度可能乱七八糟）pad / clip 成统一的 `(horizon, action_dim)`。这就是「动作维度适配」的落点。

训练时这套 transform 保证「动作专家永远看到固定维度的动作 token」，于是同一份动作专家代码能服务所有机器人，只是 transform 配置不同。这正是「通用」的工程秘密——**变的东西挡在 transform 里，不变的东西留在模型里**。

> 关键认知：pi0 的通用性不是模型结构魔法，而是「模型吃固定形状、差异全在 transform」。这对我们做自动驾驶 VLA 也很有启发：把传感器差异、动作空间差异都封进 preprocessing，主干就能复用。

## 10. 和自动驾驶 VLA 家族的对照

把 pi0 放回本系列的坐标系里，和前面几篇对照一下，你会看得更清楚：

- **AutoVLA**：动作做成**离散 token**，用 **GRPO（强化学习）** 对齐到奖励。pi0 用**连续 Flow Matching**，完全不用 RL。一个靠「试错拿奖励」，一个靠「直接回归方向场」。
- **DriveVLA-W0**：强调**世界模型**（预测未来帧 / 未来场景），动作生成建立在「先想清楚未来」之上。pi0 **不做显式世界模型**，只做「观测 → 动作」的直推，靠 VLM 主干里隐含的世界知识。
- **共同点**：都是「VLM 理解 + 独立动作头生成」的两段式。pi0 证明了这条路在**机器人**上通，AutoVLA / DriveVLA-W0 证明了在**驾驶**上通。底层思想是同一个。

> 一句话澄清：pi0 没有世界模型不代表它「不懂未来」，而是它把对未来/物理的常识**压缩进了冻结的 VLM 主干**里，不显式建模。这是工程取舍，不是能力缺失。

## 11. 个人思考

九篇写下来，我对端到端自动驾驶与通用机器人 VLA 的方法论收敛出两条互补路线：

- **生成式（扩散 / Flow Matching）** 解决「多模态 + 安全分布」：同一场景可能有多种合理动作，生成模型天然能覆盖这个分布，不会硬选一个。
- **语言化（VLA）** 解决「可解释 + 常识推理 + 易对齐」：把语言接进来，决策能被人读懂、被人用自然语言约束。

pi0 站在通用机器人视角给了一个很关键的示范：**动作完全可以是连续 Flow Matching 生成，不一定非要像 AutoVLA 那样离散成 token + RL**。两者各有取舍——离散 token 好接 RL、好对齐人类偏好，但丢掉了动作的连续平滑性；连续 Flow Matching 平滑、实时、简单，但难直接接奖励信号。真正落地的系统，大概率是「VLM 语义主干 + 连续/离散动作专家 + 强化/规则对齐」的混合体。这也是我后续在 Flow-GRPO 上想继续推进的方向：**把 Flow Matching 的连续生成和 GRPO 的奖励对齐揉到一起**。

## 12. 收官：本系列九篇串起来

用一条 Markdown 列表把九篇串成演进线，从「端到端回归」一路走到「通用 VLA」：

- 端到端回归时代
  - UniAD：全栈 query 一体化（感知-预测-规划共享 query）
  - VAD：轻量向量化表征，VADv2 走向多模态概率规划
- 生成式扩散时代
  - Diffusion Policy：把扩散引入通用动作生成（机器人先跑通）
  - DiffusionDrive：驾驶端到端扩散，解决多模态轨迹
  - Diffusion Planner：轨迹扩散 + 能量函数引导安全约束
- VLA 路线（语言接进来）
  - AutoVLA：离散动作 token + GRPO 强化对齐
  - DriveVLA-W0：VLA + 显式世界模型预测未来
- 通用 VLA 收官
  - pi0：VLM 主干（PaliGemma） + Flow Matching 动作专家，两塔并联、10 步 ODE 实时生成连续动作

这条线的内核没变过：**从「让模型直接回归轨迹」到「让模型生成轨迹分布」再到「让模型用语言理解世界再生成动作」**。pi0 是这条线在「通用机器人」上的当下答案，而 Flow-GRPO 想回答的是「生成 + 奖励」怎么在驾驶上更进一步。

---

> 思考题：如果你要设计一个「既能用语言解释决策、又能实时输出连续平滑轨迹、还能用规则/奖励在线约束」的驾驶模型，会从这九篇里各借哪一块？欢迎在评论区交流。
