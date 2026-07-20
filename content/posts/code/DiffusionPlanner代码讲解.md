---
title: "Diffusion Planner 代码讲解：轨迹级扩散 + 免训练能量引导"
date: 2026-07-20
description: "拆解 ZhengYinan-AIR/Diffusion-Planner：DiT + MLP-Mixer 去噪轨迹、DDPM 训练、classifier guidance 做碰撞/可达性无训练约束，看轨迹扩散规划如何做到 SOTA 且即插即用"
tags: ["扩散模型", "轨迹规划", "能量引导", "代码讲解"]
categories: ["代码讲解"]
summary: "「Diffusion Planner 把规划对象从『动作』升级为『整条轨迹』，用 DiT + MLP-Mixer 去噪；亮点是在 DDPM 采样时注入 classifier guidance（碰撞/可行驶区域/运动学能量），无需重新训练就能把轨迹压进可行域，本文顺着官方仓库从名词、架构、逐文件逐函数一直讲到多模态闭环评测，帮零基础读者把『轨迹级扩散 + 免训练能量引导』这条主线彻底打通」"
---

> 本文是「代码讲解」路线的第 8 篇，是 Diffusion Policy 在「规划」上的专门化（ICLR 2025 Oral，OpenDriveLab / ZhengYinan-AIR）。代码在 **ZhengYinan-AIR/Diffusion-Planner**。如果你前面读过本系列的 Diffusion Policy、DiffusionDrive、VADv2，这篇会非常顺；如果没读过也没关系，下面名词表会补齐。

## 写给零基础读者：读这篇之前先搞懂几个名词

很多同学一上来就被「轨迹级扩散」「DiT」「DPS」这些词劝退，其实它们每一个都能用大白话讲明白。我先把这些名词翻译成人话，后面看代码才不会卡壳。

- **轨迹级扩散（trajectory-level diffusion）**：传统做法（比如 Diffusion Policy）是对「每一步的动作」做扩散，模型一帧一帧往外吐动作，像边走边想。而 Diffusion Planner 直接对「未来 T 步的整条轨迹 (x, y, heading) 序列」一起扩散——相当于闭着眼睛先画出一整条路径，再慢慢把它擦清楚。规划即生成一整条路径，不是逐步动作。

- **DiT（Diffusion Transformer）**：去噪网络（denoising network）原本大家爱用 U-Net，Diffusion Planner 换成 Transformer 块来做去噪主干。它的好处是长序列建模能力更强，轨迹点之间谁跟谁相关，Transformer 比卷积看得更清楚。

- **MLP-Mixer**：一个只用多层感知机（全连接）做「混合」的结构。在 Diffusion Planner 里它负责把轨迹点之间的时序相关性「揉顺」——让相邻时刻的轨迹点不要突然跳变，起到运动学平滑的作用。你可以把它理解成「轻量级时序正则器」。

- **Classifier Guidance / DPS（Diffusion Posterior Sampling）**：标准扩散采样只靠模型学到的分布往外生成，可能生成出会撞车的轨迹。DPS 的思路是：采样每一步去噪之后，额外减去一个「能量项」对轨迹的梯度（grad of energy w.r.t. x），把样本沿着能量下降方向推离不可行区域。这种引导是**免训练**的——能量函数可以随便换，模型不用重训。

- **多模态规划（multimodal planning）**：同一个初始场景，可能有好几条都合理的轨迹（保守跟车 / 激进超车 / 左转 / 右转）。模型一次性采出多条候选，下游再用规则挑一条最好的，而不是只给一条「平均轨迹」。

- **nuPlan**：目前最主流的自动驾驶开环+闭环评测数据集之一，带真实驾驶日志和 simulator，可以拿来跑闭环（closed-loop）评测，看规划器在仿真里会不会撞、守不守交规。

- **运动学平滑（kinematic smoothing）**：真实车不能瞬移、不能急拐，轨迹的曲率、加速度得在物理上限以内。能量引导里有一项专门惩罚曲率/加速度超限。

- **可行驶区域（drivable area）**：地图里允许车开的地方，比如车道内、路口铺装路面。轨迹点跑到马路牙子外、绿化带里，就要被惩罚。

- **碰撞能量（collision energy）**：轨迹上每个点如果和别的 agent（车、行人）的未来预测位置太近，能量就高，引导就把轨迹推远。

> 一句话澄清：Diffusion Planner 不是「预测别车未来再规划」，它就是一条干净的规划器——你给它当前场景（ego 状态 + 邻居轨迹 + 地图），它吐一条 ego 的未来轨迹。碰撞能量用到的是别车「已知/预测的未来位置」作为约束，不是它自己再跑一个预测网络。

## 为什么要讲 Diffusion Planner 的代码

一句话结论：**因为它把「安全约束」从训练期彻底搬到了推理期**。这是端到端规划里非常香的一个设计——你换个城市、换个交规、换张高精地图，不用重训一个几十 G 的大模型，只要改几个能量函数就能把轨迹压进可行域。对搞落地的人来说，这比「重新训一个合规模型」实在太多。

本系列前面讲了 Diffusion Policy（动作级扩散）、DiffusionDrive（flow matching 端到端）、VADv2（多模态概率规划）。Diffusion Planner 是这几条线在「轨迹生成 + 免训练约束」上的集大成，ICLR 2025 Oral 的认可也说明社区看好这套思路。读懂它，你基本就掌握了「扩散 + 可微约束」这一派规划器的全部骨架。

## 架构总览

先把整条链路在脑子里过一遍，代码细节才不晕：

1. 输入：当前场景编码 `cond`，包含 ego 自车状态、周围 agent 的历史/当前轨迹、高精地图（车道、可行驶区域多边形）。
2. 初始化：从标准高斯噪声 `x ~ N(0, I)` 出发，这个 `x` 的形状就是 `(B, T, D)`，T 是未来步数，D 是 (x, y, heading) 之类。
3. 去噪主干：噪声轨迹 token 过 DiT 块（自带时间步条件 adaLN），再过一个 MLP-Mixer 做时序混合，最后 head 输出预测的噪声 `eps`。
4. 采样循环：从 t = T-1 反着走到 0，每步先用模型预测噪声做标准 DDPM 后验去噪，再依次调用 collision / drivable / kinematic 三个能量引导，算能量对 x 的梯度并减去，把轨迹推离不可行区。
5. 多模态：同一个场景重复采样 M 次得到 M 条候选轨迹，用规则打分 `rule_score` 挑最优那条丢给控制器。
6. 评测：在 nuPlan 闭环仿真里跑，看碰撞率、违章率、舒适度。

> 关键认知：训练阶段完全不用碰能量函数，能量只在推理采样时注入。所以训练就是最朴素的 DDPM，推理才「加装」约束。这正是它灵活的根源。

## 项目结构

先说边界，避免你找错仓库。

- 这是 **ZhengYinan-AIR/Diffusion-Planner**，属于 OpenDriveLab 体系，是 DiffusionDrive 作者在规划方向的延续工作。
- 和 DiffusionDrive 相比，最大的边界变化是：**规划对象从「动作/控制量」升级为「整条轨迹 (x, y, heading) 序列」**。也就是说它不做自回归逐步预测，而是一次性生成整段未来路径，再用能量引导修。
- 它专注「规划」这一步，不负责感知（检测/分割/地图在线构建），输入场景通常已经由上游模块或离线数据准备好。

仓库（简化）长这样，注意用 Markdown 列表表示目录树，不画框线：

- `Diffusion-Planner/`
  - `diffusion_planner/`
    - `model/`
      - `dit.py`：DiT 去噪主干，轨迹 token 过 DiT block + MLP-Mixer
      - `mixer.py`：MLP-Mixer 时序混合模块
      - `guidance/`
        - `collision.py`：碰撞能量
        - `drivable.py`：可行驶区域能量
        - `kinematic.py`：运动学平滑能量
    - `scheduler/`
      - `ddpm.py`：标准 DDPM 加噪/去噪 + 采样时注入 guidance 能量梯度
    - `env/`
      - `nuplan_wrapper.py`：闭环评测环境封装
    - `configs/`：各种训练/推理超参
  - `scripts/`
    - `train.py`：训练入口
    - `eval_closed_loop.py`：多模态候选 + 规则打分选最优 + 闭环评测

下面逐文件逐函数拆。

## 1. `model/dit.py`：DiT 去噪主干

这是整个网络最核心的一块。它的职责很简单：给一个「加噪后的轨迹」和「时间步 t」和「场景条件 cond」，输出「模型认为加在上面的是哪个噪声」。

轨迹是 `(T, D)` 的序列，思路是先切成 token（其实就是每个时间步一个 token，或者把相邻几步拼一下），然后过一堆 DiT block，最后 head 预测噪声。

```python
# model/dit.py (simplified)
import torch
import torch.nn as nn

class AdaLNNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, t_emb):
        # t_emb: (B, dim) timestep embedding
        # modulate LayerNorm by timestep
        return self.norm(x) * (1 + self.scale(t_emb).unsqueeze(1)) \
               + self.shift(t_emb).unsqueeze(1)

class MLPMixer(nn.Module):
    def __init__(self, dim, token_dim, chan_dim):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.chan_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(chan_dim, chan_dim),
            nn.GELU(),
            nn.Linear(chan_dim, chan_dim),
        )

    def forward(self, x):
        # x: (B, T, D); mix across tokens then across channels
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.chan_mix(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, dim, T, D):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.mixer = MLPMixer(dim, T, D)
        self.adaLN = AdaLNNorm(dim)

    def forward(self, x, t_emb, cond=None):
        h = self.adaLN(x, t_emb)
        if cond is not None:
            h = h + self.attn(h, cond, cond)[0]
        else:
            h = h + self.attn(h, h, h)[0]
        h = h + self.mixer(h)
        return h

class TrajectoryDiT(nn.Module):
    def __init__(self, T, D, dim, depth=6):
        super().__init__()
        self.input_proj = nn.Linear(D, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, T, dim))
        self.t_emb = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, T, dim) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, D)

    def forward(self, noisy_traj, t, cond=None):
        # noisy_traj: (B, T, D); t: (B,)
        t_emb = self.t_emb(t.float().unsqueeze(-1))
        h = self.input_proj(noisy_traj) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, t_emb, cond)
        return self.head(h)  # predict noise eps
```

几个口语化解释：

- `AdaLNNorm` 是 DiT 的标志性设计：时间步 `t` 不是简单拼到输入上，而是去「调制」LayerNorm 的 scale 和 shift。这样模型在每一步去噪时都知道「我现在是第几步」，该擦粗噪还是细噪。
- `MLPMixer` 里 `token_mix` 是在「时间步维度」上做全连接，相当于让第 3 秒的轨迹点去「看」第 1 秒、第 5 秒的点，把整条轨迹的时序相关性揉顺；`chan_mix` 是在特征维度上混合 (x, y, heading)。它比 self-attention 便宜，专门干「平滑」这一种活。
- `DiTBlock` 里的 attention 可以是 self-attention（轨迹点互看），也可以 cross-attention 到 `cond`（场景编码）。官方实现里场景条件大多是 cross-attn 或 FiLM 注入，伪代码里我两种都留了口子。
- `head` 直接输出和输入同形状的噪声预测，损失就是预测噪声和真加噪声的 MSE。

> 一句话澄清：DiT 的「预测目标」是噪声，不是轨迹本身。这是标准 DDPM 套路——网络学的是 $\epsilon_\theta(x_t, t)$，采样时再用它反推 $x_{t-1}$。

## 2. `model/mixer.py`：MLP-Mixer 时序混合

上面 dit.py 里其实已经内联了一个 MLPMixer，但这里单独成文件，说明它在仓库里是被当成一个独立模块来维护的。我们把它单独拎出来看，重点理解「token mix」和「channel mix」两件事。

```python
# model/mixer.py (simplified)
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        return self.net(x)

class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens, dim, token_hidden, chan_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = FeedForward(num_tokens, token_hidden)
        self.norm2 = nn.LayerNorm(dim)
        self.chan_mix = FeedForward(dim, chan_hidden)

    def forward(self, x):
        # x: (B, T, D)
        res = x
        x = self.norm1(x)
        x = x.transpose(1, 2)          # (B, D, T)
        x = self.token_mix(x)          # mix across time steps
        x = x.transpose(1, 2)          # (B, T, D)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.chan_mix(x)           # mix across features
        x = x + res
        return x
```

为什么轨迹规划特别需要 MLP-Mixer？因为汽车轨迹天然有强时序连续性：第 t 步的位置基本由 t-1 步决定，曲率不能突变。self-attention 能建模任意点对关系，但「揉顺时序」这种结构化先验，用 MLP-Mixer 这种轻量模块反而又快又稳。它在 DiT block 里作为 attention 之后的补充，专门收尾平滑。

> 关键认知：MLP-Mixer 不是用来「理解场景」的，它只负责把已经成形的轨迹 token 之间的时序/通道关系理顺。真正的场景理解靠的是前面的 attention 和场景编码。

## 3. `scheduler/ddpm.py`：标准 DDPM + 采样时注入 guidance

这是全篇最该精读的文件。前半部分是教科书级 DDPM（加噪、去噪、损失），后半部分是 Diffusion Planner 的独门秘籍——采样循环里塞能量引导。

先看训练侧的加噪和损失。标准 DDPM 的前向过程：

```python
# scheduler/ddpm.py (simplified, training side)
import torch
import torch.nn.functional as F

def q_sample(x0, noise, alpha_bar):
    # x0: clean trajectory (B, T, D)
    # alpha_bar: (T,) cumulative product of (1-beta)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

def training_loss(model, scene, traj, t, alpha_bar):
    noise = torch.randn_like(traj)
    noisy = q_sample(traj, noise, alpha_bar[t])
    cond = encode_scene(scene)
    pred_noise = model(noisy, t, cond=cond)
    loss = F.mse_loss(pred_noise, noise)   # predict the added noise
    return loss
```

注意 `loss` 就是预测噪声和真噪声的 MSE，目标是轨迹真值（专家驾驶轨迹）。这就是官方说的「训练标准 DDPM 损失」，没有任何花活。

再看推理侧采样。这里才是能量引导登场的地方：

```python
# scheduler/ddpm.py (simplified, sampling side)
def sample(model, scene, guides, T, guide_scale, alpha, sigma):
    x = torch.randn((B, T_steps, D))          # start from gaussian
    cond = encode_scene(scene)
    for t in reversed(range(T_steps)):
        t_batch = torch.full((B,), t)
        eps = model(x, t_batch, cond=cond)    # predicted noise
        # standard DDPM posterior step
        x0_pred = (x - sigma[t] * eps) / alpha[t]
        x = alpha[t] * x0_pred + sigma[t] * torch.randn_like(x)
        # ---- training-free guidance ----
        for guide in guides:                  # collision / drivable / kinematic
            E = guide.energy(x, scene)        # scalar energy per sample
            g = torch.autograd.grad(E, x, retain_graph=True)[0]  # dE/dx
            x = x - guide_scale * g           # push along energy descent
    return x
```

关键就在那个 `for guide in guides` 循环。每做完一次标准去噪，立刻把三个能量梯度分别减掉一点。公式上写就是：

$x = x - \text{scale} \cdot g$，其中 $g = \nabla_x E(x)$。

因为 `E` 是对不可行区域（碰撞、出界、不平滑）的惩罚，减掉它的梯度等于把 `x` 往能量更低（更可行）的地方挪。而且这一切发生在推理期，`model` 的权重一点没动——所以换地图换规则，只要换 `guides` 和 `scene` 里的地图信息即可，模型零重训。

> 一句话澄清：`torch.autograd.grad(E, x)` 要求 `E` 是 `x` 的可微函数，所以三个能量文件里写的都是「可微惩罚」，不是 if-else 硬约束。这正是 DPS 能即插即用的前提。

## 4. `guidance/collision.py`：碰撞能量

碰撞能量要做的事：拿 ego 当前这条轨迹 `x`（未来 T 步），和场景里其他 agent 的「未来位置」比距离，太近就给高能量。

```python
# guidance/collision.py (simplified)
import torch

class CollisionEnergy:
    def __init__(self, radius=1.5, margin=0.5):
        self.radius = radius
        self.margin = margin

    def energy(self, traj, scene):
        # traj: (B, T, D) with x,y at first two dims
        # scene['others_future']: (B, A, T, 2) predicted future positions
        others = scene['others_future'][..., :2]      # (B, A, T, 2)
        ego_xy = traj[..., :2].unsqueeze(1)           # (B, 1, T, 2)
        dist = torch.norm(ego_xy - others, dim=-1)    # (B, A, T)
        penetration = self.radius + self.margin - dist
        penetration = torch.clamp(penetration, min=0)
        E = penetration.pow(2).mean(dim=(1, 2))       # (B,)
        return E
```

思路非常直白：算 ego 轨迹上每个点和每个别的 agent 在每个时刻的距离，小于「车宽半径 + margin」就记一笔穿透量，平方求和当能量。平方是为了平滑、可微，梯度才好用。

> 关键认知：这里的 `others_future` 不是 Diffusion Planner 自己预测的，一般是上游给的（或评测仿真器给的真值/预测）。它只把这些位置当「固定障碍」来避，不递归预测别人怎么躲自己。

## 5. `guidance/drivable.py`：可行驶区域能量

可行驶区域能量：轨迹点如果落到地图里「不允许开」的区域（绿化带、马路牙子外、逆行车道），就惩罚。

```python
# guidance/drivable.py (simplified)
import torch

class DrivableEnergy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def energy(self, traj, scene):
        # traj: (B, T, D); scene['drivable_mask_fn']: differentiable in/out test
        xy = traj[..., :2]                       # (B, T, 2)
        # signed distance to nearest drivable boundary (negative if outside)
        sdf = scene['drivable_sdf'](xy)          # (B, T)
        outside = torch.clamp(-sdf, min=0)       # >0 means outside
        E = self.weight * outside.pow(2).mean(dim=1)   # (B,)
        return E
```

实现上常见两种：一种是用预计算的可行驶区域 SDF（有符号距离场），点在区域外时 SDF 为负，取 `-sdf` 做惩罚；另一种是对每个点做可微的「点在多边形内」判定。不管哪种，目标都是让 `x` 的梯度把轨迹点拉回车道内。

> 一句话澄清：可行驶区域能量是「软约束」，不是硬裁切。它不会瞬间把出界点弹回线内，而是每步轻轻拉一点，配合 DDPM 本身的去噪，最终轨迹既自然又在界内。

## 6. `guidance/kinematic.py`：运动学平滑能量

运动学能量管的是「车开得动不动」：相邻步之间的航向变化（曲率）太大、速度/加速度突变，就惩罚。这样生成的轨迹控制器才接得住。

```python
# guidance/kinematic.py (simplified)
import torch

class KinematicEnergy:
    def __init__(self, max_curv=0.2, max_acc=2.0):
        self.max_curv = max_curv
        self.max_acc = max_acc

    def energy(self, traj, scene):
        # traj: (B, T, D) with x,y,heading
        xy = traj[..., :2]
        heading = traj[..., 2]
        # curvature proxy: change of heading between steps
        d_head = torch.diff(heading, dim=1)
        curv = d_head.abs()
        curv_pen = torch.clamp(curv - self.max_curv, min=0).pow(2)
        # acceleration proxy: second difference of position
        acc = torch.diff(xy, dim=1)
        acc = torch.diff(acc, dim=1)
        acc_pen = torch.clamp(acc.norm(dim=-1) - self.max_acc, min=0).pow(2)
        E = curv_pen.mean(dim=1) + acc_pen.mean(dim=1)   # (B,)
        return E
```

这里用「航向差分」近似曲率、用「位置二阶差分」近似加速度，都是可微的。超限部分平方惩罚，梯度自然把轨迹往「更平滑」推。

> 关键认知：三个能量加起来才是完整引导。碰撞管「别撞别人」，可行驶管「别出界」，运动学管「车开得动」。三者正交，可以单独调权重，这也是免训练引导的最大卖点——可解释、可调参、可插拔。

## 7. `env/nuplan_wrapper.py`：闭环评测封装

模型在 nuPlan 仿真器里跑闭环，需要把「采样出的轨迹」转成「车能执行的控制」，再喂回仿真器看下一步。wrapper 干的就是这个桥接。

```python
# env/nuplan_wrapper.py (simplified)
class NuPlanWrapper:
    def __init__(self, simulator, planner):
        self.sim = simulator
        self.planner = planner

    def reset(self, scenario):
        self.state = self.sim.reset(scenario)

    def step(self, scene):
        # scene: current observation from simulator
        traj = self.planner.sample(scene, n_candidates=1)[0]
        # convert trajectory to control (speed / steering) via PID tracker
        ctrl = self.tracker.track(traj, self.state)
        self.state = self.sim.step(ctrl)
        return self.state

    def run_episode(self, scenario, steps):
        self.reset(scenario)
        for _ in range(steps):
            scene = self.sim.get_observation()
            self.step(scene)
            if self.sim.is_done():
                break
        return self.sim.metrics()
```

实际仓库里 wrapper 会更复杂（处理坐标变换、地图查询、观测编码），但骨架就是：拿观测 → 采样轨迹 → 轨迹跟踪成控制 → 仿真步进 → 收指标。闭环指标通常包括碰撞率、违章数、舒适度（加速度 RMS）。

> 一句话澄清：闭环和开环的区别在于，开环拿真值周围状态评估，闭环是规划器自己开、自己决定下一步。Diffusion Planner 的免训练引导在闭环里尤其有用——因为仿真里会出现训练时没见过的「危险临场状况」，能量引导能实时把轨迹掰回安全区。

## 8. `scripts/eval_closed_loop.py`：多模态候选 + 规则打分

这是把前面所有零件拼起来跑评测的入口。它最关键的一招是「多模态 + 规则选优」。

```python
# scripts/eval_closed_loop.py (simplified)
def rule_score(traj, scene):
    # combine guidance energies into a single pickable score
    e_col = collision_energy.energy(traj, scene)
    e_drv = drivable_energy.energy(traj, scene)
    e_kin = kinematic_energy.energy(traj, scene)
    progress = traj[..., :2].diff(dim=1).norm(dim=-1).sum()  # how far it goes
    score = progress - 5.0 * e_col - 3.0 * e_drv - 1.0 * e_kin
    return score

def evaluate(planner, sim_wrapper, scenario, M=8):
    best_traj = None
    best_score = -1e9
    for _ in range(M):
        traj = planner.sample(scene=sim_wrapper.observation(),
                              guides=all_guides, n_candidates=1)
        s = rule_score(traj, sim_wrapper.scene)
        if s > best_score:
            best_score = s
            best_traj = traj
    ctrl = tracker.track(best_traj, sim_wrapper.state)
    return sim_wrapper.sim.step(ctrl)
```

逻辑是：同一个场景采样 M 条候选（利用扩散天然的多模态），每条都算一个 `rule_score`——既看「走了多远」（鼓励前进），又扣掉三项能量（惩罚危险/出界/抖动），挑分最高的那条执行。

为什么不用模型自己给的概率选？因为轨迹级扩散采样不像 VADv2 那样每条带显式概率，它采出来的是去噪结果，没有现成置信度。所以用「能量反向构造的分数」来排序，反而更可控、更贴近安全需求。

> 关键认知：多模态采样的代价是推理要跑 M 次去噪，但因为轨迹才几十个点（T 很小），每次去噪远比图像扩散轻，所以实际 latency 完全能接受。这也是轨迹级扩散相比图像级扩散的工程优势。

## 9. 训练入口 `scripts/train.py` 串起来看

把 dit / ddpm / 数据加载串起来的训练循环大致是：

```python
# scripts/train.py (simplified)
for batch in dataloader:
    scene, traj_gt = batch['scene'], batch['traj']   # traj_gt: expert trajectory
    t = torch.randint(0, T_steps, (B,))
    loss = training_loss(model, scene, traj_gt, t, alpha_bar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

注意训练时**完全没有 energy / guidance 参与**。模型只学「给定噪声轨迹 + 时间步 + 场景，预测加上的噪声」。所有安全约束都是推理时后加的。这个解耦是 Diffusion Planner 设计上最聪明的地方——训练简单（标准 DDPM），部署灵活（换 guides 即可）。

## 为什么比回归式规划强

- 回归式（UniAD / VAD 一类）只出一条「平均轨迹」，遇到歧义场景（可左可右、可跟可超）会犹豫出一条危险的折中，比如卡在两条车道中间。
- 扩散直接建模多模态轨迹分布，再用能量引导把不可行项筛掉，安全率和舒适度都更高。
- 代价是推理要跑多步去噪 + 多候选采样，但轨迹点少（几十个），比图像扩散快几个数量级。

## 个人思考

Diffusion Planner 的「免训练能量引导」是我最想抄到 Flow-GRPO 里的设计——它把安全约束从训练期解放到推理期，意味着规则变了不用重训。结合 GRPO 的奖励信号，其实可以统一成「采样 + 可微约束」，这正是当前端到端规划很有前景的方向。另外 MLP-Mixer 在轨迹平滑上的轻量用法也值得借鉴：不一定啥都堆 attention，结构化先验用小模块收尾更稳更快。

## 和本系列其他文章的关系

- **方法论母体 → Diffusion Policy（上篇）**：Diffusion Planner 就是把 Diffusion Policy 的「动作级扩散」升级成「轨迹级扩散」，并把「动作」换成「(x, y, heading) 序列」。
- **同属扩散用于驾驶 → DiffusionDrive**：DiffusionDrive 走的是 flow matching + 生成式端到端（从感知出轨迹），Diffusion Planner 更纯粹地讲「规划 + 免训练约束」，两者在能量引导这块可以互相借。
- **离散动作对照 → VADv2**：VADv2 用多模态概率分布选轨迹，Diffusion Planner 用扩散采样 + 规则打分选轨迹。思路同源（都承认场景多解），但实现路径一个离散一个连续。

---

> 小结：读完整篇你应该建立的直觉是——Diffusion Planner = 标准 DDPM 去噪整条轨迹（DiT + MLP-Mixer 当 backbone） + 推理时免训练能量引导（碰撞/可行驶/运动学）把轨迹压进可行域 + 多模态采样配规则打分。训练朴素、部署灵活，是轨迹级扩散规划里非常工程友好的一篇。
