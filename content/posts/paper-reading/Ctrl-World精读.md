---
title: "精读 Ctrl-World：把 VLA 放进「可控世界模型」里想象着测、想着改——ICLR 2026 + 官方仓库 4 条跑法与演示全收录"
date: 2026-10-29
draft: false
categories: ["论文精读", "具身智能"]
tags: ["🌍 世界模型", "🤖 VLA", "🔍 评测", "🛠️ 官方开源", "📖 论文精读"]
summary: "斯坦福+清华《Ctrl-World: A Controllable Generative World Model for Robot Manipulation》（arXiv 2510.10125，ICLR 2026，Chelsea Finn 组，Yanjiang Guo / Lucy Xiaoyang Shi 共同一作）精读，含官方 GitHub 全解：核心命题——VLA 的评测与改进太贵（真机 rollout + 专家标注），改到「想象空间」里做 policy-in-the-loop。三个支撑件：①联合多视角预测（含腕部相机，治好单视角幻觉与「手腕视野剧变」不一致）②帧级动作条件（厘米级动作控制，1 秒动作块逐帧注入）③位姿条件记忆检索（20 秒+ 长程连贯）。从 SVD 视频扩散模型初始化，DROID 95k 轨迹/564 场景训练。两大用途实测：A) **想象中排名=真机排名**——π0/π0-fast/π0.5 在 7 任务上真机 vs 世界模型指令跟随相关 y=0.87x−0.04、成功率 y=0.81x−0.11；B) **合成成功轨迹做 SFT，策略成功率平均 +44.7%**（Spatial 0.29→0.88、Shape 0.44→0.93、平均 0.39→0.83）。官方仓库 Robert-gyj/Ctrl-World（575★，MIT，含 VLAW 后训练管线）四条跑法逐命令讲解：DROID 回放、键盘交互、π0.5 策略进环（零样本到新场景）、论文评测协议；训练/后训脚本与 ckpt 清单全解。9 张论文插图逐图讲解 + GitHub 演示媒体（README 横幅、2 个真机 vs 世界模型长程 GIF、2 组三视角合成轨迹 mp4）全部贴进博客。"
weight: 87
---

![Ctrl-World：为通用策略的 policy-in-the-loop 回放而生的多视角可控世界模型（论文 Figure 1）](/images/ctrlworld/fig1.png)

**配图说明**：本文**收录 9 张论文插图**（`/images/ctrlworld/fig1–9`）**+ 官方 GitHub 演示媒体 9 件**（横幅 JPG、2 个长程对比 GIF、2 组三视角合成轨迹 MP4），论文图逐张讲解，GitHub 媒体在第 4 章内嵌播放。论文 *Ctrl-World: A Controllable Generative World Model for Robot Manipulation*（arXiv 2510.10125，**ICLR 2026**），斯坦福 + 清华（Yanjiang Guo、Lucy Xiaoyang Shi 共同一作，Jianyu Chen，Chelsea Finn）；官网 [ctrl-world.github.io](https://ctrl-world.github.io/)；**官方实现** [github.com/Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)（575★，MIT，同时包含后续 VLAW 论文的世界模型后训管线）。

---

## 📄 论文信息

- **标题**：*Ctrl-World: A Controllable Generative World Model for Robot Manipulation*
- **机构**：Stanford University、Tsinghua University
- **时间**：2025-10 挂 arXiv（2510.10125），**ICLR 2026** 正式发表
- **关键词**：可控世界模型、policy-in-the-loop、多视角（含腕部）、帧级动作条件、位姿条件记忆、指令跟随评测与改进、DROID、SVD 初始化
- **一句话总结**：**给 VLA 配一个「能一起玩」的世界模型**——多视角联合预测 + 厘米级动作控制 + 20 秒长程记忆，让策略**在想象里被测**（排名≈真机）与**被改**（合成成功轨迹 SFT，成功率 +44.7%）。

---

## 🎯 一句话版本：核心数字

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 想象排名 ≈ 真机排名 | 指令跟随 **y=0.87x−0.04**；成功率 **y=0.81x−0.11** | π0/π0-fast/π0.5 × 7 任务，真机 vs 世界模型散点（Fig. 7） |
| 策略改进 | **平均 +44.7%**（0.39→0.83） | 合成成功轨迹 → SFT（Fig. 9） |
| 分任务涨幅 | Spatial 0.29→**0.88**；Shape 0.44→**0.93**；Towel-Dir 0.57→0.80；Novel-Obj 0.25→**0.73** | 四类指令理解短板 |
| 训练数据 | **DROID 95k 轨迹 / 564 场景** | 全开源可复现 |
| 长程生成 | 新场景/新相机位姿下 **>20 秒** 连贯多视角 | 位姿条件记忆检索 |
| 动作控制精度 | **厘米级**（帧级条件注入 1 秒动作块） | 同初始帧 + 不同动作序列 → 不同轨迹 |
| 底座 | **Stable Video Diffusion（SVD）** 初始化 | 适配为可控、时间一致的世界模型 |
| 官方仓库 | Robert-gyj/Ctrl-World，**575★**，MIT | 4 条推理跑法 + 训练/后训（VLAW） |

---

## 🗺️ 仿真 vs 真机：一眼分清

| 场景 | 类型 | 说明 |
| --- | --- | --- |
| 世界模型内 rollout（回放/键盘/π0.5 进环） | **生成式「想象」评测** | 不是物理仿真器——视频扩散模型自回归造未来 |
| 真机 rollout（π0.5 七任务 × 20 次） | **真机（对照组）** | 用来验证「想象排名=真机排名」（Fig. 6/7） |
| 合成成功轨迹 → SFT | **真机策略改进** | 改进幅度在真机上验收（Fig. 9，+44.7%） |
| 训练/后训（DROID 预训练 + VLAW 下游后训） | **离线数据** | 8×A100/H100 级别；后训见仓库（3）节 |

**一句话定位**：世界模型是「**生成式评测器**」——它既不是 MuJoCo 那种物理仿真，也不是纯视频玩具；有效性的证据就是 Fig. 7 的两条回归线。

---

## 🧭 读前 5 分钟：四个概念

1. **Policy-in-the-loop（策略进环）**：世界模型不止预测「给定动作的未来」，还要能**接住真策略的输出**再生成下一步——闭环里策略与模型互相为对手/学生。
2. **多视角联合预测（含腕部）**：现代 VLA 吃第三人称 + 腕部相机；腕部视野在一条轨迹里**剧烈变化**，单视角世界模型要么幻觉（物体无接触就「粘」进夹爪）要么干脆不兼容策略输入。
3. **帧级动作条件（frame-level action conditioning）**：不是「整段视频给个粗标签」，而是把 1 秒动作块**逐帧**钉进生成过程——同初始帧、换动作序列就换轨迹，且到**厘米级**。
4. **位姿条件记忆检索（pose-conditioned memory retrieval）**：长程自回归会漂；按**当前位姿**从记忆里检索相关历史片段补进上下文，20 秒+ 仍连贯。

---

## 🛠️ 方法：SVD 底座 + 三个可控性部件

- **初始化**（Fig. 2）：从预训练视频扩散模型 **SVD** 出发，适配成「可控、时间一致」的世界模型——保留其生成质量，注入条件通路；
- **条件三件套**：
  - **帧级动作条件**：动作块按帧注入 → 厘米级控制（Fig. 4 左半的「换动作换轨迹」实验）；
  - **多视角联合**：第三人称 ×2 + 腕部 ×1 联合去噪，腕部预测显著降低「视野剧变」带来的不一致（Fig. 5）；
  - **位姿记忆检索**：检索式记忆把长程漂移摁住（消融见 Table 2：去记忆/去帧级/去多视角全掉点）；
- **闭环用法**：
  - **测**：策略在模型里跑 rollout → 自动评指令跟随/成功率 → 与真机排名对表（Fig. 7）；
  - **改**：在模型里**筛出成功轨迹**当合成监督 → SFT 策略 → 真机涨幅 +44.7%（Fig. 9）——后续 VLAW 论文把这条「双向互相改进」扩成完整框架（仓库自带其后训代码）。

---

## 📸 逐图讲解（论文 9 张）

### Figure 1：总览——策略进环的多视角世界模型（头图）

（即文首头图。）左侧通用 VLA 策略与右侧世界模型构成闭环：初始帧 + 动作块进模型 → **联合多视角未来（含腕部）**吐回策略；标注两大用途——**(1) 想象中评测指令跟随 (2) 对未见指令做定向改进**。读图抓手：**「多视角 + 帧级控制 + 位姿记忆」三件套是闭环能转起来的前提**，缺任何一个都会在多步交互里散架。

### Figure 2：从 SVD 到可控世界模型

![初始化管线（论文 Figure 2）](/images/ctrlworld/fig2.png)

SVD 预训练权重 → 适配为**可控、时间一致**的世界模型：动作、多视角、记忆等条件通路被接入原视频扩散骨干。工程含义：**站在 8GB 级 SVD 肩膀上**——生成质量靠底座，可控性靠适配，这是小团队能训起世界模型的现实路径（与 [Sonic/SONIC 系](/posts/paper-reading/具身智能强化学习调研精读/)「站在大规模预训练上」的哲学一致）。

### Figure 3：长程 rollout 定性对比（验证集）

![长程对比（论文 Figure 3）](/images/ctrlworld/fig3.jpg)

Ctrl-World 与**单视角前作**在验证集长程回放上的多视角帧对比：前作单视角预测易出幻觉/漂移，本文多视角联合在 20 秒级轨迹上保持场景与物体恒常。配 Table 1（交互式长程生成的量化指标）阅读——定性图给「肉眼连续性」，表格给分数。

### Figure 4：可控性与消融——厘米级换动作换轨迹

![可控性（论文 Figure 4）](/images/ctrlworld/fig4.jpg)

**同一初始帧 + 不同动作序列 → 不同轨迹**的多组对照，叠加消融可视化（去帧级条件/去多视角的退化）。这是「Controllable」的字面证据：控制粒度到**厘米级**，而不是「大方向差不多」。

### Figure 5：腕部一致性

![腕部一致性（论文 Figure 5）](/images/ctrlworld/fig5.jpg)

单条轨迹内**腕部相机视野剧变**下的多视角一致性对照：腕部预测不是附属品——它既是现代 VLA 的输入刚需，也是「近距接触是否发生」的裁判；多视角联合显著降低腕部引发的不一致（配伍 Table 2 消融）。

### Figure 6：π0.5 真机 vs 世界模型（20 次交互）

![π0.5 真机 vs 想象（论文 Figure 6）](/images/ctrlworld/fig6.jpg)

每条轨迹含 π0.5 与环境的 **20 次交互**，上真机、下世界模型逐段对照——真机成功/失败的**行为模式**（何时抓偏、指令走到哪一步）在想象里被复刻。这是 Fig. 7 定量相关的定性注脚：**不是像素级复刻，是「策略行为级」复刻**。

### Figure 7：真机-世界模型定量相关

![相关性（论文 Fig. 7）](/images/ctrlworld/fig7.png)

两张散点：**指令跟随** y=0.87x−0.04、**成功率** y=0.81x−0.11（虚线=oracle）；图例 = π0/π0-fast/π0.5 × 7 任务（Pick-Place、Fold-Towel、Drawer、Wipe-table、Close-laptop、Pull-tissue、Stack）。结论一句话：**不用真机 rollout 也能给策略正确排名**——世界模型评测有效性的核心证据。

### Figure 8：后训任务与合成轨迹

![后训与合成（论文 Fig. 8)](/images/ctrlworld/fig8.jpg)

上排：**后训任务**（新物体/新指令等短板场景）；下排：世界模型内生成的**合成轨迹**。这是「改」的原材料——在想象里把失败任务的成功解法造出来。仓库 README 的两个 GIF（真机 vs 世界模型 20 秒并排）就是这条线的动态版，见第 4 章。

### Figure 9：策略改进 +44.7%

![改进柱图（论文 Fig. 9）](/images/ctrlworld/fig9.png)

Base vs Finetuned 五组柱：**Spatial 0.29→0.88、Shape 0.44→0.93、Towel-Dir 0.57→0.80、Novel-Obj 0.25→0.73、Average 0.39→0.83（+44.7%）**。四个维度恰好对应「空间理解、形状理解、方向指令、新物体」四类指令跟随短板——**合成数据专打语言接地的软肋**（配 Table 4–7 分任务细表）。

---

## 🧪 实验速查

1. **生成质量**：Table 1 交互式长程生成（验证集）+ Table 2 消融——记忆检索、帧级条件、多视角三者**缺一掉点**；
2. **评测有效性**（Fig. 6/7）：π0/π0-fast/π0.5 × 7 任务 × 真机 20 次，与世界模型排名强相关（0.87/0.81 回归）——**「想象排名」可用于策略选型**；
3. **改进有效性**（Fig. 8/9 + Table 3–7）：合成成功轨迹 SFT → 平均 **+44.7%**；按 Spatial/Shape/Towel-Dir/Novel-Obj 分表给出逐任务前后对照；
4. **零样本迁移声明**（README）：模型只在开源 DROID 上训练，**零样本**进作者自建新 DROID 场景与新相机位姿——并诚实标注「物理交互建模可能不准」。

---

## 🧰 官方 GitHub 详解：4 条跑法 + 训练/后训

**仓库**：[Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)（**575★，MIT，ICLR 2026 官方 PyTorch 实现**；同时收录 **VLAW**（arXiv 2602.12063）的世界模型后训管线）。

### 资产清单（README「CheckPoint and Dataset」表）

| 资产 | 来源 | 大小 |
| --- | --- | --- |
| CLIP text/image encoder | HF openai/clip-vit-base-patch32 | ~600M |
| SVD 视频扩散底座 | HF stabilityai/stable-video-diffusion-img2vid | ~8G |
| **Ctrl-World ckpt（DROID 训练）** | HF **yjguo/Ctrl-World** | ~8G |
| DROID 数据集 | HF cadene/droid_1.0.1（95k 轨迹/564 场景） | ~370G |

环境：`conda` Python 3.11 + `requirements.txt`；玩 π0.5 进环需另按 [openpi](https://github.com/Physical-Intelligence/openpi) 装 JAX 依赖（`GIT_LFS_SKIP_SMUDGE=1 uv sync`）。

### 跑法 ① DROID 轨迹回放（合成长视频）

从记录轨迹里采一帧作初始，**反复喂 1 秒动作块**自回归出完整 rollout：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path ${svd} --clip_model_path ${clip} \
  --ckpt_path ${ctrlworld_ckpt}
```

- 配置集中在 `config.py` 的 `__post_init__`；
- **速度提示（README 原文）**：A100 约 **10 秒/交互步**、H100 约 5 秒——一步=1 秒动作块；
- 本章贴的两组**三视角 mp4**（`dataset_example/droid_new_setup` 与 `..._full/drawer` 的 val 样本）就是这条管线的产出物形态（第三人称 + 腕部并排）。

### 跑法 ② 键盘交互（人在环造轨迹）

键盘命令逐条转动作块、**自回归顺序执行**，可一次输入多键：

```bash
... python scripts/rollout_key_board.py ... --task_type keyboard --keyboard lllrrr
```

合法键：`l/r/f/b`（左右前后）、`u/d`（升降）、`o/c`（开合夹爪）——官网 Interactive Demo 的网页版同款逻辑（见官网，另附 `keyboard_first_frame*.png` 初始帧素材）。

### 跑法 ③ π0.5 策略进环（policy-in-the-loop 的字面实现）

从**新开 DROID 场景快照**出发，让官方 **π0.5-DROID** 与世界模型闭环互动（两者都零样本迁移）；任务类型：`pickplace / towel_fold / wipe_table / tissue / close_laptop / stack`：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
python scripts/rollout_interact_pi.py ... --task_type pickplace
```

**工程细节**：π0.5 是 JAX 实现，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` 防止 JAX 预占满显存——这是把两家生态（PyTorch 世界模型 + JAX 策略）焊在一起的关键垫片。**作者声明**：只在开源 DROID 上训模型、新场景全零样本；可评指令跟随，但**物理交互可能不准**。

### 跑法 ③-New 论文评测协议（20 次/类）

论文口径：每类任务 **20 次**（5 或 10 个初始条件 × 重复 2 或 4 次），初始条件在 `dataset_example/droid_new_setup_full`；对应脚本：

```bash
... python scripts/rollout_interact_pi_eval.py ... --task_type fold_tower
```

### 跑法 ④ 训练：预训练（DROID）与后训（VLAW）

1. **潜空间预抽**（视频扩散在图像编码器潜空间训练，先抽潜提速）：`accelerate launch dataset_example/extract_latent.py ...`；
2. **元信息**：`create_meta_info.py` 生成 json、算状态/动作归一化；
3. **训练**：`accelerate launch scripts/train_wm.py ...`（仓库自带小子集 `droid_subset` 可先冒烟）；论文规模 = 1–2 节点 × **8×A100/H100**；
4. **后训（VLAW，2026-02 新增）**：接触密集/可变形任务预训练不够准时，按下游任务后训——**后训模型支持长程策略进环**，README 给出「同一初始条件下真机 20 秒 vs 世界模型 20 秒」的上下行对比 GIF（本章下方内嵌）；更多视频见 [VLAW 项目页](https://sites.google.com/view/vlaw-arxiv)（arXiv 2602.12063）。

**仓库结构速记**：`scripts/`（rollout_replay / rollout_key_board / rollout_interact_pi(_eval) / train_wm）+ `dataset_example/`（数据样例与潜空间工具）+ `synthetic_traj/gallery/`（README 演示媒体）+ `config.py`（全局配置）。致谢链：**SVD**（生成底座）+ **openpi**（π0.5 策略）。

---

## 🎬 演示媒体（GitHub 原样收录）

### 效果横幅（README 首图）

![Ctrl-World 生成轨迹画廊（GitHub README）](/images/ctrlworld/gh_banner.jpg)

同一世界模型生成的多场景轨迹拼画廊——README 开篇即挂它，对应论文多视角合成能力。

### 效果 1：真机 vs 世界模型 · 长程 20 秒并排（后训模型，上下行）

上=真机、下=世界模型，同一初始条件各 rollout 20 秒（VLAW 后训效果，README 官方素材）：

![真机上/世界模型下 对照 GIF 1（GitHub）](/images/ctrlworld/gh_post1.gif)

![真机上/世界模型下 对照 GIF 2（GitHub）](/images/ctrlworld/gh_post2.gif)

读法：盯住物体被移动的关键帧上下对齐程度——**行为级一致**（不是像素级克隆），这正是 Fig. 6 定量实验的动态版。

### 效果 2：DROID 新场景三视角合成轨迹（回放管线产出）

`dataset_example/droid_new_setup` val 样本（第三人称主视角 / 腕部视角 / 第三视角，各约 2–4 秒）：

<video controls width="32%" preload="metadata" src="/images/ctrlworld/gh_replay_v0.mp4"></video>
<video controls width="32%" preload="metadata" src="/images/ctrlworld/gh_replay_v1.mp4"></video>
<video controls width="32%" preload="metadata" src="/images/ctrlworld/gh_replay_v2.mp4"></video>

### 效果 3：抽屉场景三视角（`droid_new_setup_full/drawer`）

<video controls width="32%" preload="metadata" src="/images/ctrlworld/gh_drawer_v0.mp4"></video>
<video controls width="32%" preload="metadata" src="/images/ctrlworld/gh_drawer_v1.mp4"></video>
<video controls width="32%" preload="metadata" src="/images/ctrlworld/gh_drawer_v2.mp4"></video>

（仓库 `dataset_example/` 内还有 13+ 组同类三视角样本，均为跑法①的产物；官网另有键盘交互与对比视频合集——见 [ctrl-world.github.io](https://ctrl-world.github.io/) 的 Interactive demos 与 comparison 区。）

---

## 💡 评析：三个亮点，三条边界

### 三个亮点

1. **把「评测」做成了回归线**：Fig. 7 的 0.87/0.81 相关是世界模型从「演示玩具」变「评测基础设施」的通行证——排名一致比像素一致更便宜也更有用。
2. **改进闭环短且可复现**：合成成功轨迹 → SFT → +44.7%，全程不碰专家标注；四类短板（空间/形状/方向/新物体）全部受益——打的是 VLA 的「语言接地软肋」而非重新练操作。
3. **开源彻底**：官方实现 + ckpt（HF）+ 数据管线 + π0.5 进环脚本 + VLAW 后训——四条跑法把论文实验全部落成命令行。

### 三条边界

1. **物理可信度**：README 自陈「物理交互可能不准」——粘吸、穿透类幻觉在多视角下缓解但未根除；测「会不会做」可靠，测「力学是否对」仍需真机；
2. **算力门槛不低**：全量训练 8×A100/H100 起步，推理一步 5–10 秒——适合评几个关键策略，不适合海量在线评测；
3. **世界-策略共同漂移风险**：用同一个模型既出题又改卷（合成数据 SFT 后再回模型里验），过拟合到模型偏好的解法需要靠真机终验把关（论文 Fig. 9 有真机验收，应用时别省这步）。

---

## ❓ 自测 Q&A

**Q1：Ctrl-World 和 MuJoCo/Isaac 这类仿真器差在哪？**  
仿真器解物理方程（准但难长程好看、要资产），Ctrl-World 是**视频扩散生成**（好看、能零样本新场景、但物理可能不准）——它是**评测/改进的快速回路**，不是物理金标准；两者互补（真机终验仍是金标准）。

**Q2：为什么腕部视角这么关键？**  
现代 VLA 输入就含腕部；且腕部视野剧变是长程一致性的最大压力源——腕部预测稳了，「夹爪里到底有没有东西」才有裁判（单视角幻觉的病根）。

**Q3：+44.7% 是在想象里测的吗？**  
不是——**改进幅度在真机上验收**（Fig. 9/Table 4–7）；想象负责「造数据」，真机负责「盖章」。

**Q4：想最快上手仓库？**  
下载 HF `yjguo/Ctrl-World` ckpt + 仓库自带 `droid_subset` → 跑法①回放看 mp4 → 跑法②键盘玩交互 → 有 π0.5 环境再上跑法③。

**Q5：和 VLAW 什么关系？**  
同一作者线：Ctrl-World 是「可控世界模型」本体，VLAW（2602.12063）是「策略与世界模型**互相迭代改进**」的扩展——官方仓库（3）节的后训就是 VLAW 管线，README 的双行 GIF 即其效果。

---

## 📊 图表总索引（9 张论文图 + 9 件 GitHub 媒体）

| 图/媒体 | 内容 | 本站文件 |
| --- | --- | --- |
| Figure 1（头图） | 策略进环总览 | 本文插图 |
| Figure 2 | SVD → 可控世界模型 | 本文插图 |
| Figure 3 | 长程多视角定性对比 | 本文插图 |
| Figure 4 | 厘米级可控性 + 消融 | 本文插图 |
| Figure 5 | 腕部一致性 | 本文插图 |
| Figure 6 | π0.5 真机 vs 想象 20 次交互 | 本文插图 |
| Figure 7 | 真机-世界模型相关回归（0.87/0.81） | 本文插图 |
| Figure 8 | 后训任务 + 合成轨迹 | 本文插图 |
| Figure 9 | 策略改进 +44.7% 柱图 | 本文插图 |
| GitHub 横幅 | README 生成轨迹画廊 | 本文插图 |
| GIF ×2 | 真机 vs 世界模型 20 秒并排 | 本文插图 |
| MP4 ×6 | 两组三视角合成轨迹（视频内嵌） | 第 4 章 `<video>` 播放器 |

（Table 1–7 与 Algorithm 1 的关键数字已以文字表格复现。）

---

## 📚 资源与延伸

- 论文：arXiv [2510.10125](https://arxiv.org/abs/2510.10125)（ICLR 2026）；官网 [ctrl-world.github.io](https://ctrl-world.github.io/)（Interactive demos / comparison 视频）
- **官方代码**：[github.com/Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)（575★，MIT）· ckpt [HF yjguo/Ctrl-World](https://huggingface.co/yjguo/Ctrl-World) · 数据 [HF DROID](https://huggingface.co/datasets/cadene/droid_1.0.1)
- 姊妹工作：**VLAW**（策略-世界模型互改进）arXiv [2602.12063](https://arxiv.org/abs/2602.12063)，后训代码见官方仓库
- 同站相关：[具身智能强化学习调研精读](/posts/paper-reading/具身智能强化学习调研精读/)（世界模型/评测全景）、[OpenWAM-α 精读](/posts/paper-reading/OpenWAM-Alpha精读/)（另一路 WAM）、[π0.7 精读](/posts/paper-reading/π0.7精读/)（π 系策略，本文的被测对象家族）、[DROID 数据集精读](/posts/paper-reading/DROID机器人操作数据集精读/)
- 知识铺垫：[什么是世界模型](/posts/knowledge/什么是世界模型/)、[Flow Matching 入门详解](/posts/knowledge/Flow-Matching入门详解/)
