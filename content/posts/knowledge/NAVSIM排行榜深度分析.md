---
title: "NAVSIM 排行榜深度分析：谁在统治端到端规划？架构、创新与分数全解"
date: 2026-07-19
draft: false
categories: ["产业实践"]
tags: ["NAVSIM", "排行榜分析", "端到端自动驾驶", "PDMS", "EPDMS", "规划", "产业实践"]
summary: "NAVSIM 已成为端到端规划的事实标准benchmark，PDMS/EPDMS排行榜上群雄逐鹿。本文系统梳理navtest/navhard双榜Top方法，从Scoring-based、Diffusion-based、VLA、World Model 四大技术路线解构各家架构设计与核心创新，严格区分官方排行榜已录结果与arXiv宣称结果，并给出技术趋势判断与个人思考。数据截至2026年7月。"
---

## 📄 背景：NAVSIM 为何成为必争之地

NAVSIM 自 NeurIPS 2024 发布以来，已成为端到端自动驾驶规划的事实标准benchmark。它用**非反应式仿真**（non-reactive simulation）在开环框架下计算闭环相关性更强的指标：

- **v1 用 PDMS**：一套"是否安全合规"的多子指标加权分数；
- **v2 升级为 EPDMS**：扩展车道保持、红绿灯合规、方向合规和扩展舒适度等子指标；
- **v2 还推出 navhard 两阶段评测**：Stage 1 在原始场景评估 + Stage 2 在 3DGS 生成的偏移场景上再评估，专门测试规划器的鲁棒性。

截至 2026 年 7 月，NAVSIM 排行榜上的竞争已经白热化——Top 方法的 PDMS 从 2024 年的 66 分飙升到 94+，navhard 从个位数冲到 55。这不是一两个技巧的堆叠，而是整个技术路线的快速迭代。

> **一句话结论**：NAVSIM 是当下端到端规划最公平的"高考"——所有方法在同一批场景、同一套官方脚本下比分数，因此榜上数字比论文自报更可信。

### 三句话读懂本文

- **榜单在比什么**：模型对 12,146 个挑战场景各输出一条轨迹，官方服务器用 PDMS/EPDMS（一套"是否安全合规"的多指标加权分数）打分取均值。
- **谁在领跑**：v1 navtest 上 Scoring-based（CLOVER 94.5）最稳；v2 navhard 难例榜上 World Model（DriveFuture 55.5）反超人类；VLA+GRPO 是上升最快的路线。
- **怎么读本文**：先看图建立评测流程心智模型 → 再看数据集拆分（避免被"9000/1000/2146"误导）→ 最后按四大路线逐家拆架构。

### 本文结构导航

- [排行榜全景](#-排行榜全景) — 三张榜 Top10 速览
- [数据集与评测协议](#-数据集与评测协议谁到底在跑什么) — 拆分关系图 + 常见误解澄清
- [评测指标详解](#-评测指标详解pdms-与-epdms-到底在量什么) — PDMS/EPDMS 公式与计算链
- [四大技术路线架构详解](#-四大技术路线架构详解) — 11 家方法逐一看
- [技术路线对比与个人思考](#-技术路线对比) — 趋势判断

### NAVSIM 评测流程一图流

一句话：模型出轨迹 → 丢进非反应式仿真 → 算多个子指标 → 聚合成 PDMS/EPDMS → 全场景取均值。

![NAVSIM评测流程：输入场景→规划模型→非反应式仿真→子指标评分→聚合为PDMS/EPDMS→全场景取均值](/images/navsim/pipeline.svg)

*图1：NAVSIM 评测流程。所有榜单分数的产生都遵循这条链路——理解它，就理解了为什么"子指标"比"像不像人驾"更重要。*

**一个重要提醒**：本文区分两类分数来源：
- **官方排行榜（Leaderboard）**：通过 HuggingFace 提交、由官方评估服务器计算的分数，navtest 排名来自 [AGC2024-P/e2e-driving-navtest](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)，navhard 排名来自 [AGC2025/e2e-driving-navhard](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)
- **arXiv 宣称（Reported）**：论文中自我报告的结果，可能未提交到官方排行榜或提交了但未公开

这两者可信度不同——官方榜单一视同仁、可复现；arXiv 宣称可能有实验环境差异。

## 📊 排行榜全景

### NAVSIM v1 navtest（PDMS 排名）

| 排名 | 方法 | PDMS | 来源 | 路线 | 发表处 |
|:----:|------|:----:|:----:|:----:|:------:|
| 1 | CLOVER | **94.5** | 官方榜 | Scoring-based | arXiv 2605.15120 |
| 2 | ExploreVLA | **93.7** | arXiv | VLA+WM+RL | arXiv 2604.02714 |
| 3 | Centaur | 92.1 | 官方榜 | Scoring+TTT | arXiv 2503.11650 |
| 4 | SparseDriveV2 | 92.0 | 官方榜 | Scoring-based | arXiv 2603.29163 |
| 5 | Hydra-SE | 91.87 | 官方榜 | Scoring-based | 2025 |
| 6 | Hydra-MDP | 91.26 | 官方榜 | Scoring-based | NeurIPS 2024 |
| 7 | **DriveFuture** | 90.7 | **arXiv** | World Model | arXiv 2605.09701 |
| 8 | DriveVLA-W0 | 90.2 | 官方榜 | VLA+WM | ICLR 2026 |
| 9 | AutoVLA | 89.1 | arXiv | VLA | NeurIPS 2026 |
| 10 | WorldRFT | 87.8 | arXiv | WM+RL | AAAI 2026 |

### NAVSIM v2 navtest（EPDMS 排名）

| 排名 | 方法 | EPDMS | 来源 | 路线 | 备注 |
|:----:|------|:-----:|:----:|:----:|:----:|
| 1 | Human | 94.5 | 官方榜 | 人类 | 上限参考 |
| 2 | **Metis** | **90.3** | 官方榜 | WAM | arXiv 2606.15869 |
| 3 | CLOVER | 90.4 | 官方榜 | Scoring-based | 87.2 Variant |
| 4 | AutoDrive-P³ | 89.9 | 官方榜 | VLA+CoT+RL | perception=true |
| 5 | **DriveFuture** | 89.9 | **arXiv** | World Model | 未在官方榜出现 |
| 6 | EponaV2 | 88.9 | 官方榜 | WM+Flow | perception=false |
| 7 | ExploreVLA | 88.8 | arXiv | VLA+WM+RL | 未在官方榜出现 |
| 8 | DriveLaW | 88.6 | 官方榜 | VLA | - |
| 9 | PWM | 88.2 | 官方榜 | - | - |
| 10 | DriveWorld-VLA | 86.8 | 官方榜 | VLA+WM | - |

### NAVSIM v2 navhard（EPDMS 排名，两阶段）

| 排名 | 方法 | EPDMS | 来源 | 路线 |
|:----:|------|:-----:|:----:|:----:|
| 1 | **DriveFuture** | **55.5** | arXiv宣称为#1 | World Model |
| 2 | DrivoR | 54.6 | arXiv | - |
| 3 | SimScale | 53.2 | arXiv | 数据增强 |
| 4 | GuideFlow | 51.5 | arXiv | Diffusion |
| 5 | Expert | 51.3 | 官方榜 | 人类专家上限 |
| 6 | ZTRS | 45.5 | arXiv | - |
| 7 | World4Drive | 34.9 | 官方榜 | World Model |
| 8 | Metis | 32.2 | 官方榜 | WAM |
| 9 | MindDrive | 30.9 | arXiv | - |
| 10 | LTF | 24.4 | 官方榜 | Baseline |

navhard 的分数整体很低（Human Expert 也才 51.3），因为 Stage 2 的 3DGS 生成场景引入了观测偏移，会大量暴露规划器在分布外场景下的脆弱性。DriveFuture 以 55.5 超越人类上限本身就是一件有意思的事。

## 🗂️ 数据集与评测协议：大家到底在跑什么

在比较分数之前，必须先说清"榜单上的分数到底是在哪批数据上算出来的"。NAVSIM 不是一个从零采集的数据集，而是建立在 **OpenScene**（nuPlan 的降采样再分发版，约 120 小时城市驾驶、2Hz、每帧 8 路 1920×1080 环视相机）之上的**场景过滤器 + 评测框架**。

**基础数据集划分（OpenScene）**。OpenScene 自身提供 `trainval`（训练+验证，大）、`test`（测试，小）、`mini`（演示）三个标准划分。NAVSIM 在这之上叠加"困难场景过滤"，得到研究论文最常用的三个拆分。官方论文（NeurIPS 2024）给出的精确规模是：

| 拆分 | 基础 | 用途 | 官方规模（样本数，2Hz 帧） |
|:----:|:----:|:----:|:--------------------------:|
| **navtrain** | OpenScene `trainval` | 训练集 | **103,288** |
| **navtest** | OpenScene `test` | v1 评测集（PDMS） | **12,146** |
| **navhard_two_stage** | OpenScene `test` | v2 难例评测集（EPDMS，两阶段） | 450 真实 + 5462 合成 |

这里澄清一个常见误解：navtrain 的 10 万级样本来自 OpenScene 的 `trainval`（即你印象里"用来训练的那个大划分"），navtest 的 1.2 万样本来自 OpenScene 的 `test`（即"评测集"）——**navtrain 负责训练、navtest 负责出 PDMS 分数**，二者场景是允许重叠的（NAVSIM 的 train/test 过滤基于同一批日志做了不同难例筛选，不像标准机器学习那样严格不交叠）。所谓 "1000 calib / 2146 val" 是混淆了具体数字：真实官方数字是 **训练 103,288、评测 12,146**（12,146 容易被记成 2146，是少看了一位）。一句话总结：**大家刷的 NAVSIM v1 navtest 分数，就是模型在 OpenScene-test 上过滤出的 12,146 个挑战性场景上跑非反应式仿真算出来的 PDMS 均值**。

**关键约束**：`navtest` / `navhard` 这些测试拆分**禁止用于训练**（防止刷榜），可用其他公开数据或预训练权重，但必须在技术报告里声明数据使用。

### 一个必须澄清的社区误解：navtest 不是"9000 训练 / 1000 calib / 2146 评估"

经常有人（包括不少初学者笔记）把 NAVSIM 的拆分记成"navtest 里 9000 条训练、1000 条校准、2146 条评估"。我专门去查了官方文档、NeurIPS 2024 论文原文、GitHub issue（#139、#177 等）和主流论文的写法，**这个说法是错的，没有任何官方或主流论文把 navtest 切开当训练集用**。它的来源是把三个不同概念混在了一起：

- **"2146" 是 "12,146" 少看了一位** —— 这正是 navtest 的真实总量，不是它的一个子集。
- **"9000 训练" 对应的是 navtrain，但 navtrain 是 103k 而不是 9k** —— 你记的数量级偏小（可能把 10 万级的"10"看漏了）。
- **"1000 calib" 在 NAVSIM 官方拆分里不存在**，但方向上有个影子：**官方其实在 `navtrain` 内部提供了一份固定的验证集 `val_logs`**（见下一节代码实证）。需要澄清的是，之前流传的"85k 训练 / 18k calib"口径并不准确——`val_logs` 是按**日志（log）数**切的 2730 条 log，而非按帧数的万级划分；团队**确实会在训练集内部划出一部分做验证（calibration）**，这是从 navtrain 里划的、是官方固定划分，不是从 navtest 里划的。

所以正确的标准用法是：**navtrain（~103,288）全程训练、navtest（~12,146）全程评测**，二者场景允许重叠（官方文档明说 "the NAVSIM splits include overlapping scenes"），且 navtest 明文禁止用于训练。任何把 navtest 当训练集的做法，一旦提交官方排行榜都会被判违规（且分数不可比）。

### 🔬 代码实证：别人到底用 navtrain 的哪一部分训练、怎么切验证（扒开源 repo 得出）

前面都是从文档和论文层面讲"navtrain 训练、navtest 测试"。但"navtrain 训练"到底是不是**整个 navtrain 一把梭、完全不验证**？论文不写这种工程细节，所以我去扒了几个开源仓库的**实际训练代码**，结论很明确：**几乎所有走官方训练流程的方法，都会从 navtrain 内部切出一个固定验证集 `val_logs`，而不是整个 navtrain 全训。**

**官方训练流程的核心逻辑**。NAVSIM 官方训练入口 `navsim/planning/script/run_training.py` 并不是直接拿全部 navtrain 当训练集，而是先按一份固定的 log 名单做过滤：

```python
# navsim/planning/script/run_training.py（官方）
train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
val_scene_filter.log_names   = [l for l in val_scene_filter.log_names   if l in cfg.val_logs]
```

配套的划分文件 `navsim/planning/script/config/training/default_train_val_test_log_split.yaml` 里写死了两份 log 名单：

- **`train_logs`：13,180 条 log** —— 真正喂模型训练的部分
- **`val_logs`：2,730 条 log** —— 验证集，**且是 `train_logs` 的超集的补集、完全从 navtrain 的 log 池里切出**（不碰 navtest）

也就是说，**navtrain 整体 = train_logs ∪ val_logs**，验证集是 navtrain 的子集，规模约占总 navtrain 的 1/6。注意这里的单位是 **log（场景序列）数**，不是帧数——navtrain 的 103,288 帧分布在这 13,180 + 2,730 条 log 里，要拿精确验证帧数得实际跑 `SceneLoader` 统计。

> **一句话结论**：所谓"calib / val"不是你凭空从 navtrain 里乱切的，而是**官方早就给你切好了一份 `val_logs`（2730 logs）**——只要走官方 `run_training.py`，就自动带上它做验证。

**开源仓库实测（只要代码公开，全切）**：

| 仓库 | 是否切 val | val 规模 | 代码依据 |
|:----:|:----------:|:--------:|----------|
| **autonomousvision/navsim**（官方） | ✅ | 2730 logs | `run_training.py` + `default_train_val_test_log_split.yaml` |
| **SparseDriveV2**（swc-17） | ✅ | 2730 logs | 直接复用官方 `run_training.py` |
| **DiffusionDrive**（hustvl） | ✅ | 2730 logs | 直接复用官方 `run_training.py` |
| **DrivoR**（valeoai） | ✅ | 2730 logs | 直接复用官方 `run_training.py` |
| Hydra-MDP / Metis / AutoDrive-P³ / Centaur | 代码未公开，无法证实 | — | README 未放训练代码 |

**关键发现**：SparseDriveV2、DiffusionDrive、DrivoR 的训练脚本都只是 `train_test_split=navtrain` + 调用官方 `run_training.py`，因此它们和官方用**同一份 `val_logs` 划分文件**，验证集规模完全一致（2730 logs）。**没有任何一个可见的开源实现是"整个 navtrain 全训、零验证集"**——只要走官方训练流程，就必然带 `val_logs` 验证集。只有那些完全未公开训练代码的仓库（Hydra-MDP、Metis、AutoDrive-P³、Centaur）无法从代码证实，但它们的论文都写"trained on navtrain"。

**完整训练协议（代码层面总结）**：

- **训练用啥**：navtrain 中属于官方 `train_logs`（13,180 logs）的那部分场景，约占总 navtrain 的 5/6。
- **验证用啥**：navtrain 中属于官方 `val_logs`（2,730 logs）的那部分场景，约占总 navtrain 的 1/6，训练中途看验证 loss、选最优 checkpoint、调超参。
- **跑分用啥**：评测阶段提交轨迹到官方服务器，在 **navtest（12,146 帧，PDMS）** 或 **navhard（EPDMS，两阶段）** 上算分——这两份**禁止用于训练**。
- **为什么这样合法**：训练集和验证集都来自 navtrain（同一母集 trainval 滤出的子集），彼此不交叠；而测试集 navtest/navhard 来自 OpenScene 的 test 池，与 navtrain 来源不同，因此评测结果能真实反映泛化。这也正是为什么"把 navtest 拆 9000/1000/2146 训+验+测"既违规又自欺——你该用的是 navtrain 里的官方 `val_logs`，而不是去动 navtest。

**模型训练用的是 scene 还是 log？——答案是 scene，log 只是"切分粒度"。** 这是最容易混淆的点，必须拆开讲：

- **切分（split）按 log**：`train_logs` / `val_logs` 决定的是"哪些行车记录进训练、哪些进验证"。这一步用 log 是为了**防泄漏**——同一段 log 里的相邻帧场景高度相似，若按帧随机切，train 和 val 会共享同一段记录，验证分虚高。所以切分边界必须落在 log 之间。
- **训练（train）按 scene**：真正喂给模型、算 loss、反向传播的**最小样本单位是 scene（一个规划片段）**，不是整段 log。代码里 `SceneLoader` 会对 `train_logs` 圈定的每段 log，按场景过滤器（难例过滤 + 以某帧为中心的窗口）展开成一堆 scene，每个 scene 独立做一次前向/反向。你看到的 `len(train_data)`（日志里打印的 "Num training samples"）就是 **scene 数**，不是 log 数。
- **所以 navtrain 内部的两层关系是**：先按 log 圈范围（13,180 训练 log + 2,730 验证 log）→ 每段 log 再展开成若干 scene → 训练时 batch 里装的是 scene。103,288 这个总数就是 navtrain 全部 scene 的展开结果，它分布在 15,910 段 log 上。

一句话记忆：**log 决定"能不能用这段记录"，scene 决定"模型实际吃了哪一口数据"——切分看 log，训练吃 scene。**

![NAVSIM训练数据展开：train_logs/val_logs按log圈定→每段log展开为多个scene→batch装入scene训练](/images/navsim/train_expand.svg)

*图2c：navtrain 训练数据展开流程。切分边界在 log 层级（防泄漏），实际训练样本是 scene 层级——batch 里装的是一个一个 scene，而非整段 log。*

### 🛠️ 实操：你的自研模型怎么训、怎么验证、怎么跑分（官方脚本指向）

前面讲了协议，这里给出**真正落地跑分**的步骤。所有入口脚本都在官方仓库 `autonomousvision/navsim` 的 `navsim/planning/script/` 下，我亲自核对过它们的存在与用法。

**第一步：训练（用 navtrain，按官方 log 切分）**

复用官方训练入口，它会自动按 `train_logs` / `val_logs` 切分——你**不需要自己写切分逻辑**：

```bash
# 官方脚本 scripts/training/run_xxx_agent_training.sh 内部就是调这个
python navsim/planning/script/run_training.py \
    experiment_name=my_model \
    train_test_split=navtrain \
    agent=my_agent          # 你继承 AbstractAgent 实现的模型
```

- 训练集 = `train_logs`（13,180 logs）展开后的 scene
- 验证集 = `val_logs`（2,730 logs）展开后的 scene，每个 epoch 后自动算验证 loss
- **不要**碰 `navtest` / `navhard` 做训练，否则提交排行榜会被判违规

**第二步：本地验证（用 val_logs，自己看分数）**

训练中途/结束后，用官方 PDM 打分在 `val_logs` 上算 PDMS/EPDMS 自检（脚本 `run_pdm_score.py`）。这一步**只在本地看、不对外**，用来调超参、选 checkpoint。

**第三步：生成提交文件（在 navtest / navhard 上跑推理）**

评估阶段**不训练、只推理**。用官方提交脚本，把模型对测试集每个场景输出的轨迹存成 `submission.pkl`：

```bash
# 脚本：navsim/planning/script/run_create_submission_pickle.py
python navsim/planning/script/run_create_submission_pickle.py \
    agent=my_agent \
    train_test_split=navtest          # v1：输出 PDMS；换成 navhard_two_stage 即 v2 EPDMS
    team_name=... authors=... email=...
```

该脚本内部会：
- 对每个 token 调你的 `agent.compute_trajectory(agent_input)` 得到轨迹；
- 分 **first_stage**（原始场景）+ **second_stage**（3DGS 偏移场景）两轮输出；
- 打包成 `submission.pkl`（含 team 信息 + 两阶段预测）。

**第四步：算分（本地用 pickle，或提交官方服务器）**

- **本地先验分**（不用等排行榜排队）：用 `run_pdm_score_from_submission.py` 加载你刚生成的 `submission.pkl` + 官方 metric cache，直接算出 PDMS/EPDMS 并打印 `extended_pdm_score_combined`。
- **正式上榜**：把 `submission.pkl` 上传到 HuggingFace 官方排行榜空间（[navtest 榜](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest) / [navhard 榜](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)），由官方服务器用统一脚本算分——保证和所有人定义一致、可复现。

> **一句话流程**：`run_training.py`（navtrain 训，val_logs 验）→ `run_create_submission_pickle.py`（navtest/navhard 推理出 pkl）→ `run_pdm_score_from_submission.py`（本地算分，或传 HF 上榜）。**训练和验证只在 navtrain 内部，跑分只在 navtest/navhard——测试集永远不进训练。**

| 阶段 | 官方脚本（路径都在 `navsim/planning/script/`） | 数据 |
|:----:|:----|:----|
| 训练 | `run_training.py` | navtrain（train_logs 训 / val_logs 验） |
| 本地验分 | `run_pdm_score.py` | val_logs |
| 生成提交 | `run_create_submission_pickle.py` | navtest / navhard |
| 提交算分 | `run_pdm_score_from_submission.py`（本地）或 HF 空间（正式） | navtest / navhard |

### navtrain 那 10 万样本，别人到底怎么训？

navtrain 的 103k 样本（2Hz 采样，每帧 8 路环视相机 + 可选 LiDAR + ego 状态 + 导航命令）体量其实比 nuPlan 原版小很多（NAVSIM 故意降采样、只要相关标注），所以单机多卡就能训。从官方补充材料和 CVPR 2026 论文（DrivoR）的实现细节可以看到典型配置：

| 配置项 | 典型值（参考 DrivoR / 官方基线） | 说明 |
|:------:|:------------------------------:|------|
| 硬件 | 4×A100（v1 也可 1×3090/2080Ti） | navtrain 轻量，不需要超大集群 |
| Batch size | 16（基线 PlanCNN/TransFuser 用 64） | 视显存而定 |
| 学习率 | 2e-4（基线用 1e-4，50/75 epoch 后除以 10） | AdamW + cosine 退火 |
| Epoch 数 | v1 约 25（navtrain+navval）/ 100（小模型基线）；v2 约 10（navtrain only） | 大模型训练 1 epoch ≈ 1~1.5 小时 |
| 损失 | 轨迹回归（L1）+ 可选 score 蒸馏损失 | Scoring 类方法额外加评估器分数监督 |

关键认知：**navtrain 之所以"够训"是因为它是"难例过滤"后的子集**——官方用阈值把常量速度基线分数压到 22%、人类 95% 的场景留下，天然剔除了大量无聊直行场景，所以 10 万条里"信息密度"很高，训 10~25 个 epoch 就能收敛，不像原始 nuPlan 需要刷几百 epoch。这也是为什么榜单上很多方法光用 navtrain 就能冲到 90+ PDMS，而不必依赖外部大规模数据（除非像 EponaV2 那样主动追求 perception-free 的 data scaling）。

### 一张图理清所有名词：trainval / navtrain / navval / calib / navtest / navhard 到底谁是谁

先把最容易混的"基础划分"和"NAVSIM 过滤划分"两层关系理清楚：

![NAVSIM数据集拆分关系：OpenScene母集→trainval/test→navtrain/navtest/navhard，calib从navtrain切出](/images/navsim/dataset_split.svg)

*图2：NAVSIM 数据集拆分关系图。关键记住三点——（1）trainval 是母集，navtrain 从它滤出；（2）navtest/navhard 从 test 滤出，禁止用于训练；（3）calib/val 来自 navtrain 内部的官方固定 `val_logs`（2730 logs），不是独立拆分。*

### 先搞懂数据组织：log / scene / frame 三层到底啥关系

理解"13180 个 log"和"103288 个 scene"为什么差这么多，要先弄清 NAVSIM 数据的三层组织（官方 `docs/splits.md` 明说：标准划分指的是**可下载的 logs 集合**，而 navtrain/navtest 是**从 logs 里抽 scene 的过滤器**）：

- **log（日志）**：一辆车出去跑一次、连续录下的整段传感器数据。文件名形如 `2021.05.12.19.36.12_veh-35_00005_00204`——前段是出发时间戳、中段是车号、末段是帧区间。一个 log 是一段**连续的驾驶过程**，通常含几百到上千帧。
- **frame（帧）**：2Hz 采样的一张（每 0.5 秒一帧），是最小时间单位。
- **scene（场景）**：以某一帧为中心、截取前后若干秒的短片段，是规划任务的**基本样本单位**（模型看历史几秒、预测未来几秒）。所以 navtrain 的 **103,288** 这个"scenes/帧"数字，本质是难例过滤后留下的有效帧/场景数。

**三者关系一句话**：很多帧（103k）分布在较少的 log（~16k 段）里——一段 20 分钟的 log（约 2400 帧 @2Hz）经"难例过滤器"筛掉无聊直行段后，可能只留下几百个"有意思"的场景。

**为什么切分按 log 而不是按 frame/scene？** 这是关键设计——验证集必须按 **log 整段切**，不能按帧随机切。如果按帧随机切，同一段 log 里的相邻帧会同时出现在 train 和 val（它们场景几乎一样），验证集会"泄漏"训练信息、分数虚高；按 log 整段切，则保证**训练用的行车记录和验证用的行车记录完全不重叠**，验证结果才真实。这正是 `run_training.py` 里用 `log_names` 取交集（而非按 frame）的根本原因。

![NAVSIM数据三层组织：log(行车记录)→frame(2Hz帧)→scene(规划样本)，103288 scenes分布在15910 logs中](/images/navsim/data_hierarchy.svg)

*图2b：log → frame → scene 三层数据组织。navtrain 的 103,288 个 scene 分布在约 15,910 段 log 中；官方按 log 整段切成 13,180 训练 + 2,730 验证，避免同段行车记录泄漏到验证集。*

层级细节（用缩进表示，不画树形线）：

- **OpenScene**（nuPlan 降采样到 2Hz，底层数据源）
  - **trainval**：基础训练+验证大池（>2000GB，含全部日志）
  - **navtrain**：NAVSIM 在 trainval 上做"难例过滤"得到的训练集（103,288 帧，445GB）
       - 训练部分（官方 `train_logs`，13,180 logs）：真正喂模型训练
       - navval / calib（官方 `val_logs`，2,730 logs）：训练中途看验证损失用
  - **test**：基础测试池（217GB）
    - **navtest**：NAVSIM v1 过滤出的评测集（12,146 帧，指标 PDMS）
    - **navhard_two_stage**：NAVSIM v2 过滤出的难例评测集（指标 EPDMS，两阶段）
  - **mini**：演示小集（debug 用）

一句话记忆：**trainval 是母集，navtrain 是从它滤出的训练集，navtest/navhard 是从 test 滤出的测试集；calib/val 来自 navtrain 内部的官方固定 `val_logs`（2730 logs），不是独立拆分。**

逐个解释：

- **trainval**：OpenScene 的底层标准划分，是 navtrain 的"母集"。它包含全部日志和传感器（>2000GB），一般不整机下载，只用来抽 navtrain。
- **navtrain**：NAVSIM 在 trainval 上做"难例过滤"得到的**训练集（103,288）**。这是你模型训练时**唯一**应该用的数据。注意它和 navtest 场景允许重叠（NAVSIM 故意这么设计，见前文）。
- **navval / calib（校验集）**：**NAVSIM 官方没有单独的 `navval` 文件夹，但官方在 navtrain 内部提供了一份固定的验证集 `val_logs`（2730 条 log，见前文代码实证）**。训练流程会自动把 navtrain 按 `train_logs`（13,180 logs）/`val_logs`（2,730 logs）切成训练与验证两部分，用来监控训练 loss、选最优 checkpoint、调超参。你听人说的"calib 1000"本质上就是指这个内部验证切分，只是真实口径是按 log 数切的 2730 条、而非千级帧数。一句话：**calib = navtrain 里官方切好的 `val_logs` 验证集，不是独立拆分**。
- **navtest**：NAVSIM v1 的**测试/评测集（12,146）**，指标 PDMS。**禁止用于训练**。论文里报的"PDMS 94.5"就是在这上面算的。
- **navhard（navhard_two_stage）**：NAVSIM v2 的**难例测试集**，指标 EPDMS，两阶段（450 真实 + 5462 合成）。同样禁止训练。

所以回答你的核心疑问：**评估（看训练好不好）用的是 navtrain 内部官方切好的 `val_logs`（calib/val）；测试跑分（报给排行榜/论文）用的是 navtest（v1）或 navhard（v2）**。NAVSIM 没有"train/val/test 三独立集"的标准 ML 范式，而是"一个训练池 navtrain（内含官方 train/val 切分）+ 一个独立测试池 navtest"，验证集由官方从 navtrain 里切好、你直接拿来用。

### 端到端流程：别人到底怎么训、怎么评、怎么测

以 Scoring-based 代表 Hydra-MDP（CVPR/NeurIPS 挑战赛冠军）和主流做法为例，完整流水线如下：

**① 准备阶段（离线，一次性）**
- 下载 navtrain 传感器 + 日志，建 cache。
- Scoring 类方法会**预计算 PDM 分数当监督标签**：对 navtrain 里每条场景，用规则评估器（PDM-Closed 等）对一组候选轨迹算 NC/DAC/TTC/EP/C 等子分数，存成训练 target。这一步 Hydra-MDP 报告约 30 小时 / 32 核。
- 划分 navtrain → 训练集 + calib（val）。

**② 训练阶段（train on navtrain）**
- 输入：navtrain 训练集的环视图像（前视+左右前裁切拼成 256×1024）+ 可选 LiDAR BEV + ego 状态 + 导航命令。
- 输出：候选轨迹（或轨迹打分）。
- 监督：轨迹回归 L1 + 评估器分数蒸馏（多 head 各学一个子指标）。
- 配置（Hydra-MDP 官方）：**8×A100，总 batch 256，20 epochs，lr 1e-4，AdamW，无数据增强**。DrivoR 等后续工作用 4×A100、batch 16、25 epoch。
- 每个 epoch 后在 calib 上算验证 loss，挑最优 checkpoint。

**③ 本地评估（evaluate on calib 或 navtest 本地版）**
- 训练中途/结束后，在 calib（或自己留的 navtest 副本）上跑非反应式仿真算 PDMS，确认没过拟合、调超参。
- 注意：本地评 navtest 只能自己看，不能当官方成绩。

**④ 提交测试 / 跑分（test on navtest / navhard）**
- 官方排行榜方式：把模型对 navtest 全部 12,146 个场景输出的轨迹存成 `.pkl`，上传到 HuggingFace 排行榜空间，由**官方服务器用统一脚本算 PDMS/EPDMS**——保证所有人分数定义一致、可复现。
- navhard 同理：提交轨迹 `.pkl`，服务器在 450 真实 + 5462 合成场景上算 EPDMS。
- 你论文里写的分数，就是这步服务器返回的结果。

**⑤ 不同论文里数字不一致的原因（避坑）**
- 旧挑战赛版（Hydra-MDP 原文）写"Navtrain 1192 / Navtest 136 scenarios"——那是 **2024 挑战赛按 log 数、且过滤更严**的旧口径。
- 现在公开的 navtrain=103,288 / navtest=12,146 是**按 2Hz 帧**的新口径（issue #139 用户核对过：下全量是 103288 / 12146，下漏了会少一半）。
- 所以读到论文里"1192/136"别慌，那是 log 数；"103k/12k"是帧数，二者都对，只是计数单位不同。

### NAVTEST 是什么

**NAVTEST（通常写作 navtest）就是 NAVSIM v1 的标准测试拆分**——基于 OpenScene 的 `test` 划分、经困难场景过滤后得到的 **12,146 个场景**。它的评测指标是 **PDMS**（v1）。榜单上所有"PDMS 94.5 / 93.7 / 92.1..."之类的数字，都是模型对这 12,146 个场景各输出一条轨迹，丢进非反应式仿真算 PDMS 后取平均得到的。所以当你看到论文说"在 NAVSIM 上 PDMS 94.5"，等价于"在 navtest 的 12,146 个场景上 PDMS 均值 94.5"。

### NAVHARD 是什么，和 NAVTEST 有何不同

**NAVHARD 是 NAVSIM v2 引入的难例鲁棒性基准**（对应 `navhard_two_stage` 拆分），专门考"观测偏移 / 分布外"下的规划质量，指标升级为 **EPDMS**。它和 navtest 最大的区别是**两阶段评测**：

- **Stage 1（真实场景）**：450 个从 OpenScene `test` 里半自动筛选的真实挑战场景（人工挑选 + 对 SOTA 规划器做 failure mining，保证多样性），模型在原始观测上输出轨迹并评 EPDMS。
- **Stage 2（合成偏移场景）**：用 **3DGS（三维高斯泼溅）神经重建**把每个 Stage 1 场景重新渲染出**观测偏移**版本——比如把相机视角、物体位置做物理合理的微扰，生成 **5462 个合成场景**。模型在"看到的画面和训练时分布不一样"的情况下重新规划，再评 EPDMS。

最终 navhard 分数 = Stage 1 + Stage 2 的综合。因为 Stage 2 故意制造分布外观测，纯靠记忆训练分布的模型在这里会大幅掉分（比如 TransFuser/LTF 只有 23 左右，而懂动力学的世界模型能到 55+）。这也解释了为什么 navhard 整体分数远低于 navtest——它测的根本不是"常规驾驶"，而是"遇到没见过的观测时还稳不稳"。

**提交方式差异**：navhard 排行榜只需提交"每条测试帧的预测轨迹"，由官方服务器用 EPDMS 评测（不需要提交整个模型），所以比闭环榜单更容易规模化。

## 📐 评测指标详解：PDMS 与 EPDMS 到底在量什么

看懂分数前，先得看懂分母——尤其是 PDMS 的数学形式。NAVSIM 的分数不是"轨迹像不像人驾"，而是把一条轨迹丢进**非反应式仿真**（v1：在 BEV 抽象上展开 4 秒仿真时域，自车用 LQR 控制器跟踪轨迹、背景交通参与者按日志里的真实轨迹走，互不反应；v2：背景车改用 IDM 规则模型做反应式交互）里跑，再算多个子指标。理解子指标，才能理解各家方法为什么在某项上高、某项上低。

### PDMS 的公式（v1）

PDMS（Predictive Driver Model Score）由**乘性惩罚项**（违规直接打折）和**加权平均项**（非违规指标加权）两部分组成。计算链如下：

![PDMS计算链：NC/DAC乘性惩罚 × TTC/EP/C加权平均 → 单场景PDMS → 全场景取均值](/images/navsim/pdms_chain.svg)

*图4：PDMS 计算链。乘性项（红）是"木桶短板"——NC 或 DAC 任一为 0，整场景直接归零；加权项（黄）鼓励真的往前开（EP 权重最高）。*

官方定义为：

$$
\text{PDMS} = \underbrace{\left(\prod_{m\in\{NC, DAC\}} m(\text{agent})\right)}_{\text{乘性惩罚}} \cdot \underbrace{\left(\frac{\sum_{m\in\{TTC, EP, C\}} w_m \cdot m(\text{agent})}{\sum_{m\in\{TTC, EP, C\}} w_m}\right)}_{\text{加权平均}}
$$

其中子指标、权重与取值范围（NAVSIM v1 默认配置）为：

| 子指标 | 符号 | 类型 | 权重 $w_m$ | 取值 | 含义 |
|:------:|:----:|:----:|:----------:|:----:|------|
| 无责碰撞 | NC | 乘性 | — | $\{0, \tfrac12, 1\}$ | 自车是否引发碰撞（碰撞则该场景直接归零，最致命） |
| 可行驶区域 | DAC | 乘性 | — | $\{0, 1\}$ | 轨迹是否始终在可行驶区域内（不骑马路沿、不越野） |
| 碰撞时间 | TTC | 加权 | 5 | $\{0, 1\}$ | 与周围交通参与者的最小安全时距是否在界内 |
| 前进进度 | EP | 加权 | 5 | $[0,1]$ | 相对专家轨迹的归一化前进量（鼓励有效通行） |
| 舒适度 | C | 加权 | 2 | $\{0, 1\}$ | 加速度 / jerk 是否平缓 |

> 注：NAVSIM v1 里还有一个 **DDC（行驶方向合规）**，但**权重被设为 0**（即 PDMS 忽略它，留给 v2 启用）。所以 v1 的 PDMS 实际就是 `NC × DAC × (5·TTC + 5·EP + 2·C) / 12`。

两个关键点：
1. **乘性结构 = 木桶逻辑**：NC 或 DAC 任一为 0，整场景 PDMS 直接归零。所以"撞了 / 越野了"比"各项平庸"更致命——这也是 Scoring-based 方法拼命用评估器分数监督打分器的原因。
2. **EP 权重最高（5）**：鼓励自车真的往前开，防止模型为拿高 NC/DAC 分而"原地不动刷安全分"。

最终榜单上的 PDMS = 对所有评测场景（navtest 的 12,146 个）的 PDMS 取**均值**：

$$
\overline{\text{PDMS}} = \frac{1}{K}\sum_{i=1}^{K} \text{PDMS}_i
$$

### EPDMS 的公式（v2）

EPDMS（Extended PDMS）在 PDMS 框架上**扩展乘性项与加权项，并引入"人类过滤"**：若同一场景里人类专家也违规（如为绕开静止障碍物短暂借对向车道），则该惩罚被忽略，避免误罚合理行为。官方定义：

$$
\text{EPDMS} = \underbrace{\left(\prod_{m\in\{NC, DAC, DDC, TLC\}} \text{filter}_m(\text{agent}, \text{human})\right)}_{\text{乘性惩罚}} \cdot \underbrace{\left(\frac{\sum_{m\in\{TTC, EP, HC, LK, EC\}} w_m \cdot \text{filter}_m(\text{agent}, \text{human})}{\sum_{m\in\{TTC, EP, HC, LK, EC\}} w_m}\right)}_{\text{加权平均}}
$$

$$
\text{其中}\quad \text{filter}_m(\text{agent}, \text{human}) = \begin{cases} 1.0 & \text{若人类在该场景也违规 } m(\text{human})=0 \\ m(\text{agent}) & \text{否则} \end{cases}
$$

v2 的子指标、权重与取值范围：

| 子指标 | 符号 | 类型 | 权重 | 取值 | 含义 |
|:------:|:----:|:----:|:----:|:----:|------|
| 无责碰撞 | NC | 乘性 | — | $\{0, \tfrac12, 1\}$ | 同 PDMS |
| 可行驶区域 | DAC | 乘性 | — | $\{0, 1\}$ | 同 PDMS |
| 行驶方向合规 | DDC | 乘性 | — | $\{0, \tfrac12, 1\}$ | **v2 启用**：是否朝目标方向行进（防"为安全原地转圈"刷分） |
| 红绿灯合规 | TLC | 乘性 | — | $\{0, 1\}$ | **v2 新增**：严格考红绿灯状态 |
| 碰撞时间 | TTC | 加权 | 5 | $\{0, 1\}$ | 同 PDMS |
| 前进进度 | EP | 加权 | 5 | $[0,1]$ | 同 PDMS |
| 车道保持 | LK | 加权 | 2 | $\{0, 1\}$ | **v2 新增**：是否稳定居中车道内（路口处关闭） |
| 历史舒适度 | HC | 加权 | 2 | $\{0, 1\}$ | **v2 新增**：把预测轨迹前接 1.5s 人类驾驶历史再评舒适度 |
| 扩展舒适度 | EC | 加权 | 2 | $\{0, 1\}$ | **v2 新增**：比较相邻帧轨迹的动态状态（加速度/jerk 跳变） |

**为什么 v2 更难**：DDC / TLC / LK / HC / EC 这些新增项把"大致合规"升级成"精细合规"（不光不越野，还要居中车道；不光不撞，还要朝正确方向、按红绿灯走），所以同一方法在 navtest（PDMS）和 navtest v2（EPDMS）上分数会差一截——这也解释了为什么榜单上 EPDMS 普遍比 PDMS 低几个点。

理解了这套指标，后面的"为什么 Scoring-based 在 navtest 稳、World Model 在 navhard 强"就有了量化依据：打分器直接拟合这些子分数，自然在固定场景占优；世界模型能应对 Stage 2 的观测偏移，所以在难例榜反超。

## 🏗️ 四大技术路线架构详解

下面按 PDMS 从高到低，逐一拆解 Top 方法的架构设计和核心创新。先看一张路线总览，建立全局坐标感：

![四大技术路线定位：Scoring-based/VLA+WM/纯WorldModel/Diffusion在精度、鲁棒性、工程度上的定位](/images/navsim/route_overview.svg)

*图3：四大技术路线定位。横轴看"常规精度→难例鲁棒性"，纵轴看"工程简单→理论深度"——Scoring-based 占左上（稳而简单），World Model 占右下（难例强但重），VLA+GRPO 是中间快速上移的新势力。*

> **一句话结论**：排行榜不是"一家通吃"，而是四条路线各占一块地盘——常规场景 Scoring-based 称王，难例分布外 World Model 称王，VLA+GRPO 是上升最快的搅局者。

### 路线一：Scoring-based（评分排序范式）

这类方法占据排行榜头部最多席位。核心哲学：**不直接生成轨迹，而是先生成大量候选，再学习打分器选出最优**。

> **路线一句话**：不直接"开"，而是"先想一万个开法，再让打分器挑最好的那个"——所以它在固定场景（navtest）几乎不可战胜。

#### 1. CLOVER（94.5 PDMS / 90.4 EPDMS）

- **机构**：清华 AIR × 中科大 × 北航
- **分数定位**：navtest 官方榜第一（94.5 PDMS），navtest v2（EPDMS）90.4，是 Scoring-based 路线的天花板

**整体数据流**。CLOVER 把一个端到端规划问题拆成"生成候选集合"和"集合级打分排序"两件事，并用两个阶段分别训练：

- 图像（多视角环视）→ Image Encoder（DINOv2 ViT-S，冻结）
- → Proposal Generator（轻量 Transformer），生成 N 条候选轨迹
- → Scorer（Cross-Attention）
  - 候选轨迹 ↔ 场景特征交叉注意力
  - 输出每条轨迹的子分数（no, dj, tc, com, ...）
- → Top-1 选出（argmax over 候选）

**模块一：Proposal Generator（候选生成器）**。用冻结的 DINOv2 ViT-S 提取多视角环视图像特征（不训视觉骨干，省算力且泛化好），接一个轻量 Transformer decoder，以一组可学习的轨迹 query 为输入，一次前向直接回归出 N 条候选轨迹（通常 20~50 条）。关键点在于：生成器只负责"覆盖"动作空间，不负责"选优"——它学的是"把合理的驾驶行为都吐出来"。

**模块二：Scorer（打分器）**。这是 CLOVER 的核心。它把每条候选轨迹和场景特征做 cross-attention，预测 NAVSIM 评估器的各个子分数维度（no=无碰撞、dj=可行驶区域、tc=交通合规性、com=舒适度等）。训练时打分器的监督信号不是人驾轨迹，而是**真实评估器 R 在候选轨迹上算出来的分数**——这是它能"降维打击"的根本原因：它直接在学习排行榜的判分函数。

**核心创新一：伪专家覆盖训练（Stage 1）**。单条专家轨迹做模仿学习，模型容易 mode averaging（把所有合理行为平均成一个平庸轨迹）。CLOVER 改成"集合级覆盖"：从多类可解释动作族（横向偏移 ±k 米、加减速档位、跟车/绕行等）构造一个超大的候选池，用真实评估器 R 过滤掉低质候选，再用 FPS（最远点采样）从中挑出覆盖度最高的子集作为"伪专家集合"。监督目标从"逼近单条轨迹"变成"让打分器给伪专家集合里的轨迹打高分、给池子外打低分"——等价于教模型识别"什么是好驾驶"。

**核心创新二：保守闭环自蒸馏（Stage 2）**。Stage 1 的打分器只见过评估器分数，没见过"如何选"。Stage 2 交替优化：
- 固定生成器，更新打分器使其逼近真实子分数（回归损失）；
- 固定打分器作为"教师"，让生成器向教师选出的 Top-k 候选 + Pareto 最优目标学习（行为克隆式蒸馏）；
- 加一个稳定性正则（KL 约束生成器分布不要偏离 Stage 1 太远），防止分布漂移导致打分器失效。

**核心创新三：理论保障**。论文证明了一个关键引理：当打分器选中的目标在真实评估器 R 下统计显著优于当前分布采样时，这种"保守蒸馏"必然提升高分区（PDMS 头部）的概率质量。换句话说，它不是靠 tricks 刷分，而是有一套保证"越训越高分"的收敛理论。这也是它和一堆靠堆数据的方法最本质的区别。

- **官方榜 vs arXiv**：**官方榜**（navtest 可见，且 navtest v2 也可见）

#### 2. SparseDriveV2（92.0 PDMS / 90.1 EPDMS）

- **机构**：未公开
- **分数定位**：navtest 官方榜第四（92.0 PDMS），navtest v2（EPDMS）90.1，是 Scoring-based 路线里"词汇表设计"做得最精细的

**整体思路**。SparseDriveV2 继承自 SparseDrive 的稀疏查询范式（用一组实例 query 同时做检测和规划，避免稠密 BEV 计算），V2 把重点放在**轨迹词汇表（trajectory vocabulary）的因子化设计**上。

**模块一：因子化轨迹词汇表**。传统 anchor-based 方法把整条轨迹当成离散 token（比如 100 个固定轨迹模板），覆盖动作空间要么 vocab 太大、要么表达力不足。SparseDriveV2 把一条轨迹拆成两个因子：
- **几何路径（geometric path）**：横向/纵向的几何形状，描述"往哪走、拐多大弯"；
- **速度剖面（speed profile）**：沿路径的时间-速度曲线，描述"多快走"。

两个因子分别有各自的候选集合，再组合成完整轨迹。这样做的好处是：路径 × 速度的组合能指数级覆盖动作空间，而 vocab 尺寸只是两者之和——**Dense anchor 的组合比固定离散 anchor 更省参数、更全表达**。

**模块二：两级评分（two-level scoring）**。
- **粗粒度因子化评分**：先对"路径因子"和"速度因子"分别打分，快速筛掉明显不合理的组合（比如高速过急弯）；
- **细粒度组合评分**：对粗筛后保留的组合做联合打分（路径×速度的 cross-attention），得到最终 EPDMS 子分数。

这种两级结构平衡了效率和精度：大部分候选在粗粒度就被干掉，只有少量进入昂贵的细粒度打分。

**为什么有效**。因子化本质上是在"动作空间的结构"上做了先验归纳——驾驶行为本来就是"走哪条线"和"以多快速度走"两个相对独立的决策。把归纳偏置写进 vocab 设计，模型用更少的数据就能学出更好的打分器。

- **官方榜 vs arXiv**：**官方榜**（navtest 可见）

#### 3. Hydra-MDP 系列（91.26 PDMS / Hydra-SE 91.87）

- **机构**：NVIDIA
- **分数定位**：navtest 官方榜第六（Hydra-MDP 91.26）、Hydra-SE 91.87，是 Scoring-based 路线的"工业基准线"，被后续大量方法对标

**整体架构**。Hydra-MDP 是 NAVSIM 官方的代表性基线，也是"多目标蒸馏"思想的源头：

- 图像 / LiDAR → Transformer Encoder → 场景 token
- 固定 anchor 词汇表（K 条预定义轨迹）
- 多头打分器（Hydra heads），每个 head 对应一个专家评估器
- 聚合 → 选出 Top-1 anchor

**模块一：固定 anchor 词汇表**。和 CLOVER/SparseDriveV2 的"生成候选"不同，Hydra-MDP 用一组**预定义固定的 anchor 轨迹**（训练推理不变）。这意味着它的表达力受限于 vocab 大小，但也因此推理极快、训练稳定。

**模块二：多目标 Hydra 蒸馏（核心创新）**。NAVSIM 的 PDMS 由多个子指标组成（no/dj/tc/com 等），它们之间经常冲突（比如为了舒适就要慢，但慢可能不合规）。Hydra-MDP 不用一个总分监督，而是用**多个规则评估器（专家）各自算出的子分数**，分别监督对应的 score head：
- 每个 head 学习一种"偏好"（一个驾驶价值维度）；
- 训练时是多任务回归，每个 head 独立拟合对应专家分数；
- 推理时把多个 head 的输出按权重聚合（或按场景动态加权）得到最终排序。

这避免了"单一总分把多目标压扁"导致次优，也让模型对"哪个维度更重要"可调、可解释。

**模块三：Hydra-SE 的 Cluster Entropy（不确定性度量）**。Hydra-SE 在标准 Hydra-MDP 上增加了一个置信度信号：对 K 条 anchor 的打分分布计算**簇熵（cluster entropy）**——如果打分在多个 anchor 上都很接近（高熵），说明模型"拿不准"；如果集中在一个 anchor（低熵），说明模型很自信。这个不确定性可以：
- 用于安全兜底（高熵时降速或交给 fallback）；
- 作为 active learning 的信号（高熵样本值得再训）。

**定位意义**。Hydra-MDP 证明了"用规则评估器当老师、多 head 学多目标"这条路线在 NAVSIM 上极其有效且工程简单。后续 CLOVER、SparseDriveV2 本质上都是在它的"打分范式"上做升级（生成式候选 + 更聪明训练）。

- **官方榜 vs arXiv**：**官方榜**

### 路线二：VLA + 世界模型（Vision-Language-Action）

> **路线一句话**：把"语言推理 + 世界模型密集监督 + GRPO 在线 RL"拧成一股绳，是 2026 年上升最快的方向，但工程最重、arXiv 宣称居多。

#### 4. ExploreVLA（93.7 PDMS / 88.8 EPDMS）

- **机构**：未公开
- **分数定位**：navtest 宣布 93.7 PDMS（arXiv-only，未上官方榜）、navtest v2 88.8 EPDMS，是"VLA + 世界模型 + 在线 RL"路线的代表

**整体数据流**。ExploreVLA 的卖点是"用世界模型给自己造密集监督 + 用不确定性驱动 RL 探索"：

- 多视角图像 → ViT Encoder → 统一 Backbone（共享表征）
- 统一 Backbone 同时支撑三个分支：
  - 规划头（轨迹）
  - 世界模型头（未来 RGB）
  - 世界模型头（未来 Depth）
- 两个世界模型头的预测 → 预测不确定性 σ（像素级）
- 安全门控 GRPO 奖励：σ 高 且 PDMS 高 → 给奖励

**模块一：统一 Backbone**。ExploreVLA 用一个共享视觉 backbone 同时支撑三个头（规划、未来 RGB、未来 Depth），而非三个独立网络。好处是视觉表征被"多任务"拉扯得更鲁棒，且参数量可控。

**模块二：密集世界建模（核心创新一）**。普通端到端规划只监督"轨迹对不对"，信号稀疏（一条轨迹 vs 一张图）。ExploreVLA 额外让模型预测**未来若干帧的 RGB 图像和 Depth 图**——这给 backbone 提供了像素级、帧级的密集几何+外观监督。世界模型头本质上在强迫模型"理解场景会怎么演化"，这种理解会反哺规划头。

**模块三：不确定性驱动的探索奖励（核心创新二）**。这是它最巧妙的设计。世界模型在预测未来帧时，对"分布外/没见过的区域"预测不确定度高（σ 大）。ExploreVLA 把 σ 当成**内在探索信号**：
- 高不确定性 = 场景里有模型不懂的东西；
- 但如果此时 PDMS 也高（轨迹安全合规），说明"在未知但安全的区域里开得好"——这正是值得强化学习的"有信息量的好样本"。

**模块四：安全门控 GRPO（核心创新三）**。GRPO 是 group-relative 的 RL 算法（对一个 prompt 采样一组轨迹，用组内的相对优势做策略更新）。ExploreVLA 给 GRPO 加了一个**安全门控**：只有当"不确定性高 且 PDMS 高"同时满足时才给探索奖励，否则（不确定性高但 PDMS 低 = 危险探索）不给甚至惩罚。这从奖励设计上堵住了"为了探索去冒险"的漏洞。

**为什么分数高但没上官方榜**。93.7 是它论文自报的，评估协议（传感器、数据范围、是否用未来帧标注）可能与官方榜不完全一致，因此未进入官方 navtest 排名。这是"arXiv 宣称"的典型案例。

- **官方榜 vs arXiv**：**arXiv 宣称**（未出现在官方 navtest 排行榜）

#### 5. AutoDrive-P³（89.9 EPDMS，perception=true）

- **机构**：北京大学，**ICLR 2026**
- **分数定位**：navtest v2 官方榜（perception=true 分类）89.9 EPDMS，是"可解释推理链 + 分层 RL"路线的代表

**整体数据流**。AutoDrive-P³ 的骨架是一个 VLM，但它把规划拆成一条显式的推理链（Chain-of-Thought），并用 RL 把奖励从规划端反向渗透到感知、预测端：

- 图像 + 指令 → VLM Backbone
- VLM Backbone 同时输出三层（由 P³-CoT 结构化标签监督）：
  - 感知（Perceive）："附近有什么"
  - 预测（Predict）："它们会怎么动"
  - 规划（Plan）："我该怎么做"
- 三层汇成 P³-CoT 结构化标签
- P³-GRPO 分层渐进式强化微调（奖励从规划反传至感知、预测）

**模块一：P³-CoT 数据集（核心创新一）**。大部分 VLA 要么直接出轨迹（不可解释），要么只做语言 CoT（不落地到控制）。AutoDrive-P³ 构建了三层结构化的推理链标签：
- **感知层**：描述周围关键物体（车辆/行人/车道/红绿灯状态）；
- **预测层**：推断这些物体未来几秒的运动意图；
- **规划层**：基于前两层推导出本车轨迹，并附带语言解释。

训练时模型被要求"显式输出这三层"，而不只是末尾的轨迹。这让中间过程可被检查、可被纠错。

**模块二：P³-GRPO 分层渐进式强化微调（核心创新二）**。普通 GRPO 只给最终轨迹一个奖励，梯度很难传到前面的感知/预测。AutoDrive-P³ 把 RL 拆成三阶段渐进：
- **Stage 1**：冻结预测、规划，只 RL 优化感知（奖励来自"感知准不准"）；
- **Stage 2**：冻结感知、规划，只 RL 优化预测；
- **Stage 3**：联合 RL，奖励从规划（PDMS/EPDMS）反传到整条链。

这种"先分阶段打通、再联合精调"的策略，解决了长链条 RL 里奖励稀疏、梯度消失的问题。

**模块三：双思维模式（核心创新三）**。推理时提供两种模式：
- **Detailed Thinking**：走完整 P³-CoT，精度高，用于离线/复杂场景；
- **Fast Thinking**：跳过显式推理链直接出轨迹，延迟低，用于车端实时。

同一个权重、不同解码策略，兼顾了"可解释"和"快"。

- **官方榜 vs arXiv**：**官方榜**（perception=true 分类下可见）

#### 6. DriveVLA-W0（90.2 PDMS / 86.1-86.9 EPDMS）

- **机构**：中科院自动化所，**ICLR 2026**
- **分数定位**：navtest 官方榜 90.2 PDMS、navtest v2 86.1~86.9 EPDMS，是"世界模型放大 scaling law"路线的代表

**整体架构**。DriveVLA-W0 的核心论点是：**世界模型提供的密集监督能把规划的数据 scaling law 显著放大**——同样的数据量，有未来帧预测做监督，模型学得更准、泛化更好。

- 图像 → VLA Backbone，并行接两个分支：
  - 动作分支（轨迹，直接输出）
  - 世界模型（WM）
    - 离散版：Autoregressive 预测 visual token
    - 连续版：Diffusion 预测未来视觉特征
- 世界模型的预测作为密集监督信号 → 反哺 Backbone

**模块一：世界模型作为密集监督（核心创新一）**。规划标签（一条轨迹）很稀疏，而未来帧（连续多帧图像/特征）是密集信号。DriveVLA-W0 训练世界模型去预测"如果按某轨迹开，未来会看到什么"，这个预测误差作为额外的监督梯度回传到 backbone。直观理解：要预测准未来画面，模型必须真正理解"当前场景的几何、物体运动、自车动作后果"——这比单纯拟合轨迹逼迫模型学得更深。

**模块二：双版本世界模型（核心创新二）**。
- **离散版（Autoregressive WM）**：把未来帧量化成 visual token，用自回归方式预测 token 序列，和语言建模同源，训练稳定；
- **连续版（Diffusion WM）**：直接在连续视觉特征空间用扩散模型去噪预测未来特征，保真度更高但更难训。

两条路线论文都给了结果，工程上可按算力取舍。

**模块三：轻量级动作专家 MoE（核心创新三）**。训练时世界模型和动作分支一起训；**推理时**用一个轻量的动作专家（action expert）直接走动作分支，**bypass 掉世界模型**的前向（不预测未来帧）。这样车端部署时延迟接近纯规划模型，却享受了训练时世界模型带来的表征增益——典型的"训练贵、推理便宜"。

- **官方榜 vs arXiv**：**官方榜**

### 路线三：纯 World Model / WAM（世界动作模型）

> **路线一句话**：把"未来状态"当成规划的条件而非预测目标，因此当观测被 3DGS 偏移（navhard Stage 2）时仍能"想象"合理未来——难例榜的真正赢家。

#### 7. DriveFuture（90.7 PDMS / 89.9 EPDMS / navhard 55.5 #1）

- **机构**：未公开
- **分数定位**：navtest 宣布 90.7 PDMS、navtest v2 宣布 89.9 EPDMS，navhard 宣布 55.5（自称 #1，超越人类专家 51.3），是"未来条件化世界模型"路线的代表，也是 navhard 难例榜单的领头羊

**整体数据流**。DriveFuture 的灵魂是"把未来状态当成决策的条件，而不是预测的目标"：

- 当前观测（图像/状态）→ Encoder → 当前潜在状态 z_t
- → 未来潜在预测器
  - 训练时：用 GT 未来 z_{t+1} 做条件（teacher forcing）
  - 推理时：用预测 ẑ_{t+1} 接替（自回归）
- → Cross-Attention 精炼（候选轨迹 ↔ 条件化未来状态）
- → Diffusion 规划器 → 输出轨迹

**模块一：Encoder + 未来潜在预测器**。先把当前多模态观测编码成潜在状态 z_t，再用一个潜在动力学模型预测未来潜在状态 z_{t+1}（可以是多步）。关键在于"未来潜在"是**条件**，不是输出目标——模型生成轨迹时，会 attend 到"未来的样子"。

**核心创新一：未来条件化潜在世界模型**。常规世界模型是"给动作，预测未来"（未来是输出）；DriveFuture 反过来——"把（预测出的）未来状态作为条件，去规划动作"。两者的 train-inference gap 完全不同：
- 常规 WM：训练时未来是 GT（teacher forcing），推理时未来是自己猜的 → 分布漂移大；
- DriveFuture：训练时未来就**已经是条件输入**（用 GT 未来状态 conditioning），推理时换成预测未来状态接替 → 训练和推理用的是同一个"条件化"接口，gap 小。

**核心创新二：统一训练-推理范式**。正因为训练、推理都走"未来条件化"，DriveFuture 不存在"训练看 GT 未来、推理看预测未来"的不一致。这是它泛化好的关键。

**核心创新三：navhard 表现突出**。navhard 的 Stage 2 用 3DGS 生成"观测偏移"场景（比如把摄像头视角、物体位置做物理合理的扰动），专门考分布外鲁棒性。DriveFuture 因为内部有一个显式的"未来潜在动力学"模型，当观测被偏移时，它可以在潜在空间"想象"出偏移后合理的未来状态，再基于这个想象做规划——比纯模仿模型（看到偏移就懵）鲁棒得多，所以 navhard 55.5 反超人类上限。

- **官方榜 vs arXiv**：navtest **arXiv 宣称**（未出现在 navtest 官方榜）；navhard **arXiv 称 #1**（navhard 官方榜未公开具体排名）

#### 8. Metis（90.3 EPDMS / 32.2 navhard）

- **机构**：复旦大学 × 理想汽车 × 伦敦帝国理工
- **分数定位**：navtest v2 官方榜 90.3 EPDMS（Top）、navhard 官方榜 32.2，是"世界动作模型（WAM）+ 混合专家"路线的代表，且**完全开源**

**整体架构**。Metis 用 Mixture-of-Transformers 把"视频生成"和"动作预测"拆成两个独立 expert，在潜在空间统一做前向推演：

- 当前帧 + 历史 → Mixture-of-Transformers
  - 视频生成 Expert：预测未来帧潜在（视频）
  - 动作预测 Expert：预测自车轨迹（动作）
- 非对称注意力掩码：训练时双向，推理时动作 expert 跳过视频生成
- 世界动作模型（WAM）：潜在空间前向推演 → 轨迹

**核心创新一：解耦视频-动作建模**。如果用一个网络同时生成视频和预测动作，两种信号的表征会"混叠"（视频的高频纹理和动作的低频动力学互相干扰）。Metis 用两个 Transformer expert 分别承载，各自有独立的参数和表征空间，互不污染。

**核心创新二：非对称注意力掩码**。训练时两个 expert 通过非对称注意力联合训练（视频 expert 能看到动作信息、动作 expert 也能参考视频信息），保证两者对齐；**推理时**动作 expert 被配置成"不看视频生成分支"，直接出轨迹，从而跳过昂贵的视频生成计算，延迟大幅下降。这是"训练时共享、推理时解耦"的经典 MoE 思路。

**核心创新三：世界动作模型（WAM）**。Metis 提出把"世界模型"和"动作模型"统一成一个概念——在潜在空间里既能推演"环境会怎样"（世界），也能推演"我动了会怎样"（动作），二者共享同一套前向动力学。WAM 强调"行动"是世界推演的一部分，而不是事后接一个控制器。

**工程价值**。Metis 是榜单头部里少有的完全开源方案，对想复现 WAM 思路的团队非常友好。

- **官方榜 vs arXiv**：**官方榜**（navtest 和 navhard 都可见）

#### 9. EponaV2（88.9 EPDMS，perception-free）

- **机构**：未公开
- **分数定位**：navtest v2 官方榜 88.9 EPDMS（perception=false 分类），是"无感知标注 + 世界模型 + Flow Matching GRPO"路线的代表

**整体架构**。EponaV2 的诉求很工程化：**不要人工感知标注，也能训出强规划器**。

- 图像 → 自回归扩散世界模型，并行预测三种未来表征：
  - 未来 RGB 帧
  - 未来 Depth map（几何）
  - 未来 Semantic map（语义）
- 三种未来表征 → 综合未来表征 → 条件化规划头
- Flow Matching + GRPO → 轨迹

**核心创新一：综合未来推理**。EponaV2 的世界模型不只预测未来 RGB，还同时预测**未来深度图（3D 几何）和未来语义图（物体类别）**。RGB 提供外观、Depth 提供几何、Semantic 提供结构——三者合起来让模型对"未来世界"的理解比单预测 RGB 扎实得多，也更符合驾驶需要"空间+语义"的本质。

**核心创新二：Flow Matching GRPO**。用 Flow Matching（流匹配，扩散模型的一种连续归一化流形式）来建模轨迹分布，配合 GRPO 做 RL 微调。Flow Matching 相比传统 DDPM 扩散，训练目标更直接（回归向量场）、收敛更快，和 GRPO 的梯度流也更搭。

**核心创新三：Perception-free**。整个 pipeline 不依赖人工标好的 3D 框、车道线、语义标签——未来深度/语义图是世界模型自己从视频里自监督学出来的。这意味着它可以直接在海量无标注驾驶视频上做 data scaling，绕开了感知标注这个昂贵瓶颈。

**定位**。EponaV2 证明"无标注 + 世界模型"路线已经能摸到 88.9 EPDMS，如果数据规模拉起来，上限值得期待。

- **官方榜 vs arXiv**：**官方榜**

### 路线四：Diffusion-based / Flow Matching 范式

> **路线一句话**：把规划当成"从噪声轨迹去噪"，天然支持多模态，靠截断扩散/引导把步数压到 4~8 步——是扩散规划方法的基础基线。

#### 10. DiffusionDrive 系列（85.6-88.2 EPDMS）

- **机构**：华中科大，**CVPR 2025**
- **分数定位**：navtest v2 官方榜 85.6~88.2 EPDMS（多版本），是"截断扩散策略"路线的开创者，也是扩散规划里最常被引用的基线

**整体架构**。DiffusionDrive 把规划当成"从噪声轨迹去噪到真实轨迹"的扩散过程，但做了关键改造来解决"去噪步数多、慢"的痛点：

- 随机噪声轨迹 x_T → Denoising Network（条件：场景特征），每步从 anchor 高斯而非标准高斯采样
- → x_{T-1} ... x_0（候选轨迹，多模态）
- → 打分/聚类 → 选出最优轨迹

**核心创新一：截断扩散策略（Truncated Diffusion Policy）**。标准扩散从标准高斯噪声出发，需要几十步去噪才能成型。DiffusionDrive 改为：**从一组 anchor 高斯分布出发**——每个 anchor 对应一个"大致合理的驾驶模式"（直行/左转/跟车…），噪声只加在 anchor 附近。这样去噪步数可以从几十步砍到 4~8 步，推理速度大幅提升，同时天然支持多模态（不同 anchor 对应不同模式）。

**核心创新二：多模态输出**。因为有多个 anchor，一次去噪就吐出多个模式的候选轨迹，再配一个轻量打分器（或直接用扩散的似然）选最优。这比单峰回归更适合驾驶这种"一题多解"的任务。

**版本演进**。论文有多个变体（不同骨干、是否接感知、是否加 RL），EPDMS 落在 85.6~88.2 区间。它是后续很多扩散/流匹配规划方法的起点。

- **官方榜 vs arXiv**：**官方榜**（多版本）

#### 11. GuideFlow（51.5 navhard）

- **机构**：未公开，**CVPR 2026**
- **分数定位**：navhard 宣布 51.5 EPDMS（arXiv），是"Flow Matching + 引导"在难例榜上的代表

**整体架构**。GuideFlow 把 Flow Matching 用于轨迹生成，并引入"引导（guidance）"机制提升多模态表达：

- 噪声轨迹 → Flow Matching 向量场网络（条件：场景），含两个分支：
  - 无条件分支（纯轨迹流）
  - 引导分支（classifier/奖励引导，偏向安全合规模式）
- 引导加权 → 多模态候选轨迹

**核心创新：引导式 Flow Matching**。标准 Flow Matching 只学数据分布，生成的轨迹可能覆盖但不一定"好"。GuideFlow 在流匹配网络上加一个引导信号（类似 classifier guidance：把"安全/合规/高分"的梯度方向叠到向量场上），让生成过程偏向高质量模式，同时保留多模态（不坍缩成单峰）。在 navhard 这种难例分布外场景里，引导能把模型从"瞎生成"拉向"符合物理和安全约束"的轨迹。

- **官方榜 vs arXiv**：**arXiv**（navhard）

## 🔬 技术路线对比

| 维度 | Scoring-based | Diffusion-based | VLA+WM | Pure WM/WAM |
|:----|:-------------:|:---------------:|:------:|:-----------:|
| **代表方法** | CLOVER, SparseDriveV2, Hydra-MDP | DiffusionDrive, GuideFlow | ExploreVLA, AutoDrive-P³ | DriveFuture, Metis, EponaV2 |
| **最高 PDMS** | 94.5 | ~88 | 93.7 | 90.7 |
| **最高 EPDMS** | 90.4 | ~88 | 89.9 | 90.3 |
| **navhard EPDMS** | — | 51.5 | — | 55.5 |
| **推理速度** | 快（一次前向） | 中（4-8步去噪） | 慢（VLM 解码） | 中-慢 |
| **多模态支持** | 依赖 vocab 覆盖 | ✅ 天然 | ✅ 天然 | ✅ 天然 |
| **是否需额外数据** | 否 | 否 | 否 | 是（未来帧标注） |
| **理论深度** | 高（蒸馏理论） | 中 | 中 | 高（潜在动力学） |
| **工程复杂度** | 低-中 | 中 | 高 | 高 |
| **代码开源** | ✅ CLOVER | ✅ DiffusionDrive | ❌ 大部分否 | ✅ Metis |

### 关键洞察

**1. Scoring-based 在常规场景仍然最稳。**
CLOVER 的 94.5 PDMS 和 SparseDriveV2 的 90.1 EPDMS 说明，当你的评测环境是确定的（navtest），"生成候选 + 打分排序"这个范式几乎不可战胜。它的优势在于：
- 打分器可以直接用真实评估器的分数做监督，降维打击
- 候选生成可离线优化，推理时只做排序，延迟极小
- 集合级训练天然对抗 mode averaging

**2. World Model / WAM 在难例场景异军突起。**
navhard 上 DriveFuture 的 55.5 大幅领先（人类也只有 51.3），说明世界模型在面对分布外场景时（3DGS 偏移观测）的鲁棒性更强。这也是直觉上正确的方向——世界模型学会了环境动力学，当观测偏移时，它可以"想象"出合理的未来状态再做规划。

**3. VLA + GRPO 是上升最快的方向。**
ExploreVLA（93.7 PDMS）和 AutoDrive-P³（89.9 EPDMS）都用 GRPO 做 RL 后训练，而且都引入了某种形式的"世界模型"或"链式思维"作为推理增强。这是 2026 年最显著的趋势——**纯模仿学习已经到顶，RL 后训练成为标配**。

**4. Perception-free + WM 有望突破数据瓶颈。**
EponaV2 不需要人工感知标注，只靠未来深度/语义预测的自监督就能达到 88.9 EPDMS。如果真的把数据 scaling 跑起来，这条路线有潜力追上甚至超越依赖标注的方案。

## ⚠️ 排行榜来源透明化

一个重要的事实：**很多高分的 arXiv 论文并没有出现在官方排行榜上**。具体来说：

| 方法 | 宣称分数 | 官方榜可见 | 说明 |
|:----|:--------:|:---------:|------|
| CLOVER | 94.5 PDMS | ✅ 可见 | 官方榜 Top |
| ExploreVLA | 93.7 PDMS | ❌ 未出现 | arXiv-only |
| DriveFuture | 90.7 PDMS / 55.5 navhard | ❌ navtest 未出现，navhard 宣称 #1 | arXiv-only；navhard 官方榜未公开具体排名 |
| Metis | 90.3 EPDMS | ✅ 可见 | 官方榜 Top |
| AutoDrive-P³ | 89.9 EPDMS | ✅ 可见 | Perception=true 分类 |
| SparseDriveV2 | 92.0 PDMS | ✅ 可见 | 官方榜可见 |
| Centaur | 92.1 PDMS | ✅ 可见 | 官方榜可见 |

为什么 arXiv 宣称和官方榜有差异？可能原因：
1. **评估协议差异**：perception=true/false、传感器配置、训练数据范围等
2. **排行榜提交限制**：部分论文发表时排行榜已关闭提交
3. **分数时效性**：排行榜持续更新，论文报告的是某个时间点数据
4. **submit 意愿**：有些组只发了论文懒得提交

因此我的建议：**引用分数时优先参考官方排行榜的成绩**，arXiv 宣称的分数需谨慎对待，尤其是那些宣称 "SOTA" 但未提交到官方榜的。

## 💭 个人思考

### 1. 技术收敛了吗？

看起来 Scoring-based 和 World Model / WAM 正在形成两条互补的技术路线：
- **简单场景 → Scoring-based**：高确定性、高速度、已接近人类上限
- **难例/分布外 → World Model / WAM**：具备动力学理解能力，但计算开销大

但我认为最终会收敛到**世界模型 + 评分排序**的混合范式。DriveFuture 已经在尝试这条路（用世界模型做未来条件化，再用 scorer 选轨迹）。CLOVER 如果引入世界模型做伪专家生成，可能直接冲到 96+。

### 2. GRPO 是万能药吗？

2026 年几乎每篇高分论文都在用 GRPO。但我有两个担忧：
- **Reward hacking 风险**：当 reward 是规则化指标（PDMS/EPDMS）而非真实驾驶质量时，策略会找到投机取巧的方式最大化分数
- **在线 RL 的仿真 gap**：NAVSIM 是非反应式仿真，规划器不与环境交互。闭环 GRPO（真正在仿真器里开车学）还没大规模验证

CLOVER 的保守蒸馏理论提供了一个更安全的替代方案——不需要在线交互就能利用评估器反馈。

### 3. navhard 的意义

navhard 是一个被低估的评测维度。55.5 分超过人类上限这件事让我反思：
- 3DGS 生成的"观测偏移"可能包含了评测本身的漏洞（某些偏移在物理上不可能）
- 但也确实筛选出了真正懂动力学的模型（DriveFuture）和普通模仿模型（LTF 23.1）之间的差距

未来如果 navhard 被更广泛接受，"navtest 刷分"的军备竞赛可能会降温。

### 4. 对 VLA 工程师的启示

结合前面的 JD 分析，从 NAVSIM 排行榜可以看到用人单位到底在找什么能力：
- **RL 后训练**（GRPO/PPO）—— ExploreVLA、AutoDrive-P³、WorldRFT 都证明了 RL 的价值
- **世界模型** —— 从 DriveFuture 到 EponaV2，世界模型正在从"研究玩具"变成"部署必需品"
- **Scoring-based 框架** —— CLOVER 的蒸馏理论代表了对"损失函数设计"的深度理解，这正是工程团队需要的
- **数据 scaling 思维** —— DriveVLA-W0 的 scaling law 分析和 EponaV2 的 perception-free 范式，指向"如何经济地获取更多有效数据"

头部方法的代码和数据很少完全开源，但从论文中提取的架构设计思路是可以复现的。把这篇文章里列出的核心创新点逐个实现一遍，你的 VLA 技能栈会非常扎实。

## 📚 延伸阅读

如果想进一步深入，可以配合本博客的以下文章一起看：

- **论文精读 - DriveFuture**（2605.09701）：世界模型未来条件化的完整实现细节
- **论文精读 - ExploreVLA**（2604.02714）：不确定性驱动探索奖励与安全门控 GRPO 的工程实践
- **论文精读 - AutoDrive-P³**（2603.28116，ICLR 2026）：P³-CoT 与分层渐进式强化微调
- **产业实践 - 前沿 VLA 模型全景速览**：从产品视角横向对比主流 VLA 方案的定位与取舍
- **产业实践 - VLA 大模型训练技巧**：GRPO、蒸馏、数据 scaling 的实操经验

数据截至 2026 年 7 月，排行榜仍在快速变化。本文所有分数建议以官方 HuggingFace 排行榜为准，arXiv 宣称结果仅供参考。
