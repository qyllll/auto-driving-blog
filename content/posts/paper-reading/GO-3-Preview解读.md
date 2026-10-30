---
title: "解读智元 GO-3 Preview：WRAM 统一「推理-预测-动作」，全链路规模化学习登顶三榜——创新日全信息发布与 GO 家族三代演进（附现场图与便利店演示 GIF）"
date: 2026-10-30
draft: false
categories: ["论文精读", "具身智能"]
tags: ["🤖 具身基座", "🧠 具身智能", "🏭 智元AGIBOT", "📊 榜单", "📖 发布解读"]
summary: "智元（AGIBOT）2026-10-29 创新日发布新一代具身基座模型 GO-3 Preview——截至发文**尚无 arXiv 技术报告**，本文基于 PR Newswire 官方通稿、RAN 行业媒体现场报道、36氪/网易快讯与官方 GO 家族公开资料做全景解读。核心：GO-3 Preview 围绕 **WRAM（World Reasoning Action Model，世界-推理-动作模型）**统一推理、预测与动作，能预判动作后果、适应环境变化、泛化到未学任务序列；演示零/少样本能力；提出 **Physical Intelligence Scaling**（预训练+后训练+部署后持续改进的全链路规模化）——约 **90% 预训练数据来自人本第一视角视频**（HTX/marsbit 技术解读），后训系统 **Genie Evolver** 融合真机、物理仿真与世界模型；桌面冰壶任务数小时练习后胜过普通人类玩家，LEGO 少样本组装、跨本体操作、数百任务开箱。**榜单宣称（活动口径）**：RoboDojo 任务成功率与过程分双第一、RoboTwin 总分第一、RoboColiseum 总分第一。血缘：GO-1（ViLLA，AgiBot World Colosseo，arXiv 2503.06669，1M+轨迹开源）→ GO-2（动作链思维）→ GO-3 Preview（全链路规模化）。开源计划：**2026 年底全面开源 GO-3 Preview 与具身 Agent 框架 Genie Master**（后者宣称首次真机验证 Physical RSI 递归自进化）。同场发布 Worken（Work+Token 生产力度量）、AHA3 硬件平台、AGILE 运动智能、WITA-Omni 交互模型、G2 Max、Link-U OS 1.0。收录现场图 8 张 + GO-3 便利店任务演示 GIF（50 帧）+ GO-1 血缘 teaser，逐图讲解。"
weight: 90
---

![智元创新日 2026 现场全景（PR/RAN 报道图）](/images/go3/pano.png)

**配图说明**：本文**收录 9 件媒体素材**（`/images/go3/`）：创新日全景、Worken 发布、AHA3 硬件、姚卯青数据飞轮讲解、**GO-3 便利店任务演示 GIF**、PR 通稿现场三连、GO-1 血缘 teaser——全部来自 [PR Newswire 通稿](https://www.prnewswire.com/news-releases/agibot-introduces-worken-to-measure-productive-robot-work-at-agibot-innovation-day-2026-302923486.html) 与 [Robotics & Automation News 现场报道](https://roboticsandautomationnews.com/2026/10/29/agibot-introduces-worken-to-measure-productive-robot-work-at-agibot-innovation-day-2026/106452/)，逐图讲解。**性质声明**：GO-3 Preview 发布于**发文当日**（2026-10-29 创新日，上海），**目前无 arXiv 论文/技术报告、无 GitHub/HF 仓库**（智元计划 2026 年底开源 GO-3 Preview + Genie Master）——本文是基于官方通稿与权威行业报道的**发布解读**，数字均标注来源；榜单成绩为**厂商活动宣称**。同主题深读：[RoboDojo 精读](/posts/paper-reading/RoboDojo精读/)、[具身仿真 Benchmark 全精读](/posts/paper-reading/具身仿真Benchmark全精读/)。

---

## 📰 一句话与五个关键词

**一句话**：GO-3 Preview = 智元第三代具身基座模型——用 **WRAM** 把「想、预测、做」装进一个模型，用**全链路规模化**（预训练→后训→部署后持续改进）把学习拉通机器人全生命周期，发布即宣称**三榜第一**。

| 关键词 | 含义 |
| --- | --- |
| **WRAM**（World Reasoning Action Model） | 统一「推理-预测-动作」框架：预判动作后果、适应环境变化、泛化到未学任务序列 |
| **Physical Intelligence Scaling** | 预训练 + 后训练 + **部署后持续改进**的全链路规模化学习 |
| **Genie Evolver** | 分布式混合后训系统：**真机 + 物理仿真 + 世界模型**三源环境 |
| **~90% 人本第一视角** | 预训练数据构成（技术媒体解读口径）；人观→机动作的迁移是核心难点 |
| **Genie Master**（年底开源） | 具身 Agent 框架：真实作业中组织长程任务、沉淀现场经验；宣称首次真机验证 **Physical RSI**（递归自进化） |

---

## 🧭 发布全景：不止一个模型

2026-10-29 智元创新日（上海）的完整菜单——GO-3 Preview 只是其中一块：

| 发布 | 是什么 |
| --- | --- |
| **Worken**（彭志辉/稚晖君发布） | Work+Token：**有效智能工作小时 = 活跃机器人数 × 自主工作时长 × 产出效率比**——从「卖了多少台」转向「干了多少活」 |
| **AHA3** 硬件平台 | 第三代人形硬件架构、首个商用「三智一体」平台：电机驱动/集成算力/Wi-Fi 6+5G/联合液冷/IP67 关节；灵巧手家族 OmniHand 3 Lite/3/Ultra + OmniPicker 3 触觉夹爪 |
| **AGILE** 运动智能 | 多模态端到端运动基础模型 + 生成式运动控制 + 新增**自进化 Agent 能力**——从「运动控制」走向「空间移动智能」 |
| **GO-3 Preview** | 本篇主角（见下文） |
| **WITA-Omni 1.0** | 全模态交互基座：视觉+语音+实时对话+表情+动作；全双工；宣称 Daily-Omni 等交互榜第一 |
| **G2 Max** | 重载工业机器人：21 自由度、双臂 40kg、工作高度 0.1–2.5m、力控+全向移动 |
| **Link-U OS 1.0** | 宣称行业首个具身智能专用 OS；捐赠给开放原子基金会，**2026 年底全面开源** |
| **MEgo 手套 / Genie Sim 3.0 / AGIBOT WORLD 2026** | 触觉数据采集手套（Air/Ultra）、仿真、数据生态扩充 |
| **开源承诺** | 36氪/网易：姚卯青宣布 **2026 年底全面开源 GO-3 Preview 与 Genie Master** |

![Worken 发布：稚晖君（彭志辉）在创新日（报道图）](/images/go3/worken.png)

（Worken 发布现场。公式三因子回应的是「2 万台下线（2026-09 宣布）之后，机器人到底干了多少活」之问——与 RoboDojo 的「过程分」、[RoboCurve](/posts/paper-reading/具身仿真Benchmark全精读/) 的真机审计同属「从分数到工时」的度量转向。）

---

## 🧬 GO 家族三代：从 ViLLA 到 WRAM

![GO-1 / AgiBot World Colosseo 血缘 teaser（论文 2503.06669）](/images/go3/go1_teaser.png)

- **GO-1（2025-03）**：**ViLLA**（Vision-Language-Latent-Action）——VLM（互联网规模异构数据）+ MoE（**Latent Planner** 学跨本体/人操作的通用动作理解 + **Action Expert** 吃 100 万+真实演示做高频灵巧操作）；载于 **AgiBot World Colosseo**（arXiv [2503.06669](https://arxiv.org/abs/2503.06669)，IROS 2025 最佳论文入围、IEEE TRO 2026）：1M+ 轨迹 / 217 任务 / 100+ 同构机器人采集，**数据集与模型全开源**（[GitHub OpenDriveLab/AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World)、HF `agibot-world/GO-1`，InternVL2.5-2B 底座）——上图即该论文 teaser（数据平台+GO-1+开源生态一图流）；
- **GO-2（2025）**：主打「**动作链思维（action chain of thought）**」——在 GO-1 的潜规划上把动作推理显式成链（公开技术细节少于 GO-1，本文从简）；
- **GO-3 Preview（2026-10）**：**全链路规模化**——不再只是「预训练更强」，而是把**部署后的真实作业反馈**接回训练环（WRAM + Genie Evolver + Genie Master），宣称零/少样本、数百任务开箱、跨本体操作。

**三代一句话**：GO-1 解决「**怎么吃异构数据**」（ViLLA 潜动作）；GO-2 解决「**怎么想动作**」（动作链思维）；GO-3 Preview 解决「**怎么让部署反过来训练模型**」（全链路规模化 + Physical RSI）。

---

## 🔬 技术要点解读（基于官方口径 + 技术媒体拆解）

1. **WRAM 三件事统一**：世界模型负责**预测**（动作会把场景带向何种状态）——这是与 GO-2「只推理动作链」的代差；推理与动作在统一框架里互相条件化。官方叙事：**预判后果 → 适应变化 → 越出已学序列**；
2. **数据结构**：预训练约 **90% 人本第一视角视频**（HTX/marsbit 解读）——把「人怎么干活」的观察大规模灌入，再解决**人观→机动作**的对齐（智元既有的潜动作/跨本体路线的延续）；机器人真机数据与 MEgo 触觉数据作为接地层；
3. **Genie Evolver 混合后训**：真机 + 物理仿真 + 世界模型**三源环境**的分布式后训系统——与站内 [具身 RL 调研](/posts/paper-reading/具身智能强化学习调研精读/)里「离线-仿真-真机三级火箭」同构，但智元把它产品化成持续运行的基础设施；
4. **Physical RSI（递归自进化）**：Genie Master 在真实作业中组织长程任务并沉淀经验——宣称**首次在真实场景验证具身递归自进化**（软件工程 RSI 的机器人版：部署→反思→更新→再部署）；配套案例：**桌面冰壶**任务在真机+世界模型中练习数小时后，胜率超过普通人类玩家；
5. **少样本与跨本体**：LEGO 组装演示——少量演示微调后完成抓取-对准-插入-连续组装（Agent 编排）；并展示跨本体操纵能力（细节待技术报告）。

![姚卯青：MEgo 手套 + GO-3 Preview + 部署反馈构成数据-模型-机器人-场景飞轮（报道图）](/images/go3/yao.png)

（姚卯青（具身业务体系总裁）讲解学习飞轮：数据采集（MEgo 触觉手套）→ 模型（GO-3）→ 机器人（AHA3/G2 系）→ 场景（工厂/商超/轨交/文旅）→ **部署反馈回流**——这就是「全链路规模化」的组织形态。）

![AHA3 硬件组件展示（报道图）](/images/go3/aha3.png)

（AHA3 硬件组件特写：三智一体的物理底座——模型再强也要落在电机、算力、散热、防护等级上；GO-3 Preview 的部署故事与 AHA3/G2 Max 绑定宣讲。）

---

## 🏆 榜单宣称：三榜第一（活动口径，务必带日期）

| 榜单 | 宣称成绩 | 口径注意 |
| --- | --- | --- |
| **RoboDojo** | **任务成功率与过程分双第一** | 「过程分」= progress score；与站内 [RoboDojo 精读](/posts/paper-reading/RoboDojo精读/) 的 score/SR 双指标一致——但**本解读未能核对具体数字与快照日期**，榜单持续更新（7 月论文快照最佳仅 8.8% SR，10 月 agent 系已到 40%+ 区间），引用请等官方榜单条目 |
| **RoboTwin** | **总分第一** | 指 RoboTwin 2.0 榜（Full/C2R 轨道见 [Benchmark 全精读](/posts/paper-reading/具身仿真Benchmark全精读/) 第 09 章） |
| **RoboColiseum** | **总分第一** | 新一代竞技型评测（本站尚未收录专文，后补） |

**读榜纪律**：以上均为**发布会宣称**——无技术报告数字表、无第三方核验链接；与 OpenWAM-α 精读里「全基线同框柱图」式的披露标准差距明显，**采信前建议等待榜单页面更新或开源后复测**。

---

## 🖼️ 图集与演示

### 演示：GO-3 便利店任务（官方演示 GIF，50 帧）

![GO-3 Preview 便利店任务演示（RAN 报道转载）](/images/go3/store.gif)

发布日报道转载的官方演示：机器人在**便利店**场景中执行 GO-3 Preview 驱动的任务（推理-预测-动作闭环）。配套背景：智元宣布将在上海便利店开展**24 小时连续作业直播**（应对异常与持续进化）——便利店成为「Physical Intelligence Scaling」的公开考场。

### 现场三连（PR 通稿图）

![创新日现场 1（PR 图）](/images/go3/event1.png)

![创新日现场 2（PR 图）](/images/go3/event2.png)

![创新日现场 3（PR 图）](/images/go3/event3.png)

主论坛与产品展示现场：Worken 概念发布、AHA3/G2 Max 硬件、真实部署案例（龙旗智造平板产线、半导体封测、CNC 上下料、广州地铁、长隆文旅导览问答等——通稿点名的场景清单）。

---

## 🧩 生态与开源时间线

| 时间 | 事项 |
| --- | --- |
| 2026-10-29 | 创新日发布 GO-3 Preview / Worken / AHA3 / WITA-Omni 1.0 / G2 Max / Link-U OS 1.0 |
| 近期 | 上海便利店 24h 直播（连续作业+异常应对+持续进化验证） |
| **2026 年底** | **GO-3 Preview + Genie Master 全面开源**（姚卯青，36氪/网易）；Link-U OS 全面开源并已捐赠开放原子基金会 |
| 生态规模 | 255 高校、511 企业、8 万+ 开发者（通稿口径） |

---

## 💡 评析：三个看点，三条待验证

### 三个看点

1. **「全链路」是产品化的 scaling law**：预训练 scaling（GO-1 数据）→ 后训 scaling（Genie Evolver 三源环境）→ **部署 scaling**（现场反馈回流 + Physical RSI）——把 RL/自进化调研里的「从经验学习」（[π\*0.6 RECAP](/posts/paper-reading/π0.6与πstar0.6精读/)、FIND）做成了公司级流水线；
2. **度量自觉**：Worken 把「机器人生产力」定义成可审计公式（活跃数×自主时长×产出效率比）——与 RoboDojo 过程分、RoboCurve 真机审计同属行业「从演示视频走向可核验工时」的共识运动；
3. **开源承诺分量足**：若年底兑现「GO-3 Preview + Genie Master 双开源」，将是继 GO-1（Colosseo）之后具身基座开源的又一记重锤——**开源时间表本身已是公开承诺**。

### 三条待验证

1. **无技术报告**：WRAM 架构细节、数据配比、Genie Evolver 的算法内核全部待论文/开源——目前只能按通稿与媒体拆解读；
2. **榜单宣称待核**：三榜第一无数字表与快照日期；RoboDojo 类榜单「周更」环境下，宣称与条目之间常有时间差；
3. **Preview 的含义**：顾名思义是预览版——能力边界、失败模式、与 GO-2 的量化对比都未披露，冰壶胜过「普通人类玩家」的实验设计（人数/局数/让子）也未公开。

---

## ❓ 自测 Q&A

**Q1：WRAM 和别家「世界模型+VLA」差在哪？**  
公开口径上 WRAM 强调**推理-预测-动作在一个框架内互相条件化**，且服务的是**部署后持续改进**的闭环（Genie Evolver/Genie Master），而不是单点的视频预测或动作生成——具体差异待技术报告。

**Q2：为什么预训练 90% 是人本视频？**  
人第一视角视频在「干活过程」上几乎无限且便宜；难题在把人的观察对齐到机器人的动作空间——这正是 GO-1 潜动作（ViLLA）路线的延伸，GO-3 叠加 WRAM 的预测能力做迁移桥梁。

**Q3：Physical RSI 是什么？**  
Recursive Self-Improvement 的具身版：机器人在真实作业中把经验沉淀回模型/Agent 框架，形成「越用越强」的循环——智元宣称 Genie Master 完成**首次真机场景验证**；细节与安全边界待开源后检验。

**Q4：Worken 和 RoboDojo 的过程分什么关系？**  
RoboDojo 过程分量「单次任务内的完成度」；Worken 量「整个机队的累计有效产出」——前者是考试分，后者是工资条；两者共同指向「别只看发布会演示」。

**Q5：现在能用上 GO-3 Preview 吗？**  
不能直接下载（无 HF/GitHub）；智元自家部署（便利店直播、工厂产线）先行；**2026 年底开源**承诺兑现后可复测——届时本篇将按开源版更新。

**Q6：和站内其他具身基座精读怎么串读？**  
血统读 GO-1（Colosseo，开源）；RL 路线读 [π\*0.6](/posts/paper-reading/π0.6与πstar0.6精读/)/[具身 RL 调研](/posts/paper-reading/具身智能强化学习调研精读/)；榜单语境读 [RoboDojo](/posts/paper-reading/RoboDojo精读/) 与 [Benchmark 全精读](/posts/paper-reading/具身仿真Benchmark全精读/)。

---

## 📊 图集总索引（9 件）

| 图 | 内容 | 来源 |
| --- | --- | --- |
| 全景（头图） | 创新日现场全景 | RAN 报道 |
| Worken 发布 | 稚晖君发布 Worken | RAN 报道 |
| 飞轮讲解 | 姚卯青数据-模型-机器人-场景闭环 | RAN 报道 |
| AHA3 硬件 | 硬件组件展示 | RAN 报道 |
| **便利店 GIF** | GO-3 Preview 任务演示（50 帧） | RAN 转载官方 |
| 现场 1–3 | 主论坛/展示/案例 | PR 通稿 |
| GO-1 teaser | 血缘论文（Colosseo）总览图 | arXiv 2503.06669 |

---

## 📚 资源与延伸

- **原始信源**：[PR Newswire 通稿](https://www.prnewswire.com/news-releases/agibot-introduces-worken-to-measure-productive-robot-work-at-agibot-innovation-day-2026-302923486.html) · [RAN 现场报道](https://roboticsandautomationnews.com/2026/10/29/agibot-introduces-worken-to-measure-productive-robot-work-at-agibot-innovation-day-2026/106452/) · [36氪 开源快讯](https://www.36kr.com/newsflashes/4046405678961798) · [HTX/marsbit 技术解读](http://www.htx.com/en-us/news/524273/)
- **GO-1 血缘（已开源）**：AgiBot World Colosseo arXiv [2503.06669](https://arxiv.org/abs/2503.06669) · [GitHub OpenDriveLab/AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World) · [HF GO-1](https://huggingface.co/agibot-world/GO-1)
- **榜单语境**：[RoboDojo 精读](/posts/paper-reading/RoboDojo精读/) · [具身仿真 Benchmark 全精读](/posts/paper-reading/具身仿真Benchmark全精读/) · [RoboTwin 2.0 章节](/posts/paper-reading/具身仿真Benchmark全精读/)
- **RL 语境**：[π\*0.6 精读](/posts/paper-reading/π0.6与πstar0.6精读/) · [具身智能强化学习调研精读](/posts/paper-reading/具身智能强化学习调研精读/)
