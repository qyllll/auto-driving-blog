---
title: "自动驾驶学习路径总目录｜用我的论文精读串成的金字塔地图"
date: 2026-08-05
draft: false
categories: ["个人思考"]
tags: ["🗺️ 学习路径", "🚗 自动驾驶", "📚 论文精读", "📐 目录"]
weight: 2
summary: "参考'少读要比乱读强'的思路，把我论文精读区的文章按'地基→BEV感知→端到端→世界模型→扩散+强化学习→VLA'六层金字塔组织起来。顺着层往上走，每层解决上一层留下的问题，最后落在 VLA/世界模型这些前沿范式上。这是本站论文精读的总入口。"
---

> 这页是本站**论文精读区的总目录**。很多文章单独看都懂，但串不起来。参考学习路径的思路，我按"金字塔"把它们分层：**每一层解决上一层留下的问题**，顺着层往上读，最后正好落在 VLA / 世界模型这些前沿范式上。

## 🏔️ 先给你全局视图：六层金字塔

```text
第六层【前沿范式】 VLA / 世界模型 / 强化微调      ← 当前最热，本站的重心
第五层【生成式规划】 扩散规划 + 强化学习(GRPO)   ← AlphaDrive/DiffusionDrive/Flow-GRPO
第四层【世界模型】   DreamerV3/Cosmos/GEN-1...   ← 预测未来·做数据工厂
第三层【端到端规划】 UniAD / VAD / SparseDrive    ← 2023-2024 主战场
第二层【BEV 感知】   BEVFormer / Sparse4D         ← 行业基石
第一层【地基】       Transformer / ViT            ← 一切的源头
```

**阅读总原则**：按层往上走。跳层读会痛苦（没读 BEVFormer 直接看 VLA，感知那部分会糊涂）。每层里我标了"必读"和"进阶"，按顺序点进去即可。

---

## 📍 第 0 站：先看全景再进层

进任何层之前，先读这三份，建立坐标系：

- [自动驾驶技术全景学习指南｜六范式知识体系](/posts/thoughts/autonomous-driving-learning-guide/) ⭐ 本站总地图
- [自动驾驶感知学习指南](/posts/thoughts/autonomous-driving-perception-guide/) ⭐ BEV/感知分地图
- [自动驾驶优质博客收藏夹](/posts/thoughts/auto-driving-blog-collection/) —— 外部高质量资源

> 读完这几份，再进金字塔，就知道每篇论文"躺在哪一层、为谁服务"。

---

## 第一层：地基（Transformer / ViT）

> 不读这两块，后面全是天书——所有端到端/VLA/世界模型都建立在 Transformer 上。

- ⭐ 必读：**Attention Is All You Need (Transformer)**——注意力机制的开山作（理解"自注意力怎么并行 + 全局感受野"）
- ⭐ 必读：**ViT: An Image is Worth 16x16 Words**——图像切 patch 当 token，视觉 token 一切的源头
- 📌 我这里的底层参考：Transformer → ViT 的 patch 思想 -> 一切视觉模型输入来源

> 本站精读偏工程/前沿，地基理论靠经典论文原文补。可先用我写的 [VLA vs 感知：两种路线怎么选](/posts/thoughts/vla-vs-perception/) 建立感性认识，再回去补理论。

---

## 第二层：BEV 感知（行业基石）

> 解决"多路相机图像怎么变成统一的鸟瞰表示"。你读 VLA/V 世界模型前，感知这层必须有。

- ⭐ [BEVFormer 感知精读](/posts/paper-reading/bevformer感知精读/) —— BEV query 反向查询，Spatial Cross-Attention
- [Sparse4D 稀疏感知精读](/posts/paper-reading/sparse4d稀疏感知精读/) —— 稀疏 query 而非稠密网格
- [SparseDrive-V2 精读](/posts/paper-reading/sparsedrive-v2精读/) —— 稀疏统一感知+预测
- [SparseOccVLA 精读](/posts/paper-reading/sparseoccvla精读/) —— 稀疏占据 / VLA 结合的感知

---

## 第三层：端到端规划（2023-2024 主战场）

> 解决"感知→预测→规划各模块误差累积 / 目标不一致"。代表作是模块化端到端的巅峰。

- ⭐ [UniAD 端到端自动驾驶框架精读](/posts/paper-reading/uniad-端到端自动驾驶框架精读/) —— planning-oriented，CVPR 最佳论文
- ⭐ [VAD 向量化端到端精读](/posts/paper-reading/vad向量化端到端精读/) —— 场景矢量化，逼近实时
- [TransFuser 论文精读](/posts/paper-reading/论文精读-transfuser/) —— 多传感器融合端到端（CARLA 经典）
- [DriveTransformer 精读](/posts/paper-reading/论文精读-drivetransformer/) —— 大网络化端到端
- [EMMA-Wayve 端到端多模态精读](/posts/paper-reading/emma-wayve端到端多模态精读/) —— 多模态大模型直接出规划
- [Senna-VLM 辅助驾驶精读](/posts/paper-reading/senna-vlm辅助驾驶精读/) —— VLM 辅助端到端
- 进阶：[JEPA-DRIVE 自监督驾驶决策精读](/posts/paper-reading/jepa-drive自监督驾驶决策精读/) / [RAW2Drive 精读](/posts/paper-reading/raw2drive精读/) / [NoRD 精读](/posts/paper-reading/nord精读/)

---

## 第四层：世界模型（预测未来 + 做数据工厂）

> 解决"真车训练太危险、长尾场景太少"。用生成式模型预测/合成未来场景，既是世界模型又是闭环训练环境。

- ⭐ [Cosmos3 世界基础模型精读](/posts/paper-reading/cosmos3-世界基础模型精读/) —— NVIDA 开源世界基础模型
- [DreamerV3 世界模型精读](/posts/paper-reading/dreamerv3世界模型精读/) —— RL 决策 + 世界模型，通用智能体经典
- [DriveDreamer 世界模型精读](/posts/paper-reading/drivedreamer世界模型精读/) —— 可控驾驶场景生成（数据工厂）
- [DLWM 精读](/posts/paper-reading/dlwm精读/) / [UniSim 交互式世界模拟器精读](/posts/paper-reading/unisim交互式世界模拟器精读/)
- 我的工程向：[Cosmos3-MoE + Flow-GRPO 架构](/posts/thoughts/cosmos3-moe-flowgrpo-arch/)

---

## 第五层：生成式规划 + 强化学习（本站重心）

> 解决"模仿学习有天花板，模型最多和人类一样好"。用**扩散/Flow 生成轨迹 + 强化学习（GRPO）**突破上限。这是你读 Flow-GRPO 的落地点。

### 扩散 / Flow 生成式规划
- ⭐ [DiffusionDrive 精读](/posts/paper-reading/diffusiondrive精读/) —— 截断扩散 + anchor 先验，实时多模态规划
- [Diffusion-Planner 精读](/posts/paper-reading/diffusion-planner精读/) —— 规划当轨迹生成
- [Diffusion-Policy 精读](/posts/paper-reading/diffusion-policy精读/) —— 动作扩散，机器人/驾驶通用
- [GoalFlow 精读](/posts/paper-reading/goalflow精读/) —— Flow Matching + 目标点引导
- [TransDiffuser 论文精读](/posts/paper-reading/论文精读-2505-09315-transdiffuser/)

### 强化学习微调（GRPO 进入驾驶）
- ⭐ [AlphaDrive-GRPO 驾驶策略精读](/posts/paper-reading/alphadrive-grpo驾驶策略精读/) —— GRPO 首次引入驾驶规划，多模态规划
- [Gen-Drive 扩散强化驾驶精读](/posts/paper-reading/gen-drive扩散强化驾驶精读/) —— 扩散 + reward / RL 微调
- [AutoVLA 精读](/posts/paper-reading/autovla精读/) —— VLA + GRPO 快慢双模
- [ReCogDrive 精读](/posts/paper-reading/recogdrive精读/) —— 扩散规划器 + GRPO（DiffGRPO）
- [AlpaMayo-R1 精读](/posts/paper-reading/alpamayo-r1精读/)

### 我的深入讲解（综合）
- ⭐ [Flow-GRPO 完全讲解](/posts/thoughts/flow-grpo-complete-guide/) —— 训练/推理/梯度流逐行拆解
- [RL 与扩散轨迹规划：怎么结合](/posts/thoughts/rl-diffusion-traj/)
- [NavSim PDMS-90：闭环指标](/posts/thoughts/navsim-pdms-90/)

---

## 第六层：VLA 与具身智能（前沿范式）

> 解决"端到端网络遇到长尾场景泛化差"。用视觉-语言-动作统一模型，既能开自动驾驶车又能操作机器人。

### VLA 模型
- ⭐ [OpenVLA 开源 VLA 模型精读](/posts/paper-reading/openvla开源vla模型精读/) —— 开源落地标准起点
- [AutoVLA 精读](/posts/paper-reading/autovla精读/) / [Qwen-VLA 精读](/posts/paper-reading/qwen-vla精读/)
- [pi0 通用 VLA 模型精读](/posts/paper-reading/pi0-通用vla模型精读/)
- [One-VL 精读](/posts/paper-reading/one-vl精读/) / [Last-VLA 精读](/posts/paper-reading/last-vla精读/) / [LinkVLA 精读](/posts/paper-reading/linkvla精读/) / [DriveVLM 系列精读](/posts/paper-reading/drivevlm精读/)
- 综述：[VLA Survey 精读](/posts/paper-reading/vla-survey-2024/) ⭐ / [VLA 安全综述精读](/posts/paper-reading/vla安全综述精读/)

### 具身 / 机器人基础
- [RT-1 Robotics Transformer 精读](/posts/paper-reading/rt-1-robotics-transformer精读/) / [RT-2 精读](/posts/paper-reading/rt-2精读/)
- [PaLM-E 具身多模态语言模型精读](/posts/paper-reading/palm-e具身多模态语言模型精读/)
- [EmbodiedGPT 具身思维链精读](/posts/paper-reading/embodiedgpt具身思维链精读/)
- [SayCan 语言 grounding 精读](/posts/paper-reading/saycan语言-grounding精读/) / [VIMA 多模态 prompt 精读](/posts/paper-reading/vima-multimodal-prompts-robot-manipulation/)
- [Gato 通用智能体精读](/posts/paper-reading/gato通用智能体精读/)

### 数据 / 工程
- [RT-1 配套 Open-X-Embodiment 数据集](/posts/paper-reading/open-x-embodiment数据集精读/)
- [RH20T 拟人机器人数据集精读](/posts/paper-reading/rh20t拟人机器人数据集精读/)
- [DROID 机器人操作数据集精读](/posts/paper-reading/droid机器人操作数据集精读/)
- [LIBERO 精读](/posts/paper-reading/libero精读/) / [ACT-ALOHA 精读](/posts/paper-reading/act-aloha精读/)

---

## 🚚 商用车 / 重卡专题（按需）

工程与安全重卡方向的一批精读，按你当前项目取用：

- [AntiRollover 重型车防侧翻](/posts/paper-reading/论文精读-antirollover-apf/)
- [CargoLoad-RSS 载荷估计](/posts/paper-reading/论文精读-cargoload-rss/)
- [MANTruckScenes 卡车场景](/posts/paper-reading/论文精读-mantruckscenes/)
- [TruckDrive / TruckV2X 重卡驾驶与协同](/posts/paper-reading/论文精读-truckdrive/) · [TruckV2X](/posts/paper-reading/论文精读-truckv2x/)
- [PARA-Drive](/posts/paper-reading/论文精读-para-drive/)

---

## 🧭 建议阅读顺序（一个月路线）

| 阶段 | 读什么 | 目标 |
|---|---|---|
| 第 1 周 | 第 0 站全景 + 第二层 BEV | 建立感知坐标系 |
| 第 2 周 | 第三层 端到端（UniAD/VAD） | 懂为什么行业要端到端 |
| 第 3 周 | 第四层 世界模型 | 懂 VLA/世界模型怎么来 |
| 第 4 周 | 第五层 扩散+GRPO → **Flow-GRPO 完全讲解** | 抵达你的核心方向 |

> 更多精读文章在 [📁 论文精读分类](/categories/论文精读/) 里按时间排（上百篇）。这页是"该先读哪些"的精选骨架，那页是"全部藏货"。