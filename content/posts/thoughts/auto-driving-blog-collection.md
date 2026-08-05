---
title: "自动驾驶优质博客收藏夹｜随手存，少逛，逛精"
date: 2026-08-05
draft: false
categories: ["个人思考"]
tags: ["📌 收藏夹", "🚗 自动驾驶", "🔗 博客链接", "📚 学习资源"]
weight: 3
summary: "把平时刷到的优秀自动驾驶博客/资源链接集中存到这里，避免日后找不到。主打一个'少逛、逛精'——每篇能持续产出高质量内容才放进收藏。持续更新。"
---

> 目的很朴素：**防止好链接弄丢**。看到哪篇长期高质量的博客/专栏/资源，就记到这里。每一条都是"值得反复回来看"的那种，不是流水账资讯。

## 📍 怎么用这份收藏夹

- 每个条目记 3 件事：**平台 + 作者 + 为什么值得回访**
- 越到后面越是"我真正会回头点开"的，而非"刷到过"的
- 发现新的好博客，直接在文末"待补充"里往上加即可

---

## ⭐ 个人博客 / 学习站点

### 1. Digtime · Corwien —— 自动驾驶实战课程体系
- **链接**：https://digtime.cn/users/1
- **专栏**：`AI-Learning`（自动驾驶学习）
- **为什么收藏**：罕见地按**课程章节式**组织内容，而非散篇。有完整"从零到端到端自动驾驶实战"系列（CARLA/ROS2 环境搭建、模仿学习、BEV 感知、VLA 强化学习微调、ROSMASTER-X3 小车实战）。它的论文地图把 14 篇论文串成"四层金字塔"，非常适合建立自动驾驶全局地图。**VLA-GRPO 强化学习微调那篇讲 GRPO 原理非常通俗**，和我的 Flow-GRPO 精读互为补充。
- **可借鉴**：课程化章节体系 + "开头先给全局视图再展开"的讲法。

### 2. 智能汽车人 · CSDN —— 行业资深算法工程师
- **链接**：https://blog.csdn.net/janeiskangs
- **专栏**：自动驾驶 Planning 决策规划、感知&端到端大模型、聊聊自动驾驶技术
- **为什么收藏**：作者是自动驾驶大厂资深算法工程师，专栏覆盖决策规划、端到端大模型、行业研究。更新的"自动驾驶大模型"系列（UniUGP、FastDriveVLA、RAD 等）紧跟前沿，且带工程视角。
- **适合**：追行业前沿工作 + 了解量产/工程落地角度。

### 3. 地平线智能驾驶开发者 · 博客园
- **链接**：https://www.cnblogs.com/horizondeveloper
- **为什么收藏**：地平线官方开发者社区，VLA / 端到端 / 世界模型等方向的官方技术解读，工程味道浓，适合看"量产方案怎么落地"。

---

## 🔬 论文 & 项目官方资源

> 这些不是博客，但都是**会反复回来查的权威一手来源**，一并存着。

### Flow-GRPO（我在精读的仓库）
- Paper：https://arxiv.org/abs/2505.05470
- Code：https://github.com/yifan123/flow_grpo
- 可视化项目页：https://gongyeliu.github.io/Flow-GRPO/
- 在线 Demo：https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO

### 论文速读工具
- ar5iv / arXiv HTML 版：把 PDF 变可读网页，`https://ar5iv.labs.arxiv.org/html/<arxiv_id>`
- AlphaXiv：https://www.alphaxiv.org （带大纲/音频的论文阅读）

---

## 🧭 我的博客内部导航

这份收藏夹不是孤立的，配某几篇"地图型"文章一起用效果更好：

- [自动驾驶技术全景学习指南｜六范式](</posts/thoughts/autonomous-driving-learning-guide/>) —— 我自己做的全景知识体系
- [自动驾驶学习路径总目录](</posts/thoughts/auto-driving-learning-path/>) —— 用我自己的论文精读串起来的金字塔路径
- [Flow-GRPO 完全讲解](</posts/thoughts/flow-grpo-complete-guide/>) —— 训练/推理/梯度流逐行拆解

---

## 📝 待补充（发现好博客就往这里加）

- [ ] 世界模型方向的独立技术博客
- [ ] VLA 结合世界模型/仿真闭环的一手工程博客
- [ ] 端到端 + 强化学习（GRPO 进入驾驶落地）的最新实践

> 更新原则：**宁缺毋滥**。只收"值得二刷"的，流水线资讯不进收藏夹。