---
title: "Flow-GRPO 源码学习与策略复现"
date: 2026-07-20
draft: false
summary: "基于 Flow-GRPO 的自动驾驶策略源码学习与复现记录，含小白友好的阅读顺序与教学注释。"
tags: ["项目", "Flow-GRPO", "强化学习", "端到端"]
---

## 项目简介

Flow-GRPO 是将 **Flow Matching** 与 **GRPO 强化学习**结合用于自动驾驶策略训练的工作。本项目聚焦于**读通源码、跑通流程、理解稀疏奖励下的策略对齐**。

## 我做了什么

- 整理了一份**小白友好**的源码阅读顺序，并在关键文件加入教学注释，帮助快速建立整体认知。
- 梳理了从环境交互、轨迹采样、奖励计算到 GRPO 更新的完整数据流。
- 记录了训练启动、显存/采样步数等工程踩坑与对应解法。

## 成果

- 产出两篇源码学习笔记：总体架构笔记 + 小白版阅读顺序（带注释）。
- 形成一份可复用的「Flow-GRPO 跑通 checklist」。

## 相关链接

- GitHub: [github.com/qyllll](https://github.com/qyllll)
- 配套笔记：[Flow-GRPO 源码学习笔记](/posts/code/)、[小白版源码学习教程](/posts/code/)
