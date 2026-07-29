---
title: "论文精读｜iPad: Iterative Proposal-centric End-to-End Autonomous Driving"
date: 2026-07-29
draft: false
categories: ["论文精读", "自动驾驶"]
tags: ["🚗 端到端自动驾驶", "⚡ 迭代规划", "🧠 稀疏Proposal", "📊 CVPR 2025"]
weight: 28
summary: "iPad提出了一种以轨迹提案为中心的端到端自动驾驶框架，通过ProFormer迭代式精炼稀疏BEV提案，替代了传统的密集BEV网格范式，在NAVSIM和Bench2Drive上达到SOTA且计算效率提高10倍以上。"
---

## 📄 论文信息

<img src="/images/paper/2505.15111/figure1.png" alt="iPad与密集网格范式对比" style="width:100%; max-width:900px; display:block; margin:0 auto;">
**图1: 端到端范式对比**。(a)密集网格方法：对所有BEV网格单元提取特征后直接生成最终规划。(b)iPad：通过稀疏BEV提案的迭代精炼，将特征提取集中在与规划最相关的区域。

- **标题**：*iPad: Iterative Proposal-centric End-to-End Autonomous Driving*
- **团队**：南洋理工大学（Ke Guo, Haochen Liu, Chen Lv）、德赛西威（Xiaojun Wu）、香港大学（Jia Pan）
- **发表**：CVPR 2025（arXiv:2505.15111）
- **关键词**：端到端自动驾驶、迭代规划、稀疏BEV、ProFormer
- **一句话总结**：以规划提案为中心的端到端自动驾驶范式，通过迭代精炼稀疏BEV提案大幅提升效率和规划质量。
- **论文链接**：[arXiv:2505.15111](https://arxiv.org/abs/2505.15111)
- **代码仓库**：[github.com/Kguo-cs/iPad](https://github.com/Kguo-cs/iPad)

---

## 🤔 要解决什么问题？

现有的端到端自动驾驶方法大多基于密集BEV网格特征进行规划，存在两个主要问题：

**1. 计算效率低下**。密集BEV网格需要对每个网格单元提取特征，计算复杂度随分辨率呈二次增长。例如UniAD等方法需要高分辨率BEV特征来支持检测、跟踪、建图等辅助任务，计算开销巨大。

**2. 规划感知能力有限**。密集网格对所有空间位置一视同仁，大量计算浪费在与规划无关的区域上，导致因果混淆（causal confusion）——模型学会了关注不相关的场景元素，降低了规划性能。

iPad的核心洞察是：**规划应当成为整个架构的中心组织原则**，而非作为一个下游任务。通过将规划提案（proposals）置于特征提取和辅助任务的核心，可以同时提升效率和规划质量。

---

## 🏗️ 核心架构

<img src="/images/paper/2505.15111/figure2.png" alt="iPad框架总览" style="width:100%; max-width:900px; display:block; margin:0 auto;">
**图2: iPad框架总览**。包含四个核心组件：Scene Encoder（灰色）提取图像和自车特征；ProFormer（蓝色）基于自车特征初始化BEV提案查询并迭代精炼；Scorer（绿色）为每一提案轨迹打分；Proposal-Centric Mapping and Prediction（红色）预测通航能力和碰撞风险。

iPad由四个关键组件构成：

### 1. Scene Encoder（场景编码器）

处理多视角图像和自车状态。图像经过ResNet-34 backbone提取多视角特征图，自车状态（速度、加速度、转向指令）通过线性层编码为自车特征。

### 2. ProFormer（提案Transformer）

这是iPad的核心创新，一个基于BEVFormer构建的提案中心BEV编码器。ProFormer的工作流程是"预测-锚定-精炼"的迭代循环：

1. **初始化**：基于自车特征初始化若干BEV提案查询
2. **提案预测**：从查询解码出候选轨迹提案
3. **锚定注意力**：以各提案的角点作为锚点，聚合多视角图像特征
4. **查询精炼**：用聚合的特征更新提案查询
5. **迭代**：重复上述过程，逐步精炼提案和特征

这种设计相比密集BEV网格有两个关键优势：
- **复杂度线性增长**：计算量随提案数量线性增长，而非随网格分辨率平方增长
- **规划感知**：特征提取聚焦于与规划相关的区域，避免无关信息干扰

### 3. Scorer（评分器）

对ProFormer输出的所有精炼提案进行评估，为每个提案预测一个分数，选择分数最高的轨迹作为最终规划输出。

### 4. Proposal-Centric Mapping and Prediction

iPad引入了两个轻量级的、以提案为中心的辅助任务：
- **Mapping**：预测提案轨迹上的各点是否在道路/路线上（通航性判断）
- **Prediction**：预测与提案轨迹最可能发生碰撞的前两个物体的未来状态（碰撞风险预测）

相比于UniAD等方法的全场景检测、跟踪、建图等辅助任务，iPad的两个辅助任务：
- 完全以规划为中心，只关注与当前决策相关的信息
- 计算量极低，不需要高分辨率BEV特征
- 更符合人类驾驶直觉——只关注与决策直接相关的上下文

---

## 🔬 实验分析

<img src="/images/paper/2505.15111/figure3.png" alt="iPad实验分析" style="width:100%; max-width:900px; display:block; margin:0 auto;">
**图3: 实验结果概览**。iPad在NAVSIM和Bench2Drive基准上取得SOTA性能，同时计算量仅为UniAD的1/10以下。

### 基准测试表现

**NAVSIM（开环）**：基于真实世界nuPlan数据集的大规模开环评测。iPad在PDMS等核心指标上超越所有先前方法。

**Bench2Drive（闭环）**：基于CARLA的闭环评测。iPad在驾驶分数（Driving Score）和路线完成率（Route Completion）上均达到最优。

### 效率对比

iPad的计算效率优势显著：
- 相比UniAD：计算量降低**10倍以上**
- 相比VAD：计算量降低**5倍以上**
- 相比SparseDrive：计算量降低**2倍以上**

这种效率优势来源于稀疏提案设计——iPad不需要生成和计算密集BEV网格特征。

### 消融实验

| 组件 | PDMS | 说明 |
|------|------|------|
| 完整模型 | 93.2 | - |
| 移除迭代精炼 | 91.5 | 单次预测性能下降 |
| 移除辅助任务 | 91.8 | 辅助任务提升规划质量 |
| 替代为密集BEV | 91.0 | 密集网格反而降低性能 |

---

## 💡 个人思考

### 创新点

1. **范式转换**：从"密集BEV → 规划"到"规划提案 → 特征提取"的范式转换，将规划从下游任务变为架构的组织中心，这一思路启发性很强。

2. **极简辅助任务**：现有方法（如UniAD）的辅助任务堆砌了大量全场景感知任务，计算昂贵且与规划脱节。iPad的两个轻量级提案中心辅助任务精准定位了与规划最相关的信息，实现了"少即是多"。

3. **迭代精炼机制**：通过"预测-锚定-精炼"的迭代循环实现了规划与特征提取的联合优化，类似于DETR的迭代精炼思想在自动驾驶规划中的成功应用。

### 局限性

1. **开环评测局限**：NAVSIM的评测仍然是基于数据驱动的开环评测（非反应式），与真实闭环性能存在差距。

2. **多模态不确定性建模**：虽然iPad通过多提案实现了一定的多模态规划，但与扩散策略等方法相比，对复杂多模态分布的表达能力仍有差距。

### 延伸思考

iPad的方向代表了端到端自动驾驶从"全感知-全预测-规划"的繁重范式向"以规划为中心"的简洁范式转变的趋势。与DrivoR（通过寄存器token压缩）、DiffusionDrive（扩散规划）等工作一起，共同推动了端到端自动驾驶的轻量化与高效化发展。

---

## 📖 延伸阅读

1. **UniAD: Planning-oriented Autonomous Driving**（Hu et al., CVPR 2023）- 首个全栈端到端自动驾驶框架，CVPR 2023最佳论文
2. **VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning**（Chen et al., 2024）- 基于评分规划的概率规划方法
3. **DrivoR: Driving on Registers**（Kirby et al., CVPR 2026）- 利用ViT寄存器token进行高效端到端驾驶
4. **DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**（Liao et al., CVPR 2025）- 截断扩散模型用于规划
