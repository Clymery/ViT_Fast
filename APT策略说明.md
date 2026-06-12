# APT 策略说明

## 原有两个方案

### 1. Entropy Patch Selection

方法：对每个 `16x16` patch 计算灰度熵，熵值高于阈值的 patch 保留，低熵 patch 直接丢弃，再把保留下来的 token 输入 ViT。

不足：

- 低熵不等于无用。背景、轮廓、颜色块等信息被直接丢弃后无法恢复。
- 熵是手工统计特征，不理解分类目标，容易误删对类别判断有用的 patch。
- 阈值对数据集敏感；在 CIFAR-100 这类低分辨率图像放大后的场景，熵差异不稳定。

### 2. Fixed 16/32 Patch Merge

方法：对 `32x32` 区域计算熵；低熵区域把 4 个 `16x16` token 平均合并成 1 个粗 token，高熵区域保留 4 个细 token。

不足：

- 相比 selection 不再完全丢信息，但平均池化过于粗糙，会抹掉局部细节和 token 间差异。
- 固定合并方式没有学习能力，无法根据任务自动判断 4 个子 patch 的重要性。
- 阈值偏激时 token 压缩率很高，但精度损失也明显；历史 CIFAR-100 结果约为 73/196 tokens，Val Acc 83.13%，相对 baseline -8.56%。

## 为什么优化为 Hierarchical 16/32 Learned APT

新方案保留 APT 的核心思想：高信息区域用细粒度 `16x16` token，低信息区域用粗粒度 `32x32` token。但它针对旧方案做了三点改进：

- 不再直接丢弃低熵区域，而是用粗 token 保留其结构信息。
- 不再用简单平均合并，而是用轻量 learned aggregator 对 4 个子 token 加权聚合，让模型学习哪些局部更重要。
- 加入 scale encoding 和 masked attention，使不同尺度 token 的位置信息、尺度信息和 padding 处理更一致。

## 实验目的

A4 Learned Hierarchical APT 的实验目标是验证：在约 75% token 预算下，是否能比旧 Fixed Merge 获得更好的精度/压缩平衡。重点观察：

- 相对 Full ViT baseline 的精度下降；
- 相对历史 Fixed Merge 的精度提升；
- 平均真实 token 数、padding 后 token 数、延迟、吞吐量和峰值显存；
- CIFAR-100 与 Oxford Pets 等不同数据集上，熵值层次化策略是否稳定。

简言之，旧 selection 太容易丢信息，旧 merge 又太粗糙；A4 的设计是在“不丢区域信息”和“允许合并方式可学习”之间取更合理的折中。

# 于是单独优化方案到apt_experiments下，进行实验验证。
