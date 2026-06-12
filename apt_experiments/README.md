# APT Experiments

这是从原 ViT/MAE 工程中拆出的 APT 实验区。当前只保留两个层次化 16/32 实验包：一个主实验，一个备用对照。每个实验目录都提供独立的 `train.py`，训练、评估、checkpoint、history 和 results 写入逻辑都在同一个文件中，方便直接运行和阅读。

## 文件层级

```text
apt_experiments/
├── Hierarchical_16_32_Learned_APT/
│   ├── train.py          # A4 主实验：learned aggregator 聚合 32x32 粗 token
│   ├── README.md         # A4 方法与运行说明
│   ├── checkpoints/      # A4 训练产生的 checkpoint
│   └── results/          # A4 结果文件，可按需保存汇总表
├── Hierarchical_16_32_Average_APT/
│   ├── train.py          # A3 备用实验：average pooling 聚合 32x32 粗 token
│   ├── README.md         # A3 方法与运行说明
│   ├── checkpoints/      # A3 训练产生的 checkpoint
│   └── results/          # A3 结果文件，可按需保存汇总表
├── doc/
│   ├── APT策略说明.md
│   ├── GPU_WORKFLOW.md
│   ├── GPU训练流程及耗时估计.md
│   └── ROADMAP.md
├── scripts/
│   ├── scan_apt_thresholds.py
│   ├── generate_gpu_experiments.py
│   └── aggregate_experiment_results.py
├── experiments/
│   ├── scans/            # 阈值扫描结果
│   ├── queues/           # GPU 队列
│   └── references/       # 历史 Fixed Merge 等参考结果
├── tests/
│   └── test_apt_stage0.py
├── requirements.txt
└── __init__.py
```

## 两个实验

`Hierarchical_16_32_Learned_APT/` 是当前主实验。低熵 `32x32` 区域会合并为一个粗 token，但 4 个子 token 的聚合权重由轻量 MLP 学习得到；高熵区域保留 `16x16` 细 token。

`Hierarchical_16_32_Average_APT/` 是备用对照。区域划分与 A4 相同，但粗 token 使用简单平均池化，主要用于判断 learned aggregation 是否带来收益。

## 常用命令

所有命令从仓库根目录执行。

```bash
python apt_experiments/Hierarchical_16_32_Learned_APT/train.py --dataset cifar100 --gpu 0 --threshold32 3.25
python apt_experiments/Hierarchical_16_32_Average_APT/train.py --dataset cifar100 --gpu 0 --threshold32 3.25
python apt_experiments/scripts/scan_apt_thresholds.py --dataset cifar100
python apt_experiments/scripts/generate_gpu_experiments.py
python apt_experiments/scripts/aggregate_experiment_results.py
```
