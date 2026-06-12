# GPU 训练流程及耗时估计

当前 GPU 主线只运行 A4 Learned Hierarchical APT。

## 1. 云端文件位置

使用 VS Code Remote SSH 时，训练结果保存在云服务器，不会自动出现在本机目录。

云端输出：

```text
apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/
apt_experiments/Hierarchical_16_32_Average_APT/checkpoints/  # 仅运行 A3 时产生
apt_experiments/experiments/results/
```

释放云服务器前必须下载这两个目录。

## 2. 环境检查

在云端仓库根目录运行：

```bash
python -m pip install -r apt_experiments/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须显示 `True` 和 GPU 名称。

## 3. 生成 A4 队列

```bash
python apt_experiments/scripts/generate_gpu_experiments.py
```

默认生成两个任务：

1. CIFAR-100 A4，约 75% token，100 epochs；
2. Oxford Pets A4，约 75% token，100 epochs。

检查：

```bash
cat apt_experiments/experiments/queues/manifest.json
cat apt_experiments/experiments/queues/gpu_a4.sh
```

## 4. 启动训练

推荐使用 `tmux` 防止 SSH 断开导致训练终止：

```bash
tmux new -s apt
bash apt_experiments/experiments/queues/gpu_a4.sh 2>&1 | tee apt_experiments/gpu_a4.log
```

重新连接：

```bash
tmux attach -t apt
```

训练中断后重新执行同一条队列命令即可自动续训。

## 5. 结果文件

每个任务保存：

```text
apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/<run_name>/
├── args.json
├── history.json
├── checkpoint_epoch_*.pth
├── best_model.pth
└── results.json
```

其中包含精度、token 数、延迟、吞吐量和峰值显存。

## 6. 生成最终对比表

训练结束后运行：

```bash
python apt_experiments/scripts/aggregate_experiment_results.py
```

生成：

```text
apt_experiments/experiments/results/results.csv
apt_experiments/experiments/results/results.json
apt_experiments/experiments/results/RESULTS_TABLE.md
```

汇总器会自动加入历史 Fixed Merge 参考行，因此 CIFAR-100 表格可以直接比较：

- A4 Learned Hierarchical APT；
- 原 Fixed Merge：73 tokens、83.13% Val Acc、相对 baseline -8.56%。

## 7. A3 备用命令

A3 不进入默认队列。只有需要备用实验时运行：

```bash
python apt_experiments/Hierarchical_16_32_Average_APT/train.py \
  --dataset cifar100 --gpu 0 --batch_size 16 --accum 8 \
  --epochs 100 --seed 42 --entropy_bins 64 --threshold32 3.25
```

## 8. RTX 4090 耗时估计

A4 包含逐图层次划分和 learned aggregation，当前使用 FP32。

| 任务 | 预计时间 |
|:-----|:---------|
| CIFAR-100，100 epochs | 12-22 小时 |
| Oxford Pets，100 epochs | 2-5 小时 |
| 两个默认任务合计 | 14-27 小时 |
| 结果汇总 | 少于 5 分钟 |

实际时间以 CIFAR-100 第一轮 epoch 日志为准。

若只想先确认脚本能运行：

```bash
python apt_experiments/scripts/generate_gpu_experiments.py --epochs 5
```

5-epoch 队列约需 1-3 小时，但不能作为最终结果。

## 9. 下载回本机

在本机 PowerShell 中执行：

```powershell
scp -r USER@SERVER:/workspace/ViT_Fast/apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints `
  "C:\path\to\ViT_Fast\apt_experiments\Hierarchical_16_32_Learned_APT\"

scp -r USER@SERVER:/workspace/ViT_Fast/apt_experiments/experiments/results `
  "C:\path\to\ViT_Fast\apt_experiments\experiments\"
```

也可以在 VS Code Remote SSH 文件资源管理器中右键对应目录下载。


