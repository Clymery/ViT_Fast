$ErrorActionPreference = 'Stop'
python apt_experiments/Hierarchical_16_32_Learned_APT/train.py --dataset cifar100 --gpu 0 --batch_size 16 --accum 8 --epochs 100 --seed 42 --entropy_bins 64 --threshold32 3.25
python apt_experiments/Hierarchical_16_32_Learned_APT/train.py --dataset oxford_pets --gpu 0 --batch_size 16 --accum 8 --epochs 100 --seed 42 --entropy_bins 64 --threshold32 4.0
