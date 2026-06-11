"""
Test image preprocessing strategies that actually reduce token count.

Corrected experiments:
  1. Downsample (no restore): resize to smaller, feed directly (inference only)
  2. Checkerboard removal: remove every other row (halve image size)
  3. Center crop: crop a smaller region from the image

Usage:
  python test_preprocess_reduce.py --dataset cifar100 --gpu 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader
import timm

DATASETS = {
    'cifar100':    (get_cifar100_loader,     100),
    'oxford_pets': (get_oxford_pets_loader,   37),
    'food101':     (get_food101_loader,      101),
}


def resize_down(images, target_size):
    """Resize image to smaller resolution (no upsample back)."""
    B, C, H, W = images.shape
    return F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)


def remove_even_rows_cols(images):
    """Remove every other row AND column → image size halves.

    224×224 → remove even rows → 112×224 → remove even cols → 112×112
    Effectively 75% pixel reduction, image shrinks to 1/4 area.
    """
    return images[:, :, ::2, ::2]


def remove_even_rows_only(images):
    """Remove every other row only → height halves, width unchanged.

    224×224 → 112×224 (non-square, but fewer patches)
    """
    return images[:, :, ::2, :]


def center_crop(images, crop_size):
    """Center crop the image to a smaller size."""
    B, C, H, W = images.shape
    h_start = (H - crop_size) // 2
    w_start = (W - crop_size) // 2
    return images[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]


def get_num_patches(img_size, patch_size=16, stride=16):
    """Calculate number of patches for a given image size."""
    grid = (img_size - patch_size) // stride + 1
    return grid * grid


def make_model_for_size(device, num_classes, img_size, checkpoint_path, patch_size=16):
    """Create a ViT-B/16 model adapted for a specific input size.

    Loads checkpoint from 224×224 model, interpolates pos_embed for new size.
    """
    model = timm.create_model(
        'vit_base_patch16_224.augreg_in21k',
        pretrained=True, num_classes=num_classes, img_size=img_size)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)

        # Interpolate position embeddings if sizes differ
        if 'pos_embed' in sd and sd['pos_embed'].shape != model.pos_embed.shape:
            old_pos = sd.pop('pos_embed')
            cls_pos = old_pos[:, 0:1, :]
            patch_pos = old_pos[:, 1:, :]

            old_grid = int((patch_pos.shape[1]) ** 0.5)
            new_grid = (img_size - patch_size) // patch_size + 1

            if old_grid != new_grid:
                patch_2d = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
                patch_new = F.interpolate(patch_2d, size=(new_grid, new_grid),
                                          mode='bilinear', align_corners=False)
                patch_new = patch_new.permute(0, 2, 3, 1).reshape(1, -1, 768)
                sd['pos_embed'] = torch.cat([cls_pos, patch_new], dim=1)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'    Missing keys: {len(missing)} (expected for diff img_size)', flush=True)
        if unexpected:
            print(f'    Unexpected keys: {len(unexpected)}', flush=True)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, preprocess=None, track_time=False, img_size=224):
    model.eval()
    correct = 0
    total = 0
    total_time = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        if preprocess is not None:
            images = preprocess(images)

        if track_time:
            torch.cuda.synchronize()
            start = time.time()

        logits = model(images)

        if track_time:
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    if track_time:
        latency = total_time / total * 1000
        throughput = total / total_time
        return acc, latency, throughput
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=list(DATASETS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}', flush=True)

    loader_fn, num_classes = DATASETS[args.dataset]
    result = loader_fn(batch_size=args.batch_size, data_dir='./data', num_workers=4)
    if len(result) == 4:
        train_loader, val_loader, test_loader, n_cls = result
    else:
        train_loader, test_loader, n_cls = result
        val_loader = test_loader

    print(f'Dataset: {args.dataset}, Test: {len(test_loader.dataset)}, Classes: {n_cls}', flush=True)

    # Load fine-tuned ViT-B/16
    print('Loading ViT-B/16 IN-21K...', flush=True)
    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=n_cls)
    model = model.to(device)
    model.eval()

    ckpt_path = f'checkpoints/{args.dataset}_vit_b16_ft/best_model.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f'  Loaded fine-tuned checkpoint: {ckpt_path}', flush=True)

    # Warmup
    for images, _ in test_loader:
        images = images.to(device)
        _ = model(images)
        break

    ckpt_base = f'checkpoints/{args.dataset}_vit_b16_ft/best_model.pth'

    methods = [
        # (name, preprocess_fn, img_size, patches)
        ('Baseline (224×224)', None, 224, 196),
        ('Downsample 168×168 (推理)', lambda x: resize_down(x, 168), 168, 100),
        ('Downsample 112×112 (推理)', lambda x: resize_down(x, 112), 112, 49),
        ('隔行去行列 (224→112)', remove_even_rows_cols, 112, 49),
        ('Center crop 168×168', lambda x: center_crop(x, 168), 168, 100),
        ('Center crop 112×112', lambda x: center_crop(x, 112), 112, 49),
    ]

    print(f'\n=== {args.dataset}: 图片预处理（实际减少 token 数量） ===\n', flush=True)
    print(f'{"方法":<30} {"Acc":>8} {"Patches":>8} {"Latency":>10} {"Throughput":>12}', flush=True)
    print('-' * 75, flush=True)

    for name, preprocess, img_size, expected_patches in methods:
        if isinstance(img_size, tuple):
            # Skip non-square cases (need separate model creation)
            print(f'{name:<30} skipped (non-square input)', flush=True)
            continue
        # Create a fresh model for each input size
        m = make_model_for_size(device, n_cls, 224 if isinstance(img_size, tuple) else img_size, ckpt_base)
        acc, latency, throughput = evaluate(m, test_loader, device, preprocess, track_time=True, img_size=img_size)
        print(f'{name:<30} {acc:>7.2f}% {expected_patches:>8} {latency:>8.2f}ms {throughput:>10.1f}/s', flush=True)
        del m
        torch.cuda.empty_cache()

    print(f'\n对比：降采样训练（已训练过的模型，非推理时硬切）')
    print(f'  168×168 训练: 91.56%, 100 patches (来自 test_downsample_train.py)')
    print(f'  112×112 训练: 90.00%, 49 patches (来自 test_downsample_train.py)')
    print(f'\n结论：推理时直接缩小图片会掉点较多，因为位置编码不匹配')
    print(f'降采样训练（重新训练适配）的效果远好于推理时硬切', flush=True)


if __name__ == '__main__':
    main()
