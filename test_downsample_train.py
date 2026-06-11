"""
Test training ViT-B/16 with smaller input images (downsampled).

When input image is smaller, the patch embedding produces fewer patches:
  - 224×224 → 14×14 = 196 patches (baseline)
  - 168×168 → 10×10 = 100 patches (~49% reduction)
  - 112×112 → 7×7 = 49 patches (~75% reduction)

This tests whether training with smaller images can maintain accuracy
while reducing computation (fewer tokens → less attention computation).

Usage:
  python test_downsample_train.py --dataset cifar100 --image_size 112 --gpu 5
  python test_downsample_train.py --dataset cifar100 --image_size 168 --gpu 6
  python test_downsample_train.py --dataset cifar100 --image_size 224 --gpu 7  # baseline
"""
import torch
import torch.nn as nn
import time
import os
import sys
import argparse
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader, get_dtd_loader, get_flowers102_loader
import timm

DATASETS = {
    'cifar100':    (get_cifar100_loader,     100, 100),
    'oxford_pets': (get_oxford_pets_loader,   37, 100),
    'food101':     (get_food101_loader,      101,  30),
    'dtd':         (get_dtd_loader,           47, 100),
    'flowers102':  (get_flowers102_loader,   102, 100),
}

BASELINE_ACC = {
    'cifar100':   91.69,
    'oxford_pets': 93.81,
    'food101':    91.37,
    'dtd':        80.85,
    'flowers102': 100.00,
}


def adjust_position_embeddings(model, new_image_size, patch_size=16):
    """Adjust position embeddings for different input sizes.

    ViT-B/16 has pos_embed of shape (1, 197, 768) for 224×224 input.
    For smaller inputs, we interpolate the patch position embeddings.
    """
    old_pos_embed = model.pos_embed.data  # (1, 1+N_old, 768)
    cls_pos = old_pos_embed[:, 0:1, :]  # (1, 1, 768)
    patch_pos = old_pos_embed[:, 1:, :]  # (1, N_old, 768)

    # Compute old and new grid sizes
    old_size = int(math.sqrt(patch_pos.shape[1]))  # 14 for 224×224
    new_size = (new_image_size - patch_size) // patch_size + 1  # e.g., 7 for 112×112

    if old_size == new_size:
        return  # No change needed

    # Reshape to 2D and interpolate
    patch_pos_2d = patch_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)  # (1, 768, 14, 14)
    patch_pos_new = torch.nn.functional.interpolate(
        patch_pos_2d, size=(new_size, new_size), mode='bilinear', align_corners=False
    )
    patch_pos_new = patch_pos_new.permute(0, 2, 3, 1).reshape(1, -1, 768)  # (1, N_new, 768)

    new_pos_embed = torch.cat([cls_pos, patch_pos_new], dim=1)
    model.pos_embed = nn.Parameter(new_pos_embed)

    return new_size


def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def compute_efficiency_metrics(model, loader, device):
    """Measure latency (ms/sample) and throughput (samples/sec)."""
    model.eval()
    # Warmup
    for images, _ in loader:
        images = images.to(device)
        _ = model(images)
        break

    total_time = 0
    total_samples = 0
    for images, _ in loader:
        images = images.to(device)
        batch_size = images.size(0)

        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.time()

        _ = model(images)

        if device != 'cpu':
            torch.cuda.synchronize()
        end = time.time()

        total_time += (end - start)
        total_samples += batch_size

    latency = total_time / total_samples * 1000  # ms per sample
    throughput = total_samples / total_time  # samples/sec
    return latency, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--image_size', type=int, required=True,
                        help='Input image size (e.g., 112, 168, 224)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override default epochs for dataset')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}', flush=True)

    loader_fn, num_classes, default_epochs = DATASETS[args.dataset]
    epochs = args.epochs if args.epochs is not None else default_epochs

    # Compute expected patches
    patch_size = 16
    num_patches = ((args.image_size - patch_size) // patch_size + 1) ** 2
    reduction_pct = (1 - num_patches / 196) * 100

    print(f'Dataset: {args.dataset}', flush=True)
    print(f'Image size: {args.image_size}×{args.image_size}', flush=True)
    print(f'Patches: {num_patches} (vs 196 baseline, {reduction_pct:.1f}% reduction)', flush=True)
    print(f'Batch size: {args.batch_size}, LR: {args.lr}, WD: {args.weight_decay}', flush=True)
    print(f'Epochs: {epochs}, Label smoothing: {args.label_smoothing}', flush=True)
    print(f'Baseline Acc: {BASELINE_ACC[args.dataset]:.2f}%', flush=True)

    # Data
    result = loader_fn(batch_size=args.batch_size, data_dir='./data',
                       num_workers=args.num_workers, image_size=args.image_size)
    if len(result) == 4:
        train_loader, val_loader, test_loader, n_cls = result
        print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, '
              f'Test: {len(test_loader.dataset)}, Classes: {n_cls}', flush=True)
    else:
        train_loader, test_loader, n_cls = result
        val_loader = test_loader
        print(f'Train: {len(train_loader.dataset)}, Test/Val: {len(test_loader.dataset)}, '
              f'Classes: {n_cls}', flush=True)

    # Model
    print('Loading ViT-B/16 IN-21K...', flush=True)
    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=n_cls,
                              img_size=args.image_size)

    # Adjust position embeddings for smaller input
    if args.image_size != 224:
        new_grid_size = adjust_position_embeddings(model, args.image_size, patch_size)
        print(f'Adjusted position embeddings: {new_grid_size}×{new_grid_size} grid', flush=True)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Total params: {total_params:.2f}M', flush=True)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_epoch = -1
    test_acc_at_best = 0
    save_dir = f'./checkpoints/{args.dataset}_vit_b16_img{args.image_size}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== Training ViT-B/16 on {args.dataset} (image_size={args.image_size}, {epochs} epochs) ===\n', flush=True)

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)', flush=True)
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%', flush=True)
        print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%', flush=True)
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}', flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            test_loss_at_best, test_acc_at_best = evaluate(model, test_loader, criterion, device)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc_at_best,
                'image_size': args.image_size,
                'num_patches': num_patches,
            }, f'{save_dir}/best_model.pth')
            print(f'  -> Saved best model (Val: {best_val_acc:.2f}%, Test: {test_acc_at_best:.2f}%)', flush=True)
        print(flush=True)

    # Final evaluation
    print('=== Evaluating best model on test set ===', flush=True)
    ckpt = torch.load(f'{save_dir}/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Efficiency
    print('=== Efficiency Metrics ===', flush=True)
    latency, throughput = compute_efficiency_metrics(model, test_loader, device)

    baseline_acc = BASELINE_ACC[args.dataset]
    acc_diff = test_acc - baseline_acc

    # Estimate FLOPs reduction (attention is O(N^2), rest is O(N))
    # Rough estimate: full ViT-B/16 = 33.85G FLOPs
    flops_ratio = (num_patches / 196) ** 2  # attention dominates
    estimated_flops = 33.85 * (0.3 + 0.7 * flops_ratio)  # 30% linear + 70% quadratic

    print(f'\n========== Final Results ({args.dataset}, img_size={args.image_size}) ==========', flush=True)
    print(f'  Best Val Epoch: {best_epoch+1}', flush=True)
    print(f'  Best Val Acc:   {best_val_acc:.2f}%', flush=True)
    print(f'  Test Acc:       {test_acc:.2f}%', flush=True)
    print(f'  Baseline Acc:   {baseline_acc:.2f}%', flush=True)
    print(f'  Acc Diff:       {acc_diff:+.2f}%', flush=True)
    print(f'  --------------------------------------------', flush=True)
    print(f'  Image Size:     {args.image_size}×{args.image_size}', flush=True)
    print(f'  Patches:        {num_patches} (vs 196, {reduction_pct:.1f}% reduction)', flush=True)
    print(f'  Latency:        {latency:.2f} ms/sample', flush=True)
    print(f'  Throughput:     {throughput:.2f} samples/sec', flush=True)
    print(f'  Baseline FLOPs: 33.85G (ViT-B/16, 224×224)', flush=True)
    print(f'  Est. FLOPs:     {estimated_flops:.2f}G (rough estimate)', flush=True)
    print(f'================================================\n', flush=True)


if __name__ == '__main__':
    main()
