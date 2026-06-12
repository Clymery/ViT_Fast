"""
Train ViT-B/16 IN-21K with MAE-style patch selection on multiple datasets.
Keeps 50% patches via differentiable top-k router + reconstruction loss.

Architecture:
  Image -> Patch Embed + Pos Embed -> Router -> Top-K (STE sigmoid, no Gumbel)
    -> Lightweight Encoder (2 ViT-B blocks)
    -> split:
        (a) Main backbone (10 blocks) -> CLS -> CE Loss
        (b) MAE Decoder (4 blocks, 512-dim) -> reconstruct discarded -> MSE Loss

Usage:
  python train_patch_selection_mae.py --dataset cifar100 --gpu 6
  python train_patch_selection_mae.py --dataset oxford_pets --gpu 3
  python train_patch_selection_mae.py --dataset food101 --gpu 5
"""
import torch
import torch.nn as nn
import time
import os
import sys
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader
from models import create_model, patchify

DATASETS = {
    'cifar100':    (get_cifar100_loader,     100, 100),
    'oxford_pets': (get_oxford_pets_loader,   37, 100),
    'food101':     (get_food101_loader,      101,  30),
}

BASELINE_ACC = {
    'cifar100':   91.69,
    'oxford_pets': 93.81,
    'food101':    91.37,
}


def get_mse_weight(epoch, total_epochs, start_w=1.0, end_w=0.1):
    """Cosine anneal MSE weight from start_w to end_w over epochs."""
    frac = epoch / max(1, total_epochs - 1)
    return end_w + 0.5 * (start_w - end_w) * (1 + math.cos(math.pi * frac))


def train_one_epoch(model, loader, criterion, optimizer, device,
                    accum_steps=1, mse_weight=0.5, epoch_time=None):
    model.train()
    total_ce_loss = 0
    total_mse_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    epoch_start = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        # Forward: model returns (logits, pred, keep_mask) during training
        logits, pred, keep_mask = model(images)

        # CE loss
        ce_loss = criterion(logits, targets)

        # MSE loss on discarded patches (where keep_mask == 0)
        target_pixels = patchify(images)  # (B, N, p*p*C)
        # Per-patch MSE: (B, N)
        mse_loss = ((pred - target_pixels) ** 2).mean(dim=-1)
        # Only on discarded patches
        discard_mask = 1.0 - keep_mask  # 1 = discarded
        mse_loss = (mse_loss * discard_mask).sum() / discard_mask.sum()

        # Combined loss
        loss = ce_loss + mse_weight * mse_loss
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_ce_loss += ce_loss.item()
        total_mse_loss += mse_loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Handle remaining gradient accumulation
    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    epoch_duration = time.time() - epoch_start

    if epoch_time is not None:
        epoch_time.append(epoch_duration)

    n_batches = len(loader)
    return (total_ce_loss / n_batches, total_mse_loss / n_batches,
            100. * correct / total, epoch_duration)


@torch.no_grad()
def evaluate(model, loader, criterion, device, track_patches=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    kept_patches = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)  # eval returns only logits
        loss = criterion(logits, targets)
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if track_patches and hasattr(model, '_last_k'):
            kept_patches.append(model._last_k)
    result = (total_loss / len(loader), 100. * correct / total)
    if track_patches and kept_patches:
        avg_k = sum(kept_patches) / len(kept_patches)
        avg_n = getattr(model, '_last_n', 196)
        result = result + (avg_k, avg_n, avg_k / avg_n * 100)
    return result


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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--keep_ratio', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mse_start', type=float, default=1.0,
                        help='Starting weight for MSE loss')
    parser.add_argument('--mse_end', type=float, default=0.1,
                        help='Final weight for MSE loss (cosine anneal)')
    parser.add_argument('--decoder_dim', type=int, default=512,
                        help='MAE decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=4,
                        help='MAE decoder transformer depth')
    parser.add_argument('--router_path', type=str, default=None,
                        help='Load pretrained router weights from this path')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}', flush=True)

    loader_fn, num_classes, epochs = DATASETS[args.dataset]
    effective_bs = args.batch_size * args.accum
    k = int(196 * args.keep_ratio)

    print(f'Dataset: {args.dataset}', flush=True)
    print(f'  Batch: {args.batch_size}, Accum: {args.accum}, Effective: {effective_bs}', flush=True)
    print(f'  Epochs: {epochs}, LR: {args.lr}, WD: {args.weight_decay}', flush=True)
    print(f'  Label smoothing: {args.label_smoothing}', flush=True)
    print(f'  Keep ratio: {args.keep_ratio} ({k}/196 patches)', flush=True)
    print(f'  MSE weight: {args.mse_start} -> {args.mse_end} (cosine anneal)', flush=True)
    print(f'  Decoder: dim={args.decoder_dim}, depth={args.decoder_depth}', flush=True)
    print(f'  Baseline Acc: {BASELINE_ACC[args.dataset]:.2f}%', flush=True)

    # Data
    result = loader_fn(batch_size=args.batch_size, data_dir='./data', num_workers=4, image_size=args.image_size)
    if len(result) == 4:
        train_loader, val_loader, test_loader, n_cls = result
        print(f'  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, '
              f'Test: {len(test_loader.dataset)}, Classes: {n_cls}', flush=True)
    else:
        train_loader, test_loader, n_cls = result
        val_loader = test_loader
        print(f'  Train: {len(train_loader.dataset)}, Test/Val: {len(test_loader.dataset)}, '
              f'Classes: {n_cls}', flush=True)

    # Model
    print('Creating MAE Patch Selection ViT-B/16...', flush=True)
    model = create_model(
        model_name='mae_patch_selection_vit_b16',
        num_classes=n_cls,
        keep_ratio=args.keep_ratio,
        pretrained=True,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        img_size=args.image_size,
    )
    model = model.to(device)
    router_params = sum(p.numel() for p in model.router.parameters()) / 1e3
    decoder_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Total params: {total_params:.2f}M', flush=True)
    print(f'  Router params: {router_params:.1f}K', flush=True)
    print(f'  Decoder params: {decoder_params:.2f}M', flush=True)

    # Load pretrained router if specified (from attention distillation)
    if args.router_path is not None:
        print(f'Loading pretrained router from {args.router_path}...', flush=True)
        ckpt = torch.load(args.router_path, map_location=device)
        model.router.load_state_dict(ckpt['router_state_dict'])
        model.router.requires_grad_(True)  # still fine-tune during training
        print('  -> Loaded (router will be fine-tuned during training)', flush=True)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_epoch = -1
    test_acc_at_best = 0
    epoch_times = []
    save_dir = f'./checkpoints/{args.dataset}_mae_patchsel_b16_keep{int(args.keep_ratio*100)}'
    if args.router_path is not None:
        save_dir += '_distill'
    os.makedirs(save_dir, exist_ok=True)
    print(f'\n=== MAE Patch Selection ViT-B/16 on {args.dataset} ({epochs} epochs) ===\n', flush=True)

    for epoch in range(epochs):
        # Anneal MSE weight
        mse_w = get_mse_weight(epoch, epochs, args.mse_start, args.mse_end)

        train_ce, train_mse, train_acc, ep_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.accum, mse_w, epoch_times)
        scheduler.step()

        val_loss, val_acc, avg_k, avg_n, keep_pct = evaluate(
            model, val_loader, criterion, device, track_patches=True)

        print(f'Epoch {epoch+1}/{epochs} ({ep_time:.1f}s, MSE w={mse_w:.3f})', flush=True)
        print(f'  Train CE: {train_ce:.4f}, MSE: {train_mse:.6f}, Acc: {train_acc:.2f}%', flush=True)
        print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%', flush=True)
        print(f'  Keep:  {int(avg_k)}/{int(avg_n)} patches ({keep_pct:.1f}%)', flush=True)
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}', flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            test_loss_at_best, test_acc_at_best = evaluate(model, test_loader, criterion, device)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc_at_best,
            }, f'{save_dir}/best_model.pth')
            print(f'  -> Saved best model (Val Acc: {best_val_acc:.2f}%, '
                  f'Test Acc: {test_acc_at_best:.2f}%)', flush=True)
        print(flush=True)

    # Final evaluation
    print('=== Evaluating best model on test set ===', flush=True)
    ckpt = torch.load(f'{save_dir}/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Efficiency
    print('=== Efficiency Metrics ===', flush=True)
    latency, throughput = compute_efficiency_metrics(model, test_loader, device)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    baseline_acc = BASELINE_ACC[args.dataset]
    acc_diff = test_acc - baseline_acc

    print(f'\n========== Final Results ({args.dataset}) ==========', flush=True)
    print(f'  Best Val Epoch: {best_epoch+1}', flush=True)
    print(f'  Best Val Acc:   {best_val_acc:.2f}%', flush=True)
    print(f'  Test Acc:       {test_acc:.2f}%', flush=True)
    print(f'  Baseline Acc:   {baseline_acc:.2f}%', flush=True)
    print(f'  Acc Diff:       {acc_diff:+.2f}%', flush=True)
    print(f'  Keep Ratio:     {args.keep_ratio} ({k}/196 patches)', flush=True)
    print(f'  --------------------------------------------', flush=True)
    print(f'  Avg Epoch Time:  {avg_epoch_time:.1f}s', flush=True)
    print(f'  Latency:         {latency:.2f} ms/sample', flush=True)
    print(f'  Throughput:      {throughput:.2f} samples/sec', flush=True)
    print(f'  Baseline FLOPs:  33.85G (ViT-B/16 full)', flush=True)
    print(f'===============================================\n', flush=True)


if __name__ == '__main__':
    main()
