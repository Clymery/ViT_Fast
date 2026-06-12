import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time
import os

import timm
from timm.data import Mixup
from timm.utils import ModelEmaV2

from models import create_model


def compute_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """Estimate FLOPs using fvcore."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(input_size).to(device)
        model = model.to(device)
        flops = FlopCountAnalysis(model, dummy)
        return flops.total() / 1e9  # GFLOPs
    except Exception:
        return 0.0


class Trainer:
    def __init__(self, model, train_loader, test_loader, num_classes,
                 device='cuda', lr=1e-3, weight_decay=0.05,
                 label_smoothing=0.0, mixup_fn=None, ema_decay=0.0,
                 clip_grad=None, warmup_epochs=0, total_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.mixup_fn = mixup_fn
        self.ema_decay = ema_decay
        self.clip_grad = clip_grad
        self.warmup_epochs = warmup_epochs

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Warmup + Cosine scheduler (in iterations)
        self.total_iters = len(train_loader)
        cosine_iters = max(1, total_epochs - warmup_epochs) * self.total_iters
        if warmup_epochs > 0:
            warmup_iters = warmup_epochs * self.total_iters
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                                        total_iters=warmup_iters)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_iters)
            self.scheduler = SequentialLR(self.optimizer,
                                          schedulers=[warmup_scheduler, cosine_scheduler],
                                          milestones=[warmup_iters])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_iters)

        # EMA
        self.ema_model = ModelEmaV2(model, decay=ema_decay) if ema_decay > 0 else None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)

            # MixUp/CutMix
            if self.mixup_fn is not None:
                images, targets = self.mixup_fn(images, targets)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            # EMA update
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            total_loss += loss.item()
            if self.mixup_fn is None:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            else:
                # Approximate accuracy for mixup (count correct if both augmentations agree)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if targets.dim() > 1:
                    correct += predicted.eq(targets.argmax(dim=1)).sum().item()
                else:
                    correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self, model=None):
        model = model if model is not None else self.model
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, epochs, save_dir='./checkpoints', resume_from=None):
        os.makedirs(save_dir, exist_ok=True)
        best_acc = 0
        start_epoch = 0

        # Resume from checkpoint
        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            best_acc = ckpt.get('test_acc', 0)
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f'Resumed from epoch {ckpt.get("epoch", 0)} (best acc: {best_acc:.2f}%)')

        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            # Use raw model for evaluation tracking; EMA is for final inference only
            test_loss, test_acc = self.evaluate(self.model)
            self.scheduler.step()

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print()

            if test_acc > best_acc:
                best_acc = test_acc
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'test_acc': test_acc,
                }
                if self.ema_model is not None:
                    ckpt['ema_state_dict'] = self.ema_model.module.state_dict()
                torch.save(ckpt, os.path.join(save_dir, 'best_model.pth'))
                print(f'  -> Saved best model (Acc: {best_acc:.2f}%)')
                print()

            # Periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'test_acc': test_acc,
                }
                if self.ema_model is not None:
                    ckpt['ema_state_dict'] = self.ema_model.module.state_dict()
                torch.save(ckpt, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))
                print(f'  -> Saved periodic checkpoint (epoch {epoch+1})')
                print()

        print(f'Training complete! Best accuracy: {best_acc:.2f}%')
        return best_acc


def compute_efficiency_metrics(model, test_loader, device='cuda'):
    model.eval()
    model = model.to(device)

    total_time = 0
    total_samples = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)

            if device != 'cpu':
                torch.cuda.synchronize()
            start_time = time.time()

            _ = model(images)

            if device != 'cpu':
                torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)
            total_samples += batch_size

    avg_latency = total_time / total_samples * 1000
    throughput = total_samples / total_time

    return avg_latency, throughput


def get_model_summary(model, input_size=(1, 3, 224, 224), device='cpu'):
    """Print model parameter count and FLOPs."""
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = compute_flops(model, input_size, device)
    return params, gflops


def get_dataset_loader(dataset, batch_size, data_dir='./data', num_workers=4, use_randaugment=False):
    """Dispatch to the right dataset loader."""
    from datasets import get_cifar10_loader, get_cifar100_loader, get_tiny_imagenet_loader

    # Tiny-ImageNet at 224x224, others default
    if dataset == 'cifar10':
        train_loader, test_loader, num_classes = get_cifar10_loader(
            batch_size=batch_size, data_dir=data_dir, num_workers=num_workers)
    elif dataset == 'cifar100':
        train_loader, test_loader, num_classes = get_cifar100_loader(
            batch_size=batch_size, data_dir=data_dir, num_workers=num_workers,
            use_randaugment=use_randaugment)
    elif dataset == 'tiny_imagenet':
        train_loader, test_loader, num_classes = get_tiny_imagenet_loader(
            batch_size=batch_size, data_dir=data_dir, image_size=224, num_workers=num_workers)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return train_loader, test_loader, num_classes


if __name__ == '__main__':
    import argparse
    import sys
    from datetime import datetime

    parser = argparse.ArgumentParser(description='Vision Transformer Training with Patch Selection')
    parser.add_argument('--model', type=str, default='vit_small',
                        choices=['vit_small', 'swin_tiny', 'patch_selection_vit', 'random_prune_vit'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='Dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--keep_ratio', type=float, default=0.5,
                        help='Ratio of patches to keep (for pruning methods)')
    parser.add_argument('--selection_mode', type=str, default='topk',
                        choices=['topk', 'adaptive'],
                        help='Patch selection strategy: fixed top-k or adaptive threshold')
    parser.add_argument('--adaptive_alpha', type=float, default=0.5,
                        help='Threshold multiplier for adaptive mode: thresh = mean + alpha*std')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for image splitting (8, 16, 32)')
    parser.add_argument('--patch_stride', type=int, default=None,
                        help='Patch stride (None = same as patch_size, smaller = overlap)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Data loading workers')
    parser.add_argument('--use_randaugment', action='store_true', default=False,
                        help='Use RandAugment for data augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.0 = disabled, 0.1 recommended)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, cuda:7, etc.)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--load_mae', type=str, default=None,
                        help='Path to MAE pretrained encoder weights for stage 2 finetuning')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for log files')
    # Anti-overfitting
    parser.add_argument('--drop_path', type=float, default=0.0,
                        help='Stochastic depth drop path rate (0.0 = disabled, 0.1 recommended)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='MixUp alpha (0.0 = disabled, 0.8 recommended)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='CutMix alpha (0.0 = disabled, 1.0 recommended)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay rate (0.0 = disabled, 0.9999 recommended)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clipping max norm (None = disabled, 1.0 recommended)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of linear warmup epochs (5-10 recommended)')
    args = parser.parse_args()

    # Setup log file
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_tag = f'{args.dataset}_{args.model}'
    if 'prune' in args.model or 'selection' in args.model:
        exp_tag += f'_keep{int(args.keep_ratio*100)}_{args.selection_mode}'
    if args.model == 'patch_selection_vit' and args.selection_mode == 'adaptive':
        exp_tag += f'_alpha{args.adaptive_alpha}'
    if args.patch_size != 16:
        exp_tag += f'_ps{args.patch_size}'
    log_file = f'{args.log_dir}/{exp_tag}_{timestamp}.log'
    log_fh = open(log_file, 'w', buffering=1)  # line-buffered

    def teeprint(*args_tee, **kwargs_tee):
        """Print to both console and log file."""
        kwargs_tee.pop('file', None)
        original_print(*args_tee, **kwargs_tee)
        original_print(*args_tee, file=log_fh, **kwargs_tee)
        log_fh.flush()

    original_print = print
    print = teeprint

    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Using device: {device}')

    # Dataset
    train_loader, test_loader, num_classes = get_dataset_loader(
        args.dataset, args.batch_size, num_workers=args.num_workers)
    print(f'Dataset: {args.dataset}, {len(train_loader.dataset)} train, '
          f'{len(test_loader.dataset)} test, {num_classes} classes')

    # Model
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        keep_ratio=args.keep_ratio,
        pretrained=args.pretrained,
        selection_mode=args.selection_mode,
        adaptive_alpha=args.adaptive_alpha,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        drop_path_rate=args.drop_path,
    )

    # Load MAE pretrained weights (for Stage 2)
    if args.load_mae and hasattr(model, 'load_mae_pretrained'):
        model.load_mae_pretrained(args.load_mae)

    # Compute model summary on CPU first (fvcore doesn't always work on CUDA)
    summary_device = 'cpu' if not device.startswith('cuda') else device
    params_m, gflops = get_model_summary(model, device=summary_device)
    print(f'Model: {args.model}')
    print(f'  Parameters: {params_m:.2f}M')
    print(f'  FLOPs: {gflops:.2f}G')
    if 'prune' in args.model or 'selection' in args.model:
        print(f'  Keep ratio: {args.keep_ratio} ({int(args.keep_ratio * 100)}%)')
        print(f'  Selection mode: {args.selection_mode}')
        if args.selection_mode == 'adaptive':
            print(f'  Adaptive alpha: {args.adaptive_alpha}')
    if args.patch_size != 16 or (args.patch_stride and args.patch_stride != args.patch_size):
        print(f'  Patch config: size={args.patch_size}, stride={args.patch_stride or args.patch_size}')
        print(f'  Patches per image: {model.patch_embed.num_patches}')

    # Anti-overfitting config
    print(f'  RandAugment: {args.use_randaugment}')
    print(f'  Label smoothing: {args.label_smoothing}')
    print(f'  Drop path: {args.drop_path}')
    print(f'  MixUp alpha: {args.mixup}, CutMix alpha: {args.cutmix}')
    print(f'  EMA decay: {args.ema_decay}')
    print(f'  Gradient clipping: {args.clip_grad}')
    print(f'  Warmup epochs: {args.warmup_epochs}')

    # Experiment tag for checkpoint dir
    tag = f'{args.dataset}_{args.model}'
    if 'prune' in args.model or 'selection' in args.model:
        tag += f'_keep{int(args.keep_ratio*100)}_{args.selection_mode}'
    if args.model == 'patch_selection_vit' and args.selection_mode == 'adaptive':
        tag += f'_alpha{args.adaptive_alpha}'
    if args.patch_size != 16:
        tag += f'_ps{args.patch_size}'
    save_dir = f'{args.save_dir}/{tag}'

    # MixUp / CutMix
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )
        print(f'Using MixUp/CutMix: alpha={args.mixup}/{args.cutmix}')

    # Training
    trainer = Trainer(
        model, train_loader, test_loader, num_classes,
        device=device, lr=args.lr, weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_fn=mixup_fn,
        ema_decay=args.ema_decay,
        clip_grad=args.clip_grad,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )
    best_acc = trainer.train(epochs=args.epochs, save_dir=save_dir, resume_from=args.resume)

    # Efficiency evaluation
    print('\n=== Efficiency Metrics ===')
    latency, throughput = compute_efficiency_metrics(model, test_loader, device=device)
    print(f'  Latency: {latency:.2f} ms per sample')
    print(f'  Throughput: {throughput:.2f} samples/sec')
    print(f'  FLOPs: {gflops:.2f}G')
    print(f'  Best Accuracy: {best_acc:.2f}%')
    print(f'========================\n')
