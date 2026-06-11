"""
Train Router via Attention Distillation from frozen ViT-B/16 teacher.

Approach:
  Frozen ViT-B/16 teacher -> extract CLS->patch attention from last block
    -> average over attention heads -> target importance scores (B, N)
  MLP Router (same architecture as MAEPatchSelectionViT)
    -> predict scores from patch embeddings -> MSE loss

The trained router can then be loaded into MAEPatchSelectionViT,
replacing the randomly initialized router.

Usage:
  python train_router_distill.py --dataset cifar100 --gpu 4
  python train_router_distill.py --dataset oxford_pets --gpu 4
  python train_router_distill.py --dataset food101 --gpu 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader, get_dtd_loader, get_flowers102_loader
import timm

DATASETS = {
    'cifar100':    (get_cifar100_loader,     100),
    'oxford_pets': (get_oxford_pets_loader,   37),
    'food101':     (get_food101_loader,      101),
    'dtd':         (get_dtd_loader,           47),
    'flowers102':  (get_flowers102_loader,   102),
}


def make_router(embed_dim=768):
    """MLP Router with same architecture as MAEPatchSelectionViT."""
    return nn.Sequential(
        nn.Linear(embed_dim, embed_dim // 2),
        nn.LayerNorm(embed_dim // 2),
        nn.GELU(),
        nn.Linear(embed_dim // 2, 1),
    )


def prepare_teacher(device):
    """Load frozen ViT-B/16 teacher with monkey-patched last attention to capture weights."""
    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Monkey-patch the last attention module to store softmax attention weights
    last_attn = model.blocks[-1].attn
    orig_forward = last_attn.forward

    def _forward_with_capture(self, x, **kwargs):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Apply q_norm/k_norm if they exist (not in base ViT-B/16 but in some variants)
        q_norm = getattr(self, 'q_norm', None)
        k_norm = getattr(self, 'k_norm', None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        self._last_attn = attn.detach()  # (B, H, N, N)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    last_attn.forward = types.MethodType(_forward_with_capture, last_attn)
    return model


@torch.no_grad()
def get_teacher_attention(teacher, x):
    """Forward teacher and extract CLS->patch attention (averaged over heads)."""
    _ = teacher(x)
    attn = teacher.blocks[-1].attn._last_attn  # (B, H, N, N)
    cls_attn = attn[:, :, 0, 1:]  # (B, H, N_patches)
    cls_attn = cls_attn.mean(dim=1)  # (B, N_patches), each row sums to 1
    return cls_attn


def evaluate_correlation(teacher, router, loader, device, k_ratio=0.5):
    """Measure top-k% overlap between router scores and teacher attention."""
    router.eval()
    overlaps = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            B = images.shape[0]

            attn_targets = get_teacher_attention(teacher, images)  # (B, N)
            N = attn_targets.shape[1]

            # Patch embeddings for router
            x = teacher.patch_embed(images)
            x = x + teacher.pos_embed[:, 1:, :]
            scores = router(x).squeeze(-1)  # (B, N)

            k = max(1, int(N * k_ratio))
            for i in range(B):
                target_topk = attn_targets[i].argsort(descending=True)[:k]
                pred_topk = scores[i].argsort(descending=True)[:k]
                overlap = len(set(target_topk.tolist()) & set(pred_topk.tolist()))
                overlaps.append(overlap / k * 100)

    avg = sum(overlaps) / len(overlaps)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}', flush=True)

    # Data
    loader_fn, num_classes = DATASETS[args.dataset]
    result = loader_fn(batch_size=args.batch_size, data_dir='./data', num_workers=4)
    if len(result) == 4:
        train_loader, val_loader, test_loader, n_cls = result
    else:
        train_loader, test_loader, n_cls = result
        val_loader = test_loader

    print(f'Dataset: {args.dataset}', flush=True)
    print(f'  Train: {len(train_loader.dataset)}, Classes: {n_cls}', flush=True)

    # Teacher
    print('Loading frozen ViT-B/16 teacher...', flush=True)
    teacher = prepare_teacher(device)
    embed_dim = teacher.embed_dim  # 768
    N = teacher.patch_embed.num_patches  # 196

    # Router (student)
    router = make_router(embed_dim).to(device)
    n_params = sum(p.numel() for p in router.parameters())
    print(f'Router params: {n_params/1e3:.1f}K', flush=True)

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-compute target stats on a small sample
    print('Computing target attention statistics...', flush=True)
    sample_attns = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            attn = get_teacher_attention(teacher, images)
            sample_attns.append(attn)
            if len(sample_attns) * images.shape[0] > 500:
                break
    sample_attns = torch.cat(sample_attns, dim=0)
    print(f'  Teacher CLS attention: mean={sample_attns.mean():.6f}, '
          f'std={sample_attns.std():.6f}, mean*N={sample_attns.mean()*N:.4f}', flush=True)
    del sample_attns

    print(f'\n=== Attention Distillation: {args.dataset} ({args.epochs} epochs) ===\n',
          flush=True)

    for epoch in range(args.epochs):
        teacher.eval()
        router.train()
        total_loss = 0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            B = images.shape[0]

            # Teacher attention: CLS->patch importance
            attn_targets = get_teacher_attention(teacher, images)  # (B, N)

            # Router on patch embeddings (with positional encoding)
            x = teacher.patch_embed(images)
            x = x + teacher.pos_embed[:, 1:, :]
            scores = router(x).squeeze(-1)  # (B, N)

            # Loss: predict scaled teacher attention
            # Scale target so mean ~ 1.0 (teacher attn sums to 1 per sample)
            loss = F.mse_loss(scores, attn_targets * N)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        # Validate: correlation on test set
        val_overlap = evaluate_correlation(teacher, router, test_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s): '
              f'Loss={avg_loss:.6f}, Top-50% overlap={val_overlap:.2f}%', flush=True)

    # Final evaluation
    final_overlap = evaluate_correlation(teacher, router, test_loader, device)
    print(f'\n=== Final Results ({args.dataset}) ===', flush=True)
    print(f'  Top-50% overlap with teacher: {final_overlap:.2f}%', flush=True)
    print(f'  Random baseline:             50.00%', flush=True)
    print(f'  Router params: {n_params/1e3:.1f}K', flush=True)

    # Save router weights
    save_dir = f'./checkpoints/router_distill_{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'router_state_dict': router.state_dict(),
        'dataset': args.dataset,
        'val_overlap': final_overlap,
    }, f'{save_dir}/router.pth')
    print(f'Saved to {save_dir}/router.pth', flush=True)
    print('Done!', flush=True)


if __name__ == '__main__':
    main()
