import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np


def create_vit_small(num_classes=100, pretrained=True, drop_path_rate=0.0):
    """ViT-S/16 baseline - all patches, no pruning."""
    model = timm.create_model(
        'vit_small_patch16_224.augreg_in1k',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def create_swin_tiny(num_classes=100, pretrained=True):
    """Swin-Tiny baseline."""
    model = timm.create_model(
        'swin_tiny_patch4_window7_224.ms_in1k',
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def _replace_patch_embed(model, patch_size, patch_stride, img_size=224):
    """Replace the patch embedding layer for custom patch_size/stride."""
    old_embed = model.patch_embed
    in_chans = old_embed.proj.in_channels
    embed_dim = model.embed_dim

    new_conv = nn.Conv2d(in_chans, embed_dim,
                         kernel_size=patch_size, stride=patch_stride)
    # Compute new number of patches
    h = w = img_size
    h_out = (h - patch_size) // patch_stride + 1
    w_out = (w - patch_size) // patch_stride + 1
    num_patches = h_out * w_out

    # Initialize new conv from old (interpolate if sizes differ)
    if old_embed.proj.weight.shape[2] == patch_size == patch_stride == old_embed.proj.stride[0]:
        new_conv.weight.data.copy_(old_embed.proj.weight.data)
        if old_embed.proj.bias is not None:
            new_conv.bias.data.copy_(old_embed.proj.bias.data)

    model.patch_embed.proj = new_conv
    model.patch_embed.num_patches = num_patches
    model.patch_embed.grid_size = (h_out, w_out)

    # New positional embedding
    old_pos = model.pos_embed  # (1, N+1, D)
    cls_pos = old_pos[:, 0:1, :]
    patch_pos = old_pos[:, 1:, :]
    # Interpolate to new number of patches
    old_h = int(math.sqrt(patch_pos.shape[1]))
    patch_pos_3d = patch_pos.transpose(1, 2).reshape(1, -1, old_h, old_h)
    new_patch_pos = F.interpolate(patch_pos_3d, size=(h_out, w_out), mode='bicubic', align_corners=False)
    new_patch_pos = new_patch_pos.reshape(1, -1, h_out * w_out).transpose(1, 2)
    model.pos_embed = nn.Parameter(torch.cat([cls_pos, new_patch_pos], dim=1))

    return model


class GumbelSelection(nn.Module):
    """Differentiable patch selection supporting top-k and adaptive modes."""

    def __init__(self, num_patches, keep_ratio=0.5, selection_mode='topk',
                 adaptive_alpha=0.5, min_keep=16):
        super().__init__()
        self.num_patches = num_patches
        self.num_keep = max(1, int(num_patches * keep_ratio))
        self.selection_mode = selection_mode
        self.adaptive_alpha = adaptive_alpha
        self.min_keep = min_keep

    def forward(self, scores):
        """
        Args:
            scores: (B, N) importance scores per patch
        Returns:
            selected_indices: (B, K) indices
            mask: (B, N) soft/hard mask
            k: int (number kept)
        """
        B, N = scores.shape

        if self.selection_mode == 'topk':
            return self._topk_selection(scores, B, N)
        elif self.selection_mode == 'adaptive':
            return self._adaptive_selection(scores, B, N)
        else:
            raise ValueError(f'Unknown selection_mode: {self.selection_mode}')

    def _topk_selection(self, scores, B, N):
        k = min(self.num_keep, N)

        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            noisy_scores = scores + gumbel_noise
            _, indices = torch.topk(noisy_scores, k, dim=1)

            hard_mask = torch.zeros_like(scores)
            hard_mask.scatter_(1, indices, 1.0)

            soft_mask = F.softmax(scores / 0.1, dim=1)
            soft_mask = soft_mask / soft_mask.sum(dim=1, keepdim=True) * N

            mask = hard_mask.detach() + soft_mask - soft_mask.detach()
            return indices, mask, k
        else:
            _, indices = torch.topk(scores, k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, indices, 1.0)
            return indices, mask, k

    def _adaptive_selection(self, scores, B, N):
        """Adaptive: keep patches depending on score distribution.
        Threshold = mean(scores) + alpha * std(scores).
        Each image determines its own keep count; we pad to batch max.
        """
        if self.training:
            # Training: use concrete distribution (Gumbel-Sigmoid)
            threshold = scores.mean(dim=1, keepdim=True) + \
                        self.adaptive_alpha * scores.std(dim=1, keepdim=True)
            # Gumbel-Sigmoid for differentiable per-patch decisions
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            logits = (scores - threshold) / 0.1 + gumbel_noise
            probs = torch.sigmoid(logits)  # keep probability per patch

            # STE: hard decisions in forward, soft gradients in backward
            hard = (probs > 0.5).float()
            mask = hard.detach() + probs - probs.detach()

            # Compute per-image K (for gathering)
            k_per_image = mask.sum(dim=1).long()  # (B,)
            k_per_image = torch.clamp(k_per_image, min=self.min_keep, max=N)
            k = k_per_image.max().item()

            # Expand/collapse for variable K: pad each to k
            # Simple: just use per-image k for topk from scores
            # This keeps the gradient flow through gumbel scores
            indices_list = []
            for i in range(B):
                ki = k_per_image[i].item()
                _, idx_i = torch.topk(scores[i], ki)
                if ki < k:
                    idx_i = F.pad(idx_i, (0, k - ki), value=0)
                indices_list.append(idx_i)
            indices = torch.stack(indices_list)

            return indices, mask, k
        else:
            # Inference: per-image threshold, pad to max
            k_max = 0
            indices_list = []
            for i in range(B):
                thresh = scores[i].mean() + self.adaptive_alpha * scores[i].std()
                keep_mask = scores[i] > thresh
                ki = max(self.min_keep, keep_mask.sum().item())
                ki = min(ki, N)
                k_max = max(k_max, ki)
                _, idx_i = torch.topk(scores[i], ki)
                indices_list.append((ki, idx_i))

            # Pad all to k_max
            pad_indices = []
            for ki, idx_i in indices_list:
                if ki < k_max:
                    idx_i = F.pad(idx_i, (0, k_max - ki), value=0)
                pad_indices.append(idx_i)
            indices = torch.stack(pad_indices)

            mask = torch.zeros(B, N, device=scores.device)
            for i in range(B):
                mask[i, indices[i][:indices_list[i][0]]] = 1.0

            return indices, mask, k_max


class SemanticRouter(nn.Module):
    """
    Lightweight router for scoring patch importance.
    Uses per-patch MLP + 1-layer self-attention for context-aware scoring.
    """
    def __init__(self, embed_dim=384, hidden_dim=192, num_heads=4):
        super().__init__()
        self.per_patch_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.ln = nn.LayerNorm(hidden_dim)

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, D) patch tokens
        Returns:
            scores: (B, N) raw importance scores
        """
        B, N, D = x.shape

        # Per-patch MLP
        feat = self.per_patch_mlp(x)  # (B, N, H)

        # Self-attention across patches
        attn_feat, _ = self.cross_attn(feat, feat, feat)
        feat = self.ln(feat + attn_feat)

        # Score each patch
        scores = self.score_head(feat).squeeze(-1)  # (B, N)
        return scores


class PatchSelectionViT(nn.Module):
    """
    ViT-S/16 or ViT-B/16 with pre-tokenization patch selection.

    Supports:
    - topk: fixed K patches per image (based on keep_ratio)
    - adaptive: threshold-based, per-image variable K
    - Custom patch_size and patch_stride
    """
    def __init__(self, num_classes=100, keep_ratio=0.5, pretrained=True,
                 selection_mode='topk', adaptive_alpha=0.5,
                 patch_size=16, patch_stride=None, drop_path_rate=0.0,
                 backbone_name='vit_small_patch16_224.augreg_in1k'):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.selection_mode = selection_mode
        self.adaptive_alpha = adaptive_alpha
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size

        # Load pretrained backbone (S/16 or B/16 depending on backbone_name)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        # Override patch embedding if custom size
        if patch_size != 16 or self.patch_stride != 16:
            _replace_patch_embed(self.backbone, patch_size, self.patch_stride)

        self.patch_embed = self.backbone.patch_embed
        self.cls_token = self.backbone.cls_token
        self.pos_drop = self.backbone.pos_drop
        self.pos_embed = self.backbone.pos_embed

        embed_dim = self.backbone.embed_dim
        num_patches = self.patch_embed.num_patches

        # Selection mechanism
        self.selection = GumbelSelection(
            num_patches=num_patches,
            keep_ratio=keep_ratio,
            selection_mode=selection_mode,
            adaptive_alpha=adaptive_alpha,
            min_keep=16,
        )
        self._last_k = num_patches
        self._last_n = num_patches

        # Router (hidden_dim scales with embed_dim)
        router_hidden_dim = max(192, embed_dim // 2)
        router_num_heads = max(4, embed_dim // 64)
        self.router = SemanticRouter(
            embed_dim=embed_dim,
            hidden_dim=router_hidden_dim,
            num_heads=router_num_heads,
        )

        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm
        self.head = self.backbone.head

    def load_mae_pretrained(self, mae_encoder_path):
        """Load encoder weights from MAE pretrained checkpoint."""
        ckpt = torch.load(mae_encoder_path, map_location='cpu')
        # Support both full model and encoder-only checkpoints
        if 'encoder_state_dict' in ckpt:
            state_dict = ckpt['encoder_state_dict']
        else:
            state_dict = ckpt
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f'[PatchSelectionViT] Loaded MAE encoder: {msg}')

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]

        # 2. Score patches with semantic router
        scores = self.router(x)  # (B, N)

        # 3. Select patches
        selected_indices, mask, k = self.selection(scores)
        k = min(k, N)
        self._last_k = k  # track for logging
        self._last_n = N

        # Gather selected patches with mask gradient (STE: hard forward, soft backward)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        selected_patches = (x * mask.unsqueeze(-1))[batch_indices, selected_indices]  # (B, K, D)

        # 4. Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, selected_patches], dim=1)  # (B, K+1, D)

        # Add positional embeddings
        cls_pos = self.pos_embed[:, 0:1, :]  # (1, 1, D)
        all_patch_pos = self.pos_embed[:, 1:, :].expand(B, -1, -1)  # (B, N, D)
        selected_pos = all_patch_pos[batch_indices, selected_indices]  # (B, K, D)
        pos_embed = torch.cat([cls_pos.expand(B, -1, -1), selected_pos], dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 6. Classification head (use [CLS] token)
        x = x[:, 0]
        x = self.head(x)

        return x


class RandomPruneViT(nn.Module):
    """ViT-S/16 with random patch pruning - lower bound baseline."""
    def __init__(self, num_classes=100, keep_ratio=0.5, pretrained=True,
                 patch_size=16, patch_stride=None, drop_path_rate=0.0):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.patch_stride = patch_stride if patch_stride is not None else patch_size

        self.backbone = timm.create_model(
            'vit_small_patch16_224.augreg_in1k',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        if patch_size != 16 or self.patch_stride != 16:
            _replace_patch_embed(self.backbone, patch_size, self.patch_stride)

        self.patch_embed = self.backbone.patch_embed
        self.cls_token = self.backbone.cls_token
        self.pos_drop = self.backbone.pos_drop
        self.pos_embed = self.backbone.pos_embed

        num_patches = self.patch_embed.num_patches
        self.num_keep = max(1, int(num_patches * keep_ratio))

        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm
        self.head = self.backbone.head

    def forward(self, x):
        B = x.shape[0]
        N = self.patch_embed.num_patches
        k = self.num_keep

        x = self.patch_embed(x)

        # Random selection
        if self.training or k < N:
            indices = torch.randperm(N, device=x.device)[:k]
            indices = indices.unsqueeze(0).expand(B, -1)
        else:
            indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        x = x[batch_indices, indices]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        cls_pos = self.pos_embed[:, 0:1, :]
        all_patch_pos = self.pos_embed[:, 1:, :].expand(B, -1, -1)
        selected_pos = all_patch_pos[batch_indices, indices]
        pos_embed = torch.cat([cls_pos.expand(B, -1, -1), selected_pos], dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


# ========== MAE-style pretraining ==========

def random_masking(x, mask_ratio=0.75):
    """
    Randomly mask patches. Encoder only sees unmasked ones.
    Args:
        x: (B, N, D) patch tokens
        mask_ratio: fraction to mask
    Returns:
        x_masked: (B, N_keep, D) visible patches
        mask: (B, N) 0/1 mask (1 = masked)
        ids_restore: (B, N) indices to restore original order
    """
    B, N, D = x.shape
    n_keep = int(N * (1 - mask_ratio))

    # Random shuffle indices
    ids_shuffle = torch.rand(B, N, device=x.device).argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)

    # Keep the first n_keep, mask the rest
    ids_keep = ids_shuffle[:, :n_keep]
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
    x_masked = x[batch_idx, ids_keep]  # (B, n_keep, D)

    # Mask: 1 = masked, 0 = visible
    mask = torch.ones(B, N, device=x.device)
    mask[batch_idx, ids_keep] = 0

    return x_masked, mask, ids_restore


def patchify(images, patch_size=16):
    """Convert images to patch pixels.
    images: (B, 3, H, W)
    Returns: (B, N, patch_size*patch_size*3)
    """
    B, C, H, W = images.shape
    p = patch_size
    assert H % p == 0 and W % p == 0

    h = H // p
    w = W // p
    x = images.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.reshape(B, h * w, p * p * C)
    return x


def unpatchify(x, patch_size=16, channels=3):
    """Convert patch pixels back to images.
    x: (B, N, p*p*C)
    Returns: (B, C, H, W)
    """
    B, N, _ = x.shape
    p = patch_size
    h = w = int(N ** 0.5)
    assert h * w == N

    x = x.reshape(B, h, w, p, p, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.reshape(B, channels, h * p, w * p)
    return x


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE pretraining."""

    def __init__(self, embed_dim=384, decoder_embed_dim=192, decoder_depth=4,
                 decoder_num_heads=6, num_patches=196, patch_size=16, in_chans=3):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Mask token shared across all masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Positional embeddings for decoder (all patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))  # +1 for cls

        # Decoder transformer blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )

        # Prediction head
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans)

    def forward(self, x, ids_restore):
        """
        Args:
            x: (B, n_keep, D) encoder output (after norm)
            ids_restore: (B, N) indices to restore original order
        Returns:
            pred: (B, N, p*p*C) pixel predictions
        """
        B, n_keep, D = x.shape
        N = self.num_patches

        # Project to decoder dim
        x = self.decoder_embed(x)  # (B, n_keep, decoder_dim)

        # Append mask tokens
        n_mask = N - n_keep
        mask_tokens = self.mask_token.repeat(B, n_mask, 1)  # (B, n_mask, decoder_dim)
        x = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_dim)

        # Restore original order
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Add positional embedding
        x = x + self.decoder_pos_embed[:, 1:, :]  # skip cls pos

        # Decoder transformer
        x = self.decoder_blocks(x)

        # Prediction
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)  # (B, N, p*p*C)

        return pred


class MAEViT(nn.Module):
    """
    MAE-style pretraining for ViT-S/16.
    Encoder processes visible patches, decoder reconstructs masked ones.
    After pretraining, the encoder backbone can be used for downstream tasks.
    """

    def __init__(self, num_classes=100, mask_ratio=0.75,
                 decoder_depth=4, decoder_embed_dim=192,
                 pretrained=True):
        super().__init__()
        self.mask_ratio = mask_ratio

        # Encoder: ViT-S/16
        self.backbone = timm.create_model(
            'vit_small_patch16_224.augreg_in1k',
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.patch_embed = self.backbone.patch_embed
        self.cls_token = self.backbone.cls_token
        self.pos_drop = self.backbone.pos_drop
        self.pos_embed = self.backbone.pos_embed

        embed_dim = self.backbone.embed_dim
        num_patches = self.patch_embed.num_patches
        patch_size = 16  # ViT-S/16

        self.blocks = self.backbone.blocks
        self.encoder_norm = self.backbone.norm

        # Decoder
        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=6,
            num_patches=num_patches,
            patch_size=patch_size,
            in_chans=3,
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            loss: MSE reconstruction loss
            pred: (B, N, p*p*3) pixel predictions
            mask: (B, N) 1= masked, 0=visible
        """
        B, C, H, W = x.shape

        # Save original for loss computation
        images = x

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]
        x = self.pos_drop(x)

        # Random masking
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)
        # x_masked: (B, n_keep, D), mask: (B, N), ids_restore: (B, N)

        # Add cls token to masked sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        cls_pos = self.pos_embed[:, 0:1, :]
        x_masked = torch.cat([cls_tokens + cls_pos, x_masked], dim=1)  # (B, 1+n_keep, D)

        # Encoder
        for block in self.blocks:
            x_masked = block(x_masked)

        enc_out = self.encoder_norm(x_masked)

        # Remove cls token for decoder
        enc_patches = enc_out[:, 1:, :]  # (B, n_keep, D)

        # Decoder prediction
        pred = self.decoder(enc_patches, ids_restore)  # (B, N, p*p*3)

        # Compute loss on masked patches only
        target = patchify(images, patch_size=16)  # (B, N, p*p*3)

        # Per-patch normalization
        target_mean = target.mean(dim=-1, keepdim=True)
        target_var = target.var(dim=-1, keepdim=True) + 1e-6
        target_norm = (target - target_mean) / target_var.sqrt()

        # Loss: only on masked patches
        loss = (pred - target_norm) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum()  # average over masked patches

        return loss, pred, mask

    def get_encoder(self):
        """Return the encoder backbone for downstream tasks."""
        return self.backbone


class MAEPatchSelectionViT(nn.Module):
    """
    ViT-B/16 with MAE-style patch selection.

    Architecture:
      Image -> Patch Embed + Pos Embed -> Router (MLP scores)
        -> Differentiable Top-K (Sigmoid STE, no Gumbel) -> keep K patches
        -> Lightweight Encoder (first 2 ViT-B blocks, pretrained)
        -> split:
            (a) Main backbone (remaining 10 blocks) -> CLS head -> CE Loss
            (b) MAE Decoder (4 blocks, 512-dim) -> reconstruct discarded -> MSE Loss

    Training forward returns (logits, pred_pixels, keep_mask).
    Eval forward returns logits only.
    """

    def __init__(self, num_classes=100, keep_ratio=0.5, pretrained=True,
                 drop_path_rate=0.0, decoder_embed_dim=512, decoder_depth=4,
                 img_size=224):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.selection_temperature = 1.0
        self.img_size = img_size

        # Load pretrained ViT-B/16 backbone (source of all weights)
        backbone = timm.create_model(
            'vit_base_patch16_224.augreg_in21k',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop

        embed_dim = backbone.embed_dim  # 768
        num_patches = self.patch_embed.num_patches  # 196
        self.num_patches = num_patches
        self.num_keep = max(1, int(num_patches * keep_ratio))

        # Router: simple MLP (no self-attention, ~300K params)
        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Lightweight encoder: first 2 ViT-B blocks (pretrained weights)
        blocks = list(backbone.blocks)
        self.lightweight_encoder = nn.ModuleList(blocks[:2])

        # Main backbone: remaining 10 blocks
        self.main_blocks = nn.ModuleList(blocks[2:])

        self.norm = backbone.norm
        self.head = backbone.head

        # MAE Decoder (random init, trained from scratch)
        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=8,
            num_patches=num_patches,
            patch_size=16,
            in_chans=3,
        )

        # Clean up to avoid duplicate parameter ownership
        del backbone

        # Tracking
        self._last_k = num_patches
        self._last_n = num_patches

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            Training: (logits, pred, keep_mask)
                logits: (B, num_classes)
                pred: (B, N, p*p*C) decoder pixel predictions (all patches)
                keep_mask: (B, N) 1=selected/kept, 0=discarded
            Eval: logits (B, num_classes)
        """
        B = x.shape[0]

        # 1. Patch embedding + positional embedding (on ALL patches, before selection)
        x = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]
        x = x + self.pos_embed[:, 1:, :]  # pos-aware features for router
        x = self.pos_drop(x)

        # 2. Router scores
        scores = self.router(x).squeeze(-1)  # (B, N)

        # 3. Differentiable Top-K selection (no Gumbel noise)
        k = min(self.num_keep, N)
        _, indices = torch.topk(scores, k, dim=1)  # (B, K)

        # Build hard mask: 1 = selected
        hard_mask = torch.zeros(B, N, device=x.device)
        hard_mask.scatter_(1, indices, 1.0)

        if self.training:
            # Soft mask via sigmoid STE
            threshold = scores.topk(k, dim=1)[0][:, -1:]  # (B, 1)
            soft_mask = torch.sigmoid(
                (scores - threshold) / self.selection_temperature)
            keep_mask = hard_mask.detach() + soft_mask - soft_mask.detach()
        else:
            keep_mask = hard_mask

        self._last_k = k
        self._last_n = N

        # Gather selected patches
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        selected = x[batch_idx, indices]  # (B, K, D)

        # 4. Lightweight encoder (2 ViT-B blocks) — WITH CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        h = torch.cat([cls_tokens, selected], dim=1)  # (B, K+1, D)

        for block in self.lightweight_encoder:
            h = block(h)

        # Save patch features for decoder (before main backbone changes them)
        h_patches = h[:, 1:, :]  # (B, K, D), remove CLS

        # 5-A. Main backbone for classification (CLS goes through all 12 blocks now)
        for block in self.main_blocks:
            h = block(h)

        h = self.norm(h)
        logits = self.head(h[:, 0])

        # Eval: no decoder needed
        if not self.training:
            return logits

        # 5-B. MAE Decoder for reconstruction (training only)
        all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

        # Discarded indices (not in selected)
        discard_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        discard_mask.scatter_(1, indices, False)
        discarded = all_idx[discard_mask].reshape(B, -1)  # (B, N-K)

        # [selected | discarded] -> argsort -> ids_restore
        ids_sort = torch.cat([indices, discarded], dim=1)  # (B, N)
        ids_restore = ids_sort.argsort(dim=1)  # (B, N)

        # Decoder: reconstruct from lightweight encoder output (CLS removed)
        pred = self.decoder(h_patches, ids_restore)  # (B, N, p*p*C)

        return logits, pred, hard_mask.detach()


def create_model(model_name, num_classes=100, keep_ratio=0.5, pretrained=True,
                 selection_mode='topk', adaptive_alpha=0.5,
                 patch_size=16, patch_stride=None,
                 mask_ratio=0.75, decoder_depth=4, decoder_embed_dim=192,
                 drop_path_rate=0.0, img_size=224):
    """Factory function for all models."""
    if model_name == 'vit_small':
        return create_vit_small(num_classes, pretrained, drop_path_rate=drop_path_rate)
    elif model_name == 'swin_tiny':
        return create_swin_tiny(num_classes, pretrained)
    elif model_name == 'patch_selection_vit':
        return PatchSelectionViT(
            num_classes, keep_ratio, pretrained,
            selection_mode=selection_mode,
            adaptive_alpha=adaptive_alpha,
            patch_size=patch_size,
            patch_stride=patch_stride,
            drop_path_rate=drop_path_rate,
            backbone_name='vit_small_patch16_224.augreg_in1k',
        )
    elif model_name == 'patch_selection_vit_b16':
        return PatchSelectionViT(
            num_classes, keep_ratio, pretrained,
            selection_mode=selection_mode,
            adaptive_alpha=adaptive_alpha,
            patch_size=patch_size,
            patch_stride=patch_stride,
            drop_path_rate=drop_path_rate,
            backbone_name='vit_base_patch16_224.augreg_in21k',
        )
    elif model_name == 'patch_selection_vit_b16_in1k':
        return PatchSelectionViT(
            num_classes, keep_ratio, pretrained,
            selection_mode=selection_mode,
            adaptive_alpha=adaptive_alpha,
            patch_size=patch_size,
            patch_stride=patch_stride,
            drop_path_rate=drop_path_rate,
            backbone_name='vit_base_patch16_224.augreg_in1k',
        )
    elif model_name == 'random_prune_vit':
        return RandomPruneViT(
            num_classes, keep_ratio, pretrained,
            patch_size=patch_size,
            patch_stride=patch_stride,
            drop_path_rate=drop_path_rate,
        )
    elif model_name == 'mae_vit':
        return MAEViT(
            num_classes=num_classes,
            mask_ratio=mask_ratio,
            decoder_depth=decoder_depth,
            decoder_embed_dim=decoder_embed_dim,
            pretrained=pretrained,
        )
    elif model_name == 'mae_patch_selection_vit_b16':
        return MAEPatchSelectionViT(
            num_classes=num_classes,
            keep_ratio=keep_ratio,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
