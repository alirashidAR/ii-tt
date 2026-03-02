"""
train_segformer.py
==================
Fine-tune SegFormer-B4 (nvidia/mit-b4) on the geo-spatial segmentation dataset.

Architecture differences vs. DeepLabV3+ (train.py):
  - Backbone : Mix Transformer (MiT-B4) — hierarchical ViT encoder
  - Decoder  : LightWeight All-MLP (LWA-MLP) head, no dilated convolutions
  - Input    : standard pixel_values tensor (B, 3, H, W) normalised with
               ImageNet mean/std (handled by SegFormerImageProcessor)
  - Output   : logits dict  →  out["logits"]  shape (B, n_classes, H/4, W/4)
               Must be upsampled to (H, W) before loss / mIoU.

HuggingFace Hub ID: "nvidia/mit-b4"
Install: pip install transformers accelerate

Dataset: same GeoDataset as train.py
  E:\\dataset\\train\\images\\*.npy   (C, H, W) uint8
  E:\\dataset\\train\\masks\\*.npy    (H, W)    uint8  class ∈ {0,1,2,3}
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GeoDataset(Dataset):
    """
    Loads pre-tiled .npy patches.
    Images : (C, H, W) float32  normalised to [0, 1]   — C may be 3 or 4
    Masks  : (H, W)   int64     class labels  {0,1,2,3}

    SegFormer expects 3-channel input.  If C > 3 we take the first 3 channels.
    The ImageProcessor handles mean/std normalisation internally.
    """

    def __init__(self, root_dir, processor: SegformerImageProcessor):
        self.img_dir   = Path(root_dir) / "images"
        self.mask_dir  = Path(root_dir) / "masks"
        self.img_files = sorted(self.img_dir.glob("*.npy"))
        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path  = self.img_files[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.load(img_path).astype(np.float32) / 255.0   # (C, H, W)  [0,1]
        mask  = np.load(mask_path).astype(np.int64)             # (H, W)

        # SegFormer requires exactly 3 channels; drop any extra (e.g. NIR)
        image = image[:3]                                        # (3, H, W)

        # SegformerImageProcessor expects (H, W, C) or (C, H, W) numpy arrays
        # and returns a dict with "pixel_values" already normalised
        enc = self.processor(
            images=image,           # accepts (C, H, W) float [0,1]
            return_tensors="pt",
            do_rescale=False,       # we already divided by 255
        )
        pixel_values = enc["pixel_values"].squeeze(0)           # (3, H, W)

        return pixel_values, torch.from_numpy(mask)


# ── Loss: Dice + Focal ────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Dice + Focal  (identical to train.py rationale)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.dice  = smp.losses.DiceLoss(
            mode="multiclass", classes=num_classes, smooth=1.0, from_logits=True
        )
        self.focal = smp.losses.FocalLoss(
            mode="multiclass", gamma=2.0, normalized=True
        )

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.focal(logits, targets)


# ── Mean IoU ──────────────────────────────────────────────────────────────────

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds   = preds.view(-1)
    targets = targets.view(-1)
    ious = []
    for cls in range(num_classes):
        tp    = ((preds == cls) & (targets == cls)).sum().item()
        fp    = ((preds == cls) & (targets != cls)).sum().item()
        fn    = ((preds != cls) & (targets == cls)).sum().item()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


# ── Build SegFormer-B4 model ──────────────────────────────────────────────────

def build_segformer(num_classes: int, pretrained: bool = True):
    """
    Load nvidia/mit-b4 pretrained weights and replace the decode head for
    `num_classes` output channels.

    SegformerForSemanticSegmentation.config controls:
      id2label / label2id : required by HF but not used during raw training
      num_labels          : number of output classes
    """
    model_id = "nvidia/mit-b4"
    label2id = {str(i): i for i in range(num_classes)}
    id2label  = {i: str(i) for i in range(num_classes)}

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,   # decode head is re-initialised
        )
    else:
        from transformers import SegformerConfig
        cfg = SegformerConfig.from_pretrained(
            model_id,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )
        model = SegformerForSemanticSegmentation(cfg)

    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    # ── Config ──────────────────────────────────────────────────────────────
    DATA_DIR    = r"E:\dataset\train"
    BATCH_SIZE  = 2           # B4 is ~62 M params; reduce to 1 if OOM on <16 GB VRAM
    LR          = 6e-5        # lower LR vs DeepLabV3+; transformer encoders are sensitive
    EPOCHS      = 30
    NUM_CLASSES = 6           # 0=BG, 1=Road, 2=Built-up, 3=Water, 4=Bridge, 5=Railway
    SAVE_PATH   = "best_segformer.pth"
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP     = torch.cuda.is_available()

    # ── Processor (handles ImageNet mean/std normalisation) ─────────────────
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/mit-b4",
        do_resize=False,        # keep original tile resolution
        do_rescale=False,       # GeoDataset already provides [0,1] floats
        do_normalize=True,      # apply ImageNet mean/std
    )

    # ── Data ────────────────────────────────────────────────────────────────
    dataset    = GeoDataset(DATA_DIR, processor)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # ── Sanity check ────────────────────────────────────────────────────────
    sample_img, sample_mask = dataset[0]
    H, W = sample_img.shape[1], sample_img.shape[2]
    print(f"Tiles        : {len(dataset)}  (train {train_size} | val {val_size})")
    print(f"Tile size    : {H} × {W}")
    print(f"Image  range : [{sample_img.min():.3f}, {sample_img.max():.3f}]  (normalised)")
    print(f"Mask classes : {sample_mask.unique().tolist()}  (expect subset of 0–5)")
    print(f"Device       : {DEVICE}  |  AMP: {USE_AMP}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_segformer(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params       : {n_params:.1f}M\n")

    # ── Loss ────────────────────────────────────────────────────────────────
    criterion = CombinedLoss(num_classes=NUM_CLASSES)

    # ── Optimizer: differential LR ──────────────────────────────────────────
    # Encoder (MiT-B4 Mix Transformer) → 10× slower — it's ImageNet-pretrained
    # Decode head                       → full LR    — randomly initialised
    optimizer = optim.AdamW([
        {"params": model.segformer.parameters(),  "lr": LR * 0.1},  # encoder
        {"params": model.decode_head.parameters(), "lr": LR},        # head
    ], weight_decay=0.01)

    # Cosine schedule with warm-up
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))   # 5 % warm-up

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    global_step   = 0

    # ── Loop ────────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")

        for pixel_values, masks in pbar:
            pixel_values = pixel_values.to(DEVICE)
            masks        = masks.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                # SegFormer returns an object; logits shape: (B, C, H/4, W/4)
                outputs = model(pixel_values=pixel_values)
                logits  = outputs.logits                          # (B, C, h, w)

                # Upsample to original tile size for loss computation
                logits_up = F.interpolate(
                    logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                loss = criterion(logits_up, masks)

            scaler.scale(loss).backward()

            # Gradient clipping — important for transformer training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[1]:.2e}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for pixel_values, masks in tqdm(val_loader,
                                            desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  "):
                pixel_values = pixel_values.to(DEVICE)
                masks        = masks.to(DEVICE)

                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    outputs   = model(pixel_values=pixel_values)
                    logits    = outputs.logits
                    logits_up = F.interpolate(
                        logits, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                    loss = criterion(logits_up, masks)

                val_loss += loss.item()
                all_preds.append(torch.argmax(logits_up, dim=1).cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = val_loss / len(val_loader)
        miou = mean_iou(torch.cat(all_preds), torch.cat(all_targets), NUM_CLASSES)
        current_lr = optimizer.param_groups[1]["lr"]   # head LR

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS}  "
            f"Train: {avg_train_loss:.4f}  "
            f"Val: {avg_val_loss:.4f}  "
            f"mIoU: {miou:.4f}  "
            f"LR: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save only the state_dict (raw weights, not HF config)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model → {SAVE_PATH}  (val loss {best_val_loss:.4f})")


if __name__ == "__main__":
    train()
