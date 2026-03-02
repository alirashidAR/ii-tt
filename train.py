import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from model import get_sota_model


# ── Dataset ───────────────────────────────────────────────────────────────────
class GeoDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir  = Path(root_dir) / "images"
        self.mask_dir = Path(root_dir) / "masks"
        self.img_files = sorted(list(self.img_dir.glob("*.npy")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path  = self.img_files[idx]
        mask_path = self.mask_dir / img_path.name

        # Sanity: images in [0,1], masks in {0,1,2,3}
        image = np.load(img_path).astype(np.float32) / 255.0   # (C, H, W)
        mask  = np.load(mask_path).astype(np.int64)             # (H, W)

        return torch.from_numpy(image), torch.from_numpy(mask)


# ── Loss: Dice + Focal ────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    Dice + Focal loss for multi-class segmentation.

    Why this combination?
    - Dice loss  : maximises per-class overlap → tackles class imbalance
                   (BG >> Road/Built/Water), prevents thin-structure collapse.
    - Focal loss : down-weights easy BG pixels (gamma=2), focuses training
                   on hard, rare classes.  Better than plain CE for roads.

    Training loss > val loss is expected with DeepLabV3+ because BatchNorm
    uses noisy batch statistics during training vs stable running stats at eval.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.dice  = smp.losses.DiceLoss(
            mode="multiclass",
            classes=num_classes,
            smooth=1.0,
            from_logits=True,
        )
        self.focal = smp.losses.FocalLoss(
            mode="multiclass",
            gamma=2.0,           # standard focal gamma
            normalized=True,     # normalise per batch
        )

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.focal(logits, targets)


# ── Mean IoU ──────────────────────────────────────────────────────────────────
def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds   = preds.view(-1)
    targets = targets.view(-1)
    ious = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    # ── Config ────────────────────────────────────────────────────────────────
    DATA_DIR    = r"E:\dataset\train"
    BATCH_SIZE  = 4           # drop to 2 if OOM
    LR          = 1e-4
    EPOCHS      = 25
    NUM_CLASSES = 4           # 0=BG, 1=Road, 2=Built, 3=Water
    SAVE_PATH   = "best_model.pth"
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP     = torch.cuda.is_available()

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset    = GeoDataset(DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)   # reproducible split
    )

    # drop_last=True: prevents BatchNorm crash on the final size-1 batch
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # ── Sanity check data ─────────────────────────────────────────────────────
    sample_img, sample_mask = dataset[0]
    n_channels = sample_img.shape[0]
    print(f"Tiles        : {len(dataset)}  (train {train_size} | val {val_size})")
    print(f"Channels     : {n_channels}")
    print(f"Image  range : [{sample_img.min():.3f}, {sample_img.max():.3f}]  (expect 0–1)")
    print(f"Mask classes : {sample_mask.unique().tolist()}  (expect subset of [0,1,2,3])")
    print(f"Device       : {DEVICE}  |  AMP: {USE_AMP}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_sota_model(n_channels=n_channels, n_classes=NUM_CLASSES).to(DEVICE)
    print(f"Params       : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = CombinedLoss(num_classes=NUM_CLASSES)

    # ── Optimizer: differential LR (encoder is pretrained → fine-tune slowly) ─
    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(),           "lr": LR * 0.1},
        {"params": model.decoder.parameters(),           "lr": LR},
        {"params": model.segmentation_head.parameters(), "lr": LR},
    ], weight_decay=1e-4)

    # ── Scheduler: ReduceLROnPlateau ──────────────────────────────────────────
    # Halves LR if val loss doesn't improve for 3 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler       = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    best_val_loss = float("inf")

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
                loss    = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # -- Validate --
        model.eval()
        val_loss   = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  "):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = model(images)
                    loss    = criterion(outputs, masks)
                val_loss += loss.item()
                all_preds.append(torch.argmax(outputs, dim=1).cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = val_loss / len(val_loader)
        miou = mean_iou(torch.cat(all_preds), torch.cat(all_targets), NUM_CLASSES)

        # Step scheduler based on val loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[1]["lr"]   # decoder LR

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS}  "
            f"Train: {avg_train_loss:.4f}  "
            f"Val: {avg_val_loss:.4f}  "
            f"mIoU: {miou:.4f}  "
            f"LR: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model (val loss {best_val_loss:.4f})")


if __name__ == "__main__":
    train()
