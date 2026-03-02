"""
DeepLabV3 - 4 Class Semantic Segmentation
Dataset: water=0, background=1, built area=2, roads=3
Fixes: CUDA crash, BatchNorm, Windows multiprocessing, RGB-only (3ch)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR    = r"C:\Users\ragha\Downloads\train\train"
BATCH_SIZE  = 4
NUM_CLASSES = 4      # water=0, bg=1, built=2, roads=3
EPOCHS      = 30
LR          = 2e-4
SAVE_PATH   = "deeplabv3_4classes_final.pth"

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("❌ CPU mode")

# ============================================================
# DATASET
# ============================================================
class NpySegmentationDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir  = os.path.join(root_dir, "masks")
        self.filenames  = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".npy")])
        self.augment    = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load image: keep only RGB (first 3 channels), drop 4th channel
        img  = np.load(os.path.join(self.images_dir, fname)).astype(np.float32) / 255.0
        img  = img[:3, :, :]   # 🔥 (4,512,512) or (3,512,512) → (3,512,512)

        # Load mask: (H,W) int64 class indices [0,1,2,3]
        mask = np.load(os.path.join(self.masks_dir, fname)).astype(np.int64)

        img  = torch.from_numpy(img)    # FloatTensor (3,512,512)
        mask = torch.from_numpy(mask)   # LongTensor  (512,512)

        # Augmentation (train only)
        if self.augment:
            if torch.rand(1) > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)

        return img, mask

# ============================================================
# MODEL
# ============================================================
def build_model(num_classes, device):
    model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")

    # Replace main classifier head
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # Replace aux classifier head
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model.to(device)

# ============================================================
# TRAIN EPOCH
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            output = model(imgs)
            logits = output['out']              # (B,4,H,W)
            loss   = criterion(logits, masks)

            # Aux loss for training stability
            if 'aux' in output:
                loss = loss + 0.4 * criterion(output['aux'], masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

# ============================================================
# VAL EPOCH
# ============================================================
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Val  ", leave=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            logits = model(imgs)['out']
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

# ============================================================
# mIoU METRIC
# ============================================================
@torch.no_grad()
def compute_miou(model, loader, num_classes, device):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for imgs, masks in tqdm(loader, desc="mIoU ", leave=False):
        imgs  = imgs.to(device)
        preds = model(imgs)['out'].argmax(1).cpu()
        for p, t in zip(preds.view(-1), masks.view(-1)):
            confusion[t.long(), p.long()] += 1

    class_names = ["water", "background", "built", "roads"]
    iou_list = []
    for c in range(num_classes):
        tp    = confusion[c, c].item()
        fp    = confusion[:, c].sum().item() - tp
        fn    = confusion[c, :].sum().item() - tp
        denom = tp + fp + fn
        iou   = tp / denom if denom > 0 else 0.0
        iou_list.append(iou)
        print(f"   IoU {class_names[c]:<12}: {iou:.4f}")

    miou = np.mean(iou_list)
    print(f"   mIoU          : {miou:.4f}")
    return miou

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    # Quick shape check
    sample = np.load(os.path.join(ROOT_DIR, "images", os.listdir(os.path.join(ROOT_DIR, "images"))[0]))
    print(f"Raw image shape : {sample.shape}")
    print(f"After RGB slice : {sample[:3].shape}")

    # Datasets
    full_ds = NpySegmentationDataset(ROOT_DIR, augment=True)
    t_len   = int(0.8 * len(full_ds))
    v_len   = len(full_ds) - t_len
    train_ds, val_ds = random_split(full_ds, [t_len, v_len])
    print(f"✅ Train: {len(train_ds)} | Val: {len(val_ds)}")

    # DataLoaders
    # drop_last=True → prevents last batch size=1 (BatchNorm crash fix)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=0,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=True, drop_last=False
    )

    # Model, loss, optimizer
    model     = build_model(NUM_CLASSES, device)
    scaler    = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"✅ Model : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"✅ Config: {EPOCHS} epochs | batch={BATCH_SIZE} | LR={LR}\n")

    best_val = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS}  LR={scheduler.get_last_lr()[0]:.2e}")
        torch.cuda.empty_cache()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # mIoU every 5 epochs
        if epoch % 5 == 0:
            print("   Computing mIoU...")
            compute_miou(model, val_loader, NUM_CLASSES, device)

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch'  : epoch,
                'model'  : model.state_dict(),
                'optim'  : optimizer.state_dict(),
                'val_loss': val_loss,
                'classes': ['water', 'background', 'built', 'roads']
            }, SAVE_PATH)
            print(f"   💾 Saved BEST → {SAVE_PATH}")

    print(f"\n✅ Training complete! Best val loss: {best_val:.4f}")
    print(f"📦 Model: {SAVE_PATH}")
