import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import MiniUNet

class GeoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.img_files = sorted(list(self.img_dir.glob("*.npy")))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.load(img_path).astype(np.float32) / 255.0  # Normalize to [0, 1]
        mask = np.load(mask_path).astype(np.longlong)

        # image is (3, 512, 512), mask is (512, 512)
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask

def train():
    # --- Config ---
    DATA_DIR = r"E:\PB_training_dataSet_shp_file\dataset\train"
    BATCH_SIZE = 4
    LR = 1e-4
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4
    SAVE_PATH = "best_model.pth"

    # --- Dataset & Loader ---
    dataset = GeoDataset(DATA_DIR)
    # Simple split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model, Loss, Optimizer ---
    # Detect input channels from first sample
    sample_img, _ = dataset[0]
    n_channels = sample_img.shape[0]
    print(f"Detected {n_channels} input channels.")

    model = MiniUNet(n_channels=n_channels, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to {SAVE_PATH}")

if __name__ == "__main__":
    train()
