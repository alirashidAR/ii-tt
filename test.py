import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import MiniUNet

def visualize_prediction(model_path, data_dir, tile_idx=0):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    # Detect channels from data
    img_dir = Path(data_dir) / "images"
    img_files = sorted(list(img_dir.glob("*.npy")))
    sample_img = np.load(img_files[0])
    n_channels = sample_img.shape[0]

    model = MiniUNet(n_channels=n_channels, n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load image and mask
    img_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"
    img_files = sorted(list(img_dir.glob("*.npy")))
    
    img_path = img_files[tile_idx]
    mask_path = mask_dir / img_path.name

    image_np = np.load(img_path).astype(np.float32)
    mask_np = np.load(mask_path).astype(np.uint8)

    # Prepare for model
    image_tensor = torch.from_numpy(image_np / 255.0).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display RGB (transpose from C,H,W to H,W,C)
    # If 4 channels, take first 3 for visualization
    vis_img = image_np[:3, :, :]
    ax[0].imshow(np.transpose(vis_img, (1, 2, 0)).astype(np.uint8))
    ax[0].set_title(f"Image ({n_channels} ch)")
    ax[0].axis("off")

    ax[1].imshow(mask_np, cmap="tab10", vmin=0, vmax=3)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=3)
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_DIR = r"E:\PB_training_dataSet_shp_file\dataset\train"
    MODEL_PATH = "best_model.pth"
    
    if Path(MODEL_PATH).exists():
        visualize_prediction(MODEL_PATH, DATA_DIR, tile_idx=7)
    else:
        print(f"Model file {MODEL_PATH} not found. Train the model first using train.py.")
