import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from model import get_sota_model


# ------------------------------------------------------------------ #
# Class metadata
# ------------------------------------------------------------------ #

CLASS_COLORS = {
    0: (0.15, 0.15, 0.15),   # Background  – dark grey
    1: (0.95, 0.70, 0.10),   # Road        – amber
    2: (0.85, 0.20, 0.20),   # Built-up    – red
    3: (0.10, 0.45, 0.85),   # Water       – blue
}
CLASS_NAMES = {0: "Background", 1: "Road", 2: "Built-up", 3: "Water"}


# ------------------------------------------------------------------ #
# Model loading — auto-detects architecture from checkpoint keys
# ------------------------------------------------------------------ #

def _load_model(model_path, device):
    """
    Returns (model, n_channels, n_classes, arch_str, is_segformer).

    Detected architectures
    ----------------------
    SegFormer    keys start with  "segformer."  or  "decode_head."
    DeepLabV3+   keys start with  "encoder." / "decoder." / "segmentation_head."
    MiniUNet     keys start with  "inc."  (legacy)
    """
    checkpoint = torch.load(model_path, map_location=device)

    segformer_keys = [k for k in checkpoint
                      if k.startswith("segformer.") or k.startswith("decode_head.")]

    smp_keys       = [k for k in checkpoint
                      if k.startswith("encoder.") or k.startswith("decoder.") or
                         k.startswith("segmentation_head.")]

    # ── SegFormer ──────────────────────────────────────────────────── #
    if segformer_keys:
        from transformers import SegformerForSemanticSegmentation

        # n_classes from the classifier conv weight (decode_head.classifier.weight)
        cls_key = next(
            (k for k in checkpoint
             if "classifier.weight" in k and checkpoint[k].ndim == 4),
            None
        )
        n_classes  = checkpoint[cls_key].shape[0] if cls_key else 4
        n_channels = 3   # SegFormer is always 3-channel

        label2id = {str(i): i for i in range(n_classes)}
        id2label  = {i: str(i) for i in range(n_classes)}

        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b4",
            num_labels=n_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ).to(device)
        model.load_state_dict(checkpoint)

        print(f"[SegFormer-B4] Detected  n_channels={n_channels}, n_classes={n_classes}")
        return model, n_channels, n_classes, "SegFormer-B4", True

    # ── SMP DeepLabV3+ ─────────────────────────────────────────────── #
    elif smp_keys:
        encoder_conv_key = next(
            (k for k in checkpoint
             if "conv" in k and k.startswith("encoder.") and checkpoint[k].ndim == 4),
            None
        )
        n_channels = checkpoint[encoder_conv_key].shape[1] if encoder_conv_key else 3

        head_key = next(
            (k for k in checkpoint
             if k.startswith("segmentation_head") and checkpoint[k].ndim == 4),
            None
        )
        n_classes = checkpoint[head_key].shape[0] if head_key else 4

        print(f"[DeepLabV3+]  Detected  n_channels={n_channels}, n_classes={n_classes}")
        model = get_sota_model(n_channels=n_channels, n_classes=n_classes).to(device)
        model.load_state_dict(checkpoint)
        return model, n_channels, n_classes, "DeepLabV3+", False

    # ── Legacy MiniUNet ────────────────────────────────────────────── #
    else:
        from model import MiniUNet
        n_channels = checkpoint["inc.double_conv.0.weight"].shape[1]
        n_classes  = checkpoint["outc.weight"].shape[0]
        print(f"[MiniUNet]    Detected  n_channels={n_channels}, n_classes={n_classes}")
        model = MiniUNet(n_channels=n_channels, n_classes=n_classes).to(device)
        model.load_state_dict(checkpoint)
        return model, n_channels, n_classes, "MiniUNet", False


# ------------------------------------------------------------------ #
# Inference helper
# ------------------------------------------------------------------ #

def _preprocess_segformer(image_np):
    """
    Normalise (C, H, W) [0-255] float32 array for SegFormer:
    Scale to [0,1] then apply ImageNet mean/std.
    """
    from transformers import SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/mit-b4", do_resize=False, do_rescale=False, do_normalize=True
    )
    img = image_np[:3] / 255.0                                # (3, H, W)
    enc = processor(images=img, return_tensors="pt", do_rescale=False)
    return enc["pixel_values"]                                # (1, 3, H, W)


def _infer(model, image_np, device, is_segformer=False):
    """
    Run inference on a single (C, H, W) float32 numpy array (uint8 scale [0-255]).
    Returns pred (H, W) and confidence (H, W).
    """
    H, W = image_np.shape[1], image_np.shape[2]

    if is_segformer:
        pixel_values = _preprocess_segformer(image_np).to(device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits    # (1, C, H/4, W/4)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    else:
        image_tensor = torch.from_numpy(image_np / 255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image_tensor)
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

    pred       = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    confidence = probs.max(axis=0)
    return pred, confidence


def _label_to_rgb(label_map, n_classes):
    H, W = label_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        if cls_id < n_classes:
            rgb[label_map == cls_id] = color
    return rgb


def _legend_patches(n_classes):
    return [mpatches.Patch(color=c, label=CLASS_NAMES[i])
            for i, c in CLASS_COLORS.items() if i < n_classes]


# ------------------------------------------------------------------ #
# Single-tile visualisation
# ------------------------------------------------------------------ #

def visualize_prediction(model_path, data_dir, tile_idx=0):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_channels, n_classes, arch, is_sf = _load_model(model_path, DEVICE)
    model.eval()

    img_dir   = Path(data_dir) / "images"
    mask_dir  = Path(data_dir) / "masks"
    img_files = sorted(img_dir.glob("*.npy"))

    img_path  = img_files[tile_idx]
    mask_path = mask_dir / img_path.name

    image_np = np.load(img_path).astype(np.float32)
    mask_np  = np.load(mask_path).astype(np.uint8)

    pred, confidence = _infer(model, image_np, DEVICE, is_segformer=is_sf)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Inference — {arch}  |  tile: {img_path.name}", fontsize=13)

    vis_rgb = np.clip(np.transpose(image_np[:3], (1, 2, 0)), 0, 255).astype(np.uint8)
    axes[0].imshow(vis_rgb)
    axes[0].set_title(f"Image ({n_channels} ch)")
    axes[0].axis("off")

    axes[1].imshow(_label_to_rgb(mask_np, n_classes))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(_label_to_rgb(pred, n_classes))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    im = axes[3].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Confidence")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.legend(handles=_legend_patches(n_classes), loc="lower center",
               ncol=n_classes, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = "prediction_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualisation → {out_path}")
    plt.show()


# ------------------------------------------------------------------ #
# Batch grid: first N images   → stitched PNG
# Layout per row: [RGB | GT | Pred]
# ------------------------------------------------------------------ #

def batch_predict_grid(model_path, data_dir, n_images=20, out_path="batch_grid.png",
                       cols=4, thumb_size=160):
    """
    Run inference on the first `n_images` tiles and stitch RGB | GT | Pred
    thumbnails into a dark-background grid saved to `out_path`.
    """
    from PIL import Image as PILImage

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_channels, n_classes, arch, is_sf = _load_model(model_path, DEVICE)
    model.eval()

    img_dir   = Path(data_dir) / "images"
    mask_dir  = Path(data_dir) / "masks"
    img_files = sorted(img_dir.glob("*.npy"))[:n_images]

    if not img_files:
        raise FileNotFoundError(f"No .npy files found in {img_dir}")

    actual_n = len(img_files)
    print(f"Processing {actual_n} tiles with {arch} …")

    # ── collect thumbnails ─────────────────────────────────────────── #
    thumbs = []

    for idx, img_path in enumerate(img_files):
        mask_path = mask_dir / img_path.name
        image_np  = np.load(img_path).astype(np.float32)
        mask_np   = np.load(mask_path).astype(np.uint8) if mask_path.exists() else None

        pred, _ = _infer(model, image_np, DEVICE, is_segformer=is_sf)

        vis_rgb = np.clip(np.transpose(image_np[:3], (1, 2, 0)), 0, 255).astype(np.uint8)
        rgb_t   = np.array(PILImage.fromarray(vis_rgb).resize(
                      (thumb_size, thumb_size), PILImage.BILINEAR))

        if mask_np is not None:
            gt_rgb = (np.clip(_label_to_rgb(mask_np, n_classes), 0, 1) * 255).astype(np.uint8)
            gt_t   = np.array(PILImage.fromarray(gt_rgb).resize(
                         (thumb_size, thumb_size), PILImage.NEAREST))
        else:
            gt_t = np.full((thumb_size, thumb_size, 3), 80, dtype=np.uint8)

        pred_rgb = (np.clip(_label_to_rgb(pred, n_classes), 0, 1) * 255).astype(np.uint8)
        pred_t   = np.array(PILImage.fromarray(pred_rgb).resize(
                       (thumb_size, thumb_size), PILImage.NEAREST))

        thumbs.append((rgb_t, gt_t, pred_t))
        print(f"  [{idx + 1:>2}/{actual_n}] {img_path.name}")

    # ── build canvas ──────────────────────────────────────────────── #
    GAP      = 3
    HDR      = 24                          # header height (px)
    LBL      = 14                          # per-cell label height
    group_w  = thumb_size * 3 + GAP * 2
    group_h  = thumb_size + LBL
    n_rows   = (actual_n + cols - 1) // cols
    canvas_w = cols * group_w + (cols + 1) * GAP
    canvas_h = n_rows * group_h + (n_rows + 1) * GAP + HDR

    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)

    for gi, (rgb_t, gt_t, pred_t) in enumerate(thumbs):
        row = gi // cols
        col = gi % cols
        gx  = GAP + col * (group_w + GAP)
        gy  = HDR + GAP + row * (group_h + GAP) + LBL
        for si, sub in enumerate([rgb_t, gt_t, pred_t]):
            x0 = gx + si * (thumb_size + GAP)
            canvas[gy: gy + thumb_size, x0: x0 + thumb_size] = sub

    # ── matplotlib figure (for text labels + legend) ──────────────── #
    dpi  = 150
    fig  = plt.figure(figsize=(canvas_w / dpi, canvas_h / dpi), dpi=dpi)
    ax   = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#1c1c1c")
    ax.set_facecolor("#1c1c1c")
    ax.imshow(canvas, aspect="auto")
    ax.axis("off")
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)

    # Header
    ax.text(GAP, 4, f"{arch}  ·  first {actual_n} tiles  ·  RGB | GT | Pred",
            color="white", fontsize=6, va="top", fontfamily="monospace")

    # Per-group labels
    for gi in range(actual_n):
        row = gi // cols
        col = gi % cols
        gx  = GAP + col * (group_w + GAP)
        gy  = HDR + GAP + row * (group_h + GAP)
        name = img_files[gi].stem[:16]
        ax.text(gx, gy + 1, f"#{gi:02d} {name}", color="#cccccc",
                fontsize=4.5, va="top", fontfamily="monospace")
        for si, lbl in enumerate(["RGB", "GT", "Pred"]):
            ax.text(gx + si * (thumb_size + GAP), gy + LBL + thumb_size + 1,
                    lbl, color="#888888", fontsize=4.5, va="top", fontfamily="monospace")

    # Legend
    patches = _legend_patches(n_classes)
    fig.legend(handles=patches, loc="lower center", ncol=n_classes,
               frameon=False, fontsize=6, labelcolor="white",
               bbox_to_anchor=(0.5, 0.0))

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved stitched grid → {out_path}")
    plt.show()


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    DATA_DIR = r"E:\dataset\train"

    # ── Switch between models by changing MODEL_PATH ─────────────── #
    # MODEL_PATH = "best_model.pth"          # DeepLabV3+
    MODEL_PATH = "best_segformer.pth"    # SegFormer-B4

    if not Path(MODEL_PATH).exists():
        print(f"Model file '{MODEL_PATH}' not found. Train first.")
    else:
        # Single tile
        # visualize_prediction(MODEL_PATH, DATA_DIR, tile_idx=35)

        # Batch grid — first 20 images
        batch_predict_grid(
            model_path = MODEL_PATH,
            data_dir   = DATA_DIR,
            n_images   = 20,
            out_path   = "batch_grid.png",
            cols       = 4,
            thumb_size = 160,
        )
