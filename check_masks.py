"""
check_masks.py
--------------
Scans all mask tiles and reports:
  - Tiles with only 1 unique class (i.e., single-class masks)
  - A breakdown of class diversity across the dataset
  - Visualises a grid of the single-class tiles (first N)

Classes: 0=BG, 1=Road, 2=Built, 3=Water
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\dataset\train")
MASK_DIR   = DATA_DIR / "masks"
IMG_DIR    = DATA_DIR / "images"
MAX_VIS    = 12   # max single-class tiles to display in the grid

CLASS_NAMES  = {0: "BG only", 1: "Road only", 2: "Built only", 3: "Water only"}
CLASS_COLORS = {0: "grey",    1: "tomato",    2: "steelblue", 3: "mediumseagreen"}

# ── Scan ──────────────────────────────────────────────────────────────────────
mask_files = sorted(MASK_DIR.glob("*.npy"))
print(f"Total tiles: {len(mask_files)}")

single_class_files = []   # (path, sole_class)
class_count_dist   = Counter()   # how many unique classes a tile has

for mf in mask_files:
    mask = np.load(mf)
    unique = np.unique(mask)
    class_count_dist[len(unique)] += 1
    if len(unique) == 1:
        single_class_files.append((mf, int(unique[0])))

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Single-class tiles: {len(single_class_files)} / {len(mask_files)} "
      f"({100*len(single_class_files)/len(mask_files):.1f}%)")

# Which class dominates the single-class tiles?
sole_class_counter = Counter(cls for _, cls in single_class_files)
print("\nBreakdown of single-class tiles by class:")
for cls, count in sorted(sole_class_counter.items()):
    print(f"  Class {cls} ({CLASS_NAMES.get(cls, cls)}): {count} tiles")

print("\nClass diversity across ALL tiles (# unique classes per tile):")
for n_cls, count in sorted(class_count_dist.items()):
    print(f"  {n_cls} class(es): {count} tiles ({100*count/len(mask_files):.1f}%)")

# ── Visualise single-class tiles ──────────────────────────────────────────────
if not single_class_files:
    print("\nNo single-class tiles found. 🎉")
else:
    to_show = single_class_files[:MAX_VIS]
    n       = len(to_show)
    cols    = 4
    rows    = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3))
    axes = axes.flatten()

    for i, (mf, sole_cls) in enumerate(to_show):
        img_path = IMG_DIR / mf.name
        mask     = np.load(mf)
        img      = np.load(img_path).astype(np.uint8)

        ax_img  = axes[i * 2]
        ax_mask = axes[i * 2 + 1]

        ax_img.imshow(np.transpose(img[:3], (1, 2, 0)))
        ax_img.set_title(f"{mf.stem}\n(img)", fontsize=7)
        ax_img.axis("off")

        ax_mask.imshow(mask, cmap="tab10", vmin=0, vmax=3)
        ax_mask.set_title(f"Mask: {CLASS_NAMES.get(sole_cls, sole_cls)}", fontsize=7,
                          color=CLASS_COLORS.get(sole_cls, "black"))
        ax_mask.axis("off")

    # hide unused subplots
    for j in range(n * 2, len(axes)):
        axes[j].set_visible(False)

    legend = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
              for c in sorted(CLASS_COLORS)]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    plt.suptitle(f"Single-class tiles ({len(single_class_files)} total) — first {n} shown",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

# ── Optionally delete single-class tiles ─────────────────────────────────────
print(f"\nTo delete all {len(single_class_files)} single-class tiles, run:")
print("  python check_masks.py --delete")

import sys
if "--delete" in sys.argv:
    deleted = 0
    for mf, _ in single_class_files:
        img_path = IMG_DIR / mf.name
        mf.unlink(missing_ok=True)
        img_path.unlink(missing_ok=True)
        deleted += 1
    print(f"Deleted {deleted} tile pairs (image + mask).")
