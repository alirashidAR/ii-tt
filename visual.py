# import numpy as np
# import matplotlib.pyplot as plt

# item_path = r"E:\PB_training_dataSet_shp_file\dataset\train\images\T003_001141.npy"
# img = np.load(item_path)

# print(f"Image shape: {img.shape}")
# for i in range(img.shape[0]):
#     print(f"Channel {i} - Min: {img[i].min()}, Max: {img[i].max()}, Unique: {len(np.unique(img[i]))}")

# # Convert from (C, H, W) to (H, W, C)
# img_vis = np.transpose(img, (1, 2, 0))

# # For display, if 4 channels, just show first 3 as RGB
# if img_vis.shape[2] == 4:
#     plt.imshow(img_vis[:, :, :3].astype(np.uint8))
# else:
#     plt.imshow(img_vis.astype(np.uint8))
# plt.title("Image tile")
# plt.axis("off")
# plt.show()


# mask = np.load(r"E:\PB_training_dataSet_shp_file\dataset\train\masks\T003_001141.npy")

# plt.figure(figsize=(5, 5))
# plt.imshow(mask, cmap="tab10")
# plt.colorbar(label="Class ID")
# plt.title("Mask")
# plt.axis("off")
# plt.show()

import rasterio
print(rasterio.supported_drivers)

# with rasterio.open(r"E:\PB_training_dataSet_shp_file\PB_training_dataSet_shp_file\37458_fattu_bhila_ortho_3857.ecw") as src:
#     print("Band count:", src.count)
#     print("Band descriptions:", src.descriptions)