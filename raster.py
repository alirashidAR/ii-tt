import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import os
import pandas as pd
from rasterio.plot import show
from rasterio.enums import Resampling

# Folders
shp_folder = r"E:\CG_shp-file\shp-file"
tif_folder = r"E:\CG_live-demo\live-demo"

# 1. Load and merge all shapefiles
gdf_list = []

for file in os.listdir(shp_folder):
    if file.endswith(".shp"):
        path = os.path.join(shp_folder, file)
        gdf = gpd.read_file(path)
        gdf_list.append(gdf)

combined_gdf = gpd.GeoDataFrame(
    pd.concat(gdf_list, ignore_index=True)
)

# 2. Create plot
fig, ax = plt.subplots(figsize=(14, 14))

# 3. Plot all TIFFs
for file in os.listdir(tif_folder):
    if file.endswith(".tif"):
        tif_path = os.path.join(tif_folder, file)
        
        with rasterio.open(tif_path) as src:
            
            # Reproject vectors once based on first raster
            if combined_gdf.crs != src.crs:
                combined_gdf = combined_gdf.to_crs(src.crs)
            
            # Downsample raster
            scale = 16  # increase if slow
            data = src.read(
                out_shape=(src.count, src.height // scale, src.width // scale),
                resampling=Resampling.average
            )
            
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )
            
            if data.shape[0] >= 3:
                show(data[:3], transform=transform, ax=ax)
            else:
                show(data[0], transform=transform, ax=ax)

# 4. Plot all shapefiles on top
combined_gdf.boundary.plot(ax=ax, linewidth=0.5)

plt.title("All Orthophotos + All Shapefiles")
plt.show()
