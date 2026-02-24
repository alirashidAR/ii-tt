import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.enums import Resampling

# Paths
shp_path = r"E:\CG_shp-file\shp-file\Built_Up_Area_type.shp"
tif_path = r"E:\CG_live-demo\live-demo\PARAGAON_444686_ORTHO.tif"

# Load shapefile
gdf = gpd.read_file(shp_path)

# Open raster
with rasterio.open(tif_path) as src:
    raster_crs = src.crs
    
    # Reproject shapefile to raster CRS
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    
    # Downsample raster safely
    scale = 8
    data = src.read(
        out_shape=(src.count, src.height // scale, src.width // scale),
        resampling=Resampling.average
    )
    
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

if data.shape[0] >= 3:
    img = data[:3].transpose(1, 2, 0)
    show(data[:3], transform=transform, ax=ax)
else:
    show(data[0], transform=transform, ax=ax)

gdf.boundary.plot(ax=ax)

plt.title("Overlay After CRS Fix")
plt.show()
