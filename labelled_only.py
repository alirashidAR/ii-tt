import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.errors import RasterioIOError

ORI_DIRS = [
    r"E:\CG_Training_dataSet_3\Training_dataSet_3",
    r"E:\CG_Training_dataSet_2\Training_dataSet_2"
]

SHP_DIR = r"E:\CG_shp-file\shp-file"
OUTPUT_DIR = r"E:\CG_labeled_ori"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def raster_bounds_polygon(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        print(f'Resolution: {src}')
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return geom, crs

broken_oris = []

for ori_dir in ORI_DIRS:
    for ori_file in os.listdir(ori_dir):
        if not ori_file.lower().endswith(".tif"):
            continue

        ori_path = os.path.join(ori_dir, ori_file)

        try:
            ori_geom, ori_crs = raster_bounds_polygon(ori_path)
        except RasterioIOError:
            print(f"SKIPPING broken ORI: {ori_file}")
            broken_oris.append(ori_path)
            continue

        labeled_geometries = []

        for shp_file in os.listdir(SHP_DIR):
            if not shp_file.lower().endswith(".shp"):
                continue

            shp_path = os.path.join(SHP_DIR, shp_file)
            gdf = gpd.read_file(shp_path)

            if gdf.crs is None:
                continue

            if gdf.crs != ori_crs:
                gdf = gdf.to_crs(ori_crs)

            clipped = gdf[gdf.intersects(ori_geom)]
            if not clipped.empty:
                labeled_geometries.extend(clipped.geometry)

        if not labeled_geometries:
            print(f"NO LABELS found for ORI: {ori_file}")
            continue

        labeled_union = gpd.GeoSeries(labeled_geometries).unary_union

        try:
            with rasterio.open(ori_path) as src:
                out_image, out_transform = mask(
                    src,
                    [labeled_union],
                    crop=True,
                    all_touched=True
                )

                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                out_path = os.path.join(OUTPUT_DIR, ori_file)
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)

                print(f"SAVED labeled ORI: {ori_file}")

        except Exception as e:
            print(f"FAILED clipping {ori_file}: {e}")

print("\n==== BROKEN ORIs ====")
for b in broken_oris:
    print(b)
