import os
import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.errors import RasterioIOError

ORI_DIRS = [
    r"E:\CG_Training_dataSet_3\Training_dataSet_3",
    r"E:\CG_Training_dataSet_2\Training_dataSet_2"
]

SHP_DIR = r"E:\CG_shp-file\shp-file"

def raster_bounds_polygon(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return geom, crs

results = {}
broken_oris = []

for ori_dir in ORI_DIRS:
    for ori_file in os.listdir(ori_dir):
        if not ori_file.lower().endswith(".tif"):
            continue

        ori_path = os.path.join(ori_dir, ori_file)

        try:
            ori_geom, ori_crs = raster_bounds_polygon(ori_path)
        except RasterioIOError as e:
            print(f"SKIPPING broken ORI: {ori_file}")
            broken_oris.append(ori_path)
            continue

        matched_shps = []

        for shp_file in os.listdir(SHP_DIR):
            if not shp_file.lower().endswith(".shp"):
                continue

            shp_path = os.path.join(SHP_DIR, shp_file)
            shp_gdf = gpd.read_file(shp_path)

            if shp_gdf.crs is None:
                continue

            if shp_gdf.crs != ori_crs:
                shp_gdf = shp_gdf.to_crs(ori_crs)

            if shp_gdf.intersects(ori_geom).any():
                matched_shps.append(shp_file)

        key = f"{os.path.basename(ori_dir)}/{ori_file}"
        results[key] = matched_shps

print("\n==== MAPPING RESULT ====")
for ori, shps in results.items():
    print(f"\nORI: {ori}")
    if shps:
        for s in shps:
            print(f"  -> SHP: {s}")
    else:
        print("  -> No matching SHP found")

print("\n==== BROKEN ORIs ====")
for b in broken_oris:
    print(b)
