import rasterio
import json
from pathlib import Path

DATASETS = [
    {
        "name": "CG",
        "tif_dir": Path(r"E:\CG\tifs"),
    },
    {
        "name": "PB",
        "tif_dir": Path(r"E:\PB_training_dataSet_shp_file\PB_training_dataSet_shp_file\tifs"),
    }
]

def check_metadata():
    results = {}
    for ds in DATASETS:
        ds_name = ds["name"]
        results[ds_name] = []
        tif_dir = ds["tif_dir"]
        
        if not tif_dir.exists():
            print(f"Directory {tif_dir} does not exist.")
            continue
            
        tifs = sorted(tif_dir.glob("*.tif"))
        for tif_path in tifs:
            try:
                with rasterio.open(tif_path) as src:
                    info = {
                        "filename": tif_path.name,
                        "channels": src.count,
                        "dtypes": [str(d) for d in src.dtypes],
                        "crs": str(src.crs),
                        "width": src.width,
                        "height": src.height,
                        "transform": [float(x) for x in src.transform] if src.transform else None
                    }
                    results[ds_name].append(info)
            except Exception as e:
                results[ds_name].append({"filename": tif_path.name, "error": str(e)})

    with open("tif_metadata.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Metadata report saved to tif_metadata.json")

if __name__ == "__main__":
    check_metadata()
