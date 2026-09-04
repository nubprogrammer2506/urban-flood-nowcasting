from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEM_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
)

DEM_FILES = [
    DEM_FOLDER / "rajendra_nagar_dem_raw.tif",
    DEM_FOLDER / "rajendra_nagar_dem_large_raw.tif",
]


def inspect_dem(path):
    print("\n========================================")
    print(f"FILE: {path.name}")
    print("========================================")

    if not path.exists():
        print("ERROR: File not found.")
        return

    with rasterio.open(path) as src:

        data = src.read(1, masked=True)

        print(f"CRS             : {src.crs}")
        print(f"Width           : {src.width}")
        print(f"Height          : {src.height}")
        print(f"Resolution      : {src.res}")
        print(f"Data type       : {src.dtypes[0]}")
        print(f"NoData          : {src.nodata}")

        print("\nBounds")
        print(f"Left            : {src.bounds.left}")
        print(f"Bottom          : {src.bounds.bottom}")
        print(f"Right           : {src.bounds.right}")
        print(f"Top             : {src.bounds.top}")

        valid = data.compressed()

        if len(valid) > 0:
            print("\nElevation statistics")
            print(f"Minimum         : {valid.min():.2f} m")
            print(f"Maximum         : {valid.max():.2f} m")
            print(f"Mean            : {valid.mean():.2f} m")
            print(f"Valid pixels    : {len(valid)}")
        else:
            print("\nERROR: DEM contains no valid pixels.")


def main():
    print("\n=== DEM CANDIDATE INSPECTION ===")

    for dem_file in DEM_FILES:
        inspect_dem(dem_file)

    print("\nInspection completed.\n")


if __name__ == "__main__":
    main()