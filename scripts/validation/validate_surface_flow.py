from pathlib import Path

import numpy as np
import rasterio


TERRAIN_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

FLOW_DIR = Path(
    "data/outputs/surface_flow"
)

FILES = [
    "routed_runoff_000_030.tif",
    "routed_runoff_030_060.tif",
    "routed_runoff_060_090.tif",
    "routed_runoff_090_120.tif",
    "routed_runoff_120_150.tif",
    "routed_runoff_150_180.tif",
]


def main():

    with rasterio.open(TERRAIN_FILE) as terrain:
        crs = terrain.crs
        width = terrain.width
        height = terrain.height
        transform = terrain.transform

    all_valid = True

    print("Surface-flow raster validation")
    print()

    for filename in FILES:

        path = FLOW_DIR / filename

        if not path.exists():
            print(f"MISSING: {filename}")
            all_valid = False
            continue

        with rasterio.open(path) as src:

            data = src.read(1)

            aligned = (
                src.crs == crs
                and src.width == width
                and src.height == height
                and src.transform.almost_equals(transform)
            )

            if src.nodata is None:
                valid = np.isfinite(data)
            else:
                valid = (
                    np.isfinite(data)
                    & (data != src.nodata)
                )

            values = data[valid]

            positive = int((values > 0).sum())
            zero = int((values == 0).sum())
            negative = int((values < 0).sum())

            minimum = float(values.min())
            maximum = float(values.max())
            mean = float(values.mean())

            if not aligned:
                all_valid = False

            if negative > 0:
                all_valid = False

            print(filename)
            print(f"  CRS: {src.crs}")
            print(f"  Size: {src.width} x {src.height}")
            print(f"  Valid cells: {len(values)}")
            print(f"  Positive cells: {positive}")
            print(f"  Zero cells: {zero}")
            print(f"  Negative cells: {negative}")
            print(f"  Min: {minimum:.2f} m³")
            print(f"  Mean: {mean:.2f} m³")
            print(f"  Max: {maximum:.2f} m³")
            print(f"  Terrain aligned: {aligned}")
            print()

    print(
        f"ALL SURFACE-FLOW RASTERS VALID: {all_valid}"
    )


if __name__ == "__main__":
    main()