from pathlib import Path

import numpy as np
import rasterio


TERRAIN_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

RUNOFF_DIR = Path("data/processed/rainfall")

EXPECTED_FILES = [
    "runoff_volume_000_030.tif",
    "runoff_volume_030_060.tif",
    "runoff_volume_060_090.tif",
    "runoff_volume_090_120.tif",
    "runoff_volume_120_150.tif",
    "runoff_volume_150_180.tif",
]


def main():
    with rasterio.open(TERRAIN_FILE) as terrain:
        expected_crs = terrain.crs
        expected_width = terrain.width
        expected_height = terrain.height
        expected_transform = terrain.transform

    all_valid = True

    print("Runoff raster validation")
    print()

    for filename in EXPECTED_FILES:
        path = RUNOFF_DIR / filename

        if not path.exists():
            print(f"MISSING: {filename}")
            all_valid = False
            continue

        with rasterio.open(path) as src:
            array = src.read(1)

            aligned = (
                src.crs == expected_crs
                and src.width == expected_width
                and src.height == expected_height
                and src.transform.almost_equals(
                    expected_transform
                )
            )

            if src.nodata is not None:
                valid = (
                    np.isfinite(array)
                    & (array != src.nodata)
                )
            else:
                valid = np.isfinite(array)

            values = array[valid]

            valid_cells = int(valid.sum())

            if len(values) > 0:
                minimum = float(values.min())
                maximum = float(values.max())
                total = float(values.sum())
            else:
                minimum = float("nan")
                maximum = float("nan")
                total = 0.0
                all_valid = False

            if valid_cells != 2750:
                all_valid = False

            if not aligned:
                all_valid = False

            print(filename)
            print(f"  CRS: {src.crs}")
            print(
                f"  Size: {src.width} x {src.height}"
            )
            print(f"  Resolution: {src.res}")
            print(f"  Valid AOI cells: {valid_cells}")
            print(
                f"  Value range: "
                f"{minimum:.2f} - {maximum:.2f} m³/cell"
            )
            print(
                f"  Total volume: "
                f"{total:.2f} m³"
            )
            print(f"  Terrain aligned: {aligned}")
            print()

    print(
        "ALL RUNOFF RASTERS VALID:"
        f" {all_valid}"
    )


if __name__ == "__main__":
    main()