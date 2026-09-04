from pathlib import Path

import numpy as np
import rasterio


FLOOD_DIR = Path(
    "data/outputs/flood_depth"
)

TERRAIN_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_utm.tif"
)

INTERVALS = [
    "000_030",
    "030_060",
    "060_090",
    "090_120",
    "120_150",
    "150_180",
]


def main():

    with rasterio.open(TERRAIN_FILE) as terrain:
        crs = terrain.crs
        width = terrain.width
        height = terrain.height
        transform = terrain.transform

    all_valid = True

    print("Flood-depth validation")
    print()

    for interval in INTERVALS:

        path = (
            FLOOD_DIR
            / f"flood_depth_{interval}.tif"
        )

        with rasterio.open(path) as src:

            data = src.read(1)

            aligned = (
                src.crs == crs
                and src.width == width
                and src.height == height
                and src.transform.almost_equals(
                    transform
                )
            )

            if src.nodata is None:
                valid = np.isfinite(data)
            else:
                valid = (
                    np.isfinite(data)
                    & (data != src.nodata)
                )

            values = data[valid]

            positive = values[
                values > 0
            ]

            negative_count = int(
                (values < 0).sum()
            )

            if len(positive) > 0:

                p50 = float(
                    np.percentile(
                        positive, 50
                    )
                )

                p90 = float(
                    np.percentile(
                        positive, 90
                    )
                )

                p95 = float(
                    np.percentile(
                        positive, 95
                    )
                )

                p99 = float(
                    np.percentile(
                        positive, 99
                    )
                )

                maximum = float(
                    positive.max()
                )

                max_index = np.nanargmax(
                    np.where(
                        valid,
                        data,
                        np.nan
                    )
                )

                row, col = np.unravel_index(
                    max_index,
                    data.shape
                )

                x, y = rasterio.transform.xy(
                    src.transform,
                    row,
                    col,
                    offset="center",
                )

            else:
                p50 = p90 = p95 = p99 = 0.0
                maximum = 0.0
                x = y = float("nan")

            print(interval)
            print(
                f"  Terrain aligned: {aligned}"
            )
            print(
                f"  Negative cells: {negative_count}"
            )
            print(
                f"  Positive cells: {len(positive)}"
            )
            print(f"  P50: {p50:.3f} m")
            print(f"  P90: {p90:.3f} m")
            print(f"  P95: {p95:.3f} m")
            print(f"  P99: {p99:.3f} m")
            print(f"  Max: {maximum:.3f} m")
            print(
                f"  Max location: "
                f"{x:.2f}, {y:.2f}"
            )
            print()

            if not aligned:
                all_valid = False

            if negative_count > 0:
                all_valid = False

    print(
        "FLOOD DEPTH RASTERS VALID: "
        f"{all_valid}"
    )


if __name__ == "__main__":
    main()