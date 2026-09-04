from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DEM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "rajendra_nagar_dem_large_raw.tif"
)

OUTPUT_DEM = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "terrain"
    / "rajendra_nagar_dem_utm43n.tif"
)

TARGET_CRS = "EPSG:32643"

# Approximate source DEM resolution is ~30 m.
TARGET_RESOLUTION = 30

NODATA = -9999.0


def main():
    print("\n=== PREPARE DEM FOR DRAINAGE ===\n")

    if not INPUT_DEM.exists():
        raise FileNotFoundError(
            f"Input DEM not found: {INPUT_DEM}"
        )

    OUTPUT_DEM.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(INPUT_DEM) as src:

        print(f"Input DEM          : {INPUT_DEM.name}")
        print(f"Input CRS          : {src.crs}")
        print(
            f"Input size         : "
            f"{src.width} x {src.height}"
        )
        print(f"Input resolution   : {src.res}")

        transform, width, height = (
            calculate_default_transform(
                src.crs,
                TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
                resolution=TARGET_RESOLUTION,
            )
        )

        profile = src.profile.copy()

        profile.update(
            {
                "crs": TARGET_CRS,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "float32",
                "nodata": NODATA,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            OUTPUT_DEM,
            "w",
            **profile,
        ) as dst:

            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA,

                # Bilinear is suitable for continuous
                # elevation surfaces.
                resampling=Resampling.bilinear,
            )

    # -----------------------------------------------------
    # Validate generated raster
    # -----------------------------------------------------

    with rasterio.open(OUTPUT_DEM) as dem:

        elevation = dem.read(
            1,
            masked=True,
        )

        valid = elevation.compressed()

        print("\n=== OUTPUT DEM ===")

        print(f"CRS                : {dem.crs}")

        print(
            f"Size               : "
            f"{dem.width} x {dem.height}"
        )

        print(
            f"Resolution         : "
            f"{dem.res}"
        )

        print(f"NoData             : {dem.nodata}")

        if len(valid):

            print(
                f"Minimum elevation  : "
                f"{np.min(valid):.2f} m"
            )

            print(
                f"Maximum elevation  : "
                f"{np.max(valid):.2f} m"
            )

            print(
                f"Mean elevation     : "
                f"{np.mean(valid):.2f} m"
            )

            print(
                f"Valid pixels       : "
                f"{len(valid)}"
            )

        print("\nBounds:")
        print(dem.bounds)

    print("\nPrepared DEM saved:")
    print(OUTPUT_DEM)

    print(
        "\nDEM preparation completed."
    )


if __name__ == "__main__":
    main()