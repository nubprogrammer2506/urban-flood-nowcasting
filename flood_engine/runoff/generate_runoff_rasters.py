import json
from pathlib import Path

import numpy as np
import rasterio


RAIN_FILE = Path(
    "data/raw/rainfall/heavy_rain_demo.json"
)

TEMPLATE_RASTER = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

COEFFICIENT_FILE = Path(
    "data/processed/landuse/runoff_coefficient.tif"
)

OUTPUT_DIR = Path(
    "data/processed/rainfall"
)

OUTPUT_NODATA = -9999.0


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------
    # Rainfall scenario
    # --------------------------------------------
    with open(
        RAIN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        rainfall_data = json.load(f)

    forecast = rainfall_data["forecast"]

    # --------------------------------------------
    # Terrain template
    # --------------------------------------------
    with rasterio.open(
        TEMPLATE_RASTER
    ) as terrain:

        profile = terrain.profile.copy()

        crs = terrain.crs
        transform = terrain.transform

        width = terrain.width
        height = terrain.height

        dem = terrain.read(1)

        if terrain.nodata is None:
            terrain_valid = np.isfinite(dem)
        else:
            terrain_valid = (
                np.isfinite(dem)
                & (dem != terrain.nodata)
            )

        cell_width = abs(transform.a)
        cell_height = abs(transform.e)

        cell_area_m2 = (
            cell_width * cell_height
        )

    # --------------------------------------------
    # Spatial runoff coefficients
    # --------------------------------------------
    with rasterio.open(
        COEFFICIENT_FILE
    ) as coeff_src:

        if (
            coeff_src.crs != crs
            or coeff_src.width != width
            or coeff_src.height != height
            or not coeff_src.transform.almost_equals(
                transform
            )
        ):
            raise ValueError(
                "Runoff coefficient raster "
                "is not aligned with terrain"
            )

        coefficient = coeff_src.read(1)

        if coeff_src.nodata is None:
            coefficient_valid = (
                np.isfinite(coefficient)
            )
        else:
            coefficient_valid = (
                np.isfinite(coefficient)
                & (
                    coefficient
                    != coeff_src.nodata
                )
            )

    valid_cells = (
        terrain_valid
        & coefficient_valid
    )

    cell_count = int(
        valid_cells.sum()
    )

    coefficients = coefficient[
        valid_cells
    ]

    print(
        "Spatial rainfall -> runoff generation"
    )
    print()

    print(f"CRS: {crs}")

    print(
        f"Raster size: "
        f"{width} x {height}"
    )

    print(
        f"Cell size: "
        f"{cell_width:.2f} x "
        f"{cell_height:.2f} m"
    )

    print(
        f"Cell area: "
        f"{cell_area_m2:.2f} m²"
    )

    print(
        f"Valid AOI cells: "
        f"{cell_count}"
    )

    print(
        f"Coefficient range: "
        f"{coefficients.min():.2f} - "
        f"{coefficients.max():.2f}"
    )

    print(
        f"Mean coefficient: "
        f"{coefficients.mean():.3f}"
    )

    print()

    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    total_rainfall_mm = 0.0
    total_runoff_volume_m3 = 0.0

    # --------------------------------------------
    # Generate spatial runoff for each interval
    # --------------------------------------------
    for i in range(
        len(forecast) - 1
    ):

        current = forecast[i]
        next_step = forecast[i + 1]

        start_min = current["minutes"]
        end_min = next_step["minutes"]

        duration_hours = (
            end_min - start_min
        ) / 60.0

        start_intensity = (
            current["rainfall_mm_hr"]
        )

        end_intensity = (
            next_step["rainfall_mm_hr"]
        )

        average_intensity = (
            start_intensity
            + end_intensity
        ) / 2.0

        rainfall_mm = (
            average_intensity
            * duration_hours
        )

        rainfall_depth_m = (
            rainfall_mm / 1000.0
        )

        # ----------------------------------------
        # Spatial runoff calculation
        #
        # rainfall depth
        # × local runoff coefficient
        # × cell area
        # ----------------------------------------
        runoff_volume = np.full(
            (height, width),
            OUTPUT_NODATA,
            dtype=np.float32,
        )

        spatial_volume = (
            rainfall_depth_m
            * coefficient[valid_cells]
            * cell_area_m2
        )

        runoff_volume[
            valid_cells
        ] = spatial_volume.astype(
            np.float32
        )

        interval_total = float(
            spatial_volume.sum()
        )

        total_rainfall_mm += (
            rainfall_mm
        )

        total_runoff_volume_m3 += (
            interval_total
        )

        output_file = (
            OUTPUT_DIR
            / (
                f"runoff_volume_"
                f"{start_min:03d}_"
                f"{end_min:03d}.tif"
            )
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(
                runoff_volume,
                1
            )

            dst.update_tags(
                units="m3_per_cell",
                start_minutes=start_min,
                end_minutes=end_min,
                rainfall_mm=rainfall_mm,
                coefficient_source=(
                    "spatial_runoff_"
                    "coefficient_raster"
                ),
                scenario=(
                    rainfall_data[
                        "scenario_id"
                    ]
                ),
            )

        print(
            f"{start_min:>3}-"
            f"{end_min:>3} min | "
            f"Rain {rainfall_mm:>6.2f} mm | "
            f"Cell runoff "
            f"{spatial_volume.min():>6.2f}"
            f"-"
            f"{spatial_volume.max():>6.2f} m³ | "
            f"AOI "
            f"{interval_total:>10.2f} m³"
        )

    print()
    print(
        f"Total rainfall: "
        f"{total_rainfall_mm:.2f} mm"
    )

    print(
        "Total spatial runoff volume: "
        f"{total_runoff_volume_m3:.2f} m³"
    )

    print()
    print(
        "Spatial runoff rasters "
        "generated successfully."
    )


if __name__ == "__main__":
    main()