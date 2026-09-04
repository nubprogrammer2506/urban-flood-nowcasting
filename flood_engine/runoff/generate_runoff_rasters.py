import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


RAIN_FILE = Path("data/raw/rainfall/heavy_rain_demo.json")
AOI_FILE = Path("data/raw/aoi/rajendra_nagar_aoi.geojson")

TEMPLATE_RASTER = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

OUTPUT_DIR = Path("data/processed/rainfall")

RUNOFF_COEFFICIENT = 0.80
OUTPUT_NODATA = -9999.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load rainfall scenario
    # --------------------------------------------------
    with open(RAIN_FILE, "r", encoding="utf-8") as f:
        rainfall_data = json.load(f)

    forecast = rainfall_data["forecast"]

    # --------------------------------------------------
    # Load AOI
    # --------------------------------------------------
    aoi = gpd.read_file(AOI_FILE)

    if aoi.crs is None:
        raise ValueError("AOI has no CRS")

    # --------------------------------------------------
    # Use terrain raster as spatial template
    # --------------------------------------------------
    with rasterio.open(TEMPLATE_RASTER) as src:
        dem = src.read(1)

        profile = src.profile.copy()
        transform = src.transform
        raster_crs = src.crs
        shape = (src.height, src.width)

        if raster_crs is None:
            raise ValueError("Terrain raster has no CRS")

        if not raster_crs.is_projected:
            raise ValueError(
                "Terrain raster must use a projected CRS"
            )

        # Reproject AOI to terrain CRS
        aoi = aoi.to_crs(raster_crs)

        geometries = [
            geom
            for geom in aoi.geometry
            if geom is not None and not geom.is_empty
        ]

        # True = pixel centre lies inside AOI
        inside_aoi = geometry_mask(
            geometries,
            out_shape=shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )

        # Valid DEM cells
        if src.nodata is None:
            dem_valid = np.isfinite(dem)

        elif np.isnan(src.nodata):
            dem_valid = np.isfinite(dem)

        else:
            dem_valid = (
                np.isfinite(dem)
                & (dem != src.nodata)
            )

        valid_cells = inside_aoi & dem_valid

        # Derive area from raster instead of hard-coding 900 m²
        cell_width = abs(transform.a)
        cell_height = abs(transform.e)

        cell_area_m2 = cell_width * cell_height

    cell_count = int(valid_cells.sum())
    aoi_grid_area_km2 = (
        cell_count * cell_area_m2 / 1_000_000
    )

    print("Runoff raster generation")
    print(f"Raster CRS: {raster_crs}")
    print(
        f"Raster size: "
        f"{shape[1]} x {shape[0]}"
    )
    print(
        f"Cell size: "
        f"{cell_width:.2f} x {cell_height:.2f} m"
    )
    print(f"Cell area: {cell_area_m2:.2f} m²")
    print(f"AOI cells: {cell_count}")
    print(
        f"Rasterized AOI area: "
        f"{aoi_grid_area_km2:.3f} km²"
    )
    print()

    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    total_scenario_volume = 0.0

    # --------------------------------------------------
    # Generate one runoff raster for each interval
    # --------------------------------------------------
    for i in range(len(forecast) - 1):
        current = forecast[i]
        next_step = forecast[i + 1]

        start_min = current["minutes"]
        end_min = next_step["minutes"]

        duration_hours = (
            end_min - start_min
        ) / 60.0

        start_intensity = current["rainfall_mm_hr"]
        end_intensity = next_step["rainfall_mm_hr"]

        average_intensity = (
            start_intensity + end_intensity
        ) / 2.0

        rainfall_mm = (
            average_intensity * duration_hours
        )

        runoff_mm = (
            rainfall_mm * RUNOFF_COEFFICIENT
        )

        runoff_depth_m = runoff_mm / 1000.0

        runoff_volume_per_cell_m3 = (
            runoff_depth_m * cell_area_m2
        )

        total_interval_volume_m3 = (
            runoff_volume_per_cell_m3
            * cell_count
        )

        total_scenario_volume += (
            total_interval_volume_m3
        )

        runoff_array = np.full(
            shape,
            OUTPUT_NODATA,
            dtype=np.float32,
        )

        runoff_array[valid_cells] = (
            runoff_volume_per_cell_m3
        )

        output_file = OUTPUT_DIR / (
            f"runoff_volume_"
            f"{start_min:03d}_"
            f"{end_min:03d}.tif"
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(runoff_array, 1)

            dst.update_tags(
                units="m3_per_cell",
                runoff_coefficient=RUNOFF_COEFFICIENT,
                start_minutes=start_min,
                end_minutes=end_min,
                rainfall_mm=rainfall_mm,
                runoff_mm=runoff_mm,
                scenario=rainfall_data["scenario_id"],
            )

        print(
            f"{start_min:>3}-{end_min:>3} min | "
            f"Rain {rainfall_mm:>6.2f} mm | "
            f"Runoff {runoff_mm:>6.2f} mm | "
            f"{runoff_volume_per_cell_m3:>6.2f} m³/cell | "
            f"{total_interval_volume_m3:>10.2f} m³ AOI"
        )

    print()
    print(
        "Total generated runoff volume over AOI: "
        f"{total_scenario_volume:.2f} m³"
    )

    print()
    print("Runoff rasters generated successfully.")


if __name__ == "__main__":
    main()