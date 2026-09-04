from pathlib import Path
import math

import numpy as np
import rasterio


DEM_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_utm.tif"
)

FLOOD_FILE = Path(
    "data/outputs/flood_depth/flood_depth_090_120.tif"
)

OVERFLOW_FILE = Path(
    "data/processed/flood_inputs/overflow_volume_090_120.tif"
)

RADIUS = 5


def valid_mask(data, nodata):
    if nodata is None:
        return np.isfinite(data)

    return (
        np.isfinite(data)
        & (data != nodata)
    )


def main():

    with rasterio.open(FLOOD_FILE) as flood_src:
        flood = flood_src.read(1).astype(float)

        flood_valid = valid_mask(
            flood,
            flood_src.nodata
        )

        masked = np.where(
            flood_valid,
            flood,
            np.nan
        )

        flat_index = np.nanargmax(masked)

        row, col = np.unravel_index(
            flat_index,
            flood.shape
        )

        x, y = rasterio.transform.xy(
            flood_src.transform,
            row,
            col,
            offset="center",
        )

        max_depth = float(
            flood[row, col]
        )

    with rasterio.open(DEM_FILE) as dem_src:
        dem = dem_src.read(1).astype(float)

        dem_valid = valid_mask(
            dem,
            dem_src.nodata
        )

    with rasterio.open(OVERFLOW_FILE) as overflow_src:
        overflow = (
            overflow_src.read(1)
            .astype(float)
        )

        overflow_valid = valid_mask(
            overflow,
            overflow_src.nodata
        )

    hotspot_elevation = float(
        dem[row, col]
    )

    r0 = max(0, row - RADIUS)
    r1 = min(
        dem.shape[0],
        row + RADIUS + 1
    )

    c0 = max(0, col - RADIUS)
    c1 = min(
        dem.shape[1],
        col + RADIUS + 1
    )

    local_elevations = []
    local_depths = []

    source_count = 0
    source_volume = 0.0

    for nr in range(r0, r1):
        for nc in range(c0, c1):

            dr = nr - row
            dc = nc - col

            distance = math.sqrt(
                dr * dr + dc * dc
            )

            if distance > RADIUS:
                continue

            if dem_valid[nr, nc]:
                local_elevations.append(
                    float(dem[nr, nc])
                )

            if flood_valid[nr, nc]:
                local_depths.append(
                    float(flood[nr, nc])
                )

            if (
                overflow_valid[nr, nc]
                and overflow[nr, nc] > 0
            ):
                source_count += 1
                source_volume += float(
                    overflow[nr, nc]
                )

    elevations = np.array(
        local_elevations
    )

    depths = np.array(
        local_depths
    )

    local_min = float(
        elevations.min()
    )

    local_mean = float(
        elevations.mean()
    )

    local_max = float(
        elevations.max()
    )

    elevation_above_min = (
        hotspot_elevation
        - local_min
    )

    direct_overflow = (
        float(overflow[row, col])
        if (
            overflow_valid[row, col]
            and overflow[row, col] > 0
        )
        else 0.0
    )

    print("Flood hotspot diagnostic")
    print()

    print(f"Raster row/col: {row}, {col}")
    print(
        f"Location: {x:.2f}, {y:.2f}"
    )

    print(
        f"Peak flood depth: "
        f"{max_depth:.3f} m"
    )

    print()
    print("Terrain:")

    print(
        f"  Hotspot elevation: "
        f"{hotspot_elevation:.3f} m"
    )

    print(
        f"  Local minimum: "
        f"{local_min:.3f} m"
    )

    print(
        f"  Local mean: "
        f"{local_mean:.3f} m"
    )

    print(
        f"  Local maximum: "
        f"{local_max:.3f} m"
    )

    print(
        f"  Hotspot above local min: "
        f"{elevation_above_min:.3f} m"
    )

    print()
    print("Overflow sources within 90 m:")

    print(
        f"  Source cells: "
        f"{source_count}"
    )

    print(
        f"  Total source volume: "
        f"{source_volume:.2f} m³"
    )

    print(
        f"  Overflow directly at hotspot: "
        f"{direct_overflow:.2f} m³"
    )

    print()
    print("Nearby flood depths:")

    print(
        f"  Median: "
        f"{np.median(depths):.3f} m"
    )

    print(
        f"  P90: "
        f"{np.percentile(depths, 90):.3f} m"
    )

    print(
        f"  Maximum: "
        f"{depths.max():.3f} m"
    )

    print(
        f"  Cells >= 0.30 m: "
        f"{int((depths >= 0.30).sum())}"
    )

    print(
        f"  Cells >= 1.00 m: "
        f"{int((depths >= 1.00).sum())}"
    )


if __name__ == "__main__":
    main()