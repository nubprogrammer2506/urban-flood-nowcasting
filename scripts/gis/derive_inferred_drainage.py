from collections import deque
from heapq import heappop, heappush
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from shapely.geometry import LineString


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "terrain"
    / "rajendra_nagar_dem_utm43n.tif"
)

AOI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aoi"
    / "rajendra_nagar_aoi.geojson"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
)

FLOW_ACCUMULATION_PATH = (
    OUTPUT_FOLDER
    / "flow_accumulation.tif"
)

DRAINAGE_PATH = (
    OUTPUT_FOLDER
    / "drainage.geojson"
)

EXPECTED_CRS = "EPSG:32643"

# Initial MVP extraction threshold.
# At 30 m resolution:
# 25 cells × 900 m² = ~22,500 m² / 2.25 ha upstream area.
#
# This is a MODEL PARAMETER, not surveyed drainage truth.
MIN_ACCUMULATION_CELLS = 25


NEIGHBORS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def fill_depressions_priority_flood(
    elevation,
    valid_mask,
):
    """
    Fill DEM depressions using a simple Priority-Flood
    algorithm.

    A tiny floating-point increment is used across filled
    flats to maintain a valid downhill routing direction.
    """

    rows, cols = elevation.shape

    filled = elevation.copy().astype(
        np.float64
    )

    visited = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    heap = []

    # -----------------------------------------------------
    # Identify raster/outside-mask boundary cells
    # -----------------------------------------------------

    for row in range(rows):
        for col in range(cols):

            if not valid_mask[row, col]:
                continue

            is_boundary = (
                row == 0
                or col == 0
                or row == rows - 1
                or col == cols - 1
            )

            if not is_boundary:

                for dr, dc in NEIGHBORS:

                    nr = row + dr
                    nc = col + dc

                    if (
                        nr < 0
                        or nr >= rows
                        or nc < 0
                        or nc >= cols
                        or not valid_mask[nr, nc]
                    ):
                        is_boundary = True
                        break

            if is_boundary:

                visited[row, col] = True

                heappush(
                    heap,
                    (
                        filled[row, col],
                        row,
                        col,
                    ),
                )

    # -----------------------------------------------------
    # Priority-Flood
    # -----------------------------------------------------

    while heap:

        current_elevation, row, col = (
            heappop(heap)
        )

        for dr, dc in NEIGHBORS:

            nr = row + dr
            nc = col + dc

            if (
                nr < 0
                or nr >= rows
                or nc < 0
                or nc >= cols
            ):
                continue

            if not valid_mask[nr, nc]:
                continue

            if visited[nr, nc]:
                continue

            visited[nr, nc] = True

            neighbour_elevation = filled[nr, nc]

            if neighbour_elevation <= current_elevation:

                neighbour_elevation = np.nextafter(
                    current_elevation,
                    np.inf,
                )

                filled[nr, nc] = (
                    neighbour_elevation
                )

            heappush(
                heap,
                (
                    neighbour_elevation,
                    nr,
                    nc,
                ),
            )

    return filled


def calculate_d8_flow(
    filled_dem,
    valid_mask,
    transform,
):
    """
    Calculate one downstream D8 neighbour per raster cell.
    """

    rows, cols = filled_dem.shape

    downstream = np.full(
        rows * cols,
        -1,
        dtype=np.int64,
    )

    x_resolution = abs(transform.a)
    y_resolution = abs(transform.e)

    for row in range(rows):
        for col in range(cols):

            if not valid_mask[row, col]:
                continue

            current = filled_dem[row, col]

            best_slope = 0.0
            best_index = -1

            for dr, dc in NEIGHBORS:

                nr = row + dr
                nc = col + dc

                if (
                    nr < 0
                    or nr >= rows
                    or nc < 0
                    or nc >= cols
                ):
                    continue

                if not valid_mask[nr, nc]:
                    continue

                drop = (
                    current
                    - filled_dem[nr, nc]
                )

                if drop <= 0:
                    continue

                distance = np.hypot(
                    dc * x_resolution,
                    dr * y_resolution,
                )

                slope = drop / distance

                if slope > best_slope:

                    best_slope = slope

                    best_index = (
                        nr * cols + nc
                    )

            index = row * cols + col

            downstream[index] = best_index

    return downstream


def calculate_flow_accumulation(
    downstream,
    valid_mask,
):
    """
    Accumulate upstream cell counts through the D8 graph.
    """

    rows, cols = valid_mask.shape
    size = rows * cols

    valid_flat = valid_mask.ravel()

    accumulation = np.zeros(
        size,
        dtype=np.float64,
    )

    accumulation[valid_flat] = 1.0

    indegree = np.zeros(
        size,
        dtype=np.int32,
    )

    # Count upstream connections.
    for index in range(size):

        if not valid_flat[index]:
            continue

        target = downstream[index]

        if target >= 0:
            indegree[target] += 1

    queue = deque(
        index
        for index in range(size)
        if (
            valid_flat[index]
            and indegree[index] == 0
        )
    )

    processed = 0

    while queue:

        index = queue.popleft()
        processed += 1

        target = downstream[index]

        if target < 0:
            continue

        accumulation[target] += (
            accumulation[index]
        )

        indegree[target] -= 1

        if indegree[target] == 0:
            queue.append(target)

    valid_count = int(valid_flat.sum())

    if processed != valid_count:
        raise RuntimeError(
            "Flow network contains unresolved cycles: "
            f"{processed}/{valid_count} cells processed."
        )

    return accumulation.reshape(
        rows,
        cols,
    )


def create_drainage_lines(
    accumulation,
    downstream,
    valid_mask,
    transform,
    crs,
):
    """
    Convert high-flow raster cells into line segments.
    """

    rows, cols = accumulation.shape

    features = []

    drain_number = 1

    cell_area = (
        abs(transform.a)
        * abs(transform.e)
    )

    for row in range(rows):
        for col in range(cols):

            if not valid_mask[row, col]:
                continue

            if (
                accumulation[row, col]
                < MIN_ACCUMULATION_CELLS
            ):
                continue

            index = row * cols + col

            target = downstream[index]

            if target < 0:
                continue

            target_row = target // cols
            target_col = target % cols

            if not valid_mask[
                target_row,
                target_col
            ]:
                continue

            start_x, start_y = xy(
                transform,
                row,
                col,
                offset="center",
            )

            end_x, end_y = xy(
                transform,
                target_row,
                target_col,
                offset="center",
            )

            geometry = LineString(
                [
                    (start_x, start_y),
                    (end_x, end_y),
                ]
            )

            features.append(
                {
                    "drain_id": (
                        f"RN_DRAIN_{drain_number:05d}"
                    ),
                    "source_type": (
                        "dem_inferred_surface_flow"
                    ),
                    "is_inferred": True,
                    "acc_cells": int(
                        accumulation[row, col]
                    ),
                    "upstream_area_m2": float(
                        accumulation[row, col]
                        * cell_area
                    ),
                    "length_m": float(
                        geometry.length
                    ),
                    "geometry": geometry,
                }
            )

            drain_number += 1

    return gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs=crs,
    )


def main():

    print(
        "\n=== DEM-INFERRED DRAINAGE EXTRACTION ===\n"
    )

    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"Prepared DEM not found: {DEM_PATH}"
        )

    if not AOI_PATH.exists():
        raise FileNotFoundError(
            f"AOI not found: {AOI_PATH}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Read DEM
    # -----------------------------------------------------

    with rasterio.open(DEM_PATH) as src:

        if (
            src.crs is None
            or src.crs.to_epsg() != 32643
        ):
            raise ValueError(
                "Expected DEM in EPSG:32643."
            )

        dem_masked = src.read(
            1,
            masked=True,
        )

        valid_mask = (
            ~np.ma.getmaskarray(dem_masked)
        )

        original_dem = (
            dem_masked.filled(np.nan)
            .astype(np.float64)
        )

        transform = src.transform
        crs = src.crs

        profile = src.profile.copy()

    print(
        f"DEM size             : "
        f"{original_dem.shape[1]} x "
        f"{original_dem.shape[0]}"
    )

    print(f"DEM CRS              : {crs}")

    print(
        f"Valid terrain cells  : "
        f"{valid_mask.sum()}"
    )

    # -----------------------------------------------------
    # Depression filling
    # -----------------------------------------------------

    print(
        "\nFilling terrain depressions..."
    )

    filled_dem = fill_depressions_priority_flood(
        original_dem,
        valid_mask,
    )

    fill_difference = (
        filled_dem[valid_mask]
        - original_dem[valid_mask]
    )

    print(
        f"Maximum fill raise   : "
        f"{np.max(fill_difference):.3f} m"
    )

    # -----------------------------------------------------
    # D8 flow direction
    # -----------------------------------------------------

    print("Calculating D8 flow direction...")

    downstream = calculate_d8_flow(
        filled_dem,
        valid_mask,
        transform,
    )

    # -----------------------------------------------------
    # Flow accumulation
    # -----------------------------------------------------

    print("Calculating flow accumulation...")

    accumulation = (
        calculate_flow_accumulation(
            downstream,
            valid_mask,
        )
    )

    valid_accumulation = (
        accumulation[valid_mask]
    )

    print(
        f"Maximum accumulation : "
        f"{valid_accumulation.max():.0f} cells"
    )

    # -----------------------------------------------------
    # Save accumulation raster
    # -----------------------------------------------------

    accumulation_output = np.zeros(
        accumulation.shape,
        dtype=np.float32,
    )

    accumulation_output[valid_mask] = (
        accumulation[valid_mask]
    )

    profile.update(
        dtype="float32",
        nodata=0,
        count=1,
        compress="deflate",
    )

    with rasterio.open(
        FLOW_ACCUMULATION_PATH,
        "w",
        **profile,
    ) as dst:

        dst.write(
            accumulation_output,
            1,
        )

    # -----------------------------------------------------
    # Extract inferred drainage
    # -----------------------------------------------------

    print(
        "\nExtracting inferred drainage..."
    )

    drainage = create_drainage_lines(
        accumulation,
        downstream,
        valid_mask,
        transform,
        crs,
    )

    print(
        f"Drain segments before AOI clip: "
        f"{len(drainage)}"
    )

    if drainage.empty:
        raise RuntimeError(
            "No drainage segments were extracted. "
            "Try lowering MIN_ACCUMULATION_CELLS."
        )

    # -----------------------------------------------------
    # Clip final network to Rajendra Nagar
    # -----------------------------------------------------

    aoi = gpd.read_file(AOI_PATH)

    if aoi.crs is None:
        raise ValueError("AOI CRS is missing.")

    aoi = aoi.to_crs(crs)

    drainage = gpd.clip(
        drainage,
        aoi,
        keep_geom_type=True,
    )

    drainage = drainage[
        drainage.geometry.notna()
        & ~drainage.geometry.is_empty
    ].copy()

    drainage["length_m"] = (
        drainage.geometry.length
    ).round(2)

    drainage = drainage[
        drainage["length_m"] > 0
    ].copy()

    drainage = drainage.reset_index(
        drop=True
    )

    # Reassign stable IDs after clipping.
    drainage["drain_id"] = [
        f"RN_DRAIN_{index:05d}"
        for index in range(
            1,
            len(drainage) + 1,
        )
    ]

    drainage.to_file(
        DRAINAGE_PATH,
        driver="GeoJSON",
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n=== DRAINAGE SUMMARY ===")

    print(
        f"Accumulation threshold : "
        f"{MIN_ACCUMULATION_CELLS} cells"
    )

    print(
        f"Approx threshold area  : "
        f"{MIN_ACCUMULATION_CELLS * 900 / 10000:.2f} ha"
    )

    print(
        f"AOI drain segments     : "
        f"{len(drainage)}"
    )

    print(
        f"Total inferred length  : "
        f"{drainage['length_m'].sum() / 1000:.2f} km"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This dataset is a DEM-derived surface-flow "
        "proxy. It is NOT surveyed municipal drainage."
    )

    print("\nFlow accumulation saved:")
    print(FLOW_ACCUMULATION_PATH)

    print("\nCanonical inferred drainage saved:")
    print(DRAINAGE_PATH)

    print(
        "\nDrainage extraction completed."
    )


if __name__ == "__main__":
    main()