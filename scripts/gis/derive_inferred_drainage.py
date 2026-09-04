from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import xy
from shapely.geometry import LineString


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TERRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "terrain"

FILLED_DEM_PATH = TERRAIN_DIR / "rajendra_nagar_dem_filled.tif"
FLOW_DIRECTION_PATH = TERRAIN_DIR / "rajendra_nagar_flow_direction.tif"
FLOW_ACCUMULATION_PATH = TERRAIN_DIR / "rajendra_nagar_flow_accumulation.tif"

AOI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aoi"
    / "rajendra_nagar_aoi.geojson"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage.geojson"
)

EXPECTED_CRS = CRS.from_epsg(32643)

# 25 cells x 30 m x 30 m = 22,500 m2 = 2.25 ha.
# This is a model parameter, not surveyed drainage truth.
MIN_ACCUMULATION_CELLS = 25


# GRASS r.watershed drainage direction:
#
# 3  2  1
# 4  X  8
# 5  6  7
#
# Same convention used by flood_engine/surface_flow/route_runoff.py.
DIRECTION_OFFSETS = {
    1: (-1, 1),   # NE
    2: (-1, 0),   # N
    3: (-1, -1),  # NW
    4: (0, -1),   # W
    5: (1, -1),   # SW
    6: (1, 0),    # S
    7: (1, 1),    # SE
    8: (0, 1),    # E
}


def validate_alignment(
    src,
    reference_crs,
    reference_width,
    reference_height,
    reference_transform,
    label,
):
    """Ensure a terrain raster shares the canonical DEM grid."""

    if src.crs is None:
        raise ValueError(f"{label} has no CRS.")

    current_crs = CRS.from_user_input(src.crs)

    if not current_crs.equals(reference_crs):
        raise ValueError(
            f"{label} CRS does not match canonical DEM."
        )

    if (
        src.width != reference_width
        or src.height != reference_height
    ):
        raise ValueError(
            f"{label} dimensions do not match canonical DEM."
        )

    if not src.transform.almost_equals(reference_transform):
        raise ValueError(
            f"{label} grid transform does not match canonical DEM."
        )


def create_drainage_segments(
    accumulation,
    direction,
    dem_valid,
    transform,
    crs,
):
    """
    Extract high-accumulation cells as directed surface-flow segments.

    Segment orientation follows the canonical GRASS flow-direction raster:
    upstream -> downstream.
    """

    rows, cols = accumulation.shape
    cell_area_m2 = abs(transform.a) * abs(transform.e)

    features = []

    invalid_direction_count = 0
    raster_boundary_outlet_count = 0
    downstream_nodata_count = 0

    for row in range(rows):
        for col in range(cols):

            if not dem_valid[row, col]:
                continue

            acc_value = accumulation[row, col]

            if not np.isfinite(acc_value):
                continue

            if acc_value < MIN_ACCUMULATION_CELLS:
                continue

            code = int(direction[row, col])

            # Non-1..8 values can occur at boundaries/NoData.
            if code not in DIRECTION_OFFSETS:
                invalid_direction_count += 1
                continue

            dr, dc = DIRECTION_OFFSETS[code]

            downstream_row = row + dr
            downstream_col = col + dc

            if (
                downstream_row < 0
                or downstream_row >= rows
                or downstream_col < 0
                or downstream_col >= cols
            ):
                raster_boundary_outlet_count += 1
                continue

            if not dem_valid[downstream_row, downstream_col]:
                downstream_nodata_count += 1
                continue

            start_x, start_y = xy(
                transform,
                row,
                col,
                offset="center",
            )

            end_x, end_y = xy(
                transform,
                downstream_row,
                downstream_col,
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
                    "_row": row,
                    "_col": col,
                    "source_type": "dem_inferred_surface_flow",
                    "is_inferred": True,
                    "acc_cells": int(round(float(acc_value))),
                    "upstream_area_m2": float(
                        acc_value * cell_area_m2
                    ),
                    "length_m": float(geometry.length),
                    "geometry": geometry,
                }
            )

    print(
        f"Invalid/boundary direction cells : "
        f"{invalid_direction_count}"
    )
    print(
        f"Raster-boundary outlets          : "
        f"{raster_boundary_outlet_count}"
    )
    print(
        f"Downstream NoData cells          : "
        f"{downstream_nodata_count}"
    )

    if not features:
        return gpd.GeoDataFrame(
            columns=[
                "_row",
                "_col",
                "source_type",
                "is_inferred",
                "acc_cells",
                "upstream_area_m2",
                "length_m",
                "geometry",
            ],
            geometry="geometry",
            crs=crs,
        )

    return gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs=crs,
    )


def main():

    print(
        "\n=== CANONICAL TERRAIN DRAINAGE EXTRACTION ===\n"
    )

    required_files = [
        FILLED_DEM_PATH,
        FLOW_DIRECTION_PATH,
        FLOW_ACCUMULATION_PATH,
        AOI_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Canonical filled DEM
    # --------------------------------------------------

    with rasterio.open(FILLED_DEM_PATH) as dem_src:

        if dem_src.crs is None:
            raise ValueError(
                "Canonical DEM has no CRS."
            )

        canonical_crs = CRS.from_user_input(dem_src.crs)

        if not canonical_crs.equals(EXPECTED_CRS):
            raise ValueError(
                "Canonical DEM is not EPSG:32643."
            )

        width = dem_src.width
        height = dem_src.height
        transform = dem_src.transform
        raster_crs = dem_src.crs

        dem_masked = dem_src.read(
            1,
            masked=True,
        )

        dem = (
            dem_masked
            .filled(np.nan)
            .astype(np.float64)
        )

        dem_valid = (
            ~np.ma.getmaskarray(dem_masked)
            & np.isfinite(dem)
        )

    print(
        f"Canonical DEM       : "
        f"{FILLED_DEM_PATH.name}"
    )
    print(
        "CRS                 : EPSG:32643"
    )
    print(
        f"Grid                : "
        f"{width} x {height}"
    )
    print(
        f"Resolution          : "
        f"({abs(transform.a):.1f}, "
        f"{abs(transform.e):.1f}) m"
    )
    print(
        f"Valid terrain cells : "
        f"{int(dem_valid.sum())}"
    )

    # --------------------------------------------------
    # Canonical GRASS flow direction
    # --------------------------------------------------

    with rasterio.open(FLOW_DIRECTION_PATH) as direction_src:

        validate_alignment(
            direction_src,
            canonical_crs,
            width,
            height,
            transform,
            "Flow-direction raster",
        )

        direction_masked = direction_src.read(
            1,
            masked=True,
        )

        direction = (
            direction_masked
            .filled(0)
            .astype(np.int32)
        )

    print(
        f"Flow direction      : "
        f"{FLOW_DIRECTION_PATH.name}"
    )

    # --------------------------------------------------
    # Canonical GRASS flow accumulation
    # --------------------------------------------------

    with rasterio.open(
        FLOW_ACCUMULATION_PATH
    ) as accumulation_src:

        validate_alignment(
            accumulation_src,
            canonical_crs,
            width,
            height,
            transform,
            "Flow-accumulation raster",
        )

        accumulation_masked = accumulation_src.read(
            1,
            masked=True,
        )

        accumulation = (
            accumulation_masked
            .filled(np.nan)
            .astype(np.float64)
        )

    print(
        f"Flow accumulation   : "
        f"{FLOW_ACCUMULATION_PATH.name}"
    )

    valid_accumulation = accumulation[
        np.isfinite(accumulation)
    ]

    if valid_accumulation.size == 0:
        raise RuntimeError(
            "Flow-accumulation raster contains no valid values."
        )

    print(
        f"Maximum accumulation: "
        f"{valid_accumulation.max():.0f} cells"
    )

    # --------------------------------------------------
    # Extract inferred drainage
    # --------------------------------------------------

    print(
        "\nExtracting drainage from canonical "
        "GRASS terrain..."
    )

    drainage = create_drainage_segments(
        accumulation,
        direction,
        dem_valid,
        transform,
        raster_crs,
    )

    print(
        f"Segments before AOI clip         : "
        f"{len(drainage)}"
    )

    if drainage.empty:
        raise RuntimeError(
            "No drainage segments extracted. "
            "Check accumulation values and threshold."
        )

    # --------------------------------------------------
    # Clip to Rajendra Nagar AOI
    # --------------------------------------------------

    aoi = gpd.read_file(AOI_PATH)

    if aoi.crs is None:
        raise ValueError(
            "AOI CRS is missing."
        )

    aoi = aoi.to_crs(raster_crs)

    drainage = gpd.clip(
        drainage,
        aoi,
        keep_geom_type=True,
    )

    drainage = drainage[
        drainage.geometry.notna()
        & ~drainage.geometry.is_empty
        & drainage.geometry.is_valid
    ].copy()

    drainage = drainage[
        drainage.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ].copy()

    drainage["length_m"] = (
        drainage.geometry.length
    ).round(2)

    drainage = drainage[
        drainage["length_m"] > 0
    ].copy()

    # Stable ordering based on source raster cell.
    drainage = (
        drainage
        .sort_values(
            by=["_row", "_col"]
        )
        .reset_index(drop=True)
    )

    drainage["drain_id"] = [
        f"RN_DRAIN_{index:05d}"
        for index in range(
            1,
            len(drainage) + 1,
        )
    ]

    drainage = drainage[
        [
            "drain_id",
            "source_type",
            "is_inferred",
            "acc_cells",
            "upstream_area_m2",
            "length_m",
            "geometry",
        ]
    ]

    # --------------------------------------------------
    # Save canonical inferred drainage
    # --------------------------------------------------

    drainage.to_file(
        OUTPUT_PATH,
        driver="GeoJSON",
    )

    cell_area_m2 = (
        abs(transform.a)
        * abs(transform.e)
    )

    print(
        "\n=== DRAINAGE SUMMARY ==="
    )
    print(
        "Terrain source         : "
        "Canonical GRASS r.watershed"
    )
    print(
        f"Accumulation threshold : "
        f"{MIN_ACCUMULATION_CELLS} cells"
    )
    print(
        f"Threshold area         : "
        f"{MIN_ACCUMULATION_CELLS * cell_area_m2 / 10000:.2f} ha"
    )
    print(
        f"AOI drainage segments  : "
        f"{len(drainage)}"
    )
    print(
        f"Total inferred length  : "
        f"{drainage['length_m'].sum() / 1000:.2f} km"
    )

    print(
        "\nCanonical inferred drainage saved:"
    )
    print(OUTPUT_PATH)

    print(
        "\nIMPORTANT:"
    )
    print(
        "Drainage is inferred from the canonical "
        "GRASS terrain-routing products."
    )
    print(
        "It is NOT surveyed municipal "
        "storm-drain infrastructure."
    )

    print(
        "\nTerrain duplication removed from "
        "drainage extraction."
    )


if __name__ == "__main__":
    main()
