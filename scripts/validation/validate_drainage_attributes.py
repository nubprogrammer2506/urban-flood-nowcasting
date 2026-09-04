from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRAINAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage.geojson"
)

EXPECTED_CRS = CRS.from_epsg(32643)

REQUIRED_ATTRIBUTES = [
    "drain_id",
    "acc_cells",
    "upstream_area_m2",
    "length_m",
    "start_elevation_m",
    "end_elevation_m",
    "elevation_drop_m",
    "slope_m_per_m",
    "slope_percent",
    "terrain_slope_deg",
    "source_type",
    "is_inferred",
]

NUMERIC_ATTRIBUTES = [
    "acc_cells",
    "upstream_area_m2",
    "length_m",
    "start_elevation_m",
    "end_elevation_m",
    "elevation_drop_m",
    "slope_m_per_m",
    "slope_percent",
    "terrain_slope_deg",
]

ELEVATION_TOLERANCE_M = 0.05
SLOPE_TOLERANCE = 1e-4


def crs_matches_expected(crs):
    """
    Compare CRS semantically instead of relying only on
    an EPSG authority tag in the source file.
    """

    if crs is None:
        return False

    return CRS.from_user_input(
        crs
    ).equals(
        EXPECTED_CRS
    )


def main():

    print(
        "\n=== DRAINAGE ATTRIBUTE VALIDATION ===\n"
    )

    errors = []

    # --------------------------------------------------
    # File / dataset checks
    # --------------------------------------------------

    if not DRAINAGE_PATH.exists():
        raise FileNotFoundError(
            f"Drainage file not found: "
            f"{DRAINAGE_PATH}"
        )

    drainage = gpd.read_file(
        DRAINAGE_PATH
    )

    if drainage.empty:
        raise ValueError(
            "Drainage dataset is empty."
        )

    print(
        f"Drainage segments       : "
        f"{len(drainage)}"
    )

    print(
        f"CRS                     : "
        f"{drainage.crs}"
    )

    if not crs_matches_expected(
        drainage.crs
    ):
        errors.append(
            "drainage CRS is not EPSG:32643"
        )

        print(
            "CRS check               : FAIL"
        )

    else:
        print(
            "CRS check               : PASS "
            "(EPSG:32643)"
        )

    # --------------------------------------------------
    # Geometry checks
    # --------------------------------------------------

    null_geometry = int(
        drainage.geometry.isna().sum()
    )

    empty_geometry = int(
        drainage.geometry.is_empty.sum()
    )

    invalid_geometry = int(
        (~drainage.geometry.is_valid).sum()
    )

    wrong_geometry_type = int(
        (
            ~drainage.geometry.geom_type.isin(
                [
                    "LineString",
                    "MultiLineString",
                ]
            )
        ).sum()
    )

    print("\nGeometry checks")

    print(
        f"Null geometries         : "
        f"{null_geometry}"
    )

    print(
        f"Empty geometries        : "
        f"{empty_geometry}"
    )

    print(
        f"Invalid geometries      : "
        f"{invalid_geometry}"
    )

    print(
        f"Wrong geometry types    : "
        f"{wrong_geometry_type}"
    )

    if null_geometry:
        errors.append(
            "null drainage geometries"
        )

    if empty_geometry:
        errors.append(
            "empty drainage geometries"
        )

    if invalid_geometry:
        errors.append(
            "invalid drainage geometries"
        )

    if wrong_geometry_type:
        errors.append(
            "non-line drainage geometries"
        )

    # --------------------------------------------------
    # Required attributes
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_ATTRIBUTES
        if column not in drainage.columns
    ]

    print("\nAttribute structure")

    if missing_columns:

        print(
            "Missing columns         : "
            + ", ".join(missing_columns)
        )

        errors.append(
            "missing drainage attributes"
        )

        print(
            "\n===================================="
        )
        print(
            "DRAINAGE ATTRIBUTE VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

        print(
            "====================================\n"
        )

        return

    print(
        "Required attributes     : PASS"
    )

    # --------------------------------------------------
    # Drain ID checks
    # --------------------------------------------------

    duplicate_ids = int(
        drainage[
            "drain_id"
        ].duplicated().sum()
    )

    missing_ids = int(
        drainage[
            "drain_id"
        ].isna().sum()
    )

    print("\nID checks")

    print(
        f"Duplicate drain IDs     : "
        f"{duplicate_ids}"
    )

    print(
        f"Missing drain IDs       : "
        f"{missing_ids}"
    )

    if duplicate_ids:
        errors.append(
            "duplicate drain IDs"
        )

    if missing_ids:
        errors.append(
            "missing drain IDs"
        )

    # --------------------------------------------------
    # Missing / non-finite numeric values
    # --------------------------------------------------

    print("\nMissing-value checks")

    for column in NUMERIC_ATTRIBUTES:

        numeric = drainage[
            column
        ].astype(float)

        missing = int(
            drainage[
                column
            ].isna().sum()
        )

        non_finite = int(
            (
                ~np.isfinite(
                    numeric
                )
            ).sum()
        )

        print(
            f"{column:<22}: "
            f"missing={missing}, "
            f"non-finite={non_finite}"
        )

        if missing or non_finite:
            errors.append(
                f"invalid values in {column}"
            )

    # --------------------------------------------------
    # Hydrology attributes
    # --------------------------------------------------

    invalid_length = int(
        (
            drainage[
                "length_m"
            ] <= 0
        ).sum()
    )

    invalid_accumulation = int(
        (
            drainage[
                "acc_cells"
            ] <= 0
        ).sum()
    )

    invalid_area = int(
        (
            drainage[
                "upstream_area_m2"
            ] <= 0
        ).sum()
    )

    print("\nHydrology checks")

    print(
        f"Invalid lengths         : "
        f"{invalid_length}"
    )

    print(
        f"Invalid accumulation    : "
        f"{invalid_accumulation}"
    )

    print(
        f"Invalid upstream areas  : "
        f"{invalid_area}"
    )

    if invalid_length:
        errors.append(
            "invalid drainage lengths"
        )

    if invalid_accumulation:
        errors.append(
            "invalid accumulation values"
        )

    if invalid_area:
        errors.append(
            "invalid upstream areas"
        )

    # --------------------------------------------------
    # Elevation-direction consistency
    # --------------------------------------------------

    drops = drainage[
        "elevation_drop_m"
    ].astype(float)

    downhill = int(
        (
            drops
            > ELEVATION_TOLERANCE_M
        ).sum()
    )

    flat = int(
        (
            drops.abs()
            <= ELEVATION_TOLERANCE_M
        ).sum()
    )

    uphill = int(
        (
            drops
            < -ELEVATION_TOLERANCE_M
        ).sum()
    )

    print(
        "\nElevation-direction checks"
    )

    print(
        f"Downhill segments       : "
        f"{downhill}"
    )

    print(
        f"Flat segments           : "
        f"{flat}"
    )

    print(
        f"Uphill segments         : "
        f"{uphill}"
    )

    if uphill:
        errors.append(
            "uphill drainage segments detected"
        )

    # --------------------------------------------------
    # Elevation-drop consistency
    # --------------------------------------------------

    expected_drop = (
        drainage[
            "start_elevation_m"
        ].astype(float)
        - drainage[
            "end_elevation_m"
        ].astype(float)
    )

    drop_difference = (
        expected_drop
        - drops
    ).abs()

    inconsistent_drop = int(
        (
            drop_difference
            > 0.01
        ).sum()
    )

    print("\nElevation consistency")

    print(
        f"Inconsistent drops      : "
        f"{inconsistent_drop}"
    )

    if inconsistent_drop:
        errors.append(
            "elevation-drop calculations inconsistent"
        )

    # --------------------------------------------------
    # Slope consistency
    # --------------------------------------------------

    expected_slope = (
        drainage[
            "elevation_drop_m"
        ].astype(float)
        / drainage[
            "length_m"
        ].astype(float)
    )

    slope_difference = (
        expected_slope
        - drainage[
            "slope_m_per_m"
        ].astype(float)
    ).abs()

    inconsistent_slope = int(
        (
            slope_difference
            > SLOPE_TOLERANCE
        ).sum()
    )

    expected_percent = (
        drainage[
            "slope_m_per_m"
        ].astype(float)
        * 100.0
    )

    percent_difference = (
        expected_percent
        - drainage[
            "slope_percent"
        ].astype(float)
    ).abs()

    inconsistent_percent = int(
        (
            percent_difference
            > 0.01
        ).sum()
    )

    negative_terrain_slope = int(
        (
            drainage[
                "terrain_slope_deg"
            ].astype(float)
            < 0
        ).sum()
    )

    print("\nSlope checks")

    print(
        f"Inconsistent m/m slopes : "
        f"{inconsistent_slope}"
    )

    print(
        f"Inconsistent % slopes   : "
        f"{inconsistent_percent}"
    )

    print(
        f"Negative terrain slopes : "
        f"{negative_terrain_slope}"
    )

    if inconsistent_slope:
        errors.append(
            "segment slope calculations inconsistent"
        )

    if inconsistent_percent:
        errors.append(
            "slope-percent calculations inconsistent"
        )

    if negative_terrain_slope:
        errors.append(
            "negative terrain slope values"
        )

    # --------------------------------------------------
    # Provenance
    # --------------------------------------------------

    incorrect_source = int(
        (
            drainage[
                "source_type"
            ]
            != "dem_inferred_surface_flow"
        ).sum()
    )

    inferred_values = (
        drainage[
            "is_inferred"
        ]
        .astype(str)
        .str.lower()
    )

    incorrect_inferred = int(
        (
            ~inferred_values.isin(
                [
                    "true",
                    "1",
                ]
            )
        ).sum()
    )

    print("\nProvenance checks")

    print(
        f"Incorrect source tags   : "
        f"{incorrect_source}"
    )

    print(
        f"Not marked inferred     : "
        f"{incorrect_inferred}"
    )

    if incorrect_source:
        errors.append(
            "incorrect source provenance"
        )

    if incorrect_inferred:
        errors.append(
            "features not marked inferred"
        )

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    print("\nStatistics")

    print(
        f"Mean elevation drop     : "
        f"{drops.mean():.3f} m"
    )

    print(
        f"Maximum elevation drop  : "
        f"{drops.max():.3f} m"
    )

    print(
        f"Mean segment slope      : "
        f"{drainage['slope_percent'].mean():.3f} %"
    )

    print(
        f"Mean terrain slope      : "
        f"{drainage['terrain_slope_deg'].mean():.3f} deg"
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print(
        "\n===================================="
    )

    if errors:

        print(
            "DRAINAGE ATTRIBUTE VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

    else:

        print(
            "DRAINAGE ATTRIBUTE VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()
