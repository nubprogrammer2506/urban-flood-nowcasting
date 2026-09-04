from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRAINAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage.geojson"
)

AOI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aoi"
    / "rajendra_nagar_aoi.geojson"
)

EXPECTED_EPSG = 32643

REQUIRED_COLUMNS = [
    "drain_id",
    "source_type",
    "is_inferred",
    "acc_cells",
    "upstream_area_m2",
    "length_m",
    "geometry",
]


def main():
    print(
        "\n=== INFERRED DRAINAGE VALIDATION ===\n"
    )

    errors = []

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    if not DRAINAGE_PATH.exists():
        print("ERROR: drainage.geojson not found:")
        print(DRAINAGE_PATH)
        return

    if not AOI_PATH.exists():
        print("ERROR: AOI not found:")
        print(AOI_PATH)
        return

    # -----------------------------------------------------
    # Load drainage
    # -----------------------------------------------------

    drainage = gpd.read_file(DRAINAGE_PATH)

    if drainage.empty:
        print("ERROR: Drainage dataset is empty.")
        return

    print(
        f"Drain segments       : {len(drainage)}"
    )

    print(
        f"CRS                  : {drainage.crs}"
    )

    # -----------------------------------------------------
    # CRS
    # -----------------------------------------------------

    if drainage.crs is None:
        errors.append("missing CRS")

    elif drainage.crs.to_epsg() != EXPECTED_EPSG:
        errors.append(
            f"expected EPSG:{EXPECTED_EPSG}"
        )

    else:
        print(
            "CRS check            : "
            "PASS (EPSG:32643)"
        )

    # -----------------------------------------------------
    # Geometry
    # -----------------------------------------------------

    null_geometry = (
        drainage.geometry.isna().sum()
    )

    empty_geometry = (
        drainage.geometry.is_empty.sum()
    )

    invalid_geometry = (
        (~drainage.geometry.is_valid).sum()
    )

    valid_types = drainage.geometry.geom_type.isin(
        [
            "LineString",
            "MultiLineString",
        ]
    )

    incorrect_types = (~valid_types).sum()

    print("\nGeometry checks")

    print(
        f"Null geometries      : "
        f"{null_geometry}"
    )

    print(
        f"Empty geometries     : "
        f"{empty_geometry}"
    )

    print(
        f"Invalid geometries   : "
        f"{invalid_geometry}"
    )

    print(
        f"Wrong geometry types : "
        f"{incorrect_types}"
    )

    if null_geometry:
        errors.append("null geometries")

    if empty_geometry:
        errors.append("empty geometries")

    if invalid_geometry:
        errors.append("invalid geometries")

    if incorrect_types:
        errors.append("non-line geometries")

    # -----------------------------------------------------
    # Required attributes
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in drainage.columns
    ]

    print("\nAttribute checks")

    if missing_columns:
        print(
            "Missing columns       : "
            + ", ".join(missing_columns)
        )

        errors.append(
            "missing required attributes"
        )

    else:
        print(
            "Required columns     : PASS"
        )

    # -----------------------------------------------------
    # ID checks
    # -----------------------------------------------------

    if "drain_id" in drainage.columns:

        duplicate_ids = (
            drainage["drain_id"]
            .duplicated()
            .sum()
        )

        missing_ids = (
            drainage["drain_id"]
            .isna()
            .sum()
        )

        print(
            f"Duplicate drain IDs  : "
            f"{duplicate_ids}"
        )

        print(
            f"Missing drain IDs    : "
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

    # -----------------------------------------------------
    # Inferred-data provenance
    # -----------------------------------------------------

    if "source_type" in drainage.columns:

        incorrect_source = (
            drainage["source_type"]
            != "dem_inferred_surface_flow"
        ).sum()

        print(
            f"Incorrect source tag : "
            f"{incorrect_source}"
        )

        if incorrect_source:
            errors.append(
                "incorrect source metadata"
            )

    if "is_inferred" in drainage.columns:

        inferred_values = (
            drainage["is_inferred"]
            .astype(str)
            .str.lower()
        )

        incorrect_inferred = (
            ~inferred_values.isin(
                ["true", "1"]
            )
        ).sum()

        print(
            f"Not marked inferred  : "
            f"{incorrect_inferred}"
        )

        if incorrect_inferred:
            errors.append(
                "features not marked inferred"
            )

    # -----------------------------------------------------
    # Hydrology attributes
    # -----------------------------------------------------

    if "acc_cells" in drainage.columns:

        invalid_accumulation = (
            drainage["acc_cells"] <= 0
        ).sum()

        print(
            f"Invalid accumulation : "
            f"{invalid_accumulation}"
        )

        if invalid_accumulation:
            errors.append(
                "invalid accumulation values"
            )

    if "upstream_area_m2" in drainage.columns:

        invalid_area = (
            drainage["upstream_area_m2"] <= 0
        ).sum()

        print(
            f"Invalid upstream area: "
            f"{invalid_area}"
        )

        if invalid_area:
            errors.append(
                "invalid upstream areas"
            )

    # -----------------------------------------------------
    # Length
    # -----------------------------------------------------

    if "length_m" in drainage.columns:

        invalid_lengths = (
            drainage["length_m"] <= 0
        ).sum()

        total_length_km = (
            drainage["length_m"].sum()
            / 1000
        )

        print(
            f"Invalid lengths      : "
            f"{invalid_lengths}"
        )

        print(
            f"Total drain length   : "
            f"{total_length_km:.2f} km"
        )

        if invalid_lengths:
            errors.append(
                "invalid drainage lengths"
            )

    # -----------------------------------------------------
    # AOI clipping check
    # -----------------------------------------------------

    aoi = gpd.read_file(AOI_PATH)

    if aoi.crs is None:
        errors.append("AOI missing CRS")

    else:

        aoi = aoi.to_crs(
            drainage.crs
        )

        aoi_geometry = (
            aoi.geometry.union_all()
        )

        # Small numerical tolerance
        aoi_geometry = (
            aoi_geometry.buffer(0.05)
        )

        inside = drainage.geometry.covered_by(
            aoi_geometry
        )

        outside_count = (
            (~inside).sum()
        )

        print("\nAOI clipping check")

        print(
            f"Segments outside AOI : "
            f"{outside_count}"
        )

        if outside_count:
            errors.append(
                "drainage outside AOI"
            )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        "\nIMPORTANT:"
    )

    print(
        "This dataset represents inferred "
        "surface-flow pathways derived from DEM."
    )

    print(
        "It must not be presented as surveyed "
        "municipal storm-drain infrastructure."
    )

    print(
        "\n===================================="
    )

    if errors:

        print(
            "DRAINAGE VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

    else:

        print(
            "DRAINAGE VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()