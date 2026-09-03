from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AOI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aoi"
    / "rajendra_nagar_aoi.geojson"
)

ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "rajendra_nagar_roads.gpkg"
)

EXPECTED_CRS = 32643

REQUIRED_COLUMNS = [
    "road_id",
    "osm_id",
    "road_class",
    "name",
    "oneway",
    "lanes",
    "maxspeed",
    "surface",
    "bridge",
    "tunnel",
    "access",
    "length_m",
    "geometry",
]


def main():
    print("\n=== RAJENDRA NAGAR ROAD VALIDATION ===\n")

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if not ROADS_PATH.exists():
        print("ERROR: Processed road dataset not found:")
        print(ROADS_PATH)
        return

    if not AOI_PATH.exists():
        print("ERROR: AOI not found:")
        print(AOI_PATH)
        return

    # -----------------------------------------------------
    # Load roads
    # -----------------------------------------------------

    roads = gpd.read_file(
        ROADS_PATH,
        layer="roads",
    )

    if roads.empty:
        print("ERROR: Road dataset contains no features.")
        return

    print(f"Number of roads      : {len(roads)}")
    print(f"CRS                  : {roads.crs}")

    # -----------------------------------------------------
    # Geometry checks
    # -----------------------------------------------------

    geometry_types = roads.geometry.geom_type.value_counts()

    print("\nGeometry types:")
    print(geometry_types.to_string())

    null_geometry_count = roads.geometry.isna().sum()
    empty_geometry_count = roads.geometry.is_empty.sum()
    invalid_geometry_count = (~roads.geometry.is_valid).sum()

    print("\nGeometry checks")
    print(f"Null geometries      : {null_geometry_count}")
    print(f"Empty geometries     : {empty_geometry_count}")
    print(f"Invalid geometries   : {invalid_geometry_count}")

    # -----------------------------------------------------
    # CRS check
    # -----------------------------------------------------

    if roads.crs is None:
        print("\nERROR: Road CRS is missing.")

    elif roads.crs.to_epsg() != EXPECTED_CRS:
        print(
            f"\nWARNING: Expected EPSG:{EXPECTED_CRS}, "
            f"found {roads.crs}"
        )
    else:
        print(
            f"CRS check            : PASS "
            f"(EPSG:{EXPECTED_CRS})"
        )

    # -----------------------------------------------------
    # Required attributes
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in roads.columns
    ]

    print("\nAttribute checks")

    if missing_columns:
        print(
            "Missing columns       :",
            ", ".join(missing_columns),
        )
    else:
        print("Required columns     : PASS")

    # -----------------------------------------------------
    # Road IDs
    # -----------------------------------------------------

    if "road_id" in roads.columns:
        duplicate_ids = roads["road_id"].duplicated().sum()
        missing_ids = roads["road_id"].isna().sum()

        print(f"Duplicate road IDs   : {duplicate_ids}")
        print(f"Missing road IDs     : {missing_ids}")

    # -----------------------------------------------------
    # Length checks
    # -----------------------------------------------------

    if "length_m" in roads.columns:
        zero_lengths = (roads["length_m"] <= 0).sum()

        total_length_km = (
            roads["length_m"].sum() / 1000
        )

        print(f"Zero/negative length : {zero_lengths}")
        print(
            f"Total road length    : "
            f"{total_length_km:.2f} km"
        )

    # -----------------------------------------------------
    # AOI containment check
    # -----------------------------------------------------

    aoi = gpd.read_file(AOI_PATH)

    if aoi.crs is None:
        print("\nERROR: AOI CRS missing.")
        return

    aoi = aoi.to_crs(roads.crs)

    aoi_geometry = aoi.geometry.union_all()

    # Small tolerance for floating point boundary effects
    check_geometry = aoi_geometry.buffer(0.05)

    within_aoi = roads.geometry.covered_by(
        check_geometry
    )

    outside_count = (~within_aoi).sum()

    print("\nAOI clipping check")
    print(f"Roads outside AOI    : {outside_count}")

    # -----------------------------------------------------
    # Road class summary
    # -----------------------------------------------------

    if "road_class" in roads.columns:
        print("\nRoad classes")

        print(
            roads["road_class"]
            .value_counts(dropna=False)
            .to_string()
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    errors = []

    if null_geometry_count > 0:
        errors.append("null geometries")

    if empty_geometry_count > 0:
        errors.append("empty geometries")

    if invalid_geometry_count > 0:
        errors.append("invalid geometries")

    if missing_columns:
        errors.append("missing attributes")

    if outside_count > 0:
        errors.append("roads outside AOI")

    if (
        roads.crs is None
        or roads.crs.to_epsg() != EXPECTED_CRS
    ):
        errors.append("incorrect CRS")

    if "length_m" in roads.columns:
        if (roads["length_m"] <= 0).any():
            errors.append("invalid road lengths")

    if "road_id" in roads.columns:
        if roads["road_id"].duplicated().any():
            errors.append("duplicate road IDs")

    print("\n====================================")

    if errors:
        print("VALIDATION RESULT: FAILED")

        for error in errors:
            print(f" - {error}")

    else:
        print("VALIDATION RESULT: PASSED")

    print("====================================\n")


if __name__ == "__main__":
    main()