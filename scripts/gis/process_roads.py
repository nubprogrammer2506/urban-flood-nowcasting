from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AOI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aoi"
    / "rajendra_nagar_aoi.geojson"
)

RAW_ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "roads"
    / "rajendra_nagar_osm_roads_raw.gpkg"
)

PROCESSED_ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "rajendra_nagar_roads.gpkg"
)

WGS84 = "EPSG:4326"

# Pune / Rajendra Nagar
ANALYSIS_CRS = "EPSG:32643"


def normalize_value(value):
    """
    Convert OSM values such as lists/tuples into simple strings.
    """

    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)

    return str(value)


def main():
    print("\n=== ROAD PROCESSING ===\n")

    # -----------------------------------------------------
    # Load AOI
    # -----------------------------------------------------

    if not AOI_PATH.exists():
        raise FileNotFoundError(
            f"AOI not found: {AOI_PATH}"
        )

    aoi = gpd.read_file(AOI_PATH)

    if aoi.empty:
        raise ValueError("AOI contains no features.")

    if aoi.crs is None:
        raise ValueError("AOI CRS is missing.")

    aoi = aoi.to_crs(WGS84)

    print(f"AOI loaded          : {len(aoi)} feature(s)")
    print(f"AOI CRS             : {aoi.crs}")

    # -----------------------------------------------------
    # Load raw OSM road edges
    # -----------------------------------------------------

    if not RAW_ROADS_PATH.exists():
        raise FileNotFoundError(
            f"Raw road file not found: {RAW_ROADS_PATH}"
        )

    roads = gpd.read_file(
        RAW_ROADS_PATH,
        layer="edges",
    )

    if roads.empty:
        raise ValueError("Raw road dataset is empty.")

    if roads.crs is None:
        raise ValueError("Raw road CRS is missing.")

    roads = roads.to_crs(WGS84)

    print(f"Raw road features   : {len(roads)}")
    print(f"Raw road CRS        : {roads.crs}")

    # -----------------------------------------------------
    # Remove unusable geometries
    # -----------------------------------------------------

    roads = roads[
        roads.geometry.notna()
        & ~roads.geometry.is_empty
    ].copy()

    roads = roads[
        roads.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )
    ].copy()

    # -----------------------------------------------------
    # Exact AOI clipping
    # -----------------------------------------------------

    roads = gpd.clip(
        roads,
        aoi,
        keep_geom_type=True,
    )

    roads = roads[
        roads.geometry.notna()
        & ~roads.geometry.is_empty
    ].copy()

    print(f"After AOI clipping  : {len(roads)}")

    # -----------------------------------------------------
    # Normalize important OSM attributes
    # -----------------------------------------------------

    rename_map = {
        "osmid": "osm_id",
        "highway": "road_class",
    }

    roads = roads.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in roads.columns
        }
    )

    wanted_columns = [
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
        "geometry",
    ]

    for column in wanted_columns:
        if column not in roads.columns:
            if column != "geometry":
                roads[column] = None

    string_columns = [
        "osm_id",
        "road_class",
        "name",
        "lanes",
        "maxspeed",
        "surface",
        "bridge",
        "tunnel",
        "access",
    ]

    for column in string_columns:
        roads[column] = roads[column].apply(
            normalize_value
        )

    # Normalize oneway
    roads["oneway"] = (
        roads["oneway"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )

    roads = roads[wanted_columns].copy()

    # -----------------------------------------------------
    # Reproject to metre-based CRS
    # -----------------------------------------------------

    roads = roads.to_crs(ANALYSIS_CRS)

    # -----------------------------------------------------
    # Calculate segment length
    # -----------------------------------------------------

    roads["length_m"] = roads.geometry.length.round(2)

    # Remove zero-length pieces
    roads = roads[
        roads["length_m"] > 0
    ].copy()

    # -----------------------------------------------------
    # Create stable local road ID
    # -----------------------------------------------------

    roads = roads.reset_index(drop=True)

    roads.insert(
        0,
        "road_id",
        [
            f"RN_ROAD_{i:05d}"
            for i in range(1, len(roads) + 1)
        ],
    )

    # -----------------------------------------------------
    # Save processed roads
    # -----------------------------------------------------

    PROCESSED_ROADS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    roads.to_file(
        PROCESSED_ROADS_PATH,
        layer="roads",
        driver="GPKG",
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n=== PROCESSING SUMMARY ===")

    print(f"Processed roads     : {len(roads)}")
    print(f"Output CRS          : {roads.crs}")

    print(
        f"Total road length   : "
        f"{roads['length_m'].sum() / 1000:.2f} km"
    )

    print("\nRoad classes:")

    print(
        roads["road_class"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nProcessed road dataset saved:")
    print(PROCESSED_ROADS_PATH)

    print("\nRoad processing completed.")


if __name__ == "__main__":
    main()