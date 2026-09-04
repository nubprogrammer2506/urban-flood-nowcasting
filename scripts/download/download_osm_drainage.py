from pathlib import Path

import geopandas as gpd
import osmnx as ox
from osmnx._errors import InsufficientResponseError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AOI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aoi"
    / "rajendra_nagar_aoi.geojson"
)

RAW_DRAINAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "drainage"
    / "rajendra_nagar_osm_drainage_raw.gpkg"
)


def main():
    print("\n=== OSM DRAINAGE DOWNLOAD ===\n")

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

    aoi = aoi.to_crs("EPSG:4326")

    polygon = aoi.geometry.union_all()

    print(f"AOI CRS             : {aoi.crs}")
    print("Downloading OSM drainage features...")

    # -----------------------------------------------------
    # Query only explicitly mapped drains/ditches
    # -----------------------------------------------------

    tags = {
        "waterway": [
            "drain",
            "ditch",
        ]
    }

    try:
        drainage = ox.features.features_from_polygon(
            polygon,
            tags,
        )

    except InsufficientResponseError:
        print(
            "\nWARNING: OpenStreetMap returned no "
            "waterway=drain/ditch features inside the AOI."
        )

        print(
            "\nThis does NOT mean the area has no drainage."
        )

        print(
            "It means the drainage infrastructure is not "
            "mapped in OSM using these tags."
        )

        print(
            "\nNext source required:"
        )

        print(
            "DEM-derived/inferred surface drainage or "
            "verified municipal drainage data."
        )

        print(
            "\nNo synthetic drainage dataset was created."
        )

        return

    # -----------------------------------------------------
    # Extra empty check
    # -----------------------------------------------------

    if drainage.empty:
        print(
            "\nWARNING: OSM drainage result is empty."
        )

        print(
            "No drainage dataset was created."
        )

        return

    # -----------------------------------------------------
    # Normalize index
    # -----------------------------------------------------

    drainage = drainage.reset_index()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n=== DOWNLOAD SUMMARY ===")

    print(
        f"OSM features        : {len(drainage)}"
    )

    print(
        f"CRS                 : {drainage.crs}"
    )

    print("\nGeometry types:")

    print(
        drainage.geometry
        .geom_type
        .value_counts()
        .to_string()
    )

    if "waterway" in drainage.columns:

        print("\nWaterway types:")

        print(
            drainage["waterway"]
            .value_counts(dropna=False)
            .to_string()
        )

    # -----------------------------------------------------
    # Save raw drainage
    # -----------------------------------------------------

    RAW_DRAINAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drainage.to_file(
        RAW_DRAINAGE_PATH,
        layer="drainage",
        driver="GPKG",
    )

    print("\nRaw OSM drainage saved:")
    print(RAW_DRAINAGE_PATH)

    print("\nOSM drainage download completed.")


if __name__ == "__main__":
    main()