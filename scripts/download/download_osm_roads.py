from pathlib import Path

import geopandas as gpd
import osmnx as ox


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

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


def main():
    print("\n=== OSM ROAD DOWNLOAD ===\n")

    # -----------------------------------------------------
    # Check AOI
    # -----------------------------------------------------

    if not AOI_PATH.exists():
        raise FileNotFoundError(
            f"AOI not found: {AOI_PATH}"
        )

    print(f"Reading AOI:\n{AOI_PATH}")

    aoi = gpd.read_file(AOI_PATH)

    if aoi.empty:
        raise ValueError("AOI contains no features.")

    if aoi.crs is None:
        raise ValueError("AOI has no CRS.")

    print(f"AOI CRS: {aoi.crs}")

    # -----------------------------------------------------
    # OSMnx expects geographic coordinates
    # -----------------------------------------------------

    aoi_wgs84 = aoi.to_crs(epsg=4326)

    polygon = aoi_wgs84.geometry.union_all()

    if polygon.geom_type not in [
        "Polygon",
        "MultiPolygon",
    ]:
        raise ValueError(
            f"Expected Polygon/MultiPolygon, "
            f"found {polygon.geom_type}"
        )

    # -----------------------------------------------------
    # OSMnx settings
    # -----------------------------------------------------

    ox.settings.use_cache = True
    ox.settings.log_console = True

    print("\nDownloading drivable roads from OpenStreetMap...")
    print("This may take a little time.\n")

    # -----------------------------------------------------
    # Download road network
    # -----------------------------------------------------

    graph = ox.graph.graph_from_polygon(
        polygon,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    # -----------------------------------------------------
    # Inspect downloaded network
    # -----------------------------------------------------

    nodes, edges = ox.convert.graph_to_gdfs(graph)

    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"Nodes : {len(nodes)}")
    print(f"Edges : {len(edges)}")
    print(f"CRS   : {edges.crs}")

    if edges.empty:
        raise ValueError(
            "OSM download returned no road edges."
        )

    # -----------------------------------------------------
    # Save raw OSM network
    # -----------------------------------------------------

    RAW_ROADS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.io.save_graph_geopackage(
        graph,
        filepath=RAW_ROADS_PATH,
        directed=False,
    )

    print("\nRaw road network saved successfully:")
    print(RAW_ROADS_PATH)

    print("\nOSM road download completed.")


if __name__ == "__main__":
    main()