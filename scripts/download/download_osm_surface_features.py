from pathlib import Path

import geopandas as gpd
import osmnx as ox


AOI_FILE = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)

OUTPUT_DIR = Path(
    "data/raw/landuse"
)


def polygon_only(gdf):
    if gdf.empty:
        return gdf

    gdf = gdf[
        gdf.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        )
    ].copy()

    return gdf


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    aoi = gpd.read_file(AOI_FILE)

    if aoi.crs is None:
        raise ValueError("AOI has no CRS")

    aoi = aoi.to_crs("EPSG:4326")

    polygon = aoi.geometry.union_all()

    print("Downloading OSM surface features...")
    print()

    # ----------------------------------
    # Buildings
    # ----------------------------------
    buildings = ox.features_from_polygon(
        polygon,
        tags={
            "building": True
        }
    )

    buildings = polygon_only(buildings)

    buildings = buildings.reset_index()

    building_file = (
        OUTPUT_DIR
        / "rajendra_nagar_buildings.geojson"
    )

    buildings.to_file(
        building_file,
        driver="GeoJSON"
    )

    print(
        f"Buildings: {len(buildings)}"
    )

    # ----------------------------------
    # Green / permeable areas
    # ----------------------------------
    green = ox.features_from_polygon(
        polygon,
        tags={
            "leisure": [
                "park",
                "garden",
                "nature_reserve",
            ],
            "landuse": [
                "grass",
                "meadow",
                "forest",
                "recreation_ground",
            ],
            "natural": [
                "wood",
                "grassland",
            ],
        }
    )

    green = polygon_only(green)

    green = green.reset_index()

    green_file = (
        OUTPUT_DIR
        / "rajendra_nagar_green_areas.geojson"
    )

    green.to_file(
        green_file,
        driver="GeoJSON"
    )

    print(
        f"Green/open areas: {len(green)}"
    )

    print()
    print("Saved:")
    print(building_file)
    print(green_file)


if __name__ == "__main__":
    main()