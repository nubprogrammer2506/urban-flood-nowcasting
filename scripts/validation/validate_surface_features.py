from pathlib import Path

import geopandas as gpd


AOI_FILE = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)

BUILDINGS_FILE = Path(
    "data/raw/landuse/rajendra_nagar_buildings.geojson"
)

GREEN_FILE = Path(
    "data/raw/landuse/rajendra_nagar_green_areas.geojson"
)

TARGET_CRS = "EPSG:32643"


def validate_layer(name, path, aoi):
    if not path.exists():
        raise FileNotFoundError(path)

    gdf = gpd.read_file(path)

    print(name)
    print(f"  Features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")

    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS")

    invalid = int((~gdf.geometry.is_valid).sum())
    empty = int(gdf.geometry.is_empty.sum())
    missing = int(gdf.geometry.isna().sum())

    print(f"  Invalid geometries: {invalid}")
    print(f"  Empty geometries: {empty}")
    print(f"  Missing geometries: {missing}")

    projected = gdf.to_crs(TARGET_CRS)

    aoi_projected = aoi.to_crs(TARGET_CRS)

    # Keep only non-empty geometries
    projected = projected[
        projected.geometry.notna()
        & ~projected.geometry.is_empty
    ].copy()

    total_area_m2 = float(
        projected.geometry.area.sum()
    )

    aoi_union = aoi_projected.geometry.union_all()

    outside = 0

    for geom in projected.geometry:
        if not geom.intersects(aoi_union):
            outside += 1

    print(
        f"  Total polygon area: "
        f"{total_area_m2 / 1_000_000:.4f} km²"
    )

    print(
        f"  Features not intersecting AOI: "
        f"{outside}"
    )

    polygon_types = sorted(
        projected.geometry.geom_type.unique()
    )

    print(
        f"  Geometry types: {polygon_types}"
    )

    print()


def main():
    aoi = gpd.read_file(AOI_FILE)

    if aoi.crs is None:
        raise ValueError("AOI has no CRS")

    print("OSM surface feature validation")
    print()

    validate_layer(
        "Buildings",
        BUILDINGS_FILE,
        aoi,
    )

    validate_layer(
        "Green/open areas",
        GREEN_FILE,
        aoi,
    )

    print("Surface feature validation complete.")


if __name__ == "__main__":
    main()