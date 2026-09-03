from pathlib import Path

import geopandas as gpd


AOI_PATH = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)


def main():
    print("\n=== RAJENDRA NAGAR AOI VALIDATION ===\n")

    if not AOI_PATH.exists():
        print("ERROR: AOI not found at:")
        print(AOI_PATH)
        print("\nWaiting for GIS branch to provide the AOI.")
        return

    gdf = gpd.read_file(AOI_PATH)

    if gdf.empty:
        print("ERROR: AOI file contains no features.")
        return

    print(f"Number of features : {len(gdf)}")
    print(f"CRS                : {gdf.crs}")

    print(
        "Geometry type      :",
        gdf.geometry.geom_type.tolist()
    )

    print(
        "Geometry valid     :",
        gdf.geometry.is_valid.all()
    )

    minx, miny, maxx, maxy = gdf.total_bounds

    print("\nBounding Box")
    print(f"West  : {minx}")
    print(f"South : {miny}")
    print(f"East  : {maxx}")
    print(f"North : {maxy}")

    print("\nAttributes")
    print(gdf.drop(columns="geometry"))

    if gdf.crs is None:
        print("\nWARNING: CRS is missing.")

    elif gdf.crs.to_epsg() != 4326:
        print(
            f"\nWARNING: Expected EPSG:4326, "
            f"found {gdf.crs}"
        )

    if not gdf.geometry.is_valid.all():
        print("\nWARNING: Invalid geometry detected.")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()