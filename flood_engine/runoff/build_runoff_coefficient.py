from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize


TEMPLATE_RASTER = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

AOI_FILE = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)

BUILDINGS_FILE = Path(
    "data/raw/landuse/rajendra_nagar_buildings.geojson"
)

GREEN_FILE = Path(
    "data/raw/landuse/rajendra_nagar_green_areas.geojson"
)

# Try canonical road locations used by the project.
ROAD_CANDIDATES = [
    Path("data/processed/roads/roads.geojson"),
    Path("data/processed/roads/rajendra_nagar_roads.geojson"),
]

OUTPUT_FILE = Path(
    "data/processed/landuse/runoff_coefficient.tif"
)


DEFAULT_URBAN_COEFF = 0.75
GREEN_COEFF = 0.30
ROAD_COEFF = 0.85
BUILDING_COEFF = 0.90

OUTPUT_NODATA = -9999.0


def find_roads_file():
    for path in ROAD_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find processed roads GeoJSON. "
        f"Checked: {ROAD_CANDIDATES}"
    )


def valid_geometries(gdf):
    return [
        geom
        for geom in gdf.geometry
        if geom is not None
        and not geom.is_empty
    ]


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    roads_file = find_roads_file()

    print("Building spatial runoff coefficient raster")
    print()
    print(f"Road dataset: {roads_file}")

    # ------------------------------------------------
    # Terrain template
    # ------------------------------------------------
    with rasterio.open(TEMPLATE_RASTER) as src:

        profile = src.profile.copy()

        crs = src.crs
        transform = src.transform

        height = src.height
        width = src.width

        dem = src.read(1)

        if src.nodata is None:
            dem_valid = np.isfinite(dem)
        else:
            dem_valid = (
                np.isfinite(dem)
                & (dem != src.nodata)
            )

    if crs is None:
        raise ValueError(
            "Terrain raster has no CRS"
        )

    # ------------------------------------------------
    # AOI
    # ------------------------------------------------
    aoi = (
        gpd.read_file(AOI_FILE)
        .to_crs(crs)
    )

    inside_aoi = geometry_mask(
        valid_geometries(aoi),
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )

    valid_cells = (
        inside_aoi & dem_valid
    )

    # Start with NoData everywhere.
    coefficient = np.full(
        (height, width),
        OUTPUT_NODATA,
        dtype=np.float32,
    )

    # Every valid AOI cell starts as
    # default dense urban.
    coefficient[valid_cells] = (
        DEFAULT_URBAN_COEFF
    )

    # ------------------------------------------------
    # Load spatial features
    # ------------------------------------------------
    buildings = (
        gpd.read_file(BUILDINGS_FILE)
        .to_crs(crs)
    )

    green = (
        gpd.read_file(GREEN_FILE)
        .to_crs(crs)
    )

    roads = (
        gpd.read_file(roads_file)
        .to_crs(crs)
    )

    # ------------------------------------------------
    # GREEN
    #
    # Rasterize first because roads/buildings
    # have higher priority.
    # ------------------------------------------------
    green_mask = rasterize(
        [
            (geom, 1)
            for geom in valid_geometries(green)
        ],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    coefficient[
        green_mask & valid_cells
    ] = GREEN_COEFF

    # ------------------------------------------------
    # ROADS
    #
    # OSM roads are mostly line geometries.
    # all_touched=True is intentionally used
    # for this 30 m MVP grid.
    # ------------------------------------------------
    road_mask = rasterize(
        [
            (geom, 1)
            for geom in valid_geometries(roads)
        ],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    coefficient[
        road_mask & valid_cells
    ] = ROAD_COEFF

    # ------------------------------------------------
    # BUILDINGS
    #
    # Buildings have highest priority.
    # ------------------------------------------------
    building_mask = rasterize(
        [
            (geom, 1)
            for geom in valid_geometries(buildings)
        ],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    coefficient[
        building_mask & valid_cells
    ] = BUILDING_COEFF

    # ------------------------------------------------
    # Statistics
    # ------------------------------------------------
    values = coefficient[
        valid_cells
    ]

    classes = [
        ("Green", GREEN_COEFF),
        ("Default urban", DEFAULT_URBAN_COEFF),
        ("Road", ROAD_COEFF),
        ("Building", BUILDING_COEFF),
    ]

    print()
    print(f"CRS: {crs}")
    print(
        f"Raster size: "
        f"{width} x {height}"
    )
    print(
        f"Resolution: "
        f"{abs(transform.a):.2f} x "
        f"{abs(transform.e):.2f} m"
    )
    print(
        f"AOI cells: "
        f"{int(valid_cells.sum())}"
    )

    print()
    print("Final coefficient classes:")

    for name, value in classes:

        count = int(
            np.isclose(
                values,
                value
            ).sum()
        )

        percentage = (
            count / len(values)
        ) * 100

        print(
            f"  {name:<14} "
            f"{value:.2f} | "
            f"{count:>4} cells | "
            f"{percentage:>6.2f}%"
        )

    print()
    print(
        f"Mean runoff coefficient: "
        f"{values.mean():.3f}"
    )

    # ------------------------------------------------
    # Save
    # ------------------------------------------------
    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            coefficient,
            1
        )

        dst.update_tags(
            units="dimensionless",
            default_urban_coefficient=(
                DEFAULT_URBAN_COEFF
            ),
            green_coefficient=GREEN_COEFF,
            road_coefficient=ROAD_COEFF,
            building_coefficient=(
                BUILDING_COEFF
            ),
            source=(
                "OSM-derived MVP surface classes"
            ),
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )
    print(
        "Runoff coefficient raster "
        "generated successfully."
    )


if __name__ == "__main__":
    main()