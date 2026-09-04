from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point


COEFFICIENT_FILE = Path(
    "data/processed/landuse/runoff_coefficient.tif"
)

OUTPUT_FILE = Path(
    "data/processed/runoff/runoff_cells.geojson"
)


SURFACE_CLASS = {
    0.30: "green_open",
    0.75: "default_urban",
    0.85: "road",
    0.90: "building",
}


def get_surface_class(value):
    for coefficient, name in SURFACE_CLASS.items():
        if np.isclose(value, coefficient):
            return name

    return "unknown"


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------
    # Read coefficient raster
    # ----------------------------------------
    with rasterio.open(COEFFICIENT_FILE) as src:

        coefficient = src.read(1)

        crs = src.crs
        transform = src.transform

        height = src.height
        width = src.width

        nodata = src.nodata

        if crs is None:
            raise ValueError(
                "Coefficient raster has no CRS"
            )

        if nodata is None:
            valid = np.isfinite(coefficient)
        else:
            valid = (
                np.isfinite(coefficient)
                & (coefficient != nodata)
            )

        cell_width = abs(transform.a)
        cell_height = abs(transform.e)

        cell_area_m2 = (
            cell_width * cell_height
        )

        # ----------------------------------------
        # Build records
        # ----------------------------------------
        records = []

        rows, cols = np.where(valid)

        for row, col in zip(rows, cols):

            # Stable deterministic ID
            cell_id = (
                f"R{row:03d}_C{col:03d}"
            )

            # Pixel-centre coordinates
            x, y = rasterio.transform.xy(
                transform,
                row,
                col,
                offset="center",
            )

            coeff = float(
                coefficient[row, col]
            )

            records.append(
                {
                    "cell_id": cell_id,
                    "row": int(row),
                    "col": int(col),
                    "x": float(x),
                    "y": float(y),
                    "cell_area_m2": float(
                        cell_area_m2
                    ),
                    "runoff_coefficient": coeff,
                    "surface_class": (
                        get_surface_class(coeff)
                    ),
                    "geometry": Point(x, y),
                }
            )

    # ----------------------------------------
    # Create GeoDataFrame
    # ----------------------------------------
    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )

    # Ensure deterministic ordering
    gdf = gdf.sort_values(
        ["row", "col"]
    ).reset_index(drop=True)

    # ----------------------------------------
    # Validation before save
    # ----------------------------------------
    if gdf["cell_id"].duplicated().any():
        raise ValueError(
            "Duplicate runoff cell IDs found"
        )

    if len(gdf) != 2750:
        raise ValueError(
            f"Expected 2750 runoff cells, "
            f"found {len(gdf)}"
        )

    if (
        gdf["surface_class"]
        == "unknown"
    ).any():
        raise ValueError(
            "Unknown runoff coefficient class found"
        )

    # ----------------------------------------
    # Save
    # ----------------------------------------
    gdf.to_file(
        OUTPUT_FILE,
        driver="GeoJSON",
    )

    print("Runoff coupling-cell generation")
    print()
    print(f"CRS: {gdf.crs}")
    print(f"Cells created: {len(gdf)}")
    print(
        f"Unique cell IDs: "
        f"{gdf['cell_id'].nunique()}"
    )
    print(
        f"Cell area: "
        f"{cell_area_m2:.2f} m²"
    )

    print()
    print("Surface classes:")

    counts = (
        gdf["surface_class"]
        .value_counts()
    )

    for name, count in counts.items():
        print(
            f"  {name:<15}: {count}"
        )

    print()
    print("Example cells:")

    print(
        gdf[
            [
                "cell_id",
                "row",
                "col",
                "x",
                "y",
                "runoff_coefficient",
                "surface_class",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(
        "Runoff cells generated successfully."
    )


if __name__ == "__main__":
    main()