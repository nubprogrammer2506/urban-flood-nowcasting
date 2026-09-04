from pathlib import Path

import numpy as np
import rasterio


TERRAIN_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

COEFFICIENT_FILE = Path(
    "data/processed/landuse/runoff_coefficient.tif"
)

EXPECTED_VALUES = [
    0.30,
    0.75,
    0.85,
    0.90,
]


def main():

    with rasterio.open(TERRAIN_FILE) as terrain:

        terrain_crs = terrain.crs
        terrain_width = terrain.width
        terrain_height = terrain.height
        terrain_transform = terrain.transform

    with rasterio.open(COEFFICIENT_FILE) as src:

        data = src.read(1)

        aligned = (
            src.crs == terrain_crs
            and src.width == terrain_width
            and src.height == terrain_height
            and src.transform.almost_equals(
                terrain_transform
            )
        )

        if src.nodata is None:
            valid = np.isfinite(data)
        else:
            valid = (
                np.isfinite(data)
                & (data != src.nodata)
            )

        values = data[valid]

    print("Runoff coefficient validation")
    print()

    print(f"CRS: {src.crs}")
    print(
        f"Raster size: "
        f"{terrain_width} x {terrain_height}"
    )

    print(f"Valid AOI cells: {len(values)}")

    print(
        f"Minimum coefficient: "
        f"{values.min():.2f}"
    )

    print(
        f"Maximum coefficient: "
        f"{values.max():.2f}"
    )

    print(
        f"Mean coefficient: "
        f"{values.mean():.3f}"
    )

    print()

    unique_values = np.unique(
        np.round(values, 2)
    )

    print(
        f"Unique coefficients: "
        f"{unique_values.tolist()}"
    )

    print(
        f"Terrain aligned: {aligned}"
    )

    expected = np.array(
        EXPECTED_VALUES
    )

    classes_valid = all(
        np.any(
            np.isclose(
                unique_values,
                expected_value
            )
        )
        for expected_value in expected
    )

    range_valid = bool(
        np.all(
            (values >= 0)
            & (values <= 1)
        )
    )

    cell_count_valid = (
        len(values) == 2750
    )

    all_valid = (
        aligned
        and classes_valid
        and range_valid
        and cell_count_valid
    )

    print(
        f"Coefficient classes valid: "
        f"{classes_valid}"
    )

    print(
        f"Coefficient range valid: "
        f"{range_valid}"
    )

    print(
        f"AOI cell count valid: "
        f"{cell_count_valid}"
    )

    print()
    print(
        "RUNOFF COEFFICIENT RASTER VALID: "
        f"{all_valid}"
    )


if __name__ == "__main__":
    main()