from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


DEM_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

DIRECTION_FILE = Path(
    "data/processed/terrain/rajendra_nagar_flow_direction.tif"
)

AOI_FILE = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)


# GRASS r.watershed drainage direction:
#
# 3  2  1
# 4  X  8
# 5  6  7
#
# 1 = NE
# 2 = N
# 3 = NW
# 4 = W
# 5 = SW
# 6 = S
# 7 = SE
# 8 = E

DIRECTION_OFFSETS = {
    1: (-1, 1),
    2: (-1, 0),
    3: (-1, -1),
    4: (0, -1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1),
    8: (0, 1),
}


def main():

    # -----------------------------------------
    # Load DEM
    # -----------------------------------------
    with rasterio.open(DEM_FILE) as dem_src:

        dem = dem_src.read(1)

        dem_crs = dem_src.crs
        transform = dem_src.transform

        height = dem_src.height
        width = dem_src.width

        if dem_src.nodata is None:
            dem_valid = np.isfinite(dem)

        else:
            dem_valid = (
                np.isfinite(dem)
                & (dem != dem_src.nodata)
            )

    # -----------------------------------------
    # Load direction raster
    # -----------------------------------------
    with rasterio.open(DIRECTION_FILE) as dir_src:

        direction = dir_src.read(1)

        aligned = (
            dir_src.crs == dem_crs
            and dir_src.width == width
            and dir_src.height == height
            and dir_src.transform.almost_equals(transform)
        )

        direction_nodata = dir_src.nodata

    if not aligned:
        raise ValueError(
            "Flow direction raster is not aligned with DEM"
        )

    # -----------------------------------------
    # Load AOI
    # -----------------------------------------
    aoi = gpd.read_file(AOI_FILE).to_crs(dem_crs)

    inside_aoi = geometry_mask(
        [
            geom
            for geom in aoi.geometry
            if geom is not None and not geom.is_empty
        ],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )

    valid_aoi = inside_aoi & dem_valid

    total_cells = int(valid_aoi.sum())

    downhill = 0
    equal = 0
    uphill = 0
    exits_aoi = 0
    invalid_direction = 0
    negative_direction = 0
    zero_direction = 0

    direction_counts = {
        i: 0 for i in range(1, 9)
    }

    # -----------------------------------------
    # Check each AOI cell
    # -----------------------------------------
    rows, cols = np.where(valid_aoi)

    for row, col in zip(rows, cols):

        code = int(direction[row, col])

        # NoData
        if (
            direction_nodata is not None
            and code == int(direction_nodata)
        ):
            invalid_direction += 1
            continue

        if code == 0:
            zero_direction += 1
            continue

        if code < 0:
            negative_direction += 1
            continue

        if code not in DIRECTION_OFFSETS:
            invalid_direction += 1
            continue

        direction_counts[code] += 1

        dr, dc = DIRECTION_OFFSETS[code]

        next_row = row + dr
        next_col = col + dc

        # Outside raster
        if not (
            0 <= next_row < height
            and 0 <= next_col < width
        ):
            exits_aoi += 1
            continue

        # Outside AOI
        if not valid_aoi[next_row, next_col]:
            exits_aoi += 1
            continue

        current_elevation = dem[row, col]
        next_elevation = dem[next_row, next_col]

        difference = next_elevation - current_elevation

        tolerance = 0.01

        if difference < -tolerance:
            downhill += 1

        elif abs(difference) <= tolerance:
            equal += 1

        else:
            uphill += 1

    checked = downhill + equal + uphill

    print("Flow routing validation")
    print()
    print(f"CRS: {dem_crs}")
    print(f"Raster size: {width} x {height}")
    print(f"AOI cells: {total_cells}")
    print(f"Terrain aligned: {aligned}")

    print()
    print("Direction codes inside AOI:")

    for code, count in direction_counts.items():
        print(f"  {code}: {count}")

    print()
    print("Routing checks:")
    print(f"  Downhill: {downhill}")
    print(f"  Equal elevation: {equal}")
    print(f"  Uphill: {uphill}")
    print(f"  Leaves AOI: {exits_aoi}")
    print(f"  Negative direction: {negative_direction}")
    print(f"  Zero direction: {zero_direction}")
    print(f"  Invalid direction: {invalid_direction}")

    if checked > 0:

        good = downhill + equal

        percentage = (
            good / checked
        ) * 100

        print()
        print(
            f"Downhill/equal routing: "
            f"{percentage:.2f}%"
        )


if __name__ == "__main__":
    main()