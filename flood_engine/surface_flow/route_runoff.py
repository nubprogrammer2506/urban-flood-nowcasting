from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


DIRECTION_FILE = Path(
    "data/processed/terrain/rajendra_nagar_flow_direction.tif"
)

DEM_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

AOI_FILE = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)

RUNOFF_DIR = Path(
    "data/processed/rainfall"
)

OUTPUT_DIR = Path(
    "data/outputs/surface_flow"
)

OUTPUT_NODATA = -9999.0


# GRASS r.watershed drainage direction
#
# 3  2  1
# 4  X  8
# 5  6  7

DIRECTION_OFFSETS = {
    1: (-1, 1),   # NE
    2: (-1, 0),   # N
    3: (-1, -1),  # NW
    4: (0, -1),   # W
    5: (1, -1),   # SW
    6: (1, 0),    # S
    7: (1, 1),    # SE
    8: (0, 1),    # E
}


RUNOFF_FILES = [
    "runoff_volume_000_030.tif",
    "runoff_volume_030_060.tif",
    "runoff_volume_060_090.tif",
    "runoff_volume_090_120.tif",
    "runoff_volume_120_150.tif",
    "runoff_volume_150_180.tif",
]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------
    # Load DEM
    # --------------------------------------------
    with rasterio.open(DEM_FILE) as dem_src:

        height = dem_src.height
        width = dem_src.width

        transform = dem_src.transform
        crs = dem_src.crs

        dem = dem_src.read(1)

        if dem_src.nodata is None:
            dem_valid = np.isfinite(dem)

        else:
            dem_valid = (
                np.isfinite(dem)
                & (dem != dem_src.nodata)
            )

    # --------------------------------------------
    # Load AOI
    # --------------------------------------------
    aoi = gpd.read_file(AOI_FILE)

    aoi = aoi.to_crs(crs)

    inside_aoi = geometry_mask(
        [
            geom
            for geom in aoi.geometry
            if geom is not None
            and not geom.is_empty
        ],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )

    valid_cells = inside_aoi & dem_valid

    cell_count = int(valid_cells.sum())

    # --------------------------------------------
    # Load flow directions
    # --------------------------------------------
    with rasterio.open(DIRECTION_FILE) as dir_src:

        if (
            dir_src.crs != crs
            or dir_src.width != width
            or dir_src.height != height
            or not dir_src.transform.almost_equals(
                transform
            )
        ):
            raise ValueError(
                "Direction raster is not aligned "
                "with DEM"
            )

        direction = dir_src.read(1)

    # --------------------------------------------
    # Build downstream lookup
    # --------------------------------------------
    downstream = {}

    indegree = np.zeros(
        (height, width),
        dtype=np.int32
    )

    outlet_cells = []

    rows, cols = np.where(valid_cells)

    for row, col in zip(rows, cols):

        code = int(direction[row, col])

        if code not in DIRECTION_OFFSETS:
            raise ValueError(
                f"Invalid flow direction {code} "
                f"at ({row}, {col})"
            )

        dr, dc = DIRECTION_OFFSETS[code]

        nr = row + dr
        nc = col + dc

        # Water leaves raster / AOI
        if (
            nr < 0
            or nr >= height
            or nc < 0
            or nc >= width
            or not valid_cells[nr, nc]
        ):
            downstream[(row, col)] = None
            outlet_cells.append((row, col))

        else:
            downstream[(row, col)] = (
                nr,
                nc
            )

            indegree[nr, nc] += 1

    print("Surface-flow routing setup")
    print()
    print(f"CRS: {crs}")
    print(f"AOI cells: {cell_count}")
    print(
        f"AOI outlet cells: "
        f"{len(outlet_cells)}"
    )

    # --------------------------------------------
    # Establish topological routing order
    # --------------------------------------------
    queue = deque()

    for row, col in zip(rows, cols):

        if indegree[row, col] == 0:
            queue.append((row, col))

    routing_order = []

    indegree_work = indegree.copy()

    while queue:

        row, col = queue.popleft()

        routing_order.append(
            (row, col)
        )

        target = downstream[
            (row, col)
        ]

        if target is None:
            continue

        nr, nc = target

        indegree_work[nr, nc] -= 1

        if indegree_work[nr, nc] == 0:
            queue.append((nr, nc))

    if len(routing_order) != cell_count:

        unresolved = (
            cell_count
            - len(routing_order)
        )

        raise RuntimeError(
            f"Flow-direction graph contains "
            f"a cycle or unresolved cells: "
            f"{unresolved}"
        )

    print(
        f"Routing order cells: "
        f"{len(routing_order)}"
    )
    print("Flow graph cycle check: PASS")
    print()

    # --------------------------------------------
    # Route each rainfall interval
    # --------------------------------------------
    for filename in RUNOFF_FILES:

        input_path = (
            RUNOFF_DIR / filename
        )

        if not input_path.exists():
            raise FileNotFoundError(
                input_path
            )

        with rasterio.open(
            input_path
        ) as runoff_src:

            local_runoff = (
                runoff_src.read(1)
            )

            profile = (
                runoff_src.profile.copy()
            )

            if (
                runoff_src.crs != crs
                or runoff_src.width != width
                or runoff_src.height != height
                or not runoff_src.transform.almost_equals(
                    transform
                )
            ):
                raise ValueError(
                    f"{filename} is not aligned"
                )

            nodata = runoff_src.nodata

        # ----------------------------------------
        # Local runoff values
        # ----------------------------------------
        routed = np.zeros(
            (height, width),
            dtype=np.float64
        )

        if nodata is None:

            valid_runoff = (
                valid_cells
                & np.isfinite(local_runoff)
            )

        else:

            valid_runoff = (
                valid_cells
                & np.isfinite(local_runoff)
                & (local_runoff != nodata)
            )

        routed[valid_runoff] = (
            local_runoff[valid_runoff]
        )

        generated_volume = float(
            routed[valid_cells].sum()
        )

        exported_volume = 0.0

        # ----------------------------------------
        # Route upstream → downstream
        # ----------------------------------------
        for row, col in routing_order:

            volume = routed[row, col]

            target = downstream[
                (row, col)
            ]

            if target is None:

                exported_volume += volume

            else:

                nr, nc = target

                routed[nr, nc] += volume

        # ----------------------------------------
        # Conservation check
        # ----------------------------------------
        difference = abs(
            generated_volume
            - exported_volume
        )

        tolerance = max(
            0.01,
            generated_volume * 1e-8
        )

        conserved = (
            difference <= tolerance
        )

        # ----------------------------------------
        # Save raster
        # ----------------------------------------
        output_array = np.full(
            (height, width),
            OUTPUT_NODATA,
            dtype=np.float32
        )

        output_array[valid_cells] = (
            routed[valid_cells]
        ).astype(np.float32)

        output_filename = filename.replace(
            "runoff_volume_",
            "routed_runoff_"
        )

        output_path = (
            OUTPUT_DIR / output_filename
        )

        profile.update(
            dtype="float32",
            count=1,
            nodata=OUTPUT_NODATA,
            compress="deflate",
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                output_array,
                1
            )

            dst.update_tags(
                units="m3_flow_through_cell",
                model=(
                    "single_direction_"
                    "instantaneous_routing"
                ),
                generated_volume_m3=(
                    generated_volume
                ),
                exported_volume_m3=(
                    exported_volume
                ),
            )

        maximum = float(
            routed[valid_cells].max()
        )

        print(filename)
        print(
            f"  Generated: "
            f"{generated_volume:.2f} m³"
        )
        print(
            f"  Exported:  "
            f"{exported_volume:.2f} m³"
        )
        print(
            f"  Difference: "
            f"{difference:.6f} m³"
        )
        print(
            f"  Conservation: "
            f"{conserved}"
        )
        print(
            f"  Maximum routed volume "
            f"in one cell: "
            f"{maximum:.2f} m³"
        )
        print()

        if not conserved:
            raise RuntimeError(
                f"Water conservation failed "
                f"for {filename}"
            )

    print(
        "All runoff intervals routed "
        "successfully."
    )


if __name__ == "__main__":
    main()