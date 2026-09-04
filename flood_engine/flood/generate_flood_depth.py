from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


DEM_FILE = Path(
    "data/processed/terrain/rajendra_nagar_dem_utm.tif"
)

AOI_FILE = Path(
    "data/raw/aoi/rajendra_nagar_aoi.geojson"
)

OVERFLOW_DIR = Path(
    "data/processed/flood_inputs"
)

OUTPUT_DIR = Path(
    "data/outputs/flood_depth"
)

INTERVALS = [
    "000_030",
    "030_060",
    "060_090",
    "090_120",
    "120_150",
    "150_180",
]

OUTPUT_NODATA = -9999.0

# 3 cells × 30 m = ~90 m local spreading radius
SPREAD_RADIUS_CELLS = 5

# Controls preference for lower terrain
ELEVATION_SCALE_M = 2.0

# Controls preference for cells closer to overflow source
DISTANCE_SCALE_CELLS = 3.0


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Load original/unfilled terrain
    # -----------------------------------------
    with rasterio.open(DEM_FILE) as dem_src:

        dem = dem_src.read(1).astype(np.float64)

        crs = dem_src.crs
        transform = dem_src.transform

        width = dem_src.width
        height = dem_src.height

        profile = dem_src.profile.copy()

        if dem_src.nodata is None:
            dem_valid = np.isfinite(dem)
        else:
            dem_valid = (
                np.isfinite(dem)
                & (dem != dem_src.nodata)
            )

        cell_width = abs(transform.a)
        cell_height = abs(transform.e)

        cell_area_m2 = (
            cell_width * cell_height
        )

    # -----------------------------------------
    # AOI mask
    # -----------------------------------------
    aoi = gpd.read_file(AOI_FILE)

    if aoi.crs is None:
        raise ValueError("AOI has no CRS")

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

    valid_cells = (
        inside_aoi
        & dem_valid
    )

    print("Flood-depth generation")
    print()
    print(f"CRS: {crs}")
    print(f"Raster size: {width} x {height}")
    print(
        f"Cell size: "
        f"{cell_width:.2f} x "
        f"{cell_height:.2f} m"
    )
    print(
        f"Spreading radius: "
        f"{SPREAD_RADIUS_CELLS * cell_width:.0f} m"
    )
    print()

    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    for interval in INTERVALS:

        overflow_file = (
            OVERFLOW_DIR
            / f"overflow_volume_{interval}.tif"
        )

        if not overflow_file.exists():
            raise FileNotFoundError(
                overflow_file
            )

        with rasterio.open(
            overflow_file
        ) as src:

            if (
                src.crs != crs
                or src.width != width
                or src.height != height
                or not src.transform.almost_equals(
                    transform
                )
            ):
                raise ValueError(
                    f"{interval}: overflow raster "
                    "not aligned with DEM"
                )

            overflow = (
                src.read(1)
                .astype(np.float64)
            )

            nodata = src.nodata

        if nodata is None:
            overflow_valid = np.isfinite(
                overflow
            )
        else:
            overflow_valid = (
                np.isfinite(overflow)
                & (overflow != nodata)
            )

        source_mask = (
            valid_cells
            & overflow_valid
            & (overflow > 0)
        )

        source_rows, source_cols = (
            np.where(source_mask)
        )

        source_total = float(
            overflow[source_mask].sum()
        )

        # -------------------------------------
        # Surface storage
        # -------------------------------------
        storage_m3 = np.zeros(
            (height, width),
            dtype=np.float64,
        )

        for row, col in zip(
            source_rows,
            source_cols
        ):

            source_volume = float(
                overflow[row, col]
            )

            r0 = max(
                0,
                row - SPREAD_RADIUS_CELLS
            )

            r1 = min(
                height,
                row + SPREAD_RADIUS_CELLS + 1
            )

            c0 = max(
                0,
                col - SPREAD_RADIUS_CELLS
            )

            c1 = min(
                width,
                col + SPREAD_RADIUS_CELLS + 1
            )

            candidate_rows = []
            candidate_cols = []
            candidate_elevations = []
            candidate_distances = []

            for nr in range(r0, r1):
                for nc in range(c0, c1):

                    if not valid_cells[nr, nc]:
                        continue

                    dr = nr - row
                    dc = nc - col

                    distance = math.sqrt(
                        dr * dr + dc * dc
                    )

                    if (
                        distance
                        > SPREAD_RADIUS_CELLS
                    ):
                        continue

                    candidate_rows.append(nr)
                    candidate_cols.append(nc)

                    candidate_elevations.append(
                        dem[nr, nc]
                    )

                    candidate_distances.append(
                        distance
                    )

            if not candidate_rows:
                raise RuntimeError(
                    f"No valid spreading cells "
                    f"for source ({row}, {col})"
                )

            elevations = np.array(
                candidate_elevations,
                dtype=np.float64,
            )

            distances = np.array(
                candidate_distances,
                dtype=np.float64,
            )

            minimum_elevation = float(
                elevations.min()
            )

            # Lower cells receive more water.
            elevation_weight = np.exp(
                -(
                    elevations
                    - minimum_elevation
                )
                / ELEVATION_SCALE_M
            )

            # Nearby cells receive more water.
            distance_weight = np.exp(
                -distances
                / DISTANCE_SCALE_CELLS
            )

            weights = (
                elevation_weight
                * distance_weight
            )

            weight_sum = float(
                weights.sum()
            )

            if weight_sum <= 0:
                raise RuntimeError(
                    "Invalid spreading weights"
                )

            distributed = (
                source_volume
                * weights
                / weight_sum
            )

            for (
                nr,
                nc,
                volume,
            ) in zip(
                candidate_rows,
                candidate_cols,
                distributed,
            ):

                storage_m3[
                    nr,
                    nc
                ] += volume

        # -------------------------------------
        # Convert storage volume → depth
        # -------------------------------------
        depth_m = np.zeros(
            (height, width),
            dtype=np.float64,
        )

        depth_m[valid_cells] = (
            storage_m3[valid_cells]
            / cell_area_m2
        )

        stored_total = float(
            storage_m3[
                valid_cells
            ].sum()
        )

        difference = abs(
            source_total
            - stored_total
        )

        tolerance = max(
            0.01,
            source_total * 1e-8
        )

        conserved = (
            difference <= tolerance
        )

        if not conserved:
            raise RuntimeError(
                f"{interval}: flood storage "
                f"mass balance failed"
            )

        # -------------------------------------
        # Statistics
        # -------------------------------------
        positive = (
            valid_cells
            & (depth_m > 0)
        )

        flooded_02 = (
            valid_cells
            & (depth_m >= 0.02)
        )

        flooded_10 = (
            valid_cells
            & (depth_m >= 0.10)
        )

        flooded_30 = (
            valid_cells
            & (depth_m >= 0.30)
        )

        max_depth = float(
            depth_m[
                valid_cells
            ].max()
        )

        mean_positive = (
            float(
                depth_m[
                    positive
                ].mean()
            )
            if positive.any()
            else 0.0
        )

        # -------------------------------------
        # Save
        # -------------------------------------
        output = np.full(
            (height, width),
            OUTPUT_NODATA,
            dtype=np.float32,
        )

        output[valid_cells] = (
            depth_m[
                valid_cells
            ].astype(np.float32)
        )

        output_file = (
            OUTPUT_DIR
            / f"flood_depth_{interval}.tif"
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(output, 1)

            dst.update_tags(
                units="metres",
                model=(
                    "terrain_weighted_"
                    "local_spreading"
                ),
                spreading_radius_m=(
                    SPREAD_RADIUS_CELLS
                    * cell_width
                ),
                source=(
                    "drainage_overflow"
                ),
                note=(
                    "MVP flood-depth proxy; "
                    "not calibrated 2D hydraulics"
                ),
            )

        print(
            f"{interval} | "
            f"sources {len(source_rows):>3} | "
            f"overflow {source_total:>10.2f} m³ | "
            f">2cm {int(flooded_02.sum()):>4} cells | "
            f">10cm {int(flooded_10.sum()):>4} | "
            f">30cm {int(flooded_30.sum()):>4} | "
            f"mean+ {mean_positive:.3f} m | "
            f"max {max_depth:.3f} m | "
            f"conserved {conserved}"
        )

    print()
    print(
        "Flood-depth rasters generated successfully."
    )


if __name__ == "__main__":
    main()