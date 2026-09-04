from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


OVERFLOW_FILE = Path(
    "data/processed/coupling/drainage_overflow_points.geojson"
)

MASS_BALANCE_FILE = Path(
    "data/processed/coupling/interval_mass_balance.csv"
)

TEMPLATE_RASTER = Path(
    "data/processed/terrain/rajendra_nagar_dem_filled.tif"
)

OUTPUT_DIR = Path(
    "data/processed/flood_inputs"
)

OUTPUT_NODATA = -9999.0


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Load overflow events
    # -------------------------------------------------
    overflow = gpd.read_file(
        OVERFLOW_FILE
    )

    if overflow.empty:
        raise ValueError(
            "No drainage overflow points found"
        )

    required_fields = {
        "interval_id",
        "start_min",
        "end_min",
        "overflow_volume_m3",
    }

    missing = (
        required_fields
        - set(overflow.columns)
    )

    if missing:
        raise ValueError(
            f"Missing overflow fields: {missing}"
        )

    # -------------------------------------------------
    # Load coupling mass balance
    # -------------------------------------------------
    balance = pd.read_csv(
        MASS_BALANCE_FILE
    )

    # -------------------------------------------------
    # Terrain template
    # -------------------------------------------------
    with rasterio.open(
        TEMPLATE_RASTER
    ) as src:

        profile = src.profile.copy()

        crs = src.crs
        transform = src.transform

        width = src.width
        height = src.height

        dem = src.read(1)

        nodata = src.nodata

        if nodata is None:
            valid_cells = np.isfinite(dem)
        else:
            valid_cells = (
                np.isfinite(dem)
                & (dem != nodata)
            )

        cell_width = abs(transform.a)
        cell_height = abs(transform.e)

        cell_area_m2 = (
            cell_width * cell_height
        )

    if overflow.crs is None:
        raise ValueError(
            "Overflow dataset has no CRS"
        )

    overflow = overflow.to_crs(crs)

    print("Drainage overflow raster generation")
    print()
    print(f"CRS: {crs}")
    print(
        f"Raster size: "
        f"{width} x {height}"
    )
    print(
        f"Cell size: "
        f"{cell_width:.2f} x "
        f"{cell_height:.2f} m"
    )
    print(
        f"Cell area: "
        f"{cell_area_m2:.2f} m²"
    )
    print(
        f"Overflow records: "
        f"{len(overflow)}"
    )
    print()

    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    # -------------------------------------------------
    # Process each 30-minute interval
    # -------------------------------------------------
    interval_ids = (
        balance["interval_id"]
        .astype(str)
        .tolist()
    )

    scenario_total_points = 0.0
    scenario_total_raster = 0.0

    with rasterio.open(
        TEMPLATE_RASTER
    ) as template:

        for interval_id in interval_ids:

            interval_points = overflow[
                overflow["interval_id"]
                .astype(str)
                == interval_id
            ].copy()

            balance_row = balance[
                balance["interval_id"]
                .astype(str)
                == interval_id
            ]

            if len(balance_row) != 1:
                raise ValueError(
                    f"Mass-balance row missing "
                    f"for {interval_id}"
                )

            expected_volume = float(
                balance_row[
                    "drainage_overflow_volume_m3"
                ].iloc[0]
            )

            overflow_array = np.zeros(
                (height, width),
                dtype=np.float64,
            )

            # -----------------------------------------
            # Put each overflow point into terrain cell
            # -----------------------------------------
            for record in interval_points.itertuples():

                volume = float(
                    record.overflow_volume_m3
                )

                if not np.isfinite(volume):
                    raise ValueError(
                        "Non-finite overflow volume"
                    )

                if volume < 0:
                    raise ValueError(
                        "Negative overflow volume"
                    )

                x = record.geometry.x
                y = record.geometry.y

                row, col = template.index(
                    x,
                    y
                )

                if not (
                    0 <= row < height
                    and 0 <= col < width
                ):
                    raise ValueError(
                        f"Overflow point outside "
                        f"terrain raster: "
                        f"{record.drain_node_id}"
                    )

                if not valid_cells[row, col]:
                    raise ValueError(
                        f"Overflow point falls on "
                        f"invalid terrain cell: "
                        f"{record.drain_node_id}"
                    )

                # Multiple drains may map to
                # the same terrain cell.
                overflow_array[
                    row,
                    col
                ] += volume

            point_total = float(
                interval_points[
                    "overflow_volume_m3"
                ].sum()
            )

            raster_total = float(
                overflow_array.sum()
            )

            difference = abs(
                expected_volume
                - raster_total
            )

            tolerance = max(
                0.01,
                expected_volume * 1e-8
            )

            conserved = (
                difference <= tolerance
            )

            if not conserved:
                raise RuntimeError(
                    f"Overflow volume mismatch "
                    f"for {interval_id}: "
                    f"{difference}"
                )

            # -----------------------------------------
            # Save only valid terrain cells
            # -----------------------------------------
            output_array = np.full(
                (height, width),
                OUTPUT_NODATA,
                dtype=np.float32,
            )

            output_array[
                valid_cells
            ] = overflow_array[
                valid_cells
            ].astype(np.float32)

            output_file = (
                OUTPUT_DIR
                / f"overflow_volume_{interval_id}.tif"
            )

            with rasterio.open(
                output_file,
                "w",
                **profile
            ) as dst:

                dst.write(
                    output_array,
                    1
                )

                dst.update_tags(
                    units="m3_per_cell",
                    interval_id=interval_id,
                    source=(
                        "runoff_drainage_coupling_"
                        "overflow"
                    ),
                    capacity_source=(
                        "synthetic_manning_scenario"
                    ),
                )

            nonzero_cells = int(
                (
                    overflow_array > 0
                ).sum()
            )

            max_cell_volume = float(
                overflow_array.max()
            )

            scenario_total_points += (
                point_total
            )

            scenario_total_raster += (
                raster_total
            )

            print(
                f"{interval_id} | "
                f"events {len(interval_points):>3} | "
                f"cells {nonzero_cells:>3} | "
                f"overflow "
                f"{raster_total:>10.2f} m³ | "
                f"max cell "
                f"{max_cell_volume:>8.2f} m³ | "
                f"conserved {conserved}"
            )

    print()
    print(
        "Scenario overflow from points: "
        f"{scenario_total_points:.2f} m³"
    )

    print(
        "Scenario overflow in rasters: "
        f"{scenario_total_raster:.2f} m³"
    )

    print()
    print(
        "Overflow source rasters "
        "generated successfully."
    )


if __name__ == "__main__":
    main()