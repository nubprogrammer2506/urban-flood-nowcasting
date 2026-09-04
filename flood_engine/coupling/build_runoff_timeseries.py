from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


RUNOFF_CELLS_FILE = Path(
    "data/processed/runoff/runoff_cells.geojson"
)

RUNOFF_DIR = Path(
    "data/processed/rainfall"
)

OUTPUT_FILE = Path(
    "data/processed/runoff/runoff_cell_timeseries.csv"
)


RUNOFF_INTERVALS = [
    (0, 30, "runoff_volume_000_030.tif"),
    (30, 60, "runoff_volume_030_060.tif"),
    (60, 90, "runoff_volume_060_090.tif"),
    (90, 120, "runoff_volume_090_120.tif"),
    (120, 150, "runoff_volume_120_150.tif"),
    (150, 180, "runoff_volume_150_180.tif"),
]


def main():

    cells = gpd.read_file(
        RUNOFF_CELLS_FILE
    )

    if len(cells) != 2750:
        raise ValueError(
            f"Expected 2750 cells, found {len(cells)}"
        )

    required_columns = {
        "cell_id",
        "row",
        "col",
    }

    missing = (
        required_columns
        - set(cells.columns)
    )

    if missing:
        raise ValueError(
            f"Missing runoff-cell fields: {missing}"
        )

    records = []

    reference_crs = None
    reference_transform = None
    reference_width = None
    reference_height = None

    print("Building runoff cell time series")
    print()

    for (
        start_min,
        end_min,
        filename,
    ) in RUNOFF_INTERVALS:

        raster_file = (
            RUNOFF_DIR / filename
        )

        if not raster_file.exists():
            raise FileNotFoundError(
                raster_file
            )

        with rasterio.open(
            raster_file
        ) as src:

            data = src.read(1)

            # First raster establishes
            # the reference grid.
            if reference_crs is None:

                reference_crs = src.crs
                reference_transform = (
                    src.transform
                )
                reference_width = src.width
                reference_height = src.height

            else:

                aligned = (
                    src.crs == reference_crs
                    and src.width
                    == reference_width
                    and src.height
                    == reference_height
                    and src.transform.almost_equals(
                        reference_transform
                    )
                )

                if not aligned:
                    raise ValueError(
                        f"{filename} is not "
                        "aligned with runoff grid"
                    )

            nodata = src.nodata

            interval_total = 0.0

            for row in cells.itertuples(
                index=False
            ):

                raster_row = int(row.row)
                raster_col = int(row.col)

                value = float(
                    data[
                        raster_row,
                        raster_col,
                    ]
                )

                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite runoff for "
                        f"{row.cell_id}"
                    )

                if (
                    nodata is not None
                    and np.isclose(
                        value,
                        nodata
                    )
                ):
                    raise ValueError(
                        f"NoData runoff for "
                        f"{row.cell_id}"
                    )

                if value < 0:
                    raise ValueError(
                        f"Negative runoff for "
                        f"{row.cell_id}"
                    )

                interval_total += value

                records.append(
                    {
                        "cell_id": (
                            row.cell_id
                        ),
                        "start_min": (
                            start_min
                        ),
                        "end_min": (
                            end_min
                        ),
                        "runoff_m3": value,
                    }
                )

        print(
            f"{start_min:>3}-"
            f"{end_min:>3} min | "
            f"{len(cells)} cells | "
            f"{interval_total:>10.2f} m³"
        )

    # ----------------------------------------
    # Create time-series table
    # ----------------------------------------
    df = pd.DataFrame(records)

    expected_rows = (
        len(cells)
        * len(RUNOFF_INTERVALS)
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    # Every cell must appear exactly
    # once per rainfall interval.
    counts = (
        df.groupby("cell_id")
        .size()
    )

    if not (
        counts == len(RUNOFF_INTERVALS)
    ).all():
        raise ValueError(
            "Some runoff cells do not have "
            "all six timesteps"
        )

    # No duplicate cell/timestep records.
    duplicate_count = int(
        df.duplicated(
            subset=[
                "cell_id",
                "start_min",
                "end_min",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate time-series rows: "
            f"{duplicate_count}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Runoff cells: {len(cells)}")
    print(
        f"Timesteps per cell: "
        f"{len(RUNOFF_INTERVALS)}"
    )
    print(
        f"Time-series rows: "
        f"{len(df)}"
    )

    print()
    print("Example records:")

    print(
        df.head(12)
        .to_string(index=False)
    )

    print()
    print(
        f"Total scenario runoff: "
        f"{df['runoff_m3'].sum():.2f} m³"
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(
        "Runoff coupling time series "
        "generated successfully."
    )


if __name__ == "__main__":
    main()