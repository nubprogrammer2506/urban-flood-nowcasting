from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


CELLS_FILE = Path(
    "data/processed/runoff/runoff_cells.geojson"
)

TIMESERIES_FILE = Path(
    "data/processed/runoff/runoff_cell_timeseries.csv"
)

EXPECTED_CELLS = 2750
EXPECTED_TIMESTEPS = 6
EXPECTED_ROWS = EXPECTED_CELLS * EXPECTED_TIMESTEPS


def main():

    cells = gpd.read_file(CELLS_FILE)
    ts = pd.read_csv(TIMESERIES_FILE)

    print("Runoff coupling-interface validation")
    print()

    # -----------------------------------
    # Cell layer checks
    # -----------------------------------
    unique_cells = cells["cell_id"].nunique()

    duplicate_cells = int(
        cells["cell_id"].duplicated().sum()
    )

    valid_coefficients = np.array(
        [0.30, 0.75, 0.85, 0.90]
    )

    coefficient_ok = all(
        np.any(
            np.isclose(
                valid_coefficients,
                value
            )
        )
        for value in cells[
            "runoff_coefficient"
        ]
    )

    print(f"Runoff cells: {len(cells)}")
    print(f"Unique cell IDs: {unique_cells}")
    print(f"Duplicate cell IDs: {duplicate_cells}")
    print(f"Cell CRS: {cells.crs}")
    print(
        f"Coefficient classes valid: "
        f"{coefficient_ok}"
    )

    # -----------------------------------
    # Time-series checks
    # -----------------------------------
    duplicate_ts = int(
        ts.duplicated(
            subset=[
                "cell_id",
                "start_min",
                "end_min",
            ]
        ).sum()
    )

    missing_cell_refs = set(
        ts["cell_id"]
    ) - set(
        cells["cell_id"]
    )

    timestep_counts = (
        ts.groupby("cell_id")
        .size()
    )

    timesteps_ok = bool(
        (
            timestep_counts
            == EXPECTED_TIMESTEPS
        ).all()
    )

    negative_runoff = int(
        (ts["runoff_m3"] < 0).sum()
    )

    nonfinite_runoff = int(
        (~np.isfinite(ts["runoff_m3"])).sum()
    )

    print()
    print(f"Time-series rows: {len(ts)}")
    print(
        f"Duplicate timestep rows: "
        f"{duplicate_ts}"
    )
    print(
        f"Missing cell references: "
        f"{len(missing_cell_refs)}"
    )
    print(
        f"Six timesteps per cell: "
        f"{timesteps_ok}"
    )
    print(
        f"Negative runoff values: "
        f"{negative_runoff}"
    )
    print(
        f"Non-finite runoff values: "
        f"{nonfinite_runoff}"
    )

    print()
    print(
        f"Total runoff volume: "
        f"{ts['runoff_m3'].sum():.2f} m³"
    )

    all_valid = (
        len(cells) == EXPECTED_CELLS
        and unique_cells == EXPECTED_CELLS
        and duplicate_cells == 0
        and coefficient_ok
        and len(ts) == EXPECTED_ROWS
        and duplicate_ts == 0
        and len(missing_cell_refs) == 0
        and timesteps_ok
        and negative_runoff == 0
        and nonfinite_runoff == 0
    )

    print()
    print(
        "RUNOFF COUPLING INTERFACE VALID: "
        f"{all_valid}"
    )


if __name__ == "__main__":
    main()