from pathlib import Path

import geopandas as gpd
import numpy as np


ROAD_RISK_DIR = Path(
    "data/outputs/road_risk"
)

INTERVALS = [
    "000_030",
    "030_060",
    "060_090",
    "090_120",
    "120_150",
    "150_180",
]

EXPECTED_ROADS = 635

VALID_RISKS = {
    "safe",
    "low",
    "medium",
    "high",
}


def main():

    all_valid = True

    print("Road flood-risk validation")
    print()

    for interval in INTERVALS:

        path = (
            ROAD_RISK_DIR
            / f"road_risk_{interval}.geojson"
        )

        if not path.exists():
            print(f"MISSING: {path}")
            all_valid = False
            continue

        roads = gpd.read_file(path)

        road_count_ok = (
            len(roads) == EXPECTED_ROADS
        )

        unique_ids_ok = (
            roads["road_id"].nunique()
            == EXPECTED_ROADS
        )

        negative_depths = int(
            (
                roads["flood_max_m"] < 0
            ).sum()
        )

        nonfinite_depths = int(
            (
                ~np.isfinite(
                    roads["flood_max_m"]
                )
            ).sum()
        )

        risk_values = set(
            roads["flood_risk"].unique()
        )

        risk_classes_ok = (
            risk_values
            <= VALID_RISKS
        )

        passable_values_ok = set(
            roads["is_passable"].unique()
        ) <= {0, 1}

        blocked_logic_ok = bool(
            (
                roads.loc[
                    roads["is_passable"] == 0,
                    "flood_max_m",
                ]
                >= 0.30
            ).all()
        )

        high_logic_ok = bool(
            (
                roads.loc[
                    roads["flood_risk"] == "high",
                    "flood_max_m",
                ]
                >= 0.30
            ).all()
        )

        interval_valid = (
            road_count_ok
            and unique_ids_ok
            and negative_depths == 0
            and nonfinite_depths == 0
            and risk_classes_ok
            and passable_values_ok
            and blocked_logic_ok
            and high_logic_ok
        )

        if not interval_valid:
            all_valid = False

        print(interval)
        print(
            f"  Roads: {len(roads)}"
        )
        print(
            f"  Unique road IDs: "
            f"{roads['road_id'].nunique()}"
        )
        print(
            f"  Negative depths: "
            f"{negative_depths}"
        )
        print(
            f"  Non-finite depths: "
            f"{nonfinite_depths}"
        )
        print(
            f"  Risk classes valid: "
            f"{risk_classes_ok}"
        )
        print(
            f"  Blocked-road logic valid: "
            f"{blocked_logic_ok}"
        )
        print(
            f"  High-risk logic valid: "
            f"{high_logic_ok}"
        )
        print(
            f"  Interval valid: "
            f"{interval_valid}"
        )
        print()

    print(
        "ROAD FLOOD-RISK OUTPUTS VALID: "
        f"{all_valid}"
    )


if __name__ == "__main__":
    main()