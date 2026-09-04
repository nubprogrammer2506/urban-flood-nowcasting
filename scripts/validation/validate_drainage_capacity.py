from pathlib import Path
import math

import geopandas as gpd
import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPACITY_GEOJSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_capacity.geojson"
)

CAPACITY_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_capacity.graphml"
)

EXPECTED_SOURCE = (
    "synthetic_manning_scenario"
)

EXPECTED_SCENARIO = (
    "prototype_open_channel_base"
)

EXPECTED_MANNING_N = 0.035
EXPECTED_SLOPE_FLOOR = 0.001

REQUIRED_FIELDS = [
    "drain_id",
    "strahler_order",
    "capacity_class",
    "scenario_width_m",
    "scenario_depth_m",
    "scenario_manning_n",
    "capacity_effective_slope",
    "capacity_slope_floor",
    "capacity_m3s",
    "capacity_scenario",
    "capacity_source",
    "capacity_is_observed",
]

CAPACITY_TOLERANCE_M3S = 1e-6


def capacity_class_for_order(order):
    if order <= 1:
        return "low"
    if order == 2:
        return "medium"
    if order == 3:
        return "high"
    return "very_high"


def recompute_capacity(
    width_m,
    depth_m,
    slope_m_per_m,
    manning_n,
):
    area = (
        width_m
        * depth_m
    )

    wetted_perimeter = (
        width_m
        + 2.0 * depth_m
    )

    hydraulic_radius = (
        area
        / wetted_perimeter
    )

    return (
        (1.0 / manning_n)
        * area
        * hydraulic_radius ** (
            2.0 / 3.0
        )
        * math.sqrt(
            slope_m_per_m
        )
    )


def main():

    print(
        "\n=== DRAINAGE CAPACITY VALIDATION ===\n"
    )

    errors = []

    for path in [
        CAPACITY_GEOJSON_PATH,
        CAPACITY_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    drainage = gpd.read_file(
        CAPACITY_GEOJSON_PATH
    )

    graph = nx.read_graphml(
        CAPACITY_GRAPH_PATH,
        force_multigraph=True,
    )

    print(
        f"Capacity features       : "
        f"{len(drainage)}"
    )

    print(
        f"Capacity graph edges    : "
        f"{graph.number_of_edges()}"
    )

    # --------------------------------------------------
    # Schema
    # --------------------------------------------------

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in drainage.columns
    ]

    print(
        "\nSchema checks"
    )

    if missing_fields:

        print(
            "Missing fields          : "
            + ", ".join(
                missing_fields
            )
        )

        errors.append(
            "missing capacity fields"
        )

        print(
            "\nDRAINAGE CAPACITY VALIDATION: FAILED"
        )
        return

    print(
        "Required fields          : PASS"
    )

    # --------------------------------------------------
    # ID / geometry
    # --------------------------------------------------

    duplicate_ids = int(
        drainage[
            "drain_id"
        ].duplicated().sum()
    )

    invalid_geometry = int(
        (
            drainage.geometry.isna()
            | drainage.geometry.is_empty
            | ~drainage.geometry.is_valid
        ).sum()
    )

    print(
        "\nDataset checks"
    )

    print(
        f"Duplicate drain IDs     : "
        f"{duplicate_ids}"
    )

    print(
        f"Invalid geometries      : "
        f"{invalid_geometry}"
    )

    if duplicate_ids:
        errors.append(
            "duplicate drain IDs"
        )

    if invalid_geometry:
        errors.append(
            "invalid drainage geometries"
        )

    # --------------------------------------------------
    # Capacity values
    # --------------------------------------------------

    numeric_fields = [
        "strahler_order",
        "scenario_width_m",
        "scenario_depth_m",
        "scenario_manning_n",
        "capacity_effective_slope",
        "capacity_slope_floor",
        "capacity_m3s",
        "capacity_is_observed",
    ]

    non_finite = 0

    for field in numeric_fields:
        values = drainage[
            field
        ].astype(float)

        non_finite += int(
            (
                ~np.isfinite(
                    values
                )
            ).sum()
        )

    invalid_order = int(
        (
            drainage[
                "strahler_order"
            ].astype(int)
            < 1
        ).sum()
    )

    invalid_capacity = int(
        (
            drainage[
                "capacity_m3s"
            ].astype(float)
            <= 0
        ).sum()
    )

    wrong_source = int(
        (
            drainage[
                "capacity_source"
            ].astype(str)
            != EXPECTED_SOURCE
        ).sum()
    )

    wrong_scenario = int(
        (
            drainage[
                "capacity_scenario"
            ].astype(str)
            != EXPECTED_SCENARIO
        ).sum()
    )

    observed_capacity = int(
        (
            drainage[
                "capacity_is_observed"
            ].astype(int)
            != 0
        ).sum()
    )

    wrong_n = int(
        (
            np.abs(
                drainage[
                    "scenario_manning_n"
                ].astype(float)
                - EXPECTED_MANNING_N
            )
            > 1e-9
        ).sum()
    )

    wrong_floor = int(
        (
            np.abs(
                drainage[
                    "capacity_slope_floor"
                ].astype(float)
                - EXPECTED_SLOPE_FLOOR
            )
            > 1e-9
        ).sum()
    )

    print(
        "\nCapacity checks"
    )

    print(
        f"Non-finite values      : "
        f"{non_finite}"
    )

    print(
        f"Invalid Strahler order : "
        f"{invalid_order}"
    )

    print(
        f"Non-positive capacity  : "
        f"{invalid_capacity}"
    )

    print(
        f"Wrong capacity source  : "
        f"{wrong_source}"
    )

    print(
        f"Wrong scenario name    : "
        f"{wrong_scenario}"
    )

    print(
        f"Marked observed        : "
        f"{observed_capacity}"
    )

    print(
        f"Wrong Manning n        : "
        f"{wrong_n}"
    )

    print(
        f"Wrong slope floor      : "
        f"{wrong_floor}"
    )

    if non_finite:
        errors.append(
            "non-finite capacity values"
        )

    if invalid_order:
        errors.append(
            "invalid Strahler orders"
        )

    if invalid_capacity:
        errors.append(
            "non-positive capacity values"
        )

    if wrong_source:
        errors.append(
            "incorrect capacity provenance"
        )

    if wrong_scenario:
        errors.append(
            "incorrect scenario name"
        )

    if observed_capacity:
        errors.append(
            "synthetic capacity marked observed"
        )

    if wrong_n:
        errors.append(
            "unexpected Manning n"
        )

    if wrong_floor:
        errors.append(
            "unexpected slope floor"
        )

    # --------------------------------------------------
    # Classification / Manning consistency
    # --------------------------------------------------

    wrong_class = 0
    inconsistent_capacity = 0

    for _, row in drainage.iterrows():

        order = int(
            row[
                "strahler_order"
            ]
        )

        expected_class = (
            capacity_class_for_order(
                order
            )
        )

        if str(
            row[
                "capacity_class"
            ]
        ) != expected_class:
            wrong_class += 1

        expected_capacity = (
            recompute_capacity(
                width_m=float(
                    row[
                        "scenario_width_m"
                    ]
                ),
                depth_m=float(
                    row[
                        "scenario_depth_m"
                    ]
                ),
                slope_m_per_m=float(
                    row[
                        "capacity_effective_slope"
                    ]
                ),
                manning_n=float(
                    row[
                        "scenario_manning_n"
                    ]
                ),
            )
        )

        if abs(
            float(
                row[
                    "capacity_m3s"
                ]
            )
            - expected_capacity
        ) > CAPACITY_TOLERANCE_M3S:
            inconsistent_capacity += 1

    print(
        "\nModel consistency"
    )

    print(
        f"Wrong capacity class   : "
        f"{wrong_class}"
    )

    print(
        f"Manning mismatches     : "
        f"{inconsistent_capacity}"
    )

    if wrong_class:
        errors.append(
            "capacity class inconsistent with order"
        )

    if inconsistent_capacity:
        errors.append(
            "capacity values inconsistent with Manning equation"
        )

    # --------------------------------------------------
    # Graph / GeoJSON consistency
    # --------------------------------------------------

    graph_lookup = {}

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        drain_id = str(
            data.get(
                "source_drain_id",
                data.get(
                    "drain_id",
                    "",
                ),
            )
        )

        graph_lookup[
            drain_id
        ] = (
            int(
                float(
                    data[
                        "strahler_order"
                    ]
                )
            ),
            float(
                data[
                    "capacity_m3s"
                ]
            ),
        )

    missing_graph_match = 0
    graph_value_mismatch = 0

    for _, row in drainage.iterrows():

        drain_id = str(
            row[
                "drain_id"
            ]
        )

        if drain_id not in graph_lookup:
            missing_graph_match += 1
            continue

        graph_order, graph_capacity = (
            graph_lookup[
                drain_id
            ]
        )

        if (
            graph_order
            != int(
                row[
                    "strahler_order"
                ]
            )
            or abs(
                graph_capacity
                - float(
                    row[
                        "capacity_m3s"
                    ]
                )
            )
            > CAPACITY_TOLERANCE_M3S
        ):
            graph_value_mismatch += 1

    print(
        "\nGraph consistency"
    )

    print(
        f"Missing graph matches  : "
        f"{missing_graph_match}"
    )

    print(
        f"Graph value mismatches : "
        f"{graph_value_mismatch}"
    )

    if missing_graph_match:
        errors.append(
            "capacity GeoJSON missing graph matches"
        )

    if graph_value_mismatch:
        errors.append(
            "graph/GeoJSON capacity mismatch"
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    capacities = drainage[
        "capacity_m3s"
    ].astype(float)

    orders = drainage[
        "strahler_order"
    ].astype(int)

    print(
        "\nStatistics"
    )

    print(
        f"Maximum Strahler order : "
        f"{orders.max()}"
    )

    for order in sorted(
        orders.unique()
    ):
        count = int(
            (
                orders == order
            ).sum()
        )

        print(
            f"Order {order} features      : "
            f"{count}"
        )

    print(
        f"Minimum capacity       : "
        f"{capacities.min():.4f} m3/s"
    )

    print(
        f"Mean capacity          : "
        f"{capacities.mean():.4f} m3/s"
    )

    print(
        f"Maximum capacity       : "
        f"{capacities.max():.4f} m3/s"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These capacities are synthetic prototype "
        "Manning-scenario values."
    )

    print(
        "They are NOT surveyed or calibrated municipal "
        "storm-drain capacities."
    )

    print(
        "\n===================================="
    )

    if errors:

        print(
            "DRAINAGE CAPACITY VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )

    else:

        print(
            "DRAINAGE CAPACITY VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()
