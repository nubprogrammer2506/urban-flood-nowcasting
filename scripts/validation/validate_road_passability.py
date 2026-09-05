from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_TIMESERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_flood_risk_timeseries.csv"
)

RISK_PEAK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_flood_risk_peak.geojson"
)

PASSABILITY_TIMESERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_passability_timeseries.csv"
)

PASSABILITY_PEAK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_passability_peak.geojson"
)

PASSABILITY_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_passability.graphml"
)

POLICY_NAME = "prototype_passability_v1"

PASSABILITY_BY_RISK = {
    "minimal": 1,
    "low": 1,
    "moderate": 1,
    "high": 0,
    "unknown": 0,
}


def main():
    print(
        "\n=== ROAD PASSABILITY VALIDATION ===\n"
    )

    for path in [
        RISK_TIMESERIES_PATH,
        RISK_PEAK_PATH,
        PASSABILITY_TIMESERIES_PATH,
        PASSABILITY_PEAK_PATH,
        PASSABILITY_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    risk_ts = pd.read_csv(
        RISK_TIMESERIES_PATH
    )

    risk_peak = gpd.read_file(
        RISK_PEAK_PATH
    )

    pass_ts = pd.read_csv(
        PASSABILITY_TIMESERIES_PATH
    )

    pass_peak = gpd.read_file(
        PASSABILITY_PEAK_PATH
    )

    graph = nx.read_graphml(
        PASSABILITY_GRAPH_PATH,
        force_multigraph=True,
    )

    errors = []

    print(
        f"Risk timeseries rows       : "
        f"{len(risk_ts)}"
    )

    print(
        f"Passability timeseries rows: "
        f"{len(pass_ts)}"
    )

    print(
        f"Risk peak roads            : "
        f"{len(risk_peak)}"
    )

    print(
        f"Passability peak roads     : "
        f"{len(pass_peak)}"
    )

    print(
        f"Passability graph edges    : "
        f"{graph.number_of_edges()}"
    )

    if len(pass_ts) != len(risk_ts):
        errors.append(
            "passability timeseries row count changed"
        )

    if len(pass_peak) != len(risk_peak):
        errors.append(
            "passability peak feature count changed"
        )

    if graph.number_of_edges() != len(risk_peak):
        errors.append(
            "passability graph edge count mismatch"
        )

    duplicate_ts = int(
        pass_ts.duplicated(
            subset=[
                "edge_id",
                "interval_id",
            ]
        ).sum()
    )

    duplicate_peak = int(
        pass_peak.duplicated(
            subset=["edge_id"]
        ).sum()
    )

    print(
        f"Duplicate timeseries      : "
        f"{duplicate_ts}"
    )

    print(
        f"Duplicate peak IDs        : "
        f"{duplicate_peak}"
    )

    if duplicate_ts:
        errors.append(
            "duplicate passability timeseries rows"
        )

    if duplicate_peak:
        errors.append(
            "duplicate passability peak IDs"
        )

    wrong_ts = 0
    wrong_peak = 0
    wrong_ts_status = 0
    wrong_peak_status = 0
    wrong_policy = 0

    for row in pass_ts.itertuples(
        index=False
    ):
        expected = PASSABILITY_BY_RISK[
            str(row.flood_risk)
        ]

        if int(row.is_passable) != expected:
            wrong_ts += 1

        expected_status = (
            "passable"
            if expected == 1
            else "blocked"
        )

        if str(
            row.passability_status
        ) != expected_status:
            wrong_ts_status += 1

        if str(
            row.passability_policy
        ) != POLICY_NAME:
            wrong_policy += 1

    for row in pass_peak.itertuples(
        index=False
    ):
        expected = PASSABILITY_BY_RISK[
            str(row.flood_risk)
        ]

        if int(row.is_passable) != expected:
            wrong_peak += 1

        expected_status = (
            "passable"
            if expected == 1
            else "blocked"
        )

        if str(
            row.passability_status
        ) != expected_status:
            wrong_peak_status += 1

        if str(
            row.passability_policy
        ) != POLICY_NAME:
            wrong_policy += 1

    print("\nPolicy checks")
    print(
        f"Wrong interval passability: "
        f"{wrong_ts}"
    )
    print(
        f"Wrong peak passability    : "
        f"{wrong_peak}"
    )
    print(
        f"Wrong interval status     : "
        f"{wrong_ts_status}"
    )
    print(
        f"Wrong peak status         : "
        f"{wrong_peak_status}"
    )
    print(
        f"Wrong policy values       : "
        f"{wrong_policy}"
    )

    if wrong_ts:
        errors.append(
            "interval passability mismatch"
        )

    if wrong_peak:
        errors.append(
            "peak passability mismatch"
        )

    if wrong_ts_status:
        errors.append(
            "interval passability status mismatch"
        )

    if wrong_peak_status:
        errors.append(
            "peak passability status mismatch"
        )

    if wrong_policy:
        errors.append(
            "passability policy mismatch"
        )

    # Routing cost must still equal road length at P2.19.
    changed_cost = 0
    missing_attrs = 0
    graph_passability_bad = 0

    peak_lookup = {
        str(row.edge_id): row
        for row in pass_peak.itertuples(
            index=False
        )
    }

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):
        edge_id = str(
            data.get("road_id", "")
        )

        if edge_id not in peak_lookup:
            missing_attrs += 1
            continue

        record = peak_lookup[
            edge_id
        ]

        if int(
            float(
                data.get(
                    "is_passable",
                    -1,
                )
            )
        ) != int(
            record.is_passable
        ):
            graph_passability_bad += 1

        if str(
            data.get(
                "passability_status",
                "",
            )
        ) != str(
            record.passability_status
        ):
            graph_passability_bad += 1

        if str(
            data.get(
                "passability_policy",
                "",
            )
        ) != POLICY_NAME:
            graph_passability_bad += 1

        length_m = float(
            data["length_m"]
        )

        routing_cost = float(
            data["routing_cost"]
        )

        if abs(
            routing_cost
            - length_m
        ) > 1e-6:
            changed_cost += 1

    passability_flag = int(
        float(
            graph.graph.get(
                "passability_policy_applied",
                0,
            )
        )
    )

    cost_flag = int(
        float(
            graph.graph.get(
                "routing_cost_policy_applied",
                1,
            )
        )
    )

    safety_flag = int(
        float(
            graph.graph.get(
                "passability_policy_is_safety_standard",
                1,
            )
        )
    )

    print("\nGraph checks")
    print(
        f"Graph passability errors : "
        f"{graph_passability_bad}"
    )
    print(
        f"Missing edge mappings    : "
        f"{missing_attrs}"
    )
    print(
        f"Changed routing costs    : "
        f"{changed_cost}"
    )
    print(
        f"Passability policy flag  : "
        f"{passability_flag}"
    )
    print(
        f"Routing-cost policy flag : "
        f"{cost_flag}"
    )
    print(
        f"Safety-standard flag     : "
        f"{safety_flag}"
    )

    if graph_passability_bad:
        errors.append(
            "graph passability attributes mismatch"
        )

    if missing_attrs:
        errors.append(
            "graph edge missing passability mapping"
        )

    if changed_cost:
        errors.append(
            "routing cost changed too early"
        )

    if passability_flag != 1:
        errors.append(
            "passability policy flag not applied"
        )

    if cost_flag != 0:
        errors.append(
            "routing cost policy applied too early"
        )

    if safety_flag != 0:
        errors.append(
            "prototype passability marked as safety standard"
        )

    passable = int(
        (
            pass_peak[
                "is_passable"
            ]
            == 1
        ).sum()
    )

    blocked = int(
        (
            pass_peak[
                "is_passable"
            ]
            == 0
        ).sum()
    )

    print("\nScenario peak")
    print(
        f"Passable roads            : "
        f"{passable}"
    )
    print(
        f"Blocked roads             : "
        f"{blocked}"
    )

    print("\nIMPORTANT:")
    print(
        "Passability is a conservative prototype routing "
        "policy, not an official road-safety standard."
    )
    print(
        "Flood-aware routing cost remains a separate "
        "next milestone."
    )

    print(
        "\n===================================="
    )

    if errors:
        print(
            "ROAD PASSABILITY VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )
    else:
        print(
            "ROAD PASSABILITY VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()
