from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

ROUTING_COST_TIMESERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_routing_cost_timeseries.csv"
)

ROUTING_COST_PEAK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_routing_cost_peak.geojson"
)

ROUTING_COST_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_routing_cost.graphml"
)

POLICY_NAME = "prototype_risk_weighted_cost_v1"

RISK_MULTIPLIER = {
    "minimal": 1.0,
    "low": 1.5,
    "moderate": 3.0,
}

BLOCKED_COST_SENTINEL = 1.0e12
EPS = 1e-6


def expected_cost(
    length_m,
    flood_risk,
    is_passable,
):
    length = float(length_m)
    risk = str(flood_risk)
    passable = int(is_passable)

    if passable == 0:
        return (
            BLOCKED_COST_SENTINEL,
            0.0,
            "blocked_sentinel",
        )

    multiplier = float(
        RISK_MULTIPLIER[risk]
    )

    return (
        length * multiplier,
        multiplier,
        "length_x_risk_multiplier",
    )


def main():
    print(
        "\n=== FLOOD-AWARE ROUTING-COST VALIDATION ===\n"
    )

    for path in [
        PASSABILITY_TIMESERIES_PATH,
        PASSABILITY_PEAK_PATH,
        ROUTING_COST_TIMESERIES_PATH,
        ROUTING_COST_PEAK_PATH,
        ROUTING_COST_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    pass_ts = pd.read_csv(
        PASSABILITY_TIMESERIES_PATH
    )

    pass_peak = gpd.read_file(
        PASSABILITY_PEAK_PATH
    )

    cost_ts = pd.read_csv(
        ROUTING_COST_TIMESERIES_PATH
    )

    cost_peak = gpd.read_file(
        ROUTING_COST_PEAK_PATH
    )

    graph = nx.read_graphml(
        ROUTING_COST_GRAPH_PATH,
        force_multigraph=True,
    )

    errors = []

    print(
        f"Passability timeseries rows: "
        f"{len(pass_ts)}"
    )

    print(
        f"Routing-cost timeseries rows: "
        f"{len(cost_ts)}"
    )

    print(
        f"Passability peak roads     : "
        f"{len(pass_peak)}"
    )

    print(
        f"Routing-cost peak roads    : "
        f"{len(cost_peak)}"
    )

    print(
        f"Routing-cost graph edges   : "
        f"{graph.number_of_edges()}"
    )

    if len(
        cost_ts
    ) != len(
        pass_ts
    ):
        errors.append(
            "routing-cost timeseries row count changed"
        )

    if len(
        cost_peak
    ) != len(
        pass_peak
    ):
        errors.append(
            "routing-cost peak feature count changed"
        )

    if graph.number_of_edges() != len(
        pass_peak
    ):
        errors.append(
            "routing-cost graph edge count mismatch"
        )

    duplicate_ts = int(
        cost_ts.duplicated(
            subset=[
                "edge_id",
                "interval_id",
            ]
        ).sum()
    )

    duplicate_peak = int(
        cost_peak.duplicated(
            subset=["edge_id"]
        ).sum()
    )

    print(
        f"Duplicate timeseries     : "
        f"{duplicate_ts}"
    )

    print(
        f"Duplicate peak IDs       : "
        f"{duplicate_peak}"
    )

    if duplicate_ts:
        errors.append(
            "duplicate routing-cost timeseries rows"
        )

    if duplicate_peak:
        errors.append(
            "duplicate routing-cost peak IDs"
        )

    wrong_ts_cost = 0
    wrong_ts_multiplier = 0
    wrong_ts_basis = 0
    wrong_ts_policy = 0
    passability_changed_ts = 0

    pass_ts_lookup = {
        (
            str(row.edge_id),
            str(row.interval_id),
        ): int(row.is_passable)
        for row in pass_ts.itertuples(
            index=False
        )
    }

    for row in cost_ts.itertuples(
        index=False
    ):
        expected, multiplier, basis = (
            expected_cost(
                row.length_m,
                row.flood_risk,
                row.is_passable,
            )
        )

        tolerance = max(
            EPS,
            abs(expected) * 1e-12,
        )

        if abs(
            float(row.routing_cost)
            - expected
        ) > tolerance:
            wrong_ts_cost += 1

        if abs(
            float(row.routing_multiplier)
            - multiplier
        ) > EPS:
            wrong_ts_multiplier += 1

        if str(
            row.routing_cost_basis
        ) != basis:
            wrong_ts_basis += 1

        if str(
            row.routing_cost_policy
        ) != POLICY_NAME:
            wrong_ts_policy += 1

        key = (
            str(row.edge_id),
            str(row.interval_id),
        )

        if int(
            row.is_passable
        ) != pass_ts_lookup[
            key
        ]:
            passability_changed_ts += 1

    wrong_peak_cost = 0
    wrong_peak_multiplier = 0
    wrong_peak_basis = 0
    wrong_peak_policy = 0
    passability_changed_peak = 0

    pass_peak_lookup = {
        str(row.edge_id): int(
            row.is_passable
        )
        for row in pass_peak.itertuples(
            index=False
        )
    }

    for row in cost_peak.itertuples(
        index=False
    ):
        expected, multiplier, basis = (
            expected_cost(
                row.length_m,
                row.flood_risk,
                row.is_passable,
            )
        )

        tolerance = max(
            EPS,
            abs(expected) * 1e-12,
        )

        if abs(
            float(row.routing_cost)
            - expected
        ) > tolerance:
            wrong_peak_cost += 1

        if abs(
            float(row.routing_multiplier)
            - multiplier
        ) > EPS:
            wrong_peak_multiplier += 1

        if str(
            row.routing_cost_basis
        ) != basis:
            wrong_peak_basis += 1

        if str(
            row.routing_cost_policy
        ) != POLICY_NAME:
            wrong_peak_policy += 1

        edge_id = str(
            row.edge_id
        )

        if int(
            row.is_passable
        ) != pass_peak_lookup[
            edge_id
        ]:
            passability_changed_peak += 1

    print("\nRouting-cost checks")
    print(
        f"Wrong interval costs      : "
        f"{wrong_ts_cost}"
    )
    print(
        f"Wrong interval multipliers: "
        f"{wrong_ts_multiplier}"
    )
    print(
        f"Wrong interval basis      : "
        f"{wrong_ts_basis}"
    )
    print(
        f"Wrong interval policy     : "
        f"{wrong_ts_policy}"
    )
    print(
        f"Wrong peak costs          : "
        f"{wrong_peak_cost}"
    )
    print(
        f"Wrong peak multipliers    : "
        f"{wrong_peak_multiplier}"
    )
    print(
        f"Wrong peak basis          : "
        f"{wrong_peak_basis}"
    )
    print(
        f"Wrong peak policy         : "
        f"{wrong_peak_policy}"
    )
    print(
        f"Timeseries passability changed: "
        f"{passability_changed_ts}"
    )
    print(
        f"Peak passability changed      : "
        f"{passability_changed_peak}"
    )

    for count, message in [
        (
            wrong_ts_cost,
            "interval routing-cost mismatch",
        ),
        (
            wrong_ts_multiplier,
            "interval routing multiplier mismatch",
        ),
        (
            wrong_ts_basis,
            "interval routing-cost basis mismatch",
        ),
        (
            wrong_ts_policy,
            "interval routing-cost policy mismatch",
        ),
        (
            wrong_peak_cost,
            "peak routing-cost mismatch",
        ),
        (
            wrong_peak_multiplier,
            "peak routing multiplier mismatch",
        ),
        (
            wrong_peak_basis,
            "peak routing-cost basis mismatch",
        ),
        (
            wrong_peak_policy,
            "peak routing-cost policy mismatch",
        ),
        (
            passability_changed_ts,
            "timeseries passability changed",
        ),
        (
            passability_changed_peak,
            "peak passability changed",
        ),
    ]:
        if count:
            errors.append(message)

    # ----------------------------------------------------------
    # Graph consistency
    # ----------------------------------------------------------
    peak_lookup = {
        str(row.edge_id): row
        for row in cost_peak.itertuples(
            index=False
        )
    }

    graph_cost_bad = 0
    graph_passability_bad = 0
    missing_graph_mapping = 0

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):
        edge_id = str(
            data.get("road_id", "")
        )

        if edge_id not in peak_lookup:
            missing_graph_mapping += 1
            continue

        record = peak_lookup[
            edge_id
        ]

        expected_graph_cost = float(
            record.routing_cost
        )

        tolerance = max(
            EPS,
            abs(
                expected_graph_cost
            )
            * 1e-12,
        )

        if abs(
            float(
                data["routing_cost"]
            )
            - expected_graph_cost
        ) > tolerance:
            graph_cost_bad += 1

        if int(
            float(
                data["is_passable"]
            )
        ) != int(
            record.is_passable
        ):
            graph_passability_bad += 1

    policy_flag = int(
        float(
            graph.graph.get(
                "routing_cost_policy_applied",
                0,
            )
        )
    )

    travel_time_flag = int(
        float(
            graph.graph.get(
                "routing_cost_policy_is_travel_time_model",
                1,
            )
        )
    )

    must_filter_flag = int(
        float(
            graph.graph.get(
                "safe_route_must_filter_blocked_edges",
                0,
            )
        )
    )

    sentinel = float(
        graph.graph.get(
            "blocked_cost_sentinel",
            -1,
        )
    )

    print("\nGraph checks")
    print(
        f"Graph cost mismatches     : "
        f"{graph_cost_bad}"
    )
    print(
        f"Graph passability mismatch: "
        f"{graph_passability_bad}"
    )
    print(
        f"Missing graph mappings    : "
        f"{missing_graph_mapping}"
    )
    print(
        f"Routing-cost policy flag  : "
        f"{policy_flag}"
    )
    print(
        f"Travel-time-model flag    : "
        f"{travel_time_flag}"
    )
    print(
        f"Must-filter-blocked flag  : "
        f"{must_filter_flag}"
    )
    print(
        f"Blocked sentinel          : "
        f"{sentinel:.0f}"
    )

    if graph_cost_bad:
        errors.append(
            "graph routing-cost mismatch"
        )

    if graph_passability_bad:
        errors.append(
            "graph passability changed"
        )

    if missing_graph_mapping:
        errors.append(
            "graph edge missing routing-cost mapping"
        )

    if policy_flag != 1:
        errors.append(
            "routing-cost policy flag not applied"
        )

    if travel_time_flag != 0:
        errors.append(
            "prototype cost marked as travel-time model"
        )

    if must_filter_flag != 1:
        errors.append(
            "safe-routing blocked-edge filter flag missing"
        )

    if abs(
        sentinel
        - BLOCKED_COST_SENTINEL
    ) > 1.0:
        errors.append(
            "blocked sentinel metadata mismatch"
        )

    passable_peak = cost_peak[
        cost_peak["is_passable"]
        == 1
    ]

    blocked_peak = cost_peak[
        cost_peak["is_passable"]
        == 0
    ]

    blocked_bad = int(
        (
            abs(
                blocked_peak[
                    "routing_cost"
                ].astype(float)
                - BLOCKED_COST_SENTINEL
            )
            > 1.0
        ).sum()
    )

    passable_below_length = int(
        (
            passable_peak[
                "routing_cost"
            ].astype(float)
            + EPS
            < passable_peak[
                "length_m"
            ].astype(float)
        ).sum()
    )

    print("\nScenario peak")
    print(
        f"Passable roads            : "
        f"{len(passable_peak)}"
    )
    print(
        f"Blocked roads             : "
        f"{len(blocked_peak)}"
    )
    print(
        f"Blocked sentinel errors   : "
        f"{blocked_bad}"
    )
    print(
        f"Passable cost < length    : "
        f"{passable_below_length}"
    )

    if blocked_bad:
        errors.append(
            "blocked edge does not use sentinel cost"
        )

    if passable_below_length:
        errors.append(
            "passable routing cost below road length"
        )

    print("\nIMPORTANT:")
    print(
        "Routing multipliers are prototype preferences, "
        "not calibrated travel-time penalties."
    )
    print(
        "Safe-route generation must filter blocked edges "
        "instead of relying only on the sentinel."
    )

    print(
        "\n========================================="
    )

    if errors:
        print(
            "FLOOD-AWARE ROUTING-COST VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )
    else:
        print(
            "FLOOD-AWARE ROUTING-COST VALIDATION: PASSED"
        )

    print(
        "=========================================\n"
    )


if __name__ == "__main__":
    main()
