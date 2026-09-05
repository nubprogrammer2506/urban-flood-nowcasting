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

PASSABILITY_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_passability.graphml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
)

ROUTING_COST_TIMESERIES_OUTPUT = (
    OUTPUT_DIR
    / "road_routing_cost_timeseries.csv"
)

ROUTING_COST_PEAK_OUTPUT = (
    OUTPUT_DIR
    / "road_routing_cost_peak.geojson"
)

ROUTING_COST_GRAPH_OUTPUT = (
    OUTPUT_DIR
    / "road_graph_routing_cost.graphml"
)

# ------------------------------------------------------------------
# Prototype flood-aware routing-cost policy
# ------------------------------------------------------------------
# This is a hackathon/MVP routing preference, NOT an official
# transport standard and NOT a calibrated travel-time model.
#
# Passable edges:
#   minimal  -> 1.0 x road length
#   low      -> 1.5 x road length
#   moderate -> 3.0 x road length
#
# Blocked edges:
#   high / unknown -> BLOCKED_COST_SENTINEL
#
# Safe-route generation must STILL filter is_passable == 0.
# The sentinel is only a defensive fallback to discourage accidental
# traversal if a caller forgets to remove blocked edges.
#
POLICY_NAME = "prototype_risk_weighted_cost_v1"

RISK_MULTIPLIER = {
    "minimal": 1.0,
    "low": 1.5,
    "moderate": 3.0,
}

BLOCKED_COST_SENTINEL = 1.0e12


def routing_cost(length_m, flood_risk, is_passable):
    length = float(length_m)
    risk = str(flood_risk)
    passable = int(is_passable)

    if length <= 0:
        raise ValueError(
            f"Road length must be positive, got {length}"
        )

    if passable == 0:
        return (
            BLOCKED_COST_SENTINEL,
            0.0,
            "blocked_sentinel",
        )

    if risk not in RISK_MULTIPLIER:
        raise ValueError(
            f"Passable edge has unsupported flood risk: {risk}"
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
        "\n=== FLOOD-AWARE ROUTING COST ===\n"
    )

    for path in [
        PASSABILITY_TIMESERIES_PATH,
        PASSABILITY_PEAK_PATH,
        PASSABILITY_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

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

    # --------------------------------------------------------------
    # Per-interval routing cost
    # --------------------------------------------------------------
    cost_ts = pass_ts.copy()

    ts_cost = []
    ts_multiplier = []
    ts_basis = []

    for row in cost_ts.itertuples(
        index=False
    ):
        cost, multiplier, basis = routing_cost(
            row.length_m,
            row.flood_risk,
            row.is_passable,
        )

        ts_cost.append(cost)
        ts_multiplier.append(multiplier)
        ts_basis.append(basis)

    cost_ts["routing_cost"] = ts_cost
    cost_ts["routing_multiplier"] = (
        ts_multiplier
    )
    cost_ts["routing_cost_basis"] = (
        ts_basis
    )
    cost_ts["routing_cost_policy"] = (
        POLICY_NAME
    )

    cost_ts.to_csv(
        ROUTING_COST_TIMESERIES_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------------
    # Scenario-peak routing cost
    # --------------------------------------------------------------
    cost_peak = pass_peak.copy()

    peak_cost = []
    peak_multiplier = []
    peak_basis = []

    for row in cost_peak.itertuples(
        index=False
    ):
        cost, multiplier, basis = routing_cost(
            row.length_m,
            row.flood_risk,
            row.is_passable,
        )

        peak_cost.append(cost)
        peak_multiplier.append(multiplier)
        peak_basis.append(basis)

    cost_peak["routing_cost"] = peak_cost
    cost_peak["routing_multiplier"] = (
        peak_multiplier
    )
    cost_peak["routing_cost_basis"] = (
        peak_basis
    )
    cost_peak["routing_cost_policy"] = (
        POLICY_NAME
    )

    cost_peak.to_file(
        ROUTING_COST_PEAK_OUTPUT,
        driver="GeoJSON",
    )

    # --------------------------------------------------------------
    # Enrich graph with scenario-peak routing cost
    # --------------------------------------------------------------
    cost_lookup = {
        str(row.edge_id): row
        for row in cost_peak.itertuples(
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

        if edge_id not in cost_lookup:
            raise ValueError(
                f"Missing routing-cost record for edge {edge_id}"
            )

        record = cost_lookup[
            edge_id
        ]

        data["routing_cost"] = float(
            record.routing_cost
        )

        data["routing_multiplier"] = float(
            record.routing_multiplier
        )

        data["routing_cost_basis"] = str(
            record.routing_cost_basis
        )

        data["routing_cost_policy"] = (
            POLICY_NAME
        )

    graph.graph["routing_cost_policy_applied"] = 1
    graph.graph["routing_cost_policy"] = POLICY_NAME
    graph.graph["routing_cost_policy_is_travel_time_model"] = 0
    graph.graph["blocked_cost_sentinel"] = float(
        BLOCKED_COST_SENTINEL
    )
    graph.graph["safe_route_must_filter_blocked_edges"] = 1

    nx.write_graphml(
        graph,
        ROUTING_COST_GRAPH_OUTPUT,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    passable = cost_peak[
        cost_peak["is_passable"]
        == 1
    ].copy()

    blocked = cost_peak[
        cost_peak["is_passable"]
        == 0
    ].copy()

    print(
        f"Road edges              : "
        f"{len(cost_peak)}"
    )

    print(
        f"Passable edges          : "
        f"{len(passable)}"
    )

    print(
        f"Blocked edges           : "
        f"{len(blocked)}"
    )

    print(
        f"Routing-cost policy     : "
        f"{POLICY_NAME}"
    )

    print("\nPassable edge multipliers")
    for risk in [
        "minimal",
        "low",
        "moderate",
    ]:
        count = int(
            (
                passable[
                    "flood_risk"
                ]
                == risk
            ).sum()
        )

        print(
            f"  {risk:<8}: "
            f"{RISK_MULTIPLIER[risk]:.1f}x "
            f"| edges {count}"
        )

    print(
        f"\nBlocked sentinel       : "
        f"{BLOCKED_COST_SENTINEL:.0f}"
    )

    print("\nSaved:")
    print(ROUTING_COST_TIMESERIES_OUTPUT)
    print(ROUTING_COST_PEAK_OUTPUT)
    print(ROUTING_COST_GRAPH_OUTPUT)

    print("\nIMPORTANT:")
    print(
        "Routing multipliers are prototype preferences, "
        "not calibrated travel-time penalties."
    )
    print(
        "Safe routing must filter is_passable == 0; "
        "the blocked sentinel is only a defensive fallback."
    )

    print(
        "\nFlood-aware routing-cost policy completed."
    )


if __name__ == "__main__":
    main()
