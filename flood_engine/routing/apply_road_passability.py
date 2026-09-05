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

RISK_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_flood_risk.graphml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
)

PASSABILITY_TIMESERIES_OUTPUT = (
    OUTPUT_DIR
    / "road_passability_timeseries.csv"
)

PASSABILITY_PEAK_OUTPUT = (
    OUTPUT_DIR
    / "road_passability_peak.geojson"
)

PASSABILITY_GRAPH_OUTPUT = (
    OUTPUT_DIR
    / "road_graph_passability.graphml"
)

# ------------------------------------------------------------------
# Prototype passability policy
# ------------------------------------------------------------------
# This is a hackathon/MVP routing policy, NOT an official road-safety
# standard and NOT a calibrated vehicle-depth threshold.
#
# Passable:
#   minimal, low, moderate
#
# Blocked:
#   high, unknown
#
# "unknown" is blocked conservatively because the road edge was not
# mapped to a valid flood-depth raster cell.
#
POLICY_NAME = "prototype_passability_v1"

PASSABILITY_BY_RISK = {
    "minimal": 1,
    "low": 1,
    "moderate": 1,
    "high": 0,
    "unknown": 0,
}


def passability_from_risk(risk):
    risk = str(risk)

    if risk not in PASSABILITY_BY_RISK:
        raise ValueError(
            f"Unsupported flood-risk class: {risk}"
        )

    return PASSABILITY_BY_RISK[risk]


def main():
    print("\n=== ROAD PASSABILITY POLICY ===\n")

    for path in [
        RISK_TIMESERIES_PATH,
        RISK_PEAK_PATH,
        RISK_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    risk_ts = pd.read_csv(
        RISK_TIMESERIES_PATH
    )

    risk_peak = gpd.read_file(
        RISK_PEAK_PATH
    )

    graph = nx.read_graphml(
        RISK_GRAPH_PATH,
        force_multigraph=True,
    )

    # --------------------------------------------------------------
    # Per-interval passability
    # --------------------------------------------------------------
    passability_ts = risk_ts.copy()

    passability_ts["is_passable"] = [
        passability_from_risk(risk)
        for risk in passability_ts["flood_risk"]
    ]

    passability_ts["passability_status"] = [
        "passable" if value == 1 else "blocked"
        for value in passability_ts["is_passable"]
    ]

    passability_ts["passability_policy"] = (
        POLICY_NAME
    )

    passability_ts.to_csv(
        PASSABILITY_TIMESERIES_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------------
    # Scenario peak passability
    # --------------------------------------------------------------
    passability_peak = risk_peak.copy()

    passability_peak["is_passable"] = [
        passability_from_risk(risk)
        for risk in passability_peak["flood_risk"]
    ]

    passability_peak["passability_status"] = [
        "passable" if value == 1 else "blocked"
        for value in passability_peak["is_passable"]
    ]

    passability_peak["passability_policy"] = (
        POLICY_NAME
    )

    passability_peak.to_file(
        PASSABILITY_PEAK_OUTPUT,
        driver="GeoJSON",
    )

    # --------------------------------------------------------------
    # Enrich graph.
    # Keep routing_cost unchanged until the next milestone.
    # --------------------------------------------------------------
    passability_lookup = {
        str(row.edge_id): row
        for row in passability_peak.itertuples(
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

        if edge_id not in passability_lookup:
            raise ValueError(
                f"Missing passability record for edge {edge_id}"
            )

        record = passability_lookup[
            edge_id
        ]

        data["is_passable"] = int(
            record.is_passable
        )

        data["passability_status"] = str(
            record.passability_status
        )

        data["passability_policy"] = (
            POLICY_NAME
        )

        # Preserve baseline routing cost. Flood-aware weighting is P2.20.
        data["routing_cost"] = float(
            data.get(
                "routing_cost",
                data["length_m"],
            )
        )

    graph.graph["passability_policy_applied"] = 1
    graph.graph["passability_policy"] = POLICY_NAME
    graph.graph["passability_policy_is_safety_standard"] = 0
    graph.graph["routing_cost_policy_applied"] = 0

    nx.write_graphml(
        graph,
        PASSABILITY_GRAPH_OUTPUT,
    )

    print(
        f"Road edges              : "
        f"{len(passability_peak)}"
    )

    passable_count = int(
        (
            passability_peak[
                "is_passable"
            ]
            == 1
        ).sum()
    )

    blocked_count = int(
        (
            passability_peak[
                "is_passable"
            ]
            == 0
        ).sum()
    )

    high_blocked = int(
        (
            (
                passability_peak[
                    "flood_risk"
                ]
                == "high"
            )
            &
            (
                passability_peak[
                    "is_passable"
                ]
                == 0
            )
        ).sum()
    )

    unknown_blocked = int(
        (
            (
                passability_peak[
                    "flood_risk"
                ]
                == "unknown"
            )
            &
            (
                passability_peak[
                    "is_passable"
                ]
                == 0
            )
        ).sum()
    )

    print(
        f"Passable edges          : "
        f"{passable_count}"
    )

    print(
        f"Blocked edges           : "
        f"{blocked_count}"
    )

    print(
        f"High-risk blocked       : "
        f"{high_blocked}"
    )

    print(
        f"Unknown blocked         : "
        f"{unknown_blocked}"
    )

    print(
        f"Passability policy      : "
        f"{POLICY_NAME}"
    )

    print("\nRule")
    print(
        "  minimal / low / moderate -> passable"
    )
    print(
        "  high                     -> blocked"
    )
    print(
        "  unknown                  -> blocked conservatively"
    )

    print("\nSaved:")
    print(PASSABILITY_TIMESERIES_OUTPUT)
    print(PASSABILITY_PEAK_OUTPUT)
    print(PASSABILITY_GRAPH_OUTPUT)

    print("\nIMPORTANT:")
    print(
        "This is a prototype routing policy, "
        "not an official road-safety standard."
    )
    print(
        "routing_cost is intentionally unchanged; "
        "flood-aware routing cost is the next milestone."
    )

    print("\nRoad passability policy completed.")


if __name__ == "__main__":
    main()
