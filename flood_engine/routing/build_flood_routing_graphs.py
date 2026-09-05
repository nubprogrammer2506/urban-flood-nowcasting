from pathlib import Path

import geopandas as gpd
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_GRAPH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph.graphml"
)

ROAD_RISK_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "road_risk"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "routing_graphs"
)


INTERVALS = [
    "000_030",
    "030_060",
    "060_090",
    "090_120",
    "120_150",
    "150_180",
]


RISK_MULTIPLIER = {
    "safe": 1.0,
    "low": 1.5,
    "medium": 4.0,
    "high": 1000.0,
}


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not BASE_GRAPH.exists():
        raise FileNotFoundError(
            f"Road graph not found: {BASE_GRAPH}"
        )

    print("Flood-aware routing graph generation")
    print()

    for interval in INTERVALS:

        risk_file = (
            ROAD_RISK_DIR
            / f"road_risk_{interval}.geojson"
        )

        if not risk_file.exists():
            raise FileNotFoundError(
                risk_file
            )

        # -------------------------------------
        # Fresh baseline graph each interval
        # -------------------------------------
        graph = nx.read_graphml(
            BASE_GRAPH
        )

        roads = gpd.read_file(
            risk_file
        )

        required = {
            "road_id",
            "flood_max_m",
            "flood_risk",
            "is_passable",
        }

        missing = (
            required
            - set(roads.columns)
        )

        if missing:
            raise ValueError(
                f"{interval}: missing fields "
                f"{missing}"
            )

        # -------------------------------------
        # Lookup by canonical road ID
        # -------------------------------------
        risk_lookup = {}

        for row in roads.itertuples(
            index=False
        ):

            road_id = str(
                row.road_id
            )

            risk_lookup[road_id] = {
                "flood_depth_m": float(
                    row.flood_max_m
                ),
                "flood_risk": str(
                    row.flood_risk
                ),
                "is_passable": int(
                    row.is_passable
                ),
            }

        updated = 0
        missing_edges = 0
        blocked_edges = 0

        risk_counts = {
            "safe": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
        }

        # -------------------------------------
        # Update graph edges
        # -------------------------------------
        for (
            u,
            v,
            key,
            data,
        ) in graph.edges(
            keys=True,
            data=True,
        ):

            source_road_id = str(
                data.get(
                    "source_road_id",
                    ""
                )
            )

            if (
                source_road_id
                not in risk_lookup
            ):
                missing_edges += 1
                continue

            risk = risk_lookup[
                source_road_id
            ]

            flood_risk = risk[
                "flood_risk"
            ]

            if (
                flood_risk
                not in RISK_MULTIPLIER
            ):
                raise ValueError(
                    f"Unknown risk class: "
                    f"{flood_risk}"
                )

            edge_length = float(
                data["length_m"]
            )

            multiplier = (
                RISK_MULTIPLIER[
                    flood_risk
                ]
            )

            graph[u][v][key][
                "flood_depth_m"
            ] = risk[
                "flood_depth_m"
            ]

            graph[u][v][key][
                "flood_risk"
            ] = flood_risk

            graph[u][v][key][
                "is_passable"
            ] = risk[
                "is_passable"
            ]

            # Use this graph edge's own
            # length, not full source-road
            # length.
            graph[u][v][key][
                "routing_cost"
            ] = (
                edge_length
                * multiplier
            )

            updated += 1

            risk_counts[
                flood_risk
            ] += 1

            if (
                risk[
                    "is_passable"
                ]
                == 0
            ):
                blocked_edges += 1

        if missing_edges > 0:
            raise RuntimeError(
                f"{interval}: "
                f"{missing_edges} graph edges "
                "could not be matched to "
                "road-risk records"
            )

        # -------------------------------------
        # Graph metadata
        # -------------------------------------
        graph.graph[
            "flood_interval"
        ] = interval

        graph.graph[
            "flood_routing_ready"
        ] = 1

        graph.graph[
            "routing_model"
        ] = (
            "prototype_flood_weighted_"
            "networkx"
        )

        output_file = (
            OUTPUT_DIR
            / f"road_graph_{interval}.graphml"
        )

        nx.write_graphml(
            graph,
            output_file,
        )

        print(
            f"{interval} | "
            f"edges {graph.number_of_edges():>4} | "
            f"updated {updated:>4} | "
            f"safe {risk_counts['safe']:>4} | "
            f"low {risk_counts['low']:>3} | "
            f"medium {risk_counts['medium']:>3} | "
            f"high {risk_counts['high']:>3} | "
            f"blocked {blocked_edges:>3}"
        )

    print()
    print(
        "Flood-aware routing graphs "
        "generated successfully."
    )


if __name__ == "__main__":
    main()