from pathlib import Path
import json

import geopandas as gpd
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLOOD_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_routing_cost.graphml"
)

ROUTE_GEOJSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "routing"
    / "safe_route.geojson"
)

ROUTE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "routing"
    / "safe_route_summary.json"
)

EPS = 1e-6


def as_int(value):
    return int(float(value))


def as_float(value):
    return float(value)


def build_passable_graph(graph):
    passable = nx.MultiGraph()

    for node, data in graph.nodes(
        data=True
    ):
        passable.add_node(
            node,
            **data,
        )

    for u, v, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        if as_int(
            data.get(
                "is_passable",
                0,
            )
        ) != 1:
            continue

        passable.add_edge(
            u,
            v,
            key=key,
            **data,
        )

    return passable


def main():
    print(
        "\\n=== SAFE ROUTE VALIDATION ===\\n"
    )

    for path in [
        FLOOD_GRAPH_PATH,
        ROUTE_GEOJSON_PATH,
        ROUTE_SUMMARY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    graph = nx.read_graphml(
        FLOOD_GRAPH_PATH,
        force_multigraph=True,
    )

    passable = build_passable_graph(
        graph
    )

    route = gpd.read_file(
        ROUTE_GEOJSON_PATH
    )

    summary = json.loads(
        ROUTE_SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    errors = []

    start = str(
        summary[
            "start_node"
        ]
    )

    end = str(
        summary[
            "end_node"
        ]
    )

    print(
        f"Start node             : "
        f"{start}"
    )

    print(
        f"End node               : "
        f"{end}"
    )

    print(
        f"Route edges            : "
        f"{len(route)}"
    )

    print(
        f"Summary route edges    : "
        f"{summary['safe_route_edge_count']}"
    )

    if len(route) != int(
        summary[
            "safe_route_edge_count"
        ]
    ):
        errors.append(
            "route edge count differs from summary"
        )

    if route.empty:
        errors.append(
            "safe route is empty"
        )

    # ----------------------------------------------------------
    # Sequence and connectivity
    # ----------------------------------------------------------
    sequence_bad = 0
    connectivity_bad = 0

    route_sorted = route.sort_values(
        "sequence"
    )

    expected_sequence = list(
        range(
            1,
            len(route_sorted) + 1,
        )
    )

    actual_sequence = [
        int(value)
        for value in route_sorted[
            "sequence"
        ]
    ]

    if actual_sequence != expected_sequence:
        sequence_bad = 1

    previous_to = None

    for row in route_sorted.itertuples(
        index=False
    ):
        if previous_to is not None:
            if str(
                row.from_node
            ) != str(
                previous_to
            ):
                connectivity_bad += 1

        previous_to = str(
            row.to_node
        )

    if not route_sorted.empty:
        if str(
            route_sorted.iloc[0][
                "from_node"
            ]
        ) != start:
            connectivity_bad += 1

        if str(
            route_sorted.iloc[-1][
                "to_node"
            ]
        ) != end:
            connectivity_bad += 1

    print(
        "\\nPath structure"
    )

    print(
        f"Sequence errors         : "
        f"{sequence_bad}"
    )

    print(
        f"Connectivity errors     : "
        f"{connectivity_bad}"
    )

    if sequence_bad:
        errors.append(
            "route sequence is invalid"
        )

    if connectivity_bad:
        errors.append(
            "route edge sequence is disconnected"
        )

    # ----------------------------------------------------------
    # Safety and metric checks
    # ----------------------------------------------------------
    blocked_route_edges = int(
        (
            route[
                "is_passable"
            ].astype(int)
            != 1
        ).sum()
    )

    negative_costs = int(
        (
            route[
                "routing_cost"
            ].astype(float)
            < 0
        ).sum()
    )

    total_length = float(
        route[
            "length_m"
        ].astype(float).sum()
    )

    total_cost = float(
        route[
            "routing_cost"
        ].astype(float).sum()
    )

    route_max_depth = float(
        route[
            "flood_depth_m"
        ].astype(float).max()
    ) if not route.empty else 0.0

    length_error = abs(
        total_length
        - float(
            summary[
                "safe_route_length_m"
            ]
        )
    )

    cost_error = abs(
        total_cost
        - float(
            summary[
                "safe_route_routing_cost"
            ]
        )
    )

    depth_error = abs(
        route_max_depth
        - float(
            summary[
                "safe_route_max_flood_depth_m"
            ]
        )
    )

    print(
        "\\nSafety and metric checks"
    )

    print(
        f"Blocked edges in route  : "
        f"{blocked_route_edges}"
    )

    print(
        f"Negative routing costs  : "
        f"{negative_costs}"
    )

    print(
        f"Route length error      : "
        f"{length_error:.9f}"
    )

    print(
        f"Route cost error        : "
        f"{cost_error:.9f}"
    )

    print(
        f"Max-depth error         : "
        f"{depth_error:.9f}"
    )

    if blocked_route_edges:
        errors.append(
            "safe route traverses blocked edge"
        )

    if negative_costs:
        errors.append(
            "safe route contains negative routing cost"
        )

    if length_error > EPS:
        errors.append(
            "route length summary mismatch"
        )

    if cost_error > EPS:
        errors.append(
            "route cost summary mismatch"
        )

    if depth_error > EPS:
        errors.append(
            "route max-depth summary mismatch"
        )

    # ----------------------------------------------------------
    # Prove routing-cost optimality on the passable graph.
    # ----------------------------------------------------------
    if (
        start not in passable
        or end not in passable
        or not nx.has_path(
            passable,
            start,
            end,
        )
    ):
        optimality_error = float(
            "inf"
        )
        errors.append(
            "summary endpoints are not connected "
            "in the passable graph"
        )
    else:
        optimum = nx.shortest_path_length(
            passable,
            start,
            end,
            weight="routing_cost",
        )

        optimality_error = abs(
            total_cost
            - float(
                optimum
            )
        )

        if optimality_error > EPS:
            errors.append(
                "route is not minimum routing-cost path"
            )

    print(
        f"Optimality cost error   : "
        f"{optimality_error:.9f}"
    )

    # ----------------------------------------------------------
    # Scenario graph metadata / known limitation.
    # ----------------------------------------------------------
    blocked_in_graph = (
        graph.number_of_edges()
        - passable.number_of_edges()
    )

    summary_filtered = int(
        summary[
            "blocked_edges_filtered"
        ]
    )

    undirected = (
        not graph.is_directed()
    )

    directed_ready = int(
        float(
            graph.graph.get(
                "directed_routing_ready",
                0,
            )
        )
    )

    print(
        "\\nGraph checks"
    )

    print(
        f"Blocked edges in graph  : "
        f"{blocked_in_graph}"
    )

    print(
        f"Summary filtered edges  : "
        f"{summary_filtered}"
    )

    print(
        f"Graph is undirected     : "
        f"{undirected}"
    )

    print(
        f"Directed routing ready  : "
        f"{directed_ready}"
    )

    if blocked_in_graph != summary_filtered:
        errors.append(
            "blocked-edge filter count mismatch"
        )

    if int(
        summary[
            "safe_route_blocked_edge_count"
        ]
    ) != 0:
        errors.append(
            "summary reports blocked edge in safe route"
        )

    baseline = summary.get(
        "baseline_shortest_distance_route",
        {},
    )

    if baseline.get(
        "available",
        False,
    ):
        print(
            "\\nBaseline comparison"
        )

        print(
            f"Baseline distance      : "
            f"{float(baseline['length_m']):.2f} m"
        )

        print(
            f"Baseline blocked edges : "
            f"{int(baseline['blocked_edge_count_under_flood_policy'])}"
        )

        print(
            f"Safe route distance    : "
            f"{total_length:.2f} m"
        )

    print(
        "\\nIMPORTANT:"
    )

    print(
        "Validation confirms the route excludes blocked "
        "scenario edges and minimizes prototype routing_cost."
    )

    print(
        "The current road graph is undirected, so OSM one-way "
        "directionality is not enforced yet."
    )

    print(
        "\\n===================================="
    )

    if errors:
        print(
            "SAFE ROUTE VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )
    else:
        print(
            "SAFE ROUTE VALIDATION: PASSED"
        )

    print(
        "====================================\\n"
    )


if __name__ == "__main__":
    main()
