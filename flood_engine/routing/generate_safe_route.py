from pathlib import Path
import argparse
import json

import geopandas as gpd
import networkx as nx
from shapely import wkt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph.graphml"
)

FLOOD_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_routing_cost.graphml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "routing"
)

ROUTE_GEOJSON_OUTPUT = (
    OUTPUT_DIR
    / "safe_route.geojson"
)

ROUTE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "safe_route_summary.json"
)

WEIGHT_FIELD = "routing_cost"


def as_int(value):
    return int(float(value))


def as_float(value):
    return float(value)


def build_passable_graph(graph):
    """
    Build a graph containing only edges explicitly marked passable.

    This is required even though blocked edges already carry a very
    large sentinel routing cost. Filtering is the actual safety rule.
    """
    passable = nx.MultiGraph()

    passable.graph.update(graph.graph)

    for node, data in graph.nodes(data=True):
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


def choose_demo_pair(
    full_graph,
    passable_graph,
):
    """
    Prefer endpoints of a blocked edge that still have an alternate
    passable path. This makes the default demo visibly exercise the
    flood-aware rerouting logic.

    If no such blocked-edge pair exists, fall back to a far-apart pair
    inside the largest passable connected component.
    """
    candidates = []

    for u, v, key, data in full_graph.edges(
        keys=True,
        data=True,
    ):
        if as_int(
            data.get(
                "is_passable",
                0,
            )
        ) != 0:
            continue

        if (
            u not in passable_graph
            or v not in passable_graph
        ):
            continue

        if nx.has_path(
            passable_graph,
            u,
            v,
        ):
            candidates.append(
                (
                    str(
                        data.get(
                            "road_id",
                            "",
                        )
                    ),
                    str(u),
                    str(v),
                    str(key),
                )
            )

    if candidates:
        candidates.sort()

        road_id, start, end, key = (
            candidates[0]
        )

        return {
            "start_node": start,
            "end_node": end,
            "selection_method": (
                "blocked_edge_endpoints_with_passable_detour"
            ),
            "trigger_blocked_road_id": road_id,
            "trigger_blocked_edge_key": key,
        }

    components = list(
        nx.connected_components(
            passable_graph
        )
    )

    if not components:
        raise RuntimeError(
            "Passable road graph has no connected components."
        )

    largest = max(
        components,
        key=len,
    )

    seed = sorted(
        largest
    )[0]

    first_distances = nx.single_source_dijkstra_path_length(
        passable_graph.subgraph(
            largest
        ),
        seed,
        weight="length_m",
    )

    first = max(
        first_distances,
        key=first_distances.get,
    )

    second_distances = nx.single_source_dijkstra_path_length(
        passable_graph.subgraph(
            largest
        ),
        first,
        weight="length_m",
    )

    second = max(
        second_distances,
        key=second_distances.get,
    )

    return {
        "start_node": str(first),
        "end_node": str(second),
        "selection_method": (
            "largest_passable_component_far_pair"
        ),
        "trigger_blocked_road_id": "",
        "trigger_blocked_edge_key": "",
    }


def choose_edge_for_step(
    graph,
    u,
    v,
):
    """
    NetworkX shortest_path on a MultiGraph effectively uses the minimum
    parallel-edge weight. Resolve the concrete parallel edge using the
    same routing_cost rule.
    """
    edge_dict = graph.get_edge_data(
        u,
        v,
    )

    if not edge_dict:
        raise RuntimeError(
            f"No edge found for route step {u} -> {v}"
        )

    options = []

    for key, data in edge_dict.items():
        if as_int(
            data.get(
                "is_passable",
                0,
            )
        ) != 1:
            continue

        options.append(
            (
                as_float(
                    data[
                        WEIGHT_FIELD
                    ]
                ),
                str(key),
                key,
                data,
            )
        )

    if not options:
        raise RuntimeError(
            f"No passable edge for route step {u} -> {v}"
        )

    options.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    _, _, key, data = options[0]

    return key, data


def route_edges_from_nodes(
    graph,
    node_path,
):
    records = []

    total_length = 0.0
    total_cost = 0.0
    max_depth = 0.0
    max_risk_rank = -1
    risk_classes = []

    for sequence, (
        u,
        v,
    ) in enumerate(
        zip(
            node_path[:-1],
            node_path[1:],
        ),
        start=1,
    ):
        key, data = choose_edge_for_step(
            graph,
            u,
            v,
        )

        geometry_text = str(
            data.get(
                "geometry_wkt",
                "",
            )
        )

        if not geometry_text:
            raise RuntimeError(
                f"Missing geometry for road "
                f"{data.get('road_id', '')}"
            )

        geometry = wkt.loads(
            geometry_text
        )

        length_m = as_float(
            data["length_m"]
        )

        routing_cost = as_float(
            data[
                WEIGHT_FIELD
            ]
        )

        depth_m = as_float(
            data.get(
                "flood_depth_m",
                data.get(
                    "scenario_peak_flood_depth_m",
                    0.0,
                ),
            )
        )

        risk = str(
            data.get(
                "flood_risk",
                "unknown",
            )
        )

        risk_rank = as_int(
            data.get(
                "flood_risk_rank",
                -1,
            )
        )

        total_length += length_m
        total_cost += routing_cost

        if depth_m >= 0:
            max_depth = max(
                max_depth,
                depth_m,
            )

        max_risk_rank = max(
            max_risk_rank,
            risk_rank,
        )

        risk_classes.append(
            risk
        )

        records.append(
            {
                "sequence": sequence,
                "from_node": str(u),
                "to_node": str(v),
                "edge_key": str(key),
                "road_id": str(
                    data.get(
                        "road_id",
                        "",
                    )
                ),
                "source_road_id": str(
                    data.get(
                        "source_road_id",
                        "",
                    )
                ),
                "name": str(
                    data.get(
                        "name",
                        "",
                    )
                ),
                "road_class": str(
                    data.get(
                        "road_class",
                        "",
                    )
                ),
                "length_m": length_m,
                "routing_cost": routing_cost,
                "flood_depth_m": depth_m,
                "flood_risk": risk,
                "flood_risk_rank": risk_rank,
                "is_passable": as_int(
                    data[
                        "is_passable"
                    ]
                ),
                "geometry": geometry,
            }
        )

    return (
        records,
        total_length,
        total_cost,
        max_depth,
        max_risk_rank,
        risk_classes,
    )


def path_summary(
    graph,
    node_path,
):
    (
        records,
        total_length,
        total_cost,
        max_depth,
        max_risk_rank,
        risk_classes,
    ) = route_edges_from_nodes(
        graph,
        node_path,
    )

    return {
        "records": records,
        "length_m": total_length,
        "routing_cost": total_cost,
        "max_depth_m": max_depth,
        "max_risk_rank": max_risk_rank,
        "risk_classes": risk_classes,
    }


def baseline_comparison(
    base_graph,
    flood_graph,
    start,
    end,
):
    if not nx.has_path(
        base_graph,
        start,
        end,
    ):
        return {
            "available": False,
        }

    baseline_nodes = nx.shortest_path(
        base_graph,
        start,
        end,
        weight="length_m",
    )

    baseline_length = 0.0
    blocked_edges = 0
    baseline_road_ids = []

    for u, v in zip(
        baseline_nodes[:-1],
        baseline_nodes[1:],
    ):
        base_options = (
            base_graph.get_edge_data(
                u,
                v,
            )
            or {}
        )

        if not base_options:
            raise RuntimeError(
                "Baseline route contains a missing edge."
            )

        chosen_key, chosen_data = min(
            base_options.items(),
            key=lambda item: float(
                item[1]["length_m"]
            ),
        )

        baseline_length += float(
            chosen_data[
                "length_m"
            ]
        )

        road_id = str(
            chosen_data.get(
                "road_id",
                "",
            )
        )

        baseline_road_ids.append(
            road_id
        )

        flood_options = (
            flood_graph.get_edge_data(
                u,
                v,
            )
            or {}
        )

        corresponding = []

        for _, flood_data in (
            flood_options.items()
        ):
            if str(
                flood_data.get(
                    "road_id",
                    "",
                )
            ) == road_id:
                corresponding.append(
                    flood_data
                )

        if corresponding:
            if all(
                as_int(
                    item.get(
                        "is_passable",
                        0,
                    )
                )
                == 0
                for item in corresponding
            ):
                blocked_edges += 1

    return {
        "available": True,
        "node_count": len(
            baseline_nodes
        ),
        "edge_count": max(
            0,
            len(baseline_nodes) - 1,
        ),
        "length_m": baseline_length,
        "blocked_edge_count_under_flood_policy": (
            blocked_edges
        ),
        "road_ids": baseline_road_ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a flood-aware safe route "
            "on the scenario-peak road graph."
        )
    )

    parser.add_argument(
        "--start-node",
        default=None,
    )

    parser.add_argument(
        "--end-node",
        default=None,
    )

    args = parser.parse_args()

    print(
        "\\n=== SAFE ROUTE GENERATION ===\\n"
    )

    for path in [
        BASE_GRAPH_PATH,
        FLOOD_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_graph = nx.read_graphml(
        BASE_GRAPH_PATH,
        force_multigraph=True,
    )

    flood_graph = nx.read_graphml(
        FLOOD_GRAPH_PATH,
        force_multigraph=True,
    )

    passable_graph = build_passable_graph(
        flood_graph
    )

    blocked_edges = (
        flood_graph.number_of_edges()
        - passable_graph.number_of_edges()
    )

    print(
        f"Scenario graph nodes    : "
        f"{flood_graph.number_of_nodes()}"
    )

    print(
        f"Scenario graph edges    : "
        f"{flood_graph.number_of_edges()}"
    )

    print(
        f"Passable edges          : "
        f"{passable_graph.number_of_edges()}"
    )

    print(
        f"Blocked edges filtered  : "
        f"{blocked_edges}"
    )

    if bool(
        args.start_node
    ) != bool(
        args.end_node
    ):
        raise ValueError(
            "Provide both --start-node and --end-node, "
            "or neither for automatic demo selection."
        )

    if (
        args.start_node is not None
        and args.end_node is not None
    ):
        selection = {
            "start_node": str(
                args.start_node
            ),
            "end_node": str(
                args.end_node
            ),
            "selection_method": (
                "user_supplied_nodes"
            ),
            "trigger_blocked_road_id": "",
            "trigger_blocked_edge_key": "",
        }
    else:
        selection = choose_demo_pair(
            flood_graph,
            passable_graph,
        )

    start = selection[
        "start_node"
    ]

    end = selection[
        "end_node"
    ]

    if start not in flood_graph:
        raise ValueError(
            f"Unknown start node: {start}"
        )

    if end not in flood_graph:
        raise ValueError(
            f"Unknown end node: {end}"
        )

    if start not in passable_graph:
        raise nx.NetworkXNoPath(
            f"Start node {start} has no passable access."
        )

    if end not in passable_graph:
        raise nx.NetworkXNoPath(
            f"End node {end} has no passable access."
        )

    if not nx.has_path(
        passable_graph,
        start,
        end,
    ):
        raise nx.NetworkXNoPath(
            f"No passable route exists between "
            f"{start} and {end}."
        )

    safe_nodes = nx.shortest_path(
        passable_graph,
        start,
        end,
        weight=WEIGHT_FIELD,
    )

    safe = path_summary(
        passable_graph,
        safe_nodes,
    )

    route_gdf = gpd.GeoDataFrame(
        safe[
            "records"
        ],
        geometry="geometry",
        crs="EPSG:32643",
    )

    route_gdf.to_file(
        ROUTE_GEOJSON_OUTPUT,
        driver="GeoJSON",
    )

    baseline = baseline_comparison(
        base_graph,
        flood_graph,
        start,
        end,
    )

    unique_risks = sorted(
        set(
            safe[
                "risk_classes"
            ]
        )
    )

    summary = {
        "start_node": start,
        "end_node": end,
        "selection_method": selection[
            "selection_method"
        ],
        "trigger_blocked_road_id": selection[
            "trigger_blocked_road_id"
        ],
        "trigger_blocked_edge_key": selection[
            "trigger_blocked_edge_key"
        ],
        "road_graph_directed": bool(
            flood_graph.is_directed()
        ),
        "directed_routing_ready": int(
            float(
                flood_graph.graph.get(
                    "directed_routing_ready",
                    0,
                )
            )
        ),
        "blocked_edges_filtered": int(
            blocked_edges
        ),
        "safe_route_node_count": int(
            len(
                safe_nodes
            )
        ),
        "safe_route_edge_count": int(
            len(
                safe[
                    "records"
                ]
            )
        ),
        "safe_route_length_m": float(
            safe[
                "length_m"
            ]
        ),
        "safe_route_routing_cost": float(
            safe[
                "routing_cost"
            ]
        ),
        "safe_route_max_flood_depth_m": float(
            safe[
                "max_depth_m"
            ]
        ),
        "safe_route_max_flood_risk_rank": int(
            safe[
                "max_risk_rank"
            ]
        ),
        "safe_route_risk_classes": (
            unique_risks
        ),
        "safe_route_blocked_edge_count": 0,
        "weight_field": WEIGHT_FIELD,
        "baseline_shortest_distance_route": baseline,
        "routing_model_note": (
            "Prototype undirected flood-aware routing; "
            "OSM one-way directionality is not yet enforced."
        ),
    }

    ROUTE_SUMMARY_OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Start node              : "
        f"{start}"
    )

    print(
        f"End node                : "
        f"{end}"
    )

    print(
        f"Selection method        : "
        f"{selection['selection_method']}"
    )

    if selection[
        "trigger_blocked_road_id"
    ]:
        print(
            f"Blocked road triggering demo: "
            f"{selection['trigger_blocked_road_id']}"
        )

    print(
        f"Safe route edges        : "
        f"{len(safe['records'])}"
    )

    print(
        f"Safe route length       : "
        f"{safe['length_m']:.2f} m"
    )

    print(
        f"Safe routing cost       : "
        f"{safe['routing_cost']:.2f}"
    )

    print(
        f"Max route flood depth   : "
        f"{safe['max_depth_m']:.3f} m"
    )

    print(
        f"Route risk classes      : "
        f"{', '.join(unique_risks)}"
    )

    if baseline.get(
        "available",
        False,
    ):
        print(
            f"Baseline distance       : "
            f"{baseline['length_m']:.2f} m"
        )

        print(
            "Baseline blocked edges  : "
            f"{baseline['blocked_edge_count_under_flood_policy']}"
        )

    print(
        "\\nSaved:"
    )

    print(
        ROUTE_GEOJSON_OUTPUT
    )

    print(
        ROUTE_SUMMARY_OUTPUT
    )

    print(
        "\\nIMPORTANT:"
    )

    print(
        "Blocked edges are physically filtered before "
        "shortest-path calculation."
    )

    print(
        "Current road graph is undirected; OSM one-way "
        "directionality is not yet enforced."
    )

    print(
        "\\nSafe route generation completed."
    )


if __name__ == "__main__":
    main()
