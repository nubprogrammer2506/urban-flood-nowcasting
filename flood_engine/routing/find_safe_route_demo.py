from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERVAL = "090_120"

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "routing_graphs"
    / f"road_graph_{INTERVAL}.graphml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "routes"
)


def best_edge_data(graph, u, v, weight_field):
    edges = graph.get_edge_data(u, v)

    if edges is None:
        raise ValueError(f"No edge data for {u} -> {v}")

    candidates = []

    for key, data in edges.items():
        try:
            weight = float(data[weight_field])
        except (KeyError, TypeError, ValueError):
            continue

        candidates.append((weight, key, data))

    if not candidates:
        raise ValueError(
            f"No valid '{weight_field}' edge for {u} -> {v}"
        )

    candidates.sort(key=lambda item: item[0])

    _, key, data = candidates[0]

    return key, data


def path_metric(
    graph,
    path,
    metric_field,
    edge_weight_field,
):
    total = 0.0

    for u, v in zip(
        path[:-1],
        path[1:],
    ):
        _, data = best_edge_data(
            graph,
            u,
            v,
            edge_weight_field,
        )

        total += float(
            data[metric_field]
        )

    return total


def path_risk_stats(
    graph,
    path,
    edge_weight_field,
):
    counts = {
        "safe": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
    }

    maximum_depth = 0.0
    blocked = 0

    for u, v in zip(
        path[:-1],
        path[1:],
    ):
        _, data = best_edge_data(
            graph,
            u,
            v,
            edge_weight_field,
        )

        risk = str(
            data.get(
                "flood_risk",
                "safe",
            )
        )

        try:
            depth = float(
                data.get(
                    "flood_depth_m",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            depth = 0.0

        try:
            passable = int(
                float(
                    data.get(
                        "is_passable",
                        1,
                    )
                )
            )
        except (TypeError, ValueError):
            passable = 1

        if risk in counts:
            counts[risk] += 1

        if passable == 0:
            blocked += 1

        maximum_depth = max(
            maximum_depth,
            depth,
        )

    return (
        counts,
        maximum_depth,
        blocked,
    )


def path_geometry(
    graph,
    path,
):
    coordinates = []

    for node in path:
        data = graph.nodes[node]

        coordinates.append(
            (
                float(data["x"]),
                float(data["y"]),
            )
        )

    return LineString(
        coordinates
    )


def evaluate_pair(
    main_graph,
    safe_graph,
    source,
    target,
):
    if (
        source not in safe_graph
        or target not in safe_graph
        or source == target
    ):
        return None

    if not nx.has_path(
        safe_graph,
        source,
        target,
    ):
        return None

    normal_path = nx.shortest_path(
        main_graph,
        source=source,
        target=target,
        weight="length_m",
    )

    safe_path = nx.shortest_path(
        safe_graph,
        source=source,
        target=target,
        weight="routing_cost",
    )

    normal_distance = path_metric(
        main_graph,
        normal_path,
        "length_m",
        "length_m",
    )

    safe_distance = path_metric(
        safe_graph,
        safe_path,
        "length_m",
        "routing_cost",
    )

    safe_cost = path_metric(
        safe_graph,
        safe_path,
        "routing_cost",
        "routing_cost",
    )

    (
        normal_risks,
        normal_depth,
        normal_blocked,
    ) = path_risk_stats(
        main_graph,
        normal_path,
        "length_m",
    )

    (
        safe_risks,
        safe_depth,
        safe_blocked,
    ) = path_risk_stats(
        safe_graph,
        safe_path,
        "routing_cost",
    )

    normal_penalty = (
        normal_blocked * 10000
        + normal_risks["high"] * 1000
        + normal_risks["medium"] * 100
        + normal_risks["low"] * 10
        + normal_depth * 100
    )

    safe_penalty = (
        safe_blocked * 10000
        + safe_risks["high"] * 1000
        + safe_risks["medium"] * 100
        + safe_risks["low"] * 10
        + safe_depth * 100
    )

    improvement = (
        normal_penalty
        - safe_penalty
    )

    return {
        "source": source,
        "target": target,
        "normal_path": normal_path,
        "safe_path": safe_path,
        "normal_distance": normal_distance,
        "safe_distance": safe_distance,
        "safe_cost": safe_cost,
        "normal_risks": normal_risks,
        "safe_risks": safe_risks,
        "normal_depth": normal_depth,
        "safe_depth": safe_depth,
        "normal_blocked": normal_blocked,
        "safe_blocked": safe_blocked,
        "improvement": improvement,
        "different": normal_path != safe_path,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Flood-aware graph not found: "
            f"{GRAPH_FILE}"
        )

    graph = nx.read_graphml(
        GRAPH_FILE
    )

    print(
        "Flood-safe demo-route search"
    )
    print()

    print(
        f"Interval: {INTERVAL}"
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges()}"
    )

    # -------------------------------------
    # Original largest component
    # -------------------------------------
    original_components = list(
        nx.connected_components(
            graph
        )
    )

    largest_original = max(
        original_components,
        key=len,
    )

    main_graph = graph.subgraph(
        largest_original
    ).copy()

    # -------------------------------------
    # Build flood-safe graph
    # -------------------------------------
    safe_graph = (
        main_graph.copy()
    )

    blocked_edges = []

    for (
        u,
        v,
        key,
        data,
    ) in safe_graph.edges(
        keys=True,
        data=True,
    ):
        try:
            passable = int(
                float(
                    data.get(
                        "is_passable",
                        1,
                    )
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            passable = 1

        if passable == 0:
            blocked_edges.append(
                (
                    u,
                    v,
                    key,
                )
            )

    safe_graph.remove_edges_from(
        blocked_edges
    )

    # -------------------------------------
    # Largest remaining safe component
    # -------------------------------------
    safe_components = [
        component
        for component
        in nx.connected_components(
            safe_graph
        )
        if len(component) >= 2
    ]

    if not safe_components:
        raise RuntimeError(
            "No connected flood-safe "
            "component remains."
        )

    largest_safe = max(
        safe_components,
        key=len,
    )

    safe_graph = (
        safe_graph.subgraph(
            largest_safe
        ).copy()
    )

    print(
        f"Largest original component: "
        f"{main_graph.number_of_nodes()} "
        f"nodes"
    )

    print(
        f"Blocked edges removed: "
        f"{len(blocked_edges)}"
    )

    print(
        f"Largest safe component: "
        f"{safe_graph.number_of_nodes()} "
        f"nodes"
    )

    candidates = []

    # =====================================
    # TEST 1
    # Try endpoints of blocked roads.
    #
    # Ideal demo:
    # normal route uses blocked edge,
    # safe route detours around it.
    # =====================================
    for (
        u,
        v,
        key,
        data,
    ) in main_graph.edges(
        keys=True,
        data=True,
    ):
        try:
            passable = int(
                float(
                    data.get(
                        "is_passable",
                        1,
                    )
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            passable = 1

        if passable != 0:
            continue

        result = evaluate_pair(
            main_graph,
            safe_graph,
            u,
            v,
        )

        if (
            result is not None
            and result["different"]
            and result["improvement"] > 0
        ):
            candidates.append(
                result
            )

    # =====================================
    # TEST 2
    # Try medium-risk edge endpoints if
    # blocked-edge detours do not exist.
    # =====================================
    if not candidates:
        for (
            u,
            v,
            key,
            data,
        ) in main_graph.edges(
            keys=True,
            data=True,
        ):
            risk = str(
                data.get(
                    "flood_risk",
                    "safe",
                )
            )

            if risk != "medium":
                continue

            result = evaluate_pair(
                main_graph,
                safe_graph,
                u,
                v,
            )

            if (
                result is not None
                and result["different"]
                and result["improvement"] > 0
            ):
                candidates.append(
                    result
                )

    # =====================================
    # TEST 3
    # Broader search across graph nodes.
    # =====================================
    if not candidates:
        ordered_nodes = sorted(
            safe_graph.nodes,
            key=lambda node: (
                float(
                    safe_graph.nodes[
                        node
                    ]["x"]
                ),
                float(
                    safe_graph.nodes[
                        node
                    ]["y"]
                ),
            ),
        )

        if len(ordered_nodes) > 50:
            step = max(
                1,
                len(ordered_nodes) // 50,
            )

            sample_nodes = (
                ordered_nodes[::step]
            )

        else:
            sample_nodes = (
                ordered_nodes
            )

        for i, source in enumerate(
            sample_nodes
        ):
            for target in (
                sample_nodes[i + 1:]
            ):
                result = evaluate_pair(
                    main_graph,
                    safe_graph,
                    source,
                    target,
                )

                if (
                    result is not None
                    and result["different"]
                    and result["improvement"] > 0
                ):
                    candidates.append(
                        result
                    )

    # =====================================
    # Pick strongest contrast
    # =====================================
    if candidates:
        best = max(
            candidates,
            key=lambda item: (
                item["improvement"],
                item["normal_blocked"],
                (
                    item["normal_depth"]
                    - item["safe_depth"]
                ),
            ),
        )

        selection_note = (
            "Selected pair where "
            "flood-aware routing improves "
            "on the normal shortest route."
        )

    else:
        # ---------------------------------
        # Valid fallback
        # ---------------------------------
        source = min(
            safe_graph.nodes,
            key=lambda node: float(
                safe_graph.nodes[
                    node
                ]["x"]
            ),
        )

        target = max(
            safe_graph.nodes,
            key=lambda node: float(
                safe_graph.nodes[
                    node
                ]["x"]
            ),
        )

        best = evaluate_pair(
            main_graph,
            safe_graph,
            source,
            target,
        )

        if best is None:
            raise RuntimeError(
                "Could not create "
                "a fallback route."
            )

        selection_note = (
            "No stronger contrasting pair "
            "was found; using reachable "
            "west-east endpoints."
        )

    source = best["source"]
    target = best["target"]

    print()
    print(
        selection_note
    )

    print(
        f"Source node: {source}"
    )

    print(
        f"Target node: {target}"
    )

    # =====================================
    # Print comparison
    # =====================================
    print()
    print("NORMAL ROUTE")

    print(
        f"  Nodes: "
        f"{len(best['normal_path'])}"
    )

    print(
        f"  Distance: "
        f"{best['normal_distance']:.2f} m"
    )

    print(
        f"  Max flood depth: "
        f"{best['normal_depth']:.3f} m"
    )

    print(
        f"  Risk edges: "
        f"{best['normal_risks']}"
    )

    print(
        f"  Blocked edges crossed: "
        f"{best['normal_blocked']}"
    )

    print()
    print("FLOOD-SAFE ROUTE")

    print(
        f"  Nodes: "
        f"{len(best['safe_path'])}"
    )

    print(
        f"  Distance: "
        f"{best['safe_distance']:.2f} m"
    )

    print(
        f"  Routing cost: "
        f"{best['safe_cost']:.2f}"
    )

    print(
        f"  Max flood depth: "
        f"{best['safe_depth']:.3f} m"
    )

    print(
        f"  Risk edges: "
        f"{best['safe_risks']}"
    )

    print(
        f"  Blocked edges crossed: "
        f"{best['safe_blocked']}"
    )

    print()

    extra_distance = (
        best["safe_distance"]
        - best["normal_distance"]
    )

    print(
        "Extra distance for safer route: "
        f"{extra_distance:.2f} m"
    )

    # =====================================
    # Save for QGIS
    # =====================================
    normal_geom = path_geometry(
        main_graph,
        best["normal_path"],
    )

    safe_geom = path_geometry(
        safe_graph,
        best["safe_path"],
    )

    routes = gpd.GeoDataFrame(
        [
            {
                "route_type": "normal",
                "interval_id": INTERVAL,
                "source_node": source,
                "target_node": target,
                "distance_m": (
                    best[
                        "normal_distance"
                    ]
                ),
                "max_depth_m": (
                    best[
                        "normal_depth"
                    ]
                ),
                "blocked_edges": (
                    best[
                        "normal_blocked"
                    ]
                ),
                "high_risk_edges": (
                    best[
                        "normal_risks"
                    ]["high"]
                ),
                "medium_risk_edges": (
                    best[
                        "normal_risks"
                    ]["medium"]
                ),
                "geometry": (
                    normal_geom
                ),
            },
            {
                "route_type": (
                    "flood_safe"
                ),
                "interval_id": INTERVAL,
                "source_node": source,
                "target_node": target,
                "distance_m": (
                    best[
                        "safe_distance"
                    ]
                ),
                "max_depth_m": (
                    best[
                        "safe_depth"
                    ]
                ),
                "blocked_edges": (
                    best[
                        "safe_blocked"
                    ]
                ),
                "high_risk_edges": (
                    best[
                        "safe_risks"
                    ]["high"]
                ),
                "medium_risk_edges": (
                    best[
                        "safe_risks"
                    ]["medium"]
                ),
                "geometry": (
                    safe_geom
                ),
            },
        ],
        geometry="geometry",
        crs="EPSG:32643",
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"route_comparison_"
            f"{INTERVAL}.geojson"
        )
    )

    routes.to_file(
        output_file,
        driver="GeoJSON",
    )

    print()
    print(
        f"Saved: {output_file}"
    )

    print()
    print(
        "Flood-safe demo route "
        "generated successfully."
    )


if __name__ == "__main__":
    main()