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
    """
    Return the parallel edge between u and v with the
    lowest value for weight_field.
    """
    edges = graph.get_edge_data(u, v)

    if edges is None:
        raise ValueError(f"No edge data found for {u} -> {v}")

    valid_edges = []

    for key, data in edges.items():
        try:
            weight = float(data[weight_field])
            valid_edges.append((weight, key, data))
        except (KeyError, TypeError, ValueError):
            continue

    if not valid_edges:
        raise ValueError(
            f"No valid '{weight_field}' value for edge {u} -> {v}"
        )

    valid_edges.sort(key=lambda item: item[0])

    _, key, data = valid_edges[0]

    return key, data


def path_metric(graph, path, metric_field, edge_weight_field):
    """
    Sum metric_field along a path while selecting the same
    parallel edge that corresponds to edge_weight_field.
    """
    total = 0.0

    for u, v in zip(path[:-1], path[1:]):
        _, data = best_edge_data(
            graph,
            u,
            v,
            edge_weight_field,
        )

        total += float(data[metric_field])

    return total


def count_path_risks(graph, path, edge_weight_field):
    """
    Count risk classes and maximum flood depth along a route.
    """
    counts = {
        "safe": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
    }

    max_depth = 0.0
    blocked_edges = 0

    for u, v in zip(path[:-1], path[1:]):
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
            blocked_edges += 1

        max_depth = max(
            max_depth,
            depth,
        )

    return counts, max_depth, blocked_edges


def path_geometry(graph, path):
    """
    Build a simple LineString from graph node coordinates.
    This is sufficient for the MVP route-comparison layer.
    """
    coordinates = []

    for node in path:
        data = graph.nodes[node]

        coordinates.append(
            (
                float(data["x"]),
                float(data["y"]),
            )
        )

    return LineString(coordinates)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Flood-aware graph not found: {GRAPH_FILE}"
        )

    graph = nx.read_graphml(
        GRAPH_FILE
    )

    print("Flood-safe routing")
    print()
    print(f"Interval: {INTERVAL}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    if graph.number_of_nodes() == 0:
        raise RuntimeError("Routing graph has no nodes")

    components = list(
        nx.connected_components(
            graph
        )
    )

    largest_nodes = max(
        components,
        key=len,
    )

    main_graph = graph.subgraph(
        largest_nodes
    ).copy()

    print(
        f"Largest original component nodes: "
        f"{main_graph.number_of_nodes()}"
    )

    safe_graph = main_graph.copy()

    blocked_to_remove = []

    for u, v, key, data in safe_graph.edges(
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
        except (TypeError, ValueError):
            passable = 1

        if passable == 0:
            blocked_to_remove.append(
                (u, v, key)
            )

    safe_graph.remove_edges_from(
        blocked_to_remove
    )

    print(
        f"Blocked edges removed: "
        f"{len(blocked_to_remove)}"
    )

    safe_components = [
        component
        for component in nx.connected_components(
            safe_graph
        )
        if len(component) >= 2
    ]

    if not safe_components:
        raise RuntimeError(
            "No connected flood-safe road component remains."
        )

    largest_safe_nodes = max(
        safe_components,
        key=len,
    )

    safe_graph = safe_graph.subgraph(
        largest_safe_nodes
    ).copy()

    print(
        f"Largest flood-safe component nodes: "
        f"{safe_graph.number_of_nodes()}"
    )

    source = min(
        safe_graph.nodes,
        key=lambda node: float(
            safe_graph.nodes[node]["x"]
        ),
    )

    target = max(
        safe_graph.nodes,
        key=lambda node: float(
            safe_graph.nodes[node]["x"]
        ),
    )

    if source == target:
        raise RuntimeError(
            "Could not select two distinct demo route nodes."
        )

    print()
    print(f"Source node: {source}")
    print(f"Target node: {target}")

    if not nx.has_path(
        safe_graph,
        source,
        target,
    ):
        raise RuntimeError(
            "Unexpected error: selected flood-safe endpoints "
            "are not connected."
        )

    normal_path = nx.shortest_path(
        main_graph,
        source=source,
        target=target,
        weight="length_m",
    )

    normal_distance = path_metric(
        main_graph,
        normal_path,
        metric_field="length_m",
        edge_weight_field="length_m",
    )

    normal_risks, normal_max_depth, normal_blocked = (
        count_path_risks(
            main_graph,
            normal_path,
            edge_weight_field="length_m",
        )
    )

    safe_path = nx.shortest_path(
        safe_graph,
        source=source,
        target=target,
        weight="routing_cost",
    )

    safe_distance = path_metric(
        safe_graph,
        safe_path,
        metric_field="length_m",
        edge_weight_field="routing_cost",
    )

    safe_cost = path_metric(
        safe_graph,
        safe_path,
        metric_field="routing_cost",
        edge_weight_field="routing_cost",
    )

    safe_risks, safe_max_depth, safe_blocked = (
        count_path_risks(
            safe_graph,
            safe_path,
            edge_weight_field="routing_cost",
        )
    )

    print()
    print("NORMAL ROUTE")
    print(f"  Nodes: {len(normal_path)}")
    print(
        f"  Distance: "
        f"{normal_distance:.2f} m"
    )
    print(
        f"  Max flood depth: "
        f"{normal_max_depth:.3f} m"
    )
    print(
        f"  Risk edges: "
        f"{normal_risks}"
    )
    print(
        f"  Blocked edges crossed: "
        f"{normal_blocked}"
    )

    print()
    print("FLOOD-SAFE ROUTE")
    print(f"  Nodes: {len(safe_path)}")
    print(
        f"  Distance: "
        f"{safe_distance:.2f} m"
    )
    print(
        f"  Routing cost: "
        f"{safe_cost:.2f}"
    )
    print(
        f"  Max flood depth: "
        f"{safe_max_depth:.3f} m"
    )
    print(
        f"  Risk edges: "
        f"{safe_risks}"
    )
    print(
        f"  Blocked edges crossed: "
        f"{safe_blocked}"
    )

    extra_distance = (
        safe_distance
        - normal_distance
    )

    print()
    print(
        f"Extra distance for safer route: "
        f"{extra_distance:.2f} m"
    )

    if normal_path == safe_path:
        print(
            "Route comparison note: normal and flood-safe "
            "routes are identical for these automatic endpoints."
        )

    normal_geom = path_geometry(
        main_graph,
        normal_path,
    )

    safe_geom = path_geometry(
        safe_graph,
        safe_path,
    )

    routes = gpd.GeoDataFrame(
        [
            {
                "route_type": "normal",
                "interval_id": INTERVAL,
                "distance_m": normal_distance,
                "routing_cost": normal_distance,
                "max_depth_m": normal_max_depth,
                "blocked_edges": normal_blocked,
                "high_risk_edges": normal_risks["high"],
                "geometry": normal_geom,
            },
            {
                "route_type": "flood_safe",
                "interval_id": INTERVAL,
                "distance_m": safe_distance,
                "routing_cost": safe_cost,
                "max_depth_m": safe_max_depth,
                "blocked_edges": safe_blocked,
                "high_risk_edges": safe_risks["high"],
                "geometry": safe_geom,
            },
        ],
        geometry="geometry",
        crs="EPSG:32643",
    )

    output_file = (
        OUTPUT_DIR
        / f"route_comparison_{INTERVAL}.geojson"
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
        "Flood-safe route generated successfully."
    )


if __name__ == "__main__":
    main()
