from pathlib import Path

import geopandas as gpd
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "roads.geojson"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph.graphml"
)


REQUIRED_EDGE_ATTRIBUTES = [
    "road_id",
    "source_road_id",
    "osm_id",
    "road_class",
    "length_m",
    "geometry_wkt",
    "flood_depth_m",
    "flood_risk",
    "is_passable",
    "routing_cost",
]


def convert_graph_types(graph):
    """
    GraphML loads many values as strings.

    Convert numeric routing attributes back into
    their expected Python types.
    """

    for _, data in graph.nodes(data=True):

        if "x" in data:
            data["x"] = float(data["x"])

        if "y" in data:
            data["y"] = float(data["y"])

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        if "length_m" in data:
            data["length_m"] = float(
                data["length_m"]
            )

        if "routing_cost" in data:
            data["routing_cost"] = float(
                data["routing_cost"]
            )

        if "flood_depth_m" in data:
            data["flood_depth_m"] = float(
                data["flood_depth_m"]
            )

        if "is_passable" in data:
            data["is_passable"] = int(
                data["is_passable"]
            )

        if "oneway" in data:
            data["oneway"] = int(
                data["oneway"]
            )

    return graph


def find_test_route(graph, component_nodes):
    """
    Find a meaningful route inside the largest component.

    This uses two weighted searches to select two distant
    nodes and then calculates a shortest-distance path.
    """

    component = graph.subgraph(
        component_nodes
    ).copy()

    start = next(iter(component.nodes))

    distances = nx.single_source_dijkstra_path_length(
        component,
        start,
        weight="length_m",
    )

    node_a = max(
        distances,
        key=distances.get,
    )

    distances_from_a = (
        nx.single_source_dijkstra_path_length(
            component,
            node_a,
            weight="length_m",
        )
    )

    node_b = max(
        distances_from_a,
        key=distances_from_a.get,
    )

    route = nx.shortest_path(
        component,
        node_a,
        node_b,
        weight="length_m",
    )

    route_length = nx.shortest_path_length(
        component,
        node_a,
        node_b,
        weight="length_m",
    )

    return node_a, node_b, route, route_length


def main():

    print(
        "\n=== RAJENDRA NAGAR ROAD GRAPH VALIDATION ===\n"
    )

    errors = []

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    if not GRAPH_PATH.exists():
        print("ERROR: GraphML file not found:")
        print(GRAPH_PATH)
        return

    if not ROADS_PATH.exists():
        print("ERROR: Canonical roads file not found:")
        print(ROADS_PATH)
        return

    # -----------------------------------------------------
    # Load graph
    # -----------------------------------------------------

    graph = nx.read_graphml(
        GRAPH_PATH,
        force_multigraph=True,
    )

    graph = convert_graph_types(graph)

    roads = gpd.read_file(ROADS_PATH)

    print(f"Graph type           : {type(graph).__name__}")
    print(f"Nodes                : {graph.number_of_nodes()}")
    print(f"Edges                : {graph.number_of_edges()}")

    # -----------------------------------------------------
    # Basic graph checks
    # -----------------------------------------------------

    if graph.number_of_nodes() == 0:
        errors.append("graph contains no nodes")

    if graph.number_of_edges() == 0:
        errors.append("graph contains no edges")

    # -----------------------------------------------------
    # Node attributes
    # -----------------------------------------------------

    missing_node_coordinates = sum(
        1
        for _, data in graph.nodes(data=True)
        if "x" not in data or "y" not in data
    )

    print(
        f"Nodes missing x/y    : "
        f"{missing_node_coordinates}"
    )

    if missing_node_coordinates:
        errors.append(
            "nodes missing coordinates"
        )

    # -----------------------------------------------------
    # Edge attributes
    # -----------------------------------------------------

    missing_attribute_edges = 0
    invalid_length_edges = 0

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        missing = [
            attribute
            for attribute in REQUIRED_EDGE_ATTRIBUTES
            if attribute not in data
        ]

        if missing:
            missing_attribute_edges += 1

        if (
            "length_m" not in data
            or data["length_m"] <= 0
        ):
            invalid_length_edges += 1

    print(
        f"Edges missing attrs  : "
        f"{missing_attribute_edges}"
    )

    print(
        f"Invalid edge lengths : "
        f"{invalid_length_edges}"
    )

    if missing_attribute_edges:
        errors.append(
            "edges missing required attributes"
        )

    if invalid_length_edges:
        errors.append(
            "invalid edge lengths"
        )

    # -----------------------------------------------------
    # Connectivity
    # -----------------------------------------------------

    components = list(
        nx.connected_components(graph)
    )

    components.sort(
        key=len,
        reverse=True,
    )

    component_sizes = [
        len(component)
        for component in components
    ]

    largest_component = components[0]

    largest_fraction = (
        len(largest_component)
        / graph.number_of_nodes()
        * 100
    )

    isolated_nodes = list(
        nx.isolates(graph)
    )

    print("\nConnectivity")
    print(
        f"Connected components: {len(components)}"
    )

    print(
        f"Component sizes      : "
        f"{component_sizes}"
    )

    print(
        f"Largest component    : "
        f"{len(largest_component)} nodes "
        f"({largest_fraction:.2f}%)"
    )

    print(
        f"Isolated nodes       : "
        f"{len(isolated_nodes)}"
    )

    if isolated_nodes:
        errors.append("isolated graph nodes")

    # Multiple components are reported but not treated
    # as automatic failure because AOI clipping can
    # legitimately create disconnected road fragments.

    # -----------------------------------------------------
    # Compare GIS and graph lengths
    # -----------------------------------------------------

    gis_length = float(
        roads["length_m"].sum()
    )

    graph_length = sum(
        data["length_m"]
        for _, _, _, data in graph.edges(
            keys=True,
            data=True,
        )
    )

    difference = abs(
        gis_length - graph_length
    )

    print("\nLength consistency")

    print(
        f"GIS road length      : "
        f"{gis_length / 1000:.2f} km"
    )

    print(
        f"Graph road length    : "
        f"{graph_length / 1000:.2f} km"
    )

    print(
        f"Length difference    : "
        f"{difference:.2f} m"
    )

    # 1 metre tolerance is enough here.
    if difference > 1.0:
        errors.append(
            "GIS/graph length mismatch"
        )

    # -----------------------------------------------------
    # Flood integration placeholders
    # -----------------------------------------------------

    incorrect_flood_defaults = 0

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        if (
            data["flood_depth_m"] != -1.0
            or data["flood_risk"] != "unknown"
        ):
            incorrect_flood_defaults += 1

    print("\nFlood integration readiness")

    print(
        f"Unexpected flood data: "
        f"{incorrect_flood_defaults}"
    )

    if incorrect_flood_defaults:
        errors.append(
            "unexpected flood placeholder values"
        )

    # -----------------------------------------------------
    # Shortest path sanity test
    # -----------------------------------------------------

    print("\nShortest-path sanity test")

    try:

        (
            start_node,
            end_node,
            route,
            route_length,
        ) = find_test_route(
            graph,
            largest_component,
        )

        print(
            f"Start node           : {start_node}"
        )

        print(
            f"End node             : {end_node}"
        )

        print(
            f"Route nodes          : {len(route)}"
        )

        print(
            f"Route distance       : "
            f"{route_length / 1000:.2f} km"
        )

        if len(route) < 2:
            errors.append(
                "shortest path contains too few nodes"
            )

        if route_length <= 0:
            errors.append(
                "shortest path has invalid length"
            )

    except nx.NetworkXNoPath:

        errors.append(
            "shortest path test failed"
        )

        print(
            "ERROR: No route found inside "
            "largest component."
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    if errors:

        print(
            "ROAD GRAPH VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

    else:

        print(
            "ROAD GRAPH VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()