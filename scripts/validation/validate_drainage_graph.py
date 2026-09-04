from pathlib import Path

import geopandas as gpd
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRAINAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage.geojson"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_graph.graphml"
)


REQUIRED_EDGE_ATTRIBUTES = [
    "drain_id",
    "source_drain_id",
    "source_type",
    "is_inferred",
    "acc_cells",
    "upstream_area_m2",
    "length_m",
    "geometry_wkt",
    "hydraulic_capacity_known",
]


def convert_graph_types(graph):
    """
    GraphML may load numeric attributes as strings.
    Convert important values back to numeric types.
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

        if "acc_cells" in data:
            data["acc_cells"] = int(
                data["acc_cells"]
            )

        if "upstream_area_m2" in data:
            data["upstream_area_m2"] = float(
                data["upstream_area_m2"]
            )

        if "is_inferred" in data:
            data["is_inferred"] = int(
                data["is_inferred"]
            )

        if "hydraulic_capacity_known" in data:
            data["hydraulic_capacity_known"] = int(
                data["hydraulic_capacity_known"]
            )

    return graph


def find_directed_test_path(graph):
    """
    Find a meaningful downstream route.

    Try every source node and retain the longest
    reachable weighted path.
    """

    sources = [
        node
        for node in graph.nodes
        if (
            graph.in_degree(node) == 0
            and graph.out_degree(node) > 0
        )
    ]

    best_start = None
    best_end = None
    best_distance = -1.0

    for source in sources:

        distances = (
            nx.single_source_dijkstra_path_length(
                graph,
                source,
                weight="length_m",
            )
        )

        if not distances:
            continue

        end_node = max(
            distances,
            key=distances.get,
        )

        distance = distances[end_node]

        if distance > best_distance:
            best_start = source
            best_end = end_node
            best_distance = distance

    if (
        best_start is None
        or best_end is None
        or best_start == best_end
    ):
        return None

    route = nx.shortest_path(
        graph,
        best_start,
        best_end,
        weight="length_m",
    )

    return (
        best_start,
        best_end,
        route,
        best_distance,
    )


def main():

    print(
        "\n=== DRAINAGE GRAPH VALIDATION ===\n"
    )

    errors = []

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    if not GRAPH_PATH.exists():
        print("ERROR: Drainage graph not found:")
        print(GRAPH_PATH)
        return

    if not DRAINAGE_PATH.exists():
        print("ERROR: drainage.geojson not found:")
        print(DRAINAGE_PATH)
        return

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    graph = nx.read_graphml(
        GRAPH_PATH,
        force_multigraph=True,
    )

    graph = convert_graph_types(
        graph
    )

    drainage = gpd.read_file(
        DRAINAGE_PATH
    )

    print(
        f"Graph type           : "
        f"{type(graph).__name__}"
    )

    print(
        f"Nodes                : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges                : "
        f"{graph.number_of_edges()}"
    )

    # -----------------------------------------------------
    # Directed graph check
    # -----------------------------------------------------

    print(
        f"Directed             : "
        f"{graph.is_directed()}"
    )

    if not graph.is_directed():
        errors.append(
            "drainage graph is not directed"
        )

    # -----------------------------------------------------
    # Node validation
    # -----------------------------------------------------

    missing_coordinates = sum(
        1
        for _, data in graph.nodes(
            data=True
        )
        if (
            "x" not in data
            or "y" not in data
        )
    )

    print(
        f"Nodes missing x/y    : "
        f"{missing_coordinates}"
    )

    if missing_coordinates:
        errors.append(
            "nodes missing coordinates"
        )

    # -----------------------------------------------------
    # Edge validation
    # -----------------------------------------------------

    missing_attributes = 0
    invalid_lengths = 0
    invalid_accumulation = 0
    invalid_areas = 0
    incorrect_provenance = 0
    unexpected_capacity = 0

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        missing = [
            attribute
            for attribute
            in REQUIRED_EDGE_ATTRIBUTES
            if attribute not in data
        ]

        if missing:
            missing_attributes += 1

        if (
            "length_m" not in data
            or data["length_m"] <= 0
        ):
            invalid_lengths += 1

        if (
            "acc_cells" not in data
            or data["acc_cells"] <= 0
        ):
            invalid_accumulation += 1

        if (
            "upstream_area_m2" not in data
            or data["upstream_area_m2"] <= 0
        ):
            invalid_areas += 1

        if (
            data.get("source_type")
            != "dem_inferred_surface_flow"
            or data.get("is_inferred") != 1
        ):
            incorrect_provenance += 1

        # We must not invent pipe/channel capacity.
        if data.get(
            "hydraulic_capacity_known"
        ) != 0:
            unexpected_capacity += 1

    print("\nEdge checks")

    print(
        f"Edges missing attrs  : "
        f"{missing_attributes}"
    )

    print(
        f"Invalid lengths      : "
        f"{invalid_lengths}"
    )

    print(
        f"Invalid accumulation : "
        f"{invalid_accumulation}"
    )

    print(
        f"Invalid upstream area: "
        f"{invalid_areas}"
    )

    print(
        f"Incorrect provenance : "
        f"{incorrect_provenance}"
    )

    print(
        f"Unexpected capacity  : "
        f"{unexpected_capacity}"
    )

    if missing_attributes:
        errors.append(
            "edges missing required attributes"
        )

    if invalid_lengths:
        errors.append(
            "invalid edge lengths"
        )

    if invalid_accumulation:
        errors.append(
            "invalid accumulation values"
        )

    if invalid_areas:
        errors.append(
            "invalid upstream areas"
        )

    if incorrect_provenance:
        errors.append(
            "incorrect inferred-data provenance"
        )

    if unexpected_capacity:
        errors.append(
            "invented hydraulic capacity detected"
        )

    # -----------------------------------------------------
    # Connectivity
    # -----------------------------------------------------

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    components.sort(
        key=len,
        reverse=True,
    )

    component_sizes = [
        len(component)
        for component in components
    ]

    largest_component_size = (
        component_sizes[0]
        if component_sizes
        else 0
    )

    largest_fraction = (
        largest_component_size
        / graph.number_of_nodes()
        * 100
        if graph.number_of_nodes()
        else 0
    )

    source_nodes = [
        node
        for node in graph.nodes
        if (
            graph.in_degree(node) == 0
            and graph.out_degree(node) > 0
        )
    ]

    sink_nodes = [
        node
        for node in graph.nodes
        if (
            graph.out_degree(node) == 0
            and graph.in_degree(node) > 0
        )
    ]

    isolated_nodes = list(
        nx.isolates(graph)
    )

    print("\nConnectivity")

    print(
        f"Weak components      : "
        f"{len(components)}"
    )

    print(
        f"Component sizes      : "
        f"{component_sizes}"
    )

    print(
        f"Largest component    : "
        f"{largest_component_size} "
        f"nodes ({largest_fraction:.2f}%)"
    )

    print(
        f"Source nodes         : "
        f"{len(source_nodes)}"
    )

    print(
        f"Sink nodes           : "
        f"{len(sink_nodes)}"
    )

    print(
        f"Isolated nodes       : "
        f"{len(isolated_nodes)}"
    )

    if isolated_nodes:
        errors.append(
            "isolated drainage nodes"
        )

    if not source_nodes:
        errors.append(
            "no drainage source nodes"
        )

    if not sink_nodes:
        errors.append(
            "no drainage sink nodes"
        )

    # -----------------------------------------------------
    # DAG / loop check
    # -----------------------------------------------------

    is_dag = (
        nx.is_directed_acyclic_graph(
            graph
        )
    )

    print("\nTopology")

    print(
        f"Directed acyclic     : "
        f"{is_dag}"
    )

    if not is_dag:
        errors.append(
            "directed drainage cycles detected"
        )

    # -----------------------------------------------------
    # Forest relationship
    # -----------------------------------------------------

    expected_edges = (
        graph.number_of_nodes()
        - len(components)
    )

    forest_relationship = (
        graph.number_of_edges()
        == expected_edges
    )

    print(
        f"Forest edge relation : "
        f"{forest_relationship}"
    )

    print(
        f"Expected N-C edges   : "
        f"{expected_edges}"
    )

    # Not automatically an error in all future datasets,
    # but for this D8-derived graph it is useful evidence
    # of a clean tree/forest topology.

    # -----------------------------------------------------
    # Length consistency
    # -----------------------------------------------------

    gis_length = float(
        drainage["length_m"].sum()
    )

    graph_length = sum(
        data["length_m"]
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    )

    length_difference = abs(
        gis_length
        - graph_length
    )

    print("\nLength consistency")

    print(
        f"GIS drain length     : "
        f"{gis_length / 1000:.2f} km"
    )

    print(
        f"Graph drain length   : "
        f"{graph_length / 1000:.2f} km"
    )

    print(
        f"Length difference    : "
        f"{length_difference:.2f} m"
    )

    if length_difference > 1.0:
        errors.append(
            "GIS/graph length mismatch"
        )

    # -----------------------------------------------------
    # Directed flow-path sanity test
    # -----------------------------------------------------

    print(
        "\nDirected-flow sanity test"
    )

    route_result = (
        find_directed_test_path(
            graph
        )
    )

    if route_result is None:

        errors.append(
            "no meaningful downstream path found"
        )

        print(
            "ERROR: No directed downstream "
            "test path found."
        )

    else:

        (
            start_node,
            end_node,
            route,
            route_length,
        ) = route_result

        print(
            f"Start/source node    : "
            f"{start_node}"
        )

        print(
            f"End/downstream node  : "
            f"{end_node}"
        )

        print(
            f"Path nodes           : "
            f"{len(route)}"
        )

        print(
            f"Flow-path distance   : "
            f"{route_length / 1000:.2f} km"
        )

        if len(route) < 2:
            errors.append(
                "directed path too short"
            )

        if route_length <= 0:
            errors.append(
                "invalid directed path length"
            )

    # -----------------------------------------------------
    # Provenance statement
    # -----------------------------------------------------

    print("\nIMPORTANT:")

    print(
        "Drainage graph provenance: "
        "DEM-INFERRED SURFACE FLOW."
    )

    print(
        "It is NOT surveyed municipal "
        "stormwater infrastructure."
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    if errors:

        print(
            "DRAINAGE GRAPH VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

    else:

        print(
            "DRAINAGE GRAPH VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()