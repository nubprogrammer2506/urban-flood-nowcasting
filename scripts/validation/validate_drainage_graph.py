from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np


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
    "start_elevation_m",
    "end_elevation_m",
    "elevation_drop_m",
    "slope_m_per_m",
    "slope_percent",
    "terrain_slope_deg",
]

TERRAIN_ATTRIBUTES = [
    "start_elevation_m",
    "end_elevation_m",
    "elevation_drop_m",
    "slope_m_per_m",
    "slope_percent",
    "terrain_slope_deg",
]

ELEVATION_TOLERANCE_M = 0.05
DROP_CONSISTENCY_TOLERANCE_M = 0.01
SLOPE_CONSISTENCY_TOLERANCE = 1e-4
PERCENT_CONSISTENCY_TOLERANCE = 0.01


def as_float(value):
    return float(value)


def as_int(value):
    return int(float(value))


def convert_graph_types(graph):
    """
    Convert GraphML string-like values back into numeric
    values used by validation and graph calculations.
    """

    for _, data in graph.nodes(data=True):
        if "x" in data:
            data["x"] = as_float(data["x"])

        if "y" in data:
            data["y"] = as_float(data["y"])

    integer_attributes = [
        "is_inferred",
        "acc_cells",
        "hydraulic_capacity_known",
    ]

    float_attributes = [
        "upstream_area_m2",
        "length_m",
        "start_elevation_m",
        "end_elevation_m",
        "elevation_drop_m",
        "slope_m_per_m",
        "slope_percent",
        "terrain_slope_deg",
    ]

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):
        for attribute in integer_attributes:
            if attribute in data:
                data[attribute] = as_int(
                    data[attribute]
                )

        for attribute in float_attributes:
            if attribute in data:
                data[attribute] = as_float(
                    data[attribute]
                )

    return graph


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Drainage GraphML not found: {GRAPH_PATH}"
        )

    graph = nx.read_graphml(
        GRAPH_PATH,
        force_multigraph=True,
    )

    graph = convert_graph_types(
        graph
    )

    return graph


def directed_flow_sanity_test(graph):
    """
    Find a deterministic source-to-downstream path for a
    basic directed-flow sanity check.

    Returns:
        (start, end, path, distance)
    """

    source_nodes = sorted(
        node
        for node in graph.nodes
        if (
            graph.in_degree(node) == 0
            and graph.out_degree(node) > 0
        )
    )

    if not source_nodes:
        return None

    best_result = None

    for source in source_nodes:
        reachable = nx.descendants(
            graph,
            source,
        )

        if not reachable:
            continue

        # Work on a simple DiGraph for path-length calculations.
        simple = nx.DiGraph()

        for u, v, data in graph.edges(data=True):
            length = float(
                data.get(
                    "length_m",
                    0.0,
                )
            )

            if simple.has_edge(u, v):
                if (
                    length
                    < simple[u][v]["length_m"]
                ):
                    simple[u][v][
                        "length_m"
                    ] = length
            else:
                simple.add_edge(
                    u,
                    v,
                    length_m=length,
                )

        lengths = nx.single_source_dijkstra_path_length(
            simple,
            source,
            weight="length_m",
        )

        candidates = [
            (
                distance,
                node,
            )
            for node, distance in lengths.items()
            if node != source
        ]

        if not candidates:
            continue

        distance, end = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        path = nx.shortest_path(
            simple,
            source,
            end,
            weight="length_m",
        )

        result = (
            distance,
            source,
            end,
            path,
        )

        if (
            best_result is None
            or result[0] > best_result[0]
        ):
            best_result = result

    if best_result is None:
        return None

    distance, source, end, path = (
        best_result
    )

    return (
        source,
        end,
        path,
        distance,
    )


def main():

    print(
        "\n=== DRAINAGE GRAPH VALIDATION ===\n"
    )

    errors = []

    # --------------------------------------------------
    # Load graph
    # --------------------------------------------------

    graph = load_graph()

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

    print(
        f"Directed             : "
        f"{graph.is_directed()}"
    )

    if not graph.is_directed():
        errors.append(
            "drainage graph is not directed"
        )

    # --------------------------------------------------
    # Node checks
    # --------------------------------------------------

    nodes_missing_xy = 0

    for _, data in graph.nodes(data=True):
        if (
            "x" not in data
            or "y" not in data
        ):
            nodes_missing_xy += 1
            continue

        if (
            not np.isfinite(
                float(data["x"])
            )
            or not np.isfinite(
                float(data["y"])
            )
        ):
            nodes_missing_xy += 1

    print(
        f"Nodes missing x/y    : "
        f"{nodes_missing_xy}"
    )

    if nodes_missing_xy:
        errors.append(
            "nodes missing valid x/y coordinates"
        )

    # --------------------------------------------------
    # Edge checks
    # --------------------------------------------------

    edges_missing_attributes = 0
    invalid_lengths = 0
    invalid_accumulation = 0
    invalid_upstream_area = 0
    incorrect_provenance = 0
    unexpected_capacity = 0

    invalid_terrain_attributes = 0
    non_finite_terrain_values = 0
    uphill_edges = 0
    inconsistent_drops = 0
    inconsistent_slopes = 0
    inconsistent_percent_slopes = 0
    negative_terrain_slopes = 0

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
            edges_missing_attributes += 1
            continue

        length_m = float(
            data["length_m"]
        )

        acc_cells = int(
            data["acc_cells"]
        )

        upstream_area_m2 = float(
            data["upstream_area_m2"]
        )

        if (
            not np.isfinite(length_m)
            or length_m <= 0
        ):
            invalid_lengths += 1

        if acc_cells <= 0:
            invalid_accumulation += 1

        if (
            not np.isfinite(
                upstream_area_m2
            )
            or upstream_area_m2 <= 0
        ):
            invalid_upstream_area += 1

        if (
            data.get("source_type")
            != "dem_inferred_surface_flow"
            or int(
                data.get(
                    "is_inferred",
                    0,
                )
            ) != 1
        ):
            incorrect_provenance += 1

        if int(
            data.get(
                "hydraulic_capacity_known",
                0,
            )
        ) != 0:
            unexpected_capacity += 1

        # ----------------------------------------------
        # Terrain-attribute checks
        # ----------------------------------------------

        terrain_values = {}

        terrain_missing = False

        for attribute in TERRAIN_ATTRIBUTES:
            if attribute not in data:
                terrain_missing = True
                break

            try:
                value = float(
                    data[attribute]
                )
            except (
                TypeError,
                ValueError,
            ):
                terrain_missing = True
                break

            terrain_values[
                attribute
            ] = value

            if not np.isfinite(value):
                non_finite_terrain_values += 1

        if terrain_missing:
            invalid_terrain_attributes += 1
            continue

        start_elevation = terrain_values[
            "start_elevation_m"
        ]

        end_elevation = terrain_values[
            "end_elevation_m"
        ]

        elevation_drop = terrain_values[
            "elevation_drop_m"
        ]

        slope_m_per_m = terrain_values[
            "slope_m_per_m"
        ]

        slope_percent = terrain_values[
            "slope_percent"
        ]

        terrain_slope_deg = terrain_values[
            "terrain_slope_deg"
        ]

        if (
            elevation_drop
            < -ELEVATION_TOLERANCE_M
        ):
            uphill_edges += 1

        expected_drop = (
            start_elevation
            - end_elevation
        )

        if (
            abs(
                expected_drop
                - elevation_drop
            )
            > DROP_CONSISTENCY_TOLERANCE_M
        ):
            inconsistent_drops += 1

        if length_m > 0:
            expected_slope = (
                elevation_drop
                / length_m
            )

            if (
                abs(
                    expected_slope
                    - slope_m_per_m
                )
                > SLOPE_CONSISTENCY_TOLERANCE
            ):
                inconsistent_slopes += 1

        expected_percent = (
            slope_m_per_m
            * 100.0
        )

        if (
            abs(
                expected_percent
                - slope_percent
            )
            > PERCENT_CONSISTENCY_TOLERANCE
        ):
            inconsistent_percent_slopes += 1

        if terrain_slope_deg < 0:
            negative_terrain_slopes += 1

    print("\nEdge checks")

    print(
        f"Edges missing attrs  : "
        f"{edges_missing_attributes}"
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
        f"{invalid_upstream_area}"
    )

    print(
        f"Incorrect provenance : "
        f"{incorrect_provenance}"
    )

    print(
        f"Unexpected capacity  : "
        f"{unexpected_capacity}"
    )

    if edges_missing_attributes:
        errors.append(
            "graph edges missing required attributes"
        )

    if invalid_lengths:
        errors.append(
            "invalid graph edge lengths"
        )

    if invalid_accumulation:
        errors.append(
            "invalid graph accumulation values"
        )

    if invalid_upstream_area:
        errors.append(
            "invalid graph upstream areas"
        )

    if incorrect_provenance:
        errors.append(
            "incorrect graph provenance"
        )

    if unexpected_capacity:
        errors.append(
            "unexpected known hydraulic capacity"
        )

    # --------------------------------------------------
    # New terrain checks
    # --------------------------------------------------

    print(
        "\nTerrain attribute checks"
    )

    print(
        f"Invalid terrain attrs : "
        f"{invalid_terrain_attributes}"
    )

    print(
        f"Non-finite terrain    : "
        f"{non_finite_terrain_values}"
    )

    print(
        f"Uphill graph edges    : "
        f"{uphill_edges}"
    )

    print(
        f"Inconsistent drops    : "
        f"{inconsistent_drops}"
    )

    print(
        f"Inconsistent slopes   : "
        f"{inconsistent_slopes}"
    )

    print(
        f"Inconsistent % slopes : "
        f"{inconsistent_percent_slopes}"
    )

    print(
        f"Negative terrain slope: "
        f"{negative_terrain_slopes}"
    )

    if invalid_terrain_attributes:
        errors.append(
            "invalid terrain attributes in graph"
        )

    if non_finite_terrain_values:
        errors.append(
            "non-finite terrain values in graph"
        )

    if uphill_edges:
        errors.append(
            "uphill drainage graph edges detected"
        )

    if inconsistent_drops:
        errors.append(
            "graph elevation-drop values inconsistent"
        )

    if inconsistent_slopes:
        errors.append(
            "graph segment slopes inconsistent"
        )

    if inconsistent_percent_slopes:
        errors.append(
            "graph slope-percent values inconsistent"
        )

    if negative_terrain_slopes:
        errors.append(
            "negative terrain slope values in graph"
        )

    # --------------------------------------------------
    # Connectivity
    # --------------------------------------------------

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    component_sizes = sorted(
        (
            len(component)
            for component in components
        ),
        reverse=True,
    )

    largest_component_size = (
        component_sizes[0]
        if component_sizes
        else 0
    )

    largest_component_percent = (
        (
            largest_component_size
            / graph.number_of_nodes()
        )
        * 100.0
        if graph.number_of_nodes()
        else 0.0
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
        nx.isolates(
            graph
        )
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
        f"{largest_component_size} nodes "
        f"({largest_component_percent:.2f}%)"
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
            "isolated drainage graph nodes detected"
        )

    # --------------------------------------------------
    # Topology
    # --------------------------------------------------

    is_dag = nx.is_directed_acyclic_graph(
        graph
    )

    expected_forest_edges = (
        graph.number_of_nodes()
        - len(components)
    )

    forest_edge_relation = (
        graph.number_of_edges()
        == expected_forest_edges
    )

    print("\nTopology")

    print(
        f"Directed acyclic     : "
        f"{is_dag}"
    )

    print(
        f"Forest edge relation : "
        f"{forest_edge_relation}"
    )

    print(
        f"Expected N-C edges   : "
        f"{expected_forest_edges}"
    )

    if not is_dag:
        errors.append(
            "drainage graph contains directed cycles"
        )

    if not forest_edge_relation:
        errors.append(
            "drainage graph does not satisfy forest edge relation"
        )

    # --------------------------------------------------
    # GIS / graph length consistency
    # --------------------------------------------------

    if not DRAINAGE_PATH.exists():
        raise FileNotFoundError(
            f"Drainage GeoJSON not found: "
            f"{DRAINAGE_PATH}"
        )

    drainage = gpd.read_file(
        DRAINAGE_PATH
    )

    gis_length = float(
        drainage.geometry.length.sum()
    )

    graph_length = sum(
        float(
            data["length_m"]
        )
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

    print(
        "\nLength consistency"
    )

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

    # Millimetre coordinate rounding can create a very
    # small difference. A few metres over the full network
    # is acceptable; large differences are not.
    if length_difference > 5.0:
        errors.append(
            "GIS and graph drainage lengths differ excessively"
        )

    # --------------------------------------------------
    # Directed-flow sanity test
    # --------------------------------------------------

    sanity = directed_flow_sanity_test(
        graph
    )

    print(
        "\nDirected-flow sanity test"
    )

    if sanity is None:

        print(
            "No source-to-downstream path found."
        )

        errors.append(
            "directed-flow sanity path unavailable"
        )

    else:

        (
            start_node,
            end_node,
            path,
            distance,
        ) = sanity

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
            f"{len(path)}"
        )

        print(
            f"Flow-path distance   : "
            f"{distance / 1000:.2f} km"
        )

    # --------------------------------------------------
    # Provenance notice
    # --------------------------------------------------

    print(
        "\nIMPORTANT:"
    )

    print(
        "Drainage graph provenance: "
        "DEM-INFERRED SURFACE FLOW."
    )

    print(
        "It is NOT surveyed municipal "
        "stormwater infrastructure."
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print(
        "\n===================================="
    )

    if errors:

        print(
            "DRAINAGE GRAPH VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )

    else:

        print(
            "DRAINAGE GRAPH VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()
