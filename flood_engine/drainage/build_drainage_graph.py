from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRAINAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage.geojson"
)

GRAPH_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_graph.graphml"
)

EXPECTED_EPSG = 32643

# Millimetre-level rounding avoids creating separate
# graph nodes because of tiny floating-point differences.
COORD_PRECISION = 3


def coordinate_key(x, y):
    return (
        round(float(x), COORD_PRECISION),
        round(float(y), COORD_PRECISION),
    )


def safe_string(value):
    if value is None:
        return ""

    return str(value)


def add_drain_edge(
    graph,
    row,
    geometry,
    suffix=None,
):
    """
    Add one directed drainage segment.

    Geometry orientation is preserved:
    first coordinate = upstream
    final coordinate = downstream.
    """

    coordinates = list(geometry.coords)

    if len(coordinates) < 2:
        return

    upstream_x, upstream_y = coordinates[0]
    downstream_x, downstream_y = coordinates[-1]

    upstream_key = coordinate_key(
        upstream_x,
        upstream_y,
    )

    downstream_key = coordinate_key(
        downstream_x,
        downstream_y,
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    if upstream_key not in graph:
        graph.add_node(
            upstream_key,
            x=float(upstream_key[0]),
            y=float(upstream_key[1]),
        )

    if downstream_key not in graph:
        graph.add_node(
            downstream_key,
            x=float(downstream_key[0]),
            y=float(downstream_key[1]),
        )

    source_drain_id = safe_string(
        row.get("drain_id")
    )

    if suffix is None:
        graph_drain_id = source_drain_id
    else:
        graph_drain_id = (
            f"{source_drain_id}_{suffix}"
        )

    # -----------------------------------------------------
    # Edge attributes
    # -----------------------------------------------------

    edge_data = {
        "drain_id": graph_drain_id,
        "source_drain_id": source_drain_id,

        "source_type": safe_string(
            row.get("source_type")
        ),

        "is_inferred": 1,

        "acc_cells": int(
            row.get("acc_cells", 0)
        ),

        "upstream_area_m2": float(
            row.get(
                "upstream_area_m2",
                0.0,
            )
        ),

        "length_m": float(
            geometry.length
        ),

        # GraphML cannot store Shapely geometries,
        # therefore geometry is stored as WKT.
        "geometry_wkt": geometry.wkt,

        # Hydraulic properties are intentionally not
        # invented here.
        "hydraulic_capacity_known": 0,
    }

    graph.add_edge(
        upstream_key,
        downstream_key,
        **edge_data,
    )


def build_graph(drainage):
    """
    Convert inferred drainage lines into a directed
    NetworkX MultiDiGraph.
    """

    graph = nx.MultiDiGraph()

    graph.graph["name"] = (
        "Rajendra Nagar inferred surface drainage graph"
    )

    graph.graph["crs"] = "EPSG:32643"

    graph.graph["source_type"] = (
        "dem_inferred_surface_flow"
    )

    graph.graph["is_surveyed_municipal_drainage"] = 0

    for _, row in drainage.iterrows():

        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        if isinstance(
            geometry,
            LineString,
        ):

            add_drain_edge(
                graph,
                row,
                geometry,
            )

        elif isinstance(
            geometry,
            MultiLineString,
        ):

            for index, part in enumerate(
                geometry.geoms,
                start=1,
            ):

                add_drain_edge(
                    graph,
                    row,
                    part,
                    suffix=index,
                )

    return graph


def relabel_nodes(graph):
    """
    Replace coordinate tuple node IDs with readable,
    stable drainage node IDs.
    """

    sorted_nodes = sorted(
        graph.nodes,
        key=lambda node: (
            node[0],
            node[1],
        ),
    )

    mapping = {
        node: f"RN_DNODE_{index:05d}"
        for index, node in enumerate(
            sorted_nodes,
            start=1,
        )
    }

    return nx.relabel_nodes(
        graph,
        mapping,
        copy=True,
    )


def main():

    print(
        "\n=== DRAINAGE GRAPH BUILD ===\n"
    )

    # -----------------------------------------------------
    # Load drainage
    # -----------------------------------------------------

    if not DRAINAGE_PATH.exists():
        raise FileNotFoundError(
            f"Drainage dataset not found: "
            f"{DRAINAGE_PATH}"
        )

    drainage = gpd.read_file(
        DRAINAGE_PATH
    )

    if drainage.empty:
        raise ValueError(
            "Drainage dataset is empty."
        )

    if drainage.crs is None:
        raise ValueError(
            "Drainage CRS is missing."
        )

    print(
        f"Drain features       : "
        f"{len(drainage)}"
    )

    print(
        f"Drainage CRS         : "
        f"{drainage.crs}"
    )

    # -----------------------------------------------------
    # CRS
    # -----------------------------------------------------

    if (
        drainage.crs.to_epsg()
        != EXPECTED_EPSG
    ):

        print(
            "Reprojecting drainage "
            "to EPSG:32643..."
        )

        drainage = drainage.to_crs(
            epsg=EXPECTED_EPSG
        )

    # -----------------------------------------------------
    # Geometry filtering
    # -----------------------------------------------------

    drainage = drainage[
        drainage.geometry.notna()
        & ~drainage.geometry.is_empty
        & drainage.geometry.is_valid
    ].copy()

    print(
        f"Valid drain features : "
        f"{len(drainage)}"
    )

    # -----------------------------------------------------
    # Build graph
    # -----------------------------------------------------

    graph = build_graph(
        drainage
    )

    graph = relabel_nodes(
        graph
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    node_count = (
        graph.number_of_nodes()
    )

    edge_count = (
        graph.number_of_edges()
    )

    weak_components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    weak_components.sort(
        key=len,
        reverse=True,
    )

    largest_component = (
        len(weak_components[0])
        if weak_components
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

    total_length = sum(
        data["length_m"]
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    )

    print(
        "\n=== DRAINAGE GRAPH SUMMARY ==="
    )

    print(
        f"Graph nodes          : "
        f"{node_count}"
    )

    print(
        f"Graph edges          : "
        f"{edge_count}"
    )

    print(
        f"Weak components      : "
        f"{len(weak_components)}"
    )

    print(
        f"Largest component    : "
        f"{largest_component} nodes"
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

    print(
        f"Graph drain length   : "
        f"{total_length / 1000:.2f} km"
    )

    # -----------------------------------------------------
    # Directed acyclic graph check
    # -----------------------------------------------------

    is_dag = nx.is_directed_acyclic_graph(
        graph
    )

    print(
        f"Directed acyclic     : "
        f"{is_dag}"
    )

    # -----------------------------------------------------
    # Save GraphML
    # -----------------------------------------------------

    GRAPH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        GRAPH_OUTPUT_PATH,
    )

    print(
        "\nDrainage graph saved:"
    )

    print(
        GRAPH_OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This graph represents DEM-inferred "
        "surface-flow pathways."
    )

    print(
        "It is not surveyed municipal "
        "stormwater infrastructure."
    )

    print(
        "\nDrainage graph build completed."
    )


if __name__ == "__main__":
    main()