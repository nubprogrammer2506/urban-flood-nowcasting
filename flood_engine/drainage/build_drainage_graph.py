from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
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

EXPECTED_CRS = "EPSG:32643"

# Millimetre precision prevents tiny floating-point
# differences from creating duplicate graph nodes.
COORD_PRECISION = 3


def coordinate_key(x, y):
    return (
        round(float(x), COORD_PRECISION),
        round(float(y), COORD_PRECISION),
    )


def safe_string(value):
    """
    Convert values into GraphML-safe strings.
    """

    if value is None:
        return ""

    return str(value)


def safe_float(value, default=0.0):
    """
    Convert numeric values safely for GraphML.
    """

    try:
        value = float(value)

        if not np.isfinite(value):
            return float(default)

        return value

    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    """
    Convert numeric values safely to integer.
    """

    try:
        value = float(value)

        if not np.isfinite(value):
            return int(default)

        return int(round(value))

    except (TypeError, ValueError):
        return int(default)


def add_drain_edge(
    graph,
    row,
    geometry,
    suffix=None,
):
    """
    Add one directed inferred-drainage segment.

    Drainage geometries are oriented:

        upstream -> downstream
    """

    coordinates = list(
        geometry.coords
    )

    if len(coordinates) < 2:
        return

    upstream_x, upstream_y = (
        coordinates[0]
    )

    downstream_x, downstream_y = (
        coordinates[-1]
    )

    upstream_key = coordinate_key(
        upstream_x,
        upstream_y,
    )

    downstream_key = coordinate_key(
        downstream_x,
        downstream_y,
    )

    # --------------------------------------------------
    # Create upstream node
    # --------------------------------------------------

    if upstream_key not in graph:

        graph.add_node(
            upstream_key,
            x=float(upstream_key[0]),
            y=float(upstream_key[1]),
        )

    # --------------------------------------------------
    # Create downstream node
    # --------------------------------------------------

    if downstream_key not in graph:

        graph.add_node(
            downstream_key,
            x=float(downstream_key[0]),
            y=float(downstream_key[1]),
        )

    # --------------------------------------------------
    # Stable edge ID
    # --------------------------------------------------

    source_drain_id = safe_string(
        row.get("drain_id")
    )

    if suffix is None:

        graph_drain_id = (
            source_drain_id
        )

    else:

        graph_drain_id = (
            f"{source_drain_id}_{suffix}"
        )

    # --------------------------------------------------
    # Edge attributes
    # --------------------------------------------------

    edge_data = {

        # ----------------------------------------------
        # Identification / provenance
        # ----------------------------------------------

        "drain_id": graph_drain_id,

        "source_drain_id":
            source_drain_id,

        "source_type": safe_string(
            row.get("source_type")
        ),

        "is_inferred": 1,

        # ----------------------------------------------
        # Catchment / accumulation
        # ----------------------------------------------

        "acc_cells": safe_int(
            row.get("acc_cells")
        ),

        "upstream_area_m2": safe_float(
            row.get("upstream_area_m2")
        ),

        # ----------------------------------------------
        # Geometry
        # ----------------------------------------------

        "length_m": float(
            geometry.length
        ),

        "geometry_wkt":
            geometry.wkt,

        # ----------------------------------------------
        # Canonical terrain attributes
        # ----------------------------------------------

        "start_elevation_m": safe_float(
            row.get("start_elevation_m")
        ),

        "end_elevation_m": safe_float(
            row.get("end_elevation_m")
        ),

        "elevation_drop_m": safe_float(
            row.get("elevation_drop_m")
        ),

        "slope_m_per_m": safe_float(
            row.get("slope_m_per_m")
        ),

        "slope_percent": safe_float(
            row.get("slope_percent")
        ),

        "terrain_slope_deg": safe_float(
            row.get("terrain_slope_deg")
        ),

        # ----------------------------------------------
        # Hydraulic-data status
        # ----------------------------------------------
        # Capacity is unknown because this is inferred
        # surface drainage, not surveyed municipal
        # pipe/channel infrastructure.
        # ----------------------------------------------

        "hydraulic_capacity_known": 0,
    }

    graph.add_edge(
        upstream_key,
        downstream_key,
        **edge_data,
    )


def build_graph(drainage):
    """
    Convert canonical inferred drainage lines into
    a directed NetworkX MultiDiGraph.
    """

    graph = nx.MultiDiGraph()

    graph.graph["name"] = (
        "Rajendra Nagar inferred "
        "surface drainage graph"
    )

    graph.graph["crs"] = (
        EXPECTED_CRS
    )

    graph.graph["source_type"] = (
        "dem_inferred_surface_flow"
    )

    graph.graph[
        "is_surveyed_municipal_drainage"
    ] = 0

    graph.graph[
        "terrain_source"
    ] = "canonical_grass_r_watershed"

    for _, row in drainage.iterrows():

        geometry = row.geometry

        if (
            geometry is None
            or geometry.is_empty
        ):
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

                if (
                    part is None
                    or part.is_empty
                ):
                    continue

                add_drain_edge(
                    graph,
                    row,
                    part,
                    suffix=index,
                )

    return graph


def relabel_nodes(graph):
    """
    Replace coordinate tuple node IDs with stable,
    readable drainage-node IDs.
    """

    sorted_nodes = sorted(
        graph.nodes,
        key=lambda node: (
            node[0],
            node[1],
        ),
    )

    mapping = {
        node: (
            f"RN_DNODE_{index:05d}"
        )
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

    # --------------------------------------------------
    # Load canonical drainage dataset
    # --------------------------------------------------

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

    # --------------------------------------------------
    # CRS validation
    # --------------------------------------------------

    if drainage.crs.to_epsg() != 32643:

        print(
            "Reprojecting drainage "
            "to EPSG:32643..."
        )

        drainage = drainage.to_crs(
            EXPECTED_CRS
        )

    # --------------------------------------------------
    # Required enriched attributes
    # --------------------------------------------------

    required_attributes = [
        "drain_id",
        "source_type",
        "is_inferred",
        "acc_cells",
        "upstream_area_m2",
        "length_m",
        "start_elevation_m",
        "end_elevation_m",
        "elevation_drop_m",
        "slope_m_per_m",
        "slope_percent",
        "terrain_slope_deg",
    ]

    missing_attributes = [
        attribute
        for attribute
        in required_attributes
        if attribute not in drainage.columns
    ]

    if missing_attributes:

        raise ValueError(
            "Drainage dataset is missing "
            "required attributes: "
            + ", ".join(
                missing_attributes
            )
        )

    # --------------------------------------------------
    # Geometry validation
    # --------------------------------------------------

    drainage = drainage[
        drainage.geometry.notna()
        & ~drainage.geometry.is_empty
        & drainage.geometry.is_valid
    ].copy()

    print(
        f"Valid drain features : "
        f"{len(drainage)}"
    )

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    graph = build_graph(
        drainage
    )

    graph = relabel_nodes(
        graph
    )

    # --------------------------------------------------
    # Graph statistics
    # --------------------------------------------------

    node_count = (
        graph.number_of_nodes()
    )

    edge_count = (
        graph.number_of_edges()
    )

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    components.sort(
        key=len,
        reverse=True,
    )

    largest_component = (
        len(components[0])
        if components
        else 0
    )

    source_nodes = [
        node
        for node in graph.nodes
        if (
            graph.in_degree(node) == 0
            and
            graph.out_degree(node) > 0
        )
    ]

    sink_nodes = [
        node
        for node in graph.nodes
        if (
            graph.out_degree(node) == 0
            and
            graph.in_degree(node) > 0
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

    # --------------------------------------------------
    # Terrain statistics
    # --------------------------------------------------

    drops = [
        data["elevation_drop_m"]
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    ]

    slopes = [
        data["slope_percent"]
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    ]

    uphill_edges = sum(
        1
        for value in drops
        if value < -0.05
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

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
        f"{len(components)}"
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

    print(
        f"Uphill graph edges   : "
        f"{uphill_edges}"
    )

    if drops:

        print(
            f"Mean elevation drop  : "
            f"{np.mean(drops):.3f} m"
        )

    if slopes:

        print(
            f"Mean segment slope   : "
            f"{np.mean(slopes):.3f} %"
        )

    # --------------------------------------------------
    # DAG validation
    # --------------------------------------------------

    is_dag = (
        nx.is_directed_acyclic_graph(
            graph
        )
    )

    print(
        f"Directed acyclic     : "
        f"{is_dag}"
    )

    # --------------------------------------------------
    # Save GraphML
    # --------------------------------------------------

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
        "The graph represents DEM-inferred "
        "surface-flow pathways using the "
        "canonical terrain pipeline."
    )

    print(
        "It is NOT surveyed municipal "
        "stormwater infrastructure."
    )

    print(
        "\nDrainage graph build completed."
    )


if __name__ == "__main__":
    main()