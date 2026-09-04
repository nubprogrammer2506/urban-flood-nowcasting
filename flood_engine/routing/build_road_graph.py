from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "roads.geojson"
)

GRAPH_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph.graphml"
)

EXPECTED_CRS = "EPSG:32643"

# Coordinates are rounded to millimetre precision before
# generating node IDs. This prevents tiny floating-point
# differences from creating duplicate junction nodes.
COORD_PRECISION = 3


def coordinate_key(x, y):
    return (
        round(float(x), COORD_PRECISION),
        round(float(y), COORD_PRECISION),
    )


def normalize_attribute(value):
    """
    Convert values into GraphML-safe primitive values.
    """

    if value is None:
        return ""

    if isinstance(value, bool):
        return int(value)

    return str(value)


def add_linestring_to_graph(
    graph,
    row,
    geometry,
    segment_suffix=None,
):
    """
    Add one LineString as a road edge.
    """

    coordinates = list(geometry.coords)

    if len(coordinates) < 2:
        return

    start_x, start_y = coordinates[0]
    end_x, end_y = coordinates[-1]

    start_key = coordinate_key(start_x, start_y)
    end_key = coordinate_key(end_x, end_y)

    # -----------------------------------------------------
    # Create graph nodes
    # -----------------------------------------------------

    if start_key not in graph:
        graph.add_node(
            start_key,
            x=float(start_key[0]),
            y=float(start_key[1]),
        )

    if end_key not in graph:
        graph.add_node(
            end_key,
            x=float(end_key[0]),
            y=float(end_key[1]),
        )

    road_id = str(row.get("road_id", ""))

    if segment_suffix is not None:
        graph_road_id = (
            f"{road_id}_{segment_suffix}"
        )
    else:
        graph_road_id = road_id

    # -----------------------------------------------------
    # Edge attributes
    # -----------------------------------------------------

    edge_data = {
        "road_id": graph_road_id,
        "source_road_id": road_id,
        "osm_id": normalize_attribute(
            row.get("osm_id")
        ),
        "road_class": normalize_attribute(
            row.get("road_class")
        ),
        "name": normalize_attribute(
            row.get("name")
        ),
        "oneway": int(bool(row.get("oneway", False))),
        "lanes": normalize_attribute(
            row.get("lanes")
        ),
        "maxspeed": normalize_attribute(
            row.get("maxspeed")
        ),
        "surface": normalize_attribute(
            row.get("surface")
        ),
        "bridge": normalize_attribute(
            row.get("bridge")
        ),
        "tunnel": normalize_attribute(
            row.get("tunnel")
        ),
        "access": normalize_attribute(
            row.get("access")
        ),

        # Calculate from the actual geometry used in graph
        "length_m": float(geometry.length),

        # Preserve geometry as WKT because GraphML
        # cannot directly store Shapely objects.
        "geometry_wkt": geometry.wkt,

        # -------------------------------------------------
        # Future flood-routing fields
        # -------------------------------------------------
        # These are neutral placeholders, not flood data.
        "flood_depth_m": -1.0,
        "flood_risk": "unknown",
        "is_passable": 1,
        "routing_cost": float(geometry.length),
    }

    graph.add_edge(
        start_key,
        end_key,
        **edge_data,
    )


def build_graph(roads):
    """
    Convert processed road geometries into an undirected
    NetworkX MultiGraph.
    """

    graph = nx.MultiGraph()

    graph.graph["name"] = (
        "Rajendra Nagar baseline road graph"
    )

    graph.graph["crs"] = EXPECTED_CRS
    graph.graph["directed_routing_ready"] = 0

    for _, row in roads.iterrows():

        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        if isinstance(geometry, LineString):
            add_linestring_to_graph(
                graph,
                row,
                geometry,
            )

        elif isinstance(geometry, MultiLineString):

            for index, part in enumerate(
                geometry.geoms,
                start=1,
            ):
                add_linestring_to_graph(
                    graph,
                    row,
                    part,
                    segment_suffix=index,
                )

    return graph


def relabel_nodes(graph):
    """
    Replace coordinate tuple IDs with stable readable
    IDs while preserving x/y node attributes.
    """

    sorted_nodes = sorted(
        graph.nodes,
        key=lambda node: (node[0], node[1]),
    )

    mapping = {
        node: f"RN_NODE_{index:05d}"
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

    print("\n=== ROAD GRAPH BUILD ===\n")

    # -----------------------------------------------------
    # Load canonical road dataset
    # -----------------------------------------------------

    if not ROADS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical roads not found: {ROADS_PATH}"
        )

    roads = gpd.read_file(ROADS_PATH)

    if roads.empty:
        raise ValueError(
            "Canonical road dataset is empty."
        )

    if roads.crs is None:
        raise ValueError(
            "Canonical road CRS is missing."
        )

    print(f"Road features       : {len(roads)}")
    print(f"Road CRS            : {roads.crs}")

    # -----------------------------------------------------
    # CRS validation
    # -----------------------------------------------------

    if roads.crs.to_epsg() != 32643:
        print(
            "Reprojecting roads to EPSG:32643..."
        )

        roads = roads.to_crs(EXPECTED_CRS)

    # -----------------------------------------------------
    # Geometry validation
    # -----------------------------------------------------

    roads = roads[
        roads.geometry.notna()
        & ~roads.geometry.is_empty
        & roads.geometry.is_valid
    ].copy()

    print(
        f"Valid road features : {len(roads)}"
    )

    # -----------------------------------------------------
    # Build NetworkX graph
    # -----------------------------------------------------

    graph = build_graph(roads)

    graph = relabel_nodes(graph)

    # -----------------------------------------------------
    # Connectivity statistics
    # -----------------------------------------------------

    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()

    components = list(
        nx.connected_components(graph)
    )

    number_of_components = len(components)

    largest_component_size = max(
        (len(component) for component in components),
        default=0,
    )

    isolated_nodes = list(
        nx.isolates(graph)
    )

    total_graph_length = sum(
        data["length_m"]
        for _, _, data in graph.edges(data=True)
    )

    print("\n=== GRAPH SUMMARY ===")

    print(
        f"Graph nodes         : "
        f"{number_of_nodes}"
    )

    print(
        f"Graph edges         : "
        f"{number_of_edges}"
    )

    print(
        f"Connected components: "
        f"{number_of_components}"
    )

    print(
        f"Largest component   : "
        f"{largest_component_size} nodes"
    )

    print(
        f"Isolated nodes      : "
        f"{len(isolated_nodes)}"
    )

    print(
        f"Graph road length   : "
        f"{total_graph_length / 1000:.2f} km"
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

    print("\nRoad graph saved:")
    print(GRAPH_OUTPUT_PATH)

    print("\nRoad graph build completed.")


if __name__ == "__main__":
    main()