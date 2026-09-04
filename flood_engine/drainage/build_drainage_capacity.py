from pathlib import Path
import math

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

CAPACITY_GEOJSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_capacity.geojson"
)

CAPACITY_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_capacity.graphml"
)

# -----------------------------------------------------------------
# PROTOTYPE CAPACITY SCENARIO
# -----------------------------------------------------------------
# These are NOT surveyed drain dimensions.
#
# We use Strahler order only to select an equivalent open-channel
# geometry for a transparent Manning-capacity scenario.
#
# This allows the prototype to estimate when modeled runoff exceeds
# an assumed conveyance capacity without claiming that these values
# represent Pune municipal stormwater infrastructure.
# -----------------------------------------------------------------

SCENARIO_NAME = "prototype_open_channel_base"
CAPACITY_SOURCE = "synthetic_manning_scenario"
CAPACITY_IS_OBSERVED = 0

MANNING_N = 0.035

# A small positive slope is required by Manning's equation.
# Flat DEM-derived segments use this scenario floor.
SLOPE_FLOOR_M_PER_M = 0.001

# Equivalent rectangular channel geometry by Strahler order.
# Values are explicit prototype assumptions, not observations.
ORDER_GEOMETRY = {
    1: (0.40, 0.30),  # width m, flow depth m
    2: (0.70, 0.45),
    3: (1.00, 0.60),
    4: (1.40, 0.80),
}


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return value
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Drainage graph not found: {GRAPH_PATH}"
        )

    graph = nx.read_graphml(
        GRAPH_PATH,
        force_multigraph=True,
    )

    if not graph.is_directed():
        raise ValueError(
            "Drainage graph must be directed."
        )

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "Drainage graph must be acyclic before "
            "Strahler ordering."
        )

    return graph


def compute_node_strahler_order(graph):
    """
    Compute Strahler order for every node in a directed DAG.

    A source node receives order 1.

    At a confluence:
      - if the highest incoming order occurs at least twice,
        outgoing order = highest + 1
      - otherwise outgoing order = highest
    """

    node_order = {}

    for node in nx.topological_sort(graph):

        predecessors = list(
            graph.predecessors(node)
        )

        if not predecessors:
            node_order[node] = 1
            continue

        incoming_orders = [
            node_order[pred]
            for pred in predecessors
        ]

        maximum = max(
            incoming_orders
        )

        count_maximum = sum(
            1
            for value in incoming_orders
            if value == maximum
        )

        if count_maximum >= 2:
            node_order[node] = (
                maximum + 1
            )
        else:
            node_order[node] = (
                maximum
            )

    return node_order


def geometry_for_order(order):
    """
    Return prototype equivalent width/depth.

    Order >=4 uses the order-4 geometry so the scenario
    does not grow without an explicit calibration basis.
    """

    order = max(
        1,
        int(order),
    )

    lookup_order = min(
        order,
        4,
    )

    return ORDER_GEOMETRY[
        lookup_order
    ]


def capacity_class_for_order(order):
    if order <= 1:
        return "low"

    if order == 2:
        return "medium"

    if order == 3:
        return "high"

    return "very_high"


def manning_rectangular_capacity(
    width_m,
    depth_m,
    slope_m_per_m,
    manning_n,
):
    """
    Manning discharge for a full equivalent rectangular
    open channel.

        Q = (1/n) * A * R^(2/3) * S^(1/2)

    This is a prototype scenario calculation, not a
    calibrated municipal drain capacity.
    """

    area = (
        width_m
        * depth_m
    )

    wetted_perimeter = (
        width_m
        + 2.0 * depth_m
    )

    hydraulic_radius = (
        area
        / wetted_perimeter
    )

    discharge = (
        (1.0 / manning_n)
        * area
        * hydraulic_radius ** (
            2.0 / 3.0
        )
        * math.sqrt(
            slope_m_per_m
        )
    )

    return discharge


def enrich_graph_with_capacity(
    graph,
):
    node_order = (
        compute_node_strahler_order(
            graph
        )
    )

    for u, v, key, data in graph.edges(
        keys=True,
        data=True,
    ):

        # The outgoing reach takes the Strahler order of
        # its upstream/source node.
        stream_order = int(
            node_order[u]
        )

        width_m, depth_m = (
            geometry_for_order(
                stream_order
            )
        )

        terrain_slope = safe_float(
            data.get(
                "slope_m_per_m",
                0.0,
            )
        )

        effective_slope = max(
            terrain_slope,
            SLOPE_FLOOR_M_PER_M,
        )

        capacity_m3s = (
            manning_rectangular_capacity(
                width_m=width_m,
                depth_m=depth_m,
                slope_m_per_m=effective_slope,
                manning_n=MANNING_N,
            )
        )

        data[
            "strahler_order"
        ] = stream_order

        data[
            "capacity_class"
        ] = capacity_class_for_order(
            stream_order
        )

        data[
            "scenario_width_m"
        ] = float(
            width_m
        )

        data[
            "scenario_depth_m"
        ] = float(
            depth_m
        )

        data[
            "scenario_manning_n"
        ] = float(
            MANNING_N
        )

        data[
            "capacity_effective_slope"
        ] = float(
            effective_slope
        )

        data[
            "capacity_slope_floor"
        ] = float(
            SLOPE_FLOOR_M_PER_M
        )

        data[
            "capacity_m3s"
        ] = float(
            capacity_m3s
        )

        data[
            "capacity_scenario"
        ] = SCENARIO_NAME

        data[
            "capacity_source"
        ] = CAPACITY_SOURCE

        data[
            "capacity_is_observed"
        ] = CAPACITY_IS_OBSERVED

        # Preserve the earlier field but make its meaning
        # explicit: real hydraulic capacity is still unknown.
        data[
            "hydraulic_capacity_known"
        ] = 0

    graph.graph[
        "capacity_scenario"
    ] = SCENARIO_NAME

    graph.graph[
        "capacity_source"
    ] = CAPACITY_SOURCE

    graph.graph[
        "capacity_is_observed"
    ] = CAPACITY_IS_OBSERVED

    graph.graph[
        "scenario_manning_n"
    ] = MANNING_N

    graph.graph[
        "capacity_slope_floor"
    ] = SLOPE_FLOOR_M_PER_M

    return graph


def build_capacity_lookup(graph):
    """
    Build a one-row-per-source-drain lookup for GeoJSON.

    The current drainage network is one edge per source
    drainage feature. max() keeps the result deterministic
    if multipart features appear later.
    """

    rows = []

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        rows.append(
            {
                "drain_id": str(
                    data.get(
                        "source_drain_id",
                        data.get(
                            "drain_id",
                            "",
                        ),
                    )
                ),
                "strahler_order": safe_int(
                    data.get(
                        "strahler_order"
                    )
                ),
                "capacity_class": str(
                    data.get(
                        "capacity_class",
                        "",
                    )
                ),
                "scenario_width_m": safe_float(
                    data.get(
                        "scenario_width_m"
                    )
                ),
                "scenario_depth_m": safe_float(
                    data.get(
                        "scenario_depth_m"
                    )
                ),
                "scenario_manning_n": safe_float(
                    data.get(
                        "scenario_manning_n"
                    )
                ),
                "capacity_effective_slope": safe_float(
                    data.get(
                        "capacity_effective_slope"
                    )
                ),
                "capacity_slope_floor": safe_float(
                    data.get(
                        "capacity_slope_floor"
                    )
                ),
                "capacity_m3s": safe_float(
                    data.get(
                        "capacity_m3s"
                    )
                ),
                "capacity_scenario": str(
                    data.get(
                        "capacity_scenario",
                        "",
                    )
                ),
                "capacity_source": str(
                    data.get(
                        "capacity_source",
                        "",
                    )
                ),
                "capacity_is_observed": safe_int(
                    data.get(
                        "capacity_is_observed"
                    )
                ),
            }
        )

    if not rows:
        raise ValueError(
            "No capacity-enabled graph edges found."
        )

    import pandas as pd

    lookup = pd.DataFrame(
        rows
    )

    # One source feature should normally map to one edge.
    # Keep the first deterministic record if duplicates
    # ever occur due to multipart handling.
    lookup = (
        lookup.sort_values(
            by=[
                "drain_id",
                "strahler_order",
            ]
        )
        .drop_duplicates(
            subset=[
                "drain_id",
            ],
            keep="last",
        )
    )

    return lookup


def main():

    print(
        "\n=== DRAINAGE CAPACITY MODEL ===\n"
    )

    graph = load_graph()

    print(
        f"Graph nodes             : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Graph edges             : "
        f"{graph.number_of_edges()}"
    )

    graph = (
        enrich_graph_with_capacity(
            graph
        )
    )

    orders = [
        int(
            data[
                "strahler_order"
            ]
        )
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    ]

    capacities = [
        float(
            data[
                "capacity_m3s"
            ]
        )
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    ]

    slope_floor_edges = sum(
        1
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
        if float(
            data[
                "slope_m_per_m"
            ]
        ) < SLOPE_FLOOR_M_PER_M
    )

    print(
        f"Maximum Strahler order  : "
        f"{max(orders)}"
    )

    for order in sorted(
        set(orders)
    ):
        count = sum(
            1
            for value in orders
            if value == order
        )

        print(
            f"Order {order} edges          : "
            f"{count}"
        )

    print(
        f"Slope-floor edges       : "
        f"{slope_floor_edges}"
    )

    print(
        f"Minimum capacity        : "
        f"{min(capacities):.4f} m3/s"
    )

    print(
        f"Mean capacity           : "
        f"{np.mean(capacities):.4f} m3/s"
    )

    print(
        f"Maximum capacity        : "
        f"{max(capacities):.4f} m3/s"
    )

    # --------------------------------------------------
    # Save capacity graph
    # --------------------------------------------------

    CAPACITY_GRAPH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        CAPACITY_GRAPH_PATH,
    )

    # --------------------------------------------------
    # Join capacity attributes to drainage GeoJSON
    # --------------------------------------------------

    if not DRAINAGE_PATH.exists():
        raise FileNotFoundError(
            f"Drainage GeoJSON not found: "
            f"{DRAINAGE_PATH}"
        )

    drainage = gpd.read_file(
        DRAINAGE_PATH
    )

    lookup = build_capacity_lookup(
        graph
    )

    capacity_gdf = drainage.merge(
        lookup,
        on="drain_id",
        how="left",
        validate="one_to_one",
    )

    missing_capacity = int(
        capacity_gdf[
            "capacity_m3s"
        ].isna().sum()
    )

    if missing_capacity:
        raise ValueError(
            f"{missing_capacity} drainage features "
            "did not receive capacity attributes."
        )

    capacity_gdf.to_file(
        CAPACITY_GEOJSON_PATH,
        driver="GeoJSON",
    )

    print(
        "\nSaved capacity GeoJSON:"
    )
    print(
        CAPACITY_GEOJSON_PATH
    )

    print(
        "\nSaved capacity GraphML:"
    )
    print(
        CAPACITY_GRAPH_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "capacity_m3s is a SYNTHETIC PROTOTYPE "
        "Manning scenario."
    )

    print(
        "It is NOT a surveyed or calibrated municipal "
        "storm-drain capacity."
    )

    print(
        "Terrain slope and Strahler order are derived; "
        "equivalent channel width/depth and Manning n "
        "are scenario assumptions."
    )

    print(
        "\nDrainage capacity model completed."
    )


if __name__ == "__main__":
    main()
