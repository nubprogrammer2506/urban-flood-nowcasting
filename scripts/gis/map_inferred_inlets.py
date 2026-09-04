from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from pyproj import CRS
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "roads.geojson"
)

DRAINAGE_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_graph.graphml"
)

DEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "terrain"
    / "rajendra_nagar_dem_filled.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "inferred_inlets.geojson"
)

EXPECTED_CRS = CRS.from_epsg(32643)

# Candidate search horizon tied to the 30 m canonical DEM.
# This is a mapping/search tolerance, NOT a physical inlet
# capture radius.
MAX_ROAD_DISTANCE_M = 30.0

# Operational threshold used later for coupling.
# Candidates farther than this are preserved but marked
# as not coupling-ready.
COUPLING_DISTANCE_M = 20.0

# Very small road distance treated as an approximate
# road/drainage intersection.
INTERSECTION_TOLERANCE_M = 0.50


def crs_matches_expected(crs):
    if crs is None:
        return False

    return CRS.from_user_input(
        crs
    ).equals(
        EXPECTED_CRS
    )


def sample_raster(
    raster,
    x,
    y,
):
    """
    Safely sample one raster value.
    """

    try:
        row, col = raster.index(
            x,
            y,
        )

    except Exception:
        return np.nan

    if (
        row < 0
        or row >= raster.height
        or col < 0
        or col >= raster.width
    ):
        return np.nan

    value = raster.read(
        1,
        window=(
            (
                row,
                row + 1,
            ),
            (
                col,
                col + 1,
            ),
        ),
    )[0, 0]

    if not np.isfinite(value):
        return np.nan

    if (
        raster.nodata is not None
        and np.isclose(
            value,
            raster.nodata,
        )
    ):
        return np.nan

    return float(value)


def load_drainage_graph():
    """
    Load the canonical inferred drainage graph.
    """

    if not DRAINAGE_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Drainage graph not found: "
            f"{DRAINAGE_GRAPH_PATH}"
        )

    graph = nx.read_graphml(
        DRAINAGE_GRAPH_PATH,
        force_multigraph=True,
    )

    if not graph.is_directed():
        raise ValueError(
            "Drainage graph must be directed."
        )

    return graph


def build_node_geodataframe(
    graph,
):
    """
    Convert drainage graph nodes into a GeoDataFrame.

    Only nodes with at least one downstream edge are
    considered candidate coupling locations.
    Sink nodes are excluded.
    """

    records = []

    for node_id, data in graph.nodes(
        data=True
    ):

        if graph.out_degree(
            node_id
        ) <= 0:
            continue

        if (
            "x" not in data
            or "y" not in data
        ):
            continue

        x = float(
            data["x"]
        )

        y = float(
            data["y"]
        )

        records.append(
            {
                "drain_node_id": str(
                    node_id
                ),
                "graph_in_degree": int(
                    graph.in_degree(
                        node_id
                    )
                ),
                "graph_out_degree": int(
                    graph.out_degree(
                        node_id
                    )
                ),
                "geometry": Point(
                    x,
                    y,
                ),
            }
        )

    if not records:
        raise ValueError(
            "No usable drainage nodes found."
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:32643",
    )


def outgoing_edge_attributes(
    graph,
    node_id,
):
    """
    Return representative downstream-edge attributes.

    The drainage network is D8-derived and should normally
    have one downstream edge from each non-sink node.

    If multiple edges exist, choose deterministically by
    greatest accumulation.
    """

    edges = list(
        graph.out_edges(
            node_id,
            keys=True,
            data=True,
        )
    )

    if not edges:
        return None

    edges.sort(
        key=lambda edge: float(
            edge[3].get(
                "acc_cells",
                0,
            )
        ),
        reverse=True,
    )

    _, _, _, data = edges[0]

    return {
        "drain_id": str(
            data.get(
                "source_drain_id",
                data.get(
                    "drain_id",
                    "",
                ),
            )
        ),
        "acc_cells": int(
            float(
                data.get(
                    "acc_cells",
                    0,
                )
            )
        ),
        "upstream_area_m2": float(
            data.get(
                "upstream_area_m2",
                0.0,
            )
        ),
    }


def spatial_class(
    distance,
):
    """
    Assign a transparent geometric class based only on
    road proximity.

    This is not a physical inlet-capacity classification.
    """

    if distance <= INTERSECTION_TOLERANCE_M:
        return "intersection"

    if distance <= 10.0:
        return "near"

    if distance <= COUPLING_DISTANCE_M:
        return "moderate"

    return "distant_candidate"


def main():

    print(
        "\n=== INFERRED INLET MAPPING ===\n"
    )

    # --------------------------------------------------
    # Load roads
    # --------------------------------------------------

    if not ROADS_PATH.exists():
        raise FileNotFoundError(
            f"Road dataset not found: "
            f"{ROADS_PATH}"
        )

    roads = gpd.read_file(
        ROADS_PATH
    )

    if roads.empty:
        raise ValueError(
            "Road dataset is empty."
        )

    if not crs_matches_expected(
        roads.crs
    ):
        roads = roads.to_crs(
            "EPSG:32643"
        )

    roads = roads[
        roads.geometry.notna()
        & ~roads.geometry.is_empty
        & roads.geometry.is_valid
    ].copy()

    print(
        f"Road features          : "
        f"{len(roads)}"
    )

    # --------------------------------------------------
    # Load drainage graph
    # --------------------------------------------------

    graph = load_drainage_graph()

    print(
        f"Drainage graph nodes   : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Drainage graph edges   : "
        f"{graph.number_of_edges()}"
    )

    node_gdf = build_node_geodataframe(
        graph
    )

    print(
        f"Eligible drain nodes   : "
        f"{len(node_gdf)}"
    )

    # --------------------------------------------------
    # Prepare roads for nearest-neighbour search
    # --------------------------------------------------

    road_columns = [
        "geometry",
    ]

    if "road_id" in roads.columns:
        road_columns.append(
            "road_id"
        )

    road_subset = roads[
        road_columns
    ].copy()

    if "road_id" not in road_subset.columns:
        road_subset[
            "road_id"
        ] = ""

    # --------------------------------------------------
    # Find drainage nodes near roads
    # --------------------------------------------------

    joined = gpd.sjoin_nearest(
        node_gdf,
        road_subset,
        how="left",
        max_distance=MAX_ROAD_DISTANCE_M,
        distance_col="distance_to_road_m",
    )

    joined = joined[
        joined[
            "distance_to_road_m"
        ].notna()
    ].copy()

    if joined.empty:
        raise ValueError(
            "No drainage nodes found within "
            f"{MAX_ROAD_DISTANCE_M:.1f} m "
            "of roads."
        )

    # sjoin_nearest can return more than one row when
    # multiple roads are exactly equidistant.
    # Keep one deterministic road reference per node.
    joined = joined.sort_values(
        by=[
            "drain_node_id",
            "distance_to_road_m",
            "road_id",
        ]
    )

    joined = joined.drop_duplicates(
        subset=[
            "drain_node_id",
        ],
        keep="first",
    )

    # --------------------------------------------------
    # Add drainage-edge metadata
    # --------------------------------------------------

    drain_ids = []
    acc_cells = []
    upstream_areas = []

    for node_id in joined[
        "drain_node_id"
    ]:

        attributes = (
            outgoing_edge_attributes(
                graph,
                node_id,
            )
        )

        if attributes is None:

            drain_ids.append("")
            acc_cells.append(0)
            upstream_areas.append(
                0.0
            )

            continue

        drain_ids.append(
            attributes[
                "drain_id"
            ]
        )

        acc_cells.append(
            attributes[
                "acc_cells"
            ]
        )

        upstream_areas.append(
            attributes[
                "upstream_area_m2"
            ]
        )

    joined[
        "drain_id"
    ] = drain_ids

    joined[
        "acc_cells"
    ] = acc_cells

    joined[
        "upstream_area_m2"
    ] = upstream_areas

    # --------------------------------------------------
    # Sample canonical DEM
    # --------------------------------------------------

    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"Canonical DEM not found: "
            f"{DEM_PATH}"
        )

    elevations = []

    with rasterio.open(
        DEM_PATH
    ) as dem:

        if not crs_matches_expected(
            dem.crs
        ):
            raise ValueError(
                "Canonical DEM is not "
                "EPSG:32643."
            )

        for geometry in joined.geometry:

            elevations.append(
                sample_raster(
                    dem,
                    geometry.x,
                    geometry.y,
                )
            )

    joined[
        "elevation_m"
    ] = np.round(
        elevations,
        3,
    )

    # --------------------------------------------------
    # Candidate method
    # --------------------------------------------------

    joined[
        "candidate_method"
    ] = np.where(
        joined[
            "distance_to_road_m"
        ]
        <= INTERSECTION_TOLERANCE_M,
        "road_intersection",
        "road_proximity",
    )

    joined[
        "distance_to_road_m"
    ] = joined[
        "distance_to_road_m"
    ].round(
        3
    )

    # --------------------------------------------------
    # Coupling eligibility
    # --------------------------------------------------
    # Keep all candidates within the 30 m search horizon,
    # but only mark <=20 m as operationally eligible.
    # --------------------------------------------------

    joined[
        "coupling_eligible"
    ] = (
        joined[
            "distance_to_road_m"
        ]
        <= COUPLING_DISTANCE_M
    ).astype(
        int
    )

    # --------------------------------------------------
    # Spatial class
    # --------------------------------------------------

    joined[
        "spatial_class"
    ] = joined[
        "distance_to_road_m"
    ].apply(
        spatial_class
    )

    # --------------------------------------------------
    # Explicit provenance
    # --------------------------------------------------

    joined[
        "source_type"
    ] = (
        "inferred_surface_to_drainage_coupling"
    )

    joined[
        "is_observed"
    ] = 0

    joined[
        "mapping_tolerance_m"
    ] = MAX_ROAD_DISTANCE_M

    joined[
        "coupling_threshold_m"
    ] = COUPLING_DISTANCE_M

    # --------------------------------------------------
    # Stable inlet IDs
    # --------------------------------------------------

    joined = joined.sort_values(
        by=[
            "drain_node_id",
        ]
    ).reset_index(
        drop=True
    )

    joined[
        "inlet_id"
    ] = [
        f"RN_INLET_{index:05d}"
        for index in range(
            1,
            len(joined) + 1,
        )
    ]

    # --------------------------------------------------
    # Final output schema
    # --------------------------------------------------

    output_columns = [
        "inlet_id",
        "drain_node_id",
        "drain_id",
        "road_id",
        "candidate_method",
        "spatial_class",
        "coupling_eligible",
        "distance_to_road_m",
        "elevation_m",
        "acc_cells",
        "upstream_area_m2",
        "graph_in_degree",
        "graph_out_degree",
        "source_type",
        "is_observed",
        "mapping_tolerance_m",
        "coupling_threshold_m",
        "geometry",
    ]

    output = joined[
        output_columns
    ].copy()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_file(
        OUTPUT_PATH,
        driver="GeoJSON",
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    intersections = int(
        (
            output[
                "candidate_method"
            ]
            == "road_intersection"
        ).sum()
    )

    proximity = int(
        (
            output[
                "candidate_method"
            ]
            == "road_proximity"
        ).sum()
    )

    coupling_eligible = int(
        output[
            "coupling_eligible"
        ].sum()
    )

    deferred_candidates = int(
        len(output)
        - coupling_eligible
    )

    missing_elevation = int(
        output[
            "elevation_m"
        ].isna().sum()
    )

    intersection_class = int(
        (
            output[
                "spatial_class"
            ]
            == "intersection"
        ).sum()
    )

    near_class = int(
        (
            output[
                "spatial_class"
            ]
            == "near"
        ).sum()
    )

    moderate_class = int(
        (
            output[
                "spatial_class"
            ]
            == "moderate"
        ).sum()
    )

    distant_class = int(
        (
            output[
                "spatial_class"
            ]
            == "distant_candidate"
        ).sum()
    )

    print(
        "\n=== INLET MAPPING SUMMARY ==="
    )

    print(
        f"Candidate inlets       : "
        f"{len(output)}"
    )

    print(
        f"Road intersections     : "
        f"{intersections}"
    )

    print(
        f"Road proximity points  : "
        f"{proximity}"
    )

    print(
        f"Coupling eligible      : "
        f"{coupling_eligible}"
    )

    print(
        f"Deferred candidates    : "
        f"{deferred_candidates}"
    )

    print(
        f"Missing DEM elevation  : "
        f"{missing_elevation}"
    )

    print(
        f"Maximum road distance  : "
        f"{output['distance_to_road_m'].max():.2f} m"
    )

    print(
        "\nSpatial classes"
    )

    print(
        f"Intersection           : "
        f"{intersection_class}"
    )

    print(
        f"Near (<=10 m)          : "
        f"{near_class}"
    )

    print(
        f"Moderate (<=20 m)      : "
        f"{moderate_class}"
    )

    print(
        f"Distant candidate      : "
        f"{distant_class}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are inferred surface-to-drainage "
        "coupling candidates."
    )

    print(
        "They are NOT surveyed or observed "
        "municipal storm-drain inlets."
    )

    print(
        "The 30 m road-distance threshold is "
        "a search/mapping horizon tied to DEM "
        "resolution, not a physical inlet "
        "capture radius."
    )

    print(
        "The 20 m coupling threshold is an "
        "operational spatial-filter threshold "
        "for this prototype, not a measured "
        "hydraulic property."
    )

    print(
        "\nInlet mapping completed."
    )


if __name__ == "__main__":
    main()
