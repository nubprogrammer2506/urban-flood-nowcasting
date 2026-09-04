from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from pyproj import CRS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INLETS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "inferred_inlets.geojson"
)

DRAINAGE_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage_graph.graphml"
)

ROADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "roads.geojson"
)

EXPECTED_CRS = CRS.from_epsg(32643)

EXPECTED_SOURCE_TYPE = (
    "inferred_surface_to_drainage_coupling"
)

INTERSECTION_TOLERANCE_M = 0.50
MAX_MAPPING_TOLERANCE_M = 30.0
COORDINATE_TOLERANCE_M = 0.01
AREA_TOLERANCE_M2 = 0.01


REQUIRED_COLUMNS = [
    "inlet_id",
    "drain_node_id",
    "drain_id",
    "road_id",
    "candidate_method",
    "distance_to_road_m",
    "elevation_m",
    "acc_cells",
    "upstream_area_m2",
    "graph_in_degree",
    "graph_out_degree",
    "source_type",
    "is_observed",
    "mapping_tolerance_m",
    "geometry",
]


def crs_matches_expected(crs):
    if crs is None:
        return False

    return CRS.from_user_input(
        crs
    ).equals(
        EXPECTED_CRS
    )


def as_bool_false(value):
    """
    Accept common false representations written/read
    through GeoJSON drivers.
    """

    normalized = str(value).strip().lower()

    return normalized in {
        "0",
        "false",
        "0.0",
    }


def representative_outgoing_edge(
    graph,
    node_id,
):
    """
    Match the deterministic edge-selection rule used by
    map_inferred_inlets.py: choose the outgoing edge with
    greatest flow accumulation.
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

    return edges[0][3]


def main():

    print(
        "\n=== INFERRED INLET VALIDATION ===\n"
    )

    errors = []

    # --------------------------------------------------
    # File checks
    # --------------------------------------------------

    for path in [
        INLETS_PATH,
        DRAINAGE_GRAPH_PATH,
        ROADS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------

    inlets = gpd.read_file(
        INLETS_PATH
    )

    roads = gpd.read_file(
        ROADS_PATH
    )

    graph = nx.read_graphml(
        DRAINAGE_GRAPH_PATH,
        force_multigraph=True,
    )

    if inlets.empty:
        raise ValueError(
            "Inferred inlet dataset is empty."
        )

    print(
        f"Candidate inlets       : "
        f"{len(inlets)}"
    )

    print(
        f"Drainage graph nodes   : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Drainage graph edges   : "
        f"{graph.number_of_edges()}"
    )

    # --------------------------------------------------
    # CRS
    # --------------------------------------------------

    print("\nCRS checks")

    inlet_crs_ok = crs_matches_expected(
        inlets.crs
    )

    road_crs_ok = crs_matches_expected(
        roads.crs
    )

    print(
        f"Inlet CRS              : "
        f"{'PASS' if inlet_crs_ok else 'FAIL'}"
    )

    print(
        f"Road CRS               : "
        f"{'PASS' if road_crs_ok else 'FAIL'}"
    )

    if not inlet_crs_ok:
        errors.append(
            "inlet CRS is not EPSG:32643"
        )

    if not road_crs_ok:
        errors.append(
            "road CRS is not EPSG:32643"
        )

    # --------------------------------------------------
    # Schema
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in inlets.columns
    ]

    print("\nSchema checks")

    if missing_columns:
        print(
            "Missing columns         : "
            + ", ".join(
                missing_columns
            )
        )

        errors.append(
            "missing required inlet columns"
        )

        print(
            "\n===================================="
        )
        print(
            "INFERRED INLET VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

        print(
            "====================================\n"
        )

        return

    print(
        "Required columns       : PASS"
    )

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    null_geometry = int(
        inlets.geometry.isna().sum()
    )

    empty_geometry = int(
        inlets.geometry.is_empty.sum()
    )

    invalid_geometry = int(
        (~inlets.geometry.is_valid).sum()
    )

    non_point_geometry = int(
        (
            inlets.geometry.geom_type
            != "Point"
        ).sum()
    )

    print("\nGeometry checks")

    print(
        f"Null geometries        : "
        f"{null_geometry}"
    )

    print(
        f"Empty geometries       : "
        f"{empty_geometry}"
    )

    print(
        f"Invalid geometries     : "
        f"{invalid_geometry}"
    )

    print(
        f"Non-point geometries   : "
        f"{non_point_geometry}"
    )

    if null_geometry:
        errors.append(
            "null inlet geometries"
        )

    if empty_geometry:
        errors.append(
            "empty inlet geometries"
        )

    if invalid_geometry:
        errors.append(
            "invalid inlet geometries"
        )

    if non_point_geometry:
        errors.append(
            "inlet geometries are not points"
        )

    # --------------------------------------------------
    # IDs
    # --------------------------------------------------

    duplicate_inlet_ids = int(
        inlets[
            "inlet_id"
        ].duplicated().sum()
    )

    duplicate_node_ids = int(
        inlets[
            "drain_node_id"
        ].duplicated().sum()
    )

    missing_inlet_ids = int(
        inlets[
            "inlet_id"
        ].isna().sum()
    )

    missing_node_ids = int(
        inlets[
            "drain_node_id"
        ].isna().sum()
    )

    missing_road_ids = int(
        (
            inlets[
                "road_id"
            ].isna()
            | (
                inlets[
                    "road_id"
                ]
                .astype(str)
                .str.strip()
                == ""
            )
        ).sum()
    )

    print("\nID checks")

    print(
        f"Duplicate inlet IDs    : "
        f"{duplicate_inlet_ids}"
    )

    print(
        f"Duplicate drain nodes  : "
        f"{duplicate_node_ids}"
    )

    print(
        f"Missing inlet IDs      : "
        f"{missing_inlet_ids}"
    )

    print(
        f"Missing drain node IDs : "
        f"{missing_node_ids}"
    )

    print(
        f"Missing road IDs       : "
        f"{missing_road_ids}"
    )

    if duplicate_inlet_ids:
        errors.append(
            "duplicate inlet IDs"
        )

    if duplicate_node_ids:
        errors.append(
            "more than one inlet candidate per drain node"
        )

    if missing_inlet_ids:
        errors.append(
            "missing inlet IDs"
        )

    if missing_node_ids:
        errors.append(
            "missing drainage node IDs"
        )

    if missing_road_ids:
        errors.append(
            "missing road IDs"
        )

    # --------------------------------------------------
    # Numeric values
    # --------------------------------------------------

    numeric_columns = [
        "distance_to_road_m",
        "elevation_m",
        "acc_cells",
        "upstream_area_m2",
        "graph_in_degree",
        "graph_out_degree",
        "mapping_tolerance_m",
    ]

    invalid_numeric = 0

    print("\nNumeric checks")

    for column in numeric_columns:
        values = inlets[
            column
        ].astype(float)

        invalid = int(
            (
                ~np.isfinite(
                    values
                )
            ).sum()
        )

        print(
            f"{column:<22}: "
            f"non-finite={invalid}"
        )

        invalid_numeric += invalid

    if invalid_numeric:
        errors.append(
            "non-finite numeric inlet values"
        )

    # --------------------------------------------------
    # Mapping tolerance / candidate classification
    # --------------------------------------------------

    distances = inlets[
        "distance_to_road_m"
    ].astype(float)

    tolerances = inlets[
        "mapping_tolerance_m"
    ].astype(float)

    negative_distance = int(
        (
            distances < 0
        ).sum()
    )

    outside_own_tolerance = int(
        (
            distances
            > tolerances + 1e-6
        ).sum()
    )

    excessive_tolerance = int(
        (
            tolerances
            > MAX_MAPPING_TOLERANCE_M
            + 1e-6
        ).sum()
    )

    expected_method = np.where(
        distances
        <= INTERSECTION_TOLERANCE_M,
        "road_intersection",
        "road_proximity",
    )

    incorrect_method = int(
        (
            inlets[
                "candidate_method"
            ].astype(str)
            != expected_method
        ).sum()
    )

    print(
        "\nMapping checks"
    )

    print(
        f"Negative road distance : "
        f"{negative_distance}"
    )

    print(
        f"Outside own tolerance  : "
        f"{outside_own_tolerance}"
    )

    print(
        f"Tolerance > 30 m       : "
        f"{excessive_tolerance}"
    )

    print(
        f"Wrong candidate method : "
        f"{incorrect_method}"
    )

    if negative_distance:
        errors.append(
            "negative road distances"
        )

    if outside_own_tolerance:
        errors.append(
            "candidate exceeds its mapping tolerance"
        )

    if excessive_tolerance:
        errors.append(
            "mapping tolerance exceeds 30 m"
        )

    if incorrect_method:
        errors.append(
            "candidate-method classification inconsistent"
        )

    # --------------------------------------------------
    # Provenance
    # --------------------------------------------------

    incorrect_source = int(
        (
            inlets[
                "source_type"
            ].astype(str)
            != EXPECTED_SOURCE_TYPE
        ).sum()
    )

    incorrectly_observed = int(
        (
            ~inlets[
                "is_observed"
            ].apply(
                as_bool_false
            )
        ).sum()
    )

    print("\nProvenance checks")

    print(
        f"Incorrect source tags  : "
        f"{incorrect_source}"
    )

    print(
        f"Marked as observed     : "
        f"{incorrectly_observed}"
    )

    if incorrect_source:
        errors.append(
            "incorrect inlet provenance"
        )

    if incorrectly_observed:
        errors.append(
            "inferred inlet marked as observed"
        )

    # --------------------------------------------------
    # Graph-node and edge consistency
    # --------------------------------------------------

    missing_graph_nodes = 0
    sink_nodes_used = 0
    coordinate_mismatch = 0
    degree_mismatch = 0
    drain_id_mismatch = 0
    accumulation_mismatch = 0
    upstream_area_mismatch = 0

    for _, row in inlets.iterrows():

        node_id = str(
            row[
                "drain_node_id"
            ]
        )

        if node_id not in graph:
            missing_graph_nodes += 1
            continue

        graph_node = graph.nodes[
            node_id
        ]

        graph_x = float(
            graph_node["x"]
        )

        graph_y = float(
            graph_node["y"]
        )

        point = row.geometry

        coordinate_error = (
            (
                point.x
                - graph_x
            ) ** 2
            + (
                point.y
                - graph_y
            ) ** 2
        ) ** 0.5

        if (
            coordinate_error
            > COORDINATE_TOLERANCE_M
        ):
            coordinate_mismatch += 1

        graph_in_degree = int(
            graph.in_degree(
                node_id
            )
        )

        graph_out_degree = int(
            graph.out_degree(
                node_id
            )
        )

        if graph_out_degree <= 0:
            sink_nodes_used += 1

        if (
            int(
                float(
                    row[
                        "graph_in_degree"
                    ]
                )
            )
            != graph_in_degree
            or int(
                float(
                    row[
                        "graph_out_degree"
                    ]
                )
            )
            != graph_out_degree
        ):
            degree_mismatch += 1

        edge_data = (
            representative_outgoing_edge(
                graph,
                node_id,
            )
        )

        if edge_data is None:
            continue

        expected_drain_id = str(
            edge_data.get(
                "source_drain_id",
                edge_data.get(
                    "drain_id",
                    "",
                ),
            )
        )

        if (
            str(
                row[
                    "drain_id"
                ]
            )
            != expected_drain_id
        ):
            drain_id_mismatch += 1

        expected_acc = int(
            float(
                edge_data.get(
                    "acc_cells",
                    0,
                )
            )
        )

        if (
            int(
                float(
                    row[
                        "acc_cells"
                    ]
                )
            )
            != expected_acc
        ):
            accumulation_mismatch += 1

        expected_area = float(
            edge_data.get(
                "upstream_area_m2",
                0.0,
            )
        )

        if (
            abs(
                float(
                    row[
                        "upstream_area_m2"
                    ]
                )
                - expected_area
            )
            > AREA_TOLERANCE_M2
        ):
            upstream_area_mismatch += 1

    print(
        "\nDrainage graph consistency"
    )

    print(
        f"Missing graph nodes     : "
        f"{missing_graph_nodes}"
    )

    print(
        f"Sink nodes used         : "
        f"{sink_nodes_used}"
    )

    print(
        f"Coordinate mismatches   : "
        f"{coordinate_mismatch}"
    )

    print(
        f"Degree mismatches       : "
        f"{degree_mismatch}"
    )

    print(
        f"Drain ID mismatches     : "
        f"{drain_id_mismatch}"
    )

    print(
        f"Accumulation mismatches : "
        f"{accumulation_mismatch}"
    )

    print(
        f"Upstream area mismatch  : "
        f"{upstream_area_mismatch}"
    )

    if missing_graph_nodes:
        errors.append(
            "inlet references missing graph nodes"
        )

    if sink_nodes_used:
        errors.append(
            "sink nodes used as inlet candidates"
        )

    if coordinate_mismatch:
        errors.append(
            "inlet geometry does not match graph node"
        )

    if degree_mismatch:
        errors.append(
            "stored graph degrees are inconsistent"
        )

    if drain_id_mismatch:
        errors.append(
            "stored drain IDs are inconsistent"
        )

    if accumulation_mismatch:
        errors.append(
            "stored accumulation values are inconsistent"
        )

    if upstream_area_mismatch:
        errors.append(
            "stored upstream areas are inconsistent"
        )

    # --------------------------------------------------
    # Road-reference consistency
    # --------------------------------------------------

    if "road_id" in roads.columns:

        valid_road_ids = set(
            roads[
                "road_id"
            ]
            .astype(str)
        )

        unknown_road_ids = int(
            (
                ~inlets[
                    "road_id"
                ]
                .astype(str)
                .isin(
                    valid_road_ids
                )
            ).sum()
        )

    else:
        unknown_road_ids = len(
            inlets
        )

    print(
        "\nRoad-reference checks"
    )

    print(
        f"Unknown road IDs        : "
        f"{unknown_road_ids}"
    )

    if unknown_road_ids:
        errors.append(
            "inlet references unknown road IDs"
        )

    # --------------------------------------------------
    # Distance distribution for tolerance review
    # --------------------------------------------------

    bins = [
        (
            "0-0.5 m",
            distances <= 0.5,
        ),
        (
            ">0.5-5 m",
            (
                distances > 0.5
            )
            & (
                distances <= 5
            ),
        ),
        (
            ">5-10 m",
            (
                distances > 5
            )
            & (
                distances <= 10
            ),
        ),
        (
            ">10-15 m",
            (
                distances > 10
            )
            & (
                distances <= 15
            ),
        ),
        (
            ">15-20 m",
            (
                distances > 15
            )
            & (
                distances <= 20
            ),
        ),
        (
            ">20-30 m",
            (
                distances > 20
            )
            & (
                distances <= 30
            ),
        ),
    ]

    print(
        "\nRoad-distance distribution"
    )

    for label, mask in bins:
        count = int(
            mask.sum()
        )

        print(
            f"{label:<18}: "
            f"{count}"
        )

    print(
        f"Mean distance          : "
        f"{distances.mean():.2f} m"
    )

    print(
        f"Median distance        : "
        f"{distances.median():.2f} m"
    )

    print(
        f"Maximum distance       : "
        f"{distances.max():.2f} m"
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print(
        "\nIMPORTANT:"
    )

    print(
        "These points are inferred "
        "surface-to-drainage coupling candidates."
    )

    print(
        "They are NOT surveyed municipal "
        "storm-drain inlets."
    )

    print(
        "\n===================================="
    )

    if errors:

        print(
            "INFERRED INLET VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )

    else:

        print(
            "INFERRED INLET VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()
