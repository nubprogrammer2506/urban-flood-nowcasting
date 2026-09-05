from pathlib import Path
import re

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from shapely import wkt
from shapely.geometry import mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROAD_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph.graphml"
)

FLOOD_DEPTH_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "flood_depth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
)

TIMESERIES_OUTPUT = (
    OUTPUT_DIR
    / "road_flood_depth_timeseries.csv"
)

PEAK_GEOJSON_OUTPUT = (
    OUTPUT_DIR
    / "road_flood_depth_peak.geojson"
)

ENRICHED_GRAPH_OUTPUT = (
    OUTPUT_DIR
    / "road_graph_flood_depth.graphml"
)

EXPECTED_CRS = CRS.from_epsg(32643)

SAMPLING_METHOD = (
    "road_line_intersected_raster_cells_all_touched"
)


def crs_equals(left, right):
    if left is None or right is None:
        return False

    return CRS.from_user_input(left).equals(
        CRS.from_user_input(right)
    )


def parse_interval(path):
    match = re.search(
        r"flood_depth_(\d+)_(\d+)\.tif$",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse flood interval from {path.name}"
        )

    start_min = int(match.group(1))
    end_min = int(match.group(2))

    if end_min <= start_min:
        raise ValueError(
            f"Invalid interval in {path.name}"
        )

    return {
        "interval_id": (
            f"{start_min:03d}_{end_min:03d}"
        ),
        "start_min": start_min,
        "end_min": end_min,
    }


def graph_edge_id(u, v, key, data):
    road_id = str(
        data.get("road_id", "")
    )

    if road_id:
        return road_id

    return f"{u}__{v}__{key}"


def load_graph_edges(graph):
    records = []
    seen_ids = set()

    for u, v, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        edge_id = graph_edge_id(
            u,
            v,
            key,
            data,
        )

        if edge_id in seen_ids:
            raise ValueError(
                f"Duplicate graph road_id/edge ID: {edge_id}"
            )

        seen_ids.add(edge_id)

        geometry_text = str(
            data.get("geometry_wkt", "")
        )

        if not geometry_text:
            raise ValueError(
                f"Missing geometry_wkt for {edge_id}"
            )

        geometry = wkt.loads(
            geometry_text
        )

        if geometry.is_empty:
            raise ValueError(
                f"Empty geometry for {edge_id}"
            )

        records.append(
            {
                "graph_u": str(u),
                "graph_v": str(v),
                "graph_key": str(key),
                "edge_id": edge_id,
                "road_id": str(
                    data.get("road_id", "")
                ),
                "source_road_id": str(
                    data.get(
                        "source_road_id",
                        "",
                    )
                ),
                "road_class": str(
                    data.get(
                        "road_class",
                        "",
                    )
                ),
                "name": str(
                    data.get(
                        "name",
                        "",
                    )
                ),
                "length_m": float(
                    data.get(
                        "length_m",
                        geometry.length,
                    )
                ),
                "geometry": geometry,
            }
        )

    return records


def sample_edge(
    geometry,
    depth,
    nodata,
    transform,
):
    cell_mask = geometry_mask(
        [mapping(geometry)],
        out_shape=depth.shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )

    if nodata is None:
        valid = (
            cell_mask
            & np.isfinite(depth)
        )
    else:
        valid = (
            cell_mask
            & np.isfinite(depth)
            & (depth != nodata)
        )

    values = depth[
        valid
    ].astype(
        np.float64
    )

    if values.size == 0:
        return {
            "sampling_status": (
                "outside_valid_flood_grid"
            ),
            "sample_cell_count": 0,
            "max_depth_m": np.nan,
            "mean_depth_m": np.nan,
            "p90_depth_m": np.nan,
            "fraction_gt_0m": np.nan,
            "fraction_gt_0_02m": np.nan,
            "fraction_gt_0_10m": np.nan,
            "fraction_gt_0_30m": np.nan,
        }

    if np.any(values < 0):
        raise ValueError(
            "Flood-depth raster contains "
            "negative valid values."
        )

    count = int(
        values.size
    )

    return {
        "sampling_status": "mapped",
        "sample_cell_count": count,
        "max_depth_m": float(
            values.max()
        ),
        "mean_depth_m": float(
            values.mean()
        ),
        "p90_depth_m": float(
            np.percentile(
                values,
                90,
            )
        ),
        "fraction_gt_0m": float(
            np.mean(
                values > 0.0
            )
        ),
        "fraction_gt_0_02m": float(
            np.mean(
                values > 0.02
            )
        ),
        "fraction_gt_0_10m": float(
            np.mean(
                values > 0.10
            )
        ),
        "fraction_gt_0_30m": float(
            np.mean(
                values > 0.30
            )
        ),
    }


def main():
    print(
        "\n=== ROAD FLOOD-DEPTH SAMPLING ===\n"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ROAD_GRAPH_PATH.exists():
        raise FileNotFoundError(
            ROAD_GRAPH_PATH
        )

    flood_files = sorted(
        FLOOD_DEPTH_DIR.glob(
            "flood_depth_*.tif"
        )
    )

    if not flood_files:
        raise FileNotFoundError(
            f"No flood-depth rasters found in "
            f"{FLOOD_DEPTH_DIR}"
        )

    graph = nx.read_graphml(
        ROAD_GRAPH_PATH,
        force_multigraph=True,
    )

    edge_records = load_graph_edges(
        graph
    )

    print(
        f"Road graph nodes       : "
        f"{graph.number_of_nodes()}"
    )
    print(
        f"Road graph edges       : "
        f"{graph.number_of_edges()}"
    )
    print(
        f"Flood intervals        : "
        f"{len(flood_files)}"
    )

    with rasterio.open(
        flood_files[0]
    ) as template:
        if not crs_equals(
            template.crs,
            EXPECTED_CRS,
        ):
            raise ValueError(
                "Flood-depth raster is not EPSG:32643."
            )

        template_crs = template.crs
        template_transform = template.transform
        template_width = template.width
        template_height = template.height

        units = str(
            template.tags().get(
                "units",
                "",
            )
        ).lower()

        if units not in {
            "metres",
            "meters",
            "metre",
            "meter",
            "m",
        }:
            raise ValueError(
                "Flood-depth raster units are not metres."
            )

        model = str(
            template.tags().get(
                "model",
                "",
            )
        )

    print(
        f"Flood CRS              : "
        f"{template_crs}"
    )
    print(
        f"Flood raster size      : "
        f"{template_width} x "
        f"{template_height}"
    )
    print(
        f"Sampling method        : "
        f"{SAMPLING_METHOD}"
    )
    print(
        f"Flood model            : "
        f"{model}\n"
    )

    timeseries_rows = []

    for flood_path in flood_files:
        interval = parse_interval(
            flood_path
        )

        with rasterio.open(
            flood_path
        ) as src:
            if (
                not crs_equals(
                    src.crs,
                    template_crs,
                )
                or src.width
                != template_width
                or src.height
                != template_height
                or not src.transform.almost_equals(
                    template_transform
                )
            ):
                raise ValueError(
                    f"{flood_path.name} is not "
                    "aligned with the flood template."
                )

            tags = src.tags()

            current_units = str(
                tags.get(
                    "units",
                    "",
                )
            ).lower()

            if current_units not in {
                "metres",
                "meters",
                "metre",
                "meter",
                "m",
            }:
                raise ValueError(
                    f"{flood_path.name} has "
                    "unexpected units."
                )

            depth = src.read(1)
            nodata = src.nodata

            mapped_edges = 0
            interval_peak = 0.0

            for edge in edge_records:
                metrics = sample_edge(
                    geometry=edge[
                        "geometry"
                    ],
                    depth=depth,
                    nodata=nodata,
                    transform=src.transform,
                )

                if (
                    metrics[
                        "sampling_status"
                    ]
                    == "mapped"
                ):
                    mapped_edges += 1
                    interval_peak = max(
                        interval_peak,
                        metrics[
                            "max_depth_m"
                        ],
                    )

                timeseries_rows.append(
                    {
                        **interval,
                        "edge_id": edge[
                            "edge_id"
                        ],
                        "graph_u": edge[
                            "graph_u"
                        ],
                        "graph_v": edge[
                            "graph_v"
                        ],
                        "graph_key": edge[
                            "graph_key"
                        ],
                        "road_id": edge[
                            "road_id"
                        ],
                        "source_road_id": edge[
                            "source_road_id"
                        ],
                        "road_class": edge[
                            "road_class"
                        ],
                        "name": edge[
                            "name"
                        ],
                        "length_m": edge[
                            "length_m"
                        ],
                        "sampling_method": (
                            SAMPLING_METHOD
                        ),
                        **metrics,
                    }
                )

        print(
            f"{interval['start_min']:>3}-"
            f"{interval['end_min']:>3} min | "
            f"mapped edges "
            f"{mapped_edges:>3}/"
            f"{len(edge_records)} | "
            f"road peak "
            f"{interval_peak:.3f} m"
        )

    timeseries = pd.DataFrame(
        timeseries_rows
    )

    timeseries.to_csv(
        TIMESERIES_OUTPUT,
        index=False,
    )

    peak_records = []

    enriched_graph = graph.copy()

    edge_lookup = {
        edge["edge_id"]: edge
        for edge in edge_records
    }

    for edge_id, group in timeseries.groupby(
        "edge_id",
        sort=False,
    ):
        valid_group = group[
            group[
                "sampling_status"
            ]
            == "mapped"
        ].copy()

        edge = edge_lookup[
            edge_id
        ]

        if valid_group.empty:
            peak_max_depth = -1.0
            peak_mean_depth = -1.0
            peak_p90_depth = -1.0
            peak_interval = ""
            max_fraction_002 = -1.0
            max_fraction_010 = -1.0
            max_fraction_030 = -1.0
            status = (
                "outside_valid_flood_grid"
            )
        else:
            peak_row = valid_group.loc[
                valid_group[
                    "max_depth_m"
                ].idxmax()
            ]

            peak_max_depth = float(
                peak_row[
                    "max_depth_m"
                ]
            )

            peak_mean_depth = float(
                valid_group[
                    "mean_depth_m"
                ].max()
            )

            peak_p90_depth = float(
                valid_group[
                    "p90_depth_m"
                ].max()
            )

            peak_interval = str(
                peak_row[
                    "interval_id"
                ]
            )

            max_fraction_002 = float(
                valid_group[
                    "fraction_gt_0_02m"
                ].max()
            )

            max_fraction_010 = float(
                valid_group[
                    "fraction_gt_0_10m"
                ].max()
            )

            max_fraction_030 = float(
                valid_group[
                    "fraction_gt_0_30m"
                ].max()
            )

            status = "mapped"

        peak_records.append(
            {
                "edge_id": edge_id,
                "road_id": edge[
                    "road_id"
                ],
                "source_road_id": edge[
                    "source_road_id"
                ],
                "road_class": edge[
                    "road_class"
                ],
                "name": edge[
                    "name"
                ],
                "length_m": edge[
                    "length_m"
                ],
                "sampling_status": status,
                "sampling_method": (
                    SAMPLING_METHOD
                ),
                "sampled_intervals": int(
                    len(
                        valid_group
                    )
                ),
                "scenario_peak_max_depth_m": (
                    peak_max_depth
                ),
                "scenario_peak_mean_depth_m": (
                    peak_mean_depth
                ),
                "scenario_peak_p90_depth_m": (
                    peak_p90_depth
                ),
                "scenario_peak_interval": (
                    peak_interval
                ),
                "scenario_max_fraction_gt_0_02m": (
                    max_fraction_002
                ),
                "scenario_max_fraction_gt_0_10m": (
                    max_fraction_010
                ),
                "scenario_max_fraction_gt_0_30m": (
                    max_fraction_030
                ),
                "geometry": edge[
                    "geometry"
                ],
            }
        )

    peak_gdf = gpd.GeoDataFrame(
        peak_records,
        geometry="geometry",
        crs="EPSG:32643",
    )

    peak_gdf.to_file(
        PEAK_GEOJSON_OUTPUT,
        driver="GeoJSON",
    )

    peak_lookup = {
        row.edge_id: row
        for row in peak_gdf.itertuples(
            index=False
        )
    }

    for u, v, key, data in enriched_graph.edges(
        keys=True,
        data=True,
    ):
        edge_id = graph_edge_id(
            u,
            v,
            key,
            data,
        )

        peak = peak_lookup[
            edge_id
        ]

        data[
            "flood_sampling_status"
        ] = str(
            peak.sampling_status
        )

        data[
            "flood_sampling_method"
        ] = SAMPLING_METHOD

        data[
            "sampled_flood_intervals"
        ] = int(
            peak.sampled_intervals
        )

        data[
            "scenario_peak_flood_depth_m"
        ] = float(
            peak.scenario_peak_max_depth_m
        )

        data[
            "scenario_peak_mean_depth_m"
        ] = float(
            peak.scenario_peak_mean_depth_m
        )

        data[
            "scenario_peak_p90_depth_m"
        ] = float(
            peak.scenario_peak_p90_depth_m
        )

        data[
            "scenario_peak_interval"
        ] = str(
            peak.scenario_peak_interval
        )

        data[
            "scenario_max_fraction_gt_0_02m"
        ] = float(
            peak.scenario_max_fraction_gt_0_02m
        )

        data[
            "scenario_max_fraction_gt_0_10m"
        ] = float(
            peak.scenario_max_fraction_gt_0_10m
        )

        data[
            "scenario_max_fraction_gt_0_30m"
        ] = float(
            peak.scenario_max_fraction_gt_0_30m
        )

        data[
            "flood_depth_m"
        ] = float(
            data.get(
                "flood_depth_m",
                -1.0,
            )
        )

        data[
            "flood_risk"
        ] = str(
            data.get(
                "flood_risk",
                "unknown",
            )
        )

        data[
            "is_passable"
        ] = int(
            float(
                data.get(
                    "is_passable",
                    1,
                )
            )
        )

        data[
            "routing_cost"
        ] = float(
            data.get(
                "routing_cost",
                data[
                    "length_m"
                ],
            )
        )

    enriched_graph.graph[
        "flood_depth_units"
    ] = "metres"

    enriched_graph.graph[
        "flood_sampling_method"
    ] = SAMPLING_METHOD

    enriched_graph.graph[
        "flood_depth_model"
    ] = model

    enriched_graph.graph[
        "risk_policy_applied"
    ] = 0

    nx.write_graphml(
        enriched_graph,
        ENRICHED_GRAPH_OUTPUT,
    )

    mapped_peak = peak_gdf[
        peak_gdf[
            "sampling_status"
        ]
        == "mapped"
    ]

    print(
        "\n=== ROAD FLOOD-DEPTH SUMMARY ==="
    )
    print(
        f"Road edges             : "
        f"{len(peak_gdf)}"
    )
    print(
        f"Mapped road edges      : "
        f"{len(mapped_peak)}"
    )
    print(
        f"Unmapped road edges    : "
        f"{len(peak_gdf) - len(mapped_peak)}"
    )

    if not mapped_peak.empty:
        print(
            f"Scenario road max      : "
            f"{mapped_peak['scenario_peak_max_depth_m'].max():.3f} m"
        )
        print(
            f"Edges > 0.02 m         : "
            f"{int((mapped_peak['scenario_peak_max_depth_m'] > 0.02).sum())}"
        )
        print(
            f"Edges > 0.10 m         : "
            f"{int((mapped_peak['scenario_peak_max_depth_m'] > 0.10).sum())}"
        )
        print(
            f"Edges > 0.30 m         : "
            f"{int((mapped_peak['scenario_peak_max_depth_m'] > 0.30).sum())}"
        )

    print(
        "\nSaved:"
    )
    print(
        TIMESERIES_OUTPUT
    )
    print(
        PEAK_GEOJSON_OUTPUT
    )
    print(
        ENRICHED_GRAPH_OUTPUT
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "This stage samples the MVP flood-depth proxy "
        "onto road edges."
    )
    print(
        "No road-risk, passability, or routing-cost "
        "policy has been applied yet."
    )
    print(
        "\nRoad flood-depth sampling completed."
    )


if __name__ == "__main__":
    main()
