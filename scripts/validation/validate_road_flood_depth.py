from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_GRAPH_PATH = (
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

TIMESERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_flood_depth_timeseries.csv"
)

PEAK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_flood_depth_peak.geojson"
)

ENRICHED_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "road_graph_flood_depth.graphml"
)

EPS = 1e-8


def main():
    print(
        "\n=== ROAD FLOOD-DEPTH SAMPLING VALIDATION ===\n"
    )

    for path in [
        BASE_GRAPH_PATH,
        TIMESERIES_PATH,
        PEAK_PATH,
        ENRICHED_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    flood_files = sorted(
        FLOOD_DEPTH_DIR.glob(
            "flood_depth_*.tif"
        )
    )

    if not flood_files:
        raise FileNotFoundError(
            "No flood-depth rasters found."
        )

    base_graph = nx.read_graphml(
        BASE_GRAPH_PATH,
        force_multigraph=True,
    )

    enriched_graph = nx.read_graphml(
        ENRICHED_GRAPH_PATH,
        force_multigraph=True,
    )

    timeseries = pd.read_csv(
        TIMESERIES_PATH
    )

    peak = gpd.read_file(
        PEAK_PATH
    )

    errors = []

    edge_count = (
        base_graph.number_of_edges()
    )

    interval_count = len(
        flood_files
    )

    expected_rows = (
        edge_count
        * interval_count
    )

    print(
        f"Baseline road edges       : "
        f"{edge_count}"
    )
    print(
        f"Flood intervals           : "
        f"{interval_count}"
    )
    print(
        f"Timeseries rows           : "
        f"{len(timeseries)}"
    )
    print(
        f"Expected timeseries rows  : "
        f"{expected_rows}"
    )
    print(
        f"Peak road features        : "
        f"{len(peak)}"
    )

    if len(
        timeseries
    ) != expected_rows:
        errors.append(
            "unexpected road/interval row count"
        )

    if len(
        peak
    ) != edge_count:
        errors.append(
            "peak feature count differs from road graph"
        )

    if (
        enriched_graph.number_of_nodes()
        != base_graph.number_of_nodes()
        or enriched_graph.number_of_edges()
        != edge_count
    ):
        errors.append(
            "enriched graph topology differs from baseline"
        )

    duplicate_rows = int(
        timeseries.duplicated(
            subset=[
                "edge_id",
                "interval_id",
            ]
        ).sum()
    )

    duplicate_peak = int(
        peak.duplicated(
            subset=[
                "edge_id"
            ]
        ).sum()
    )

    print(
        f"Duplicate interval rows   : "
        f"{duplicate_rows}"
    )
    print(
        f"Duplicate peak edge IDs   : "
        f"{duplicate_peak}"
    )

    if duplicate_rows:
        errors.append(
            "duplicate edge interval rows"
        )

    if duplicate_peak:
        errors.append(
            "duplicate peak edge IDs"
        )

    mapped = timeseries[
        timeseries[
            "sampling_status"
        ]
        == "mapped"
    ].copy()

    unmapped = timeseries[
        timeseries[
            "sampling_status"
        ]
        != "mapped"
    ].copy()

    allowed_status = {
        "mapped",
        "outside_valid_flood_grid",
    }

    bad_status = int(
        (
            ~timeseries[
                "sampling_status"
            ].astype(str).isin(
                allowed_status
            )
        ).sum()
    )

    print(
        "\nSampling coverage"
    )
    print(
        f"Mapped interval rows      : "
        f"{len(mapped)}"
    )
    print(
        f"Unmapped interval rows    : "
        f"{len(unmapped)}"
    )
    print(
        f"Bad status rows           : "
        f"{bad_status}"
    )

    if bad_status:
        errors.append(
            "unexpected sampling status"
        )

    if mapped.empty:
        errors.append(
            "no road edges mapped to flood grid"
        )

    numeric_fields = [
        "max_depth_m",
        "mean_depth_m",
        "p90_depth_m",
        "fraction_gt_0m",
        "fraction_gt_0_02m",
        "fraction_gt_0_10m",
        "fraction_gt_0_30m",
    ]

    non_finite = 0

    for field in numeric_fields:
        values = mapped[
            field
        ].astype(float)

        non_finite += int(
            (
                ~np.isfinite(
                    values
                )
            ).sum()
        )

    negative_depth = int(
        (
            mapped[
                [
                    "max_depth_m",
                    "mean_depth_m",
                    "p90_depth_m",
                ]
            ].astype(float)
            < -EPS
        ).any(
            axis=1
        ).sum()
    )

    mean_over_max = int(
        (
            mapped[
                "mean_depth_m"
            ].astype(float)
            - mapped[
                "max_depth_m"
            ].astype(float)
            > EPS
        ).sum()
    )

    p90_over_max = int(
        (
            mapped[
                "p90_depth_m"
            ].astype(float)
            - mapped[
                "max_depth_m"
            ].astype(float)
            > EPS
        ).sum()
    )

    fraction_columns = [
        "fraction_gt_0m",
        "fraction_gt_0_02m",
        "fraction_gt_0_10m",
        "fraction_gt_0_30m",
    ]

    fraction_values = mapped[
        fraction_columns
    ].astype(float)

    invalid_fraction = int(
        (
            (fraction_values < -EPS)
            | (fraction_values > 1.0 + EPS)
        ).any(
            axis=1
        ).sum()
    )

    invalid_sample_count = int(
        (
            mapped[
                "sample_cell_count"
            ].astype(int)
            <= 0
        ).sum()
    )

    fraction_order_bad = int(
        (
            (
                mapped[
                    "fraction_gt_0_30m"
                ].astype(float)
                - mapped[
                    "fraction_gt_0_10m"
                ].astype(float)
                > EPS
            )
            |
            (
                mapped[
                    "fraction_gt_0_10m"
                ].astype(float)
                - mapped[
                    "fraction_gt_0_02m"
                ].astype(float)
                > EPS
            )
            |
            (
                mapped[
                    "fraction_gt_0_02m"
                ].astype(float)
                - mapped[
                    "fraction_gt_0m"
                ].astype(float)
                > EPS
            )
        ).sum()
    )

    print(
        "\nDepth metric checks"
    )
    print(
        f"Non-finite mapped values : "
        f"{non_finite}"
    )
    print(
        f"Negative depth rows       : "
        f"{negative_depth}"
    )
    print(
        f"Mean > max rows           : "
        f"{mean_over_max}"
    )
    print(
        f"P90 > max rows            : "
        f"{p90_over_max}"
    )
    print(
        f"Invalid fractions         : "
        f"{invalid_fraction}"
    )
    print(
        f"Invalid sample counts     : "
        f"{invalid_sample_count}"
    )
    print(
        f"Fraction ordering errors  : "
        f"{fraction_order_bad}"
    )

    if non_finite:
        errors.append(
            "non-finite mapped flood metrics"
        )

    if negative_depth:
        errors.append(
            "negative road flood depth"
        )

    if mean_over_max:
        errors.append(
            "mean road depth exceeds max depth"
        )

    if p90_over_max:
        errors.append(
            "P90 road depth exceeds max depth"
        )

    if invalid_fraction:
        errors.append(
            "road flooded fractions outside [0,1]"
        )

    if invalid_sample_count:
        errors.append(
            "mapped road has zero sampled cells"
        )

    if fraction_order_bad:
        errors.append(
            "flooded fraction thresholds are inconsistent"
        )

    mapped_peak = peak[
        peak[
            "sampling_status"
        ]
        == "mapped"
    ].copy()

    peak_negative = int(
        (
            mapped_peak[
                [
                    "scenario_peak_max_depth_m",
                    "scenario_peak_mean_depth_m",
                    "scenario_peak_p90_depth_m",
                ]
            ].astype(float)
            < -EPS
        ).any(
            axis=1
        ).sum()
    )

    wrong_interval_count = int(
        (
            mapped_peak[
                "sampled_intervals"
            ].astype(int)
            != interval_count
        ).sum()
    )

    print(
        "\nPeak summary checks"
    )
    print(
        f"Mapped peak roads         : "
        f"{len(mapped_peak)}"
    )
    print(
        f"Negative peak depths      : "
        f"{peak_negative}"
    )
    print(
        f"Wrong interval counts     : "
        f"{wrong_interval_count}"
    )

    if peak_negative:
        errors.append(
            "negative peak road metrics"
        )

    if wrong_interval_count:
        errors.append(
            "mapped peak road missing intervals"
        )

    changed_risk = 0
    changed_passable = 0
    changed_cost = 0
    missing_sampling_attrs = 0

    for _, _, _, data in enriched_graph.edges(
        keys=True,
        data=True,
    ):
        if str(
            data.get(
                "flood_risk",
                "",
            )
        ) != "unknown":
            changed_risk += 1

        if int(
            float(
                data.get(
                    "is_passable",
                    0,
                )
            )
        ) != 1:
            changed_passable += 1

        length_m = float(
            data[
                "length_m"
            ]
        )

        routing_cost = float(
            data[
                "routing_cost"
            ]
        )

        if abs(
            routing_cost
            - length_m
        ) > 1e-6:
            changed_cost += 1

        if (
            "scenario_peak_flood_depth_m"
            not in data
            or "flood_sampling_status"
            not in data
        ):
            missing_sampling_attrs += 1

    policy_flag = int(
        float(
            enriched_graph.graph.get(
                "risk_policy_applied",
                1,
            )
        )
    )

    print(
        "\nBaseline-policy preservation"
    )
    print(
        f"Risk labels changed       : "
        f"{changed_risk}"
    )
    print(
        f"Passability changed       : "
        f"{changed_passable}"
    )
    print(
        f"Routing costs changed     : "
        f"{changed_cost}"
    )
    print(
        f"Missing sampling attrs    : "
        f"{missing_sampling_attrs}"
    )
    print(
        f"Risk policy flag          : "
        f"{policy_flag}"
    )

    if changed_risk:
        errors.append(
            "risk labels changed during sampling milestone"
        )

    if changed_passable:
        errors.append(
            "passability changed during sampling milestone"
        )

    if changed_cost:
        errors.append(
            "routing costs changed during sampling milestone"
        )

    if missing_sampling_attrs:
        errors.append(
            "enriched graph missing flood sampling attributes"
        )

    if policy_flag != 0:
        errors.append(
            "risk policy incorrectly marked as applied"
        )

    print(
        "\nIMPORTANT:"
    )
    print(
        "Flood depths come from the terrain-weighted "
        "MVP flood-depth proxy, not calibrated 2D hydraulics."
    )
    print(
        "This validation covers sampling only; "
        "passability/routing policy is intentionally absent."
    )

    print(
        "\n============================================"
    )

    if errors:
        print(
            "ROAD FLOOD-DEPTH SAMPLING VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )
    else:
        print(
            "ROAD FLOOD-DEPTH SAMPLING VALIDATION: PASSED"
        )

    print(
        "============================================\n"
    )


if __name__ == "__main__":
    main()
