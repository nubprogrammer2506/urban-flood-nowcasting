from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


ROADS_FILE = Path(
    "data/processed/roads/roads.geojson"
)

FLOOD_DIR = Path(
    "data/outputs/flood_depth"
)

OUTPUT_DIR = Path(
    "data/outputs/road_risk"
)

INTERVALS = [
    "000_030",
    "030_060",
    "060_090",
    "090_120",
    "120_150",
    "150_180",
]

SAMPLE_SPACING_M = 10.0


def classify_risk(depth_m):

    if depth_m < 0.05:
        return "safe"

    if depth_m < 0.15:
        return "low"

    if depth_m < 0.30:
        return "medium"

    return "high"


def is_passable(depth_m):

    # MVP routing threshold:
    # >= 30 cm is treated as avoid / blocked.
    return int(depth_m < 0.30)


def routing_multiplier(risk):

    if risk == "safe":
        return 1.0

    if risk == "low":
        return 1.5

    if risk == "medium":
        return 4.0

    return 1000.0


def sample_line(geometry, spacing):

    if geometry.length == 0:
        return [geometry.interpolate(0)]

    distances = np.arange(
        0,
        geometry.length,
        spacing,
    )

    distances = np.append(
        distances,
        geometry.length,
    )

    return [
        geometry.interpolate(
            float(distance)
        )
        for distance in distances
    ]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    roads = gpd.read_file(
        ROADS_FILE
    )

    if roads.empty:
        raise ValueError(
            "Road dataset is empty"
        )

    if roads.crs is None:
        raise ValueError(
            "Road dataset has no CRS"
        )

    print("Road flood-risk generation")
    print()
    print(
        f"Road features: {len(roads)}"
    )
    print(
        f"Road CRS: {roads.crs}"
    )
    print()

    for interval in INTERVALS:

        flood_file = (
            FLOOD_DIR
            / f"flood_depth_{interval}.tif"
        )

        if not flood_file.exists():
            raise FileNotFoundError(
                flood_file
            )

        with rasterio.open(
            flood_file
        ) as src:

            if src.crs is None:
                raise ValueError(
                    f"{interval}: flood raster "
                    "has no CRS"
                )

            interval_roads = (
                roads.to_crs(src.crs)
                .copy()
            )

            flood = src.read(1)

            nodata = src.nodata

            max_depths = []
            mean_depths = []
            p90_depths = []
            sample_counts = []

            for geometry in (
                interval_roads.geometry
            ):

                if (
                    geometry is None
                    or geometry.is_empty
                ):
                    max_depths.append(0.0)
                    mean_depths.append(0.0)
                    p90_depths.append(0.0)
                    sample_counts.append(0)
                    continue

                points = sample_line(
                    geometry,
                    SAMPLE_SPACING_M,
                )

                coordinates = [
                    (point.x, point.y)
                    for point in points
                ]

                values = []

                for value in src.sample(
                    coordinates
                ):

                    depth = float(
                        value[0]
                    )

                    if not np.isfinite(
                        depth
                    ):
                        continue

                    if (
                        nodata is not None
                        and np.isclose(
                            depth,
                            nodata
                        )
                    ):
                        continue

                    if depth < 0:
                        continue

                    values.append(depth)

                if len(values) == 0:

                    max_depth = 0.0
                    mean_depth = 0.0
                    p90_depth = 0.0

                else:

                    values = np.array(
                        values,
                        dtype=float,
                    )

                    max_depth = float(
                        values.max()
                    )

                    mean_depth = float(
                        values.mean()
                    )

                    p90_depth = float(
                        np.percentile(
                            values,
                            90,
                        )
                    )

                max_depths.append(
                    max_depth
                )

                mean_depths.append(
                    mean_depth
                )

                p90_depths.append(
                    p90_depth
                )

                sample_counts.append(
                    len(values)
                )

        # -------------------------------------
        # Risk classification
        #
        # Use maximum sampled depth for safety.
        # -------------------------------------
        interval_roads[
            "flood_max_m"
        ] = max_depths

        interval_roads[
            "flood_mean_m"
        ] = mean_depths

        interval_roads[
            "flood_p90_m"
        ] = p90_depths

        interval_roads[
            "flood_samples"
        ] = sample_counts

        interval_roads[
            "flood_risk"
        ] = interval_roads[
            "flood_max_m"
        ].apply(
            classify_risk
        )

        interval_roads[
            "is_passable"
        ] = interval_roads[
            "flood_max_m"
        ].apply(
            is_passable
        )

        interval_roads[
            "routing_cost"
        ] = interval_roads.apply(
            lambda row:
                float(row["length_m"])
                * routing_multiplier(
                    row["flood_risk"]
                ),
            axis=1,
        )

        interval_roads[
            "interval_id"
        ] = interval

        # -------------------------------------
        # Statistics
        # -------------------------------------
        risk_counts = (
            interval_roads[
                "flood_risk"
            ]
            .value_counts()
            .to_dict()
        )

        high_count = int(
            (
                interval_roads[
                    "flood_risk"
                ]
                == "high"
            ).sum()
        )

        blocked_count = int(
            (
                interval_roads[
                    "is_passable"
                ]
                == 0
            ).sum()
        )

        maximum = float(
            interval_roads[
                "flood_max_m"
            ].max()
        )

        output_file = (
            OUTPUT_DIR
            / f"road_risk_{interval}.geojson"
        )

        interval_roads.to_file(
            output_file,
            driver="GeoJSON",
        )

        print(
            f"{interval} | "
            f"safe {risk_counts.get('safe', 0):>4} | "
            f"low {risk_counts.get('low', 0):>4} | "
            f"medium {risk_counts.get('medium', 0):>4} | "
            f"high {high_count:>4} | "
            f"blocked {blocked_count:>4} | "
            f"max depth {maximum:.3f} m"
        )

    print()
    print(
        "Road flood-risk layers "
        "generated successfully."
    )


if __name__ == "__main__":
    main()