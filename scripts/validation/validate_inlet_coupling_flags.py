from pathlib import Path

import geopandas as gpd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INLETS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "inferred_inlets.geojson"
)

REQUIRED_COLUMNS = [
    "spatial_class",
    "coupling_eligible",
    "distance_to_road_m",
    "mapping_tolerance_m",
    "coupling_threshold_m",
]

INTERSECTION_TOLERANCE_M = 0.50
COUPLING_DISTANCE_M = 20.0
MAPPING_DISTANCE_M = 30.0


def expected_spatial_class(distance):
    if distance <= INTERSECTION_TOLERANCE_M:
        return "intersection"

    if distance <= 10.0:
        return "near"

    if distance <= COUPLING_DISTANCE_M:
        return "moderate"

    return "distant_candidate"


def main():

    print(
        "\n=== INLET COUPLING FLAG VALIDATION ===\n"
    )

    errors = []

    if not INLETS_PATH.exists():
        raise FileNotFoundError(
            f"Inferred inlets not found: "
            f"{INLETS_PATH}"
        )

    inlets = gpd.read_file(
        INLETS_PATH
    )

    if inlets.empty:
        raise ValueError(
            "Inferred inlet dataset is empty."
        )

    print(
        f"Candidate inlets       : "
        f"{len(inlets)}"
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in inlets.columns
    ]

    if missing_columns:
        print(
            "Missing columns         : "
            + ", ".join(missing_columns)
        )

        errors.append(
            "missing coupling fields"
        )

        print(
            "\n===================================="
        )
        print(
            "INLET COUPLING FLAG VALIDATION: FAILED"
        )

        for error in errors:
            print(f" - {error}")

        print(
            "====================================\n"
        )

        return

    print(
        "Required coupling fields: PASS"
    )

    distances = inlets[
        "distance_to_road_m"
    ].astype(float)

    eligibility = inlets[
        "coupling_eligible"
    ].astype(int)

    mapping_tolerance = inlets[
        "mapping_tolerance_m"
    ].astype(float)

    coupling_threshold = inlets[
        "coupling_threshold_m"
    ].astype(float)

    invalid_numeric = int(
        (
            ~np.isfinite(
                distances
            )
        ).sum()
        + (
            ~np.isfinite(
                mapping_tolerance
            )
        ).sum()
        + (
            ~np.isfinite(
                coupling_threshold
            )
        ).sum()
    )

    invalid_eligibility_values = int(
        (
            ~eligibility.isin(
                [
                    0,
                    1,
                ]
            )
        ).sum()
    )

    expected_eligibility = (
        distances
        <= COUPLING_DISTANCE_M
    ).astype(int)

    wrong_eligibility = int(
        (
            eligibility
            != expected_eligibility
        ).sum()
    )

    wrong_mapping_tolerance = int(
        (
            np.abs(
                mapping_tolerance
                - MAPPING_DISTANCE_M
            )
            > 1e-6
        ).sum()
    )

    wrong_coupling_threshold = int(
        (
            np.abs(
                coupling_threshold
                - COUPLING_DISTANCE_M
            )
            > 1e-6
        ).sum()
    )

    expected_classes = distances.apply(
        expected_spatial_class
    )

    wrong_spatial_class = int(
        (
            inlets[
                "spatial_class"
            ].astype(str)
            != expected_classes
        ).sum()
    )

    outside_mapping_horizon = int(
        (
            distances
            > MAPPING_DISTANCE_M
            + 1e-6
        ).sum()
    )

    eligible_count = int(
        eligibility.sum()
    )

    deferred_count = int(
        len(inlets)
        - eligible_count
    )

    intersection_count = int(
        (
            inlets[
                "spatial_class"
            ]
            == "intersection"
        ).sum()
    )

    near_count = int(
        (
            inlets[
                "spatial_class"
            ]
            == "near"
        ).sum()
    )

    moderate_count = int(
        (
            inlets[
                "spatial_class"
            ]
            == "moderate"
        ).sum()
    )

    distant_count = int(
        (
            inlets[
                "spatial_class"
            ]
            == "distant_candidate"
        ).sum()
    )

    print(
        "\nCoupling-field checks"
    )

    print(
        f"Non-finite values      : "
        f"{invalid_numeric}"
    )

    print(
        f"Invalid eligible values: "
        f"{invalid_eligibility_values}"
    )

    print(
        f"Wrong eligibility      : "
        f"{wrong_eligibility}"
    )

    print(
        f"Wrong mapping tolerance: "
        f"{wrong_mapping_tolerance}"
    )

    print(
        f"Wrong coupling thresh. : "
        f"{wrong_coupling_threshold}"
    )

    print(
        f"Wrong spatial class    : "
        f"{wrong_spatial_class}"
    )

    print(
        f"Outside 30 m horizon   : "
        f"{outside_mapping_horizon}"
    )

    print(
        "\nCoupling summary"
    )

    print(
        f"Intersection           : "
        f"{intersection_count}"
    )

    print(
        f"Near                   : "
        f"{near_count}"
    )

    print(
        f"Moderate               : "
        f"{moderate_count}"
    )

    print(
        f"Distant candidate      : "
        f"{distant_count}"
    )

    print(
        f"Coupling eligible      : "
        f"{eligible_count}"
    )

    print(
        f"Deferred candidates    : "
        f"{deferred_count}"
    )

    for count, message in [
        (
            invalid_numeric,
            "non-finite coupling values",
        ),
        (
            invalid_eligibility_values,
            "invalid coupling_eligible values",
        ),
        (
            wrong_eligibility,
            "coupling eligibility inconsistent",
        ),
        (
            wrong_mapping_tolerance,
            "mapping tolerance inconsistent",
        ),
        (
            wrong_coupling_threshold,
            "coupling threshold inconsistent",
        ),
        (
            wrong_spatial_class,
            "spatial classification inconsistent",
        ),
        (
            outside_mapping_horizon,
            "candidate outside mapping horizon",
        ),
    ]:
        if count:
            errors.append(message)

    print(
        "\nIMPORTANT:"
    )

    print(
        "20 m is an operational prototype "
        "spatial threshold, not a measured "
        "storm-drain inlet capture radius."
    )

    print(
        "\n===================================="
    )

    if errors:

        print(
            "INLET COUPLING FLAG VALIDATION: FAILED"
        )

        for error in errors:
            print(
                f" - {error}"
            )

    else:

        print(
            "INLET COUPLING FLAG VALIDATION: PASSED"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()
