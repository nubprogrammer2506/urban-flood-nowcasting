from pathlib import Path

import rasterio
from pyproj import CRS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TERRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "terrain"
)

FILES = [
    "rajendra_nagar_dem_utm.tif",
    "rajendra_nagar_dem_filled.tif",
    "rajendra_nagar_flow_accumulation.tif",
    "rajendra_nagar_flow_direction.tif",
    "rajendra_nagar_slope.tif",
]

EXPECTED_CRS = CRS.from_epsg(32643)


def main():

    print("\n=== TERRAIN ALIGNMENT VALIDATION ===\n")

    reference = None
    errors = []

    for filename in FILES:

        path = TERRAIN_DIR / filename

        if not path.exists():
            errors.append(
                f"Missing file: {filename}"
            )
            continue

        with rasterio.open(path) as src:

            info = {
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
                "transform": src.transform,
                "bounds": src.bounds,
            }

            print(filename)
            print(f"  CRS        : {src.crs}")
            print(
                f"  Size       : "
                f"{src.width} x {src.height}"
            )
            print(
                f"  Resolution : {src.res}"
            )
            print(
                f"  Bounds     : {src.bounds}"
            )

            # -----------------------------------------
            # Semantic CRS validation
            # -----------------------------------------

            if src.crs is None:

                errors.append(
                    f"{filename}: missing CRS"
                )

                print(
                    "  CRS match  : FAIL"
                )

            else:

                actual_crs = CRS.from_user_input(
                    src.crs
                )

                crs_matches = actual_crs.equals(
                    EXPECTED_CRS
                )

                print(
                    f"  CRS match  : "
                    f"{crs_matches}"
                )

                if not crs_matches:
                    errors.append(
                        f"{filename}: "
                        "expected EPSG:32643"
                    )

            print()

            # -----------------------------------------
            # Grid alignment
            # -----------------------------------------

            if reference is None:

                reference = info

            else:

                if info["crs"] != reference["crs"]:

                    # Rasterio CRS text may differ while
                    # representing the same CRS.
                    current_crs = CRS.from_user_input(
                        info["crs"]
                    )

                    reference_crs = CRS.from_user_input(
                        reference["crs"]
                    )

                    if not current_crs.equals(
                        reference_crs
                    ):
                        errors.append(
                            f"{filename}: CRS mismatch"
                        )

                if (
                    info["width"]
                    != reference["width"]
                    or
                    info["height"]
                    != reference["height"]
                ):
                    errors.append(
                        f"{filename}: "
                        "dimension mismatch"
                    )

                if not (
                    abs(
                        info["resolution"][0]
                        - reference["resolution"][0]
                    ) < 1e-6
                    and
                    abs(
                        info["resolution"][1]
                        - reference["resolution"][1]
                    ) < 1e-6
                ):
                    errors.append(
                        f"{filename}: "
                        "resolution mismatch"
                    )

                if not info[
                    "transform"
                ].almost_equals(
                    reference["transform"]
                ):
                    errors.append(
                        f"{filename}: "
                        "transform mismatch"
                    )

    print(
        "======================================"
    )

    if errors:

        print(
            "TERRAIN ALIGNMENT: FAILED"
        )

        for error in errors:
            print(f" - {error}")

    else:

        print(
            "TERRAIN ALIGNMENT: PASSED"
        )

        print(
            "All canonical terrain rasters "
            "share the same CRS and grid."
        )

    print(
        "======================================\n"
    )


if __name__ == "__main__":
    main()