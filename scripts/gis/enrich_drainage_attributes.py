from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRAINAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drainage"
    / "drainage.geojson"
)

FILLED_DEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "terrain"
    / "rajendra_nagar_dem_filled.tif"
)

SLOPE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "terrain"
    / "rajendra_nagar_slope.tif"
)

EXPECTED_CRS = CRS.from_epsg(32643)

# Small tolerance for numerical/raster effects.
ELEVATION_TOLERANCE_M = 0.05


def crs_matches_epsg32643(crs):
    if crs is None:
        return False

    return CRS.from_user_input(
        crs
    ).equals(
        EXPECTED_CRS
    )


def validate_raster_alignment(
    reference,
    other,
    label,
):
    """
    Validate that another terrain raster shares
    the canonical DEM grid.
    """

    if not crs_matches_epsg32643(
        other.crs
    ):
        raise ValueError(
            f"{label} is not EPSG:32643."
        )

    if (
        other.width != reference.width
        or other.height != reference.height
    ):
        raise ValueError(
            f"{label} dimensions do not "
            "match canonical DEM."
        )

    if not other.transform.almost_equals(
        reference.transform
    ):
        raise ValueError(
            f"{label} transform does not "
            "match canonical DEM."
        )


def geometry_endpoints(geometry):
    """
    Return the first and final coordinate while
    preserving the existing drainage direction.

    The drainage geometry is already oriented:
        upstream -> downstream
    """

    if isinstance(
        geometry,
        LineString,
    ):
        coords = list(
            geometry.coords
        )

        if len(coords) < 2:
            return None

        return (
            coords[0],
            coords[-1],
        )

    if isinstance(
        geometry,
        MultiLineString,
    ):
        parts = [
            part
            for part in geometry.geoms
            if (
                part is not None
                and not part.is_empty
                and len(part.coords) >= 2
            )
        ]

        if not parts:
            return None

        return (
            list(parts[0].coords)[0],
            list(parts[-1].coords)[-1],
        )

    return None


def sample_raster(
    raster,
    x,
    y,
):
    """
    Sample one raster value safely.
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


def main():

    print(
        "\n=== DRAINAGE ATTRIBUTE ENRICHMENT ===\n"
    )

    # --------------------------------------------------
    # Input checks
    # --------------------------------------------------

    required_files = [
        DRAINAGE_PATH,
        FILLED_DEM_PATH,
        SLOPE_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------
    # Load drainage
    # --------------------------------------------------

    drainage = gpd.read_file(
        DRAINAGE_PATH
    )

    if drainage.empty:
        raise ValueError(
            "Drainage dataset is empty."
        )

    if not crs_matches_epsg32643(
        drainage.crs
    ):
        raise ValueError(
            "Drainage dataset must use "
            "EPSG:32643."
        )

    print(
        f"Drainage segments     : "
        f"{len(drainage)}"
    )

    print(
        f"Drainage CRS          : "
        f"EPSG:32643"
    )

    # --------------------------------------------------
    # Open canonical terrain
    # --------------------------------------------------

    with (
        rasterio.open(
            FILLED_DEM_PATH
        ) as dem_src,
        rasterio.open(
            SLOPE_PATH
        ) as slope_src,
    ):

        if not crs_matches_epsg32643(
            dem_src.crs
        ):
            raise ValueError(
                "Canonical DEM is not "
                "EPSG:32643."
            )

        validate_raster_alignment(
            dem_src,
            slope_src,
            "Slope raster",
        )

        print(
            f"Canonical DEM         : "
            f"{FILLED_DEM_PATH.name}"
        )

        print(
            f"Canonical slope       : "
            f"{SLOPE_PATH.name}"
        )

        print(
            f"Raster grid           : "
            f"{dem_src.width} x "
            f"{dem_src.height}"
        )

        # ----------------------------------------------
        # Attribute containers
        # ----------------------------------------------

        start_elevations = []
        end_elevations = []

        elevation_drops = []

        slopes_m_per_m = []
        slopes_percent = []

        terrain_slopes_deg = []

        invalid_endpoint_count = 0
        missing_dem_count = 0
        missing_slope_count = 0

        # ----------------------------------------------
        # Process every drainage segment
        # ----------------------------------------------

        for _, feature in drainage.iterrows():

            geometry = feature.geometry

            endpoints = geometry_endpoints(
                geometry
            )

            if endpoints is None:

                invalid_endpoint_count += 1

                start_elevations.append(
                    np.nan
                )

                end_elevations.append(
                    np.nan
                )

                elevation_drops.append(
                    np.nan
                )

                slopes_m_per_m.append(
                    np.nan
                )

                slopes_percent.append(
                    np.nan
                )

                terrain_slopes_deg.append(
                    np.nan
                )

                continue

            (
                start_point,
                end_point,
            ) = endpoints

            start_x, start_y = (
                start_point
            )

            end_x, end_y = (
                end_point
            )

            # ------------------------------------------
            # Sample DEM
            # ------------------------------------------

            start_elevation = sample_raster(
                dem_src,
                start_x,
                start_y,
            )

            end_elevation = sample_raster(
                dem_src,
                end_x,
                end_y,
            )

            if (
                not np.isfinite(
                    start_elevation
                )
                or not np.isfinite(
                    end_elevation
                )
            ):
                missing_dem_count += 1

            # ------------------------------------------
            # Elevation drop
            # ------------------------------------------

            if (
                np.isfinite(
                    start_elevation
                )
                and np.isfinite(
                    end_elevation
                )
            ):

                elevation_drop = (
                    start_elevation
                    - end_elevation
                )

            else:

                elevation_drop = np.nan

            # ------------------------------------------
            # Segment slope
            # ------------------------------------------

            length_m = float(
                feature["length_m"]
            )

            if (
                np.isfinite(
                    elevation_drop
                )
                and length_m > 0
            ):

                slope_m_per_m = (
                    elevation_drop
                    / length_m
                )

                slope_percent = (
                    slope_m_per_m
                    * 100.0
                )

            else:

                slope_m_per_m = np.nan
                slope_percent = np.nan

            # ------------------------------------------
            # Sample canonical slope raster
            # at geometry midpoint
            # ------------------------------------------

            midpoint = geometry.interpolate(
                0.5,
                normalized=True,
            )

            terrain_slope_deg = (
                sample_raster(
                    slope_src,
                    midpoint.x,
                    midpoint.y,
                )
            )

            if not np.isfinite(
                terrain_slope_deg
            ):
                missing_slope_count += 1

            start_elevations.append(
                start_elevation
            )

            end_elevations.append(
                end_elevation
            )

            elevation_drops.append(
                elevation_drop
            )

            slopes_m_per_m.append(
                slope_m_per_m
            )

            slopes_percent.append(
                slope_percent
            )

            terrain_slopes_deg.append(
                terrain_slope_deg
            )

    # --------------------------------------------------
    # Add attributes
    # --------------------------------------------------

    drainage[
        "start_elevation_m"
    ] = np.round(
        start_elevations,
        3,
    )

    drainage[
        "end_elevation_m"
    ] = np.round(
        end_elevations,
        3,
    )

    drainage[
        "elevation_drop_m"
    ] = np.round(
        elevation_drops,
        3,
    )

    drainage[
        "slope_m_per_m"
    ] = np.round(
        slopes_m_per_m,
        6,
    )

    drainage[
        "slope_percent"
    ] = np.round(
        slopes_percent,
        3,
    )

    drainage[
        "terrain_slope_deg"
    ] = np.round(
        terrain_slopes_deg,
        3,
    )

    # --------------------------------------------------
    # Direction statistics
    # --------------------------------------------------

    valid_drop = drainage[
        "elevation_drop_m"
    ].dropna()

    downhill_count = int(
        (
            valid_drop
            > ELEVATION_TOLERANCE_M
        ).sum()
    )

    flat_count = int(
        (
            valid_drop.abs()
            <= ELEVATION_TOLERANCE_M
        ).sum()
    )

    uphill_count = int(
        (
            valid_drop
            < -ELEVATION_TOLERANCE_M
        ).sum()
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    drainage.to_file(
        DRAINAGE_PATH,
        driver="GeoJSON",
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(
        "\n=== ATTRIBUTE SUMMARY ==="
    )

    print(
        f"Segments enriched     : "
        f"{len(drainage)}"
    )

    print(
        f"Invalid geometries     : "
        f"{invalid_endpoint_count}"
    )

    print(
        f"Missing DEM samples    : "
        f"{missing_dem_count}"
    )

    print(
        f"Missing slope samples  : "
        f"{missing_slope_count}"
    )

    print(
        f"Downhill segments      : "
        f"{downhill_count}"
    )

    print(
        f"Flat segments          : "
        f"{flat_count}"
    )

    print(
        f"Uphill segments        : "
        f"{uphill_count}"
    )

    if not valid_drop.empty:

        print(
            f"Mean elevation drop    : "
            f"{valid_drop.mean():.3f} m"
        )

        print(
            f"Maximum elevation drop : "
            f"{valid_drop.max():.3f} m"
        )

    valid_slope = drainage[
        "slope_percent"
    ].dropna()

    if not valid_slope.empty:

        print(
            f"Mean segment slope     : "
            f"{valid_slope.mean():.3f} %"
        )

    print(
        "\nEnriched drainage saved:"
    )

    print(
        DRAINAGE_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Elevation and slope attributes are "
        "derived from canonical terrain."
    )

    print(
        "No municipal pipe capacity, diameter, "
        "material, or surveyed drain properties "
        "have been invented."
    )

    print(
        "\nDrainage attribute enrichment completed."
    )


if __name__ == "__main__":
    main()