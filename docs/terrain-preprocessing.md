# Rajendra Nagar Terrain Preprocessing

## Pilot Area

Rajendra Nagar, Pune, Maharashtra.

## Source DEM

Copernicus GLO-30 DSM obtained through OpenTopography.

Source resolution: approximately 30 metres.

## Processing CRS

EPSG:32643  
WGS 84 / UTM Zone 43N

All terrain calculations are performed in metres.

## DEM Processing Workflow

The terrain-processing workflow used for the MVP is:

Copernicus GLO-30
→ buffered terrain download
→ reprojection to EPSG:32643
→ sink filling
→ flow accumulation
→ flow direction
→ slope calculation

## Terrain Buffer

The first DEM download used a smaller surrounding buffer.

When flow accumulation was generated, approximately 30% of cells
inside the Rajendra Nagar AOI were marked by GRASS as potentially
underestimated because runoff could originate outside the DEM region.

A larger terrain region was therefore downloaded.

The larger DEM reduced this issue significantly and was used for the
final terrain products.

The final accumulation raster was generated using the GRASS
r.watershed positive accumulation option.

This does not mean the entire upstream catchment is represented.
External upstream inflow remains a limitation of the MVP.

## Tools Used

- QGIS 3.44 LTR
- GDAL
- GRASS GIS r.watershed
- Fill sinks (Wang & Liu)
- Rasterio
- GeoPandas
- NumPy

## Final Terrain Outputs

Generated locally under:

data/processed/terrain/

Files:

- rajendra_nagar_dem_utm.tif
- rajendra_nagar_dem_filled.tif
- rajendra_nagar_flow_accumulation.tif
- rajendra_nagar_flow_direction.tif
- rajendra_nagar_slope.tif

These generated raster files are not committed to GitHub.

## Raster Properties

CRS:

EPSG:32643

Resolution:

30 m × 30 m

Raster dimensions:

205 × 184 pixels

## Elevation

Approximate processed DEM elevation range:

535.5 m to 719.22 m

## Slope

Approximate slope range:

0.015° to 26.51°

Mean slope:

approximately 4.32°

## Flow Direction

Generated using GRASS r.watershed.

The resulting raster uses GRASS drainage-direction codes.

## Flow Accumulation

Generated using GRASS r.watershed.

The final accumulation raster contains positive accumulation values
for downstream terrain analysis.

## Validation

All final terrain rasters were checked for matching:

- CRS
- width
- height
- pixel resolution
- affine transform
- spatial extent

Final automated validation result:

ALL TERRAIN RASTERS ALIGNED: True

## MVP Limitation

Copernicus GLO-30 is approximately 30 m resolution and is a DSM rather
than a high-resolution bare-earth municipal DEM.

It is suitable for development and demonstration of the SIH MVP
terrain-routing pipeline.

For production deployment, the terrain layer should be replaced with
higher-resolution municipal, LiDAR, or equivalent elevation data.