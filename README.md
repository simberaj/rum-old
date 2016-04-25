# ReconfigurableUrbanModeler
**Reconfigurable Urban Modeler** (RUM) is a framework for spatial characteristic estimation,
mainly for social indicators in urban areas.

It can be used to train machine learning models that can then be used to disaggregate coarse
data into spatially finer resolution using free ancillary data sources. As of now, the data
used by the model is mostly limited to pan-European coverage.

The framework was developed as a master's thesis at the Faculty of Science, Charles University
in Prague and licensed under the MIT License. Please report all bugs and problems to simberaj@natur.cuni.cz.

## Installation
The framework is a plug-in Python toolbox for *ArcGIS 10*. It is directly usable with it after unpacking,
no build process or installation is required.

### Dependencies
The toolbox only depends on the libraries that are bundled with ArcGIS by default (ArcPy, NumPy, and Matplotlib).

To use the General Regression (GR) machine learning model, SciPy in the version compatible with the NumPy
version you use is required. *Do not* install it using a package manager, use the Superpack installer that
is available from [SourceForge](https://sourceforge.net/projects/scipy/files/scipy/).

## How-To
The toolbox works in three main phases:
- Ancillary data download and integration.
- Creation of the modeling areas (*segments*).
- Disaggregation of the input coarse data.

The disaggregation step requires a trained machine learning model that estimates the disaggregation weights. Its
parameters are stored in a `.mod` file. There are some examples (concerning population density and rent level)
in the `models` directory. If you need a better or new one, see the *Model training* section.

### Ancillary data integration
#### Data sources used
The model uses the following data sources:
- [OpenStreetMap](http://www.openstreetmap.org). Download is automatic within the toolbox by default,
  using the Overpass API gateway.
- [Urban Atlas](http://land.copernicus.eu/local/urban-atlas). Download is possible upon free registration.
  Only available for European urban areas as of 2016. A similar layer can be used if the land use classification
  uses the Urban Atlas nomenclature.
- Digital elevation models used to model the terrain. The recommended source is
  [SRTM](http://srtm.csi.cgiar.org/SELECTION/inputCoord.asp) but any similar raster-based DEM is allowed, such as
  [ASTER](http://asterweb.jpl.nasa.gov/gdem.asp) which is available for higher latitudes as well. The download of both
  sources is possible after a free registration.

#### Integration
The `1 - Integrate` tool integrates the data sources above into the format used by the model - a handful of
*direct layers* in a single geodatabase.

The direct layers can also be supplied directly from a different source, provided they meet the attribute schema and
the coordinate reference systems match.

A single user-selected coordinate reference system is forced upon the data. It should be a projected coordinate system
(i. e., one that uses Cartesian coordinates instead of latitude and longitude), such as UTM. You may
also specify a study area to clip all the sources to.

#### Troubleshooting
When processing large areas, memory and network bandwidth may be an issue. In such cases, a lengthier process
can be applied using the tools in the toolset `1 - Integrate phases`:
- If the Overpass API takes too long to respond, use the [Geofabrik](http://download.geofabrik.de/) data extracts
  instead (the `.osm.bz2` format), unpack them and feed them into the `1b - OSM File to Direct Layers`.
- If the OSM conversion is too lengthy, use the `1b - OSM conversion phases` toolset:
  - `1ba - OSM to GeoJSON` transforms the OSM XML into a standard GeoJSON file.
  - `1bb - Split GeoJSON` splits the huge GeoJSON file into smaller parts for processing.
  - `1bc - OSM File to Direct Layers per Parts` converts the parts into direct layer chunks.
  - `1bd - Merge Workspaces` merges the part directories with direct layers to the target workspace.
- If the OSM is acquired using the steps above, the `1c - Integrate Other Sources` can be used to convert all the
  other sources into the target workspace.
  
### Modeling area (segment) creation
The tool `2 - Create Segments` delimits inhabited areas (*segments*) for which the target values are obtained.

### Model application
If you have a `.mod` file with the model definition, the direct layers and segments in a workspace,
and some data to disaggregate (in the worst case, a single value for the whole area, assigned to the area polygon),
you can run the `3 - Apply Model` to obtain a disaggregated distribution. The target values are saved into the
segment attribute table under the name that the disaggregation data attribute has, and can be easily visualised
or used further.

Do not forget to distinguish absolute (*spatially extensive*) values that grow with area, such as population count,
from relative (*spatially intensive*) values, such as population density - the internal disaggregation process is
different for these.

#### Stepwise application
The model application can also be used step-wise (in case more control over the process is desired or some errors
occur) with the tools in the `3 - Application phases` toolset:
- `3a - Prepare Model Application` intersects the segments with the disaggregation data areas.
- `3b - Calculate Features` calculates the characteristics of the physical environment of the segments from the ancillary data
  and stores them in the segment attribute table.
- `3c - Apply Model with Assigned Segments` takes the `.mod` file, computes the segment weights from the features
  and performs the disaggregation.
  
### Model training
If you do not have the `.mod` file you need, you can create it using the training routines from some high-resolution data
you have. Use the `0 - Training` toolset for that - the tool `0 - Train Model` does it all in one step.

For Model Type, use either:
- OLS for [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) estimation, or
- GR for [General Regression](http://research.vuse.vanderbilt.edu/vuwal/paul/paper/references/grnn.pdf)
  which is generally more precise.
  
#### Training data
The training data should capture the spatial distribution of the particular indicator that you wish to model
at a different location, in precision comparable to the delimited segments (which usually have the size of an
urban fabric block). Either point or polygon data may be used.

### Validation
If you wish to validate the precision of the modeled spatial distribution using a *validation dataset*
obtained a different way, use the `4 - Validate` tool from the `4 - Validation` toolset. It intersects the
segments (containing the modeled values) with the validation dataset (deemed to contain the ground
truth - real - values), performs some statistical analyses of the differences and produces a simple validation
report.

## Further description and citations
Šimbera, Jan (2016): *Modeling population from topographic data*. Master's thesis, Faculty of Science,
Charles University in Prague.
