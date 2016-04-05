import regression
import arcpy
import common
import numpy

NODATA = -2147483648

with common.runtool(3) as params:
  modelrast, realrast, outfile = params
  models = arcpy.RasterToNumPyArray(modelrast, nodata_to_value=NODATA).flatten()
  reals = arcpy.RasterToNumPyArray(realrast, nodata_to_value=NODATA).flatten()
  condition = (models != NODATA) & (reals != NODATA)
  val = regression.Validator(
    numpy.extract(condition, models), 
    numpy.extract(condition, reals))
  val.validate()
  val.output(outfile)