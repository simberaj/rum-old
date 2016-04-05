import common
import raster

with common.runtool(3) as parameters:
  dem, perc, samples = parameters
  raster.heightPercentiles(dem, perc, int(samples))