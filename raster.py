import arcpy

class Raster:
  def __init__(self, source):
    self.source = source
    self.raster = arcpy.RasterToNumPyArray(self.source)
    self.rows, self.cols = self.raster.shape
    dsc = arcpy.Describe(self.source)
    self.dx = dsc.MeanCellWidth
    self.dy = abs(dsc.MeanCellHeight)
    # coordinates of (0, 0)
    self.x0 = dsc.extent.XMin + self.dx / 2.0
    self.y0 = dsc.extent.YMax - self.dy / 2.0
    self.bottom = dsc.extent.XMin
    self.left = dsc.extent.YMin
    
  def toIndex(self, x, y):
    j = int(round((x - self.x0) / self.dx))
    i = int(-round((y - self.y0) / self.dy))
    # print(i, j)
    # print(self.cols, self.rows)
    if i < 0 or i > self.rows or j < 0 or j > self.cols:
      return None
    else:
      return i, j

  def toCells(self, dist):
    return dist / float(self.dx)
  
  def get(self, i, j):
    return self.raster[i,j]
  
  def getRect(self, i1, i2, j1, j2):
    return self.raster[i1:i2,j1:j2]
  
  def getArray(self):
    return self.raster

  def getByLoc(self, x, y):
    idx = self.toIndex(x, y)
    if idx is None:
      return None
    else:
      return self.get(*idx)

  def alignedRaster(self, array):
    return arcpy.NumPyArrayToRaster(array, arcpy.Point(self.bottom, self.left), self.source)
      
  @property
  def shape(self):
    return self.raster.shape


def windowToDims(window):
  winx = len(window[0])
  winy = len(window)
  if winx != winy or winx % 2 != 1 or winy % 2 != 1:
    raise IndexError, 'invalid window size: odd-sized square required'
  return (range(winx), range(winy), int(winy / 2))


def getMargins(array, windim):
  arrx = len(array)
  arry = len(array[0])
  return [(range(arrx), range(windim + 1)),
          (range(arrx), range(arry - windim - 1, arry)),
          (range(windim + 1), range(windim, arry - windim)),
          (range(arrx - windim - 1, arrx), range(windim, arry - windim))]
  
def window(inarray, window):
  import arcpy, numpy
  winrangei, winrangej, windim = windowToDims(window)
  outarray = numpy.empty_like(inarray)
  # margins (indexerror may occur)
  for irange, jrange in getMargins(inarray, windim):
    for i in irange:
      for j in jrange:
        winval = 0
        used = 0
        for k in winrangei:
          for l in winrangej:
            try: # ignore indexerror
              winval += inarray[i+k-windim][j+l-windim] * window[k][l]
              used += window[k][l]
            except IndexError:
              pass
        outarray[i][j] = winval / float(used) # convert to same scale, ignoring indexerrors
  # central cells (no indexerror will occur)
  for i in range(windim, len(inarray) - windim):
    for j in range(windim, len(inarray[i]) - windim):
      winval = 0
      for k in winrangei:
        for l in winrangej:
          winval += inarray[i+k-windim][j+l-windim] * window[k][l]
      outarray[i][j] = winval
  return outarray

def medianfilter(inarray, windim):
  import arcpy, numpy
  winrangei = range(windim * 2 + 1)
  winrangej = range(windim * 2 + 1)
  medianloc = 2 * (windim ** 2 + windim)
  outarray = numpy.empty_like(inarray)
  # central cells (no indexerror will occur) - optimized for speed
  for i in range(windim, len(inarray) - windim):
    for j in range(windim, len(inarray[i]) - windim):
      winvals = []
      for k in winrangei:
        for l in winrangej:
          winvals.append(inarray[i+k-windim][j+l-windim])
      winvals.sort()
      outarray[i][j] = winvals[medianloc]
  # margins (indexerror may occur)
  for irange, jrange in getMargins(inarray, windim):
    for i in irange:
      for j in jrange:
        winvals = []
        for k in winrangei:
          for l in winrangej:
            try: # ignore indexerror
              winvals.append(inarray[i+k-windim][j+l-windim])
            except IndexError:
              pass
        try:
          outarray[i][j] = median(winvals)
        except IndexError:
          raise IndexError, 'invalid index: %i %i within %i %i' % (i, j, len(inarray), len(inarray[0]))
  return outarray

def median(values):
  if len(values) % 2 == 0:
    return (values[int(len(values) / 2) - 1] + values[int(len(values) / 2)]) / 2.0
  else:
    return values[int(len(values) / 2)]
  
  
def heightPercentiles(demFile, outFile, samples=1000):
  import bisect, numpy
  dem = Raster(demFile)
  demArray = dem.getArray()
  percentileKeys = list(numpy.arange(samples) * 100.0 / samples) # 0 to 100
  percentileVals = numpy.percentile(demArray, percentileKeys)
  percentileArray = numpy.empty(demArray.shape)
  rows, cols = percentileArray.shape
  factor = 100.0 / samples
  for i in xrange(rows):
    for j in xrange(cols):
      percentileArray[i,j] = bisect.bisect_left(percentileVals, demArray[i,j]) * factor
  percentiles = dem.alignedRaster(percentileArray)
  percentiles.save(outFile)
  
# TMP_GDB_NAME = 'tmp_mosaic.gdb'
# TMP_MOSAIC_NAME = 'tmp_mos'
  
# def merge(tiles, outPath, bands=1, bitdepth='16_BIT_SIGNED'):
  # import common, os
  # if common.isInDatabase(outPath):
    # workspace = os.path.dirname(outPath)
  # else:
    # folder = common.folder(outPath)
    # arcpy.CreateFileGDB_management(folder, TMP_GDB_NAME)
    # workspace = os.path.join(folder, TMP_GDB_NAME)
  # arcpy.CreateMosaicDataset_management(workspace, TMP_MOSAIC_NAME, arcpy.Describe(tiles[0]).spatialReference, bands, bitdepth)
  # mosaic = common.rasterPath(workspace, TMP_MOSAIC_NAME)
  # arcpy.AddRastersToMosaicDataset_management(mosaic, 'Raster Dataset', tiles)
  
