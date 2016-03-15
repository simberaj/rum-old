# coding: utf8
import arcpy
import numpy
import math
import random
import operator
import sys
import collections

import loaders
import common
import stats


# points in polygon - spatial join, by count - přidám id a pak procházím 1:1
#   spatial join přiřadí bodům ID segmentu
#   pak se prochází body a spočítají se počty/atributy
# points around polygon - gaussian kernel around centroid in points 
 # - counting categories (absolute, ratios)
 # - summarizing attributes
# polygons in polygon - intersector, by area
# polygons around polygon - gaussian kernel around centroid in raster
 # - counting categories (absolute, ratios)
 # - summarizing attributes
 
# kalkulátor - má už uložené zdrojové vrstvy, zavolá se metoda se segmenty a on to vyřeší
# 
 
# pt-po: point in polygon, weight 1, generator
# pt@po: in range, weight~distance, generator
# po-po: intersecting, weight by area, generator
# po@po: in range, weight~distance by mask, generator

  
class Aggregator(object):
  def __init__(self, name, field):
    self.inField = field
    self.name = name
    self.prefix = ''
  
  def reset(self, prefix=''):
    self.prefix = prefix
    self.outFields = self._outputNames()
  
  def get(self):
    return self.values
  
  def getWhere(self):
    return ''
  
  def getReadSlots(self):
    return {self.name : self.inField}
  
  def getWriteSlots(self):
    return dict(zip(self.outFields, self.outFields))
  
  def getWriteTypes(self):
    return {fld : float for fld in self.outFields}
    
    
class CategoricalAggregator(Aggregator):
  CATEGORIES = []
  
  def __init__(self, name, field, selCats=None):
    self.categories = self.CATEGORIES if selCats is None else selCats
    Aggregator.__init__(self, name, field)
    self.reset()
  
  def reset(self, prefix=''):
    Aggregator.reset(self, prefix)
    self.values = collections.defaultdict(lambda: collections.defaultdict(float))
  
  def _outputNames(self):
    names = []
    self.translation = collections.defaultdict(lambda: None) # all unknown values add to None
    for cat in self.categories:
      fldname = self.prefix + self.name + 'f' + str(cat)
      names.append(fldname)
      self.translation[cat] = fldname
    return names
  
  def getWhere(self):
    return "{} IN ({})".format(self.inField, ",".join("'" + cat + "'" if cat == str(cat) else str(cat) for cat in self.categories))
  
  def add(self, id, weight, category):
    if id >= 0:
      # print(self.name, 'adding', id, weight, category)
      try:
        self.values[id][self.translation[category]] += weight
      except TypeError: # sequence
        vals = self.values[id]
        try:
          for i in xrange(len(category)):
            vals[self.translation[category[i]]] += weight[i]
        except IndexError:
          print(category, weight)
          raise
    
  
class CategoryFractionAggregator(CategoricalAggregator):
  def get(self):
    self.normalize()
    return self.values
  
  def normalize(self):
    for id in self.values:
      vals = self.values[id]
      weightSum = sum(vals.itervalues()) - vals[None]
      for key in vals:
        vals[key] /= weightSum
  
class SummarizingAggregator(Aggregator):
  def __init__(self, name, field, functions):
    self.functions = functions
    Aggregator.__init__(self, name, field)
    self.weightedFunctions = set(self.functions).intersection(stats.WEIGHTED_FUNCTIONS)
    self.weight = bool(self.weightedFunctions)
    self.reset()
  
  def reset(self, prefix=''):
    Aggregator.reset(self, prefix)
    self.values = collections.defaultdict(list)
    if self.weight:
      self.weights = collections.defaultdict(list)
  
  def _outputNames(self):
    names = []
    self.translation = {}
    for fx in self.functions:
      fldname = self.prefix + self.name + '_' + fx.__name__
      names.append(fldname)
      self.translation[fx] = fldname
    return names
  
  def add(self, id, weight, value):
    if id >= 0: # filter out non-matches from containment etc.
      print(self.name, 'adding', id, weight, value)
      try:
        for val in value:
          self.values[id].append(val)
        if self.weight:
          for w in weight:
            self.weights[id].append(w)
      except TypeError: # single value
        self.values[id].append(value)
        if self.weight:
          self.weights[id].append(weight)
  
  def get(self):
    print(self.name, 'aggregating', self.values, self.weights if self.weight else None)
    summarized = {}
    fxitems = self.translation.items()
    for id in self.values:
      sums = summarized[id] = {}
      vals = self.values[id]
      if self.weight: wts = self.weights[id]
      for fx, fld in fxitems:
        sums[fld] = fx(vals, wts) if fx in self.weightedFunctions else fx(vals)
    return summarized
    
  
  
class FeatureCalculator(object):
  def __init__(self, name, source, aggregators):
    self.name = name
    self.source = source
    self.aggregators = aggregators
    wheres = set(agg.getWhere() for agg in self.aggregators)
    self.where = wheres.pop() if len(wheres) == 1 else ''
    print(self.name, self.source, self.where)
    for agg in self.aggregators: agg.reset(prefix=self.name)
    
  def calculate(self, segments):
    start = True
    for id, weight, values in self.load(segments):
      if start:
        if isinstance(values, dict):
          getter = lambda values, aggname: values[aggname]
        elif hasattr(values, '__iter__') and isinstance(next(iter(values)), dict):
          getter = lambda values, aggname: (item[aggname] for item in values)
        else:
          getter = lambda value, aggname: value
        start = False
      for agg in self.aggregators:
        agg.add(id, weight, getter(values, agg.name))
    self.write(segments, self.merge(agg.get() for agg in self.aggregators))
    
  @staticmethod
  def merge(dicts):
    main = next(iter(dicts)) # take first
    for d in dicts:
      # recursive update, copy everything from d to main
      for k in d:
        main[k].update(d[k])
    return main
    
  def getReadSlots(self, aggs=None):
    if aggs is None: aggs = self.aggregators
    slots = {}
    for agg in aggs:
      slots.update(agg.getReadSlots())
    return slots
    
  def getWriteSlots(self, aggs=None):
    if aggs is None: aggs = self.aggregators
    slots = {}
    for agg in aggs:
      slots.update(agg.getWriteSlots())
    return slots
    
  def getWriteTypes(self, aggs=None):
    if aggs is None: aggs = self.aggregators
    slots = {}
    for agg in aggs:
      slots.update(agg.getWriteTypes())
    return slots
    
  def write(self, segments, results):
    loaders.ObjectMarker(segments, {'id' : common.ensureIDField(segments)}, 
        self.getWriteSlots(), {}, self.getWriteTypes()).mark(results)
    
    
class ContainmentCalculator(FeatureCalculator): # počítá vnitřky
  def __init__(self, name, source, aggregators, weightField=None):
    FeatureCalculator.__init__(self, name, source, aggregators)
    self.weightField = weightField
    
  def load(self, segments):
    with common.PathManager(segments, shout=False) as pathman:
      # if the generators desire less, restrict the source
      if self.where:
        print(self.__class__.__name__, 'selecting where: ' + self.where)
        selected = pathman.tmpLayer()
        arcpy.MakeFeatureLayer_management(self.source, selected,
            common.safeQuery(self.where, self.source))
      else:
        selected = self.source
      # intersect the source and the segments
      featIdent = pathman.tmpFile()
      arcpy.Identity_analysis(selected, segments, featIdent, 'ONLY_FID')
      data = loaders.BasicReader(featIdent, self._slots(featIdent, segments)).read()
      for feat in data:
        # print(feat['id'], (1 if self.weightField is None else feat['weight'], feat))
        yield feat['id'], 1 if self.weightField is None else feat['weight'], feat
    
  def _slots(self, source, segments):
    slots = self.getReadSlots()
    slots['id'] = common.joinIDName(segments, source) # TODO
    print(slots['id'])
    if self.weightField is not None:
      slots['weight'] = self.weightField
    return slots

    
class PolygonInPolygonCalculator(ContainmentCalculator):
  def _slots(self, source, segments):
    slots = ContainmentCalculator._slots(self, source, segments)
    areaFld = common.ensureShapeAreaField(source)
    if self.weightField:
      # multiply the weights with area
      multFld = common.PathManager(source, shout=False).tmpField(source, float, delete=False)
      # do not delete the field, will be torn down with the whole layer
      arcpy.CalculateField_management(source, multFld,
          '!{}! * !{}!'.format(self.weightField, areaFld), 'PYTHON_9.3')
      self.weightField = multFld
    else:
      self.weightField = areaFld
    slots['weight'] = self.weightField
    return slots

    
class ProximityCalculator(FeatureCalculator):
  DEFAULT_RANGE_FACTOR = 5
  
  def __init__(self, name, source, aggregators):
    FeatureCalculator.__init__(self, name, source, aggregators)
    self._sourceLoaded = False
    
  def _find(self, x, y):
    raise NotImplementedError
    
  def load(self, segments):
    if not self._sourceLoaded:
      self.loadSource()
      self._sourceLoaded = True
    # convert to centroids
    self.centroids = common.PathManager(segments).tmpFile(delete=False) # delete in write
    arcpy.FeatureToPoint_management(segments, self.centroids, 'INSIDE')
    shapeSlot = loaders.SHAPE_SLOT
    ptSlots = {'id' : common.ensureIDField(self.centroids), shapeSlot : None}
    pts = loaders.BasicReader(self.centroids, ptSlots).read()
    for pt in pts:
      found = self._find(*pt[shapeSlot])
      if found:
        yield pt['id'], found[0], found[1]
  
  def write(self, segments, results):
    FeatureCalculator.write(self, self.centroids, results)
    with common.PathManager(segments, shout=False) as pathman:
      pathman.registerFile(self.centroids)
      tmpSegments = pathman.tmpFile()
      arcpy.SpatialJoin_analysis(segments, self.centroids, tmpSegments, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', '', 'CONTAINS')
      arcpy.Delete_management(segments)
      arcpy.CopyFeatures_management(tmpSegments, segments)
  
  
class GaussianProximityCalculator(ProximityCalculator):
  def __init__(self, name, source, aggregators, sd=100, range=None):
    ProximityCalculator.__init__(self, name, source, aggregators)    
    if range is None: range = sd * self.DEFAULT_RANGE_FACTOR
    self.sd = sd
    self.range = range
    self.denom = 2 * (sd ** 2)

    
class GaussianPointCalculator(GaussianProximityCalculator):
  BUCKET_SIZE = 5
  SAMPLING_COEF = 0.25
  
  def loadSource(self):
    data = []
    shapeSlot = loaders.SHAPE_SLOT
    slotsIn = self.getReadSlots()
    slotsIn[shapeSlot] = None
    for row in loaders.BasicReader(self.source, slotsIn, where=self.where).read():
      data.append(row[shapeSlot] + [row])
    self.count = len(data)
    self.cells = self._quad(data)
      
  @classmethod
  def _quad(cls, pts):
    pts.sort(key=operator.itemgetter(0))
    if len(pts) <= cls.BUCKET_SIZE:
      return pts
    else:
      # mean x value is in the middle (pts sorted by x)
      xDivI = int(len(pts) / 2.0)
      xDiv = 0.5 * (pts[xDivI][0] + pts[xDivI+1][0])
      # mean y value determined by random sampling
      tries = int(max(cls.BUCKET_SIZE, cls.SAMPLING_COEF * len(pts)))
      yDiv = sum(random.choice(pts)[1] for k in xrange(tries)) / tries
      nw, ne, sw, se = [], [], [], []
      for i in xrange(xDivI+1): # all smaller than xDiv
        (sw if pts[i][1] < yDiv else nw).append(pts[i])
      for i in xrange(xDivI+1, len(pts)):
        (se if pts[i][1] < yDiv else ne).append(pts[i])
      return (xDiv, yDiv, cls._quad(nw), cls._quad(ne), cls._quad(sw), cls._quad(se))
  
  def _traverse(self, cell, x, y):
    rg = self.range
    if len(cell) == 6: # non-leaf cell
      if x + rg > cell[0]:
        if y + rg > cell[1]:
          for res in self._traverse(cell[3], x, y): yield res
        if y - rg < cell[1]:
          for res in self._traverse(cell[5], x, y): yield res
      if x - rg < cell[0]:
        if y + rg > cell[1]:
          for res in self._traverse(cell[2], x, y): yield res
        if y - rg < cell[1]:
          for res in self._traverse(cell[4], x, y): yield res
    else: # leaf cell, count for all
      for pt in cell:
        dist = math.hypot(pt[0] - x, pt[1] - y)
        if dist <= rg:
          yield math.exp(-(dist ** 2) / self.denom), pt[2]
  
  def _find(self, x, y):
    weights = []
    values = []
    for res in self._traverse(self.cells, x, y):
      weights.append(res[0])
      values.append(res[1])
    return weights, values
  
      
class GaussianPolygonCalculator(GaussianProximityCalculator):
  def __init__(self, name, source, aggregators, resolution, *args, **kwargs):
    GaussianProximityCalculator.__init__(self, name, source, aggregators, *args, **kwargs)
    self.field = self.getReadSlots(aggregators).items()[0][1]
    self.resolution = resolution
  
  def loadSource(self):
    with common.PathManager(self.source) as pathman:
      if self.where:
        # print(self.__class__.__name__, 'selecting where: ' + self.where)
        selection = pathman.tmpLayer()
        arcpy.MakeFeatureLayer_management(self.source, selection,
            common.safeQuery(self.where, self.source))
      else:
        selection = self.source
      rast = pathman.tmpRaster()
      arcpy.PolygonToRaster_conversion(selection, self.field, rast,
          'MAXIMUM_COMBINED_AREA', '', self.resolution)
      self.raster = loaders.Raster(rast)
      self.rows, self.cols = self.raster.shape
      self.cellRange = self.raster.toCells(self.range)
      self.mask = self.createMask(self.raster.toCells(self.sd),
                                  self.cellRange)
                                  
      print(self.mask.shape)

  @staticmethod
  def createMask(sd, range):
    range = int(math.ceil(range))
    dim = 2*range + 1
    denom = 2 * (sd ** 2)
    mask = numpy.empty((dim, dim))
    for i in xrange(dim):
      for j in xrange(dim):
        dist = math.hypot(i - range, j - range)
        mask[i,j] = math.exp(-(dist ** 2) / denom)
    return mask / mask.sum()

  def _find(self, x, y):
    idx = self.raster.toIndex(x, y)
    if idx is None:
      return None
    else:
      return self._findByIndex(*idx)
    
  def _findByIndex(self, i, j):
    range = int(math.ceil(self.cellRange))
    result = []
    fullMask = True
    if i > range:
      iFrom = i - range
      maskIFrom = 0
    else:
      iFrom = 0
      maskIFrom = range - i
      fullMask = False
    if i + range < self.rows:
      iTo = i + range + 1
      maskITo = 2 * range + 1
    else:
      iTo = self.rows
      maskITo = range - i + self.rows
      fullMask = False
    if j > range:
      jFrom = j - range
      maskJFrom = 0
    else:
      jFrom = 0
      maskJFrom = range - j
      fullMask = False
    if j + range < self.cols:
      jTo = j + range + 1
      maskJTo = 2 * range + 1
    else:
      jTo = self.cols
      maskJTo = range - j + self.cols
      fullMask = False
    print(i, j, iFrom, iTo, jFrom, jTo, maskIFrom, maskITo, maskJFrom, maskJTo, self.rows, self.cols, self.mask[maskIFrom:maskITo,maskJFrom:maskJTo].shape)
    return self.mask[maskIFrom:maskITo,maskJFrom:maskJTo].flatten(), self.raster.getRect(iFrom, iTo, jFrom, jTo).flatten()
       
class RasterCalculator(ProximityCalculator):
  def loadSource(self):
    self.raster = loaders.Raster(self.source)
    
  def _find(self, x, y):
    return 1, self.raster.getByLoc(x, y)
        
    
def calculateStandard(segments, buildings, landuse):
  calculateBuildings(segments, buildings)
  calculateLandUse(segments, landuse)
  
BUILDING_HEIGHT_QUERY = '(height > 0 and height < 50) or floors <> 0'
HEIGHT_TO_FLOORS = 3.5 # TODO
AREA_FIELD = 'AREA'
CF_FIELD = 'CF'
WAGNER_FIELD = 'WAG'
BUILDING_CALC_FIELDS = [AREA_FIELD, WAGNER_FIELD, CF_FIELD]
BUILDING_FEATURE_NAME = 'building_pts'
BUILDING_FLOOR_TRANSFER_DISTANCE = 50
BUILDING_AGGREGATORS = [
  SummarizingAggregator('bsize', AREA_FIELD, (sum, stats.mean, stats.median)),
  SummarizingAggregator('bwag', WAGNER_FIELD, (stats.mean, stats.median, stats.wmean, stats.wmedian))
]

UA_CODES = [11100, 11210, 11220, 11230, 11240, 11300, 12100, 12210, 12230, 12300, 12400, 13100, 13300, 13400, 14100, 14200, 20000, 30000, 50000]
UA_CODE_FIELD = 'CODE'
UA_INT_CODE_FLD = 'CODE_INT'
LANDUSE_AGGREGATORS = [CategoryFractionAggregator('lu', UA_INT_CODE_FLD, UA_CODES)]
  
def calculateBuildings(segments, buildings):
  with common.PathManager(buildings) as pathman:
    common.progress('creating temporary building copy')
    tmpBuilds = pathman.tmpFile() # do not compromise the original file
    arcpy.CopyFeatures_management(buildings, tmpBuilds)
    # extract the features from geometry
    common.progress('computing building footprint characteristics')
    common.addFields(tmpBuilds, BUILDING_CALC_FIELDS, [float for fld in BUILDING_CALC_FIELDS])
    arcpy.CalculateField_management(tmpBuilds, AREA_FIELD, '!shape.area!', 'PYTHON_9.3')
    arcpy.CalculateField_management(tmpBuilds, CF_FIELD, '!shape.length!', 'PYTHON_9.3')
    arcpy.CalculateField_management(tmpBuilds, WAGNER_FIELD,
        '!{0}! / (2 * math.sqrt(math.pi * !{1}!)) if !{1}! > 1e-6 else 0'.format(
            CF_FIELD, AREA_FIELD),
        'PYTHON_9.3') # wagner index
    # convert to points
    common.progress('locating buildings')
    tmpPts = pathman.tmpFile()
    arcpy.FeatureToPoint_management(tmpBuilds, tmpPts, 'CENTROID')
    # calculate and assign heights
    common.progress('computing building height characteristics')
    tmpHeight = pathman.tmpFile()
    arcpy.Select_analysis(tmpBuilds, tmpHeight, BUILDING_HEIGHT_QUERY)
    # calculate approximate number of floors from height
    tmpFloors = pathman.tmpLayer()
    arcpy.MakeFeatureLayer_management(tmpHeight, tmpFloors, 'floors=0 or floors is null')
    arcpy.CalculateField_management(tmpFloors, 'floors', '!height! / {:f}'.format(HEIGHT_TO_FLOORS), 'PYTHON_9.3')
    # now join the attributes to the calculated points via spatialjoin
    common.progress('joining the characteristics to building location')
    # outBuilds = common.featurePath(pathman.getLocation(), BUILDING_FEATURE_NAME)
    outBuildPts = pathman.tmpFile()
    outBuildPolys = pathman.tmpFile()
    arcpy.SpatialJoin_analysis(tmpPts, tmpHeight, outBuildPts, 'JOIN_ONE_TO_ONE', 'KEEP_ALL', '', 'CLOSEST', '{} Meters'.format(BUILDING_FLOOR_TRANSFER_DISTANCE))
    arcpy.SpatialJoin_analysis(tmpBuilds, tmpHeight, outBuildPolys, 'JOIN_ONE_TO_ONE', 'KEEP_ALL', '', 'CLOSEST', '{} Meters'.format(BUILDING_FLOOR_TRANSFER_DISTANCE))
    # and calculate features
    common.progress('calculating building containment features')
    PolygonInPolygonCalculator('c', outBuildPolys, BUILDING_AGGREGATORS).calculate(segments)
    common.progress('calculating building proximity features')
    GaussianPointCalculator('p', outBuildPts, BUILDING_AGGREGATORS, sd=100).calculate(segments)
    
def calculateLandUse(segments, landuse):
  common.progress('converting Urban Atlas land use codes')
  common.addField(landuse, UA_INT_CODE_FLD, int)
  arcpy.CalculateField_management(landuse, UA_INT_CODE_FLD, '!' + UA_CODE_FIELD + '!', 'PYTHON_9.3')
  common.progress('calculating landuse containment features')
  PolygonInPolygonCalculator('c', landuse, LANDUSE_AGGREGATORS).calculate(segments)
  common.progress('calculating landuse proximity features')
  GaussianPolygonCalculator('p', landuse, LANDUSE_AGGREGATORS, resolution=50, sd=100).calculate(segments)
  
    
if __name__ == '__main__':
  # calculateStandard(sys.argv[1], sys.argv[2])
  calculateLandUse(sys.argv[1], sys.argv[2])
  # createBuildingCalculators(sys.argv[1])
  # import sys
  # # calc = PointDensityCalculator(sys.argv[1], sd=50, difFld='type', difs=['traffic', 'stop', 'housing', 'recreation'])
  # calc = PointAveragingCalculator(sys.argv[1], sd=300, fields=['SHAPE_AREA'])
  # # calc = LanduseFractionCalculator(sys.argv[1])
  # # print(calc.CODES)
  # # for i in range(min(calc.cols, calc.rows)):
    # # print(i, calc.calculateByIndex(i, i))
  # print('ready')
  # import cProfile
  # def main():
    # for i in range(1000):
      # x = calc.calculate(float(sys.argv[2]), float(sys.argv[3]))
    # # print(calc.i)
    # # print(calc.difs)
    # print(x)
  # cProfile.run('main()')