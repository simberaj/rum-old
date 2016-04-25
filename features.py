# coding: utf8
import arcpy
import numpy
import math
import random
import operator
import sys
import os
import collections

import loaders
import common
import stats
import subdivide_polygons
import raster



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
 
# extraktor - má už uložené zdrojové vrstvy, zavolá se metoda se segmenty a on to vyřeší
# 
 
# pt-po: point in polygon, weight 1, generator
# pt@po: in range, weight~distance, generator
# po-po: intersecting, weight by area, generator
# po@po: in range, weight~distance by mask, generator

GLOBAL_PREFIX = 'FF'
  
class Aggregator(object):
  def __init__(self, name, field, selField=None):
    self.inField = field
    self.selField = field if selField is None else selField
    self.name = name
    self.prefix = ''
  
  def reset(self, prefix=''):
    self.prefix = GLOBAL_PREFIX + prefix
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
  
  def setSelectionField(self, fld):
    self.selField = fld
  
  def getExtensiveIndicators(self):
    return []
    
    
class CategoricalAggregator(Aggregator):
  CATEGORIES = []
  MARKER = 'c'
  
  def __init__(self, name, field, selCats=None, selField=None):
    self.categories = self.CATEGORIES if selCats is None else selCats
    Aggregator.__init__(self, name, field, selField)
    self.reset()
  
  def reset(self, prefix=''):
    Aggregator.reset(self, prefix)
    self.values = collections.defaultdict(lambda: collections.defaultdict(float))
  
  def _outputNames(self):
    names = []
    self.translation = collections.defaultdict(lambda: None) # all unknown values add to None
    common = self.prefix + self.name + self.MARKER
    for cat in self.categories:
      fldname = common + str(cat)
      names.append(fldname)
      self.translation[cat] = fldname
    return names
  
  def getWhere(self):
    return "{} IN ({})".format(self.selField, ",".join("'" + cat + "'" if cat == str(cat) else str(cat) for cat in self.categories))
  
  def add(self, id, weight, category):
    if id >= 0:
      # print(self.name, 'adding', id, weight, category)
      try:
        self.values[id][self.translation[category]] += weight
      except TypeError: # sequence
        vals = self.values[id]
        for i in xrange(len(category)):
          vals[self.translation[category[i]]] += weight[i]
  
  def getExtensiveIndicators(self):
    return self.outFields
    
  
class CategoryFractionAggregator(CategoricalAggregator):
  MARKER = 'f'

  def get(self):
    self.normalize() # in-place, not easy to refactor!
    return self.values
  
  def getExtensiveIndicators(self):
    return [] # normalized by fractioning
  
  def normalize(self):
    for vals in self.values.itervalues():
      weightSum = sum(vals.itervalues()) - vals[None]
      if weightSum:
        for key in vals:
          vals[key] /= weightSum
        
class CategoryPrefixFractionAggregator(CategoryFractionAggregator):
  # instead of specifying precise categories, specifies just prefixes
  # useful for UA as different versions go into different levels of detail

  def getWhere(self):
    return ' OR '.join("({} LIKE '{}%')".format(self.selField, cat) for cat in self.categories)
  
  def normalize(self):
    CategoryFractionAggregator.normalize(self)
    # common.debug(self.values)
    allkeys = set()
    # find out all categories encountered in the data
    for vals in self.values.itervalues():
      allkeys.update(vals.iterkeys())
    # build a translation dict
    translateCat = {}
    strCats = [str(cat) for cat in self.categories]
    for key in allkeys:
      for i in range(len(strCats)):
        if str(key).startswith(strCats[i]):
          translateCat[key] = self.translation[self.categories[i]]
          break
    # translate all the values
    for id in self.values:
      oldvals = self.values[id]
      newvals = collections.defaultdict(float)
      for key in translateCat:
        newvals[translateCat[key]] += oldvals[key]
      self.values[id] = newvals
    
  def add(self, id, weight, category):
    if id >= 0:
      # print(self.name, 'adding', id, weight, category)
      try:
        self.values[id][category] += weight
      except TypeError: # sequence
        vals = self.values[id]
        for i in xrange(len(category)):
          vals[category[i]] += weight[i]
      
        
class CategoryMergingAggregator(CategoricalAggregator):
  MARKER = 'p'

  def __init__(self, name, field, mergeRules={}, selField=None):
    self.mergeRules = mergeRules
    CategoricalAggregator.__init__(self, name, field,
        [item for sublist in mergeRules.itervalues() for item in sublist], selField)

  def _outputNames(self):
    # init translation and throw away the result
    CategoricalAggregator._outputNames(self)
    self.prefixAll = self.prefix + self.name + self.MARKER
    self.merges = {(self.prefixAll + toName) : [(self.prefixAll + fromName) for fromName in fromNames] for toName, fromNames in self.mergeRules.iteritems()}
    return self.merges.keys()
    
  def get(self):
    self.merge()
    return self.values
  
  def merge(self):
    merges = self.merges.items()
    for id, oldvals in self.values.iteritems():
      newvals = collections.defaultdict(float)
      for target, sources in merges:
        newvals[target] = sum(oldvals[source] for source in sources)
      self.values[id] = newvals
  
  
class SummarizingAggregator(Aggregator):
  def __init__(self, name, field, functions, selField=None):
    self.functions = functions
    Aggregator.__init__(self, name, field, selField)
    self.weightedFunctions = set(self.functions).intersection(stats.WEIGHTED_FUNCTIONS)
    self.weight = bool(self.weightedFunctions)
    self.reset()
  
  def reset(self, prefix=''):
    Aggregator.reset(self, prefix)
    self.values = collections.defaultdict(list)
    if self.weight:
      self.weights = collections.defaultdict(list)
    self.sums = collections.defaultdict(lambda: collections.defaultdict(float)) # summarizing right away
    self.lastID = None
  
  def _outputNames(self):
    names = []
    self.translation = {}
    for fx in self.functions:
      fldname = self.prefix + self.name + '_' + fx.__name__
      names.append(fldname)
      self.translation[fx] = fldname
    self.transItems = self.translation.items()
    return names
  
  def add(self, id, weight, value):
    if id >= 0: # filter out non-matches from containment etc.
      if id != self.lastID:
        self.summarize(self.lastID)
        self.lastID = id
      # print(self.name, 'adding', id, weight, value)
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
    # print(self.name, 'aggregating', self.values, self.weights if self.weight else None)
    remain = list(self.values.keys())
    for id in remain: # aggregate the remaining
      self.summarize(id)
    return self.sums
  
  def summarize(self, id):
    values = self.values[id]
    del self.values[id]
    if self.weight:
      weights = self.weights[id]
      del self.weights[id]
    if values:
      sumdict = collections.defaultdict(float)
      for fx, fld in self.transItems:
        sumdict[fld] = fx(values, weights) if fx in self.weightedFunctions else fx(values)
      self.sums[id] = sumdict
      
  def getExtensiveIndicators(self):
    # everything that is summed is extensive
    return [fld for fx, fld in self.transItems if fx in (sum, stats.wsum)]
    
  
  
class FeatureExtractor(object):
  def __init__(self, name, source, aggregators):
    self.name = name
    self.source = source
    self.aggregators = aggregators
    wheres = set(agg.getWhere() for agg in self.aggregators)
    self.where = wheres.pop() if len(wheres) == 1 else ''    
    
  def extract(self, segments):
    for agg in self.aggregators:
      agg.reset(prefix=self.name)
    start = True
    areas = {}
    for id, weight, area, values in self.load(segments):
      try:
        if start:
          if isinstance(values, dict):
            getter = lambda values, aggname: values[aggname]
          elif hasattr(values, '__iter__') and isinstance(next(iter(values)), dict):
            getter = lambda values, aggname: [item[aggname] for item in values]
          else:
            getter = lambda value, aggname: value
          start = False
        for agg in self.aggregators:
          agg.add(id, weight, getter(values, agg.name))
      except StopIteration:
        continue
      areas[id] = area
    self.write(segments, self.normalize(self.merge(agg.get() for agg in self.aggregators), areas))
    return self.names()
  
  def names(self):
    return self.getWriteSlots().values()
    
  @staticmethod
  def merge(dicts):
    main = next(iter(dicts)) # take first
    for d in dicts:
      # recursive update, copy everything from d to main
      for k in d:
        main[k].update(d[k])
    return main
  
  def normalize(self, main, areas):
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
    # common.debug(segments, results)
    loaders.ObjectMarker(segments, {'id' : common.ensureIDField(segments)}, 
        self.getWriteSlots(), {}, self.getWriteTypes()).mark(results)
    
    
class ContainmentExtractor(FeatureExtractor): # počítá vnitřky
  def __init__(self, name, source, aggregators, weightField=None):
    FeatureExtractor.__init__(self, name, source, aggregators)
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
      # make the area available for the normalization
      areaFld = pathman.tmpField(segments, float)
      common.calcField(segments, areaFld, '!shape.area!')
      # intersect the source and the segments
      featIdent = pathman.tmpFile()
      arcpy.Identity_analysis(selected, segments, featIdent, 'ONLY_FID')
      # add the area field from the segments
      arcpy.JoinField_management(featIdent, common.joinIDName(segments, featIdent), segments, common.ensureIDField(segments), areaFld)
      data = loaders.BasicReader(featIdent, self._slots(featIdent, segments, areaFld)).read()
      data.sort(key=operator.itemgetter('id')) # sort by feature id so the aggregator can summarize right away
      for feat in data:
        # print(feat['id'], (1 if self.weightField is None else feat['weight'], feat))
        yield feat['id'], 1 if self.weightField is None else feat['weight'], feat['area'], feat
  
  def normalize(self, mainDict, areas):
    extensive = self.getExtensiveIndicators()
    # common.debug('normalizing', extensive)
    if extensive:
      for id, valueDict in mainDict.iteritems():
        for ind in extensive:
          valueDict[ind] /= areas[id]
    return mainDict
  
  def getExtensiveIndicators(self):
    inds = []
    for agg in self.aggregators:
      inds.extend(agg.getExtensiveIndicators())
    return inds
    
  def _slots(self, source, segments, areaFld=None):
    slots = self.getReadSlots()
    slots['id'] = common.joinIDName(segments, source)
    if self.weightField is not None:
      slots['weight'] = self.weightField
    if areaFld is not None:
      slots['area'] = areaFld
    # common.debug(slots, common.fieldList(source))
    return slots

    
class PolygonInPolygonExtractor(ContainmentExtractor):
  def _slots(self, source, segments, areaFld=None):
    slots = ContainmentExtractor._slots(self, source, segments, areaFld)
    areaFld = common.ensureShapeAreaField(source)
    if self.weightField:
      # multiply the weights with area
      multFld = common.PathManager(source, shout=False).tmpField(source, float)
      # do not delete the field, will be torn down with the whole layer
      arcpy.CalculateField_management(source, multFld,
          '!{}! * !{}!'.format(self.weightField, areaFld), 'PYTHON_9.3')
      self.weightField = multFld
    else:
      self.weightField = areaFld
    slots['weight'] = self.weightField
    return slots

    
class ProximityExtractor(FeatureExtractor):
  DEFAULT_RANGE_FACTOR = 5
  
  def __init__(self, name, source, aggregators):
    FeatureExtractor.__init__(self, name, source, aggregators)
    self._sourceLoaded = False
    
  def _find(self, x, y):
    raise NotImplementedError
    
  def load(self, centroids):
    if not self._sourceLoaded:
      self.loadSource()
      self._sourceLoaded = True
    # convert to centroids
    # self.centroids = common.PathManager(segments).tmpFile(delete=False) # delete in write
    # arcpy.FeatureToPoint_management(segments, self.centroids, 'INSIDE')
    shapeSlot = loaders.SHAPE_SLOT
    ptSlots = {'id' : common.ensureIDField(centroids), shapeSlot : None}
    for pt in loaders.BasicReader(centroids, ptSlots).read():
      found = self._find(*pt[shapeSlot])
      if found:
        yield pt['id'], found[0], 1, found[1]
  
  
class GaussianProximityExtractor(ProximityExtractor):
  def __init__(self, name, source, aggregators, sd=100, range=None):
    ProximityExtractor.__init__(self, name, source, aggregators)    
    if range is None: range = sd * self.DEFAULT_RANGE_FACTOR
    self.sd = sd
    self.range = range
    self.denom = 2 * (sd ** 2)

    
class GaussianPointExtractor(GaussianProximityExtractor):
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
      # mean y value determined by random sampling from a quarter of points
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
  
      
class GaussianPolygonExtractor(GaussianProximityExtractor):
  def __init__(self, name, source, aggregators, resolution, *args, **kwargs):
    GaussianProximityExtractor.__init__(self, name, source, aggregators, *args, **kwargs)
    self.field = self.getReadSlots(aggregators).items()[0][1]
    self.resolution = resolution
  
  def loadSource(self):
    with common.PathManager(self.source, shout=False) as pathman:
      if self.where:
        # common.debug(self.__class__.__name__, 'selecting where: ' + self.where)
        selection = pathman.tmpLayer()
        arcpy.MakeFeatureLayer_management(self.source, selection,
            common.safeQuery(self.where, self.source))
      else:
        selection = self.source
      rast = pathman.tmpRaster()
      arcpy.PolygonToRaster_conversion(selection, self.field, rast,
          'MAXIMUM_COMBINED_AREA', '', self.resolution)
      self.raster = raster.Raster(rast)
      self.rows, self.cols = self.raster.shape
      self.cellRange = self.raster.toCells(self.range)
      self.mask = self.createMask(self.raster.toCells(self.sd),
                                  self.cellRange)
      # print(self.mask.shape)

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
       
class RasterExtractor(ProximityExtractor):
  def loadSource(self):
    self.raster = loaders.Raster(self.source)
    
  def _find(self, x, y):
    return 1, self.raster.getByLoc(x, y)
        
      
      
class CalculationLayer:
  AUXILIARY_LAYERS = []
  descriptions = []

  def addAuxiliarySource(self, name, aux):
    if not hasattr(self, 'auxiliary'):
      self.auxiliary = {}
    self.auxiliary[name] = aux
  
  def input(self, source):
    self.source = source
    return self
  
  def hasSource(self):
    return hasattr(self, 'source')
  
  def __enter__(self):
    self.pathman = common.PathManager(self.source, shout=False)
    self.precalculate()
  
  def __exit__(self, *args):
    self.pathman.exit()
    
  def getName(self):
    return self.NAME
  
  def getSource(self):
    if self.hasSource():
      return self.source
    else:
      raise ValueError, 'source for layer {} missing'.format(self.NAME)
  
  def calculate(self, segments, points):
    polyfeats = []
    ptfeats = []
    for i in range(len(self.extractors)):
      extr = self.extractors[i]
      desc = self.descriptions[i] if i < len(self.descriptions) else None
      if desc:
        common.progress(' '.join(['calculating', desc, 'features']))
      if isinstance(extr, ProximityExtractor):
        ptfeats.extend(extr.extract(points))
      else:
        polyfeats.extend(extr.extract(segments))
    return polyfeats, ptfeats
  
  def sourceField(self, fld, source=None):
    source = source if source else self.source
    if fld.endswith('*'):
      fldList = common.fieldList(source)
      # common.debug(fldList)
      for srcFld in fldList:
        if srcFld.startswith(fld[:-1]):
          return srcFld
    else:
      return fld
  
  def names(self):
    for extr in self.extractors:
      for name in extr.names():
        yield name
 
 
class BuildingLayer(CalculationLayer):
  AREA_FIELD = 'AREA'
  # CF_FIELD = 'CF'
  WAGNER_FIELD = 'WAG'
  # CALC_FIELDS = [AREA_FIELD, WAGNER_FIELD, CF_FIELD]
  NAME = 'buildings'

  def __init__(self, config):
    # common.debug(config)
    self.heightQry = config['height_query'].format(config['max_height'])
    self.floorHeight = config['floor_height']
    self.floorTransferDist = config['floor_transfer_distance']
    self.sd = config['sd']
    self.aggregators = [
      SummarizingAggregator('bsize', self.AREA_FIELD, (sum, stats.mean, stats.median)),
      SummarizingAggregator('bwag', self.WAGNER_FIELD, (stats.mean, stats.median, stats.wmean, stats.wmedian))
    ]
    self.descriptions = ['buildings', 'building proximity']
  
  def precalculate(self):
    # common.progress('creating temporary building copy')
    # tmpBuilds = self.pathman.tmpFile() # do not compromise the original file
    # arcpy.CopyFeatures_management(self.source, tmpBuilds)
    # extract the features from geometry
    self.computeFootprint(self.source)
    # convert to points
    common.progress('locating buildings')
    tmpPts = self.pathman.tmpFile()
    arcpy.FeatureToPoint_management(self.source, tmpPts, 'INSIDE')
    # calculate and assign heights
    # tmpHeight = self.computeHeight(tmpBuilds)
    # now join the attributes to the calculated points via spatialjoin
    # common.progress('joining the characteristics to building location')
    # calcPoints = self.pathman.tmpFile()
    # calcPolygons = self.pathman.tmpFile()
    # arcpy.SpatialJoin_analysis(tmpPts, tmpHeight, calcPoints, 'JOIN_ONE_TO_ONE', 'KEEP_ALL', '', 'CLOSEST', '{} Meters'.format(self.floorTransferDist))
    # arcpy.SpatialJoin_analysis(tmpBuilds, tmpHeight, calcPolygons, 'JOIN_ONE_TO_ONE', 'KEEP_ALL', '', 'CLOSEST', '{} Meters'.format(self.floorTransferDist))
    self.extractors = [
      PolygonInPolygonExtractor('c', self.source, self.aggregators),
      GaussianPointExtractor('p', tmpPts, self.aggregators, sd=self.sd)
    ]
  
  # def computeHeight(self, tmpBuilds):
    # common.progress('computing building height characteristics')
    # tmpHeight = self.pathman.tmpFile()
    # arcpy.Select_analysis(tmpBuilds, tmpHeight, self.heightQry)
    # # calculate approximate number of floors from height
    # tmpFloors = self.pathman.tmpLayer()
    # arcpy.MakeFeatureLayer_management(tmpHeight, tmpFloors, 'floors=0 or floors is null')
    # arcpy.CalculateField_management(tmpFloors, 'floors', '!height! / {:f}'.format(self.floorHeight), 'PYTHON_9.3')
    # return tmpHeight
  
  def computeFootprint(self, source):
    common.progress('computing building footprint characteristics')
    # common.addFields(tmpBuilds, self.CALC_FIELDS, [float] * len(self.CALC_FIELDS))
    # even default fields recreated because we transfer them on to points and such
    common.copyField(source, common.ensureShapeAreaField(source), self.AREA_FIELD)
    common.calcField(source, self.WAGNER_FIELD, 
      '!{0}! / (2 * math.sqrt(math.pi * !{1}!)) if !{1}! > 1e-6 else 0'.format(
            common.ensureShapeLengthField(source), self.AREA_FIELD), float)
    # arcpy.CalculateField_management(tmpBuilds, self.AREA_FIELD, '!shape.area!', 'PYTHON_9.3')
    # arcpy.CalculateField_management(tmpBuilds, self.CF_FIELD, '!shape.length!', 'PYTHON_9.3')
    # arcpy.CalculateField_management(tmpBuilds, self.WAGNER_FIELD,
        # '!{0}! / (2 * math.sqrt(math.pi * !{1}!)) if !{1}! > 1e-6 else 0'.format(
            # self.CF_FIELD, self.AREA_FIELD),
        # 'PYTHON_9.3') # wagner index
   
class LandUseLayer(CalculationLayer):
  NAME = 'landuse'

  def __init__(self, config):
    self.intCode = config['int_code_field']
    self.code = config['code_field']
    self.resolution = config['resolution']
    self.sd = config['sd']
    self.aggregator = CategoryPrefixFractionAggregator('lu', self.intCode, config['codes'], selField=self.code)
    self.descriptions = ['land use', 'neighbourhood land use']
  
  def precalculate(self):
    common.addField(self.source, self.intCode, int)
    realCodeFld = self.sourceField(self.code)
    self.aggregator.setSelectionField(realCodeFld)
    arcpy.CalculateField_management(self.source, self.intCode, '!' + realCodeFld + '!', 'PYTHON_9.3')
    self.extractors = [
      PolygonInPolygonExtractor('c', self.source, [self.aggregator]),
      GaussianPolygonExtractor('p', self.source, [self.aggregator], resolution=self.resolution, sd=self.sd)
    ]
    
class POILayer(CalculationLayer):
  NAME = 'poi'

  def __init__(self, config):
    self.aggregator = CategoryMergingAggregator('', config['field'], config['categories'])
    self.sd = config['sd']
    self.descriptions = ['functional use', 'neighbourhood functional use']
 
  def precalculate(self):
    self.extractors = [
      ContainmentExtractor('c', self.source, [self.aggregator]),
      GaussianPointExtractor('p', self.source, [self.aggregator], sd=self.sd)
    ]
    
class TransportLayer(CalculationLayer):
  NAME = 'transport'
  FEATURE = 'FFacc'
  AUXILIARY_LAYERS = [LandUseLayer]
  ONEWAY_FIELD = 'oneway'
  LENGTH_FIELD = 'length'
  
  def __init__(self, config):
    self.typeField = config['type_field']
    self.levelField = config['level_field']
    self.speeds = collections.defaultdict(lambda: collections.defaultdict(float),
      {key : collections.defaultdict(float, subdict)
          for key, subdict in config['speeds'].iteritems()})
    self.tolerance = config['tolerance']
    self.builtupQuery = config['builtup_landuse_query']
    self.codeField = config['landuse_code_field']
    self.bufferDist = config['builtup_buffer']
  
  def names(self):
    return [self.FEATURE]
  
  def precalculate(self):
    tgtDir = os.path.join(self.pathman.getFolder(), 'network')
    if os.path.isdir(tgtDir):
      common.message('Network dataset already exists, skipping')
      self.net = os.path.join(tgtDir, 'rum_nd.nd')
    else:
      common.progress('intersecting lines to routable')
      import lines_to_routable
      routable = self.pathman.tmpFC()
      lines_to_routable.linesToRoutable(self.source, routable, levelFld=self.levelField, groundLevel=0, transferFlds=[self.typeField, self.ONEWAY_FIELD])
      common.progress('intersecting with builtup areas')
      intraroads = self.pathman.tmpFC()
      intravilan = self.builtupBuffer(self.auxiliary[LandUseLayer.NAME])
      arcpy.Identity_analysis(routable, intravilan, intraroads, 'ONLY_FID', self.tolerance)
      self.calcFields(intraroads, common.joinIDName(intravilan, intraroads))
      self.prepareNetwork(intraroads, tgtDir)
  
  def builtupBuffer(self, lu):
    common.progress('finding builtup areas')
    builtup = self.pathman.tmpLayer()
    arcpy.MakeFeatureLayer_management(lu, builtup, common.safeQuery(
      self.builtupQuery.format(self.sourceField(self.codeField, lu)), lu))
    common.progress('buffering builtup areas')
    buffer = self.pathman.tmpFC()
    arcpy.Buffer_analysis(builtup, buffer, self.bufferDist, '', '', 'ALL')
    return buffer
  
  def calcFields(self, intravilan, builtupFIDFld):
    common.progress('determining intravilan lines')
    common.calcField(intravilan, 'builtup', '!{}! > -1'.format(builtupFIDFld), int)
    common.progress('calculating speeds')
    common.debug(u'speedDict = ' + self.speedDictStr(self.speeds))
    common.debug('speedDict[str(!builtup!)][!{}!]'.format(self.typeField))
    common.calcField(intravilan, 'speed', 'speedDict[str(!builtup!)][!{}!]'.format(self.typeField), float, prelogic=(u'import collections;speedDict = ' + self.speedDictStr(self.speeds)))
    common.progress('calculating length')
    common.copyField(intravilan, common.ensureShapeLengthField(intravilan), self.LENGTH_FIELD)
    common.progress('calculating time')
    common.calcField(intravilan, 'time', '!{}! / !speed! * 3.6'.format(self.LENGTH_FIELD), float)
  
  @staticmethod
  def speedDictStr(spdict):
    return ('collections.defaultdict(lambda: collections.defaultdict(float), ' + 
        unicode({key : dict(value) for key, value in spdict.iteritems()}).replace(
          '{', 'collections.defaultdict(float, {').replace(
              '}', '})') + ')')
  
  def prepareNetwork(self, lines, tgtDir):
    common.progress('preparing transport network')
    os.mkdir(tgtDir)
    self.net = self.copyNetwork(tgtDir, 'rum_nd')
    common.progress('filling transport network')
    arcpy.Append_management(lines, common.featurePath(tgtDir, 'transport1'),
      'NO_TEST')
    common.progress('building transport network')
    # common.debug(self.net)
    arcpy.BuildNetwork_na(self.net)
  
  @staticmethod
  def copyNetwork(tgtFolder, ndName):
    # import shutil
    srcND = os.path.join(os.path.dirname(__file__), 'network', ndName + '.nd')
    tgtND = os.path.join(tgtFolder, ndName + '.nd')
    arcpy.Copy_management(srcND, tgtND)
    return tgtND
    # common.debug(srcND, tgtND)
    # if os.path.isdir(tgtND):
      # shutil.rmtree(tgtND)
    # shutil.copytree(srcND, tgtND)
    # for line in lineNames + [(ndName + '_Junctions')]:
      # arcpy.CopyFeatures_management(common.featurePath(srcFolder, line), common.featurePath(tgtFolder, line))
    # return tgtND
    
  def calculate(self, segments, points):
    import accessibility
    common.progress('indexing inputs')
    self.index()
    accessibility.TOLERANCE = '200 Meters'
    accessibility.accessibility(points, self.net, 'time', accFld=self.FEATURE)
    return [], [self.FEATURE]
  
  def index(self):
    wsp = common.location(self.net)
    for fc in common.listLayers(wsp):
      # common.debug('indexing', fc)
      arcpy.AddSpatialIndex_management(common.featurePath(wsp, fc))
    
    
class RasterCalculationLayer(CalculationLayer):
  def calculate(self, segments, points):
    common.progress('extracting raster values to points')
    arcpy.sa.ExtractMultiValuesToPoints(points, zip(self.rasters, self.FEATURES), 'NONE')
    return [], self.FEATURES

  def names(self):
    return self.FEATURES
  
class BarrierLayer(RasterCalculationLayer):
  NAME = 'barrier'
  FEATURES = ['FFrepel']
  
  def __init__(self, config):
    self.field = config['weight_field']
    self.cellSize = config['cellsize']
    self.searchRadius = config['radius']
  
  def precalculate(self):
    arcpy.CheckOutExtension('Spatial')
    common.progress('calculating barrier density')
    self.density = self.pathman.tmpRaster()
    arcpy.sa.LineDensity(self.source, self.field, self.cellSize, self.searchRadius, 'SQUARE_KILOMETERS').save(self.density)
    self.rasters = [self.density]

class DEMLayer(RasterCalculationLayer):
  NAME = 'dem'
  FEATURES = ['FFslope', 'FFwest', 'FFsouth', 'FFhperc']
  
  def __init__(self, config):
    self.smoothing = config['smoothing']
  
  def precalculate(self):
    arcpy.CheckOutExtension('Spatial')
    common.progress('calculating terrain features')
    common.progress('smoothing raster')
    smoothed = self.pathman.tmpRaster()
    arcpy.sa.FocalStatistics(self.source, arcpy.sa.NbrCircle(self.smoothing, 'MAP'), 'MEDIAN', 'DATA').save(smoothed) # smoothing in map units (meters), DATA = ignore nodata in neighbourhood
    common.progress('calculating terrain slope')
    self.slope = self.pathman.tmpRaster()
    arcpy.sa.Slope(smoothed, 'DEGREE').save(self.slope)
    common.progress('calculating terrain aspect')
    asp = arcpy.sa.Aspect(smoothed)
    self.westerness = self.pathman.tmpRaster()
    (abs((asp + 90) % 360 - 180) / 180.0).save(self.westerness)
    self.southness = self.pathman.tmpRaster()
    (1 - abs(asp - 180) / 180.0).save(self.southness)
    common.progress('calculating terrain height percentiles')
    self.percentiles = self.pathman.tmpRaster()
    import raster
    raster.heightPercentiles(smoothed, self.percentiles)
    self.rasters = [self.slope, self.westerness, self.southness, self.percentiles]
  
    
    
class FeatureCalculator:
  DEFAULT_LAYERS = [BuildingLayer, LandUseLayer, POILayer, TransportLayer, BarrierLayer, DEMLayer]
  # DEFAULT_LAYERS = [BarrierLayer]
  AREA_FIELD = 'WT_AREA'

  def __init__(self, layerClasses, config):
    self.layers = {}
    for cls in layerClasses:
      # common.debu
      lyr = cls(config['layers'].get(cls.NAME, {}))
      self.layers[lyr.getName()] = lyr
    self.configure(config)
  
  def configure(self, config):
    self.minArea = config['minarea']
    self.maxArea = config['maxarea']
    self.maxWagner = config['maxwagner']
    self.directFields = config['direct_fields']
    self.exact = {}
    for fld in self.directFields:
      self.exact[fld] = not fld.endswith('*')
  
  def getDirectFields(self, layer):
    direct = []
    fieldList = common.fieldList(layer)
    for fld in self.directFields:
      if self.exact[fld]:
        direct.append(fld)
      else:
        for matchFld in fieldList:
          if matchFld.startswith(fld[:-1]):
            direct.append(matchFld)
            break
    return direct
  
  def input(self, sources):
    for layerName, source in sources.iteritems():
      if layerName in self.layers:
        self.layers[layerName].input(source)
      else:
        common.warning('input layer {} unused'.format(layerName))
    for layer in self.layers.itervalues():
      if layer.AUXILIARY_LAYERS:
        for auxLayerClass in layer.AUXILIARY_LAYERS:
          layer.addAuxiliarySource(auxLayerClass.NAME, sources[auxLayerClass.NAME])

  def names(self):
    nameList = []
    for layer in self.layers.itervalues():
      nameList.extend(layer.names())
    return nameList

  def createSegmentCentroids(self, segments, centroids):
    with common.PathManager(segments, shout=False) as pathman:
      # create representative points for extractors that operate on points
      subdiv = pathman.tmpFile()
      # common.debug(segments, common.fieldTypeList(segments), self.directFields, self.getDirectFields(segments))
      subdivide_polygons.subdivide(segments, subdiv, self.maxArea, self.maxWagner, self.minArea, self.getDirectFields(segments))
      common.progress('creating segment centroids')
      common.clearFields(subdiv)
      common.copyField(subdiv, common.ensureShapeAreaField(subdiv), self.AREA_FIELD)
      arcpy.FeatureToPoint_management(subdiv, centroids, 'INSIDE')
    
    
  def calculate(self, segments, centroids=None):
    # create representative points for extractors that operate on points
    if centroids is None:
      centroids = createSegmentCentroids(segments,
          common.addFeatureExt(os.path.splitext(segments)[0] + '_centroids'))
    # calculate the features
    polyFeats, ptFeats = self._calcFeatures(segments, centroids)
    # aggregate the ptFeats from points to segments (mean, weighted by area repres. by point)
    self.aggregate(segments, centroids, ptFeats)
    return polyFeats, ptFeats
      
  def _calcFeatures(self, segments, points):
    polyFeats = []
    ptFeats = []
    for layer in self.layers.itervalues():
      if layer.hasSource():
      # common.debug(layer, dir(layer))
        with layer:
          forpoly, forpts = layer.calculate(segments, points)
          ptFeats.extend(forpts)
          polyFeats.extend(forpoly)
      else:
        common.warning('layer {} not initialized'.format(layer.getName()))
    return polyFeats, ptFeats
  
  @classmethod
  def aggregate(cls, segments, points, fields=None):
    if fields is None:
      fields = [fld for fld in common.fieldList(points) if fld.startswith('FF')]
    # aggregate data from fields in points to segments using mean weighted by weightFld
    with common.PathManager(segments) as pathman:
      # first, find which points belong into which segment
      common.progress('matching neighbourhood data')
      ptsIdent = pathman.tmpFile()
      segIDFld = common.ensureIDField(segments)
      # _to_many means there will be a JOIN_FID in the output to identify the segment of the point
      arcpy.SpatialJoin_analysis(points, segments, ptsIdent, 'JOIN_ONE_TO_MANY', 'KEEP_COMMON', '', 'WITHIN')
      # now load the point data in fields, the segment id and weighting area
      common.progress('aggregating neighbourhood data')
      ptSlots = {'sid' : 'JOIN_FID', 'weight' : cls.AREA_FIELD}
      segSlots, segTypes = {}, {}
      for fld in fields:
        ptSlots[fld] = fld
        segSlots[fld] = fld
        segTypes[fld] = float
      ptFeats = loaders.BasicReader(ptsIdent, ptSlots).read()
      segFeats = cls.mergeToSegments(ptFeats, fields)
      loaders.ObjectMarker(segments, {'id' : segIDFld}, segSlots, {}, segTypes).mark(segFeats)
  
  @classmethod
  def mergeToSegments(cls, ptFeats, fields):
    # common.debug(ptFeats)
    # common.debug(fields)
    segFeats = collections.defaultdict(lambda: collections.defaultdict(float))
    segWeightSums = collections.defaultdict(float)
    for pt in ptFeats:
      segID = pt['sid']
      weight = pt['weight']
      for fld in fields:
        if pt[fld]:
          segFeats[segID][fld] += pt[fld] * weight
      segWeightSums[segID] += weight
    for segID, seg in segFeats.iteritems():
      wtSum = segWeightSums[segID]
      for fld in fields:
        seg[fld] /= wtSum
    return segFeats
  
  

def calculate(segments, layers, config, layerClasses=None):
  if layerClasses is None:
    layerClasses = FeatureCalculator.DEFAULT_LAYERS
  calc = FeatureCalculator(layerClasses, config)
  calc.input(layers)
  centroids = common.addFeatureExt(os.path.splitext(segments)[0] + '_centroids')
  if not arcpy.Exists(centroids):
    calc.createSegmentCentroids(segments, centroids)
  calc.calculate(segments, centroids)
  return calc.names()
    
if __name__ == '__main__':
  with common.runtool(2) as parameters:
    segments, layer = parameters
    import json
    with open('e:\\school\\tools\\config\\features.json') as conffile:
      configCont = json.load(conffile)
    calculate(segments, {'transport' : layer, 'landuse' : r'E:\school\dp\test\malesice\malesice.gdb\ua'}, configCont, [TransportLayer])
  # with common.runtool(4) as parameters:
    # segments, layer, name, config = parameters
    # import json
    # with open(config) as conffile:
      # configCont = json.load(conffile)
    # calculate(segments, {name : layer, 'landuse' : r'E:\school\dp\test\smichov\ua.shp'}, configCont)
