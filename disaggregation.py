import json
import os

import arcpy

import common
import loaders
import features

# PROGRAMMING TODOS:
# stop connections?
# calculate inhabitants and transport
# transport network creation
# debug transformer (road with empty class and zero speed)

    
class Disaggregator:
  @staticmethod
  def disaggregateInPlace(layer, mainIDFld, mainFld, weightFld=None, targetFld=None, shout=False):
    if not targetFld:
      targetFld = mainFld
    if not weightFld:
      weightFld = common.ensureShapeAreaField(layer)
    # disaggregate by area
    # load data
    idFld = common.ensureIDField(layer)
    inputSlots = {'id' : idFld,
                  'mainid' : mainIDFld,
                  'mainval' : mainFld,
                  'weight' : weightFld}
    input = loaders.BasicReader(layer, inputSlots).read(text=('reading values to disaggregate' if shout else None))
    if shout: common.progress('computing weight sums')
    weightSums = {}
    for item in input:
      mainID = item['mainid']
      if mainID not in input:
        weightSums[mainID] = 0.0
      weightSums[mainID] += item['weight']
    if shout: common.progress('disaggregating values')
    output = {}
    for item in input:
      item['val'] = item['mainval'] * item['weight'] / weightSums[item['mainid']]
      output[item['id']] = item
    # write back
    loaders.ObjectMarker(layer, {'id' : idFld}, {'val' : targetFld}, outTypes={'val' : float}).mark(output, text=('writing disaggregated values' if shout else None))
    
    # with common.PathManager(layer) as pathman:
      # weightSums = pathman.tmpTable()
      # arcpy.Statistics_analysis(layer, weightSums, [(weightFld, 'SUM')], mainIDFld)
      # weightSumFld = 'SUM_' + weightFld
      # # join the weight sums back
      # arcpy.JoinField_management(layer, mainIDFld, weightSums, mainIDFld, weightSumFld)
      # weightSumFld = common.fieldNameIn(layer, weightSumFld)
      # pathman.registerField(layer, weightSumFld)
      # # and calculate the disaggregated values
      # common.debug(layer, targetFld, mainFld, weightFld, weightSumFld, common.fieldList(layer))
      # common.calcField(layer, targetFld, '!{}! * !{}! / !{}!'.format(mainFld, weightFld, weightSumFld), float if targetFld != mainFld else None)
  
  @classmethod
  def overlayToValue(cls, segments, overlay, valueFld, target):
    if common.getShapeType(overlay) == 'POINT':
      cls.overlayPointsToValue(segments, overlay, valueFld, target)
    else:
      cls.overlayPolygonsToValue(segments, overlay, valueFld, target)
  
  @classmethod
  def overlayPointsToValue(cls, segments, overlay, valueFld, target):
    fm = arcpy.FieldMappings()
    fm.addFieldMap(common.fieldMap(overlay, valueFld, valueFld, cls.AGGREGATION))
    arcpy.SpatialJoin_analysis(segments, overlay, target, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', fm, 'INTERSECT')

  @staticmethod
  def overlayPolygons(segments, overlay, target, keepFlds=[], keepOverlayID=True, keepSegmentID=False):
    arcpy.Intersect_analysis([segments, overlay], target, 'ALL')
    if keepOverlayID:
      overlayID = common.joinIDName(overlay, target)
      keepFlds += [overlayID]
    if keepSegmentID:
      segmentID = common.joinIDName(segments, target)
      keepFlds += [segmentID]
    common.clearFields(target, exclude=keepFlds)
    if keepOverlayID and keepSegmentID:
      return overlayID, segmentID
    elif keepOverlayID:
      return overlayID
    elif keepSegmentID:
      return segmentID
    else:
      return None
      
  @classmethod
  def overlayToID(cls, segments, overlay, valueFld, target, segWeightFld=None, valueFldSuffix=''):
    keepFlds = [valueFld]
    if segWeightFld:
      segIDFld = common.joinIDName(segments, target)
      keepFlds += [segWeightFld, segIDFld]
    overIDFld = cls.overlayPolygons(segments, overlay, target, keepFlds, keepOverlayID=True)
    if segWeightFld:
      cls.disaggregateInPlace(target, segIDFld, segWeightFld)
      arcpy.DeleteField_management(target, segIDFld)
    if valueFldSuffix:
      common.copyField(target, valueFld, valueFld + valueFldSuffix)
      arcpy.DeleteField_management(target, valueFld)
    return overIDFld
  
  @classmethod
  def disaggregate(cls, segments, mainIDFld, valueFld, weightFld, apriWeightFld=None, mainFldSuffix=''):
    coefFld = valueFld + '_TCOEF'
    weightFld = cls.computeTransferCoefs(segments, mainIDFld, weightFld, apriWeightFld, coefFld)
    valExpr = '!{}! * !{}! * !{}!'.format(coefFld, valueFld + mainFldSuffix, weightFld)
    common.debug(valExpr)
    common.calcField(segments, valueFld, valExpr, float)
            
  @classmethod
  def computeTransferCoefs(cls, segments, mainIDFld, weightFld, apriWeightFld, tgtFld):
    # summarization
    with common.PathManager(segments) as pathman:
      coefTable = pathman.tmpTable()
      weightFld, stats = cls.prepareWeights(segments, weightFld, apriWeightFld)
      common.debug(weightFld, stats)
      arcpy.Statistics_analysis(segments, coefTable, stats, mainIDFld)
      cls.divideWeights(coefTable, weightFld, apriWeightFld, tgtFld)
      arcpy.JoinField_management(segments, mainIDFld, coefTable, mainIDFld, tgtFld)
      return weightFld
  
  @classmethod
  def overlayValidate(cls, segments, valueFld, valid, validFld, target):
    return cls.overlayPolygons(segments, valid, target, [valueFld, validFld], keepSegmentID=True, keepOverlayID=True)
    

      
class AbsoluteDisaggregator(Disaggregator):
  AGGREGATION = 'SUM'

  @classmethod
  def overlayPolygonsToValue(cls, segments, overlay, valueFld, target, keepID=False):
    overIDFld = cls.overlayPolygons(segments, overlay, target, [valueFld], keepOverlayID=True)
    cls.disaggregateInPlace(target, overIDFld, valueFld)
    if not keepID:
      arcpy.DeleteField_management(target, overIDFld)
  
  @classmethod
  def modelInValue(cls, segments, valueFld, targetFld):
    # apply correction factor so we do not lose precision
    densExpr = '!{}! / !shape.area! * 1000000'.format(valueFld)
    common.calcField(segments, targetFld, densExpr, float)
  
  @classmethod
  def prepareWeights(cls, segments, weightFld, apriWeightFld=None):
    absFld = weightFld + '_ABS'
    absExpr = '!{}! * !{}! / 1000000.0'.format(weightFld, common.ensureShapeAreaField(segments))
    common.calcField(segments, absFld, absExpr, float)
    return absFld, [(absFld, 'SUM')]
  
  @classmethod
  def divideWeights(cls, table, weightFld, apriWeightFld=None, tgtFld=None):
    if not tgtFld: tgtFld = weightFld
    sumWeightFld = 'SUM_' + weightFld
    common.calcField(table, tgtFld, '1.0 / !{}!'.format(sumWeightFld), float)
    
  @classmethod
  def overlayValidate(cls, segments, valueFld, valid, validFld, target):
    validIDFld, segmentIDFld = Disaggregator.overlayValidate(segments, valueFld, valid, validFld, target)
    cls.disaggregateInPlace(target, segmentIDFld, valueFld)
    cls.disaggregateInPlace(target, validIDFld, validFld)

    
class RelativeDisaggregator(Disaggregator):
  AGGREGATION = 'MEAN'
  
  @classmethod
  def overlayPolygonsToValue(cls, segments, overlay, valueFld, target, keepID=False):
    cls.overlayPolygons(segments, overlay, target, [valueFld], keepOverlayID=keepID)
  
  @classmethod
  def modelInValue(cls, segments, valueFld, targetFld):
    common.copyField(segments, valueFld, targetFld)
    
  @classmethod
  def prepareWeights(cls, segments, weightFld, apriWeightFld=None):
    if apriWeightFld:
      combFld = weightFld + '_MUL'
      combExpr = '!{}! * !{}!'.format(weightFld, apriWeightFld)
      common.calcField(segments, combFld, combExpr, float)
      return weightFld, [(combFld, 'SUM'), (apriWeightFld, 'SUM')]
    else:
      return weightFld, [(common.ensureIDField(segments), 'COUNT'), (weightFld, 'SUM')]
  
  @classmethod
  def divideWeights(cls, table, weightFld, apriWeightFld=None, tgtFld=None):
    if not tgtFld: tgtFld = weightFld
    if apriWeightFld:
      sumCombFld = 'SUM_' + weightFld + '_MUL'
      coefExpr = '!{}! / !{}!'.format('SUM_' + apriWeightFld, sumCombFld)
    else:
      countFld = [fld for fld in common.fieldList(table) if fld.startswith('COUNT_')].pop()
      coefExpr = '!{}! / !{}!'.format(countFld, 'SUM_' + weightFld)
    common.calcField(table, tgtFld, coefExpr, float)
  
def overlayToID(segments, overlay, valueFld, target, segWeightFld=None, valueFldSuffix=''):
  return Disaggregator.overlayToID(segments, overlay, valueFld, target, segWeightFld, valueFldSuffix=valueFldSuffix)
  
  # TRAIN_VAL_FLD = 'TRAIN_VAL'
  # INPUT_ID_FLD = 'DISAG_ID'
  # TRAIN_ID_P = 'ID_TRAIN_'
  # APPLY_ID_P = 'ID_APPLY_'
  # TRAIN_SEGS_P = 'segments_train_'
  
  # def __init__(self, calcProxy):
    # self.calcProxy = calcProxy

  # @staticmethod
  # def idField(layer, name):
    # common.copyField(layer, common.ensureIDField(layer), name)
    # return name
  
  # @staticmethod
  # def name(prefix, layer):
    # return prefix + common.fcName(layer)
  
  # @staticmethod
  # def path(segments, name):
    # return common.featurePath(common.location(segments), name)
    
  # def train(self, model, segments, train, trainValFld, segWeightFld=None, save=True):
    # common.progress('matching segments with training data')
    # idFldName = self.idField(segments, self.name(self.TRAIN_ID_P, train))
    # segmentsWithTrainVals = self.path(segments, self.name(self.TRAIN_SEGS_P, train))
    # self.valuesToSegments(segments, train, trainValFld, self.TRAIN_VAL_FLD, segmentsWithTrainVals, segWeightFld)
    # common.progress('calculating features')
    # self.calcProxy.calculate(segmentsWithTrainVals)
    # self.trainFromValuedSegments(model, segmentsWithTrainVals, idFldName, outSegments=segments)
  
  # def trainFromValuedSegments(self, model, segmentsWithTrainVals, idFldName, outSegments=None, save=True):
    # # load the segment data (features and train values)
    # segmentSlots = {'id' : idFldName, 'trainVal' : self.TRAIN_VAL_FLD, 'area' : common.ensureShapeAreaField(segmentsWithTrainVals)}
    # featureNames = self.featureFields(segmentsWithTrainVals)
    # for feat in featureNames:
      # segmentSlots[feat] = feat
    # segmentData = loaders.BasicReader(segmentsWithTrainVals, segmentSlots).read(text='loading segment data')
    # # use the model, luke
    # common.progress('initializing regression model')
    # model.addFeatureNames(featureNames)
    # for seg in segmentData:
      # seg['features'] = model.featuresToList(seg)
      # seg['inVal'] = self.modelInputValue(seg)
      # model.addExample(seg['features'], seg['inVal'])
    # common.progress('training regression model')
    # model.train()
    # if save:
      # self.save(segmentData, outSegments if outSegments else segmentsWithTrainVals, self.trainIDName(model), {'inVal' : 'MODIN_VAL', 'trainVal' : self.TRAIN_VAL_FLD})

  # @staticmethod
  # def save(data, layer, idFld, outSlots):
    # common.debug('saving data')
    # if isinstance(data, list):
      # data = {item['id'] : item for item in data}
    # loaders.ObjectMarker(layer, {'id' : idFld}, outSlots, outTypes={key : float for key in outSlots}).mark(data)
  
  ############################## TODO
  
  def apply(self, model, segments, input, inputValFld, segWeightFld=None, save=True):
    with common.PathManager(segments) as pathman:
      self.pathman = pathman
      # match the segments to the input
      self.idField(segments, self.applyIDName(model))
      segmentsWithInputIDs = common.featurePath(self.pathman.getLocation(), 'segments_apply_' + model.getName())
      self.idsToSegments(segments, input, common.ensureIDField(input), self.INPUT_ID_FLD, segmentsWithInputIDs)
      common.progress('calculating features')
      features.calculate(segmentsWithInputIDs, self.directLayers, self.featConfig)
      self.applyFromAssignedSegments(self, model, segmentsWithInputIDs, input, inputValFld, segWeightFld, outSegments=segments, save=save)
  
  # def applyFromAssignedSegments(self, model, segmentsWithInputIDs, input, inputValFld, segWeightFld=None, outSegments=None, save=True):
    # if not segWeightFld:
      # segWeightFld = common.ensureShapeAreaField(segmentsWithInputIDs)
    # # load the segment data (features and input ids values)
    # segmentSlots = {'id' : self.applyIDName(model), 'inputID' : self.INPUT_ID_FLD, 'area' : common.ensureShapeAreaField(segmentsWithInputIDs), 'aprioriWeight' : segWeightFld}
    # features = self.featureFields(segmentsWithInputIDs)
    # for feat in features: segmentSlots[feat] = feat
    # segmentData = loaders.BasicReader(segmentsWithInputIDs, segmentSlots).read(text='loading segment data')
    # # load input values
    # inputData = loaders.DictReader(input, {'id' : common.ensureIDField(input), 'value' : inputValFld}).read(text='loading disaggregation data')
    # # prepare disag weights
    # for item in inputData.itervalues():
      # item['modelWeights'] = []
      # item['aprioriWeights'] = []
    # # apply model
    # modelFX = model.get()
    # for seg in segmentData:
      # seg['features'] = model.featuresToList(seg)
      # common.debug(len(seg['features']))
      # seg['outVal'] = modelFX(seg['features'])
      # seg['modelWeight'] = self.modelOutputValue(seg)
      # # sum weights to disaggregation polygons
      # inputItem = inputData[seg['inputID']]
      # inputItem['modelWeights'].append(seg['modelWeight'])
      # inputItem['aprioriWeights'].append(seg['aprioriWeight'])
    # # compute the disaggregation coefficients
    # for item in inputData.itervalues():
      # item['coef'] = self.transferCoefficient(item)
    # # disaggregate
    # for seg in segmentData:
      # seg['modelVal'] = inputData[seg['inputID']]['coef'] * seg['modelWeight']
    # if save:
      # self.save(segmentData, outSegments if outSegments else segmentsWithInputIDs, self.applyIDName(model), {'outVal' : 'MODOUT_VAL', 'inputID' : self.INPUT_ID_FLD, 'modelWeight' : 'MODEL_WEIG', 'modelVal' : 'MODEL_VAL'})
  
  # def validate(self, model, segments, segValFld, valid, validValFld, segWeightFld=None, save=True):
    # with common.PathManager(segments) as pathman:
      # self.pathman = pathman
      # validIDFld = common.ensureIDField(valid)
      # # disaggregate the validValFld and segValFld across segments according to segWeightFld
      # segmentsWithValidVals = self.valuesToSegments(segments, valid, validIDFld, validValFld, self.VALID_VAL_FLD, segWeightFld, keepVal=segValFld)
      # # load the segment and validation values
      # if not segWeightFld:
        # segWeightFld = common.ensureShapeAreaField(segmentsWithValidVals)
      # segmentSlots = {'resultVal' : segValFld, 'validVal' : self.VALID_VAL_FLD, 'aprioriWeight' : segWeightFld}
      # segmentData = loaders.DictReader(segmentsWithValidVals, segmentSlots).read(text='loading segment data')
      # # perform some analysis on the segment data and create an HTML report
    
  ############################## END TODO

    
  # def idsToSegments(self, segments, overlay, overlayIDFld, targetIDFld, targetPath):
    # fm = arcpy.FieldMappings()
    # fm.addTable(segments)
    # fm.addFieldMap(common.fieldMap(overlay, overlayIDFld, targetIDFld, 'FIRST'))
    # arcpy.SpatialJoin_analysis(segments, overlay, targetPath, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', fm, 'HAVE_THEIR_CENTER_IN')
    
  # def valuesToSegments(self, segments, overlay, overlayFld, targetFld, targetPath, segWeightFld=None, keepVal=None):
    # if common.getShapeType(overlay) == 'POINT':
      # return self.valuesToSegmentsFromPoints(segments, overlay, overlayFld, targetFld, targetPath)
    # else:
      # return self.valuesToSegmentsFromPolygons(segments, overlay, overlayFld, targetFld, targetPath, segWeightFld, keepVal)
      
  # def valuesToSegmentsFromPoints(self, segments, overlay, overlayFld, targetFld, targetPath):
    # # join the segment ids to the points
    # # segIDFld = self.pathman.tmpIDField(segments)
    # joined = self.pathman.tmpFC()
    # fm = arcpy.FieldMappings()
    # fm.addTable(segments)
    # fm.addFieldMap(common.fieldMap(overlay, overlayFld, targetFld, self.AGGREGATION))
    # arcpy.SpatialJoin_analysis(overlay, segments, joined, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', fm, 'INTERSECT')
    # perform the statistic to aggregate the point feats
    # aggregated = self.pathman.tmpTable()
    # arcpy.Statistics_analysis(joined, aggregated, [(overlayFld, self.AGGREGATION)], segIDFld)
    # aggFld = self.AGGREGATION + '_' + overlayFld
    # # join the result to the segments
    # arcpy.JoinField_management(segments, segIDFld, aggregated, segIDFld, aggFld)
    # common.copyField(segments, aggFld, targetFld)
    # select those with not-null values
    # selected = self.pathman.tmpFC(delete=False, suffix='segs') # TODO
    # arcpy.Select_analysis(segments, selected, common.safeQuery('[{0}] is not null and [{0}] <> 0'.format(targetFld), segments))
    # arcpy.Select_analysis(joined, targetPath, common.safeQuery('[{0}] is not null and [{0}] <> 0'.format(targetFld), joined))
    # self.pathman.registerField(segments, aggFld)
    # self.pathman.registerField(segments, targetFld)
      
  # def valuesToSegmentsFromPolygons(self, segments, overlay, overlayFld, targetFld, targetPath, segWeightFld=None, keepVal=None):
    # # assign the overlay polygons to the segments, cutting the segments if necessary
    # # keeps only the segments overlaid!
    # overlayIDFld = self.pathman.tmpIDField(overlay)
    # if keepVal is not None or segWeightFld is not None:
      # segIDFld = self.pathman.tmpIDField(segments)
    # arcpy.Intersect_analysis([segments, overlay], targetPath, 'ALL')
    # common.clearFields(targetPath, exclude=common.fieldList(segments) + [overlayIDFld, overlayFld])
    # # now, every segment has its unique overlay polygon
    # # first, if there are different apriori segment weights than area, we need to disaggregate them
    # # and the segment keep values (such as previously calculated values) as well
    # if not segWeightFld:
      # segWeightFld = common.ensureShapeAreaField(targetPath)
    # else:
      # self.disaggregateInPlaceByArea(targetPath, segIDFld, segWeightFld)
    # if keepVal:
      # self.disaggregateInPlaceByArea(targetPath, segIDFld, keepVal)
    # # we need to disaggregate the overlayFld values by weight (default: area)
    # self.disaggregateToField(targetPath, overlayIDFld, overlayFld, segWeightFld, targetFld)
  
  # def disaggregateInPlaceByArea(self, layer, idFld, valFld):
    # swapFld = self.pathman.tmpField(layer, common.pyTypeOfField(layer, valFld))
    # common.copyField(layer, valFld, swapFld)
    # arcpy.DeleteField_management(layer, valFld)
    # self.disaggregateToField(layer, idFld, swapFld, common.ensureShapeAreaField(layer), valFld)

    
# class AbsoluteDisaggregator(Disaggregator):
  # AGGREGATION = 'SUM'

  # def disaggregateToField(self, layer, mainIDFld, mainFld, weightFld, tgtFld):
    # # first, we need the sum of weights for each line
    # weightSums = self.pathman.tmpTable()
    # arcpy.Statistics_analysis(layer, weightSums, [(weightFld, 'SUM')], mainIDFld)
    # weightSumFld = 'SUM_' + weightFld
    # # join the weight sums back
    # arcpy.JoinField_management(layer, mainIDFld, weightSums, mainIDFld, weightSumFld)
    # # and calculate the disaggregated values
    # common.addField(layer, tgtFld, float)
    # arcpy.CalculateField_management(layer, tgtFld, '!{}! * !{}! / !{}!'.format(mainFld, weightFld, weightSumFld), 'PYTHON_9.3')
  
  # @staticmethod
  # def modelInputValue(seg):
    # return seg['trainVal'] / seg['area']
  
  # @staticmethod
  # def modelOutputValue(seg):
    # return seg['outVal'] * seg['area']
  
  # @staticmethod
  # def transferCoefficient(item):
    # return item['value'] / sum(item['modelWeights'])
  
  
# class RelativeDisaggregator(Disaggregator):
  # AGGREGATION = 'MEAN'

  # def disaggregateToField(self, layer, mainIDFld, mainFld, weightFld, tgtFld):
    # # 
    # common.copyField(layer, mainFld, tgtFld)

  # @staticmethod
  # def modelInputValue(seg):
    # return seg['trainVal']

  # @staticmethod
  # def modelOutputValue(seg):
    # return seg['outVal']

  # @staticmethod
  # def transferCoefficient(item):
    # modelWts = item['modelWeights']
    # aprioriWts = item['aprioriWeights']
    # weightedWeightSum = sum(modelWts[i] * aprioriWts[i] for i in range(len(modelWts)))
    # return item['value'] * sum(aprioriWts) / weightedWeightSum
  
    
# def assignIDs(target, source, srcIDFld, tgtFld):
  # with common.PathManager(target) as pathman:
    # tgtIDFld = pathman.tmpIDField(target)
    # if srcIDFld is None:
      # srcIDFld = pathman.tmpIDField(source)
    # tgtPts = pathman.tmpFile()
    # arcpy.FeatureToPoint_management(target, tgtPts, 'INSIDE')
    # tgtLocated = pathman.tmpFile()
    # # for polygon source, match the encompassing polygon; for point, match the closest
    # mode = 'WITHIN' if common.getShapeType(source) == 'POLYGON' else 'CLOSEST'
    # arcpy.SpatialJoin_analysis(tgtPts, source, tgtLocated, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', '', 'WITHIN')
    # arcpy.JoinField_management(target, tgtIDFld, tgtLocated, tgtIDFld, srcIDFld)
    # if srcIDFld != tgtFld:
      # common.copyField(target, srcIDFld, tgtFld)
      # pathman.registerField(target, srcIDFld)
    
# def assignValues(tgtData, srcData, linkID, tgtSlot, valSlot):
  # # transfer the values by id
  # for tgt in tgtData.itervalues():
    # tgt[tgtSlot] = srcData[tgtData[linkID]]['value']
  
# def disaggregate(data, idKey, mainKey, weightKey, tgtKey, absolute=True):
  # # first, find out the sums of weights for each main value
  # weightSums = {}
  # mainValues = {}
  # for item in data.itervalues():
    # mainID = item[idKey]
    # weightSums[mainID] = weightSums.get(mainID, 0.0) + item[weightKey]
    # if absolute:
      # mainValues[mainID] = item[mainKey]
    # else:
      # mainValues[mainID] += item[mainKey] # we need to compensate for the count of subunits
  # # disaggregate!
  # for item in data.itervalues():
    # mainID = item[idKey]
    # item[tgtKey] = item[weightKey] * mainValues[mainID] / weightSums[mainID]

# def computeDensities(data, value, area, density):
  # for item in data.itervalues():
    # data[density] = data[value] / data[area]
    
# def trainModel(segments, directLayers, disag, disagValFld, train, trainValFld, featConfig, absolute=False):
  # # common.progress('creating segments')
  # # segments = common.featurePath(os.path.dirname(disag), 'segments')  
  # # createSegments(landuse, [barriers, disag, train, geostat], UA_CODE_FIELD, MIN_SEGMENT_SIZE, MAX_SEGMENT_SIZE, MAX_SEGMENT_WAGNER, segments)
  # # # create transport network
  # # transnet = createTransportNetwork(transport, pois)
  # # calculate segment features
  # common.progress('calculating features')
  # feats = features.calculate(segments, directLayers, featConfig)
  # # assign segments to train and disag areas by centroid
  # common.progress('matching segments with disaggregation and training data')
  # assignIDs(segments, disag, None, 'DISAG_ID')
  # assignIDs(segments, train, None, 'TRAIN_ID')
  # # load segment id, all features, area, train id and disag id
  # common.progress()
  # segmentSlots = {'id' : common.ensureIDField(segments), 'disagID' : 'DISAG_ID', 'trainID' : 'TRAIN_ID', 'area' : common.ensureShapeAreaField(segments)}
  # for feat in features: segmentSlots[feat] = feat
  # segmentData = loaders.DictReader(segments, segmentSlots).read(text='loading segment data')
  # # load disag values to disaggregate
  # disagData = loaders.DictReader(disag, {'id' : common.ensureIDField(disag), 'value' : disagValFld}).read(text='loading disaggregation data')
  # # load train values
  # trainData = loaders.DictReader(train, {'id' : common.ensureIDField(train), 'value' : trainValFld}).read(text='loading training data')
  # # merge the train and disaggregation values to segments
  # common.progress('transferring training and disaggregation values')
  # assignValues(segmentData, trainData, 'trainID', 'trainMain', 'value')
  # assignValues(segmentData, disagData, 'disagID', 'disagMain', 'value')
  # if absolute:
    # # we have to disaggregate the trainMain values by area and obtain densities
    # disaggregate(segmentData, 'trainID', 'trainMain', 'area', 'trainVal')
    # computeDensities(segmentData, 'trainVal', 'area', 'trainDens')
    # trainSlot = 'trainDens'
  # else:
    # trainSlot = 'trainMain'
  # # use the model, luke
  # common.progress('initializing regression model')
  # model = OLSModel()
  # for seg in segmentData:
    # seg['features'] = [seg[feat] for feat in features]
    # model.addExample(seg['features'], seg[trainSlot])
  # common.progress('training regression model')
  # model.train()
  # common.debug(model.report())
  # # the model is trained, apply the regression
  # common.progress('computing model outputs')
  # modelSlot = 'modelDens' if absolute else 'modelVal'
  # disagSlot = 'modelVal' if absolute else 'modelWeight'
  # for seg in segmentData:
    # seg[modelSlot] = model.compute(seg['features'])
    # seg[disagSlot] = seg[modelSlot] * seg['area']
  # # and disaggregate
  # common.progress('disaggregating')
  # disaggregate(segmentData, 'disagID', 'disagMain', disagSlot, 'outVal', absolute=absolute)
  # # output the results
  # common.progress('saving results')
  # outSlots = {
    # 'outVal' : 'OUT_VAL', # the output value (absolute)
    # disagSlot : 'DISAG_WEIGHT', # disaggregation weight
    # 'modelVal' : 'MODEL_VAL', # value on the model output
    # '
    

    
class FeatureCalculatorProxy:
  def __init__(self, directLayers, featConfigFile):
    self.directLayers = directLayers
    self.featConfigFile = featConfigFile
    self.featConfig = self.loadConfig(self.featConfigFile)
  
  def calculate(self, segments):
    features.calculate(segments, self.directLayers, self.featConfig)
  
  @staticmethod
  def loadConfig(file):
    if file:
      with open(file) as fcFO:
        return json.load(fcFO)
    else:
      return None
    
    
def create(absolute):
  if absolute:
    return AbsoluteDisaggregator()
  else:
    return RelativeDisaggregator()

if __name__ == '__main__':
  with common.runtool(5) as parameters:
    trainedModel = trainModel(*parameters[:-1])
    trainedModel.serialize(parameters[-1])