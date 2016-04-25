import json
import os
import collections

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
    # common.debug('by area', layer, mainIDFld, mainFld, weightFld, targetFld)
    # disaggregate by area
    # load data
    idFld = common.ensureIDField(layer)
    inputSlots = {'id' : idFld,
                  'mainid' : mainIDFld,
                  'mainval' : mainFld,
                  'weight' : weightFld}
    # common.debug(layer, inputSlots)
    input = loaders.BasicReader(layer, inputSlots).read(text=('reading values to disaggregate' if shout else None))
    if shout: common.progress('computing weight sums')
    weightSums = {}
    for item in input:
      mainID = item['mainid']
      if mainID not in weightSums:
        weightSums[mainID] = 0.0
      weightSums[mainID] += item['weight']
    # common.debug(weightSums)
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
    with common.PathManager(segments) as pathman:
      # determine which segment the points lie in (or near to)
      # first, record the segment id
      segIDFld = pathman.tmpField(segments, int)
      common.copyField(segments, common.ensureIDField(segments), segIDFld)
      # join the points
      ptsIdent = pathman.tmpFC()
      fm = arcpy.FieldMappings()
      fm.addFieldMap(common.fieldMap(segments, segIDFld, segIDFld, 'FIRST'))
      fm.addFieldMap(common.fieldMap(overlay, valueFld, valueFld, 'FIRST'))
      arcpy.SpatialJoin_analysis(overlay, segments, ptsIdent, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', fm, 'CLOSEST', '10 Meters')
      # load the point values and segment ids
      ptData = loaders.BasicReader(ptsIdent, {'sid' : segIDFld, 'value' : valueFld}).read()
      segData = collections.defaultdict(list)
      for pt in ptData:
        segData[pt['sid']].append(pt['value'])
      segVals = collections.defaultdict(lambda: {'value' : 0.0},
        {id : {'value' : cls.AGGREGATOR(values)} for id, values in segData.iteritems()})
      arcpy.CopyFeatures_management(segments, target)
      pathman.registerField(target, segIDFld)
      loaders.ObjectMarker(target, {'id' : segIDFld}, {'value' : valueFld}, outTypes={'value' : float}).mark(segVals)
      

  @staticmethod
  def overlayPolygons(segments, overlay, target, keepFlds=[], keepOverlayID=True, keepSegmentID=False):
    arcpy.Intersect_analysis([segments, overlay], target, 'ALL')
    if keepOverlayID:
      overlayID = common.joinIDName(overlay, target)
      keepFlds += [overlayID]
    if keepSegmentID:
      segmentID = common.joinIDName(segments, target)
      keepFlds += [segmentID]
    # common.debug(overlay, target, overlayID)
    # common.debug(common.fieldList(target))
    # common.debug(keepFlds)
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
    weightFld = cls.prepareWeights(segments, weightFld, apriWeightFld)
    segID = common.ensureIDField(segments)
    slots = {'id' : segID, 'mainid' : mainIDFld, 'mainval' : valueFld + mainFldSuffix, 'wt' : weightFld}
    if apriWeightFld: slots['awt'] = apriWeightFld
    segdata = loaders.BasicReader(segments, slots).read('reading disaggregation data')
    coefs = cls.transferCoefs(segdata)
    outdata = {}
    for seg in segdata:
      seg['val'] = seg['mainval'] * seg['wt'] * coefs[seg['mainid']]
      outdata[seg['id']] = seg
    common.debug(outdata.items()[:10])
    loaders.ObjectMarker(segments, {'id' : segID}, {'val' : valueFld}, outTypes={'val' : float}).mark(outdata, text='saving disaggregation data')
    # coefFld = valueFld + '_TCOEF'
    # weightFld = cls.computeTransferCoefs(segments, mainIDFld, weightFld, apriWeightFld, coefFld)
    # valExpr = '!{}! * !{}! * !{}!'.format(coefFld, valueFld + mainFldSuffix, weightFld)
    # common.debug(valExpr)
    # common.calcField(segments, valueFld, valExpr, float)
            
  # @classmethod
  # def computeTransferCoefs(cls, segments, mainIDFld, weightFld, apriWeightFld, tgtFld):
    # # summarization
    # common.debug('segments', segments)
    # with common.PathManager(segments) as pathman:
      # coefTable = pathman.tmpTable()
      # weightFld, stats = cls.prepareWeights(segments, weightFld, apriWeightFld)
      # common.debug(weightFld, stats)
      # arcpy.Statistics_analysis(segments, coefTable, stats, mainIDFld)
      # cls.divideWeights(coefTable, weightFld, apriWeightFld, tgtFld)
      # arcpy.JoinField_management(segments, mainIDFld, coefTable, mainIDFld, tgtFld)
      # return weightFld
  
  @classmethod
  def addUnmatched(cls, toadd, exclude, target, toaddID):
    # add validation polygons untouched by any segment
    unmatched = arcpy.MakeFeatureLayer_management(toadd)
    arcpy.SelectLayerByLocation_management(unmatched, 'INTERSECT', exclude)
    arcpy.SelectLayerByAttribute_management(unmatched, 'SWITCH_SELECTION')
    # append them to the result; will keep nothing but validFld
    arcpy.Append_management([unmatched], target, 'NO_TEST')
    # make fictive IDs for the unmatched (needs unique)
    cls.extraIDs(target, toaddID)
  
  @classmethod
  def extraIDs(cls, target, toaddID, empty=None):
    common.calcField(target, toaddID, '(!{1}! + {2}) if !{0}!=={3} else !{0}!'.format(toaddID, common.ensureIDField(target), 10 * common.count(target), str(empty)))
    

      
class AbsoluteDisaggregator(Disaggregator):
  AGGREGATION = 'SUM'
  AGGREGATOR = sum

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
    absExpr = '!{0}! * !{1}! / 1000000.0 if !{0}! > 0 else 0.0'.format(weightFld, common.ensureShapeAreaField(segments))
    common.calcField(segments, absFld, absExpr, float)
    return absFld
  
  @classmethod
  def transferCoefs(cls, segdata):
    wtsums = collections.defaultdict(float)
    for seg in segdata:
      wtsums[seg['mainid']] += seg['wt']
    return {mainid : (1.0 / wtsum if wtsum else 0.0) for mainid, wtsum in wtsums.iteritems()}
  
  # @classmethod
  # def divideWeights(cls, table, weightFld, apriWeightFld=None, tgtFld=None):
    # common.debug(weightFld)
    # if not tgtFld: tgtFld = weightFld
    # sumWeightFld = 'SUM_' + weightFld
    # common.debug('dividing', table, tgtFld)
    # common.calcField(table, tgtFld, '1.0 / !{}!'.format(sumWeightFld), float)
    
  @classmethod
  def overlayValidate(cls, segments, valueFld, valid, validFld, target):
    with common.PathManager(target) as pathman:
      inters = pathman.tmpFC()
      arcpy.Identity_analysis(segments, valid, inters, 'ALL')
      segIDFld = common.joinIDName(segments, inters)
      validIDFld = common.joinIDName(valid, inters)
      cls.disaggregateInPlace(inters, segIDFld, valueFld)
      cls.extraIDs(inters, validIDFld, empty=-1)
      arcpy.Dissolve_management(inters, target, validIDFld, [(valueFld, 'SUM'), (validFld, 'FIRST')], 'MULTI_PART')
      common.copyField(target, 'SUM_' + valueFld, valueFld)
      common.copyField(target, 'FIRST_' + validFld, validFld)
      cls.addUnmatched(valid, segments, target, validIDFld)
      # return overlayID, segmentID
  
    
class RelativeDisaggregator(Disaggregator):
  AGGREGATION = 'MEAN'
  AGGREGATOR = lambda lst: sum(lst) / len(lst)

  @classmethod
  def overlayValidate(cls, segments, valueFld, valid, validFld, target):
    arcpy.Intersect_analysis([segments, valid], target, 'ALL')
  
  @classmethod
  def overlayPolygonsToValue(cls, segments, overlay, valueFld, target, keepID=False):
    cls.overlayPolygons(segments, overlay, target, [valueFld], keepOverlayID=keepID)
  
  @classmethod
  def modelInValue(cls, segments, valueFld, targetFld):
    common.copyField(segments, valueFld, targetFld)
    
  @classmethod
  def prepareWeights(cls, segments, weightFld, apriWeightFld=None):
    return weightFld
  
  @classmethod
  def transferCoefs(cls, segdata):
    if 'awt' in segdata[0]:
      awt = lambda seg: seg['awt']
    else:
      awt = lambda seg: 1.0
    awtsums = collections.defaultdict(float)
    combsums = collections.defaultdict(float)
    for seg in segdata:
      mainID = seg['mainid']
      awtsums[mainID] += awt(seg)
      combsums[mainID] += seg['wt'] * awt(seg)
    return {mainid : awtsums[mainid] / combsums[mainid] for mainid in awtsums.iterkeys()}      
    
  
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