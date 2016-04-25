import json

import common
import modelhandling
import regression
import disaggregation
import features

# Core Reconfigurable Urban Modeler dispatcher.
# Combines the feature calculation, disaggregation, model training, application and validation.

DIRECT_LAYERS = {'landuse' : 'ua', 'poi' : 'poi', 'dem' : 'dem', 'buildings' : 'building', 'transport' : 'transport', 'barrier' : 'barrier'}

FEATURE_LAYERS = {'landuse' : features.LandUseLayer,
                  'poi' : features.POILayer,
                  'dem' : features.DEMLayer,
                  'buildings' : features.BuildingLayer,
                  'transport' : features.TransportLayer,
                  'barrier' : features.BarrierLayer}

ABSOLUTE_FIELD_DESC = 'absolute/relative training value switch'

def loadConfig(confFile):
  '''Loads RUM JSON configuration from a file.'''
  with open(confFile) as fileObj:
    return json.load(fileObj)

def directLayerDict(workspace):
  '''Given a workspace, returns a dictionary of paths to the direct layers
  that should be contained in it.'''
  dldict = {}
  for name, fileName in DIRECT_LAYERS.iteritems():
    dldict[name] = common.featurePath(workspace, fileName)
  return dldict

def trainModel(workspace, segments, train, trainValFld, modelType, featConfig, outFile, absolute=True):
  '''Routine for 0 - Train Model, includes segment intersection and feature calculation.'''
  tgtSegments, inFld = prepareTraining(workspace, segments, train, trainValFld, featConfig, absolute)
  calculateFeatures(tgtSegments, workspace, featConfig)
  trainFromSegments(tgtSegments, inFld, modelType, outFile)

def prepareTraining(workspace, segments, train, trainValFld, featConfig, absolute=True):
  
  if not segments:
    segments = common.featurePath(workspace, 'segments')
  tgtSegments = common.featurePath(workspace, 'segments_train_' + trainValFld)
  disag = disaggregation.create(absolute)
  common.progress('calculating training values')
  disag.overlayToValue(segments, train, trainValFld, tgtSegments)
  inFld = calcModelInput(disag, tgtSegments, trainValFld)
  return tgtSegments, inFld
  
def dualPrepareTraining(workspace, segments, trainPts, ptsValFld, trainPolys, polyValFld, featConfig, absolute=True):
  if not segments:
    segments = common.featurePath(workspace, 'segments')
  tgtSegments = common.featurePath(workspace, 'segments_train_' + ptsValFld)
  disag = disaggregation.create(absolute)
  with common.PathManager(tgtSegments) as pathman:
    common.progress('calculating training values from points')
    ptSegments = pathman.tmpFC()#delete=False, suffix='ptres')
    disag.overlayToValue(segments, trainPts, ptsValFld, ptSegments)
    common.progress('calculating training values from polygons')
    polySegments = pathman.tmpFC()#delete=False, suffix='polyres')
    disag.overlayToValue(segments, subtractedLayer(trainPolys, trainPts),
      polyValFld, polySegments)
    # remove from ptSegments those overlapped by polySegments
    mergeSegments([subtractedLayer(ptSegments, polySegments), polySegments],
        tgtSegments, [ptsValFld, polyValFld])
  inFld = calcModelInput(disag, tgtSegments, ptsValFld)
  return tgtSegments, inFld

def subtractedLayer(what, minus):
  import arcpy
  common.progress('subtracting')
  mLayer = 'subtracted'
  arcpy.MakeFeatureLayer_management(what, mLayer)
  arcpy.SelectLayerByLocation_management(mLayer, 'INTERSECT', minus)
  arcpy.SelectLayerByAttribute_management(mLayer, 'SWITCH_SELECTION')
  return mLayer

  
def calcModelInput(disag, segments, valFld):    
  common.progress('calculating model input values')
  inFld = valFld + '_IN'
  disag.modelInValue(segments, valFld, inFld)
  return inFld
   
# assuming the segments in seglist are spatially disjoint
def mergeSegments(segList, target, fields):
  import arcpy
  fm = arcpy.FieldMappings()
  fm.addFieldMap(common.multiFieldMap(segList, fields, 'FIRST'))
  arcpy.Merge_management(segList, target, fm)

  
def trainFromSegments(segmentsWithVals, inFld, modelType, outFile):
  model = regression.create(modelType)
  common.progress('training model')
  modelhandling.train(model, segmentsWithVals, inFld)
  model.serialize(outFile)

def calculateFeatures(tgtSegments, workspace, featConfig, layers=None):
  if layers is None:
    layers = FEATURE_LAYERS.values()
  else:
    layers = [FEATURE_LAYERS[key] for key in layers]
  common.progress('calculating features')
  features.calculate(tgtSegments, directLayerDict(workspace), loadConfig(featConfig), layers)
  
  
def applyModel(workspace, modelFile, segments, segWeightFld, disag, disagValFld, featConfig, absolute=True):
  tgtSegments, disagIDFld = prepareModeling(workspace, segments, segWeightFld, disag, disagValFld, absolute)
  calculateFeatures(tgtSegments, workspace, featConfig)
  applyToSegments(modelFile, tgtSegments, disagIDFld, disagValFld, segWeightFld, absolute)
  
def prepareModeling(workspace, segments, segWeightFld, disag, disagValFld, absolute=True):
  if not segments:
    segments = common.featurePath(workspace, 'segments')
  tgtSegments = common.featurePath(workspace, 'segments_model_' + disagValFld)
  common.progress('identifying disaggregation areas')
  disagIDFld = disaggregation.overlayToID(segments, disag, disagValFld, tgtSegments, segWeightFld, valueFldSuffix='_MAIN')
  return tgtSegments, disagIDFld
  
def applyToSegments(modelFile, segmentsWithIDs, disagIDFld, disagValFld, aprioriWeightFld, absolute=True):
  if disagValFld.endswith('_MAIN'):
    disagValFld = disagValFld[:-5]
  common.progress('applying regression model')
  model = regression.load(modelFile)
  modelWeightFld = disagValFld + '_MWT'
  modelhandling.apply(model, segmentsWithIDs, modelWeightFld)
  common.debug(disagValFld, modelWeightFld)
  common.progress('disaggregating')
  disag = disaggregation.create(absolute)
  disag.disaggregate(segmentsWithIDs, disagIDFld, disagValFld, modelWeightFld, aprioriWeightFld, mainFldSuffix='_MAIN')
    
    
def validate(segments, valueFld, valid, validFld, reportFile=None, absolute=True, keepTarget=True):
  if keepTarget:
    target, validFld = prepareValidation(segments, valueFld, valid, validFld, absolute=absolute)
  else:
    with common.PathManager(segments) as pathman:
      target = pathman.tmpFC()
      target, validFld = prepareValidation(segments, valueFld, valid, validFld, target=target, absolute=absolute)
  validateIntersected(target, valueFld, validFld, reportFile)

def prepareValidation(segments, valueFld, valid, validFld, target=None, absolute=True):
  if target is None:
    target = common.featurePath(common.location(segments), common.fcName(segments) + '_valid_' + validFld)
  if validFld == valueFld:
    common.copyField(valid, validFld, validFld + '_VALID')
    validFld += '_VALID'
  disag = disaggregation.create(absolute)
  disag.overlayValidate(segments, valueFld, valid, validFld, target)
  return target, validFld
  
  
def validateIntersected(layer, modelFld, realFld, reportFile=None):
  modelhandling.validate(layer, modelFld, realFld, reportFile)