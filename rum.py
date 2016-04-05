import json

import common
import modelhandling
import regression
import disaggregation
import features

DIRECT_LAYERS = {'landuse' : 'ua', 'poi' : 'poi', 'dem' : 'dem', 'buildings' : 'building', 'transport' : 'transport', 'barrier' : 'barrier'}

ABSOLUTE_FIELD_DESC = 'absolute/relative training value switch'

def loadConfig(confFile):
  with open(confFile) as fileObj:
    return json.load(fileObj)

def directLayerDict(workspace):
  dldict = {}
  for name, fileName in DIRECT_LAYERS.iteritems():
    dldict[name] = common.featurePath(workspace, fileName)
  return dldict

def trainModel(workspace, segments, train, trainValFld, modelType, featConfig, outFile, absolute=True):
  tgtSegments, inFld = prepareTraining(workspace, segments, train, trainValFld, featConfig, absolute)
  # train the model
  trainFromSegments(tgtSegments, inFld, modelType, outFile)

def prepareTraining(workspace, segments, train, trainValFld, featConfig, absolute=True):
  if not segments:
    segments = common.featurePath(workspace, 'segments')
  tgtSegments = common.featurePath(workspace, 'segments_train_' + trainValFld)
  disag = disaggregation.create(absolute)
  common.progress('calculating training values')
  disag.overlayToValue(segments, train, trainValFld, tgtSegments)
  common.progress('calculating model input values')
  inFld = trainValFld + '_IN'
  disag.modelInValue(tgtSegments, trainValFld, inFld)
  common.progress('calculating features')
  features.calculate(tgtSegments, directLayerDict(workspace), loadConfig(featConfig))
  return tgtSegments, inFld
  
def trainFromSegments(segmentsWithVals, inFld, modelType, outFile):
  model = regression.create(modelType)
  common.progress('training model')
  modelhandling.train(model, segmentsWithVals, inFld)
  model.serialize(outFile)
  
  
def applyModel(workspace, modelFile, segments, segWeightFld, disag, disagValFld, featConfig, absolute=True):
  tgtSegments, disagIDFld = prepareModeling(workspace, segments, segWeightFld, disag, disagValFld, featConfig, absolute)
  applyToSegments(modelFile, tgtSegments, disagIDFld, disagValFld, segWeightFld, absolute)
  
def prepareModeling(workspace, segments, segWeightFld, disag, disagValFld, featConfig, absolute=True):
  if not segments:
    segments = common.featurePath(workspace, 'segments')
  tgtSegments = common.featurePath(workspace, 'segments_model_' + disagValFld)
  common.progress('identifying disaggregation areas')
  disagIDFld = disaggregation.overlayToID(segments, disag, disagValFld, tgtSegments, segWeightFld, valueFldSuffix='_MAIN')
  common.progress('calculating features')
  features.calculate(tgtSegments, directLayerDict(workspace), loadConfig(featConfig))
  return tgtSegments, disagIDFld
  
def applyToSegments(modelFile, segmentsWithIDs, disagIDFld, disagValFld, aprioriWeightFld, absolute=True):
  common.progress('applying regression model')
  model = regression.load(modelFile)
  modelWeightFld = disagValFld + '_MWT'
  modelhandling.apply(model, segmentsWithIDs, modelWeightFld)
  common.progress('disaggregating')
  disag = disaggregation.create(absolute)
  disag.disaggregate(segmentsWithIDs, disagIDFld, disagValFld, modelWeightFld, aprioriWeightFld, mainFldSuffix='_MAIN')
    
    
def validate(segments, valueFld, valid, validFld, reportFile=None, absolute=True, keepTarget=True):
  if keepTarget:
    target = common.featurePath(common.location(segments), common.fcName(segments) + '_valid_' + valueFld)
    validateInner(segments, valueFld, valid, validFld, target, reportFile, absolute)    
  else:
    with common.PathManager(segments) as pathman:
      target = pathman.tmpFC()
      validateInner(segments, valueFld, valid, validFld, target, reportFile, absolute)

def validateInner(segments, valueFld, valid, validFld, target, reportFile=None, absolute=True):
  if validFld == valueFld:
    common.copyField(valid, validFld, validFld + '_VALID')
    validFld += '_VALID'
  disag = disaggregation.create(absolute)
  disag.overlayValidate(segments, valueFld, valid, validFld, target)
  modelhandling.validate(target, valueFld, validFld, reportFile)
  
def validateIntersected(layer, modelFld, realFld, reportFile=None):
  modelhandling.validate(layer, modelFld, realFld, reportFile)