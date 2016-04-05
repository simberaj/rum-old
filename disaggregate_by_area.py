import common
import disaggregation

with common.runtool(4) as parameters:
  layer, mainIDFld, mainFld, tgtFld = parameters
  areaFld = common.ensureShapeAreaField(layer)
  disaggregation.Disaggregator.disaggregateInPlace(layer, mainIDFld, mainFld, areaFld, tgtFld, shout=True)