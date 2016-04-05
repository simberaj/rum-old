import common
import loaders
import features
import regression

def featureFields(layer):
  fields = common.fieldList(layer)
  prefix = features.GLOBAL_PREFIX
  return sorted(fld for fld in fields if fld.startswith(prefix))
      
def load(layer, slots=None, id=False, dict=False, features=True, text=''):
  if slots is None:
    slots = {}
  if features:
    for feat in featureFields(layer):
      slots[feat] = feat
  if id:
    slots['id'] = common.ensureIDField(layer)
  readerClass = loaders.DictReader if dict else loaders.BasicReader
  return readerClass(layer, slots).read(text=(' '.join(('loading', text, 'data'))))
  
def save(data, layer, slots, idFld=None):
  if idFld is None: idFld = common.ensureIDField(layer)
  common.progress('saving data')
  if isinstance(data, list):
    data = {item['id'] : item for item in data}
  types = {key : float for key in slots}
  loaders.ObjectMarker(layer, {'id' : idFld}, slots, outTypes=types).mark(data)
  
def train(model, segments, valueFld, fit=False):
  data = load(segments, {'value' : valueFld}, text='training')
  common.debug(valueFld)
  common.debug(data[0])
  common.progress('initializing regression model')
  model.setFeatureNames(featureFields(segments))
  for seg in data:
    model.addDictExample(seg, 'value')
  common.progress('training regression model')
  model.train()

def apply(model, segments, saveFld):
  data = load(segments, id=True, text='feature')
  common.progress('applying regression model')
  model.setFeatureNames(featureFields(segments))
  modelFX = model.get()
  for seg in data:
    seg['value'] = float(modelFX(model.featuresToList(seg)))
  common.debug(data)
  save(data, segments, {'value' : saveFld})
  
def validate(segments, valueFld, validFld, reportFile=None):
  data = load(segments, id=True, features=False, slots={'model' : valueFld, 'real' : validFld, 'xy' : 'shape@xy'}, text='segment')
  regression.validate(data, 'model', 'real', outfile=reportFile, shapekey='xy')
