import neuro
import common

import numpy

import cPickle
import os

class Model(object):
  def __init__(self, name=''):
    self.features = []
    self.results = []
    self.name = name + self.__class__.__name__[:-5] # strip Model, leave e.g. OLS
    self.trained = False
    self.featureNames = []
    self.coefs = None
    # self.function = None
  
  def addExample(self, features, result):
    self.features.append(features)
    self.results.append(result)
  
  def addDictExample(self, dict, valueFld):
    self.addExample([dict[feat] for feat in self.featureNames], dict[valueFld])
  
  def addExamples(self, features, results):
    self.features.extend(features)
    self.results.extend(results)
    
  def setFeatureNames(self, names):
    self.featureNames = names
    self.checkFeatures()
  
  def featuresToList(self, featDict):
    common.debug([featDict[featName] for featName in self.featureNames])
    return [featDict[featName] for featName in self.featureNames]
    
  def getName(self):
    return self.name
    
  def train(self):
    pass
  
  def checkFeatures(self):
    pass
   
  def serialize(self, file, keepData=False):
    if not keepData:
      self.features = []
      self.results = []
    cPickle.dump(self, open(file, 'w'))
  
  def report(self):
    return self.check(self.features, self.results)
  
  def check(self, features, results):
    fx = self.get()
    return self.r2(results, [fx(feat) for feat in features])
  
  @staticmethod
  def r2(model, data):
    ncheck = len(data)
    resMean = sum(data) / ncheck
    ssres = 0.0
    sstot = 0.0
    for i in range(len(data)):
      ssres += (data[i] - model[i]) ** 2
      sstot += (data[i] - resMean) ** 2
    return 1 - (ssres / sstot)
  
  @staticmethod
  def tae(model, data):
    return sum(abs(model[i] - data[i]) for i in range(len(data)))
  
  @staticmethod
  def rtae(model, data): # gives [0,1]
    return 0.5 * sum(abs(model[i] - data[i]) for i in range(len(data))) / sum(data)
  
  def get(self):
    if self.trained:
      return self.getFunction()
    else:
      raise RuntimeError, 'model not yet trained'
  
  def compute(self, features):
    return self.get()(features)
  
  def train(self):
    self._train(self._prepareFeatures(), self._prepareTargets())
  
  def _prepareFeatures(self):
    input = numpy.matrix([list(row) + [1] for row in self.features])
    # exclude zero features
    self.excluded = []
    cols = input.shape[1]
    for i in xrange(cols):
      if input[:,i].sum() == 0:
        self.excluded.append(i)
    return numpy.delete(input, self.excluded, 1)
  
  def _prepareTargets(self):
    return numpy.matrix(self.results).reshape(len(self.results), 1)
  
  @staticmethod
  def deserialize(file):
    with open(file) as fileobj:
      return cPickle.load(fileobj)

class OLSModel(Model):
  def _train(self, input, train):
    unknowns = list(numpy.asarray(self.solve(input, train)).flatten())
    self.coefs, self.absolute = self._unknownsToCoefs(unknowns)
    self.trained = True
    # common.debug(len(self.featureNames), len(self.coefs))
    # common.debug([(self.featureNames[i], self.coefs[i]) for i in range(len(self.coefs))])
    common.debug(self.coefs)
    common.debug(self.featureNames)
    common.debug(self.absolute)

    
  def checkFeatures(self):
    if self.coefs and len(self.featureNames) != len(self.coefs):
      raise ValueError, 'feature count ({}) does not match trained model coefficients ({})'.format(len(self.featureNames), len(self.coefs))
    
  def getFunction(self):
    coefs = numpy.array(self.coefs)
    # def fx(row):
      # common.debug('row len', len(row))
      # return coefs.dot(row) + self.absolute
    return lambda row: coefs.dot(row) + self.absolute
    # return fx
  
  @staticmethod
  def solve(input, train):
    inputT = input.transpose()
    return (inputT * input).I * inputT * train
  
  def _unknownsToCoefs(self, unknowns):
    for i in self.excluded:
      unknowns.insert(i, 0.0)
    return unknowns[:-1], unknowns[-1]

    
class ARDModel(OLSModel):
  ALPHA_0 = 1e-10
  MAX_ITER = 100

  @classmethod
  def solve(cls, A, y):
    # y = Ax + e
    # rescaling to make stopping constants meaningful
    maxy = abs(y).max()
    A /= maxy
    y /= maxy
    Atrans = A.transpose()
    AA = Atrans * A
    Ay = Atrans * y
    # initial approximation - TODO: he uses something different!
    # print(A)
    # print(AA)
    # print(y)
    x = OLSModel.solve(A, y)
    # variance calc
    resid = y - A * x
    omega = float(1 / ((numpy.asarray(resid) ** 2).sum() / len(resid)))
    oldOmega = omega
    sigmaX = numpy.diag(abs(x) * 0.1 + 0.01 * x.sum() / len(x))
    iter = 0
    while iter < cls.MAX_ITER:
      varX = numpy.asarray(x) ** 2 + numpy.diag(sigmaX)
      omegaX = numpy.diag((cls.ALPHA_0 + 0.5) / (cls.ALPHA_0 + 0.5 * varX))
      sigmaX = (omega * AA + omegaX).I
      x = omega * sigmaX * Ay
      resid = y - A * x
      omega = float(len(y) / (resid.transpose() * resid + numpy.diag(sigmaX * AA).sum()))
      # print(omega)
      iter += 1
      # if abs(oldOmega - omega) < omega / 1e4:
        # iter = cls.MAX_ITER
      oldOmega = omega
    return x
    
class ANNMLPModel(Model):
  INLAYER_COEF = 2
  # INLAYER2_COEF = 1

  def _train(self, feats, tgts, epochs=100, mindiff=1e-3):
    feats = numpy.asarray(feats)
    featCount = len(feats[0])
    self.net = self._createNet(featCount)
    # print('inmodel', tgts.shape, numpy.asarray(tgts).flatten().shape)
    self.net.train(numpy.asarray(feats), numpy.asarray(tgts).flatten(), maxiter=epochs)
    self.trained = True
    
  def getFunction(self):
    infx = self.net.get()
    return lambda row: infx(row + [1])
    # from pybrain.datasets import SupervisedDataSet
    # from pybrain.supervised.trainers import BackpropTrainer
    # traindata = SupervisedDataSet(featCount, 1)
    # for i in range(len(tgts)):
      # print(list(feats[i].flatten()), float(tgts[i]))
      # traindata.addSample(tuple(feats[i].flatten()), float(tgts[i]))
    # trainer = BackpropTrainer(net, traindata)
    # errors = [1e20]
    # for i in range(epochs):
      # errors.append(trainer.train())
      # if abs(errors[-1] - errors[-2]) < mindiff:
        # break
    # self.errors = errors[1:]
    # def fx(row):
      # print('activating', list(row) + [1], net.activate(list(row) + [1]))
      # return net.activate(list(row) + [1])
    # # self.function = lambda row: net.activate(list(row) + [1])
    # self.function = fx
    # self.trained = True
  
  def _createNet(self, featCount):
    net = neuro.NeuralNetwork([neuro.InputLayer(featCount), neuro.TanhLayer(self.INLAYER_COEF * featCount), neuro.TanhLayer(1)])
    return net
    # import pybrain
    # from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
    # net = FeedForwardNetwork()
    # inLayer = LinearLayer(featCount)
    # hiddenLayer = SigmoidLayer(self.INLAYER1_COEF * featCount)
    # hiddenLayer2 = SigmoidLayer(self.INLAYER2_COEF * featCount)
    # outLayer = LinearLayer(1)
    # net.addInputModule(inLayer)
    # net.addModule(hiddenLayer)
    # net.addModule(hiddenLayer2)
    # net.addOutputModule(outLayer)
    # in_to_hidden = FullConnection(inLayer, hiddenLayer)
    # hh2 = FullConnection(hiddenLayer, hiddenLayer2)
    # hidden_to_out = FullConnection(hiddenLayer2, outLayer)
    # net.addConnection(in_to_hidden)
    # net.addConnection(hh2)
    # net.addConnection(hidden_to_out)
    # net.sortModules()
    # return net
   
MODELS = {'OLS' : OLSModel, 'ARD' : ARDModel, 'ANN' : ANNMLPModel}   
    
def create(modelType):
  try:
    return MODELS[modelType]()
  except KeyError:
    raise KeyError, 'model type {} not found'.format(modelType)
    
def load(modelFile):
  return Model.deserialize(modelFile)
    
def validate(data, modelkey, realkey, outfile=None, shapekey=None):
  vdtor = Validator.fromDict(data, modelkey, realkey, shapekey)
  vdtor.validate()
  if outfile:
    vdtor.output(outfile)
  
class Validator:
  TEMPLATE_FILE = 'report.html'
  ERRAGG_NAMES = [
    ('mismatch', 'Absolute value sum difference'),
    ('tae', 'Total absolute error (TAE)'),
    ('rtae', 'Relative total absolute error'),
    ('r2', 'Coefficient of determination (R<sup>2</sup>)')]
  DESC_NAMES = [
    ('set', 'Dataset'),
    ('sum', 'Sum'),
    ('min', 'Minimum'),
    ('q2l', 'Q2,5'),
    ('q10', 'Q10 - 1st Decile'),
    ('q25', 'Q25 - 1st Quartile'),
    ('median', 'Q50 - Median'),
    ('mean', 'Mean'),
    ('q75', 'Q75 - 3rd Quartile'),
    ('q90', 'Q90 - 9th Decile'),
    ('q2h', 'Q97,5'),
    ('max', 'Maximum')]
  FORMATS = {'mismatch' : '{:g}', 'tae' : '{:.0f}', 'rtae' : '{:.2%}', 'r2' : '{:.3%}'}
  FORMATS.update({item[0] : '{:g}' for item in DESC_NAMES})

  def __init__(self, models, reals, shapes=None):
    self.models = models
    self.reals = reals
    self.shapes = shapes
    common.debug(self.models[:10])
    common.debug(self.reals[:10])
  
  @classmethod
  def fromDict(cls, data, modelkey, realkey, shapekey=None):
    return cls(*cls.transform(data, modelkey, realkey, shapekey))
  
  def validate(self):
    self.realSum = self.reals.sum()
    self.realMean = self.realSum / len(self.reals)
    self.modelSum = self.models.sum()
    self.mismatch = self.modelSum - self.realSum
    self.resid = self.models - self.reals
    self.absResid = abs(self.resid)
    self.tae = self.absResid.sum()
    self.rtae = 0.5 * self.tae / self.realSum
    self.r2 = 1 - (self.absResid ** 2).sum() / ((self.models - self.realMean) ** 2).sum()
    common.debug('TAE ', self.tae)
    common.debug('RTAE', self.rtae)
    common.debug('R2  ', self.r2)
  
  def describe(self, data):
    return {'min' : data.min(), 'max' : data.max(), 'sum' : data.sum(), 'mean' : data.mean(), 'median' : numpy.median(data), 'q25' : numpy.percentile(data, 25), 'q75' : numpy.percentile(data, 75), 'q10' : numpy.percentile(data, 10), 'q90' : numpy.percentile(data, 90), 'q2l' : numpy.percentile(data, 2.5), 'q2h' : numpy.percentile(data, 97.5)}
  
  def descriptions(self):
    descs = []
    for name, data in (('Modeled', self.models), ('Real', self.reals), ('Residuals', self.resid), ('Abs(Residuals)', self.absResid)):
      desc = self.format(self.describe(data))
      desc['set'] = name
      descs.append(desc)
    return descs
  
  def globals(self):
    return dict(mismatch=self.mismatch, tae=self.tae, rtae=self.rtae, r2=self.r2)
    
  def format(self, fdict):
    return {key : self.FORMATS[key].format(float(value)).replace('.', ',') for key, value in fdict.iteritems()}
  
  def output(self, fileName):
    import html
    try:
      with open(os.path.join(os.path.dirname(__file__), self.TEMPLATE_FILE)) as templFile:
        template = templFile.read()
    except IOError:
      raise IOError, 'report template file not found'
    template = template.replace('{', '[').replace('}', ']').replace('[[', '{').replace(']]', '}')
    text = template.format(
        erragg=html.dictToTable(self.format(self.globals()), self.ERRAGG_NAMES),
        setdescstat=html.dictToTable(self.descriptions(), self.DESC_NAMES, rowHead=True)
    )
    with open(fileName, 'w') as outfile:
      outfile.write(text.replace('[', '{').replace(']', '}'))
    try:
      import matplotlib
      self.outputImages(os.path.dirname(fileName))
    except ImportError:
      common.message('Matplotlib unavailable, skipping image output')
  
  def outputImages(self, directory):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(self.reals, self.models, 'b.')
    plt.xlabel('Real values')
    plt.ylabel('Modeled values')
    plt.savefig(os.path.join(directory, 'correl.png'), bbox_inches='tight')
    
    
  @staticmethod
  def transform(data, modelkey, realkey, shapekey=None):
    models = []
    reals = []
    if shapekey:
      shapes = []
    for item in data:
      models.append(item[modelkey])
      reals.append(item[realkey])
      if shapekey:
        shapes.append(item[shapekey])
    return numpy.array(models), numpy.array(reals), numpy.array(shapes) if shapekey else None
    
if __name__ == '__main__':
  atest = numpy.arange(0,10,0.5)
  print(atest)
  Atest = numpy.array([(x, x ** 2, x ** 3) for x in atest], dtype=float)
  ktrue = numpy.array((2,0,0.5))
  # print(Atest, ktrue, Atest.dot(ktrue.transpose()))
  ytrain = Atest.dot(ktrue.transpose()) + numpy.random.random_sample(len(atest))
  # print(Atest.dot(ktrue.transpose()), ytrain)
  ols = OLSModel()
  ols.addExamples(Atest, ytrain)
  ols.train()
  print(ols.coefs, ols.absolute)
  print(ols.report())
  sls = ARDModel()
  sls.addExamples(Atest, ytrain)
  sls.train()
  print(sls.coefs, sls.absolute)
  print(sls.report())
  nn = ANNMLPModel()
  nn.addExamples(Atest, ytrain)
  nn.train()
  print(nn.errors)
  print(nn.report())
  # tryinp = [
    # (0.5, 0.2, 0.3),
    # (0.1, 0.4, 0.5),
    # (0.3, 0.1, 0.6),
    # (0.7, 0.1, 0.2),
    # (0.4, 0.4, 0.2),
    # (0.3, 0.4, 0.3)
  # ]
  # trytrain = [15.0, 8.0, 7.0, 25.0, 16.0, 12.0]
  # check = [
    # (0.7, 0.2, 0.1),
    # (0.1, 0.1, 0.8),
    # (0.5, 0.1, 0.4)
  # ]
  # checkres = [30, 1, 12]
  # tryinp = [(0.1, 0.2), (0.5, 0.6), (0.4, 0.1), (0.7, 0.3), (0.8, 0.2), (0.3, 0.6)]
  # trytrain = [float(x) for x in [20, 120, 40, 110, 90, 100]]
  # check = [(0.2, 0.5), (0.8, 0.7), (0.4, 0.8), (0.9, 0.1)]
  # checkres = [70, 150, 120, 100]
  # model = OLSModel()
  # model.addExamples(tryinp, trytrain)
  # model.train()
  # print(model.report())
  # print(model.check(check, checkres))
  # print(model.compute((0.3, 0.8)))