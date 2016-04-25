import neuro
import common
import pca

import numpy

import cPickle
import os
import math

class Model(object):
  def __init__(self, name=''):
    self.features = []
    self.results = []
    self.name = name + self.__class__.__name__[:-5] # strip Model, leave e.g. OLS
    self.trained = False
    self.featureNames = []
    self.weights = 1.0
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
    common.debug(names)
    self.checkFeatures()
    
  def setWeights(self, weights):
    self.weights = numpy.array(weights)
  
  def setWeightsAsDict(self, weights):
    self.weights = numpy.array([weights[feat] for feat in self.featureNames])
  
  def getFeatureNames(self):
    return self.featureNames
  
  def featuresToList(self, featDict):
    # common.debug([featDict[featName] for featName in self.featureNames])
    return [featDict[featName] for featName in self.featureNames]
    
  def getName(self):
    return self.name
    
  def getCoefficients(self):
    raise NotImplementedError
  
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
    return sum(abs(model[i] - data[i]) for i in range(len(data))) / sum(data)
  
  def get(self):
    if self.trained:
      return self.getFunction()
    else:
      raise RuntimeError, 'model not yet trained'
  
  def compute(self, features):
    return self.get()(features)
  
  def train(self, **kwargs):
    self._findExcludedFeatures()
    self._train(self._prepareFeatures(), self._prepareTargets(), **kwargs)
  
  def _findExcludedFeatures(self):
    feats = numpy.array(self.features)
    self.excluded = self._zeroIndexes(feats)
    self.featureNames = [self.featureNames[i] for i in xrange(len(self.featureNames)) if i not in self.excluded]
    # return numpy.delete(feats, self.excluded, 1)
    
  @staticmethod
  def _zeroIndexes(array):
    cols = array.shape[1]
    colsums = abs(array.sum(axis=0)) < 1e-10 # column sums
    zerois = []
    for i in xrange(cols):
      if colsums[i]:
        zerois.append(i)
    return zerois
  
  def _prepareFeatures(self):
    feats = numpy.delete(numpy.array(self.features), self.excluded, 1)
    # print(feats)
    maxims = feats.max(axis=0) # column maxima
    self.normalizers = self.weights / maxims
    common.debug(feats.shape)
    self.decorrelator = pca.PCA(normalize=False)
    # print(feats * self.normalizers)
    transformed = self.decorrelator.feed(feats * self.normalizers)['transformed']
    # print(transformed)
    common.debug(self.featureNames)
    common.debug('explvar', self.decorrelator.varianceExplained)
    common.debug('eigvec', self.decorrelator.eigenvectors)
    # common.debug(self.decorrelator.transNorm.shape)
    nrows = transformed.shape[0]
    return numpy.append(transformed, numpy.ones((nrows, 1)), axis=1)
  
  def _prepareTargets(self):
    return numpy.array(self.results)
    
  @staticmethod
  def deserialize(file):
    with open(file) as fileobj:
      return cPickle.load(fileobj)

class OLSModel(Model):
  def _train(self, input, train, **kwargs):
    coefs = numpy.asarray(self.solve(input, train)).flatten()
    self.coefs = coefs[:-1]
    self.absolute = coefs[-1]
    self.trained = True
    # common.debug(len(self.featureNames), len(self.coefs))
    # common.debug([(self.featureNames[i], self.coefs[i]) for i in range(len(self.coefs))])
  
  def _prepareTargets(self):
    return Model._prepareTargets(self).reshape(len(self.results), 1)
    
  def checkFeatures(self):
    if self.coefs and len(self.featureNames) != len(self.coefs):
      raise ValueError, 'feature count ({}) does not match trained model coefficients ({})'.format(len(self.featureNames), len(self.coefs))
    
  def getFunction(self):
    coefs = self.coefs
    decor = self.decorrelator.transform
    dele = numpy.delete
    excl = self.excluded
    ar = numpy.array
    abso = self.absolute
    # common.debug(coefs.shape)
    # common.debug(self.decorrelator.mean.shape)
    # common.debug(decor.transform(ar([0] * coefs.shape[0])))
    # common.debug(coefs.dot(decor.transform(ar([0] * coefs.shape[0]))))
    # def fx(row):
      # common.debug('row len', len(row))
      # return coefs.dot(row) + self.absolute
    return lambda row: coefs.dot(decor(ar(row))) + abso
    # return fx
  
  @staticmethod
  def solve(input, train):
    inputT = input.transpose()
    return numpy.linalg.inv(inputT.dot(input)).dot(inputT).dot(train)
  
  # def _unknownsToCoefs(self, unknowns):
    # for i in self.excluded:
      # unknowns.insert(i, 0.0)
    # return unknowns[:-1], unknowns[-1]
  
  def getCoefficients(self):
    return list(self.coefs) + [self.absolute]

    
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
    AA = Atrans.dot(A)
    Ay = Atrans.dot(y)
    # initial approximation - TODO: he uses something different!
    # print(A)
    # print(AA)
    # print(y)
    x = OLSModel.solve(A, y)
    # variance calc
    resid = y - A.dot(x)
    omega = float(1 / ((numpy.asarray(resid) ** 2).sum() / len(resid)))
    oldOmega = omega
    sigmaX = numpy.diag(abs(x) * 0.1 + 0.01 * x.sum() / len(x))
    iter = 0
    while iter < cls.MAX_ITER:
      varX = numpy.asarray(x) ** 2 + numpy.diag(sigmaX)
      omegaX = numpy.diag((cls.ALPHA_0 + 0.5) / (cls.ALPHA_0 + 0.5 * varX))
      sigmaX = numpy.linalg.inv(omega * AA + omegaX)
      x = omega * sigmaX.dot(Ay)
      resid = y - A.dot(x)
      omega = float(len(y) / (resid.transpose().dot(resid) + numpy.diag(sigmaX.dot(AA)).sum()))
      # print(omega)
      iter += 1
      # if abs(oldOmega - omega) < omega / 1e4:
        # iter = cls.MAX_ITER
      oldOmega = omega
    return x
    
class GRModel(Model):
  SAMPLING_MAX = 1000

  def __init__(self, sigma=0.2, **kwargs):
    Model.__init__(self, **kwargs)
    self.sigma = sigma

  def _train(self, feats, tgts):
    self.feats = feats
    print self.feats
    self.tgts = tgts
    self.sampling = max(int(len(tgts) / self.SAMPLING_MAX), 1)
    self.sigma = self._determineSigma()
    print 'sigma', self.sigma
    self.trained = True
  
  def _determineSigma(self):
    from scipy.optimize import minimize_scalar
    # self.samplI = 0
    # determine the kernel range by cross-validation
    # raise RuntimeError
    minres = minimize_scalar(self._getCVError, bracket=(0.01, 1.0), tol=1e-6, options={'maxiter' : 50, 'disp' : True}) # 
    print minres
    return float(minres.x)
    
    
  def _getCVError(self, sigma):
    common.debug('iter', sigma)
    err = 0.0
    feats = self.feats
    tgts = self.tgts
    for i in xrange(0,feats.shape[0],self.sampling):
      wts = numpy.exp(-abs(feats - feats[i,:][numpy.newaxis,:]).sum(axis=1) / sigma)
      err += abs(((wts * tgts).sum() - tgts[i]) / wts.sum() - tgts[i])
    # self.samplI = (self.samplI + 1) % self.sampling
    common.debug('iter finished')
    return err
  
  def getFunction(self):
    feats = self.feats
    tgts = self.tgts
    sigma = self.sigma
    ar = numpy.array
    decor = self.decorrelator
    def result(featvec):
      wts = numpy.exp(-abs(feats - numpy.append(decor.transform(featvec * self.normalizers), 1)[numpy.newaxis,:]).sum(axis=1) / sigma)
      return (wts * tgts).sum() / wts.sum()
    return result
    
class ANNMLPModel(Model):
  INLAYER_COEF = 2

  def _train(self, feats, tgts, epochs=100, mindiff=1e-3, seed=True):
    featCount = len(feats[0])
    self.net = self._createNet(featCount)
    if seed:
      sCoefs = self._seedCoefs(feats, tgts)
      common.debug(sCoefs)
      self.net.seed(sCoefs)
    self.net.train(feats, numpy.asarray(tgts).flatten(), maxiter=epochs)
    self.trained = True
  
  @staticmethod
  def _seedCoefs(feats, tgts):
    return ARDModel.solve(feats, tgts.reshape(tgts.size, 1)).flatten()
    
  def getFunction(self):
    infx = self.net.get()
    ar = numpy.array
    ap = numpy.append
    decor = self.decorrelator.transform
    return lambda row: float(infx(ap(decor(ar(row)), [1])))
    
  def _createNet(self, featCount):
    net = neuro.NeuralNetwork([neuro.InputLayer(featCount + 1), neuro.TanhLayer(self.INLAYER_COEF * featCount), neuro.TanhLayer(1)])
    # net = neuro.NeuralNetwork([neuro.InputLayer(featCount), neuro.TransferLayer(self.INLAYER_COEF * featCount), neuro.TransferLayer(1)])
    return net

    
MODELS = {'OLS' : OLSModel, 'ARD' : ARDModel, 'ANN' : ANNMLPModel, 'GR' : GRModel}   
    
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
    ('rmse', 'Root mean square error (RMSE)'),
    ('tae', 'Total absolute error (TAE)'),
    ('rtae', 'Relative total absolute error (RTAE)'),
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
  FORMATS = {'mismatch' : '{:g}', 'tae' : '{:.0f}', 'rtae' : '{:.2%}', 'r2' : '{:.3%}', 'rmse' : '{:.2f}'}
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
    self.sqResidSum = (self.absResid ** 2).sum()
    self.tae = self.absResid.sum()
    self.rtae = self.tae / self.realSum
    self.r2 = 1 - self.sqResidSum / ((self.reals - self.realMean) ** 2).sum()
    common.debug(self.sqResidSum, self.realMean, ((self.reals - self.realMean) ** 2).sum())
    self.rmse = math.sqrt(self.sqResidSum / len(self.reals))
    common.debug('TAE ', self.tae)
    common.debug('RTAE', self.rtae)
    common.debug('R2  ', self.r2)
    common.debug('RMSE', self.rmse)
  
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
    return dict(mismatch=self.mismatch, tae=self.tae, rtae=self.rtae, r2=self.r2, rmse=self.rmse)
    
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
    plt.gca().set_aspect(True)
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
      models.append(0 if item[modelkey] is None else item[modelkey])
      reals.append(0 if item[realkey] is None else item[realkey])
      if shapekey:
        shapes.append(item[shapekey])
    return numpy.array(models), numpy.array(reals), numpy.array(shapes) if shapekey else None
    
if __name__ == '__main__':
  atest = numpy.arange(0,10,0.2)
  print(atest)
  Atest = numpy.array([(x, x ** 2, x ** 3) for x in atest], dtype=float)
  ktrue = numpy.array((2,0,0.5))
  # print(Atest, ktrue, Atest.dot(ktrue.transpose()))
  ytrain = Atest.dot(ktrue.transpose()) + numpy.random.random_sample(len(atest))
  gr = GRModel(sigma=0.1)
  # gr = OLSModel()
  gr.addExamples(Atest, ytrain)
  gr.train()
  grfx = gr.get()
  models = []
  reals = []
  for row in Atest:
    models.append(grfx(row))
    reals.append((row * ktrue).sum())
    print row[0], models[-1], reals[-1]
  print (((numpy.array(models) - numpy.array(reals)) ** 2).sum() / len(models)) ** 0.5
    
    
  
  # # print(Atest.dot(ktrue.transpose()), ytrain)
  # ols = OLSModel()
  # ols.addExamples(Atest, ytrain)
  # ols.train()
  # print(ols.coefs, ols.absolute)
  # print(ols.report())
  # sls = ARDModel()
  # sls.addExamples(Atest, ytrain)
  # sls.train()
  # print(sls.coefs, sls.absolute)
  # print(sls.report())
  # nn = ANNMLPModel()
  # nn.addExamples(Atest, ytrain)
  # nn.train()
  # print(nn.errors)
  # print(nn.report())
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