import numpy

class Model:
  def __init__(self):
    self.features = []
    self.results = []
    self.trained = False
    self.function = None
  
  def addExample(self, features, result):
    self.features.append(features)
    self.results.append(result)
  
  def addExamples(self, features, results):
    self.features.extend(features)
    self.results.extend(results)
    
  def train(self):
    pass
  
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
      return self.function
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

class OLSModel(Model):
  def _train(self, input, train):
    unknowns = list(numpy.asarray(self.solve(input, train)).flatten())
    self.coefs, self.absolute = self._unknownsToCoefs(unknowns)
    self.function = lambda row: numpy.array(row).dot(self.coefs) + self.absolute
    self.trained = True
  
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
    net = self._createNet(featCount)
    net.train(feats, tgts, maxiter=epochs)
    infx = net.get()
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
    import neuro
    net = neuro.NeuralNetwork([neuro.InputLayer(featCount), neuro.SoftplusLayer(self.INLAYER_COEF * featCount), neuro.SoftplusLayer(1)])
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