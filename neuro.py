from __future__ import print_function
import numpy

class NeuronLayer:
  def __init__(self, size):
    self.size = size
    self.next = None
  
  def getSize(self):
    return self.size
  
  def __call__(self, values):
    return values
  
  def __repr__(self):
    return self.__class__.__name__ + '({})'.format(self.size)

  def connect(self, layer):
    self.next = layer
    layer.precede(self)
  

class TransferLayer(NeuronLayer):
  @staticmethod
  def activate(values):
    return values
  
  @staticmethod
  def inverse(values):
    return numpy.ones(values.size)
  
  def precede(self, layer):
    self.previous = layer
    self.weights = numpy.random.random_sample((self.getSize(), layer.getSize()))
  
  def __call__(self, values):
    return self.activate(self.weights.dot(self.previous(values)))
  
  def train(self, input, target, rate, shout=True):
    # print(shout)
    if shout: print('input', input, target, rate)
    # print('TRAINING', self.size)
    if shout: print('weights', self.weights)
    weighted = self.weights.dot(input)
    if shout: print('weighted', weighted)
    myOutput = self.activate(weighted)
    if shout: print('myoutput', myOutput)
    if self.next is None:
      error = target - myOutput
    else:
      error = self.next.train(myOutput, target, rate, shout)
    if shout: print('error', error)
    delta = self.inverse(weighted) * error
    if shout: print('delta', delta)
    # if shout:
      # # print(self.weights)
      # print(self.size, self.weights, input, myOutput, target, weighted, delta)
      # # print(rate, delta, input, delta[:,numpy.newaxis], input[numpy.newaxis,:])
      # print(rate * delta[:,numpy.newaxis] * input[numpy.newaxis,:])
    # adjust my weights
    # print(rate, weighted, delta, delta[:,numpy.newaxis], input[numpy.newaxis,:])
    # raise RuntimeError
    self.weights += rate * delta[:,numpy.newaxis] * input[numpy.newaxis,:]
    if shout: print('newweights', self.weights)
    # backpropagate the error
    return delta.dot(self.weights)
  
  def __repr__(self):
    return NeuronLayer.__repr__(self) + '\n  ' + '\n  '.join((repr(self.weights) + '\n' + repr(self.previous)).split('\n'))
  
class SoftplusLayer(TransferLayer):
  @staticmethod
  def activate(values):
    return numpy.log(1 + numpy.exp(values))
  
  @staticmethod
  def inverse(values):
    exps = numpy.exp(values)
    return exps / (1 + exps)
    
class QuadraticLayer(TransferLayer):
  @staticmethod
  def activate(values):
    return values ** 2
  
  @staticmethod
  def inverse(values):
    return 2 * values
    
class SigmoidLayer(TransferLayer):
  @staticmethod
  def activate(values):
    return 1 / (1 + numpy.exp(-values))
  
  @staticmethod
  def inverse(values):
    exps = numpy.exp(-values)
    return exps / (1 + exps) ** 2

class TanhLayer(TransferLayer):
  @staticmethod
  def activate(values):
    exps = numpy.exp(2 * values)
    return (exps - 1) / (exps + 1)
  
  @staticmethod
  def inverse(values):
    exps = numpy.exp(values)
    return 4 / (exps + 1 / exps) ** 2
    
class InputLayer(NeuronLayer):
  def train(self, input, target, rate, shout=False):
    self.next.train(input, target, rate, shout)
    
  
class NeuralNetwork:
  SEED_RANDOMIZATION_VARIANCE = 0.25

  def __init__(self, layers):
    self.input = layers[0]
    for i in range(len(layers)-1):
      layers[i].connect(layers[i+1])
    self.output = layers[-1]
    self._layers = layers
    # self.inpmax = 1.0
    # self.outmax = 1.0
    self._seedCoefs = None
  
  def __repr__(self):
    return repr(self.output)
  
  def seed(self, coefs):
    import common
    common.debug('SEEDING INPUT', coefs)
    self._seedCoefs = coefs
  
  def _seed(self):
    import common
    # common.debug('SEEDING', self._seedCoefs)
    if self._seedCoefs is not None:
      # print(self._seedCoefs)
      # print(self.inpmax)
      # print(self.outmax)
      coefs = self._seedCoefs # * self.inpmax / self.outmax
      # print(coefs)
      # print(self._layers[1].weights)
      # print(self._layers[1].weights.dot(numpy.array([1,1,1,1,1])))
      self._layers[1].weights = self.randomizedCoefs(coefs, self._layers[1].getSize())
      for i in range(2, len(self._layers)):
        prevSize = self._layers[i-1].getSize()
        self._layers[i].weights = self.randomizedCoefs(
            numpy.array([1.0 / prevSize] * prevSize), self._layers[i].getSize())
    self.debugWeights()
    # print(self.
  
  def debugWeights(self):
    import common
    for i in range(1, len(self._layers)):
      common.debug(i, self._layers[i].weights)
      
  def randomizedCoefs(cls, coefs, layerSize):
    return coefs[numpy.newaxis,:] + \
      numpy.random.randn(layerSize, coefs.size) * coefs * cls.SEED_RANDOMIZATION_VARIANCE
  
  def train(self, inputs, outputs, rate=0.01, maxiter=20):
    # print('MAIN TRAINER', inputs.shape, outputs.shape)
    # rescale the inputs and outputs
    import common
    common.debug('input data')
    common.debug(inputs[:5], outputs[:5])
    # self.inpmax = inputs.max(axis=0)
    # self.outmax = outputs.max()
    # inp = inputs / self.inpmax
    # out = outputs / self.outmax
    self._seed()
    common.debug('training data')
    common.debug(inputs[:5], outputs[:5])
    for iter in xrange(maxiter):
      for i in xrange(len(outputs)):
        self.input.train(inputs[i], outputs[i], rate, shout=False)
        # raise RuntimeError
    return self.get()
  
  def get(self):
    import common
    # common.debug(self.inpmax, self.outmax)
    for i in range(1, len(self._layers)):
      common.debug(i, self._layers[i].weights)
    # return lambda input: self.output(input / self.inpmax) * self.outmax
    return self.output
      
  
if __name__ == '__main__':
  net = NeuralNetwork([InputLayer(2), TanhLayer(2), TanhLayer(1)])
  print(net)
  inputs = numpy.random.random_sample((100,2)) * 5
  outputs = 3 * inputs.sum(axis=1) + numpy.random.random_sample(100) * 0.5
  xs = inputs[:,0]
  ys = inputs[:,1]
  import matplotlib
  import matplotlib.pyplot as pl
  from mpl_toolkits.mplot3d import Axes3D
  pl.figure().add_subplot(111, projection='3d')
  pl.ion()
  pl.show()
  pl.plot(xs, ys, outputs, 'b.')
  fx = net.get()
  pl.plot(xs, ys, [float(fx(inputs[i])) for i in range(len(inputs))], 'r.')
  print(inputs, outputs)
  for epoch in range(20):
    x = raw_input()
    pl.cla()
    net.train(inputs, outputs, maxiter=1)
    # print(net)
    fx = net.get()
    # for i in range(10):
      # print(inputs[i], fx(inputs[i]), outputs[i], (inputs[i] ** 2).sum())
    # Plot result
    pl.plot(xs, ys, outputs, 'b.')
    pl.plot(xs, ys, [float(fx(inputs[i])) for i in range(len(inputs))], 'r.')
