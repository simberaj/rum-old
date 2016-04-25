# A simple example of PCA
import numpy

class PCA:
  def __init__(self, reduceDim=0.0, varExp=1.0, normalize=True):
    self.reduceDim = reduceDim
    self.varExp = varExp
    self.normalize = normalize
    
  def feed(self, data):
    # subtract the mean, center the data at origin
    self.mean = data.mean(axis=0)
    centered = data - self.mean
    # covariance matrix
    C = numpy.cov(centered.transpose())
    # Compute eigenvalues and sort into descending order
    evals,evecs = numpy.linalg.eig(C)
    indices = numpy.argsort(evals)
    indices = indices[::-1]
    self.eigenvectors = evecs[:,indices]
    self.eigenvalues = evals[indices]
    # fractions of variance explained
    self.varianceExplained = self.eigenvalues.cumsum() / self.eigenvalues.sum()
    # print(evals, varfrac, (varfrac < 0.95).sum())

    self.dimensions = self.eigenvectors.shape[1]
    if self.reduceDim > 0:
      self.dimensions = min(int((1 - reduceDim) * self.eigenvectors.shape[1]), self.dimensions)
    if self.varExp < 1.0:
      self.dimensions = min((self.varianceExplained < self.varExp).sum(), self.dimensions)
    self.eigenvectors = self.eigenvectors[:,:self.dimensions]
    
    if self.normalize:
      for i in range(self.eigenvectors.shape[1]):
        self.eigenvectors[:,i] / numpy.linalg.norm(self.eigenvectors[:,i]) * numpy.sqrt(self.eigenvalues[i])

    # Produce the new data matrix
    # x = numpy.dot(numpy.transpose(evecs),numpy.transpose(data))
    # print(x)
    transformed = centered.dot(self.eigenvectors)
    # self.dividers = abs(numpy.array((
        # transformed.max(axis=0),
        # transformed.min(axis=0)))).max(axis=0)
    # print(x)
    # Compute the original data again
    # self.reconstructed = numpy.transpose(numpy.dot(self.eigenvectors,x.transpose()))+m
    reconstructed = self.eigenvectors.dot(transformed.transpose()).transpose() + self.mean
    return {'centered' : centered, 'transformed' : transformed, 'reconstructed' : reconstructed}
    
  def transform(self, input):
    return (input - self.mean).dot(self.eigenvectors)# / self.dividers

    
if __name__ == '__main__':
  import pylab as pl
  x = numpy.random.normal(5,.2,1000)
  y = numpy.random.normal(3,1,1000)
  a = x*numpy.cos(numpy.pi/4) + y*numpy.sin(numpy.pi/4)
  b = -x*numpy.sin(numpy.pi/4) + y*numpy.cos(numpy.pi/4)

  pl.plot(a,b,'.')
  ex = numpy.array((5.5,-2))
  pl.plot(ex[0],ex[1],'r.')
  pl.xlabel('x')
  pl.ylabel('y')
  pl.title('Original dataset')
  data = numpy.zeros((1000,2))
  data[:,0] = a
  data[:,1] = b

  pca = PCA(data)
  # print x
  # print y
  # print evals
  # print evecs
  x = pca.transNorm
  y = pca.reconstructed
  pl.figure()
  pl.plot(x[:,0],x[:,1],'.')
  pl.xlabel('x')
  pl.ylabel('y')
  pl.title('Reconstructed data after PCA')
  trex = (ex - pca.mean).dot(pca.eigenvectors) / pca.dividers
  print(trex)
  pl.plot(trex[0],trex[1],'r.')
  pl.show()
  # print(pca.transformedNorm)