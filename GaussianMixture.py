from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
import numpy as np
from dpgmm import *
from random import betavariate
import sys

def sampleMixture(mix):
    weight = []
    gauss = []
    stick = 1.0
    for i in xrange(mix.stickCap):
      val = betavariate(mix.v[i,0], mix.v[i,1])
      wi = stick * val
      if wi > 1e-3:
        weight.append(wi)
        gauss.append(mix.n[i].sample())
      stick *= 1.0 - val
    weight = np.array(weight)
    if np.sum(weight)<1.0:
      weight[weight.argmin()] = 1.0 - (np.sum(weight)-weight.min())
    return (np.array(weight),gauss)

def fitDPGMM(dims,data):
  maxStick = 32
  model = DPGMM(dims[0])
  means_array = np.array([np.mean(data[:,i]) for i in xrange(dims[0])])
  std_array = np.array([np.std(data[:,i]) for i in xrange(dims[0])])
  try:
    for i,p in enumerate(data):
      model.add((p-means_array)/std_array)

    model.setPrior()#covar = dims[1], mean = np.zeros(dims[0]))
    model.setThreshold(1e-3)
    model.setConcGamma(1.0,1.0)

    it = model.solve(iterCap=1024)
    best = model
    bestNLL = model.nllData()

    current = model
    lastScore = None

    while True and current.getStickCap() < maxStick:
      current = DPGMM(current)
      current.incStickCap()
      it = current.solve(iterCap=1024)
      score = current.nllData()
      if score>bestNLL:
        best = current
      bestNLL = score
      if score<lastScore: break
      lastScore = score
    model = best
    density = None
    while density==None:
      try:
        density = sampleMixture(model)
        return density
      except:
        pass
  except:
    sys.stderr.write("dpgmm failed!\n")
    return None

def getParams(mixture):
  means, covars = [], []
  for i,p in enumerate(mixture[1]):
    means.append(p.getMean())
    covars.append(p.getCovariance())
  return means, covars, mixture[0]

class GMM(object):
  def __init__(self,dims,mixture=None):
    if mixture !=None:
      self.n = len(mixture[1])
      self.components=[multivariate_normal(mean=np.zeros(dims[0]), cov=p.getCovariance()) for i,p in enumerate(mixture[1])]
      self.w = mixture[0]
    else:
      self.n = 1
      self.w = [1.0]
      self.components=[multivariate_normal(mean=np.zeros(dims[0]), cov=dims[1])]
    self.density=None
  def logProb(self,x):
    return logsumexp([comp.logprob(x) for comp in self.components],b=self.w)
  def sample(self):
    i = np.random.choice(np.arange(self.n),p=self.w)
    return self.components[i].rvs()

#dims = 23
#data = np.zeros((256,dims))
#for i in xrange(dims):
#  for j in xrange(256):
#    data[j,i] = np.random.uniform(-1.0,1.0)
##data = np.loadtxt("live_stack.txt")
##
##from pylab import *
##matshow(cov(data.T))
##colorbar()
##show()
#
#dims = data.shape[1]
#a = fitDPGMM([dims,np.cov(data.T)],data)
#gmm = GMM([dims,np.cov(data.T)],a)
#print gmm.sample()

