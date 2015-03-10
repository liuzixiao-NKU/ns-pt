import numpy as np
from dpgmm import *
import multiprocessing as mp

def solve_dpgmm(args):
  nc = args[0]
  dpgmm = args[1]
  for _ in xrange(nc): dpgmm.incStickCap()
  it = dpgmm.solve(iterCap=1024)
  return (nc+1,dpgmm.nllData(),dpgmm)

def sample_dpgmm(args):
  x = args[1]
  w,p = args[0]
  if len(x)==3:
    return np.log(w)+np.array([[[p.logProb([a,b,c]) for a in x[0]] for b in x[1]] for c in x[2]])
  if len(x)==2:
    return np.log(w)+np.array([[p.logProb([a,b]) for a in x[0]] for b in x[1]])
  else:
    density = np.zeros(len(x))
    for i in xrange(len(x)):
      density[i] = p.logProb([x[i]])
    return density+np.log(w)

def ndm(*args):
  return [x[(None,)*i+(slice(None),)+(None,)*(len(args)-i-1)] for i, x in enumerate(args)]

if __name__=='__main__':
  max_comp = 16
  x = np.linspace(-5.0,5.0,128)
  y = np.linspace(-5.0,5.0,128)
  data = np.zeros((1000,2))
  for i in xrange(1000):
    if np.random.uniform(0.0,1.0)< 0.1: data[i,:] = np.random.normal(0.0,0.2,size=2)
    else: data[i,:] = np.random.normal(-1.0,0.1,size=2)
  model = DPGMM(2)
  for l,point in enumerate(data):
    model.add([point])
    model.setPrior()
    model.setThreshold(1e-3)
    model.setConcGamma(1.0,1.0)
  pool = mp.Pool(2*mp.cpu_count())
  jobs = [(i,model) for i in xrange(max_comp)]
  results = pool.map(solve_dpgmm,jobs)
  scores = np.zeros(max_comp)
  for i,r in enumerate(results):
    scores[i] = r[1]
  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm
  plt.figure(1)
  plt.plot(scores)

  print scores.argmax()
  for i,r in enumerate(results):
    if i==scores.argmax():
      model = r[-1]
      break
  density = model.intMixture()
  logprob = np.zeros((len(x),len(y)))
  logprob[:] = -np.inf
  jobs = [((density[0][ind],prob),(x,y)) for ind,prob in enumerate(density[1])]
  results = pool.map(sample_dpgmm,jobs)
  logprob = reduce(np.logaddexp,results)
  plt.figure(2)
  plt.hist2d(data[:,0],data[:,1],bins=[x,y],norm=LogNorm())
  plt.colorbar()
  X,Y = np.meshgrid(x,y)
  plt.contour(X,Y,logprob,10,linewidth=2.0,color='k')
  plt.show()