import numpy as np
from collections import defaultdict
from ComputePKparameters import *
import libstempo as T
from libstempo import tempopulsar
import sys

par_names = ['RAJ','DECJ','F0_0','F1_0','F0_1','F1_1','DM','PMRA','PMDEC','PX','SINI','PB','T0','A1_0','A1_1','OM_0','OM_1','ECC','PBDOT','OMDOT','M2_0','M2_1','GAMMA_0','GAMMA_1','GOB','EPS','XI','KAPPA']
par_formats = [np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128]
vary_formats = [np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int]

width = 100.

class Parameter(object):
  """
    Class holding the necessary functions and informations for the nested sampling algorithm
  """
  def __init__(self,model=None):
    self.bounds = None
    self.logP = -np.inf
    self.logL = -np.inf
    self.vary = np.zeros(1,dtype={'names':par_names,'formats':vary_formats})
    self.model = model
    self.values = np.zeros(1,dtype={'names':par_names,'formats':par_formats})
    self._internalvalues = np.zeros(1,dtype={'names':par_names,'formats':par_formats})
  def set_vary(self):
    """
      Sets the parameters that will have to vary given a certain model and a certain pulsar pair
    """
    self.vary['OM_1'] = -1
    if self.model==None or self.model=="Free":
      for n in ['RAJ','DECJ','F0_0','F1_0','F0_1','F1_1','DM','PMRA','PMDEC','PX','SINI','PB','T0','A1_0','A1_1','OM_0','ECC','PBDOT','OMDOT','M2_0','M2_1','GAMMA_0','GAMMA_1']:
        self.vary[n] = 1
    elif self.model=='GR':
      for n in ['RAJ','DECJ','F0_0','F1_0','F0_1','F1_1','DM','PMRA','PMDEC','PX','PB','T0','A1_0','A1_1','OM_0','ECC','M2_0','M2_1']:
        self.vary[n] = 1
      for n in ['GAMMA_0','GAMMA_1','PBDOT','OMDOT','SINI']:
        self.vary[n] = -1
    elif self.model=='CG':
      for n in ['RAJ','DECJ','F0_0','F1_0','F0_1','F1_1','DM','PMRA','PMDEC','PX','PB','T0','A1_0','A1_1','OM_0','PBDOT','ECC','M2_0','M2_1','GOB','EPS','XI','KAPPA']:
        self.vary[n] = 1
      for n in ['GAMMA_0','GAMMA_1','OMDOT','SINI']:
        self.vary[n] = -1
    if self.model=='Test':
      for n in ['RAJ','DECJ']:
        self.vary[n] = 1
  def set_bounds(self,pulsars):
    """
      Sets the bounds on the varying parameters given a tempo2 fit and a timing model
    """
    self.bounds = {}
    for n in pulsars[0].pars:
      if n in ['RAJ','DECJ','PMRA','PMDEC','SINI','PB','ECC','PBDOT','OMDOT']:
        a = np.minimum(pulsars[0][n].val-width*pulsars[0][n].err,pulsars[1][n].val-width*pulsars[1][n].err)
        b = np.maximum(pulsars[0][n].val+width*pulsars[0][n].err,pulsars[1][n].val+width*pulsars[1][n].err)
        self.vary[n] = np.all([p[n].fit for p in pulsars])
      if n=='T0':
        a = np.minimum(pulsars[0][n].val-pulsars[0][n].err,pulsars[1][n].val-pulsars[1][n].err)
        b = np.maximum(pulsars[0][n].val+pulsars[0][n].err,pulsars[1][n].val+pulsars[1][n].err)
        self.vary[n] = np.all([p[n].fit for p in pulsars])
      if n=='SINI':
        b = 1.0
        self.vary[n] = np.all([p[n].fit for p in pulsars])
      if n=='M2':
        amax = 1.
        bmin = 3.
        for j,p in enumerate(pulsars):
          a = np.maximum(amax,p[n].val-width*p[n].err)
          b = np.minimum(bmin,p['M2'].val+width*p[n].err)
          self.bounds[n+'_%d'%j] = [a,b]
          self.vary[n+'_%d'%j] = p[n].fit
      if 'GAMMA' in n:
        a = 0.0
      if n=='DM':
        a = 1.0
        b = 100.0
        self.vary[n] = np.all([p[n].fit for p in pulsars])
      if n=='PX':
        a = 1.0
        b = 10.0
        self.vary[n] = np.all([p[n].fit for p in pulsars])
      if (n == 'F0' or n == 'F1' or n == 'A1' or n == 'OM' or n =='GAMMA'):
        for j,p in enumerate(pulsars):
          a = p[n].val-width*p[n].err
          b = p[n].val+width*p[n].err
          self.bounds[n+'_%d'%j] = [a,b]
#  #        sys.stderr.write("%.30f %.30f %.30f %.30f\n"%(a,b,pulsars[0][n].val,(b-pulsars[0][n].val)/pulsars[0][n].err))
#          a = pulsars[1][n].val-width*pulsars[1][n].err
#          b = pulsars[1][n].val+width*pulsars[1][n].err
#  #        sys.stderr.write("%.30f %.30f %.30f %.30f\n"%(a,b,pulsars[1][n].val,(b-pulsars[1][n].val)/pulsars[1][n].err))
#          self.bounds[n+'_1'] = [a,b]
#          self.vary[n+'_0'] = pulsars[0][n].fit
          self.vary[n+'_%d'%j] = p[n].fit
#        exit()
        self.vary['OM_1'] = -1
      else:
        self.bounds[n] = [a,b]
    if self.model=='GR':
      for n in ['GAMMA_0','GAMMA_1','PBDOT','OMDOT','SINI']:
        self.vary[n] = -1
    elif self.model=='CG':
      self.bounds['GOB'] = [0.99,1.01]
      self.bounds['EPS'] = [2.99,3.01]
      self.bounds['XI'] = [0.99,1.01]
      self.bounds['KAPPA'] = [-0.01,0.01]
      self.vary['KAPPA'] = 1
      self.vary['GOB'] = 1
      self.vary['EPS'] = 1
      self.vary['XI'] = 1
      for n in ['GAMMA_0','GAMMA_1','OMDOT','SINI']:
        self.vary[n] = -1
    for n in par_names:
      if self.vary[n] == 0:
        if n in pulsars[0].allpars:
          self.values[n] = pulsars[0].prefit[n].val
        elif '_0' in n:
          self.values[n] = pulsars[0].prefit[str(n.split("_")[0])].val
        elif '_1' in n:
          self.values[n] = pulsars[1].prefit[str(n.split("_")[0])].val
  def inbounds(self):
    """
      Checks whether the values of the parameters are in bound
    """
    for n in self.values.dtype.names:
      if (self.vary[n]==1):
        if (self.values[n] < self.bounds[n][0] or self.values[n] > self.bounds[n][1]):
          return False
    self.constraint()
    for n in self.values.dtype.names:
      if (self.vary[n]==-1):
        if (self.values[n] < self.bounds[n][0] or self.values[n] > self.bounds[n][1]):
          return False
    return True
  
  def map(self):
    """
      Maps the bounds of the parameters onto [0,1]
    """
    for name in par_names:
      if (self.vary[name]==1):
        self.values[name] = self.bounds[name][0]+self._internalvalues[name]*(self.bounds[name][1]-self.bounds[name][0])

  def inverse_map(self):
    """
      Maps [0,1] to the bounds of the parameters
    """
    for name in par_names:
      if (self.vary[name]==1):
        self._internalvalues[name] = (self.values[name]-self.bounds[name][0])/(self.bounds[name][1]-self.bounds[name][0])

  def constraint(self):
    """
      Imposes the relevant constraints for the model under consideration
    """
    self.values['OM_1'] = self.values['OM_0']+180.
    if self.model is not("Free"):
      m1 = self.values['M2_1']
      m2 = self.values['M2_0']
      pb = self.values['PB']
      a1 = self.values['A1_0']
      a2 = self.values['A1_1']
      ecc = self.values['ECC']
      if self.model=='GR':
        self.values['GAMMA_0'] = gamma(pb,ecc,m1,m2)
        self.values['GAMMA_1'] = gamma(pb,ecc,m2,m1)
        self.values['PBDOT'] = pbdot(pb,ecc,m1,m2)
        self.values['OMDOT'] = omega_dot(pb,ecc,m1,m2)
        self.values['SINI'] = shapiroS(pb,m1,m2,a1)
      #sys.stderr.write('pb: %.30f m1: %.30f m2: %.30f a1: %.30f sini: %.30f\n'%(pb,m1,m2,a1,self.values['SINI']))
      elif self.model=='CG':
        gob = self.values['GOB']
        xi = self.values['XI']
        eps = self.values['EPS']
        kappa = self.values['KAPPA']
        mtot = m1+m2
        beta = beta0(pb,mtot,gob)
        self.values['GAMMA_0'] = gammaAG(pb,m2/mtot,gob,kappa,beta,ecc)
        self.values['GAMMA_1'] = gammaAG(pb,m1/mtot,gob,kappa,beta,ecc)
        self.values['OMDOT'] = omdotAG(pb,eps,xi,beta,ecc)
        self.values['SINI'] = shapiroSAG(pb,a1,m2/mtot,beta)
  def logPrior(self):
    """
      Prior function, flat on every parameter for the time being
    """
    self.map()
    if self.inbounds():
      self.logP = 0.0
    else:
      self.logP = -np.inf
    return self.logP
  def initialise(self):
    for n in par_names:
      if (self.vary[n]==1):
        self._internalvalues[n] = np.random.uniform(0.0,1.0)#self.bounds[n][0],self.bounds[n][1])
    self.map()
    self.constraint()
  def logLikelihood(self,pulsars):
    """
      Likelihood function for white uncorrelated gaussian noise
    """
    # fill the pulsars
    self.logL=0.0
    for i,p in enumerate(pulsars):
      for name in par_names:
        if self.vary[name]!=0:
          if name in ['RAJ','DECJ','PMRA','PMDEC','SINI','PB','T0','ECC','PBDOT','OMDOT','DM']:
            p[name].val = np.copy(self.values[name])
          if name in ['F0','F1','A1','OM','M2','GAMMA']:
            p[name].val = np.copy(self.values[name+'_'+str(i)])
      err = 1.0e-6 * p.toaerrs
      Cdiag = (err)**2
      Cinv = np.diag(1.0/Cdiag)
      #logCdet = np.sum(np.log(Cdiag))
      res = np.array(p.residuals(updatebats=True,formresiduals=True),dtype=np.float128)
      self.logL+= -0.5 * np.dot(res,np.dot(Cinv,res))#- 0.5 * logCdet - 0.5 * len(res) * np.log(2.0*np.pi)
    return self.logL



if __name__ == '__main__':
  import libstempo as T
  import matplotlib.pyplot as plt

  T.data = "/projects/pulsar_timing/nested_sampling/"
  parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par"]
  timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b.simulate"]
  
  psrA = tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
  psrB = tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
  psrA.fit()
  psrB.fit()

  N = 1
  param = [None]*N
  v = []
  for i in xrange(N):
    param[i] = Parameter()
    param[i].set_bounds([psrA,psrB])
    param[i].initialise()
    print param[i].logLikelihood([psrA,psrB])
    print param[i].values
    print param[i]._internalvalues