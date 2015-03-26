#! /usr/bin/env python
# coding: utf-8

import sys
import os
import optparse as op
from scipy import integrate
from random import shuffle
import optparse as op
import cPickle as pickle
import numpy as np
from ComputePKparameters import *
from logLikelihoodsPT import *
from GaussianMixture import *
from numpy.linalg import eig
import libstempo as T
from Parameters_v2 import *
from collections import deque
from scipy.stats import multivariate_normal
import multiprocessing as mp
from Queue import Queue

def autocorr(x):
    """
    Compute autocorrelation by convolution
    """
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result 

def autocorrelation(x):
	"""
	Compute autocorrelation using FFT
	"""
	x = np.asarray(x)
	N = len(x)
	x = x-x.mean()
	s = np.fft.fft(x, N*2-1)
	result = np.real(np.fft.ifft(s * np.conjugate(s), N*2-1))
	result = result[:N]
	result /= result[0]
	return result
	
class NestedSampler(object):
    """
    Nested Sampling class for pulsar timing data analysis
    Requirements are tempo2, the python wrapper libstempo
    For red noise analysis the gaussian process package george is necessary
    
    Initialisation arguments:
    
    parfiles: list of parameter files in tempo2 format 
    
    timfiles: list of timing files in tempo2 format
    
    Nlive: 
        number of live points to be used for the integration
        default: 1024
    
    maxmcmc: 
        maximum number of mcmc steps to be used in the sampler
        default: 4096
    
    model: 
        specify if the anlaysis should be done assuming the tempo2 model, the DDGR model of the DDCG model
        default: None
    
    output: 
        folder where the output will be stored
    
    verbose: 
        display information on screen
        default: True
        
    seed:
        seed for the initialisation of the pseudorandom chain
        
    nthreads: (UNUSED)
    
    prior:
        produce N samples from the prior
        
    noise:
        model to use for the noise
        default: None
    """
    def __init__(self,parfiles,timfiles,Nlive=1024,maxmcmc=4096,model=None,noise=None,output=None,verbose=True,seed=1,nthreads=None,prior=None):
        """
        Initialise all necessary arguments and variables for the algorithm
        """
        if nthreads == None:
          self.nthreads = mp.cpu_count()*2
        else:
          self.nthreads = np.minimum(nthreads,mp.cpu_count()*2)
        self.prior_sampling = prior
        self.model = model
        self.noise = noise
        self.setup_random_seed(seed)
        self.verbose = verbose
        self.task_queue =  deque(maxlen=self.nthreads)
        self.result_queue = mp.JoinableQueue()
        self.active_index = None
        self.accepted = 0
        self.rejected = 1
        self.dimension = None
        self.Nlive = Nlive
        self.Nmcmc = maxmcmc
        self.cache = deque(maxlen=2*maxmcmc)
        self.maxmcmc = maxmcmc
        self.output,self.evidence_out = self.setup_output(output)
        self.pulsars = [tempopulsar(parfile = parfiles[i], timfile = timfiles[i]) for i in xrange(len(parfiles))]
        for p in self.pulsars: p.fit()
        self.samples = []
        self.params = [None] * Nlive
        self.logZ = np.finfo(np.float128).min
        self.tolerance = 0.1
        self.condition = None
        self.logLmin = None
        self.information = 0.0
        self.worst = 0
        self.logLmax = None
        self.iteration = 0
        self.jumps = 0
        self.thinning = 1
        self.logLinj = computeLogLinj(self.pulsars)
        for n in xrange(self.Nlive):
            while True:
                if self.verbose: sys.stderr.write("sprinkling --> %.3f %% complete\r"%(100.0*float(n+1)/float(Nlive)))
                self.params[n] = Parameter(self.pulsars,model=self.model,noise=self.noise)
                self.params[n].set_bounds()
                self.params[n].initialise()
                self.params[n].logPrior()
                if not(isinf(self.params[n].logP)): break
        sys.stderr.write("\n")
        self.dimension = 0
        for n in self.params[0].par_names:
          if self.params[0].vary[n]==1:
            self.dimension+=self.params[0].vary[n]
            if self.verbose: sys.stderr.write("%s in [%.15f,%.15f]\n"%(n,self.params[0].bounds[n][0],self.params[0].bounds[n][1]))
        self.scale = 2.0
        self.active_live = Parameter(self.pulsars,model=self.model,noise=self.noise)
        self.active_live.set_bounds()
        self.active_live.initialise()
        self.active_live.logPrior()
        self.acceptance_probability = None
        cov_array = np.zeros((self.dimension,self.Nlive))
        for j,p in enumerate(self.params):
          i = 0
          for n in self.params[0].par_names:
            if p.vary[n]==1:
              cov_array[i,j] = p._internalvalues[n]
              i+=1
        self.covariance = np.cov(cov_array)
#        a = fitDPGMM([self.dimension,self.covariance],np.transpose(cov_array))
#        if a!=None:
#          self.gmm = GMM(self.dimension,a)
#        else:
#          self.gmm = GMM((self.dimension,self.covariance))
        ev,evec = np.linalg.eigh(self.covariance)
        self.eigen_values,self.eigen_vectors = ev.real,evec.real
        self.mvn = multivariate_normal(np.zeros(self.dimension),self.covariance)
        if self.verbose: sys.stderr.write("Dimension --> %d\n"%self.dimension)
        header = open(os.path.join(output,'header.txt'),'w')
        for n in self.active_live.par_names:
          header.write(n+'\t')
        header.write('logL\n')
        header.close()
    
    def setup_output(self,output):
        """
        Set up the output folder
        """
        os.system("mkdir -p %s"%output)
        outputfile = "chain_"+self.model+"_"+str(self.Nlive)+"_"+str(self.seed)+".txt"
        return open(os.path.join(output,outputfile),"w"),open(os.path.join(output,outputfile+"_evidence.txt"), "wb" )

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def setup_proposals(self):
        """
        initialise the proposals cycle
        """
        self.proposal_function = [None]*100
        for i in xrange(100):
          jump_select = np.random.uniform(0.,1.)
          if jump_select<0.2: self.proposal_function[i]=self._mvn
          elif jump_select<0.5: self.proposal_function[i]=self._eigen
          elif jump_select<0.8: self.proposal_function[i]=self._stretch
          else: self.proposal_function[i]=self._walk

    def _select_live(self):
        """
        select a live point
        """
        while True:
            j = np.random.randint(self.Nlive)
            if j!= self.worst:
                return j
    def _evolve_live(self):
        """
        evolve a live point
        """
        self.proposal_function[self.jumps%100]()
    def _mvn(self):
        """
        multivariate normal proposal function
        will be replaced by a gaussian mixture model proposal once it is working
        """
        variates = self.mvn.rvs()#self.gmm.sample()#
        for n in self.active_live.par_names:
          k = 0
          if self.active_live.vary[n]==1:
            self.active_live._internalvalues[n]+=variates[k]
            k+=1
        self.acceptance_probability = np.log(np.random.uniform(0.,1.))
    def _eigen(self):
        """
        normal jump along the live points covariance matrix eigen directions
        """
        i = np.random.randint(self.dimension)
        k = 0
        for n in self.active_live.par_names:
          if self.active_live.vary[n]==1:
            jumpsize = np.sqrt(np.abs(self.eigen_values[i]))*np.random.normal(0,1)
            self.active_live._internalvalues[n]+=jumpsize*self.eigen_vectors[k,i]
            k+=1
        self.acceptance_probability = np.log(np.random.uniform(0.,1.))
    def _update_mvn(self):
        """
        update the live points covariance matrix
        """
        cov_array = np.zeros((self.dimension,self.Nlive))
        for j,p in enumerate(self.params):
            i = 0
            for n in self.params[0].par_names:
                if p.vary[n]==1:
                    cov_array[i,j] = p._internalvalues[n]
                    i+=1
        self.covariance = np.cov(cov_array)
        #        a = fitDPGMM([self.dimension,self.covariance],np.transpose(cov_array))
        #        if a!=None:
        #          self.gmm = GMM(self.dimension,a)
        #        else:
        #          self.gmm = GMM([self.dimension,self.covariance])
        ev,evec = np.linalg.eigh(self.covariance)
        self.eigen_values,self.eigen_vectors = ev.real,evec.real
        try:
            self.mvn = multivariate_normal(np.zeros(self.dimension),self.covariance.T)
        except:
            pass

    def _walk(self):
        """
        ensemble sampler walk move
        """
        Nsubset = 3
        while True:
          indeces = random.sample(range(self.Nlive),Nsubset)
          if self.active_index not in indeces: break
        subset = [self.params[i] for i in indeces]
        cm = np.zeros(1,dtype={'names':self.active_live.par_names,'formats':self.active_live.par_types})
        for n in self.active_live.par_names:
          cm[n] = np.sum([p._internalvalues[n] for p in subset])/Nsubset
        w = np.zeros(1,dtype={'names':self.active_live.par_names,'formats':self.active_live.par_types})
        for n in self.active_live.par_names:
          w[n] = np.sum([np.random.normal(0,1)*(p._internalvalues[n]-cm[n]) for p in subset])
        for n in self.active_live.par_names:
          if self.active_live.vary[n]==1: self.active_live._internalvalues[n]+=w[n]
        self.acceptance_probability = np.log(np.random.uniform(0.,1.))
    def _stretch(self):
        """
        ensemble sampler stretch move
        """
        a=0
        while True:
            a = np.random.randint(self.Nlive)
            if a!=self.active_index: break
        wa = np.copy(self.params[a]._internalvalues[:])
        u = np.random.uniform(0.0,1.0)
        x = 2.0*u*np.log(self.scale)-np.log(self.scale)
        Z = np.exp(x)
        for n in self.active_live.par_names:
          if self.active_live.vary[n]==1: self.active_live._internalvalues[n] = wa[n]+Z*(self.active_live._internalvalues[n]-wa[n])
        if (Z<1.0/self.scale)or(Z>self.scale): self.acceptance_probability = -np.inf
        else: self.acceptance_probability = np.log(np.random.uniform(0.,1.))-(self.dimension)*np.log(Z)
    def sample_prior(self):
        """
        generate samples from the prior
        """
        first = 1
        for i in xrange(self.Nlive):
          if self.verbose: sys.stderr.write("sampling the prior --> %.3f %% complete\r"%(100.0*float(i+1)/float(self.Nlive)))
          self.copy_params(self.params[i],self.active_live)
          while self.jumps<self.Nmcmc or self.accepted==0 or self.params[i].logL == -np.inf:
            logP0 = self.active_live.logPrior()
            self._evolve_live()
            logP = self.active_live.logPrior()
            if logP-logP0 > self.acceptance_probability:
              logLnew = self.active_live.logLikelihood()
              if logLnew > -np.inf:
                self.copy_params(self.active_live,self.params[i])
                self.accepted+=1
              else:
                self.copy_params(self.params[i],self.active_live)
                self.rejected+=1
            else:
              self.copy_params(self.params[i],self.active_live)
              self.rejected+=1
            self.cache.append(self.params[i]._internalvalues[:])
            self.jumps+=1
          if first and len(self.cache)==2*self.maxmcmc:
            first = 0
            self.autocorrelation()
            self._update_mvn()
          self.jumps = 0
        if self.verbose: sys.stderr.write("\n")

    def evolve(self):
        """
        mcmc loop to generate the new live point taking nmcmc steps
        """
        self.accepted = 0
        self.rejected = 1
        self.jumps = 0
        self.copy_params(self.params[self.worst],self.active_live)
        while (self.jumps < self.Nmcmc or self.accepted==0):
          logP0 = self.active_live.logPrior()
          self._evolve_live()
          logP = self.active_live.logPrior()
          if logP-logP0 > self.acceptance_probability:
            logLnew = self.active_live.logLikelihood()
            if logLnew > self.logLmin:
              self.copy_params(self.active_live,self.params[self.worst])
              self.accepted+=1
            else:
              self.copy_params(self.params[self.worst],self.active_live)
              self.rejected+=1
          else:
            self.copy_params(self.params[self.worst],self.active_live)
            self.rejected+=1
          self.cache.append(self.active_live._internalvalues[:])
          self.jumps+=1
          if self.nthreads>1: self.result_queue.put((self.params[self.worst].values,self.params[self.worst]._internalvalues,self.params[self.worst].logP,self.params[self.worst].logL,self.jumps,float(self.accepted)/float(self.rejected+self.accepted)))

    def copy_params(self,param_in,param_out):
        """
        helper function to copy live points
        """
        param_out._internalvalues = np.copy(param_in._internalvalues)
        param_out.values = np.copy(param_in.values)
        param_out.logL = np.copy(param_in.logL)
        param_out.logP = np.copy(param_in.logP)

    def autocorrelation(self):
        """
        estimates the autocorrelation length of the mcmc chain from the cached samples
        """
        try:
            ACLs = []
            cov_array = np.squeeze([[np.float64(p[n]) for n in self.active_live.par_names] for p in self.cache])
            N = len(self.cache)
            for i,n in enumerate(self.active_live.par_names):
              if self.params[0].vary[n]==1:
                ACF = autocorrelation(cov_array[:,i])
                ACL = np.min(np.where((ACF > -2./np.sqrt(N)) & (ACF < 2./np.sqrt(N)))[0])#sum(ACF**2)
                if not(np.isnan(ACL)):
                  ACLs.append(ACL)
                  if self.verbose: sys.stderr.write("autocorrelation length %s = %.1f mean = %g standard deviation = %g\n"%(n,ACLs[-1],np.mean(cov_array[:,i]),np.std(cov_array[:,i])))
            self.Nmcmc =int(self.thinning*(np.max(ACLs)))
            if self.Nmcmc < 16: self.Nmcmc = 16
            if self.Nmcmc > self.maxmcmc:
                if self.verbose: sys.stderr.write("Warning ACL --> %d!\n"%self.Nmcmc)
                self.Nmcmc = self.maxmcmc
        except:
            sys.stderr.write("ACL computation failed!\n")
            self.Nmcmc = self.maxmcmc
        if self.nthreads>1: self.Nmcmc = self.maxmcmc

    def wrapper_evolve(self,index):
        """
        wrapper function for the evolution of the active live point
        UNUSED
        """
        self.copy_params(self.params[index],self.params[self.worst])
        self.evolve()

    def nested_sampling(self):
        """
        main nested sampling loop
        """
        self.setup_proposals()
        logwidth = np.log(1.0 - np.exp(-1.0 / float(self.Nlive)))
        self.sample_prior()
        self._update_mvn()
        self.autocorrelation()
        self.condition = np.inf
        running_jobs = 0
        # if requested drop the stack of live points after sampling the prior and return
        if self.prior_sampling:
          for i in xrange(self.Nlive):
            line = ""
            for n in self.params[i].par_names:
              line+='%.30e\t'%self.params[i].values[n]
            line+='%30e\n'%self.params[i].logL
            self.output.write(line)
          self.output.close()
          return

        while self.condition > self.tolerance:
            logL_array = np.array([p.logL for p in self.params])
            self.worst = logL_array.argmin()
            self.logLmin = np.min(logL_array)
            self.logLmax = np.max(logL_array)
            logWt = self.logLmin+logwidth;
            logZnew = np.logaddexp(self.logZ, logWt)
            self.information = np.exp(logWt - logZnew) * self.params[self.worst].logL + np.exp(self.logZ - logZnew) * (self.information + self.logZ) - logZnew
            self.logZ = logZnew
            self.condition = np.logaddexp(self.logZ,self.logLmax-self.iteration/(float(self.Nlive))-self.logZ)
            line = ""
            for n in self.params[self.worst].par_names:
              line+='%.30e\t'%self.params[self.worst].values[n]
            line+='%30e\n'%self.params[self.worst].logL
            self.output.write(line)
            self.jumps = 0

            if self.nthreads==1:
              while True:
                self.active_index =self._select_live()
                self.copy_params(self.params[self.active_index],self.params[self.worst])
                self.evolve()
                self.rejected+=1
                acceptance = float(self.accepted)/float(self.accepted+self.rejected)
                if self.params[self.worst].logL>self.logLmin: break
            else:
              while True:
                if not(self.result_queue.empty()):
                  vals,invals,lp,ll,self.jumps,acceptance = self.result_queue.get()
                  self.params[self.worst].values = np.copy(vals)
                  self.params[self.worst]._internalvalues = np.copy(invals)
                  self.params[self.worst].logP = np.copy(lp)
                  self.params[self.worst].logL = np.copy(ll)
                else:
                  running_jobs = 0
                  for p in self.task_queue:
                    running_jobs += p.is_alive()
                  for n in xrange(self.nthreads-running_jobs):
                    self.active_index = self._select_live()
                    self.copy_params(self.params[self.active_index],self.params[self.worst])
                    p = mp.Process(target=self.wrapper_evolve,args=(self.active_index,))
                    p.daemon = False
                    p.start()
                    self.task_queue.append(p)
  #                self.task_queue.task_done()
  #
  ##print self.task_queue.task_done()
  #                print self.task_queue.get()
  #                print self.task_queue.task_done()
                  running_jobs = 0
                  for p in self.task_queue:
                    running_jobs += p.is_alive()
                  #exit()
                  vals,invals,lp,ll,self.jumps,acceptance = self.result_queue.get()
                  self.params[self.worst].values = np.copy(vals)
                  self.params[self.worst]._internalvalues = np.copy(invals)
                  self.params[self.worst].logP = np.copy(lp)
                  self.params[self.worst].logL = np.copy(ll)
                  #self.copy_params(self.result_queue.get(),self.params[self.worst])
                  self.rejected+=1
                if self.params[self.worst].logL > self.logLmin: break

            if self.verbose: sys.stderr.write("%d: n:%4d acc:%.3f H: %.2f logL %.5f --> %.5f dZ: %.3f logZ: %.3f logLmax: %.5f logLinj: %.5f cache: %4d processes: %d\n"%(self.iteration,self.jumps,acceptance,self.information,self.logLmin,self.params[self.worst].logL,self.condition,self.logZ,self.logLmax,self.logLinj,len(self.cache),running_jobs))
            running_jobs = 0
            logwidth-=1.0/float(self.Nlive)
            self.iteration+=1
            if self.iteration%(self.Nlive/4)==0:
              self.autocorrelation()
              self._update_mvn()
            if self.condition < self.tolerance: break
        sys.stderr.write("\n")
        # final adjustments
        i = 0
        logL_array = [p.logL for p in self.params]
        logL_array = np.array(logL_array)
        idx = logL_array.argsort()
        logL_array = logL_array[idx]
        for i in idx:
          line = ""
          for n in self.params[i].par_names:
            line+='%.30e\t'%self.params[i].values[n]
          line+='%30e\n'%self.params[i].logL
          self.output.write(line)
          i+=1
        self.output.close()
        self.evidence_out.write('%.5f %.5f %.5f\n'%(self.logZ,self.logLmax,self.logLinj))
        self.evidence_out.close()
        return

def parse_to_list(option, opt, value, parser):
    """
    parse a comma separated string into a list
    """
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
  parser = op.OptionParser()
  parser.add_option("-N", type="int", dest="Nlive", help="Number of Live points",default=1000)
  parser.add_option("-o", "--output", type="string", dest="output", help="Output folder", default="Free")
  parser.add_option("-t", "--type", type="string", dest="model", help="Class of model to assume (Free, GR, CG)", default=None)
  parser.add_option("-s", type="int", dest="seed", help="seed for the chain", default=0)
  parser.add_option("--verbose", action="store_true", dest="verbose", help="display progress information", default=False)
  parser.add_option("--maxmcmc", type="int", dest="maxmcmc", help="maximum number of mcmc steps", default=5000)
  parser.add_option("--nthreads", type="int", dest="nthreads", help="number of parallel threads to spawn", default=None)
  parser.add_option("--parameters", type="string", dest="parfiles", help="pulsar parameter files", default=None, action='callback',
                    callback=parse_to_list)
  parser.add_option("--times", type="string", dest="timfiles", help="pulsar time files, they must be ordered as the parameter files", default=None, action='callback',
                                      callback=parse_to_list)
  parser.add_option( "--sample-prior", action="store_true", dest="prior", help="draw samples from the prior", default=False)
  parser.add_option( "--noise", dest="noise", type="string", help="noise model to assume", default=None)
  (options, args) = parser.parse_args()
  if len(options.parfiles)!= len(options.timfiles):
    sys.stderr.write("Fatal error! The number of par files is different from the number of times!\n")
    exit(-1)

  NS = NestedSampler(options.parfiles,options.timfiles,model=options.model,seed=options.seed,noise=options.noise,Nlive=options.Nlive,maxmcmc=options.maxmcmc,output=options.output,verbose=options.verbose,nthreads=options.nthreads,prior=options.prior)
  NS.nested_sampling()
