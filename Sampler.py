import sys
import os
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
from multiprocessing import Process, Lock, Queue
import proposals
import signal
import time

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

class Sampler(object):
    def __init__(self,pulsars,maxmcmc,model="Free",noise=None,verbose=True):
        self.model = model
        self.noise = noise
        self.pulsars = pulsars
        self.param = Parameter(self.pulsars,model=self.model,noise=self.noise)
        self.param.set_bounds()
        self.param.initialise()
        self.param.logPrior()
        self.cache = deque(maxlen=5*maxmcmc)
        self.inParam = Parameter(self.pulsars,model=self.model,noise=self.noise)
        self.inParam.set_bounds()
        self.inParam.initialise()
        self.inParam.logPrior()
        self.maxmcmc=maxmcmc
        self.Nmcmc=maxmcmc
        self.proposals = proposals.setup_proposals_cycle()
        self.poolsize = 100
        self.evolution_points = [None]*self.poolsize
        self.verbose=verbose
        for n in xrange(self.poolsize):
            while True:
                if self.verbose: sys.stderr.write("process %s --> generating pool of %d points for evolution --> %.3f %% complete\r"%(os.getpid(),self.poolsize,100.0*float(n+1)/float(self.poolsize)))
                self.evolution_points[n] = Parameter(self.pulsars,model=self.model,noise=self.noise)
                self.evolution_points[n].set_bounds()
                self.evolution_points[n].initialise()
                self.evolution_points[n].logPrior()
                if not(isinf(self.evolution_points[n].logP)): break
        if self.verbose: sys.stderr.write("\n")
        self.dimension = 0
        for n in self.evolution_points[0].par_names:
            if self.evolution_points[0].vary[n]==1:
                self.dimension+=self.evolution_points[0].vary[n]
        self.kwargs = proposals._setup_kwargs(self.evolution_points,self.poolsize,self.dimension)
        self.kwargs = proposals._update_kwargs(**self.kwargs)

    def copy_params(self,param_in,param_out):
        """
        helper function to copy live points
        """
        param_out._internalvalues = np.copy(param_in._internalvalues)
        param_out.values = np.copy(param_in.values)
        param_out.logL = np.copy(param_in.logL)
        param_out.logP = np.copy(param_in.logP)

    def produce_sample(self, consumer_lock, queue, work_queue, seed):
        self.seed = seed
        np.random.seed(seed=self.seed)
        counter=1
        while(1):
            if not(work_queue.empty()):
                counter += 1
                logLmin = work_queue.get()
                if logLmin=='pill': break
                acceptance,jumps,outParam = self.MetropolisHastings(self.inParam,logLmin,self.Nmcmc,**self.kwargs)
                if (counter%4==0):
                    j = np.random.randint(self.poolsize)
                    self.copy_params(outParam,self.evolution_points[j])
                queue.put((acceptance,jumps,outParam._internalvalues,outParam.values,outParam.logP,outParam.logL))

#            else:
#                print 'Work queue looks empty'
#            print "consumer lock -->",consumer_lock
#            consumer_lock.release()
#            time.sleep(1)
#            print consumer_lock
            if (counter%(self.poolsize/4))==0 and len(self.cache)==5*self.maxmcmc:
                counter=1
                self.autocorrelation()
                self.kwargs=proposals._update_kwargs(**self.kwargs)
        sys.stderr.write("process %s, exiting\n"%os.getpid())
        return 0

    def MetropolisHastings(self,inParam,logLmin,nsteps,**kwargs):
        """
        mcmc loop to generate the new live point taking nmcmc steps
        """
        accepted = 0
        rejected = 1
        jumps = 0
        self.copy_params(inParam,self.param)
        while (jumps < nsteps or accepted==0):
            logP0 = self.param.logPrior()
            self.param,log_acceptance = self.proposals[jumps%100].get_sample(self.param,**kwargs)
            logP = self.param.logPrior()
            if logP-logP0 > log_acceptance:
                logLnew = self.param.logLikelihood()
                if logLnew > logLmin:
                    self.copy_params(self.param,inParam)
                    accepted+=1
                else:
                    self.copy_params(inParam,self.param)
                    rejected+=1
            else:
                self.copy_params(inParam,self.param)
                rejected+=1
            self.cache.append(inParam._internalvalues[:])
            jumps+=1
            if jumps==10*self.maxmcmc:
                return (0.0,jumps,inParam)
        return (float(accepted)/float(rejected+accepted),jumps,inParam)

    def autocorrelation(self):
        """
        estimates the autocorrelation length of the mcmc chain from the cached samples
        """
        try:
            ACLs = []
            cov_array = np.squeeze([[np.float64(p[n]) for n in self.evolution_points[0].par_names] for p in self.cache])
            N = len(self.cache)
            for i,n in enumerate(self.evolution_points[0].par_names):
              if self.evolution_points[0].vary[n]==1:
                ACF = autocorrelation(cov_array[:,i])
                ACL = np.min(np.where((ACF > -2./np.sqrt(N)) & (ACF < 2./np.sqrt(N)))[0])#sum(ACF**2)
                if not(np.isnan(ACL)):
                  ACLs.append(ACL)
                  if self.verbose: sys.stderr.write("autocorrelation length %s = %.1f mean = %g standard deviation = %g\n"%(n,ACLs[-1],np.mean(cov_array[:,i]),np.std(cov_array[:,i])))
            self.Nmcmc =int((np.max(ACLs)))
            if self.Nmcmc < 16: self.Nmcmc = 16
            if self.Nmcmc > self.maxmcmc:
                if self.verbose: sys.stderr.write("Warning ACL --> %d!\n"%self.Nmcmc)
                self.Nmcmc = self.maxmcmc
        except:
            sys.stderr.write("ACL computation failed!\n")
            self.Nmcmc = self.maxmcmc

if __name__=='__main__':
    import libstempo as T
    import optparse as op
    from libstempo import tempopulsar
    T.data = "/projects/pulsar_timing/nested_sampling/"
    parfiles =["/projects/pulsar_timing/ns-pt/pulsar_a.par","/projects/pulsar_timing/ns-pt/pulsar_b.par"]
    timfiles =["/projects/pulsar_timing/ns-pt/pulsar_a_zero_noise.simulate","/projects/pulsar_timing/ns-pt/pulsar_b.simulate","/projects/pulsar_timing/ns-pt/pulsar_b_zero_noise.simulate"]
    
    psrA = tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
    psrB = tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
    mcmc =  Sampler([psrA,psrB],128,model='Free',noise=None)
    mcmc.MetropolisHastings(mcmc.param,-np.inf,10)