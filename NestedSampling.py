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
from multiprocessing import Process, Lock, Queue
import proposals
import signal
from Sampler import *

import copy_reg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def runSamplerInParallel(listOf_FuncAndArgLists):
  """
    Take a list of lists like [function, arg1, arg2, ...]. Run those functions in parallel, wait for them all to finish, and return the list of their return values, in order.
    
    (This still needs error handling ie to ensure everything returned okay.)
    
    """
  from multiprocessing import Process, Queue
  
  def storeOutputFFF(fff,theArgs,que,**kwargs): #add a argument to function for assigning a queue
    print "MULTIPROCESSING: Launching %s in parallel "%fff.func_name
    que.put(fff(*theArgs,**kwargs)) #we're putting return value into queue
  
  queues=[Queue() for fff in listOf_FuncAndArgLists] #create a queue object for each function
  jobs = [Process(target=storeOutputFFF,args=[funcArgs[0],funcArgs[1:-1],queues[iii]],kwargs=funcArgs[-1]) for iii,funcArgs in enumerate(listOf_FuncAndArgLists)]
  for job in jobs: job.start() # Launch them all
  #for job in jobs: job.join() # Wait for them all to finish
  # And now, collect all the outputs:
  return([queue.get() for queue in queues])

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
        specify if the analysis should be done assuming the tempo2 model, the DDGR model of the DDCG model
        default: None
    
    output: 
        folder where the output will be stored
    
    verbose: 
        display information on screen
        default: True
        
    seed:
        seed for the initialisation of the pseudorandom chain
        
    nthreads: 
        number of sampling threads to spawn in parallel. Note that this is still an experimental feature.
        Default: 1
    
    prior:
        produce N samples from the prior
        
    noise:
        model to use for the noise
        default: None
    """
    def __init__(self,parfiles,timfiles,Nlive=1024,maxmcmc=4096,model=None,noise=None,output=None,verbose=True,seed=1,nthreads=1,prior=None):
        """
        Initialise all necessary arguments and variables for the algorithm
        """

        self.prior_sampling = prior
        self.model = model
        self.noise = noise
        self.setup_random_seed(seed)
        self.verbose = verbose
        self.active_index = None
        self.accepted = 0
        self.rejected = 1
        self.dimension = None
        self.Nlive = Nlive
        self.Nmcmc = maxmcmc
        self.maxmcmc = maxmcmc
        self.output,self.evidence_out,self.checkpoint = self.setup_output(output)
        self.pulsars = [tempopulsar(parfile = parfiles[i], timfile = timfiles[i]) for i in xrange(len(parfiles))]
        for p in self.pulsars: p.fit()
        self.samples = []
        self.params = [None] * self.Nlive
        self.logZ = np.finfo(np.float128).min
        self.tolerance = 0.01
        self.condition = None
        self.logLmin = None
        self.information = 0.0
        self.worst = 0
        self.logLmax = None
        self.iteration = 0
        self.logLinj = computeLogLinj(self.pulsars)
        for n in xrange(self.Nlive):
            while True:
                if self.verbose: sys.stderr.write("sprinkling %d live points --> %.3f %% complete\r"%(self.Nlive,100.0*float(n+1)/float(self.Nlive)))
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
        self.active_live = Parameter(self.pulsars,model=self.model,noise=self.noise)
        self.active_live.set_bounds()
        self.active_live.initialise()
        self.active_live.logPrior()
        if self.loadState()==0:
            sys.stderr.write("Loaded state %s, resuming run\n"%self.checkpoint)
            self.new_run=False
        else:
            sys.stderr.write("Checkpoint not found, starting anew\n")
            self.new_run=True

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
        return open(os.path.join(output,outputfile),"a"),open(os.path.join(output,outputfile+"_evidence.txt"), "wb" ),os.path.join(output,outputfile+"_resume")

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def _select_live(self):
        """
        select a live point
        """
        while True:
            j = np.random.randint(self.Nlive)
            if j!= self.worst:
                return j

    def copy_params(self,param_in,param_out):
        """
        helper function to copy live points
        """
        param_out._internalvalues = np.copy(param_in._internalvalues)
        param_out.values = np.copy(param_in.values)
        param_out.logL = np.copy(param_in.logL)
        param_out.logP = np.copy(param_in.logP)

    def consume_sample(self, producer_lock, queue, work_queue):
        """
        main nested sampling loop
        """
        logwidth = np.log(1.0 - np.exp(-1.0 / float(self.Nlive)))
        if self.new_run==True:
            """
            generate samples from the prior
            """
            for i in xrange(self.Nlive):
                while True:
                    if (work_queue.empty()):
                        work_queue.put(-np.inf)
                    if not(queue.empty()):
                        acceptance,jumps,_internalvalues,values,logP,logL = queue.get()
                        self.params[i].values = np.copy(values)
                        self.params[i]._internalvalues = np.copy(_internalvalues)
                        self.params[i].logP = np.copy(logP)
                        self.params[i].logL = np.copy(logL)
                        if self.params[i].logP!=-np.inf or self.params[i].logL!=-np.inf: break
                if self.verbose: sys.stderr.write("sampling the prior --> %.3f %% complete\r"%(100.0*float(i+1)/float(self.Nlive)))

            if self.verbose: sys.stderr.write("\n")
        while not(work_queue.empty()): work_queue.get()
        self.condition = np.inf
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
        self.active_index =self._select_live()
        self.copy_params(self.params[self.active_index],self.params[self.worst])
        work_queue.put(self.logLmin)
        while self.condition > self.tolerance:
            while not(queue.empty()):
                self.rejected+=1
                acceptance,jumps,_internalvalues,values,logP,logL = queue.get()
                self.params[self.worst].values = np.copy(values)
                self.params[self.worst]._internalvalues = np.copy(_internalvalues)
                self.params[self.worst].logP = np.copy(logP)
                self.params[self.worst].logL = np.copy(logL)
                
                if self.params[self.worst].logL>self.logLmin:
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
                    self.active_index =self._select_live()
                    self.copy_params(self.params[self.active_index],self.params[self.worst])
                    if self.verbose: sys.stderr.write("%d: n:%4d acc:%.3f H: %.2f logL %.5f --> %.5f dZ: %.3f logZ: %.3f logLmax: %.5f logLinj: %.5f\n"%(self.iteration,jumps,acceptance,self.information,self.logLmin,self.params[self.worst].logL,self.condition,self.logZ,self.logLmax,self.logLinj))
                    logwidth-=1.0/float(self.Nlive)
                    self.iteration+=1
            if work_queue.empty():
                work_queue.put(self.logLmin)
        # empty the queue
        while not(work_queue.empty()): work_queue.get()
        # put as many None as sampler processes
        for _ in xrange(NUMBER_OF_PRODUCER_PROCESSES): work_queue.put(None)
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
        sys.stderr.write("process %s, exiting\n"%os.getpid())
        return 0

    def saveState(self):
        try:
            livepoints_stack = np.zeros(self.Nlive,dtype={'names':self.params[0].par_names,'formats':self.params[0].par_types})
            for i in xrange(self.Nlive):
                for n in self.params[0].par_names:
                    livepoints_stack[i][n] = self.params[i]._internalvalues[n]
            resume_out = open(self.checkpoint,"wb")
            pickle.dump((livepoints_stack,np.random.get_state(),self.iteration,self.cache),resume_out)
            sys.stderr.write("Checkpointed %d live points.\n"%self.Nlive)
            resume_out.close()
            return 0
        except:
            sys.stderr.write("Checkpointing failed!\n")
            return 1

    def loadState(self):
        try:
            resume_in = open(self.checkpoint,"rb")
            livepoints_stack,RandomState,self.iteration,self.cache = pickle.load(resume_in)
            resume_in.close()
            for i in xrange(self.Nlive):
                for n in self.params[0].par_names:
                    self.params[i]._internalvalues[n] = livepoints_stack[i][n]
                self.params[i].logPrior()
                self.params[i].logLikelihood()
            np.random.set_state(RandomState)
            self.kwargs=proposals._setup_kwargs(self.params,self.Nlive,self.dimension)
            self.kwargs=proposals._update_kwargs(**self.kwargs)
            sys.stderr.write("Resumed %d live points.\n"%self.Nlive)
            return 0
        except:
            sys.stderr.write("Resuming failed!\n")
            return 1


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
    parser.add_option("--nthreads", type="int", dest="nthreads", help="number of sampling threads to spawn", default=None)
    parser.add_option("--parameters", type="string", dest="parfiles", help="pulsar parameter files", default=None, action='callback',
                    callback=parse_to_list)
    parser.add_option("--times", type="string", dest="timfiles", help="pulsar time files, they must be ordered as the parameter files", default=None, action='callback',
                                      callback=parse_to_list)
    parser.add_option( "--sample-prior", action="store_true", dest="prior", help="draw samples from the prior", default=False)
    parser.add_option( "--noise", dest="noise", type="string", help="noise model to assume", default=None)
    (options, args) = parser.parse_args()
    if len(options.parfiles)!= len(options.timfiles):
        sys.stderr.write("Fatal error! The number of par files is different from the number of time files!\n")
        exit(-1)

    NS = NestedSampler(options.parfiles,options.timfiles,model=options.model,seed=options.seed,noise=options.noise,Nlive=options.Nlive,maxmcmc=options.maxmcmc,output=options.output,verbose=options.verbose,nthreads=options.nthreads,prior=options.prior)
    Evolver = Sampler(NS.pulsars,options.maxmcmc,model=options.model,noise=options.noise)

    NUMBER_OF_PRODUCER_PROCESSES = options.nthreads
    NUMBER_OF_CONSUMER_PROCESSES = 1

    process_pool = []
    ns_lock = Lock()
    sampler_lock = Lock()
    queue = Queue()
    work_queue = Queue()
    for i in xrange(0,NUMBER_OF_PRODUCER_PROCESSES):
        p = Process(target=Evolver.produce_sample, args=(ns_lock, queue, work_queue, options.seed+i, ))
        process_pool.append(p)
    for i in xrange(0,NUMBER_OF_CONSUMER_PROCESSES):
        p = Process(target=NS.consume_sample, args=(sampler_lock, queue, work_queue,))
        process_pool.append(p)
    for each in process_pool:
        each.start()
