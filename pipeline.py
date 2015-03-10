import optparse as op
import sys
import os
import multiprocessing as mp
import threading
from NestedSampling import *

def NSwrapper(q,NS):
  q.put(NS.nested_sampling)

def parse_to_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
  parser = op.OptionParser()
  parser.add_option("-N", type="int", dest="Nlive", help="Number of Live points",default=1000)
  parser.add_option("-o", "--output", type="string", dest="output", help="Output folder", default="Free")
  parser.add_option("-t", "--type", type="string", dest="model", help="Class of model to assume (Free, GR, CG)", default=None)
  parser.add_option("-s", type="int", dest="seed", help="seed for the chain", default=1000)
  parser.add_option("--maxmcmc", type="int", dest="maxmcmc", help="maximum number of mcmc steps", default=5000)
  parser.add_option("--nthreads", type="int", dest="nthreads", help="number of processes", default=4)
  parser.add_option("--nruns", type="int", dest="nruns",help="number of parallel runs",default=1)
  parser.add_option("--parameters", type="string", dest="parfiles", help="pulsar parameter files", default=None, action='callback',
                    callback=parse_to_list)
  parser.add_option("--times", type="string", dest="timfiles", help="pulsar time files, they must be ordered as the parameter files", default=None, action='callback',
                                      callback=parse_to_list)
  (options, args) = parser.parse_args()
  NLIVE = options.Nlive
  MAXMCMC = options.maxmcmc
  NRUNS = options.nruns
  NProcesses = options.nthreads
  output = options.output
  queueLock = threading.Lock()
  NSpool = Queue()
  threads = []
  for i in xrange(options.seed,options.seed+NRUNS):
    Ns = NestedSampler(options.parfiles,options.timfiles,model=options.model,Nlive=options.Nlive,maxmcmc=options.maxmcmc,verbose=True,output=options.output,seed=i,nthreads=options.nthreads)
    thread = threading.Thread(target=Ns.nested_sampling,name=str(i))
    threads.append(thread)
    thread.start()
  queueLock.acquire()
  queueLock.release()
  while not NSpool.empty():
    pass
  for t in threads:
    t.join()
  outfiles = ""

  arguments="-N %d -o %s/merged_chain.txt -p %s/posterior_samples.txt --header %s/header.txt "%(NLIVE,output,output,output)
  for i in xrange(options.seed,options.seed+NRUNS):
    arguments+=os.path.join(output,"chain_"+options.model+"_"+str(options.Nlive)+"_"+str(i)+".txt")
    arguments+=" "
  os.system("python merge_runs.py %s"%arguments)
  arguments="-i %s/posterior_samples.txt -e %s/merged_chain.txt_evidence -o %s/posteriors/ --parameters %s --times %s"%(output,output,output,','.join(options.parfiles),','.join(options.timfiles))
  os.system("python PTPostProcess.py %s"%arguments)