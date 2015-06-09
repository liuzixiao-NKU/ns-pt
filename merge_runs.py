from optparse import OptionParser
import matplotlib
matplotlib.use("Agg")
import scipy
from pylab import *
import sys
import math
import numpy
import os
import cPickle as pickle
from functools import partial

parser=OptionParser()
parser.add_option('-N','--Nlive',action='store',type='int',dest='Nlive',default=None,help='Number of live points in each chain loaded',metavar='NUM')
parser.add_option('-o','--out',action='store',type='string',default=None,help='Output file for merged chain',metavar='outfile.dat')
parser.add_option('-p','--pos',action='store',type='string',default=None,help='Output file for posterior samples if requested',metavar='posterior.dat')
parser.add_option('-e','--header',action='store',type='string',default=None,help='Output file for posterior samples if requested',metavar='header.dat')

logLcol=-1 # default to last column

(opts,args)=parser.parse_args()
print opts
header_file=open(opts.header,'r')

headers=header_file.readline().split(None)
header_file.close()
logLcol=headers.index('logL')

if opts.Nlive is None:
  parser.error('You must specify a number of live points using --Nlive')
if opts.out is None:
  parser.error('You must specify an output file')

if len(args)==0:
  parser.error('No input files specified, exiting')

def logadd(a,b):
  if(a>b): (a,b)=(b,a)
  return(b+log(1.0+exp(a-b)))

def nestPar(d,Nlive):
  """
  Do nested sampling on parallel runs, taking into account
  different run lengths
  """
  maxes=[]
  for set in d:
    maxes.append(set[-1,logLcol]) # Max L point for each chain
  maxes=array(maxes)
  maxes.sort()
  N=len(d)
  print 'N chains = '+str(N)
  logw = log(1.0-exp(-1.0/(N*Nlive)))
  H=0
  alldat=reduce(lambda x,y: hstack([x,y]) , map(lambda x: x[:,logLcol],d))
  sidx=argsort(alldat)
  alldat=alldat[sidx]
  logZ=logw + alldat[0]
  logw-=1.0/(N*Nlive)
  weights = zeros(size(alldat,0))
  weights[0]=logw
  j=0
  numsamp=size(alldat,0)
  for samp in alldat[1:]:
    if samp>maxes[0]:
      maxes=maxes[1:]
      N-=1
      print str(N)+' Parallel chains left at %d/%d'%(j,numsamp)
    logZnew = logadd(logZ,logw+samp)
    H = exp(logw + samp -logZnew)*samp \
    + exp(logZ-logZnew)*(H+logZ) - logZnew
    logw = logw -1.0/(N*Nlive)
    j+=1
    weights[j]=logw
    logZ=logZnew
  bigdata=reduce(lambda x,y: vstack([x,y]), d)
  bigdata=bigdata[sidx]
  return (logZ,H,bigdata,weights)

mapfunc = partial(genfromtxt, invalid_raise=False)

def loaddata(datalist):
  out = list(map(mapfunc,datalist))
  Bfiles = list(map(getBfile,datalist))
  return out,Bfiles

def getBfile(datname):
  
  Bfile=datname+'_evidence'
  print 'Looking for '+Bfile
  if os.access(Bfile,os.R_OK):
    outstat = pickle.load(open(Bfile, "rb" ) )
    return outstat
  else:
    return None

def nest2pos_par(samps,weights):
  """
  resample nested sampling samples to get posterior samples
  """
  randoms=rand(size(samps,0))
  wt=weights+samps[:,logLcol]
  maxwt=max(wt)
  posidx=find(wt>maxwt+log(randoms))
  print posidx
  pos=samps[posidx,:]
  return pos

Nlive=int(opts.Nlive)

data,Bfiles=loaddata(args)
#for d in data:
#    data[0] = transpose(sort(transpose(data[0]),axis=-1))

(logZ,H,d_sorted,weights)=nestPar(data,Nlive)

logLmax=d_sorted[-1,logLcol]

outfile=open(opts.out,'w')
for row in d_sorted:
  for i in row:
    outfile.write('%.30f\t'%(i))
  outfile.write('\n')
outfile.close()

Bfile=open(opts.out+'_evidence','w')
Bfile.write('%lf %lf\n'%(logZ,sqrt(H/Nlive)))
Bfile.close()

if opts.pos is not None:
  pos=nest2pos_par(d_sorted,weights)
  posfile=open(opts.pos,'w')
  for h in headers:
    posfile.write('%s\t'%(h))
  posfile.write('\n')
  for row in pos:
    for i in row:
      posfile.write('%.30f\t'%(i))
    posfile.write('\n')
  posfile.close()
if(os.access(args[0]+'_params.txt',os.R_OK)):
  import shutil
  if opts.pos:
    shutil.copy(args[0]+'_params.txt',opts.pos+'_params.txt')
  if opts.out:
    shutil.copy(args[0]+'_params.txt',opts.out+'_params.txt')