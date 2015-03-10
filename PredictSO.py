#! /usr/bin/env python

import numpy as np
import scipy
import scipy.stats.kde
import matplotlib.pyplot as plt
import sys
import math
from pylab import *
import string
import random
import os
import optparse as op
import cPickle as pickle
from scipy import integrate
#from gcp import gcp
#from gcp.gaussian import Gaussian
from dpgmm import *
import libstempo as T
from Parameters import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
from ComputePKparameters import *

fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

from matplotlib import rc

rc('text', usetex=False)
rc('font', family='serif')
rc('font', serif='times')
#rc('font', weight='bolder')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=1)
rc("lines", linewidth=3)
rc('axes', labelsize=18) #24
rc("axes", linewidth=0.5) #2)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('legend', fontsize=12) #16
rc('xtick.major', pad=8) #8)
rc('ytick.major', pad=8) #8)
rc('xtick.major', size=13) #8)
rc('ytick.major', size=13) #8)
rc('xtick.minor', size=7) #8)
rc('ytick.minor', size=7) #8)

def set_tick_sizes(ax, major, minor):
  for l in ax.get_xticklines() + ax.get_yticklines():
    l.set_markersize(major)
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks():
      tick.tick1line.set_markersize(minor)
      tick.tick2line.set_markersize(minor)
    ax.xaxis.LABELPAD=10.
    ax.xaxis.OFFSETTEXTPAD=10.

if __name__=='__main__':
  parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par"]
  timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b.simulate"]
  psrA = T.tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
  psrB = T.tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
  grfile = 'GR/posterior_samples.txt'
  pos=np.genfromtxt(grfile,names=True)
  SOA = array([omegaSO(pbi,ecci,mci,mpi) for pbi,ecci,mci,mpi in zip(pos['PB_PSR'],pos['ECC_PSR'],pos['M2_PSRB'],pos['M2_PSRA'])])
  SOB = array([omegaSO(pbi,ecci,mci,mpi) for pbi,ecci,mci,mpi in zip(pos['PB_PSR'],pos['ECC_PSR'],pos['M2_PSRA'],pos['M2_PSRB'])])
  myfig = figure()
  ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
  p, bins = numpy.histogram(SOA, bins=32, normed=True)
  db = bins[1]-bins[0]
  bincenters = 0.5*(bins[1:]+bins[:-1])
  ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
  xlabel('SOA')
  ylabel(r"$probability$ $density$")
  myfig.savefig("SOA_GR.pdf",bbox_inches='tight')
  close(myfig)
  myfig = figure()
  ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
  p, bins = numpy.histogram(SOB, bins=32, normed=True)
  db = bins[1]-bins[0]
  bincenters = 0.5*(bins[1:]+bins[:-1])
  ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
  xlabel('SOB')
  ylabel(r"$probability$ $density$")
  myfig.savefig("SOB_GR.pdf",bbox_inches='tight')
  close(myfig)
  # now the CG
  cgfile = 'CG/posterior_samples.txt'
  pos=np.genfromtxt(cgfile,names=True)
  #beta0(pb,Mtot,GOB)
  beta = array([beta0(pbi,mpi+mci,gobi) for pbi,mpi,mci,gobi in zip(pos['PB_PSR'],pos['M2_PSRB'],pos['M2_PSRA'],pos['GOB'],)])
  #omegaSOAG(pb,mp,mc,GammaAB,GOB,beta0,ecc)
  SOA = array([omegaSOAG(pbi,mpi,mci,gammai,gobi,betai,ecci) for pbi,mpi,mci,gammai,gobi,betai,ecci in zip(pos['PB_PSR'],pos['M2_PSRB'],pos['M2_PSRA'],2.0*G*pos['GOB'],pos['GOB'],beta,pos['ECC_PSR'])])
  SOB = array([omegaSOAG(pbi,mpi,mci,gammai,gobi,betai,ecci) for pbi,mpi,mci,gammai,gobi,betai,ecci in zip(pos['PB_PSR'],pos['M2_PSRA'],pos['M2_PSRB'],2.0*G*pos['GOB'],pos['GOB'],beta,pos['ECC_PSR'])])
  myfig = figure()
  ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
  p, bins = numpy.histogram(SOA, bins=32, normed=True)
  db = bins[1]-bins[0]
  bincenters = 0.5*(bins[1:]+bins[:-1])
  ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
  xlabel('SOA')
  ylabel(r"$probability$ $density$")
  myfig.savefig("SOA_CG.pdf",bbox_inches='tight')
  close(myfig)
  myfig = figure()
  ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
  p, bins = numpy.histogram(SOB, bins=32, normed=True)
  db = bins[1]-bins[0]
  bincenters = 0.5*(bins[1:]+bins[:-1])
  ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
  xlabel('SOB')
  ylabel(r"$probability$ $density$")
  myfig.savefig("SOB_CG.pdf",bbox_inches='tight')
  close(myfig)
