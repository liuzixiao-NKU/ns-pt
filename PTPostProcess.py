#! /usr/bin/env python

import numpy as np
import scipy
import scipy.stats.kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import math
from pylab import *
import string
import random
import os
import optparse as op
import cPickle as pickle
from scipy import integrate
import george
from george import kernels
from dpgmm import *
import libstempo as T
from Parameters_v2 import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
from matplotlib.colors import LogNorm
from dpgmm_parallel_solver import *
import multiprocessing as mp

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
rc('lines', markeredgewidth=1)
rc('lines', linewidth=3)
rc('axes', labelsize=18) #24
rc('axes', linewidth=0.5) #2)
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

maxStick=16

class Posterior:
  def __init__(self,data_file,evidence_file,pulsars=None,dpgmm=0):
    
    data=np.genfromtxt(data_file,skip_header=1)
    f = open(data_file,'r') #"double_pulsar/whitenoise/prior/free/header.txt" data_file
    self.csv_names = f.readline().split(None)
    f.close()
    self.samples= data.view(dtype=[(n, 'float64') for n in self.csv_names]).reshape(len(data))
    for n in self.csv_names:
      if 'log' in n and n!='logL':
        self.samples[n]=np.exp(self.samples[n])
    self.logZ = np.loadtxt(evidence_file)
    self.pulsars = Parameter(pulsars,model='Free')
    self.dpgmm = dpgmm
    self.gp = {}
  def oneDpos(self,fig,name,width,nbins):
    n, bins = np.histogram(self.samples[name], bins=linspace(np.min(self.samples[name]),np.max(self.samples[name]),nbins), normed=True)
    db = bins[1]-bins[0]
    p=np.cumsum(n*db)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    low=bincenters[find_nearest(p,0.025)]
    med=bincenters[find_nearest(p,0.5)]
    high=bincenters[find_nearest(p,0.975)]
    sys.stderr.write("plotting "+name+" --> ")
    sys.stderr.write("median: %.30f low 2.5: %.30f high 97.5: %.30f\n"%(med,low,high))
    ax = plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    # transform the samples in standard format
    m = np.mean(self.samples[name])
    s = np.std(self.samples[name])
    #if (self.pulsars!=None) and (name!='GOB' and name!='XI' and name!='EPS' and name!='KAPPA' and 'TAU' not in name and 'SIGMA' not in name):
      #fields = name.split('_')
      #if fields[-1]=='PSR':
        #injection = self.pulsars.pulsars['binaries'][0][0].prefit[str(fields[0])].val
      #elif fields[-1]=='PSRA':
        #injection = self.pulsars.pulsars['binaries'][0][0].prefit[str(fields[0])].val
      #elif fields[-1]=='PSRB':
        #injection = self.pulsars.pulsars['binaries'][0][1].prefit[str(fields[0])].val
      #elif self.pulsars.pulsars['singles']!=None and fields[0]!='logL':
        #injection = self.pulsars.pulsars['singles'][0].prefit[str(fields[0])].val
      #else:
    injection = None
    if self.dpgmm:
      model = DPGMM(1)
      if s > 0.0:
        try:
          for j,point in enumerate(self.samples[name]):
            model.add([(point-m)/s])
          model.setPrior()
          model.setThreshold(1e-3)
          model.setConcGamma(1.0,1.0)
          jobs = [(i,model) for i in xrange(maxStick)]
          results = pool.map(solve_dpgmm,jobs)
          scores = np.zeros(maxStick)
          for i,r in enumerate(results):
            scores[i] = r[1]
          for i,r in enumerate(results):
            if i==scores.argmax():
              model = r[-1]
              break

          xplot=np.linspace((np.min(self.samples[name])-m)/s,(np.max(self.samples[name])-m)/s,128)
          density = model.intMixture()
          density_estimate=np.zeros(len(xplot),dtype=np.float128)
          density_estimate[:] = -np.inf
          print "Best model has %d components"%len(density[0])
          jobs = [((density[0][ind],prob),(xplot)) for ind,prob in enumerate(density[1])]
          results = pool.map(sample_dpgmm,jobs)
          density_estimate = reduce(np.logaddexp,results)
          ax.plot(m+s*xplot,np.exp(density_estimate)/s,"k",linewidth=2.0)
        except:
          sys.stderr.write("DPGMM fit failed!\n")
    ax.bar(bincenters,n,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
    x = bincenters[find_nearest(p,0.025):find_nearest(p,0.975)+1]
    px=n[find_nearest(p,0.025):find_nearest(p,0.975)+1]
    ax.bar(x,px,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.5,edgecolor='white')
    ax.axvline(low,linewidth=2, color='r',linestyle="--",alpha=0.5)
    ax.axvline(high,linewidth=2, color='r',linestyle="--",alpha=0.5)
    if (self.pulsars!=None) and (name!='GOB' and name!='XI' and name!='EPS' and name!='KAPPA' and 'TAU' not in name and 'SIGMA' not in name):
      ax.axvline(injection,linewidth=2, color='k',linestyle="--",alpha=0.5)
    #    axvline((self.injections[index]-self.tempo2values[index])/self.errors[index],linewidth=2, color='r',linestyle="--",alpha=0.5)
    #plt.xticks(linspace(-width,width,11),rotation=45.0)
    majorFormatterX = FormatStrFormatter('%.15f')
    majorFormatterY = FormatStrFormatter('%.15f')
    ax.xaxis.set_major_formatter(majorFormatterX)
#    ax.yaxis.set_major_formatter(majorFormatterY)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.xticks(np.linspace(np.min(self.samples[name]),np.max(self.samples[name]),11),rotation=45.0)
    xlabel(r"$\mathrm{"+name+"}$")
    ylabel(r"$probability$ $density$")
    return fig
  
  def twoDpos(self,name1,name2,width,nbins):
    from matplotlib.colors import LogNorm
    if (self.pulsars!=None) and (name1!='GOB' and name1!='XI' and name1!='EPS' and name1!='KAPPA' and name2!='GOB' and name2!='XI' and name2!='EPS' and name2!='KAPPA'):
      
      fields1 = name1.split('_')
      fields2 = name2.split('_')
      if (fields1[-1]=='PSR' and fields2[-1]=='PSR') or (fields1[-1]=='PSR' and fields2[-1]=='PSRA') or (fields1[-1]=='PSRA' and fields2[-1]=='PSR') or (fields1[-1]=='PSRA' and fields2[-1]=='PSRA'):
        injection = [self.pulsars.pulsars['binaries'][0][0].prefit[str(fields1[0])].val,self.pulsars.pulsars['binaries'][0][0].prefit[str(fields2[0])].val]
      elif (fields1[-1]=='PSR' and fields2[-1]=='PSRB') or (fields1[-1]=='PSRB' and fields2[-1]=='PSRB'):
        injection = [self.pulsars.pulsars['binaries'][0][1].prefit[str(fields1[0])].val,self.pulsars.pulsars['binaries'][0][1].prefit[str(fields2[0])].val]
      elif (fields1[-1]=='PSRB' and fields2[-1]=='PSR') or (fields1[-1]=='PSRB' and fields2[-1]=='PSRA'):
        injection = [self.pulsars.pulsars['binaries'][0][1].prefit[str(fields1[0])].val,self.pulsars.pulsars['binaries'][0][0].prefit[str(fields2[0])].val]
      elif (fields1[-1]=='PSRA' and fields2[-1]=='PSRB'):
        injection = [self.pulsars.pulsars['binaries'][0][0].prefit[str(fields1[0])].val,self.pulsars.pulsars['binaries'][0][1].prefit[str(fields2[0])].val]
      elif self.pulsars.pulsars['singles']!=None and fields1[0]!='logL' and fields2[0]!='logL' and 'logTAU' not in fields1[0] and 'logSIGMA' not in fields1[0] and 'logTAU' not in fields2[0] and 'logSIGMA' not in fields2[0]:
          injection = [self.pulsars.pulsars['singles'][0].prefit[str(fields1[0])].val,self.pulsars.pulsars['singles'][0].prefit[str(fields2[0])].val]
      else:
        injection = None
    sys.stderr.write("plotting "+name1+" vs "+name2+"\n")
    m1 = np.mean(self.samples[name1])
    s1 = np.std(self.samples[name1])
    m2 = np.mean(self.samples[name2])
    s2 = np.std(self.samples[name2])
    if s1 > 0.0 and s2 > 0.0:
      myhist,xedges,yedges=np.histogram2d(self.samples[name1],self.samples[name2],bins=[np.linspace(np.min(self.samples[name1]),np.max(self.samples[name1]),nbins),np.linspace(np.min(self.samples[name2]),np.max(self.samples[name2]),nbins)])
    
      if self.dpgmm:
        try:
          model = DPGMM(2)

          xplot=np.linspace((np.min(self.samples[name1])-m1)/s1,(np.max(self.samples[name1])-m1)/s1,128)
          yplot=np.linspace((np.min(self.samples[name2])-m2)/s2,(np.max(self.samples[name2])-m2)/s2,128)

          points = np.column_stack(((self.samples[name1]-m1)/s1,(self.samples[name2]-m2)/s2))
          for l,point in enumerate(points):
            model.add(point)
          model.setPrior()
          model.setThreshold(1e-3)
          model.setConcGamma(1.0,1.0)
          
          jobs = [(i,model) for i in xrange(maxStick)]
          results = pool.map(solve_dpgmm,jobs)
          scores = np.zeros(maxStick)
          for i,r in enumerate(results):
              scores[i] = r[1]
          for i,r in enumerate(results):
            if i==scores.argmax():
              model = r[-1]
              break
          density = model.intMixture()
          print "Best model has %d components"%len(density[0])
          density_estimate = np.zeros((len(xplot),len(yplot)))
          density_estimate[:] = -np.inf
      
          jobs = [((density[0][ind],prob),(xplot,yplot)) for ind,prob in enumerate(density[1])]
          results = pool.map(sample_dpgmm,jobs)
          density_estimate = reduce(np.logaddexp,results)
          plotME=1
        except:
          plotME=0
          print "failed to converge!"
          pass
      else: plotME=0
      myfig=figure(1)
      ax = plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
      ax.scatter((self.samples[name1]-m1)/s1,(self.samples[name2]-m2)/s2,s=2,c="0.5",marker='.',alpha=0.25)
      if (plotME==1 and self.dpgmm):
        levels = FindHeightForLevel(density_estimate,[0.68,0.95,0.99])#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])#
        C = ax.contour(density_estimate,levels,linestyles='-',colors='k',linewidths=1.5, hold='on',origin='lower',extent=[np.min(xplot),np.max(xplot),np.min(yplot),np.max(yplot)])
      elif (plotME==1):
        levels = FindHeightForLevel(np.log(myhist),[0.68,0.95,0.99])#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])#
        C = ax.contour(np.log(myhist).T,levels,linestyles='-',colors='k',linewidths=1.5, hold='on',origin='lower',extent=[np.min(xplot),np.max(xplot),np.min(yplot),np.max(yplot)])
      if (self.pulsars!=None) and (name1!='GOB' and name1!='XI' and name1!='EPS' and name1!='KAPPA' and name2!='GOB' and name2!='XI' and name2!='EPS' and name2!='KAPPA' and 'logTAU' not in name1 and 'logSIGMA' not in name1 and 'logTAU' not in name2 and 'logSIGMA' not in name2 and 'logL' not in name1 and 'logL' not in name2):
        ax.axvline((injection[0]-m1)/s1,linewidth=2, color='k',linestyle="--",alpha=0.5)
        ax.axhline((injection[1]-m2)/s2,linewidth=2, color='k',linestyle="--",alpha=0.5)
      majorFormatterX = FormatStrFormatter('%.15f')
      majorFormatterY = FormatStrFormatter('%.15f')
      ax.xaxis.set_major_formatter(majorFormatterX)
      ax.yaxis.set_major_formatter(majorFormatterY)
      plt.xticks(np.linspace(np.min(xplot),np.max(xplot),11),rotation=45.0)
      plt.yticks(np.linspace(np.max(xplot),np.min(yplot),11),rotation=45.0)
#        plt.xlim(np.min(self.samples[name1]),np.max(self.samples[name1]))
#        plt.ylim(np.min(self.samples[name2]),np.max(self.samples[name2]))
      plt.xlabel(r"$\mathrm{"+name1+"}$")
      plt.ylabel(r"$\mathrm{"+name2+"}$")
      return myfig

  def plotresiduals(self):
    myfig=plt.figure(1)
    ax = plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    for binaries in self.pulsars.pulsars['binaries']:
      for n in binaries[0].pars:
        if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
          name = n+"_"+binaries[0].name[:-1]
          binaries[0][n].val = np.copy(np.median(self.samples[name]))
          binaries[1][n].val = np.copy(np.median(self.samples[name]))
        else:
          for p in binaries:
            name = n+"_"+p.name
            p[n].val = np.copy(np.median(self.samples[name]))
      for p in binaries:
        i = np.argsort(p.toas())
        residuals = 1e9*p.residuals(updatebats=True,formresiduals=True)[i]
#        ax.plot(p.toas()[i],self.gp[p.name].sample_conditional(residuals[i], p.toas()[i]))
        ax.errorbar(p.toas()[i],residuals,yerr=1e3*p.toaerrs[i],fmt='.',label=p.name+" rms = %.3f ns"%np.sqrt(np.mean(residuals**2)));

    for singles in self.pulsars.pulsars['singles']:
      for n in singles.pars:
        name = n+"_"+singles.name
        singles[n].val = np.copy(np.median(self.samples[name]))
      i = np.argsort(singles.toas())
      residuals = 1e9*singles.residuals(updatebats=True,formresiduals=True)[i]
      mu, cov = self.gp[singles.name].predict(residuals,singles.toas()[i])
      full_residuals = residuals
      full_residuals_errors = (1e3*singles.toaerrs[i])
      plt.errorbar(singles.toas()[i],full_residuals,yerr=full_residuals_errors,fmt='.',color='k',label="$"+singles.name+"$ $\mathrm{rms} = %.3f ns $"%np.sqrt(np.mean(full_residuals**2)));
      singles.fit()
      tempo_residuals = 1e9*singles.residuals(updatebats=True,formresiduals=True)[i]
      plt.errorbar(singles.toas()[i],tempo_residuals,yerr=full_residuals_errors,fmt='.',color='r',label="$"+singles.name+"$ $\mathrm{rms} = %.3f ns (\mathrm{TEMPO2})$"%np.sqrt(np.mean(tempo_residuals**2)));
    plt.xlabel("$\mathrm{MJD}$",fontsize=18)
    plt.ylabel(r"$\mathrm{residuals}/\mathrm{ns}$",fontsize=18)
    plt.legend(loc='best')
    return myfig

  def plotCovariance(self):
    N = 2*len(self.pulsars.pulsars['binaries'])+len(self.pulsars.pulsars['singles'])
    myfig=plt.figure(1)
    j = 1
    for binaries in self.pulsars.pulsars['binaries']:
      for n in binaries[0].pars:
        if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
          name = n+"_"+binaries[0].name[:-1]
          binaries[0][n].val = np.copy(np.median(self.samples[name]))
          binaries[1][n].val = np.copy(np.median(self.samples[name]))
        else:
          for p in binaries:
            name = n+"_"+p.name
            p[n].val = np.copy(np.median(self.samples[name]))
    for binaries in self.pulsars.pulsars['binaries']:
      for p in binaries:
        ax = myfig.add_subplot(1,N,j)
        i = np.argsort(p.toas())
        tau = np.median(self.samples['logTAU_'+p.name])/86400.0 # we are translating it in days to be consistent with the toas untis
        sigma = np.median(1e-9*self.samples['logSIGMA_'+p.name])
        equad = np.median(1e-9*self.samples['logEQUAD_'+p.name])
        err = 1.0e3 * p.toaerrs # in ns
        gp = george.GP(sigma * sigma* kernels.ExpSquaredKernel(tau*tau)+kernels.WhiteKernel(equad*equad))
        gp.compute(p.toas()[i], err[i])
        mu, cov = gp.predict(1e9*p.residuals(updatebats=True,formresiduals=True)[i],p.toas()[i])
        self.gp[p.name]=gp
        mat = ax.matshow(cov, cmap=cm.seismic, vmin=-cov.max(), vmax=cov.max())
        plt.title(r"$\mathrm{noise}$ $\mathrm{covariance}$ $\mathrm{%s}$"%p.name, y=1.10)
        cb = plt.colorbar(mat, orientation='horizontal')
        ticks = cb.ax.get_yticklabels()
        cb.ax.set_yticklabels(ticks, rotation=45.0)
        j+=1
    for singles in self.pulsars.pulsars['singles']:
      ax = myfig.add_subplot(1,N,j)
      tau = np.median(self.samples['logTAU_'+singles.name])/86400.0 # we are translating it in days to be consistent with the toas untis
      sigma = np.median(1e-9*self.samples['logSIGMA_'+singles.name])
      equad = np.median(1e-9*self.samples['logEQUAD_'+singles.name])
      for n in singles.pars:
        name = n+"_"+singles.name
        singles[n].val = np.copy(np.median(self.samples[name]))
      i = np.argsort(singles.toas())
      err = 1.0e3 * singles.toaerrs # in ns
      gp = george.GP(sigma *sigma * kernels.ExpSquaredKernel(tau*tau)+kernels.WhiteKernel(equad*equad))
      gp.compute(singles.toas()[i], err[i])
      mu, cov = gp.predict(1e9*singles.residuals(updatebats=True,formresiduals=True)[i],singles.toas()[i])
      self.gp[singles.name]=gp
      mat = ax.matshow(cov, cmap=cm.seismic, vmin=-cov.max(), vmax=cov.max())
      plt.title(r"$\mathrm{noise}$ $\mathrm{covariance}$ $\mathrm{%s}$"%singles.name, y=1.10)
      cb = plt.colorbar(mat, orientation='horizontal')
      ticks = cb.ax.get_yticklabels()
      cb.ax.set_yticklabels(ticks, rotation=45.0)
      cb.set_label(r"$\mathrm{d}^2$")
      j+=1

    return myfig

  def plotNoisePSD(self):
    N = 2*len(self.pulsars.pulsars['binaries'])+len(self.pulsars.pulsars['singles'])
    myfig=plt.figure(1)
    j = 1
    for binaries in self.pulsars.pulsars['binaries']:
      for n in binaries[0].pars:
        if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
          name = n+"_"+binaries[0].name[:-1]
          binaries[0][n].val = np.copy(np.median(self.samples[name]))
          binaries[1][n].val = np.copy(np.median(self.samples[name]))
        else:
          for p in binaries:
            name = n+"_"+p.name
            p[n].val = np.copy(np.median(self.samples[name]))
    for binaries in self.pulsars.pulsars['binaries']:
      for p in binaries:
        ax = myfig.add_subplot(1,N,j)
        i = np.argsort(p.toas())
        tau = np.median(self.samples['logTAU_'+p.name])/86400.0 # we are translating it in days to be consistent with the toas untis
        sigma = np.median(1e-9*self.samples['logSIGMA_'+p.name])
        equad = np.median(1e-9*self.samples['logEQUAD_'+p.name])
        err = 1.0e3 * p.toaerrs # in ns
        gp = george.GP(sigma * sigma* kernels.ExpSquaredKernel(tau*tau)+kernels.WhiteKernel(equad*equad))
        gp.compute(p.toas()[i], err[i])
        mu, cov = gp.predict(1e9*p.residuals(updatebats=True,formresiduals=True)[i],p.toas()[i])
        self.gp[p.name]=gp
        mat = ax.matshow(cov, cmap=cm.seismic, vmin=-cov.max(), vmax=cov.max())
        plt.title(r"$\mathrm{noise}$ $\mathrm{covariance}$ $\mathrm{%s}$"%p.name, y=1.10)
        cb = plt.colorbar(mat, orientation='horizontal')
        ticks = cb.ax.get_yticklabels()
        cb.ax.set_yticklabels(ticks, rotation=45.0)
        j+=1
    for singles in self.pulsars.pulsars['singles']:
      ax = myfig.add_subplot(1,N,j)
      M = 16*4096
      samps = xrange(np.size(self.samples['logL']))
      autocovariance = np.zeros((np.size(self.samples['logL']),M))
      r = np.linspace(1.0/86400.0 ,np.exp(21.)/86400.0 ,M)
      taus = self.samples['logTAU_'+singles.name]/86400.0 # we are translating it in days to be consistent with the toas untis
      sigmas = self.samples['logSIGMA_'+singles.name]
      equads = self.samples['logEQUAD_'+singles.name]
      for i,tau,sigma,equad in zip(xrange(np.size(self.samples['logL'])),taus,sigmas,equads):
          autocovariance[i,:] = (sigma *sigma * np.exp(-0.5*(r/tau)**2))
          autocovariance[i,0] += equad
      colors = ['r','b','k','b','r']
      frequency = []
      psds = []
      for i in xrange(np.size(self.samples['logL'])):
        frequency.append(np.fft.rfftfreq(np.size(autocovariance[i,:]),np.diff(r)[0]))
        psds.append(1e-18*(np.fft.rfft(autocovariance[i,:]).real))
      ax.fill_between(frequency[0],np.percentile(psds,97.5,axis=0),np.percentile(psds,2.5,axis=0),facecolor='r',alpha=0.5)
      ax.fill_between(frequency[0],np.percentile(psds,84.,axis=0),np.percentile(psds,16.,axis=0),facecolor='b',alpha=0.5)
      ax.plot(frequency[0],np.percentile(psds,2.5,axis=0),color='r')
      ax.plot(frequency[0],np.percentile(psds,97.5,axis=0),color='r')
      ax.plot(frequency[0],np.percentile(psds,16.,axis=0),color='b')
      ax.plot(frequency[0],np.percentile(psds,84.,axis=0),color='b')
#      ax.plot(frequency[0],np.percentile(psds,1.5,axis=0),color='g')
#      ax.plot(frequency[0],np.percentile(psds,98.5,axis=0),color='g')
      ax.plot(frequency[0],np.median(psds,axis=0),color='k')
      #ax.axhline(1e-18*np.median(equads),color='g')
      ax.axvline(1.0/365.,color='k')#31556926.
      plt.yscale('log', nonposy='clip')
      plt.xscale('log')
      plt.ylabel("$P(f)/[s^2 d]$")
      plt.xlabel("$\mathrm{d}^{-1}$")
#      plt.ylim(1e-20,1e-8)
#      plt.xlim(1./np.exp(21.),0.5/M)
#      plt.grid(alpha=0.5)
      plt.title(r"$\mathrm{power}$ $\mathrm{spectral}$ $\mathrm{density}$ $\mathrm{%s}$"%singles.name, y=1.10)
      j+=1
          # plot hc(f) = sqrt(12 pi**2 f**3 psd(f))
    return myfig

def find_nearest(array,value):
  idx=(numpy.abs(array-value)).argmin()
  return idx

def CheckPath(location):
  if os.path.exists(location):
    print "path exists..."
    #string="rm -rf "+location
    #os.system(string)
    #os.makedirs(location)
  else:
    print "creating "+location
    os.makedirs(location)
    print "done"

def FindHeightForLevel(inArr, adLevels):
  # flatten the array
  oldshape = shape(inArr)
  adInput= reshape(inArr,oldshape[0]*oldshape[1])
 # GET ARRAY SPECIFICS
  nLength = np.size(adInput)
      
  # CREATE REVERSED SORTED LIST
  adTemp = -1.0 * adInput
  adSorted = np.sort(adTemp)
  adSorted = -1.0 * adSorted

  # CREATE NORMALISED CUMULATIVE DISTRIBUTION
  adCum = np.zeros(nLength)
  adCum[0] = adSorted[0]
  for i in xrange(1,nLength):
    adCum[i] = lnPLUS(adCum[i-1], adSorted[i])
  adCum = adCum - adCum[-1]

  # FIND VALUE CLOSEST TO LEVELS
  adHeights = []
  for item in adLevels:
    idx=(np.abs(adCum-np.log(item))).argmin()
    adHeights.append(adSorted[idx])

  adHeights = np.array(adHeights)
  return adHeights

def lnPLUS(x,y):
  max1 = maximum(x,y)
  min1 = minimum(x,y)
  return max1 + log1p(exp(min1 - max1))

def hms_rad(value):
  p=list(value.split(":"))
  return 360.0*(numpy.pi/180.0)*(float(p[0])/24.0+float(p[1])/1440.0+float(p[2])/86400.0)
def dms_rad(value):
  p=list(value.split(":"))
  return 360.0*(numpy.pi/180.0)*(float(p[0])/360.0+float(p[1])/21600.0+float(p[2])/1296000.0)

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i

def parse_to_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

#-------------------
# start the program
#-------------------

# parse arguments
if __name__=='__main__':
#  parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a_nongr.par","/projects/pulsar_timing/nested_sampling/pulsar_b_nongr.par"]
#  timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a_nongr.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b_nongr.simulate"]
#  parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par"]
#  timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b.simulate"]
#  psrA = T.tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
#  psrB = T.tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
  parser = op.OptionParser()
  parser.add_option("-i", "--input", type="string", dest="input", help="Input file")
  parser.add_option("-e", "--evidence", type="string", dest="evidence", help="Evidence file")
  parser.add_option("-o", "--output", type="string", dest="output", help="Output folder", default="posteriors")
  parser.add_option( "--2D", type="int", dest="twodplot", help="enable 2d posteriors", default="0")
  parser.add_option( "--DPGMM", type="int", dest="dpgmm", help="fit a dpgmm to the posteriors", default="0")
  parser.add_option("--parameters", type="string", dest="parfiles", help="pulsar parameter files", default=None, action='callback',
                    callback=parse_to_list)
  parser.add_option("--times", type="string", dest="timfiles", help="pulsar time files, they must be ordered as the parameter files", default=None, action='callback',
                                      callback=parse_to_list)
  parser.add_option("--noise-model",type="string",dest="noisemodel",help="noise model assumed in the analysis", default="white")
  (options, args) = parser.parse_args()
  psrs = [T.tempopulsar(parfile = par,timfile = tim) for par,tim in zip(options.parfiles,options.timfiles)]
  param = Parameter(psrs,model='Free')

  if options.dpgmm:
    pool = mp.Pool(mp.cpu_count())
  posteriors = Posterior(options.input,options.evidence,psrs,options.dpgmm)

  location=options.output+'/'
  CheckPath(location)
  nbins1D = 64
  nbins2D = 64

  htmlfile=open(location+'/posterior.html','w')
  htmlfile.write('<HTML><HEAD><TITLE>Posterior PDFs</TITLE></HEAD><BODY><h3>Posterior PDFs</h3>')
  htmlfile.write('Produced using '+str(np.size(posteriors.samples['logL']))+' posterior samples <br>')
  htmlfile.write('logZ = '+str(posteriors.logZ[0])+' error = '+str(posteriors.logZ[1])+'<br>')

  from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator

  htmlfile.write('<h4> Medians <h4><br>')
  htmlfile.write('<table border=1><tr>')
  namesline=reduce(lambda a,b:a+'<td>'+b,posteriors.samples.dtype.names)
  htmlfile.write('<td>'+namesline+'</tr><tr>')
  valuesline=""
  for n in posteriors.samples.dtype.names:
    valuesline+='%.20e<td>'%(np.median(posteriors.samples[n]))

  htmlfile.write('<td>'+valuesline+'</tr></table>')
  htmlfile.write('<br><hr>')

  htmlfile.write('<h4> Means <h4><br>')
  htmlfile.write('<table border=1><tr>')
  namesline=reduce(lambda a,b:a+'<td>'+b,posteriors.samples.dtype.names)
  htmlfile.write('<td>'+namesline+'</tr><tr>')
  valuesline=""
  for n in posteriors.samples.dtype.names:
    valuesline+='%.20e<td>'%(np.mean(posteriors.samples[n]))

  htmlfile.write('<td>'+valuesline+'</tr></table>')
  htmlfile.write('<br><hr>')

  htmlfile.write('<h4> Maximum Likelihood <h4><br>')
  htmlfile.write('<table border=1><tr>')
  namesline=reduce(lambda a,b:a+'<td>'+b,posteriors.samples.dtype.names)
  htmlfile.write('<td>'+namesline+'</tr><tr>')
  valuesline=""
  for n in posteriors.samples.dtype.names:
    valuesline+='%.20e<td>'%(posteriors.samples[n][-1])

  htmlfile.write('<td>'+valuesline+'</tr></table>')
  htmlfile.write('<br><hr>')

  htmlfile.write('<h4> Standard deviation <h4><br>')
  htmlfile.write('<table border=1><tr>')
  namesline=reduce(lambda a,b:a+'<td>'+b,posteriors.samples.dtype.names)
  htmlfile.write('<td>'+namesline+'</tr><tr>')
  valuesline=""
  for n in posteriors.samples.dtype.names:
    valuesline+='%.20e<td>'%(np.std(posteriors.samples[n]))

  htmlfile.write('<td>'+valuesline+'</tr></table>')
  htmlfile.write('<br><hr>')

  majorFormatterX = FormatStrFormatter('%g')
  majorFormatterY = FormatStrFormatter('%g')
  WIDTH = 5.0

  if options.noisemodel=="red":
    htmlfile.write('<br><hr>')
    myfig_pos = posteriors.plotCovariance()
    myfig_pos.savefig(location+'/covariance.png',bbox_inches='tight')
    myfig_pos.savefig(location+'/covariance.pdf',bbox_inches='tight')
    myfig_pos.clf()
    htmlfile.write('<tr><td><img src="covariance.png">')
    htmlfile.write('<br><hr>')
  
    htmlfile.write('<br><hr>')
    myfig_pos = posteriors.plotNoisePSD()
    myfig_pos.savefig(location+'/psd.png',bbox_inches='tight')
    myfig_pos.savefig(location+'/psd.pdf',bbox_inches='tight')
    myfig_pos.clf()
    htmlfile.write('<tr><td><img src="psd.png">')
    htmlfile.write('<br><hr>')

  myfig_pos = posteriors.plotresiduals()
  myfig_pos.savefig(location+'/residuals.png',bbox_inches='tight')
  myfig_pos.savefig(location+'/residuals.pdf',bbox_inches='tight')
  myfig_pos.clf()
  htmlfile.write('<tr><td><img src="residuals.png">')
  exit()
  for n in posteriors.csv_names:
    myfig_pos=plt.figure(1)
    myfig_pos = posteriors.oneDpos(myfig_pos,n,WIDTH,nbins1D)
    myfig_pos.savefig(location+'/'+n+ '_pos.png',bbox_inches='tight')
    myfig_pos.savefig(location+'/'+n+ '_pos.pdf',bbox_inches='tight')
    myfig_pos.clf()
    htmlfile.write('<tr><td><img src="'+n+'_pos.png">')

  if (options.twodplot):
    for i,n1 in enumerate(posteriors.csv_names):
      for j,n2 in enumerate(posteriors.csv_names):
        if j>i:
          myfig=posteriors.twoDpos(n1,n2,WIDTH,nbins2D)
          myfig.savefig(location+n1+'_'+n2 + '_pos.png',bbox_inches='tight')
          myfig.savefig(location+n1+'_'+n2 + '_pos.pdf',bbox_inches='tight')
          myfig.clf()
          htmlfile.write('<img src="'+n1+'_'+n2 + '_pos.png">')
          htmlfile.write('<br />')
  htmlfile.write('</BODY></HTML>')
  htmlfile.close()
  exit()
