# coding: utf-8
#! /usr/bin/env python
from pylab import *
import numpy
import cPickle as pickle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
from ComputePKparameters import *
import optparse as op
import libstempo as T
import triangle
import os
import sys

par_names = ['RAJ','DECJ','F0_0','F1_0','F0_1','F1_1','DM','PMRA','PMDEC','PX','SINI','PB','T0','A1_0','A1_1','OM_0','OM_1','ECC','PBDOT','OMDOT','M2_0','M2_1','GAMMA_0','GAMMA_1','GOB','EPS','XI','KAPPA']
par_formats = [np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128]
tex_labels=[r"$$",r"$$",r"$\mathrm{Hz}$",r"$\mathrm{Hz}/s$",r"$\mathrm{Hz}$",r"$\mathrm{Hz}/s$",r"$\mathrm{DM}$",r"$M[M_\odot]$",r"$M[M_\odot]$",r"$a[lt\cdot s^{-1}]$",r"$a[lt\cdot s^{-1}]$",r"$\gamma[ms]$",r"$\gamma[ms]$",r"$P_b[\mathrm{days}]$",r"$\dot{P}_b[10^{-12}s^{-1}]$",r"$\dot{\omega}[\mathrm{deg}\cdot \mathrm{yr}^{-1}]$",r"$e$",r"$s$"]

fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'png',
    'axes.labelsize': 24,
    'text.fontsize': 28,
    'legend.fontsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False,
    'figure.figsize': fig_size}

rcParams.update(params)

if __name__=='__main__':
  parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par"]
  timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b.simulate"]
  psrA = T.tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
  psrB = T.tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
  parser = op.OptionParser()
  parser.add_option("-N", type="int", dest="Nlive", help="Number of Live points")
  parser.add_option("-t", type="string", dest="model", help="Model to analyse")
  parser.add_option("-o", type="string", dest="output", help="output folder")
  parser.add_option("-i", type="string", dest="input", help="input folder")
  (options, args) = parser.parse_args()
  Nlive = str(options.Nlive)
  pos = genfromtxt(options.input+"/posterior_samples_"+options.model+"_"+str(options.Nlive)+"_1.txt",names=True)
  logZ = pickle.load(open( options.input+"/Chain_"+options.model+"_"+str(options.Nlive)+"_1.p_evidence", "rb" ) )
  print "N samples -->",pos.shape[0]
  print "logZ -->",logZ
  os.system("mkdir -p joint/"+options.model)
  os.system("mkdir -p joint/"+options.output+"/"+options.model)
#  os.chdir("joint/")
  injections = []
  tridata = []
  trilabel = []
  for n in pos.dtype.names:
    if numpy.std(pos[n])>0.0:
      tridata.append(pos[n])
      trilabel.append(n)
    myfig = figure()
    ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
    p, bins = numpy.histogram(pos[n], bins=32, normed=True)
    db = bins[1]-bins[0]
    bincenters = 0.5*(bins[1:]+bins[:-1])
    ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
    #xticks(linspace(bincenters.min(),bincenters.max(),10),rotation=90.0)
    if "_0" in n:
      ax.axvline(psrA.prefit[str(n.split("_")[0])].val,color='r')
      injections.append(psrA.prefit[str(n.split("_")[0])].val)
    elif "_1" in n:
      ax.axvline(psrB.prefit[str(n.split("_")[0])].val,color='r')
      injections.append(psrB.prefit[str(n.split("_")[0])].val)
    else:
      try:
        ax.axvline(psrA.prefit[n].val,color='r')
        injections.append(psrA.prefit[n].val)
      except:
        pass
    mu = (p*db*bincenters).sum()
    sig = sqrt((p*db*(bincenters-mu)**2).sum())
    sys.stderr.write("%s & %.30f & %.30f & %.30f & %.30f\\ \n"%(n,mu,sig,injections[-1],(mu-injections[-1])/sig))
#    ax.xaxis.set_major_formatter(majorFormatterX)
    xlabel(n)
    ylabel(r"$probability$ $density$")
    myfig.savefig("joint/"+options.output+"/"+options.model+"/"+n+".pdf",bbox_inches='tight')
    close(myfig)
  if options.model=="CG":
    injections.append(1)
    injections.append(3)
    injections.append(1)
    injections.append(0)
  """
  tridata = numpy.array(tridata)
  trifigure = triangle.corner(tridata.T, labels=trilabel, truth = injections,
                         quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_args={"fontsize": 12})
  trifigure.savefig("joint/"+options.output+"/"+options.model+"/triangle.pdf")
  """
  # predict the spin orbit precession
  if options.model == "GR":
    SOA = array([omegaSO(pbi,ecci,mci,mpi) for pbi,ecci,mci,mpi in zip(pos['PB'],pos['ECC'],pos['M2_0'],pos['M2_1'])])
    SOB = array([omegaSO(pbi,ecci,mci,mpi) for pbi,ecci,mci,mpi in zip(pos['PB'],pos['ECC'],pos['M2_1'],pos['M2_0'])])
    myfig = figure()
    ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
    p, bins = numpy.histogram(SOA, bins=32, normed=True)
    db = bins[1]-bins[0]
    bincenters = 0.5*(bins[1:]+bins[:-1])
    ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
    xlabel('SOA')
    ylabel(r"$probability$ $density$")
    myfig.savefig("joint/"+options.output+"/"+options.model+"/SOA.pdf",bbox_inches='tight')
    close(myfig)
    myfig = figure()
    ax = axes([0.125,0.2,0.95-0.125,0.95-0.2])
    p, bins = numpy.histogram(SOB, bins=32, normed=True)
    db = bins[1]-bins[0]
    bincenters = 0.5*(bins[1:]+bins[:-1])
    ax.bar(bincenters,p,width=0.9*diff(bincenters)[0],color="0.5",alpha=0.25,edgecolor='white')
    xlabel('SOB')
    ylabel(r"$probability$ $density$")
    myfig.savefig("joint/"+options.output+"/"+options.model+"/SOB.pdf",bbox_inches='tight')
    close(myfig)
  elif options.model == "CG":
    SO = omegaSOAG (pb,mp,mc,GammaAB,GOB,beta0,ecc)
#  os.chdir("..")