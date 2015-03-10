from matplotlib import rc
import numpy as np

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

maxmcmc = [128,256,512,1024]
nlive = [256,512,1024,2048,4096]

logZ = np.zeros((len(maxmcmc),len(nlive)))
dlogZ = np.zeros((len(maxmcmc),len(nlive)))
for i,live in enumerate(nlive):
  for j,mc in enumerate(maxmcmc):
    try:
      f = open("double_pulsar/whitenoise/logZ_acc/%d_%d/merged_chain_%d_%d.txt_evidence"%(live,mc,live,mc),"r")
      ev,dev = f.readline().split(None)
      logZ[j,i] = np.float(ev)
      dlogZ[j,i] = np.float(dev)
      f.close()
    except:
      print "double_pulsar/whitenoise/logZ_acc/%d_%d/merged_chain_%d_%d.txt_evidence not found!"%(live,mc,live,mc)
      pass

import matplotlib.pyplot as plt
plt.figure()
colors = ["k","r","g","b"]
for j,mc in enumerate(maxmcmc):
  plt.plot(nlive,logZ[j,:],'.', color=colors[j], label="$\mathrm{MCMC}$ $\mathrm{steps} = %d$"%(mc))
  plt.errorbar(nlive,logZ[j,:],yerr=3*dlogZ[j,:],fmt='.',color=colors[j])
plt.xticks(nlive)
plt.legend(fancybox=True,loc='lower right')
plt.grid(alpha=0.5)
plt.xlabel("$\mathrm{Live}$ $\mathrm{Points}$")
plt.ylabel("$\log Z$")
plt.show()