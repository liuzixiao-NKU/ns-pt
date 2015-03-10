from pylab import *
import cPickle as pickle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
from ComputePKparameters import *
import optparse as op
import libstempo as T
from dpgmm import *
from dpgmm_parallel_solver import *
import multiprocessing as mp

maxStick=32

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
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False,
    'figure.figsize': fig_size}

rcParams.update(params)
if __name__=='__main__':
  parser = op.OptionParser()
  parser.add_option("-N", type="int", dest="Nlive", help="Number of Live points",default=1000)
  (options, args) = parser.parse_args()
  Nlive = str(options.Nlive)
  parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par"]
  timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b.simulate"]
  tex_labels=[r"$M[M_\odot]$",r"$M[M_\odot]$",r"$a[lt\cdot s^{-1}]$",r"$a[lt\cdot s^{-1}]$",r"$\gamma[ms]$",r"$\gamma[ms]$",r"$P_b[\mathrm{days}]$",r"$\dot{P}_b[10^{-12}s^{-1}]$",r"$\dot{\omega}[\mathrm{deg}\cdot \mathrm{yr}^{-1}]$",r"$e$",r"$s$"]
  psrA = T.tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
  psrB = T.tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
  posterior_samples = [genfromtxt( "double_pulsar/whitenoise/free/posterior_samples.txt",names=True)]
  posterior_samples.append(genfromtxt( "double_pulsar/whitenoise/cg/posterior_samples.txt",names=True))
  posterior_samples.append(genfromtxt( "double_pulsar/whitenoise/gr/posterior_samples.txt",names=True))
  # now do the 2D mass posterior
  myfig = figure()
  ax=axes([0.125,0.2,0.95-0.125,0.95-0.2])
  colors = ['k','r','g','b','y']
  from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
  from mpl_toolkits.axes_grid1.inset_locator import mark_inset
  ax_inset = zoomed_inset_axes(ax, 20, loc = 1) #300, loc=1)
  mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
  
  pool = mp.Pool(mp.cpu_count())
  # free masses
  for k,p in enumerate(posterior_samples):
      n,xedges,yedges = histogram2d(p['M2_PSRB'],p['M2_PSRA'],bins=128)
      x = 0.5*(xedges[1:]+xedges[:-1])
      y = 0.5*(yedges[1:]+yedges[:-1])
      model = DPGMM(2)
      m1 = np.mean(p['M2_PSRB'])
      s1 = np.std(p['M2_PSRB'])
      m2 = np.mean(p['M2_PSRA'])
      s2 = np.std(p['M2_PSRA'])
      points = np.column_stack(((p['M2_PSRB']-m1)/s1,(p['M2_PSRA']-m2)/s2))
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
      density_estimate = np.zeros((len(x),len(y)))
      density_estimate[:] = -np.inf
      print "Best model has %d components"%len(density[0])
      jobs = [((density[0][ind],prob),((x-m1)/s1,(y-m2)/s2)) for ind,prob in enumerate(density[1])]
      results = pool.map(sample_dpgmm,jobs)
      density_estimate = reduce(np.logaddexp,results)

      levels = FindHeightForLevel(density_estimate,[0.68,0.95,0.99])
      if k==0:
        ax.scatter(p['M2_PSRB'],p['M2_PSRA'],s=2,c="0.5",marker='.',alpha=0.25)
        C = ax.contour(density_estimate,levels,linestyles='-',colors='k',linewidths=1.5, hold='on',origin='lower',extent=[np.min(p['M2_PSRB']),np.max(p['M2_PSRB']),np.min(p['M2_PSRA']),np.max(p['M2_PSRA'])])
#        fmt = {}
#        strs = ['$68\%$','$95\%$','$99\%$']
#        for l,s in zip( C.levels, strs ):
#          fmt[l] = s
#        clabel(C,C.levels,inline=True,fmt=fmt,fontsize=12)

      if k==1:
        ax_inset.scatter(p['M2_PSRB'],p['M2_PSRA'],s=2,c="r",marker='.',alpha=0.25)
        levels = FindHeightForLevel(density_estimate,[0.68,0.95,0.99])
        C = ax_inset.contour(density_estimate,levels,linestyles='-',colors='r',linewidths=1.5, hold='on',origin='lower',extent=[np.min(p['M2_PSRB']),np.max(p['M2_PSRB']),np.min(p['M2_PSRA']),np.max(p['M2_PSRA'])])
        C = ax.contour(density_estimate,levels,linestyles='-',colors='r',linewidths=1.5, hold='on',origin='lower',extent=[np.min(p['M2_PSRB']),np.max(p['M2_PSRB']),np.min(p['M2_PSRA']),np.max(p['M2_PSRA'])])
      if k==2:
        ax_inset.scatter(p['M2_PSRB'],p['M2_PSRA'],s=2,c="g",marker='.',alpha=0.25)
        levels = FindHeightForLevel(density_estimate,[0.68,0.95,0.99])
        C = ax_inset.contour(density_estimate,levels,linestyles='-',colors='g',linewidths=1.5, hold='on',origin='lower',extent=[np.min(p['M2_PSRB']),np.max(p['M2_PSRB']),np.min(p['M2_PSRA']),np.max(p['M2_PSRA'])])
        C = ax.contour(density_estimate,levels,linestyles='-',colors='g',linewidths=1.5, hold='on',origin='lower',extent=[np.min(p['M2_PSRB']),np.max(p['M2_PSRB']),np.min(p['M2_PSRA']),np.max(p['M2_PSRA'])])

  ax.axvline(psrB.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
  ax.axhline(psrA.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
  ax_inset.axvline(psrB.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
  ax_inset.axhline(psrA.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
  majorFormatter = FormatStrFormatter('%.1f')
  ax.xaxis.set_major_formatter(majorFormatter)
  ax.yaxis.set_major_formatter(majorFormatter)
  majorFormatter = FormatStrFormatter('%.2f')
#  ax_inset.set_xlim([1.32,1.35])
#  ax_inset.set_ylim([1.24,1.27])
  ax_inset.xaxis.set_major_formatter(majorFormatter)
  ax_inset.yaxis.set_major_formatter(majorFormatter)
  ax.set_xlabel(r"$m_A[M_\odot]$")
  ax.set_ylabel(r"$m_B[M_\odot]$")
  ax_inset.set_xlabel(r"$m_A[M_\odot]$",fontsize=16)
  ax_inset.set_ylabel(r"$m_B[M_\odot]$",fontsize=16)
  myfig.savefig("m1m2.pdf",bbox_inches='tight')