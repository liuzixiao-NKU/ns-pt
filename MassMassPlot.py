from pylab import *
import numpy as np
import cPickle as pickle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
from ComputePKparameters import *
import optparse as op
import libstempo as T
from dpgmm import *
from dpgmm_parallel_solver import *
import multiprocessing as mp
from scipy.optimize  import newton,brentq
from collections import defaultdict
from scipy.stats import binned_statistic
import matplotlib.cm as cm

maxStick=2

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
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]
    
    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])
    
    adHeights = np.array(adHeights)

    return adHeights

from matplotlib import rc

rc('text', usetex=False)
rc('font', family='serif')
rc('font', serif='times')
#rc('font', weight='bolder')
rc('mathtext', default='sf')
rc('lines', markeredgewidth=1)
rc('lines', linewidth=2)
rc('axes', labelsize=18) #24
rc('axes', linewidth=0.5) #2)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('legend', fontsize=12) #16
rc('xtick.major', pad=8) #8)
rc('ytick.major', pad=8) #8)
rc('xtick.major', size=10) #8)
rc('ytick.major', size=10) #8)
rc('xtick.minor', size=5) #8)
rc('ytick.minor', size=5) #8)

def gamma_m1_wrapper(mc,gamma_samps,pb,ecc,mp):
    return gamma_samps-gamma(pb,ecc,mp,mc)

def omega_dot_wrapper(mc,omd_samps,pb,ecc,mp):
    return omd_samps-omega_dot(pb,ecc,mp,mc)

def pb_dot_wrapper(mc,pbd_samps,pb,ecc,mp):
    return pbd_samps-pbdot(pb,ecc,mp,mc)

def s_wrapper(mc,s_samps,pb,ap,mp):
    return s_samps-shapiroS(pb,mp,mc,ap)

def binned_mean(bins,vec):
    digitized = np.digitize(vec, bins)
    bin_means = np.array([vec[digitized == i].mean() for i in range(1, len(bins))])
    return bin_means

def binned_std(bins,vec):
    digitized = np.digitize(vec, bins)
    bin_std = np.array([vec[digitized == i].std() for i in range(1, len(bins))])
    return bin_std

def chunkplot(x, y, chunksize, ax=None, line_kwargs=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if line_kwargs is None:
        line_kwargs = {}
    # first sort the array in increasing x
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    # remove any invalid value
    idx = xrange(len(x))
    invalid=[]
    for i,xi,yi in zip(idx,x,y):
		if xi > 10.0 or yi > 10.0:
			invalid.append(i)
    x = np.delete(x,invalid)
    y = np.delete(y,invalid)
    
    # Wrap the array into a 2D array of chunks, truncating the last chunk if
    # chunksize isn't an even divisor of the total size.
    # (This part won't use _any_ additional memory)
    numchunks = y.size // chunksize
    ychunks = y[:chunksize*numchunks].reshape((-1, chunksize))
    xchunks = x[:chunksize*numchunks].reshape((-1, chunksize))

    # Calculate the max, min, and means of chunksize-element chunks...
#    max_env = ychunks.max(axis=1)
#    min_env = ychunks.min(axis=1)
    max_env = 3.0*ychunks.std(axis=1)+ychunks.mean(axis=1)
    min_env = ychunks.mean(axis=1)-3.0*ychunks.std(axis=1)
    ycenters = ychunks.mean(axis=1)
    xcenters = xchunks.mean(axis=1)

    # Now plot the bounds and the mean...
    fill = ax.fill_between(xcenters, min_env, max_env, **kwargs)
#    line = ax.plot(xcenters, ycenters, **line_kwargs)[0]
    return fill#, line

if __name__=='__main__':
    parser = op.OptionParser()
    parser.add_option("-N", type="int", dest="Nlive", help="Number of Live points",default=1000)
    (options, args) = parser.parse_args()
    Nlive = str(options.Nlive)
    parfiles =["/home/wdp/src/ns-pt/pulsar_a_nongr.par","/home/wdp/src/ns-pt/pulsar_b_nongr.par"]
    timfiles =["/home/wdp/src/ns-pt/pulsar_a_zero_noise_nongr.simulate","/home/wdp/src/ns-pt/pulsar_b_zero_noise_nongr.simulate"]
    tex_labels=[r"$M[M_\odot]$",r"$M[M_\odot]$",r"$a[lt\cdot s^{-1}]$",r"$a[lt\cdot s^{-1}]$",r"$\gamma[ms]$",r"$\gamma[ms]$",r"$P_b[\mathrm{days}]$",r"$\dot{P}_b[10^{-12}s^{-1}]$",r"$\dot{\omega}[\mathrm{deg}\cdot \mathrm{yr}^{-1}]$",r"$e$",r"$s$"]
    psrA = T.tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
    psrB = T.tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
    #posterior_samples = [genfromtxt( "dbl_psr/Free/posterior_samples.txt",names=True)]
    #posterior_samples.append(genfromtxt( "dbl_psr/CG/posterior_samples.txt",names=True))
    #posterior_samples.append(genfromtxt( "dbl_psr/GR/posterior_samples.txt",names=True))
    posterior_samples = [genfromtxt( "/home/wdp/src/ns-pt/Free/nonGR/rerun/posterior_samples.txt",names=True)]
    posterior_samples.append(genfromtxt( "/home/wdp/src/ns-pt/CG/nonGR/rerun/posterior_samples.txt",names=True))
    posterior_samples.append(genfromtxt( "/home/wdp/src/ns-pt/GR/nonGR/rerun/posterior_samples.txt",names=True))    
    # now do the 2D mass posterior
    myfig = figure()
    ax=axes([0.125,0.2,0.95-0.125,0.95-0.2])
    colors = ['k','r','g','b','y']
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    ax_inset = zoomed_inset_axes(ax, 10, loc = 1) #300, loc=1)
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



    invM2 = defaultdict(list)
    invM1 = defaultdict(list)

    for m1,m2,g1,g2,pb,e,a1,a2,om,pbd,s in zip(posterior_samples[0]["M2_PSRB"],posterior_samples[0]["M2_PSRA"],posterior_samples[0]["GAMMA_PSRA"],posterior_samples[0]["GAMMA_PSRB"],posterior_samples[0]["PB_PSR"],posterior_samples[0]["ECC_PSR"],posterior_samples[0]["A1_PSRA"],posterior_samples[0]["A1_PSRB"],posterior_samples[0]["OMDOT_PSR"],posterior_samples[0]["PBDOT_PSR"],posterior_samples[0]["SINI_PSR"]):
        try:
			if m1<3.0 and m2<3.0:
				invM2['GAMMA'].append((m1,newton(gamma_m1_wrapper,m2,args=(g1,pb,e,m1),maxiter=1000)))
				invM2['OMDOT'].append((m1,newton(omega_dot_wrapper,m2,args=(om,pb,e,m1),maxiter=1000)))
				invM2['PBDOT'].append((m1,newton(pb_dot_wrapper,m2,args=(pbd,pb,e,m1),maxiter=1000)))
				invM2['SINI'].append((m1,newton(s_wrapper,m2,args=(s,pb,a2,m1),maxiter=1000)))
				invM1['GAMMA'].append((m2,newton(gamma_m1_wrapper,m1,args=(g2,pb,e,m2),maxiter=1000)))
				invM1['OMDOT'].append((m2,newton(omega_dot_wrapper,m1,args=(om,pb,e,m2),maxiter=1000)))
				invM1['PBDOT'].append((m2,newton(pb_dot_wrapper,m1,args=(pbd,pb,e,m2),maxiter=1000)))
				invM1['SINI'].append((m2,newton(s_wrapper,m1,args=(s,pb,a1,m2),maxiter=1000)))
        except:
            pass

    colors = cm.rainbow(np.linspace(0, 1, 4))

    for lab,mark in zip(['GAMMA','OMDOT','PBDOT','SINI'],colors):

        invM1[lab] = np.array(invM1[lab])
        invM2[lab] = np.array(invM2[lab])
        chunkplot(invM1[lab][:,0],invM1[lab][:,1], chunksize=5, ax=ax,edgecolor=mark, alpha=0.5, color=mark, label = lab)
        chunkplot(invM2[lab][:,1],invM2[lab][:,0], chunksize=5, ax=ax,edgecolor=mark, alpha=0.5, color=mark,label = lab)
        chunkplot(invM1[lab][:,0],invM1[lab][:,1], chunksize=5, ax=ax_inset,edgecolor=mark, alpha=0.2, color=mark,label = lab)
        chunkplot(invM2[lab][:,1],invM2[lab][:,0], chunksize=5, ax=ax_inset,edgecolor=mark, alpha=0.2, color=mark,label = lab)

    ax.axvline(psrB.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
    ax.axhline(psrA.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
    ax_inset.axvline(psrB.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
    ax_inset.axhline(psrA.prefit['M2'].val,color='k',linestyle='dotted',alpha=0.5,linewidth=1.5)
    majorFormatter = FormatStrFormatter('%.1f')
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)
    majorFormatter = FormatStrFormatter('%.2f')
    ax_inset.set_xlim([1.1,1.7])
    ax_inset.set_ylim([1.1,1.7])
    ax_inset.xaxis.set_major_formatter(majorFormatter)
    ax_inset.yaxis.set_major_formatter(majorFormatter)
    ax.set_xlabel(r"$m_A[M_\odot]$")
    ax.set_ylabel(r"$m_B[M_\odot]$")
    ax.set_xlim(0.5,10)
    ax.set_ylim(0.5,10)
    plt.legend(loc=2,fancybox=True,shadow=True)
    ax_inset.set_xlabel(r"$m_A[M_\odot]$",fontsize=16)
    ax_inset.set_ylabel(r"$m_B[M_\odot]$",fontsize=16)
    myfig.savefig("m1m2_nongr.pdf",bbox_inches='tight')
