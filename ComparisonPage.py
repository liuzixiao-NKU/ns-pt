import optparse as op
from PTPostProcess import *

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

maxStick=8

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
  def oneDpos(self,ax,name,width,nbins,color):
    n, bins = np.histogram(self.samples[name], bins=linspace(np.min(self.samples[name]),np.max(self.samples[name]),nbins), normed=True)
    db = bins[1]-bins[0]
    p=np.cumsum(n*db)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    low=bincenters[find_nearest(p,0.025)]
    med=bincenters[find_nearest(p,0.5)]
    high=bincenters[find_nearest(p,0.975)]
    sys.stderr.write("plotting "+name+" --> ")
    sys.stderr.write("median: %.30f low 2.5: %.30f high 97.5: %.30f\n"%(med,low,high))
    # transform the samples in standard format
    m = np.mean(self.samples[name])
    s = np.std(self.samples[name])
    if (self.pulsars!=None) and (name!='GOB' and name!='XI' and name!='EPS' and name!='KAPPA' and 'TAU' not in name and 'SIGMA' not in name):
      fields = name.split('_')
      if fields[-1]=='PSR':
        injection = self.pulsars.pulsars['binaries'][0][0].prefit[str(fields[0])].val
      elif fields[-1]=='PSRA':
        injection = self.pulsars.pulsars['binaries'][0][0].prefit[str(fields[0])].val
      elif fields[-1]=='PSRB':
        injection = self.pulsars.pulsars['binaries'][0][1].prefit[str(fields[0])].val
      elif self.pulsars.pulsars['singles']!=None and fields[0]!='logL':
        injection = self.pulsars.pulsars['singles'][0].prefit[str(fields[0])].val
      else:
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
          p = np.exp(density_estimate)/s
          ax.plot(m+s*xplot,p/p.max(),color=color,linewidth=2.0)
        except:
          sys.stderr.write("DPGMM fit failed!\n")
#    ax.bar(bincenters,n/n.max(),width=0.9*diff(bincenters)[0],color=color,alpha=0.25,edgecolor='white')
#    x = bincenters[find_nearest(p,0.025):find_nearest(p,0.975)+1]
#    px=n[find_nearest(p,0.025):find_nearest(p,0.975)+1]
#    ax.bar(x,px/px.max(),width=0.9*diff(bincenters)[0],color=color,alpha=0.5,edgecolor='white')
    ax.axvline(low,linewidth=2, color=color,linestyle="--",alpha=0.5)
    ax.axvline(high,linewidth=2, color=color,linestyle="--",alpha=0.5)
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
    return ax
  
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
          binaries[0][n].val = np.copy(self.samples[name][-1])
          binaries[1][n].val = np.copy(self.samples[name][-1])
        else:
          for p in binaries:
            name = n+"_"+p.name
            p[n].val = np.copy(self.samples[name][-1])
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
      full_residuals = residuals+mu
      full_residuals_errors = np.sqrt((1e3*singles.toaerrs[i])**2+1.0/np.diag(cov))
      plt.errorbar(singles.toas()[i],full_residuals,yerr=full_residuals_errors,fmt='.',label=singles.name+" rms = %.3f ns"%np.sqrt(np.mean(full_residuals_errors**2)));
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
        tau = np.median(self.samples['logTAU_'+p.name])
        sigma = np.median(self.samples['logSIGMA_'+p.name])
        err = 1.0e3 * p.toaerrs # in ns
        gp = george.GP(sigma * sigma* kernels.ExpSquaredKernel(tau*tau))
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
      tau = np.median(self.samples['logTAU_'+singles.name])
      sigma = np.median(self.samples['logSIGMA_'+singles.name])
      for n in singles.pars:
        name = n+"_"+singles.name
        singles[n].val = np.copy(np.median(self.samples[name]))
      i = np.argsort(singles.toas())
      err = 1.0e3 * singles.toaerrs # in ns
      gp = george.GP(sigma *sigma * kernels.ExpSquaredKernel(tau*tau))
      gp.compute(singles.toas()[i], err[i])
      mu, cov = gp.predict(1e9*singles.residuals(updatebats=True,formresiduals=True)[i],singles.toas()[i])
      self.gp[singles.name]=gp
      mat = ax.matshow(cov, cmap=cm.seismic, vmin=-cov.max(), vmax=cov.max())
      plt.title(r"$\mathrm{noise}$ $\mathrm{covariance}$ $\mathrm{%s}$"%singles.name, y=1.10)
      cb = plt.colorbar(mat, orientation='horizontal')
      ticks = cb.ax.get_yticklabels()
      cb.ax.set_yticklabels(ticks, rotation=45.0)
      j+=1

    return myfig

if __name__=='__main__':
  parser = op.OptionParser()
  parser.add_option("-o","--output", type="string", dest="output", help="output location")
  parser.add_option("--parameters", type="string", dest="parfiles", help="pulsar parameter files", default=None, action='callback',callback=parse_to_list)
  parser.add_option("--times", type="string", dest="timfiles", help="pulsar time files, they must be ordered as the parameter files", default=None, action='callback',callback=parse_to_list)
  parser.add_option( "--DPGMM", type="int", dest="dpgmm", help="fit a dpgmm to the posteriors", default="1")
  (options, args) = parser.parse_args()

  pos = ["double_pulsar/whitenoise/free/posterior_samples.txt"]
  pos.append("double_pulsar/whitenoise/gr/posterior_samples.txt")
  pos.append("double_pulsar/whitenoise/cg/posterior_samples.txt")
  
  ev = ["double_pulsar/whitenoise/free/merged_chain.txt_evidence"]
  ev.append("double_pulsar/whitenoise/gr/merged_chain.txt_evidence")
  ev.append("double_pulsar/whitenoise/cg/merged_chain.txt_evidence")
  
  psrs = [T.tempopulsar(parfile = par,timfile = tim) for par,tim in zip(options.parfiles,options.timfiles)]
  param = Parameter(psrs,model='Free')
  
  if options.dpgmm:
    pool = mp.Pool(mp.cpu_count())
  
  posteriors = [Posterior(p,e,psrs,1) for p,e in zip(pos,ev)]

  htmlfile=open(options.output+'/comparison.html','w')
  htmlfile.write('<HTML><HEAD><TITLE>Comparison Page</TITLE></HEAD><BODY><h3>Comparison Page</h3>')
  WIDTH = 5.0
  nbins1D = 64
  colors = ['k','g','r']
  for n in posteriors[0].csv_names:
    for p,c in zip(posteriors,colors):
      myfig_pos=plt.figure(1)
      ax = plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
      ax = p.oneDpos(ax,n,WIDTH,nbins1D,c)
    myfig_pos.savefig(options.output+'/'+n+ '_comp.png',bbox_inches='tight')
    myfig_pos.savefig(options.output+'/'+n+ '_comp.pdf',bbox_inches='tight')
    myfig_pos.clf()
    htmlfile.write('<tr><td><img src="'+n+'_comp.png">')

  htmlfile.write('</BODY></HTML>')
  htmlfile.close()