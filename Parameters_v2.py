import numpy as np
from collections import defaultdict
from ComputePKparameters import *
import libstempo as T
from libstempo import tempopulsar
import sys
import george
from george import kernels

par_names = ['RAJ','DECJ','F0_0','F1_0','F0_1','F1_1','DM','PMRA','PMDEC','PX','SINI','PB','T0','A1_0','A1_1','OM_0','OM_1','ECC','PBDOT','OMDOT','M2_0','M2_1','GAMMA_0','GAMMA_1','GOB','EPS','XI','KAPPA']
par_formats = [np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128,np.float128]
vary_formats = [np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int,np.int]

width = 5.

def computeLogLinj(pulsars):
	"""
	Injection Likelihood function for white uncorrelated gaussian noise
	"""
	# fill the pulsars
	logL=0.0
	for p in pulsars:
		for n in p.pars:
			p[n].val = np.copy(p.prefit[n].val)
		err = 1.0e3 * p.toaerrs
		Cdiag = (err)**2
		Cinv = np.diag(1.0/Cdiag)
		logCdet = np.sum(np.log(Cdiag))
		res = 1e9*np.array(p.residuals(updatebats=True,formresiduals=True),dtype=np.float128)
		logL+= -0.5 * np.dot(res,np.dot(Cinv,res))- 0.5 * logCdet - 0.5 * len(res) * np.log(2.0*np.pi)
	return logL

class Parameter(object):
    """
    Class holding the necessary functions and informations for the nested sampling algorithm over an arbitrary number of pulsars and binary pulsars systems
    """
    def __init__(self,pulsars,model='Free',noise="white"):
        self.model = model
        self.noisemodel=noise
        self.pulsar_names = [p.name for p in pulsars]
        self.pulsars = self.get_system_types(pulsars)
        self.values,self._internalvalues,self.vary = self.get_parameters_names()
        self.set_bounds()
        self.Npulsars = len(self.pulsar_names)
        self.logP = -np.inf
        self.logL = -np.inf
        if self.noisemodel==None:
            self.logLikelihood = self.logLikelihood_naive
        elif self.noisemodel=="white":
            self.logLikelihood = self.logLikelihood_whitenoise
            self.gp={}
        elif self.noisemodel=="red":
            self.logLikelihood = self.logLikelihood_rednoise
            self.gp={}
        else:
            sys.stderr.write("Noise model not implemented!")
            exit(-1)

    def get_system_types(self,pulsars):
        """
          checks for the pulsar names to identify potential binary pulsar systems and sets the
          pulsar dictionary accordingly
        """
        psrs=defaultdict(list)
        # first let's filter the binary pulsar systems
        for i in xrange(len(pulsars)):
            for j in xrange(i+1,len(pulsars)):
                if pulsars[i].name[:-1]==pulsars[j].name[:-1] and pulsars[i].name!=pulsars[j].name:
                    psrs['binaries'].append([pulsars[i],pulsars[j]])
        # let's now add the single pulsars
        for i in xrange(len(pulsars)):
            if len(psrs['binaries'])>0:
                for j in xrange(len(psrs['binaries'])):
                    if pulsars[i] not in psrs['binaries'][j]:
                        psrs['singles'].append(pulsars[i])
            else:
                psrs['singles'].append(pulsars[i])
        return psrs

    def get_parameters_names(self):
        """
        gets the parameters names from the input pulsars
        it checks for binary pulsar systems and makes sure that common parameters
        are not counted twice as independent ones
        """
        self.par_names = []
        self.par_types = []
        self.vary_types = []
        # first let's deal with binary systems by adding the relevant parameters
        for binaries in self.pulsars['binaries']:
            # loop only over the first component of the binary system to add its parameters and
            # check against the prefit values to identify common parameters
            # otherwise just add the parameters
            for n in binaries[0].pars:
                if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
                    self.par_names.append(n+"_"+binaries[0].name[:-1])
                    self.par_types.append(np.float128)
                    self.vary_types.append(np.int)
                else:
                    self.par_names.append(n+"_"+binaries[0].name)
                    self.par_types.append(np.float128)
                    self.vary_types.append(np.int)
                    self.par_names.append(n+"_"+binaries[1].name)
                    self.par_types.append(np.float128)
                    self.vary_types.append(np.int)
            if self.noisemodel=="white" or self.noisemodel=="red":
                self.par_names.append('logEQUAD_'+binaries[0].name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
                self.par_names.append('logEQUAD_'+binaries[1].name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
            if self.noisemodel=="red":
                # now let's add the noise parameters, 2 per pulsar
                self.par_names.append('logTAU_'+binaries[0].name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
                self.par_names.append('logTAU_'+binaries[1].name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
                self.par_names.append('logSIGMA_'+binaries[0].name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
                self.par_names.append('logSIGMA_'+binaries[1].name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)

        # now let's add the single pulsars parameters
        for singles in self.pulsars['singles']:
            for n in singles.pars:
                self.par_names.append(n+"_"+singles.name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
            if self.noisemodel=="white" or self.noisemodel=="red":
                self.par_names.append('logEQUAD_'+singles.name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
            if self.noisemodel=="red":
                self.par_names.append('logTAU_'+singles.name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
                self.par_names.append('logSIGMA_'+singles.name)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
                # now append the CG parameters
        if self.model=="CG":
            for n in ['GOB','EPS','XI','KAPPA']:
                self.par_names.append(n)
                self.par_types.append(np.float128)
                self.vary_types.append(np.int)
        # finally, we create the parameter values, internal values and variation information arrays
        return np.zeros(1,dtype={'names':self.par_names,'formats':self.par_types}),np.zeros(1,dtype={'names':self.par_names,'formats':self.par_types}),np.zeros(1,dtype={'names':self.par_names,'formats':self.vary_types})

    def set_bounds(self):
        """
        Sets the bounds on the varying parameters given a tempo2 fit and a timing model
        """
        self.bounds = {}
        # first let's deal with binary systems by adding the relevant parameters
        for binaries in self.pulsars['binaries']:
            # loop only over the first component of the binary system to add its parameters and
            # check against the prefit values to identify common parameters
            # otherwise just add the parameters
            if self.noisemodel=="white" or self.noisemodel=="red":
                # first set the noise parameters
                self.bounds['logEQUAD_'+binaries[0].name] = [-3.,10.0] # in ns
                self.bounds['logEQUAD_'+binaries[1].name] = [-3.,10.0] # in ns
                self.vary['logEQUAD_'+binaries[0].name] = 1
                self.vary['logEQUAD_'+binaries[1].name] = 1
            if self.noisemodel=="red":
                self.bounds['logTAU_'+binaries[0].name] = [-3.,10.0]
                self.bounds['logTAU_'+binaries[1].name] = [-3.,10.0]
                self.bounds['logSIGMA_'+binaries[0].name] = [-3.,10.0] # in ns
                self.bounds['logSIGMA_'+binaries[1].name] = [-3.,10.0] # in ns
                self.vary['logTAU_'+binaries[0].name] = 1
                self.vary['logTAU_'+binaries[1].name] = 1
                self.vary['logSIGMA_'+binaries[0].name] = 1
                self.vary['logSIGMA_'+binaries[1].name] = 1

            for n in binaries[0].pars:
                if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
                    name = n+"_"+binaries[0].name[:-1]
                    a = np.minimum(binaries[0][n].val-width*binaries[0][n].err,binaries[1][n].val-width*binaries[1][n].err)
                    b = np.maximum(binaries[0][n].val+width*binaries[0][n].err,binaries[1][n].val+width*binaries[1][n].err)
                    self.vary[name] = np.all([p[n].fit for p in binaries])
                    self.values[name] = binaries[0].prefit[n].val
                    # now let's impose additional constraints on some physical parameters that are either due to physical constraints
                    # or to the model of choice
                    if n=='SINI':
                        a = np.maximum(-1.0,a)
                        b = np.minimum(1.0,b)
                    if n=='DM':
                        a = np.maximum(1.0,a)
                        b = np.minimum(1000.0,b)
                    if n=='PX':
                        a = np.maximum(0.1,a)
                    if self.model=='GR' or self.model=='CG':
                        if n == 'OMDOT':
                            self.vary[name] = -1
                        if n == 'SINI':
                            self.vary[name] = -1
                        if self.model=='GR':
                            if n == 'PBDOT':
                                self.vary[name] = -1
                    self.bounds[name] = [a,b]
                else:
                    for p in binaries:
                        a = p[n].val-width*p[n].err
                        b = p[n].val+width*p[n].err
                        self.values[n+'_'+p.name] = binaries[0].prefit[n].val
                        if n=='M2':
                            a = np.maximum(0.1,a)
                            b = np.minimum(10.0,b)
                        if n=='GAMMA':
                            a = np.maximum(0.0,a)
                        self.bounds[n+'_'+p.name] = [a,b]
                        self.vary[n+'_'+p.name] = p[n].fit
                        if self.model=='GR' or self.model=='CG':
                            if n=='GAMMA':
                                self.vary[n+'_'+p.name] = -1
            if 'OM_'+binaries[0].name in self.par_names: self.vary['OM_'+binaries[1].name] = -1
            if self.model=='CG':
                self.bounds['GOB'] = [0.95,1.05]
                self.bounds['EPS'] = [2.95,3.05]
                self.bounds['XI'] = [0.95,1.05]
                self.bounds['KAPPA'] = [-0.05,0.05]
                self.vary['KAPPA'] = 1
                self.vary['GOB'] = 1
                self.vary['EPS'] = 1
                self.vary['XI'] = 1

        # now let's add the single pulsars parameters
        for singles in self.pulsars['singles']:
            if self.noisemodel=="white" or self.noisemodel=="red":
                self.bounds['logEQUAD_'+singles.name] = [-3.,10.0] # in log(ns)
                self.vary['logEQUAD_'+singles.name] = 1
            if self.noisemodel=="red":
                # first set the noise parameters
                self.bounds['logTAU_'+singles.name] = [-3.,10.0]
                self.bounds['logSIGMA_'+singles.name] = [-3.,10.0] # in log(ns)
                self.vary['logTAU_'+singles.name] = 1
                self.vary['logSIGMA_'+singles.name] = 1
            
            for n in singles.pars:
                a = singles[n].val-width*singles[n].err
                b = singles[n].val+width*singles[n].err
                if n=='SINI':
                    b = np.minimum(1.0,b)
                if n=='DM':
                    a = np.maximum(0.1,a)
                if n=='PX':
                    a = np.maximum(0.1,a)
                if n=='M2':
                    a = np.maximum(0.1,a)
                if n=='GAMMA':
                    a = np.maximum(0.0,a)
                self.bounds[n+'_'+singles.name] = [a,b]
                self.vary[n+'_'+singles.name] = singles[n].fit


    def inbounds(self):
        """
        Checks whether the values of the parameters are in bound
        """
        for n in self.values.dtype.names:
            if (self.vary[n]==1):
                if (self.values[n] < self.bounds[n][0] or self.values[n] > self.bounds[n][1]):
                    return False
        self.constraint()
        try:
            if self.values['SINI_PSR']>1.0: return False
        except:
            pass
        #    for n in self.values.dtype.names:
        #      if (self.vary[n]==-1):
        #        if (self.values[n] < self.bounds[n][0] or self.values[n] > self.bounds[n][1]):
        #          return False
        return True

    def map(self):
        """
        Maps [0,1] to the bounds of the parameters
        """
        for name in self.par_names:
            if (self.vary[name]==1):
        #        self.values[name] = self.bounds[name][0]+self._internalvalues[name]*(self.bounds[name][1]-self.bounds[name][0])
                self.values[name] = 0.5*(self._internalvalues[name]+1.0)*(self.bounds[name][1]-self.bounds[name][0])+self.bounds[name][0]

    def invmap(self):
        """
        Maps the bounds of the parameters onto [0,1] BROKEN!
        """
        for name in self.par_names:
            if (self.vary[name]==1):
                self._internalvalues[name] = (self.values[name]-self.bounds[name][0])/(self.bounds[name][1]-self.bounds[name][0])

    def constraint(self):
        """
        Imposes the relevant constraints for the model under consideration
        """
        for binaries in self.pulsars['binaries']:
            if 'OM_'+binaries[0].name in self.par_names:
                self.values['OM_'+binaries[1].name] = self.values['OM_'+binaries[0].name]+180.
            if self.model!="Free":
                m1 = self.values['M2_'+binaries[1].name]
                m2 = self.values['M2_'+binaries[0].name]
                pb = self.values['PB_'+binaries[0].name[:-1]]
                a1 = self.values['A1_'+binaries[0].name]
                a2 = self.values['A1_'+binaries[1].name]
                ecc = self.values['ECC_'+binaries[0].name[:-1]]
            if self.model=='GR':
                self.values['GAMMA_'+binaries[0].name] = gamma(pb,ecc,m1,m2)
                self.values['GAMMA_'+binaries[1].name] = gamma(pb,ecc,m2,m1)
                self.values['PBDOT_'+binaries[0].name[:-1]] = pbdot(pb,ecc,m1,m2)
                self.values['OMDOT_'+binaries[0].name[:-1]] = omega_dot(pb,ecc,m1,m2)
                self.values['SINI_'+binaries[0].name[:-1]] = shapiroS(pb,m1,m2,a1)
                #sys.stderr.write('pb: %.30f m1: %.30f m2: %.30f a1: %.30f sini: %.30f\n'%(pb,m1,m2,a1,self.values['SINI_'+binaries[0].name[:-1]]))
            elif self.model=='CG':
                gob = self.values['GOB']
                xi = self.values['XI']
                eps = self.values['EPS']
                kappa = self.values['KAPPA']
                mtot = m1+m2
                beta = beta0(pb,mtot,gob)
                self.values['GAMMA_'+binaries[0].name] = gammaAG(pb,m2/mtot,gob,kappa,beta,ecc)
                self.values['GAMMA_'+binaries[1].name] = gammaAG(pb,m1/mtot,gob,kappa,beta,ecc)
                self.values['OMDOT_'+binaries[0].name[:-1]] = omdotAG(pb,eps,xi,beta,ecc)
                self.values['SINI_'+binaries[0].name[:-1]] = shapiroSAG(pb,a1,m2/mtot,beta)

    def logPrior(self):
        """
        Prior function
        """
        self.map()
        if self.inbounds():
            self.logP = 0.0
            for n in self.par_names:
                if 'PX' in n:
                    self.logP+=-np.log(self.values[n])
                if 'DM' in n[:2] and 'DM1' not in n and 'DM2' not in n:
                    self.logP+=-np.log(self.values[n])
        #        if 'TAU' in n:
        #          self.logP+=-np.log(self.values[n])
        #        if 'SIGMA' in n:
        #          self.logP+=-np.log(self.values[n])
        else:
            self.logP = -np.inf
        return self.logP

    def initialise(self):
        """
        Initialise the internal values
        """
        for n in self.par_names:
            if (self.vary[n]==1):
                self._internalvalues[n] = np.random.uniform(-1.0,1.0)
        self.map()
        self.constraint()
#        # now initialise the Gaussian process for the red noise if necessary
#        if self.noisemodel=="white" or self.noisemodel=="red":
#            for binaries in self.pulsars['binaries']:
#                for p in binaries:
#                    equad = np.exp(self.values['logEQUAD_'+p.name])[0]
#                    kernel = kernels.WhiteKernel(equad)
#                    if self.noisemodel=="red":
#                        sigma = np.exp(self.values['logSIGMA_'+p.name])
#                        tau = np.exp(self.values['logTAU_'+p.name])
#                        kernel = kernel+sigma * sigma * kernels.ExpSquaredKernel(tau*tau)
#                    self.gp[p.name] = george.GP(kernel)
#                    err = 1.0e3 * p.toaerrs # in ns
#                    i = np.argsort(p.toas())
#                    self.gp[p.name].compute(p.toas()[i], err[i])
#            for singles in self.pulsars['singles']:
#                    equad = np.exp(self.values['logEQUAD_'+singles.name])[0]
#                    kernel = kernels.WhiteKernel(equad)
#                    if self.noisemodel=="red":
#                        tau = np.exp(self.values['logTAU_'+singles.name])
#                        sigma = np.exp(self.values['logSIGMA_'+singles.name])
#                        kernel = kernel+sigma * sigma * kernels.ExpSquaredKernel(tau*tau)
#                    self.gp[singles.name] = george.GP(kernel)
#                    err = 1.0e3 * singles.toaerrs # in ns
#                    i = np.argsort(singles.toas())		
#                    self.gp[singles.name].compute(singles.toas()[i], err[i])

    def logLikelihood_naive(self):
        """
        Likelihood function for naive measurement only uncertainty
        """
        # fill the pulsars
        self.logL=0.0
        for binaries in self.pulsars['binaries']:
            for n in binaries[0].pars:
                if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
                    name = n+"_"+binaries[0].name[:-1]
                    if self.vary[name]!=0:
                        binaries[0][n].val = np.copy(self.values[name])
                        binaries[1][n].val = np.copy(self.values[name])
                else:
                    for p in binaries:
                        name = n+"_"+p.name
                        if self.vary[name]!=0:
                            p[n].val = np.copy(self.values[name])
            for p in binaries:
                err = 1.0e3 * p.toaerrs
                Cdiag = (err)**2
                Cinv = np.diag(1.0/Cdiag)
                logCdet = np.sum(np.log(Cdiag))
                res =  1e9*np.array(p.residuals(updatebats=True,formresiduals=True),dtype=np.float128)
                self.logL+= -0.5 * np.dot(res,np.dot(Cinv,res))- 0.5 * logCdet - 0.5 * len(res) * np.log(2.0*np.pi)
                
        for singles in self.pulsars['singles']:
            for n in singles.pars:
                name = n+"_"+singles.name
                if self.vary[name]!=0:
                    singles[n].val = np.copy(self.values[name])
            err = 1.0e3 *  singles.toaerrs
            Cdiag = (err)**2
            Cinv = np.diag(1.0/Cdiag)
            logCdet = np.sum(np.log(Cdiag))
            res = 1e9*np.array(singles.residuals(updatebats=True,formresiduals=True),dtype=np.float128)
            self.logL+= -0.5 * np.dot(res,np.dot(Cinv,res))- 0.5 * logCdet - 0.5 * len(res) * np.log(2.0*np.pi)
        return self.logL

    def logLikelihood_whitenoise(self):
        """
        Likelihood function for white noise
        """
        # fill the pulsars
        self.logL=0.0
        for binaries in self.pulsars['binaries']:
            for n in binaries[0].pars:
                if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
                    name = n+"_"+binaries[0].name[:-1]
                    if self.vary[name]!=0:
                        binaries[0][n].val = np.copy(self.values[name])
                        binaries[1][n].val = np.copy(self.values[name])
                else:
                    for p in binaries:
                        name = n+"_"+p.name
                        if self.vary[name]!=0:
                            p[n].val = np.copy(self.values[name])
            for p in binaries:
                equad = np.exp(self.values['logEQUAD_'+p.name])[0]
                kernel = kernels.ExpSquaredKernel(tau*tau)
                gp = george.GP(kernel)
                i = np.argsort(p.toas())
                err = 1.0e3 * p.toaerrs # in ns
                gp.compute(p.toas()[i], err[i])
                res = 1e9*np.array(p.residuals(updatebats=True,formresiduals=True)[i],dtype=np.float128) # in ns
                try:
                    self.logL+=gp.lnlikelihood(res,quiet=True)
                except:
                    self.logL = -np.inf
                    return self.logL

        for singles in self.pulsars['singles']:
            for n in singles.pars:
                name = n+"_"+singles.name
                if self.vary[name]!=0:
                    singles[n].val = np.copy(self.values[name])
            equad = np.exp(self.values['logEQUAD_'+singles.name])[0]
            kernel = kernels.WhiteKernel(equad)
            gp = george.GP(kernel)
            err = 1.0e3 * singles.toaerrs # in ns
            i = np.argsort(singles.toas())
            gp.compute(singles.toas()[i], err[i])
            res = 1e9*np.array(singles.residuals(updatebats=True,formresiduals=True)[i],dtype=np.float128) # in ns
            try:
                self.logL+=gp.lnlikelihood(res,quiet=True)
            except:
                self.logL = -np.inf
                return self.logL
        return self.logL

    def logLikelihood_rednoise(self):
        """
        Likelihood function for red noise
        """
        # fill the pulsars
        self.logL=0.0
        for binaries in self.pulsars['binaries']:
            for n in binaries[0].pars:
                if binaries[0].prefit[n].val == binaries[1].prefit[n].val:
                    name = n+"_"+binaries[0].name[:-1]
                    if self.vary[name]!=0:
                        binaries[0][n].val = np.copy(self.values[name])
                        binaries[1][n].val = np.copy(self.values[name])
                else:
                    for p in binaries:
                        name = n+"_"+p.name
                        if self.vary[name]!=0:
                            p[n].val = np.copy(self.values[name])
            for p in binaries:
                tau = np.exp(self.values['logTAU_'+p.name])
                sigma = np.exp(self.values['logSIGMA_'+p.name])
                equad = np.exp(self.values['logEQUAD_'+p.name])[0]
                kernel = kernels.WhiteKernel(equad)+sigma * sigma * kernels.ExpSquaredKernel(tau*tau)
                gp = george.GP(kernel)
                i = np.argsort(p.toas())
                err = 1.0e3 * p.toaerrs # in ns
                gp.compute(p.toas()[i], err[i])
                res = 1e9*np.array(p.residuals(updatebats=True,formresiduals=True)[i],dtype=np.float128) # in ns
                try:
                    self.logL+=gp.lnlikelihood(res,quiet=True)
                except:
                    self.logL = -np.inf
                    return self.logL

        for singles in self.pulsars['singles']:
            for n in singles.pars:
                name = n+"_"+singles.name
                if self.vary[name]!=0:
                    singles[n].val = np.copy(self.values[name])
            tau = np.exp(self.values['logTAU_'+singles.name])
            sigma = np.exp(self.values['logSIGMA_'+singles.name])
            equad = np.exp(self.values['logEQUAD_'+singles.name])[0]
            kernel = kernels.WhiteKernel(equad)+sigma * sigma * kernels.ExpSquaredKernel(tau*tau)
            err = 1.0e3 * singles.toaerrs # in ns
            i = np.argsort(singles.toas())
            gp = george.GP(kernel)
            gp.compute(singles.toas()[i], err[i])
            res = 1e9*np.array(singles.residuals(updatebats=True,formresiduals=True)[i],dtype=np.float128) # in ns
            try:
                self.logL+=gp.lnlikelihood(res,quiet=True)
            except:
                self.logL = -np.inf
                return self.logL
        return self.logL

if __name__ == '__main__':
	import libstempo as T
	import matplotlib.pyplot as plt
	
	T.data = "/projects/pulsar_timing/nested_sampling/"
	parfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par","/projects/pulsar_timing/nested_sampling/pulsar_b.par"]
	timfiles =["/projects/pulsar_timing/nested_sampling/pulsar_a.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b.simulate","/projects/pulsar_timing/nested_sampling/pulsar_b_py.simulate"]
	
	psrA = tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
	psrB = tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
	psrA.fit()
	psrB.fit()
	
	N = 1
	param = [None]*N
	v = []
	for i in xrange(N):
		param[i] = Parameter([psrA,psrB],model='Free')
		#print param[i].par_names
		if len(param[i].pulsars['singles'])>0: print 'single',param[i].pulsars['singles'][0].name
		if len(param[i].pulsars['binaries'])>0: print 'binaries',param[i].pulsars['binaries'][0][0].name,param[i].pulsars['binaries'][0][1].name
		print param[i].values.dtype.names
		print param[i].bounds
		print param[i].vary
		param[i].initialise()
		#    param[i].set_bounds([psrA,psrB])
		#    param[i].initialise()
		print param[i].logLikelihood([psrA,psrB])
		print param[i].values
		print param[i]._internalvalues
	
