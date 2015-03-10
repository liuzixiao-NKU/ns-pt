# -*- coding: utf-8 -*-

import numpy as np
from math import pow
from scipy.special import cbrt
import optparse as op


C = 299792458.0 # m/s
G = 6.67e-11 # SI m^3 kg^-1 s^-2
Msun = 1.98892e30 # kg
GM = 1.3271243999e20
SYEAR = 31558149.7676
T0 = GM/C**3.# 4.925490947e-6 # sec

def omega_dot(pb,ecc,mp,mc):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    omdot = 3.*pow(T0,2./3.)*pow(nb,5./3.)*pow(mp+mc,2./3.)/(1.0-ecc*ecc)
    return (omdot*SYEAR)*180./np.pi # return deg/yr

def gamma(pb,ecc,mp,mc):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    return pow(T0,2./3.)*pow(nb,-1./3.)*ecc*mc*(mp+2.*mc)/pow(mp+mc,4./3.)

def shapiroR(mc):
    return T0*mc

def shapiroS(pb,mp,mc,ap):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    return pow(T0,-1./3.)*pow(nb,2./3.)*ap*pow(mp+mc,2./3.)/mc

def pbdot(pb,ecc,mp,mc):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    pbdot = -(192.*np.pi/5.)*pow(T0,5./3.)*pow(nb,5./3.)*mp*mc/pow(mp+mc,1./3.)*((1.0+(73./24.)*ecc*ecc+(37./96.)*ecc*ecc*ecc*ecc)/pow(1.0-ecc*ecc,7./2.))
    return pbdot

def omegaSO(pb,ecc,mc,mp):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    omega = pow(T0,2./3.)*pow(nb,5./3.)*(mc*(4.*mp+3.*mc))/(2.*pow(mp+mc,4./3.))/(1.0-ecc*ecc)
    return (omega*SYEAR)*180./np.pi # return deg/yr

def read_param_names(file):
    paramnames=[]
    paraminj=[]
    paramerr=[]
    f=open(file,"r")
    for line in f:
        l = list(line.split(None))
        
        if int(l[-1])==1:
            paramnames.append(l[0])
            paraminj.append(np.float64(l[1]))
            paramerr.append(np.float64(l[2]))
    f.close()
    return paramnames,paraminj,paramerr

def omdotAG(pb,eps,xi,beta0,ecc):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    omdot = nb*(eps-0.5*xi+0.5)*beta0*beta0/(1.0-ecc*ecc)
    return (omdot*SYEAR)*180./np.pi # deg/yr

def gammaAG(pb,Xb,gob,kappa,beta0,ecc):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    return (ecc/nb)*Xb*(gob+kappa+Xb)*beta0*beta0

def beta0(pb,Mtot,GOB):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    return pow(GOB*Mtot*nb*GM,1./3.)/C

def shapiroSAG(pb,ap,Xb,beta0):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    return (nb*ap)/(beta0*Xb)

def omegaSOAG(pb,mp,mc,GammaAB,GOB,beta0,ecc):
    # convert pb in sec
    pb = pb*86400.
    nb = 2.0*np.pi/pb
    mtot = mp+mc
    omega = nb*(mp*mc/(mtot*mtot))*((1.+mp/mc)*GammaAB/(GOB*G)-0.5*mp/mc)*beta0*beta0/(1.0-ecc*ecc)
    return (omega*SYEAR)*180./np.pi # return deg/yr

def semimaxjoraxis(pb,mp,mc):
    # convert pb in sec
    pb = pb*86400.
    mu = mp*mc/(mp+mc)
    mtot = mp+mc
    a = cbrt(mtot*GM*pb*pb/(4.0*np.pi*np.pi))/C
    return a/(1.+mp/mc)

if __name__ == '__main__':
    paramnamesA,injectionsA,errorsA=read_param_names("/projects/pulsar_timing/nested_sampling/prefit_PSRA.par")
    paramnamesB,injectionsB,errorsB=read_param_names("/projects/pulsar_timing/nested_sampling/prefit_PSRB.par")
    si = 0.997
    m1 = injectionsB[paramnamesB.index("M2")]
    m2 = injectionsA[paramnamesA.index("M2")]
    mtot = m1+m2
    pb = injectionsA[paramnamesA.index("PB")]
    GOB = 1.005
    GEFF = GOB*G
    beta = beta0(pb,mtot,GOB)
    ecc = injectionsA[paramnamesA.index("ECC")]
    omdotA = omega_dot(pb,ecc,m1,m2)
    omdotB = omega_dot(pb,ecc,m2,m1)
    gammaA = gamma(pb,ecc,m1,m2)
    gammaB = gamma(pb,ecc,m2,m1)
    pbdotA = pbdot(pb,ecc,m1,m2)
    pbdotB = pbdot(pb,ecc,m2,m1)
    aA = semimaxjoraxis(pb,m1,m2)*si
    aB = semimaxjoraxis(pb,m2,m1)*si
    sA = shapiroS(pb,m1,m2,aA)
    sB = shapiroS(pb,m2,m1,aB)
    print "parameter    computed    injected"
    print "m A:",m1
    print "m B:",m2
    print "a A:",aA,injectionsA[paramnamesA.index("A1")]
    print "a B:",aB,injectionsB[paramnamesB.index("A1")]
    print "omdot A:",omdotA,injectionsA[paramnamesA.index("OMDOT")]
    print "omdot B:",omdotB,injectionsB[paramnamesB.index("OMDOT")]
    print "pbdot A:",pbdotA,injectionsA[paramnamesA.index("PBDOT")]
    print "pbdot B:",pbdotB,injectionsB[paramnamesB.index("PBDOT")]
    print "gamma A:",gammaA,injectionsA[paramnamesA.index("GAMMA")]
    print "gamma B:",gammaB,injectionsB[paramnamesB.index("GAMMA")]
    print "s A:",sA,injectionsA[paramnamesA.index("SINI")]
    print "s B:",sB,injectionsB[paramnamesB.index("SINI")]
    print "Omega SO A:",omegaSO(pb,ecc,m1,m2)
    print "Omega SO B:",omegaSO(pb,ecc,m2,m1)
    print "AG:"
    GammaAB = 2.*GEFF
    si = shapiroSAG(pb,aA,m2/mtot,beta)
    aA = semimaxjoraxis(pb,m1,m2)*si
    aB = semimaxjoraxis(pb,m2,m1)*si
    print "a A:",aA,injectionsA[paramnamesA.index("A1")]
    print "a B:",aB,injectionsB[paramnamesB.index("A1")]
    print "beta:",beta
    print "shapiro s(A):",shapiroSAG(pb,aA,m2/mtot,beta)
    print "shapiro s(B):",shapiroSAG(pb,aB,m1/mtot,beta)
    print "omega:",omdotAG(pb,3.0,1.0,beta,ecc)
    print "gamma A:",gammaAG(pb,m2/mtot,1.,0.0,beta,ecc)
    print "gamma B:",gammaAG(pb,m1/mtot,1.,0.0,beta,ecc)
    print "Omega SO A:",omegaSOAG(pb,m1,m2,GammaAB,GOB,beta,ecc)
    print "Omega SO B:",omegaSOAG(pb,m2,m1,GammaAB,GOB,beta,ecc)