# -*- coding: utf-8 -*-

import numpy as np
from math import pow
from scipy.special import cbrt

C = 299792458.0 # m/s
G = 6.67e-11 # SI m^3 kg^-1 s^-2
Msun = 1.98892e30 # kg
GM = 1.3271243999e20
SYEAR = 31558149.7676
T0 = GM/C**3.# 4.925490947e-6 # sec

def ms2mcq(m1,m2):
    q = m2/m1
    mc = (m1+m2)*pow(m1*m2/(m1+m2)**2,3./5.)
    return mc,q

def q2ms(mc,q):
    factor = mc * np.power(1+q, 1.0/5.0);
    m1 = factor * np.power(q, -3.0/5.0);
    m2 = factor * np.power(q, 2.0/5.0);
    return (m1,m2)
#    if np.random.uniform(0.0,1.0) <= 0.5:
#        return (m1,m2)
#    else:
#        return (m2,m1)

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
    a = pow(mtot*GM*pb*pb/(4.0*np.pi*np.pi),1./3.)/C
    return a/(1.+mp/mc)

def write_to_file(filename,psr,fit):
    file = open(filename,"w")
    file.write('PSRJ\t%s\n'%(psr['NAME']))
    file.write('RAJ\t%s\t%d\n'%(psr['RAJ'],fit))
    file.write('DECJ\t%s\t%d\n'%(psr['DECJ'],fit))
    file.write('F0\t%.15f\t%d\n'%(psr['F0'],fit))
    file.write('F1\t%.15e\t%d\n'%(psr['F1'],fit))
    file.write('DM\t%.15f\t%d\n'%(psr['DM'],fit))
    file.write('DMEPOCH\t53156\n')
    file.write('PEPOCH\t53156\n')
    file.write('POSEPOCH\t53156\n')
    file.write('PMRA\t%.15f\t%d\n'%(psr['PMRA'],fit))
    file.write('PMDEC\t%.15f\t%d\n'%(psr['PMDEC'],fit))
    file.write('PX\t%.15f\t%d\n'%(psr['PX'],fit))
    file.write('SINI\t%.15f\t%d\n'%(psr['SINI'],fit))
    file.write('PB\t%.15f\t%d\n'%(psr['PB'],fit))
    file.write('PBDOT\t%.15e\t%d\n'%(psr['PBDOT'],fit))
    file.write('T0\t%.15f\t%d\n'%(psr['T0'],fit))
    file.write('A1\t%.15f\t%d\n'%(psr['A1'],fit))
    file.write('OM\t%.15f\t%d\n'%(psr['OM'],fit))
    file.write('ECC\t%.15f\t%d\n'%(psr['ECC'],fit))
    file.write('OMDOT\t%.15f\t%d\n'%(psr['OMDOT'],fit))
    file.write('GAMMA\t%.15f\t%d\n'%(psr['GAMMA'],fit))
    file.write('M2\t%.15f\t%d\n'%(psr['M2'],fit))
    file.write('BINARY\tDD\n')
    file.write('START\t53000\n')
    file.write('FINISH\t54000\n')
    file.write('TZRMJD\t53000\n')
    file.write('TZRFRQ\t1440\n')
    file.write('TZRSITE\t7\n')
    file.write('CLK\tTT(BIPM2011)\n')
    file.write('EPHEM\tDE405\n')
    file.write('EPHVER\t5\n')
    file.close()
    return

if __name__ == '__main__':
    import optparse as op
    import libstempo as T
    pA = T.tempopulsar(parfile = "/projects/pulsar_timing/ns-pt/results/double_pulsar_2/pulsar_a.par", timfile = "pulsar_a_zero_noise.simulate")
    pB = T.tempopulsar(parfile = "/projects/pulsar_timing/ns-pt/results/double_pulsar_2/pulsar_b.par", timfile = "pulsar_b_zero_noise.simulate")
    parser = op.OptionParser()
    parser.add_option("--gob", type="float", dest="gob", help="vslue of the effective gravitational constant to assume", default=1.0)
    (options, args) = parser.parse_args()
    
    
    np.random.seed(259)
    pulsar_a,pulsar_b = {}, {}
    pulsar_a['NAME'] = 'FAKEPSRA'
    pulsar_b['NAME'] = 'FAKEPSRB'
    for p1,p2 in zip(pA.pars,pB.pars):
        pulsar_a[p1] = 0.0
        pulsar_b[p2] = 0.0
    
    # fill in the masses
    m1 = np.random.uniform(1.0,1.5)
    m2 = np.random.uniform(1.0,1.5)
    pulsar_a['M2'] = m2 # this is M2
    pulsar_b['M2'] = m1 # this is M1
    # the inclination angle
    inc = np.random.uniform(0.0,np.pi)
    pulsar_a['SINI'] = np.sin(inc)
    pulsar_b['SINI'] = np.sin(inc)
    # the eccenticity
    ecc = np.random.uniform(0.0,1.0)
    pulsar_a['ECC'] = ecc
    pulsar_b['ECC'] = ecc
    # the orbital period
    pb = np.random.uniform(0.0,1.0)
    pulsar_a['PB'] = pb
    pulsar_b['PB'] = pb
    # now fill the GR derived parameters
    pulsar_a['GAMMA'] = gamma(pb,ecc,m1,m2)
    pulsar_b['GAMMA'] = gamma(pb,ecc,m2,m1)
    pulsar_a['A1'] = semimaxjoraxis(pb,m1,m2)*np.sin(inc)
    pulsar_b['A1'] = semimaxjoraxis(pb,m2,m1)*np.sin(inc)
    pulsar_a['PBDOT'] = pbdot(pb,ecc,m1,m2)
    pulsar_b['PBDOT'] = pbdot(pb,ecc,m2,m1)
    pulsar_a['OMDOT'] = omega_dot(pb,ecc,m1,m2)
    pulsar_b['OMDOT'] = omega_dot(pb,ecc,m2,m1)
    pulsar_a['OM'] = np.random.uniform(0.0,180.0)
    pulsar_b['OM'] = pulsar_a['OM']+180.0

    # now fill the "extrinsic" parameters
    ra = str(np.random.randint(0,24))+":"+str(np.random.randint(0,60))+":"+str(np.random.uniform(0.0,60.0))
    pulsar_a['RAJ'] = ra
    pulsar_b['RAJ'] = ra
    dec = str(np.random.randint(-90,90))+":"+str(np.random.randint(0,60))+":"+str(np.random.uniform(0.0,60.0))
    pulsar_a['DECJ'] = dec
    pulsar_b['DECJ'] = dec
    pmra = np.random.uniform(-100.0,100.0)
    pulsar_a['PMRA'] = pmra
    pulsar_b['PMRA'] = pmra
    pmdec = np.random.uniform(-100.0,100.0)
    pulsar_a['PMDEC'] = pmdec
    pulsar_b['PMDEC'] = pmdec
    px = np.exp(np.random.uniform(np.log(0.1),np.log(100.0)))
    pulsar_a['PX'] = px
    pulsar_b['PX'] = px
    dm = np.exp(np.random.uniform(np.log(1.0),np.log(1000.0)))
    pulsar_a['DM'] = dm
    pulsar_b['DM'] = dm
    T0 = np.random.uniform(53000,54000)
    pulsar_a['T0'] = T0
    pulsar_b['T0'] = T0
    # finally the frequency and its derivative
    f0a = np.random.uniform(0.0,100.0)
    f0b = np.random.uniform(0.0,100.0)
    pulsar_a['F0'] = f0a
    pulsar_b['F0'] = f0b
    f1a = 1e-15*np.random.uniform(0.0,100.0)
    f1b = 1e-15*np.random.uniform(0.0,100.0)
    pulsar_a['F1'] = f1a
    pulsar_b['F1'] = f1b
    for p1,p2 in zip(pA.pars,pB.pars):
        print p1,pulsar_a[p1]
        print p2,pulsar_b[p2]
    # now save to a par file each of the pulsars:
    write_to_file("results/double_pulsar_2/pulsar_a_zero_noise.par",pulsar_a,0)
    write_to_file("results/double_pulsar_2/pulsar_b_zero_noise.par",pulsar_b,0)
    write_to_file("results/double_pulsar_2/pulsar_a.par",pulsar_a,1)
    write_to_file("results/double_pulsar_2/pulsar_b.par",pulsar_b,1)
    """
    si = 0.96
    m1 = pB.prefit["M2"].val
    m2 = pA.prefit["M2"].val
    mtot = m1+m2
    pb =  pA.prefit["PB"].val
    GOB = 1.00
    GEFF = GOB*G
    beta = beta0(pb,mtot,GOB)
    ecc = pA.prefit["ECC"].val
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
    print "a A:",aA,pA.prefit["A1"].val
    print "a B:",aB,pB.prefit["A1"].val
    print "omdot A:",omdotA,pA.prefit["OMDOT"].val
    print "omdot B:",omdotB,pB.prefit["OMDOT"].val
    print "pbdot A:",pbdotA,pA.prefit["PBDOT"].val
    print "pbdot B:",pbdotB,pB.prefit["PBDOT"].val
    print "gamma A:",gammaA,pA.prefit["GAMMA"].val
    print "gamma B:",gammaB,pB.prefit["GAMMA"].val
    print "s A:",sA,pA.prefit["SINI"].val
    print "s B:",sB,pB.prefit["SINI"].val
    print "Omega SO A:",omegaSO(pb,ecc,m1,m2)
    print "Omega SO B:",omegaSO(pb,ecc,m2,m1)
    print "AG:"
    GammaAB = 2.*GEFF
    si = shapiroSAG(pb,aA,m2/mtot,beta)
    aA = semimaxjoraxis(pb,m1,m2)*si
    aB = semimaxjoraxis(pb,m2,m1)*si
    print "a A:",aA,pA.prefit["A1"].val
    print "a B:",aB,pB.prefit["A1"].val
    print "beta:",beta
    print "shapiro s(A):",shapiroSAG(pb,aA,m2/mtot,beta)
    print "shapiro s(B):",shapiroSAG(pb,aB,m1/mtot,beta)
    print "omega:",omdotAG(pb,3.0,1.0,beta,ecc)
    print "gamma A:",gammaAG(pb,m2/mtot,1.,0.0,beta,ecc)
    print "gamma B:",gammaAG(pb,m1/mtot,1.,0.0,beta,ecc)
    print "Omega SO A:",omegaSOAG(pb,m1,m2,GammaAB,GOB,beta,ecc)
    print "Omega SO B:",omegaSOAG(pb,m2,m1,GammaAB,GOB,beta,ecc)
    """
