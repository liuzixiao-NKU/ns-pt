# coding: utf-8
#! /usr/bin/env python

import numpy
import scipy
import scipy.stats.kde
import matplotlib
import sys
import math
from pylab import *
import string
import random
import os
import optparse as op
from scipy import integrate
from random import shuffle
import optparse as op
import cPickle as pickle
import numpy as np
import sys
from ComputePKparameters import *
from scipy.misc import logsumexp
"""
we approximated p(d|x1,x2,...,xn) with a DPGMM(x)
each xi = (yi-ai)/bi we need to tranform the distribution into
p(d|x1,x2,...,xn) = p(d|y1,y2,...,yn)|J(x->y)|
and |J(x->y)| = \prod dyi/dxi = \prod bi

"""

width = 5.0
def lnPLUS(x,y):
    max1 = np.maximum(x,y)
    min1 = np.minimum(x,y)
    return max1 + np.log1p(np.exp(min1 - max1))

def transform_coordinates(x,means,scale):
    xout = []
    for xi,mi,si in zip(x,means,scale):
        xout.append((xi-mi)/si)
    return array(xout)

def inverse_transform_coordinates(x,means,scale):
    xout = []
    for xi,mi,si in zip(x,means,scale):
        xout.append(xi*si+mi)
    return array(xout)

def logLikelihoodTest(ns,pdfs,x):
    logL=0.0
    for j,pdf in enumerate(pdfs):
        # each likelihood contains ["M2","PB","A1","PBDOT","OMDOT","GAMMA","SINI","ECC"]
        if j==0:
            mass = x.values[x.names.index("M2")]
        else:
            mass = x.values[x.names.index("M1")]
        logL_event = -np.inf
        y = transform_coordinates([mass,x.values[x.names.index("PB")]],ns.means[j],ns.scales[j])
        for ind,prob in enumerate(pdf[1]):
            logL_event=lnPLUS(logL_event,np.log(pdf[0][ind])+prob.logProb(y))
        logL+=logL_event#+sum([log(s) for s in ns.scales[j]])
    if isnan(logL): return -np.inf
    return logL

def constraint_gr(x,m1,m2,pb,ecc,a,psr):
    x.values[x.names.index("GAMMA"+str(psr))] = gamma(pb,ecc,m1,m2)
    x.values[x.names.index("PBDOT")] = pbdot(pb,ecc,m1,m2)
    x.values[x.names.index("OMDOT")] = omega_dot(pb,ecc,m1,m2)
    x.values[x.names.index("SINI")] = shapiroS(pb,m1,m2,a)
    return

def constraint_ag(x,m1,m2,pb,ecc,a,gob,kappa,eps,xi,psr):
    mtot = m1+m2
    beta = beta0(pb,mtot,gob)
    x.values[x.names.index("GAMMA%d"%psr)] = gammaAG(pb,m2/mtot,gob,kappa,beta,ecc)
    x.values[x.names.index("OMDOT")] = omdotAG(pb,eps,xi,beta,ecc)
    x.values[x.names.index("SINI")] = shapiroSAG(pb,a,m2/mtot,beta)
    return

def logLikelihood(x,ns,pdfs):
    logL = 0
    pb = x.values[x.names.index("PB")]
    ecc = x.values[x.names.index("ECC")]
    for j,pdf in enumerate(pdfs):
        # each likelihood contains ["M2","PB","A1","PBDOT","OMDOT","GAMMA","SINI","ECC"]
        if j==0:
            mp = x.values[x.names.index("M1")]
            mc = x.values[x.names.index("M2")]
        else:
            mc = x.values[x.names.index("M1")]
            mp = x.values[x.names.index("M2")]
        ap = x.values[x.names.index("A%d"%(j+1))]
        if ns.constraint !=None:
            if "GOB" in x.names:
                gob = x.values[x.names.index("GOB")]
                kappa = x.values[x.names.index("KAPPA")]
                eps = x.values[x.names.index("EPS")]
                xi = x.values[x.names.index("XI")]
                ns.constraint(x,mp,mc,pb,ecc,ap,gob,kappa,eps,xi,j+1)
            else:
                ns.constraint(x,mp,mc,pb,ecc,ap,j+1)
        if x.values[x.names.index("SINI")]>1.0:
            x.logL = -np.inf
            return -np.inf
        y = transform_coordinates([mc,pb,ap,x.values[x.names.index("PBDOT")],x.values[x.names.index("OMDOT")],x.values[x.names.index("GAMMA%d"%(j+1))],x.values[x.names.index("SINI")],ecc],ns.means[j],ns.scales[j])
#        if np.min(y)<-width or np.max(y)>width:
#            return -np.inf
        components = zeros(len(pdf[1]))
        weights = zeros(len(pdf[1]))
        for ind,prob in enumerate(pdf[1]):
            components[ind] = prob.logProb(y)
            weights[ind] = pdf[0][ind]
        logL+=logsumexp(components,b=weights)+sum([log(s) for s in ns.scales[j]])
    if isnan(logL):
        x.logL = -np.inf
        return -np.inf
    x.logL = logL
    return x.logL

def logprior(x):
    if x.inbounds():
        x.logP = 0.0
    else:
        x.logP =  -np.inf
    return x.logP