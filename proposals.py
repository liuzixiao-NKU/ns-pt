import numpy as np
from Parameters_v2 import *
from scipy.stats import multivariate_normal
import random


def _setup_kwargs(LivePoints,Nlive,dimension):
    kwargs = {}
    kwargs['Nlive']=Nlive
    kwargs['LivePoints']=LivePoints
    kwargs['dimension']=dimension
    return kwargs

def _update_kwargs(**kwargs):
    """
    update the live points covariance matrix, eigen values and eigen vectors
    """
    Nlive = kwargs['Nlive']
    dimension = kwargs['dimension']
    LivePoints = kwargs['LivePoints']
    cov_array = np.zeros((dimension,Nlive))
    for j,p in enumerate(LivePoints):
        i = 0
        for n in p.par_names:
            if p.vary[n]==1:
                cov_array[i,j] = p._internalvalues[n]
                i+=1
    covariance = np.cov(cov_array)
    #        a = fitDPGMM([self.dimension,self.covariance],np.transpose(cov_array))
    #        if a!=None:
    #          self.gmm = GMM(self.dimension,a)
    #        else:
    #          self.gmm = GMM([self.dimension,self.covariance])
    ev,evec = np.linalg.eigh(covariance)
    eigen_values,eigen_vectors = ev.real,evec.real
    mvn_object = multivariate_normal(np.zeros(dimension),covariance.T)
    kwargs['eigen_values']=eigen_values
    kwargs['eigen_vectors']=eigen_vectors
    kwargs['mvn']=mvn_object
    return kwargs

def _MultivariateNormal(inParam,**kwargs):
    """
    multivariate normal proposal function
    will be replaced by a gaussian mixture model proposal once it is working
    """
    mvn_object = kwargs['mvn']
    variates = mvn_object.rvs()#self.gmm.sample()#
    for n in inParam.par_names:
        k = 0
        if inParam.vary[n]==1:
            inParam._internalvalues[n]+=variates[k]
            k+=1
    log_acceptance_probability = np.log(np.random.uniform(0.,1.))
    return inParam,log_acceptance_probability

def _EigenDirection(inParam,**kwargs):
    """
    normal jump along the live points covariance matrix eigen directions
    """
    eigen_values = kwargs['eigen_values']
    eigen_vectors = kwargs['eigen_vectors']
    dimension = kwargs['dimension']
    i = np.random.randint(dimension)
    k = 0
    for n in inParam.par_names:
      if inParam.vary[n]==1:
        jumpsize = np.sqrt(np.abs(eigen_values[i]))*np.random.normal(0,1)
        inParam._internalvalues[n]+=jumpsize*eigen_vectors[k,i]
        k+=1
    log_acceptance_probability = np.log(np.random.uniform(0.,1.))
    return inParam,log_acceptance_probability

def _EnsembleWalk(inParam,**kwargs):
    """
    ensemble sampler walk move
    """
    Nsubset = 3
    Nlive = kwargs['Nlive']
    LivePoints = kwargs['LivePoints']
    indeces = random.sample(range(Nlive),Nsubset)

    subset = [LivePoints[i] for i in indeces]
    cm = np.zeros(1,dtype={'names':inParam.par_names,'formats':inParam.par_types})
    for n in inParam.par_names:
        cm[n] = np.sum([p._internalvalues[n] for p in subset])/Nsubset
    w = np.zeros(1,dtype={'names':inParam.par_names,'formats':inParam.par_types})
    for n in inParam.par_names:
        w[n] = np.sum([np.random.normal(0,1)*(p._internalvalues[n]-cm[n]) for p in subset])
    for n in inParam.par_names:
        if inParam.vary[n]==1: inParam._internalvalues[n]+=w[n]
    log_acceptance_probability = np.log(np.random.uniform(0.,1.))
    return inParam,log_acceptance_probability

def _EnsembleStretch(inParam,**kwargs):
    """
    ensemble sampler stretch move
    """
    Nlive = kwargs['Nlive']
    LivePoints = kwargs['LivePoints']
    dimension = kwargs['dimension']
    scale = 2.0
    a = np.random.randint(Nlive)
    
    wa = np.copy(LivePoints[a]._internalvalues[:])
    u = np.random.uniform(0.0,1.0)
    x = 2.0*u*np.log(scale)-np.log(scale)
    Z = np.exp(x)
    for n in inParam.par_names:
        if inParam.vary[n]==1: inParam._internalvalues[n] = wa[n]+Z*(inParam._internalvalues[n]-wa[n])
    if (Z<1.0/scale)or(Z>scale): log_acceptance_probability = -np.inf
    else: log_acceptance_probability = np.log(np.random.uniform(0.,1.))-(dimension)*np.log(Z)
    return inParam,log_acceptance_probability


proposals = {}
proposal_list_name = ['MultiVariateNormal','EigenDirections','EnsembleWalk','EnsembleStretch']
proposal_list = [_MultivariateNormal,_EigenDirection,_EnsembleWalk,_EnsembleStretch]

for name,algorithm in zip(proposal_list_name,proposal_list):
    proposals[name]=algorithm

class Proposal(object):
    """
    Proposal distribution class. 
    It takes a Parameter object and returns an updated 
    one
    """
    def __init__(self,name):
        """
        name: name of the proposal
        """
        self.name = name
        self.algorithm = proposals[name]

    def get_sample(self,inParam,**kwargs):
        return self.algorithm(inParam,**kwargs)

def setup_proposals_cycle():
    """
    initialise the proposals cycle
    """
    jump_proposals = []
    for i in xrange(100):
        jump_select = np.random.uniform(0.,1.)
        if jump_select<0.2: jump_proposals.append(Proposal(proposal_list_name[0]))
        elif jump_select<0.5: jump_proposals.append(Proposal(proposal_list_name[1]))
        elif jump_select<0.8: jump_proposals.append(Proposal(proposal_list_name[2]))
        else: jump_proposals.append(Proposal(proposal_list_name[3]))
    return jump_proposals

if __name__=="__main__":
    import libstempo as T
    from libstempo import tempopulsar
    kwargs = {}
    kwargs['mvn'] = multivariate_normal(0.0,1.0)
    p = Proposal('MultiVariateNormal')
    T.data = "/projects/pulsar_timing/nested_sampling/"
    parfiles =["/projects/pulsar_timing/ns-pt/pulsar_a.par","/projects/pulsar_timing/ns-pt/pulsar_b.par","/projects/pulsar_timing/ns-pt/pulsar_b.par"]
    timfiles =["/projects/pulsar_timing/ns-pt/pulsar_a_zero_noise.simulate","/projects/pulsar_timing/ns-pt/pulsar_b.simulate","/projects/pulsar_timing/ns-pt/pulsar_b_zero_noise.simulate"]

    psrA = tempopulsar(parfile = parfiles[0], timfile = timfiles[0])
    psrB = tempopulsar(parfile = parfiles[1], timfile = timfiles[1])
    psrA.fit()
    psrB.fit()
    x = Parameter([psrA,psrB],model='Free',noise=None)
    proposals_cycle = setup_proposals_cycle()
    for i,p in enumerate(proposals_cycle):
        print i,p.name,p.algorithm