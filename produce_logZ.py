import optparse as op
import sys
import os
import multiprocessing as mp
import threading
from NestedSampling import *

def parse_to_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))



for nlive in [256,512,1024,2048,4096]:#256,
  for maxmcmc in [128,256,512]:
    cmd = "python NestedSampling.py -N %d -t Free -s 1 --verbose --maxmcmc %d --nthreads 1 --parameters pulsar_a.par,pulsar_b.par --times pulsar_a.simulate,pulsar_b.simulate -o double_pulsar/whitenoise/logZ_acc/%d_%d"%(nlive,maxmcmc,nlive,maxmcmc)
    os.system(cmd)
    cmd = "python merge_runs.py -N %d -o double_pulsar/whitenoise/logZ_acc/%d_%d/merged_chain_%d_%d.txt -p double_pulsar/whitenoise/logZ_acc/%d_%d/posterior_samples_%d_%d.txt -e double_pulsar/whitenoise/logZ_acc/%d_%d/header.txt double_pulsar/whitenoise/logZ_acc/%d_%d/chain_Free_%d_1.txt"%(nlive,nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc,nlive)
    os.system(cmd)
    cmd = "python PTPostProcess.py -i double_pulsar/whitenoise/logZ_acc/%d_%d/posterior_samples_%d_%d.txt -e double_pulsar/whitenoise/logZ_acc/%d_%d/merged_chain_%d_%d.txt_evidence -o double_pulsar/whitenoise/logZ_acc/%d_%d/posplots --parameters pulsar_a.par,pulsar_b.par --times pulsar_a.simulate,pulsar_b.simulate --noise-model white --2D 1 --DPGMM 1"%(nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc,nlive,maxmcmc)
    os.system(cmd)