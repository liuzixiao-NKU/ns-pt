import optparse as op
import sys
import os
import numpy as np
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import matplotlib.pyplot as plt

T.data = '/projects/pulsar_timing/nested_sampling/python/PTparallel'

def parse_to_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

if __name__=="__main__":
	parser = op.OptionParser()
	parser.add_option("-s", "--seed", type="int", dest="seed", help="seed for the run",default=0)
	(options, args) = parser.parse_args()
	seed = options.seed
	psr1 = LT.fakepulsar(parfile=T.data+'pulsar_a.par',
					obstimes=np.arange(53000,54000,30),
					toaerr=20)
	psr2 = LT.fakepulsar(parfile=T.data+'pulsar_b.par',
					obstimes=np.arange(53000,54000,30),
					toaerr=20)
	LT.add_efac(psr1,efac=1.0,seed=seed)
	LT.add_efac(psr2,efac=1.0,seed=seed)
	psr1.savetim('pulsar_a_%d.simulate'%seed)
	psr2.savetim('pulsar_b_%d.simulate'%seed)
	T.purgetim('pulsar_a_%d.simulate'%seed)
	T.purgetim('pulsar_b_%d.simulate'%seed)
	cmd = "python NestedSampling.py -N 2048 -t Free -s 1 --verbose --maxmcmc 1024 --nthreads 1 --parameters pulsar_a.par,pulsar_b.par --times pulsar_a_%d.simulate,pulsar_b_%d.simulate -o /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d"%(seed,seed,seed)
	os.system(cmd)
	cmd = "python merge_runs.py -N 2048 -o /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/merged_chain.txt -p /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/posterior_samples.txt -e /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/header.txt /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/chain_Free_1.txt"%(seed,seed,seed,seed)
	os.system(cmd)
	cmd = "python PTPostProcess.py -i /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/posterior_samples.txt -e /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/merged_chain.txt_evidence -o /projects/pulsar_timing/nested_sampling/python/PTparallel/zeronoise/%d/posplots --parameters pulsar_a.par,pulsar_b.par --times pulsar_a_%d.simulate,pulsar_b_%d.simulate --noise-model white --2D 1 --DPGMM 1"%(seed,seed,seed,seed,seed)
	os.system(cmd)
	
