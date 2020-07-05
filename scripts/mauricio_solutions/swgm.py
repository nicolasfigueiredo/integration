import util_m
import numpy as np

def SWGM(specs, beta=0.3):
	"""
	Gets a list of spectrograms (normalized, interpolated) and returns their SWGM combination
	"""
	P = len(specs)
	gammas = []
	X_exp_gamma = [] # X[p] ^ gamma[p]

	for i in range(len(specs)):
	    other_specs = util_m.get_other_specs(specs, i)
	    spec = specs[i]
	    gamma = (np.prod(other_specs ** (1/(P-1)), axis=0) / spec) ** beta
	    gammas.append(gamma)
	    X_exp_gamma.append(spec ** gamma)

	prod = np.prod(X_exp_gamma, axis=0)
	gamma_sum = 1/(np.sum(gammas, axis=0))
	SWGM_spec = prod ** gamma_sum
	norm_gain = np.sum(specs[0]) / np.sum(SWGM_spec)

	return SWGM_spec * norm_gain