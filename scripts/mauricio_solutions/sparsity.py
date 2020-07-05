import numpy as np

def calc_sparsity(spec, window, mode='lukin todd'):
	"""
    According to an analysis window and figure-of-merit, compute the LT sparsity for 
    each bin of the spectrogram 'spec'

    mode = {'lukin todd', 'local sparsity'}
	"""
	kernel_size = window.shape
	sparsity = np.zeros(spec.shape)
	guard_cols = int(np.floor(kernel_size[1] / 2)) # number of cols on each side of evaluated bin
	guard_rows = int(np.floor(kernel_size[0] / 2)) # number of rows on each side of evaluated bin

	for i in range(guard_rows, spec.shape[0] - guard_rows): # o que fazer com as bordas?
	    for j in range(guard_cols, spec.shape[1] - guard_cols):
	        subregion = spec[i-guard_rows:i+guard_rows+1, j-guard_cols:j+guard_cols+1] * window
	        if mode == 'lukin todd':
	        	sparsity[i,j] = LT_sparsity(subregion)
	        elif mode == 'local sparsity':
	        	sparsity[i,j] = gini(subregion)
	return sparsity

def get_sparsities(specs, window, mode='lukin todd'):
	"""
	Interpolate, normalize and get sparsities for a group of spectrograms
	"""
	sparsities = []
	for spec in specs:
	    sparsities.append(calc_sparsity(spec, window, mode=mode))
	return np.array(sparsities)


def gini(subregion):
	N = np.size(subregion)
	sorted = np.sort(subregion.flatten())
	norm_1 = np.sum(np.abs(subregion))
	j = np.array(list(range(1, 1+len(sorted))))
	return 1 - 2 * np.sum((sorted/norm_1) * (N - j + 0.5)/N)

def LT_sparsity(subregion):
	"""
	Measure of sparsity used by Lukin & Todd
	"""
	sorted = np.sort(subregion.flatten())[::-1]
	idx = list(range(1, len(sorted)+1))
	return (np.sum(sorted*idx)) / (np.sqrt(np.sum(sorted)) + 0.0000001)