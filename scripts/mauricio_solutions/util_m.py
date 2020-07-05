from cv2 import resize
import librosa
import numpy as np

def interpol_and_normalize(specs):
	""" Interpolates all spectrograms to the same dimensions and normalizes them"""
	return np.array(normalize_specs(fix_dimensions(specs))) # interpol. and normalize

def fix_dimensions(specs, interpolation='linear'):
	""" Interpolates all spectrograms so that they have the same dimensions of the larger one """
	new_shape = specs[-1].shape[::-1]
	new_specs = []
	for spec in specs[:-1]:
		new_specs.append(resize(spec, new_shape))
	new_specs.append(specs[-1])
	return new_specs

def get_spectrograms(y, windows=[512, 1024, 4096]):
	""" Gets 'n' energy spectrograms according to the list of FFT lengths given (fixed hop size)"""
	spec_list = []
	for window in windows:
	    spec = librosa.stft(y, n_fft=window, hop_length=512)
	    spec_list.append(np.abs(spec) ** 2)
	return spec_list

def get_other_specs(specs, i):
	""" Returns all items of specs that are not specs[i] """
	# (deve ter algum jeito mais inteligente de fazer isso...)
	other_specs = []
	for j in range(0, len(specs)):
	    if j != i:
	        other_specs.append(specs[j])
	return np.array(other_specs)

def normalize_specs(specs):
	""" Normalizes the energy of each spectrogram so that they have the same total sum of coefficients"""
	E = np.sum(specs[0])
	norm_specs = [specs[0]]
	for spec in specs[1:]:
	    norm_specs.append(spec * (E/np.sum(spec)))
	return norm_specs