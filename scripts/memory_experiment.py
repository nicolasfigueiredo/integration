import librosa
import numpy as np
import scipy.signal
import math
import pandas as pd

import sys
sys.path.insert(0, '../scripts')

import stft_zoom, display, detect_musical_regions
from util import *
import mappings
import pickle
import PIL
import IPython.display
from classes import SingleResSpectrogram, MultiResSpectrogram
import glob
from aug_density_map import *
from mappings import *

import csv
import mido
from cost_experiment_lib import *

#do mauricio
sys.path.insert(0, '../scripts/mauricio_solutions/')
import lukin_todd, swgm, local_sparsity, util_m

def get_size(obj, seen=None):
	"""Recursively finds size of objects"""
	size = sys.getsizeof(obj)
	if seen is None:
		seen = set()
	obj_id = id(obj)
	if obj_id in seen:
		return 0
	# Important mark as seen *before* entering recursion to gracefully handle
	# self-referential objects
	seen.add(obj_id)
	if isinstance(obj, dict):
		size += sum([get_size(v, seen) for v in obj.values()])
		size += sum([get_size(k, seen) for k in obj.keys()])
	elif hasattr(obj, '__dict__'):
		size += get_size(obj.__dict__, seen)
	elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
		size += sum([get_size(i, seen) for i in obj])
	return size

def to_list(list_of_strings):
	return list(map(float, list_of_strings.strip('][').split(', ')))

def treat_pcts(pcts):
	for col in ['pct_multilevel_2', 'pct_multilevel_3', 'pct_multilevel_4']:
		pcts[col] = pcts[col].apply(to_list)
	return pcts

def get_pcts(pcts, file):
	pcts_file = (pcts[pcts['file name'] == file]).iloc[0]
	return(pcts_file['pct_200'], pcts_file['pct_500'], pcts_file['pct_800'], pcts_file['pct_1600'],
			pcts_file['pct_multilevel_2'], pcts_file['pct_multilevel_3'], pcts_file['pct_multilevel_4'])

def main(num_file):
	# result_file = '../notebooks/cost-experiment/results/memory-results/results_130820.csv'
	# with open(result_file, 'w') as csvfile:
	#     writer = csv.writer(csvfile, delimiter=';')
	#     writer.writerow(['file name', 'representation', 'max res', 'memory', 'pct refine'])
	# print("csv overwritten")

	# pcts = pd.read_csv('../notebooks/cost-experiment/results/pcts.csv', sep=';', dtype={'pct_200':float, 'pct_500':float, 'pct_800':float, 'pct_1600':float})
	# pcts = treat_pcts(pcts)

	# file_names = []
	# for year in ['2006','2008','2009','2011','2013','2014','2015']:
	#     path = '/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/' + year + '/*.wav'
	#     for file in glob.glob(path):
	#         file_names.append(file[:-4])

	file_names = []
	for year in ['2004']:
		path = '/Users/nicolas/Documents/ime/compmus/mestrado/maestro-dataset/' + year + '/*.wav'
		for file in glob.glob(path):
			file_names.append(file[:-4])

	print("files fetched: ",len(file_names))

	res_hz = [86.13, 43.06, 21.53, 10.77, 5.38, 2.69, 1.34]
	res_window = [512, 1024, 2048, 4096, 8192, 16384, 32768]

	res_lists_2lvl = [[1,1], [2,2], [2,4], [4,8], [8, 16], [16, 32], [32, 64]]
	res_lists_3lvl = [[1,1,1], [2,2,2], [2,3,4], [3, 6, 8], [5, 10, 16], [10, 21, 32], [21, 42, 64]]
	res_lists_4lvl = [[1,1,1,1], [2,2,2,2], [1,2,3,4], [2, 4, 6, 8], [4, 8, 12, 16], [8, 16, 24, 32], [16, 32, 48, 64]]

	model_200 = pickle.load(open('../notebooks/renyi_shannon_prollharm_200.sav', 'rb'))
	model_500 = pickle.load(open('../notebooks/renyi_shannon_prollharm_500.sav', 'rb'))
	model_800 = pickle.load(open('../notebooks/renyi_shannon_prollharm_800.sav', 'rb'))

	size_batch = 1
	j = size_batch*(num_file)
	stop_file = size_batch*(num_file+1)
	if stop_file > len(file_names):
		stop_file = len(file_names)

	black_list = ['/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2008/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--2',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2011/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_09_Track09_wav',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2011/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_05_Track05_wav',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--1']

	for N in range(1):
		for file in file_names[size_batch*(num_file):stop_file]:
		# for file in file_names:
			if file in black_list:
				continue

			y, sr = librosa.load(file + '.wav', sr=44100)
			timestamp = int((len(y) / sr) // 2) # pega o meio da gravação (só analisaremos 30 segundos do meio da música)
			y = y[sr*timestamp:sr*(timestamp+30)]
			
			print("\n\nProcessing file ", j, "of ", len(file_names)) 
			print(file, '\n')
			j += 1

			n_fft = 512
			hop_size = n_fft
			midi = file + '.midi'
			
			# try:
			# 	pct_200 ,pct_500, pct_800, pct_1600, pct_multilevel_2, pct_multilevel_3, pct_multilevel_4 = get_pcts(pcts, file)
			# except:
			# 	print("deu ruim nas pcts")
			# 	continue

			for i in range(len(res_window)):
				res = res_window[i]
				print("Resolution: ", res)

				#STFT
				# result = librosa.stft(y, n_fft=res, hop_length=res)
				# result_stft = get_size(result)
				# print(result_stft)

				# #CQT
				# bpo = hz_to_binperoct(res_hz[i])
				# result = librosa.cqt(y, sr=sr, bins_per_octave=bpo, fmin=20, n_bins=10*bpo)
				# result_cqt = get_size(result)
				# print(result_cqt)

				# LUKIN-TODD, SLS, SWGM
				max_window = res_window[i]
				window_lengths = [min(int(max_window/8), 512), min(int(max_window/2), 2048), max_window]
				# result = swgm_time(y, window_lengths)
				# result_swgm = get_size(result)
				# print(result_swgm)
				kernel_anal = calc_kernel_size(window_lengths)
				kernel_energy = calc_kernel_size(window_lengths, energy=True)
				result = LT(y, window_lengths, kernel_anal)
				result_LT = get_size(result)
				print(result_LT)
				result = SLS_time(y, window_lengths, kernel_anal, kernel_energy)
				result_SLS = get_size(result)


				#OUR SOLUTION
				res = np.max([int(res_window[i] // 512), 1])
				if res == 1:
					res += 1

				# try:
				# 	# result = our_solution(y, res, [200,200], model_200, pct_200, n_fft=n_fft, hop_size=hop_size)
				# 	# result_OURS_200 = get_size(result)
				# 	# print(result_OURS_200)

				# 	# result = our_solution(y, res, [500,500], model_500, pct_500, n_fft=n_fft, hop_size=hop_size)
				# 	# result_OURS_500 = get_size(result)
				# 	# print(result_OURS_500)

				# 	# result = our_solution(y, res, [800,800], model_800, pct_800, n_fft=n_fft, hop_size=hop_size)
				# 	# result_OURS_800 = get_size(result)
				# 	# print(result_OURS_800)

				# 	# result = our_solution(y, res, [1600,1600], model_800, pct_1600, n_fft=n_fft, hop_size=hop_size)        
				# 	# result_OURS_1600 = get_size(result)
				# 	# print(result_OURS_1600)

				# 	result = our_solution_multilevel(y, res_lists_2lvl[i], [[1600,1600], [800,800]], model_800, pct_multilevel_2, sr=44100, n_fft=n_fft, hop_size=n_fft)
				# 	result_OURS_2lvl = get_size(result)
				# 	print(result_OURS_2lvl)

				# 	result = our_solution_multilevel(y, res_lists_3lvl[i], [[1600,1600], [800,800], [400,400]], model_800, pct_multilevel_3, sr=44100, n_fft=n_fft, hop_size=n_fft)
				# 	result_OURS_3lvl = get_size(result)
				# 	print(result_OURS_3lvl)

				# 	result = our_solution_multilevel(y, res_lists_4lvl[i], [[1600,1600], [800,800], [400,400], [200,200]], model_800, pct_multilevel_4, sr=44100, n_fft=n_fft, hop_size=n_fft)
				# 	result_OURS_4lvl = get_size(result)
				# 	print(result_OURS_4lvl)

				# except:
				# 	print("erro nos timers")

				result_file = '../notebooks/cost-experiment/results/memory-results/results_130820.csv'

				with open(result_file, 'a') as csvfile:
					writer = csv.writer(csvfile, delimiter=';')
					# writer.writerow([file, 'STFT', res_window[i], str(result_stft)])
					# writer.writerow([file, 'CQT', res_window[i], str(result_cqt)])
					# writer.writerow([file, 'SWGM', res_window[i], str(result_swgm)])
					writer.writerow([file, 'LT', res_window[i], str(result_LT)])
					writer.writerow([file, 'SLS', res_window[i], str(result_SLS)])
					# writer.writerow([file, 'economic 200', res_window[i], str(result_OURS_200), pct_200])
					# writer.writerow([file, 'economic 500', res_window[i], str(result_OURS_500), pct_500])
					# writer.writerow([file, 'economic 800', res_window[i], str(result_OURS_800), pct_800])
					# writer.writerow([file, 'economic 1600', res_window[i], str(result_OURS_1600), pct_1600])
					# writer.writerow([file, 'economic 2 lvl', res_window[i], str(result_OURS_2lvl), pct_multilevel_2])
					# writer.writerow([file, 'economic 3 lvl', res_window[i], str(result_OURS_3lvl), pct_multilevel_3])
					# writer.writerow([file, 'economic 4 lvl', res_window[i], str(result_OURS_4lvl), pct_multilevel_4])

if __name__ == '__main__':
	# print(sys.argv)
	main(int(sys.argv[1]))