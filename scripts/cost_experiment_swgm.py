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
from timeit import default_timer as timer

#do mauricio
sys.path.insert(0, '../scripts/mauricio_solutions/')
import lukin_todd, swgm, local_sparsity, util_m

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
	# result_file = '../notebooks/cost-experiment/results/results_040820_swgm.csv'
	# with open(result_file, 'w') as csvfile:
	#     writer = csv.writer(csvfile, delimiter=';')
	#     writer.writerow(['file name', 'representation', 'max res', 'timeit', 'pct refine'])
	# print("csv overwritten")

	# pcts = pd.read_csv('../notebooks/cost-experiment/results/pcts.csv', sep=';', dtype={'pct_200':float, 'pct_500':float, 'pct_800':float, 'pct_1600':float})
	# pcts = treat_pcts(pcts)


	# # file_name = '../data/MIDI-Unprocessed_XP_09_R1_2004_05_ORIG_MID--AUDIO_09_R1_2004_06_Track06_wav.wav'
	# # y, sr = librosa.load(file_name, sr=44100)
	# # y = y[:44100*10]

	# # _ = librosa.cqt(y)

	file_names = []
	for year in ['2006','2008','2009','2011','2013','2014','2015']:
	    path = '/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/' + year + '/*.wav'
	    for file in glob.glob(path):
	        file_names.append(file[:-4])

	# file_names = []
	# for year in ['2004']:
	# 	path = '/Users/nicolas/Documents/ime/compmus/mestrado/maestro-dataset/' + year + '/*.wav'
	# 	for file in glob.glob(path):
	# 		file_names.append(file[:-4])

	print("files fetched: ",len(file_names))

	res_hz = [86.13, 43.06, 21.53, 10.77, 5.38, 2.69, 1.34]
	res_window = [512, 1024, 2048, 4096, 8192, 16384, 32768]

	# res_lists_2lvl = [[1,1], [2,2], [2,4], [4,8], [8, 16], [16, 32], [32, 64]]
	# res_lists_3lvl = [[1,1,1], [2,2,2], [2,3,4], [3, 6, 8], [5, 10, 16], [10, 21, 32], [21, 42, 64]]
	# res_lists_4lvl = [[1,1,1,1], [2,2,2,2], [1,2,3,4], [2, 4, 6, 8], [4, 8, 12, 16], [8, 16, 24, 32], [16, 32, 48, 64]]

	# model_200 = pickle.load(open('../notebooks/renyi_shannon_prollharm_200.sav', 'rb'))
	# model_500 = pickle.load(open('../notebooks/renyi_shannon_prollharm_500.sav', 'rb'))
	# model_800 = pickle.load(open('../notebooks/renyi_shannon_prollharm_800.sav', 'rb'))


	size_batch = 30
	j = size_batch*(num_file)
	stop_file = size_batch*(num_file+1)
	if stop_file > len(file_names):
		stop_file = len(file_names)


	# file_name = '../../data/MIDI-Unprocessed_XP_09_R1_2004_05_ORIG_MID--AUDIO_09_R1_2004_06_Track06_wav'

	black_list = ['/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2008/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--2',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2011/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_09_Track09_wav',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2011/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_05_Track05_wav',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3',
					'/Volumes/HD-NICO/vaio-backup/Documents/ime/compmus/mestrado/maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--1']

	start_abs = timer()

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

				# #STFT
				# start = timer()
				# _ = librosa.stft(y, n_fft=res, hop_length=res)
				# end = timer()
				# result_stft = (end - start)
				# print(result_stft)

				# #CQT
				# bpo = hz_to_binperoct(res_hz[i])
				# start = timer()
				# _ = librosa.cqt(y, sr=sr, bins_per_octave=bpo, fmin=20, n_bins=10*bpo)
				# end = timer()
				# result_cqt = (end - start)
				# print(result_cqt)

				# LUKIN-TODD, SLS, SWGM
				max_window = res_window[i]
				window_lengths = [int(max_window/8), int(max_window/2), max_window]
				start = timer()
				_ = swgm_time(y, window_lengths)
				end = timer()
				result_swgm = (end - start)
				print(result_swgm)

				# OUR SOLUTION
				# res = np.max([int(res_window[i] // 512), 1])
				# if res == 1:
				# 	res += 1

				# try:
				# 	# start = timer()
				# 	# _ = our_solution(y, res, [200,200], model_200, pct_200, n_fft=n_fft, hop_size=hop_size)
				# 	# end = timer()
				# 	# result_OURS_200 = end - start
				# 	# print(result_OURS_200)

				# 	# start = timer()
				# 	# _ = our_solution(y, res, [500,500], model_500, pct_500, n_fft=n_fft, hop_size=hop_size)
				# 	# end = timer()
				# 	# result_OURS_500 = end - start
				# 	# print(result_OURS_500)

				# 	# start = timer()
				# 	# _ = our_solution(y, res, [800,800], model_800, pct_800, n_fft=n_fft, hop_size=hop_size)
				# 	# end = timer()
				# 	# result_OURS_800 = end - start
				# 	# print(result_OURS_800)

				# 	# start = timer()
				# 	# _ = our_solution(y, res, [1600,1600], model_800, pct_1600, n_fft=n_fft, hop_size=hop_size)        
				# 	# end = timer()
				# 	# result_OURS_1600 = end - start
				# 	# print(result_OURS_1600)

				# 	start = timer()
				# 	_ = our_solution_multilevel(y, res_lists_2lvl[i], [[1600,1600], [800,800]], model_800, pct_multilevel_2, sr=44100, n_fft=n_fft, hop_size=n_fft)
				# 	end = timer()
				# 	result_OURS_2lvl = end - start
				# 	print(result_OURS_2lvl)

				# 	start = timer()
				# 	_ = our_solution_multilevel(y, res_lists_3lvl[i], [[1600,1600], [800,800], [400,400]], model_800, pct_multilevel_3, sr=44100, n_fft=n_fft, hop_size=n_fft)
				# 	end = timer()
				# 	result_OURS_3lvl = end - start
				# 	print(result_OURS_3lvl)

				# 	start = timer()
				# 	_ = our_solution_multilevel(y, res_lists_4lvl[i], [[1600,1600], [800,800], [400,400], [200,200]], model_800, pct_multilevel_4, sr=44100, n_fft=n_fft, hop_size=n_fft)
				# 	end = timer()
				# 	result_OURS_4lvl = end - start
				# 	print(result_OURS_4lvl)

				# except:
				# 	print("erro nos timers")

				result_file = '../notebooks/cost-experiment/results/results_040820_swgm.csv'

				with open(result_file, 'a') as csvfile:
					writer = csv.writer(csvfile, delimiter=';')
					# writer.writerow([file, 'STFT', res_window[i], str(result_stft)])
					# writer.writerow([file, 'CQT', res_window[i], str(result_cqt)])
					writer.writerow([file, 'SWGM', res_window[i], str(result_swgm)])
					# writer.writerow([file, 'economic 200', res_window[i], str(result_OURS_200), pct_200])
					# writer.writerow([file, 'economic 500', res_window[i], str(result_OURS_500), pct_500])
					# writer.writerow([file, 'economic 800', res_window[i], str(result_OURS_800), pct_800])
					# writer.writerow([file, 'economic 1600', res_window[i], str(result_OURS_1600), pct_1600])
					# writer.writerow([file, 'economic 2 lvl', res_window[i], str(result_OURS_2lvl), pct_multilevel_2])
					# writer.writerow([file, 'economic 3 lvl', res_window[i], str(result_OURS_3lvl), pct_multilevel_3])
					# writer.writerow([file, 'economic 4 lvl', res_window[i], str(result_OURS_4lvl), pct_multilevel_4])

	end_abs = timer()

	print("TOTAL TIME: ", end_abs - start_abs)

if __name__ == '__main__':
	# print(sys.argv)
	main(int(sys.argv[1]))