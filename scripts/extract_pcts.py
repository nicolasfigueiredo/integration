import librosa
import numpy as np
import scipy.signal
import math

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

j = 0

# file_name = '../../data/MIDI-Unprocessed_XP_09_R1_2004_05_ORIG_MID--AUDIO_09_R1_2004_06_Track06_wav'

# for file in file_names[:5]:

result_file = '../notebooks/cost-experiment/results/pcts.csv'
# with open(result_file, 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     writer.writerow(['file name', 'pct_200', 'pct_500', 'pct_800', 'pct_1600', 'pct_multilevel_2', 'pct_multilevel_3', 'pct_multilevel_4'])

for file in file_names:
	y, sr = librosa.load(file + '.wav', sr=44100)
	timestamp = int((len(y) / sr) // 2) # pega o meio da gravação (só analisaremos 30 segundos do meio da música)
	y = y[sr*timestamp:sr*(timestamp+30)]
	print(j, file)
	j += 1

	n_fft = 512
	hop_size = n_fft
	midi = file + '.midi'
	
	try:
		pct_200 = get_pct_to_refine(midi, [200,200], timestamp, n_fft=n_fft, hop_size=hop_size)
		pct_500 = get_pct_to_refine(midi, [500,500], timestamp, n_fft=n_fft, hop_size=hop_size)
		pct_800 = get_pct_to_refine(midi, [800,800], timestamp, n_fft=n_fft, hop_size=hop_size)
		pct_1600 = get_pct_to_refine(midi, [1600,1600], timestamp, n_fft=n_fft, hop_size=hop_size)

		pct_multilevel_2 = get_pct_to_refine_multilevel(midi, [[1600,1600], [800,800]], timestamp, n_fft=n_fft, hop_size=n_fft, y=y)
		pct_multilevel_3 = get_pct_to_refine_multilevel(midi, [[1600,1600], [800,800], [400,400]], timestamp, n_fft=n_fft, hop_size=n_fft, y=y)
		pct_multilevel_4 = get_pct_to_refine_multilevel(midi, [[1600,1600], [800,800], [400,400], [200,200]], timestamp, n_fft=n_fft, hop_size=n_fft, y=y)
	except:
		print("erro nos pct")
		continue
	

	result_file = '../notebooks/cost-experiment/results/pcts.csv'

	with open(result_file, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';')
		writer.writerow([file, pct_200, pct_500, pct_800, pct_1600, pct_multilevel_2, pct_multilevel_3, pct_multilevel_4])