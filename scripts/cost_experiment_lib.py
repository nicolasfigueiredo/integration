import librosa
import numpy as np
import scipy.signal
import math

import sys
sys.path.insert(0, '../scripts')

sys.path.insert(0, '../scripts/mauricio_solutions/')
import lukin_todd, swgm, local_sparsity, util_m

import stft_zoom, detect_musical_regions
from util import *
import mappings
import pickle
import PIL
from classes import SingleResSpectrogram, MultiResSpectrogram
import glob
# from aug_density_map import *
from mappings import *

import timeit
import csv
import mido

def energy_reference(y, time_range, freq_range, sr=44100):
    y_slice = y[math.floor(time_range[0] * sr): math.floor(time_range[1] * sr)]
    spec = np.fft.rfft(y_slice)
    freqs = np.fft.rfftfreq(len(y_slice), 1./sr)
    idx_start = find_nearest(freqs, freq_range[0])
    idx_stop = find_nearest(freqs, freq_range[1])
    return np.sum(np.abs(spec[idx_start:idx_stop])**2) * 2 / len(y_slice)

def normalize_subregion(spec_zoom, time_range, freq_range, y):
    energy_ref = energy_reference(y, time_range, freq_range)
    energy_old = np.sum(spec_zoom ** 2)
    return math.sqrt(energy_ref / energy_old) * spec_zoom


def calc_kernel_size(window_lengths, energy=False):
    if energy:
        P = 2
    else:
        P = 5
    f_size = P * window_lengths[-1]/window_lengths[0]
    if not f_size % 2 > 0:
        f_size += 1
    
    return [int(f_size), 5]

def LT(y, window_lengths, kernel):
    specs = util_m.get_spectrograms(y, windows=window_lengths)
    specs = util_m.interpol_and_normalize(specs)
    return lukin_todd.lukin_todd(specs, kernel)

def swgm_time(y, window_lengths):
    specs = util_m.get_spectrograms(y, windows=window_lengths)
    specs = util_m.interpol_and_normalize(specs)
    return swgm.SWGM(specs)

def SLS_time(y, window_lengths, kernel_anal, kernel_energy):
    specs = util_m.get_spectrograms(y, windows=window_lengths)
    specs = util_m.interpol_and_normalize(specs)
    return local_sparsity.smoothed_local_sparsity(specs, kernel_anal, kernel_energy)

def ers(y, res, kernel, model, pct, sr=44100, n_fft=512, hop_size=512, normalize=True):
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size))
    time_span = [0,len(y)/sr]
    x_axis, y_axis = stft_zoom.get_axes_values(sr, 0, time_span, spec.shape) 
    base_spec = SingleResSpectrogram(spec, x_axis, y_axis)
    multires_spec = MultiResSpectrogram(base_spec)
    indices, original_shape = detect_musical_regions.detect_musical_regions(model, spec, kernel=kernel, mode='pct', pct_or_threshold=pct, n_fft=n_fft, hop_size=hop_size)
    to_be_refined = detect_musical_regions.musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, n_fft=n_fft, hop_size=hop_size)

    stft_zoom.set_signal_bank(y,kernel, n_fft=n_fft)

    for subregion in to_be_refined:
        freq_range = subregion[0]
        time_range = subregion[1]
        spec_zoom, x_axis, y_axis, new_sr, window_size, hop_size = stft_zoom.stft_zoom(y, freq_range, time_range, sr=sr, original_window_size=n_fft, k=res)
        refined_subspec = SingleResSpectrogram(spec_zoom, x_axis, y_axis)
        multires_spec.insert_zoom(multires_spec.base_spec, refined_subspec, freq_range, zoom_level=1, normalize=normalize)
        
    return multires_spec

def ers_multilevel(y, res_list, kernel_list, model, pct_list, sr=44100, n_fft=512, hop_size=512, normalize=True):
    o_n_fft = n_fft
    o_hop_size = hop_size
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size))
    time_span = [0,len(y)/sr]
    x_axis, y_axis = stft_zoom.get_axes_values(sr, 0, time_span, spec.shape) 
    base_spec = SingleResSpectrogram(spec, x_axis, y_axis)
    multires_spec = MultiResSpectrogram(base_spec)
    
    kernel = kernel_list[0]
    indices, original_shape = detect_musical_regions.detect_musical_regions(model, spec, kernel=kernel, mode='pct', pct_or_threshold=pct_list[0], n_fft=n_fft, hop_size=hop_size)
    to_be_refined = detect_musical_regions.musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, n_fft=n_fft, hop_size=hop_size)

    stft_zoom.set_signal_bank(y,kernel, n_fft=n_fft)
    
    for subregion in to_be_refined:
        freq_range = subregion[0]
        time_range = subregion[1]
        spec_zoom, x_axis, y_axis, new_sr, window_size, hop_size = stft_zoom.stft_zoom(y, freq_range, time_range, sr=sr, original_window_size=n_fft, k=res_list[0])
        refined_subspec = SingleResSpectrogram(spec_zoom, x_axis, y_axis, n_fft=window_size, hop_size=hop_size, sr=new_sr)
        multires_spec.insert_zoom(multires_spec.base_spec, refined_subspec, freq_range, zoom_level=1, normalize=normalize)
        
    i = 1
    for kernel in kernel_list[1:]:
        i += 1
#         print(res_list[i-1])
        if i == 2:
            spec_list = multires_spec.first_zoom
        elif i == 3:
            spec_list = multires_spec.second_zoom
        elif i == 4:
            spec_list = multires_spec.third_zoom

        to_be_further_refined = []
        for spec_zoom in spec_list:
            spec        = np.array(spec_zoom.spec, dtype=float)
            x_axis      = spec_zoom.x_axis
            y_axis      = spec_zoom.y_axis
            sr          = spec_zoom.sr
            window_size = spec_zoom.n_fft
            hop_size    = spec_zoom.hop_size
            
            if len(y_axis) == 1 or len(x_axis) == 1:
                continue
            
            indices, original_shape = detect_musical_regions.detect_musical_regions(model, spec, mode='pct', pct_or_threshold=pct_list[i-1], kernel=kernel, n_fft=window_size, hop_size=hop_size, sr=sr, y_axis=y_axis)
            to_be_further_refined.append([spec_zoom, detect_musical_regions.musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, sr=sr, hop_size=hop_size)])

        sr = 44100
        hop_size = o_hop_size
        n_fft = o_n_fft
        time_step = o_hop_size / sr
        for subregion in to_be_further_refined:
            base_spec = subregion[0]
            for ranges in subregion[1]:
                freq_range = ranges[0]
                time_range = ranges[1]
                # Compensacao da centralizacao das janelas no tempo e STFT centrada em 0
                time_range[0] -= time_step/2
                time_range[1] += time_step/2
                if time_range[0] < 0:
                    time_range[0] = 0

                spec_zoom, x_axis, y_axis, new_sr, window_size, hop_size = stft_zoom.stft_zoom(y, freq_range, time_range, sr=sr, original_window_size=n_fft, k=res_list[i-1])
                refined_subspec = SingleResSpectrogram(spec_zoom, x_axis, y_axis, n_fft=window_size, hop_size=hop_size, sr=new_sr)
                multires_spec.insert_zoom(base_spec, refined_subspec, freq_range, zoom_level=i, normalize=normalize)
    return multires_spec

def hz_to_cents(hz_resolution):
    # Converts from Hz res to cent resolution in the octave from 100-200Hz
    delta_f = 100
    cent_delta_f = delta_f/1200
    return hz_resolution / cent_delta_f

def hz_to_binperoct(hz_resolution):
    return int(np.ceil(1200 / hz_to_cents(hz_resolution)))

def get_refine_map(midi_file, kernel, timestamp, num_time_frames, n_fft=512, hop_size=512):
    midi = mido.MidiFile(midi_file)
    proll = midi_to_piano_roll_new(midi, timestamp)
    proll_den = calc_map_aug2(sound_event_to_tfp(proll, num_time_frames, n_fft=n_fft), kernel, type='avg', n_fft=n_fft, hop_size=hop_size)
    return np.array(proll_den > 0, dtype=int)

def get_pct_to_refine(midi_file, kernel, timestamp, n_fft=512, hop_size=512):
    num_time_frames = int(np.ceil(44100 * 30 / hop_size))
    refine_map = get_refine_map(midi_file, kernel, timestamp, num_time_frames, n_fft=n_fft, hop_size=hop_size)
    return 100*(np.sum(refine_map)/refine_map.size)

# fazer: overlap dos detectados com kernel grande e todos kernel pequeno: N
#        overlap dos detectados com kernel pequeno e detectados kernel grande: n
#        o numero que queremos é n/N

# no caso real: N continua o mesmo, overlap dos detectado com kernel pequeno e refinados kernel grande = n

def in_range_list(range, range_list):
    for r in range_list:
        if range[0][0] >= r[0][0] and range[0][1] <= r[0][1]:
            if range[1][0] >= r[1][0] and range[1][1] <= r[1][1]:
                return True
    return False

def all_overlapping_ranges(range_list_1, range_list_2):
    # returns ranges in list_1 that are contained inside the regions defined by range_list_2
    overlapping_ranges = []
    for range in range_list_1:
        if in_range_list(range, range_list_2):
            overlapping_ranges.append(range)
    return overlapping_ranges

def get_pct_to_refine_multilevel(midi_file, kernel_list, timestamp, n_fft=512, hop_size=512, y=None):
# 1. pegar kernel primeiro nivel, ver no ground truth as regioes relevantes, guardar os ranges.
# 2. pegar kernel segundo, ver no ground truth as regioes relevantes, guardar os ranges.
# 3. ver overlap entre passo 2 e 1 = n
# 4. pegar kernel segundo, pegar ranges de todas as subregioes definidas por ele
# 5. ver overlap entre passo 4 e 1 = N
# 6. calcular pct, guardar regiões do passo 3 como as novas subregioes refinadas.
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size))
    x_axis, y_axis = stft_zoom.get_axes_values(44100, 0, [0.0, 30.0], spec.shape) 
    num_time_frames = int(np.ceil(44100 * 30 / hop_size))
    first_level = get_refine_map(midi_file, kernel_list[0], timestamp, num_time_frames, n_fft=n_fft, hop_size=hop_size)
    pct_list = [100*(np.sum(first_level)/first_level.size)]
    indices_first_level = np.where(first_level.flatten() > 0)[0]
    range_list_first = detect_musical_regions.musical_regions_to_ranges(indices_first_level, first_level.shape, x_axis, y_axis, kernel_list[0], n_fft=n_fft, hop_size=hop_size)
    for kernel in kernel_list[1:]:
        second_level = get_refine_map(midi_file, kernel, timestamp, num_time_frames, n_fft=n_fft, hop_size=hop_size)
        indices_second_level = np.where(second_level.flatten() > 0)[0]
        all_subregions_second_level = np.where(second_level.flatten() > -1)[0]
        all_ranges_second_level = detect_musical_regions.musical_regions_to_ranges(all_subregions_second_level, second_level.shape, x_axis, y_axis, kernel, n_fft=n_fft, hop_size=hop_size)
        range_list_second = detect_musical_regions.musical_regions_to_ranges(indices_second_level, second_level.shape, x_axis, y_axis, kernel, n_fft=n_fft, hop_size=hop_size)
        n = len(all_overlapping_ranges(range_list_second, range_list_first))
        N = len(all_overlapping_ranges(all_ranges_second_level, range_list_first))
        pct_list.append((n/N) * 100)
        range_list_first = range_list_second
    return pct_list