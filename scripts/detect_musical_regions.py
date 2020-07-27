import mappings
from util import *
import numpy as np
import librosa
import PIL

# This script contains functions that perform the detection of musical subregions of a spectrogram.
# detect_musical_regions() performs feature extraction and prediction, and the other functions compute the
# mapping between the output of detect_musical_regions() and the corresponding time range x freq. range that
# define these subregions.

def detect_musical_regions(model, spectrogram, mode='threshold', pct_or_threshold=0.75, kernel=[800, 800], n_fft=2048, hop_size=512, sr=44100, y_axis=None):
    # Takes a spectrogram and detects its musically interesting subregions, according to a NB model given by 'model'
    # and a threshold of probability or percentile.
    # The returned list are the flattened indices that represent interesting subregions in the feature matrix, that
    # should be converted later
    
    # Compute features
    shannon, renyi = mappings.extract_features(spectrogram, kernel, n_fft=n_fft, hop_size=hop_size, sr=sr, fft_freqs=y_axis)
    X = np.array([shannon.flatten(), renyi.flatten()]).T
    
    # Predict probability
    predicted_probs = model.predict_proba(X)
    
    # Filter results according to threshold and return a sorted list of indices
    if mode == 'threshold':
        sorted_probs = np.sort(predicted_probs[:,1])[::-1]
        idx = sorted_probs.size - np.searchsorted(sorted_probs[::-1], pct_or_threshold, side = "right")
    elif mode == 'pct':
        idx = int(len(renyi.flatten()) * (pct_or_threshold/100))
    return np.argsort(predicted_probs[:,1])[::-1][:idx], shannon.shape  # shape of feature map is used later in other functions

def musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, sr=44100, n_fft=2048, hop_size=512):
    # Takes as argument a list of indices of musically interesting regions and returns the corresponding
    # frequency and time ranges of such regions.
    ranges = []
    for idx in indices:
        ranges.append(index_to_range(np.unravel_index(idx, original_shape), x_axis, y_axis, kernel, sr=sr, n_fft=n_fft, hop_size=hop_size))
    return ranges

def index_to_range(idx, x_axis, y_axis, kernel, sr=44100, n_fft=2048, hop_size=512):
    # Given an index of the feature matrix and the kernel dimensions that were used to compute such feature matrix,
    # return the corresponding freq. range and time range that determine this subregion of the time-freq. plane    
    idx_x = idx[1]
    idx_y = idx[0]
    freq_idx_list = mappings.find_freq_list(y_axis, kernel[1]) # find frequencies that correspond to the y_axis of the feature map
    ms_per_frame = hop_size * 1000 / sr
    delta_x_idx = int(np.round(kernel[0] / ms_per_frame)) # each "block" of the feature map lies between x_axis[i] and x_axis[i+delta_x_idx] 
    # if delta_x_idx == 1:
    #     delta_x_idx += 1
    freq_range = [y_axis[freq_idx_list[idx_y]], y_axis[freq_idx_list[idx_y+1]]]
    end_point = (idx_x+1)*delta_x_idx
    if end_point >= len(x_axis):
        end_point = len(x_axis) - 1
    start_point = idx_x*delta_x_idx
    if start_point == end_point:
        start_point -= 1

    time_range = [x_axis[start_point], x_axis[end_point]]
    # if kernel == [100,100]:
    #     print(ms_per_frame, delta_x_idx, time_range)
    return freq_range, time_range

# Should be moved to the MultiResSpec() class in the future
def insert_zoom(spec_img, zoom, time_range, freq_range, x_axis, y_axis):
    x_start = find_nearest(x_axis, time_range[0])
    x_end   = find_nearest(x_axis, time_range[1])
    y_start = find_nearest(y_axis, freq_range[0])
    y_end   = find_nearest(y_axis, freq_range[1])
    
    zoom_img = PIL.Image.fromarray(zoom).resize((x_end - x_start,y_end - y_start))
    box = (x_start, y_start, x_end, y_end)
    spec_img.paste(zoom_img, box)
    
    return spec_img




# def kernel_to_ranges(x_span, x_size, kernel_size=[800,800], sr=44100, n_fft=2048, hop_size=512):
#     freq_list = fft_frequencies(sr=sr, n_fft=n_fft)
#     idx_list = mappings.find_freq_list(freq_list, kernel_size[1])
#     freq_ranges = freq_list[idx_list]

    
#     time_ranges = np.linspace(x_span[0], x_span[1], x_size+1)  # discussão: levar p/ reunião 
#     return freq_ranges, time_ranges

# def index_to_range(idx, freq_ranges, time_ranges):
#     freq_range = [freq_ranges[idx[0]], freq_ranges[idx[0]+1]]
#     time_range = [time_ranges[idx[1]], time_ranges[idx[1]+1]]
#     return freq_range, time_range