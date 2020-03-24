import mappings
from util import *
import numpy as np
import librosa
import PIL


# Recebe um espectrograma, calcula um mapa de interesse musical e devolve uma lista ordenada de regiões
# mais interessantes
# A lista retornada são os índices flattened (retornar índices ou probs?) 
# ainda precisamos fazer o "unravel" e transformar em ranges
def detect_musical_regions(model, spectrogram, mode='threshold', pct_or_threshold=0.75, kernel=[800, 800], n_fft=2048, hop_size=512, sr=44100, y_axis=None):
    spec_amp = np.abs(spectrogram)
    spec_db = librosa.amplitude_to_db(spec_amp, ref=np.min)
    
    renyi = mappings.calc_map_aug2(spec_amp, kernel, type='renyi', n_fft=n_fft, hop_size=hop_size, sr=sr, fft_freqs=y_axis)
    shannon = mappings.calc_map_aug2(spec_db, kernel, type='shannon', n_fft=n_fft, hop_size=hop_size, sr=sr, fft_freqs=y_axis)
    
    X = np.array([shannon.flatten(), renyi.flatten()]).T
    predicted_probs = model.predict_proba(X)
    
    if mode == 'threshold':
        sorted_probs = np.sort(predicted_probs[:,1])[::-1]
        idx = sorted_probs.size - np.searchsorted(sorted_probs[::-1], pct_or_threshold, side = "right")
    elif mode == 'pct':
        idx = int(len(renyi.flatten()) * (pct_or_threshold/100))
    return np.argsort(predicted_probs[:,1])[::-1][:idx], shannon.shape  # shape do mapa é usado depois em outras funções

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

def index_to_range(idx, x_axis, y_axis, kernel, sr=44100, n_fft=2048, hop_size=512):
    idx_x = idx[1]
    idx_y = idx[0]
    freq_idx_list = mappings.find_freq_list(y_axis, kernel[1])
    ms_per_frame = hop_size * 1000 / sr
    delta_x_idx = int(np.round(kernel[0] / ms_per_frame)) # cada "bloco" do mapa estimador está entre x_axis[i] e x_axis[i+delta_x_idx] 
    freq_range = [y_axis[freq_idx_list[idx_y]], y_axis[freq_idx_list[idx_y+1]]]
    time_range = [x_axis[idx_x*delta_x_idx], x_axis[(idx_x+1)*delta_x_idx]]
    return freq_range, time_range

# Recebe índices de regiões e devolve time_ranges e freq_ranges correspondentes
def musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, sr=44100, n_fft=2048, hop_size=512):
    ranges = []
    for idx in indices:
        ranges.append(index_to_range(np.unravel_index(idx, original_shape), x_axis, y_axis, kernel, sr=sr, n_fft=n_fft, hop_size=hop_size))
    return ranges

def insert_zoom(spec_img, zoom, time_range, freq_range, x_axis, y_axis):
    # tem uma redundância com x_axis e time_range...
    # TEM QUE CORTAR O ZOOM ANTES DE INSERIR

    # y_axis = fft_frequencies(sr=sr, n_fft=n_fft)
    # x_axis = librosa.core.frames_to_time(list(range(base_spec_shape)), sr=sr, hop_length=hop_size)
    
    x_start = find_nearest(x_axis, time_range[0])
    x_end   = find_nearest(x_axis, time_range[1])
    y_start = find_nearest(y_axis, freq_range[0])
    y_end   = find_nearest(y_axis, freq_range[1])
    
    zoom_img = PIL.Image.fromarray(zoom).resize((x_end - x_start,y_end - y_start))
#     spec_img = PIL.Image.fromarray(base_spec).resize((base_spec.shape[1] * 3,base_spec.shape[0] * 3))
    box = (x_start, y_start, x_end, y_end)
    spec_img.paste(zoom_img, box)
    
    return spec_img