import scipy.signal
import numpy as np
import librosa
from util import *

def compose_alpha_list(sr):
	alpha_list = []
	for i in range(1,sr):
		if int(sr % i) == 0:
			alpha_list.append(i)
	return alpha_list

alpha_list = compose_alpha_list(44100)

def closest_alpha(possible_alpha):
    idx = np.searchsorted(alpha_list, possible_alpha, side="left")
    if alpha_list[idx] != possible_alpha:
        return alpha_list[idx-1]
    else:
        return possible_alpha

def slice_signal(y, time_range, sr):
    return y[int(sr * time_range[0]) : int(sr * time_range[1])]
    
    
# filter_and_mod chooses between the following three options:
#     1) if freq_range[1] < 200, filter with low-pass and do not modulate
#     2) else, filter with band-pass and:
#            a) if possible, find an undersampling frequency. check if the spectrum will be inverted by subsampling with the found frequency rate
#            b) else, perform ring-modulation and filter with low-pass (this is a cheaper version of SSB modulation)
# Returns the filtered and modulated signal, the new sampling rate, the freq mapped to DC and a boolean variable that
# tells if the spectrum is inverted or not (it happens with undersampling) 
    
def filter_and_mod(y, freq_range, sr):
    inverted = False # se for feito undersampling com n par, o espectro é espelhado e temos que inverter depois. essa é uma flag para essa situação

    if freq_range[0] <= 200:
        return filter_lowpass(y, freq_range[1], sr), 2*(freq_range[1]+100), 0, inverted
      
    wp = np.array([freq_range[0] - 50, freq_range[1] + 50])
    ws = np.array([wp[0] - 50, wp[1] + 150]) # [alpha, beta]

    new_sr = find_undersample_fs(ws)

    wp = wp / (sr/2)
    ws = ws / (sr/2)
    
    y_filt = filter_bandpass(y, wp, ws, sr)
    
    if not new_sr: # ringmod + lpf
        new_sr = (ws[1] - ws[0] + 100/(sr/2)) * sr
        print("ring mod + lpf")
        return filter_lowpass(ring_mod(y_filt, ws[0]*(sr/2), sr), new_sr/2 - 100, sr), new_sr, ws[0]*(sr/2), inverted         
        
    print("undersampling")
    new_freq_range = treat_undersampling(new_sr[0], new_sr[1], ws*(sr/2))
    if new_sr[1] == 0: # undersampling frequency found using even n, mirroring of spectrum is needed
        print(new_sr, new_freq_range)
        inverted = True
        return y_filt, new_sr[0], [ws*(sr/2), new_freq_range], inverted

    print(new_sr, new_freq_range)
    return y_filt, new_sr[0], ws[0]*(sr/2) - new_freq_range[0], inverted

    
def filter_bandpass(y, wp, ws, sr):
    N, wn = scipy.signal.buttord(wp, ws, 3, 30)
    sos = scipy.signal.butter(N, wn, 'band', output='sos')
    return scipy.signal.sosfilt(sos, y)

def filter_lowpass(y, f_c, sr):
    anti_alias = scipy.signal.ellip(7, 3, 80, f_c / (sr/2), output='sos')
    return scipy.signal.sosfilt(anti_alias, y)
    
# retorna 0 se não é possível fazer undersampling
# retorna a nova taxa de amostragem caso contrário
def find_undersample_fs(freq_range):
    f_l = freq_range[0]
    f_h = freq_range[1]

    n_upperlim = int(np.floor(f_h/(f_h - f_l)))
    
    for n in range(n_upperlim, 1, -1):
        new_sr_h = 2 * f_l / (n-1)
        new_sr_l = 2 * f_h / n

        i_l = np.searchsorted(alpha_list, new_sr_l)
        i_h = np.searchsorted(alpha_list, new_sr_h)

        if i_l != i_h:
            return alpha_list[i_h-1], n % 2

    return(0)

def treat_undersampling(undersample_freq, n_parity, freq_range):
    print(freq_range)
    if n_parity == 0:
        new_fl = np.ceil(freq_range[1]/undersample_freq) * undersample_freq - freq_range[1]
        new_fh = new_fl + freq_range[1] - freq_range[0]
    else:
        new_fl = freq_range[0] - np.floor(freq_range[0]/undersample_freq) * undersample_freq
        new_fh = new_fl + freq_range[1] - freq_range[0]
    return [new_fl, new_fh]

# retorna 0 se não é possível fazer a subamostragem simples
# retorna a nova taxa de amostragem caso contrário
def check_subsample(sr, ws):
    M_upperlim = int(ws[0] // (ws[1] - ws[0]))
    if M_upperlim < 1:
        return False
    for M in range(M_upperlim, 0, -1):
        if test_new_sr(2 * ws[0] / M, sr):
            return (2 * ws[0] / M)
    return False  # não é possível fazer o "aliasing inteligente"
                             
        
# Testa se a nova taxa de amostragem pode ser obtida 
# com uma subamostragem simples (selecionado 1 a cada X amostras de y)
def test_new_sr(new_sr, sr):
    if int(sr % new_sr) == 0:
        return True
    else:
        return False

def ring_mod(y, freq, sr):
    t_final = len(y) / sr
    t = np.linspace(0,t_final, int(sr*t_final))
    x = np.cos(2*np.pi*freq*t)
    return x*y

def subsample_signal(y, new_sr, sr):
    subsample_step = int(np.ceil(sr/new_sr)) # pegar 1 em cada subsample_step amostras de y
    return y[::subsample_step], sr/subsample_step  # sinal subamostrado, new_sr

# def analyze_slice(y, freq_range, sr, freq_res_type, freq_res, time_res_type, time_res):
#     # devolve matriz da FFT de y no intervalo freq_range, time_range de acordo com alguma
#     # "heuristica de resolução", por ex: quero 10 bins de freq nesse intervalo, 10 frames de tempo
#     if freq_res_type == 'freq. bins':
#         freq_res_hz = (freq_range[1] - freq_range[0]) / freq_res
#         window_size = int(sr // freq_res_hz)
#     else: # freq res specified in Hz per bin
#         window_size = int(sr // freq_res)

#     if window_size > len(y):
#         window_size = len(y) 
        
#     if time_res == 0:
#         hop_size = window_size // 4
#     elif time_res_type == 'time frames':
#         hop_size = int(len(y) / time_res)
#     else:
#         hop_size = int(sr*time_res/1000)

#     return np.abs(librosa.stft(y, n_fft=window_size, hop_length=hop_size))

# def stft_zoom(y, freq_range, time_range, sr, freq_res_type='freq. bins', freq_res=40, time_res_type='time frames', time_res=0):
#     inverted = False
#     y_mod, new_sr, f_min, inverted = filter_and_mod(slice_signal(y, time_range, sr), freq_range, sr)
#     y_sub, new_sr = subsample_signal(y_mod, new_sr, sr)

#     D = analyze_slice(y_sub, freq_range, new_sr, freq_res_type, freq_res, time_res_type, time_res)
    
#     if type(f_min) is list: # undersampling que inverteu o espectro entre f_min[0] e f_min[1]
#         ws = f_min[0]
#         new_freq_range = f_min[1]        
#         f_min = ws[0] - new_freq_range[0]
#         x_axis, y_axis = get_axes_values(new_sr, f_min, time_range, D.shape)
#         D = unmirror(D, y_axis, ws)        
#     else:
#         x_axis, y_axis = get_axes_values(new_sr, f_min, time_range, D.shape)

#     return D, x_axis, y_axis

def analyze_slice(y, new_sr, original_resolution, k=2):
    # devolve matriz da FFT de y no intervalo freq_range, time_range de acordo com alguma
    # "heuristica de resolução", por ex: quero 10 bins de freq nesse intervalo, 10 frames de tempo
    new_resolution = original_resolution / k
    window_size = int(new_sr / new_resolution)
    hop_size = window_size // 4
    return np.abs(librosa.stft(y, n_fft=window_size, hop_length=hop_size)), window_size, hop_size

def stft_zoom(y, freq_range, time_range, sr=44100, original_window_size=2048, k=2):
    inverted = False
    y_mod, new_sr, f_min, inverted = filter_and_mod(slice_signal(y, time_range, sr), freq_range, sr)
    y_sub, new_sr = subsample_signal(y_mod, new_sr, sr)

    original_resolution = sr / original_window_size
    D, new_window_size, new_hop_size = analyze_slice(y_sub, new_sr, original_resolution, k=k)
    
    if type(f_min) is list: # undersampling que inverteu o espectro entre f_min[0] e f_min[1]
        ws = f_min[0]
        new_freq_range = f_min[1]        
        f_min = ws[0] - new_freq_range[0]
        x_axis, y_axis = get_axes_values(new_sr, f_min, time_range, D.shape)
        D = unmirror(D, y_axis, ws)        
    else:
        x_axis, y_axis = get_axes_values(new_sr, f_min, time_range, D.shape)

    # Aqui, cortamos o espectrograma só na faixa de frequẽncias especificada
    y_start = find_nearest(y_axis, freq_range[0])
    y_end   = find_nearest(y_axis, freq_range[1])

    return D[y_start:y_end,:], x_axis, y_axis[y_start:y_end], new_sr, new_window_size, new_hop_size

def unmirror(D, y_axis, freq_range):
    i_start = np.searchsorted(y_axis, freq_range[0])
    i_stop  = np.searchsorted(y_axis, freq_range[1]) + 1
    D[i_start:i_stop, :] = D[i_start:i_stop, :][::-1]
    return D


def get_axes_values(sr, f_min, time_range, spec_shape):
    x_axis = np.linspace(time_range[0], time_range[1], spec_shape[1])
    f_max = f_min + (sr / 2)
    y_axis = np.linspace(f_min, f_max, spec_shape[0])
    return x_axis, y_axis