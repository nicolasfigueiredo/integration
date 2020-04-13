import scipy.signal
import numpy as np
import librosa
from util import *

# This script contains all functions that perform subband processing in order to compute a low cost STFT
# representation of a specific region (frequency band x time interval) of a signal. The main function in
# this script is stft_zoom().

def compose_alpha_list(sr):

#   Computes the list of subsampling frequencies that can be achieved by the method of taking
#   each one sample out of every X samples of the original signal (for example, the sampling 
#   frequency of 22050 can be achieved by taking one sample out of every 2 samples (if sr==44100) 
#   This will list will be used in undersampling.

	alpha_list = []
	for i in range(1,sr):
		if int(sr % i) == 0:
			alpha_list.append(i)
	return alpha_list

alpha_list = compose_alpha_list(44100)

def slice_signal(y, time_range, sr):
    return y[int(sr * time_range[0]) : int(sr * time_range[1])]
    
def filter_and_mod(y, freq_range, sr):

#   filter_and_mod() chooses between the following three options:
#         1) if freq_range[1] < 200, filter with low-pass and do not modulate
#         2) else, filter with band-pass and:
#            a) if possible, find an undersampling frequency. Check if the spectrum will be inverted by subsampling with the found frequency rate
#            b) else, perform ring-modulation and filter with low-pass (this is a cheaper version of SSB modulation)
#   Returns the filtered and modulated signal, the new sampling rate, the freq. mapped to DC and a boolean variable that
#   tells if the spectrum is inverted or not (it can happen with undersampling)

    inverted = False # if undersampling is performed with an even 'n', the spectrum is mirrored and will be unmirrored afterwards

    if freq_range[0] <= 200:
        return filter_lowpass(y, freq_range[1], sr), 2*(freq_range[1]+100), 0, inverted
      
    wp = np.array([freq_range[0] - 50, freq_range[1] + 50]) # lower freq. for bandpass filter 
    ws = np.array([wp[0] - 50, wp[1] + 150]) # higher freq. for bandpass filter

    new_sr = find_undersample_fs(ws) # if new_sr, an undersampling frequency was found

    wp = wp / (sr/2)
    ws = ws / (sr/2)
    
    y_filt = filter_bandpass(y, wp, ws, sr) # bandpass filter the signal
    
    if not new_sr: # if undersampling is not possible, perform ringmod + lpf
        new_sr = (ws[1] - ws[0] + 100/(sr/2)) * sr
        print("ring mod + lpf")
        return filter_lowpass(ring_mod(y_filt, ws[0]*(sr/2), sr), new_sr/2 - 100, sr), new_sr, ws[0]*(sr/2), inverted         
        
    print("undersampling")
    new_freq_range = treat_undersampling(new_sr[0], new_sr[1], ws*(sr/2)) # where will the frequency band of interest be shifted to?
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
    
def find_undersample_fs(freq_range):

#   returns 0 if undersampling is not possible
#   returns a new sampling rate if undersampling is possible

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

#   Returns the frequency band where the original frequency band of interest will be located in the
#   undersampled signal

    print(freq_range)
    if n_parity == 0:
        new_fl = np.ceil(freq_range[1]/undersample_freq) * undersample_freq - freq_range[1]
        new_fh = new_fl + freq_range[1] - freq_range[0]
    else:
        new_fl = freq_range[0] - np.floor(freq_range[0]/undersample_freq) * undersample_freq
        new_fh = new_fl + freq_range[1] - freq_range[0]
    return [new_fl, new_fh]

def ring_mod(y, freq, sr):
    t_final = len(y) / sr
    t = np.linspace(0,t_final, int(sr*t_final))
    x = np.cos(2*np.pi*freq*t)
    return x*y

def subsample_signal(y, new_sr, sr):
    subsample_step = int(np.ceil(sr/new_sr)) # take 1 out of every subsample_step samples of y
    return y[::subsample_step], sr/subsample_step  # subsampled signal, new_sr

def analyze_slice(y, new_sr, original_resolution, k=2):
    # Returns the STFT matrix of y in the freq_range x time_range, with a frequency resolution
    # determined by the factor k: new_resolution = k * original_resolution
    new_resolution = original_resolution / k
    window_size = int(new_sr / new_resolution)
    hop_size = window_size // 4
    return np.abs(librosa.stft(y, n_fft=window_size, hop_length=hop_size)), window_size, hop_size

def unmirror(stft_zoom, y_axis, freq_range):
#   Unmirror spectrum originally mirrored by the undersampling process
    i_start = np.searchsorted(y_axis, freq_range[0])
    i_stop  = np.searchsorted(y_axis, freq_range[1]) + 1
    stft_zoom[i_start:i_stop, :] = stft_zoom[i_start:i_stop, :][::-1]
    return stft_zoom

def get_axes_values(sr, f_min, time_range, spec_shape):
    x_axis = np.linspace(time_range[0], time_range[1], spec_shape[1])
    f_max = f_min + (sr / 2)
    y_axis = np.linspace(f_min, f_max, spec_shape[0])
    return x_axis, y_axis

def stft_zoom(y, freq_range, time_range, sr=44100, original_window_size=2048, k=2):
    # Returns an STFT representation of the interval freq_range x time_range of signal y with
    # its frequency resolution determined by the factor k

    inverted = False # suppose that the spectrum is not inverted to begin with (it could be if undersampling is performed)
    y_mod, new_sr, f_min, inverted = filter_and_mod(slice_signal(y, time_range, sr), freq_range, sr)
    y_sub, new_sr = subsample_signal(y_mod, new_sr, sr)

    original_resolution = sr / original_window_size
    stft_zoom, new_window_size, new_hop_size = analyze_slice(y_sub, new_sr, original_resolution, k=k)
    
    if type(f_min) is list: # undersampling inverted the spectrum between f_min[0] and f_min[1]
        ws = f_min[0]
        new_freq_range = f_min[1]        
        f_min = ws[0] - new_freq_range[0]
        x_axis, y_axis = get_axes_values(new_sr, f_min, time_range, stft_zoom.shape)
        stft_zoom = unmirror(stft_zoom, y_axis, ws)        
    else:
        x_axis, y_axis = get_axes_values(new_sr, f_min, time_range, stft_zoom.shape)

    # Slice the spectrogram in order to represent only the specified frequency range
    # ("guard bands" are used in the bandpass and lowpass filters in order to not distort the frequency
    # band specified)
    y_start = find_nearest(y_axis, freq_range[0])
    y_end   = find_nearest(y_axis, freq_range[1])

    # stft matrix, x axis, y axis, new sampling rate, window size and hop size used in this new STFT
    return stft_zoom[y_start:y_end,:], x_axis, y_axis[y_start:y_end], new_sr, new_window_size, new_hop_size



# # Returns 0 if it is not possible to perform "simple" subsampling
# # Returns the new sampling rate otherwise
# def check_subsample(sr, ws):
#     M_upperlim = int(ws[0] // (ws[1] - ws[0]))
#     if M_upperlim < 1:
#         return False
#     for M in range(M_upperlim, 0, -1):
#         if test_new_sr(2 * ws[0] / M, sr):
#             return (2 * ws[0] / M)
#     return False  # não é possível fazer o "aliasing inteligente"
                             
        
# Testa se a nova taxa de amostragem pode ser obtida 
# com uma subamostragem simples (selecionado 1 a cada X amostras de y)
# def test_new_sr(new_sr, sr):
#     if int(sr % new_sr) == 0:
#         return True
#     else:
#         return False

# def closest_alpha(possible_alpha):
#     idx = np.searchsorted(alpha_list, possible_alpha, side="left")
#     if alpha_list[idx] != possible_alpha:
#         return alpha_list[idx-1]
#     else:
#         return possible_alpha