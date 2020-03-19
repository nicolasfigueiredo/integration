import numpy as np
import librosa

def get_axes_values(sr, f_min, time_range, spec_shape):
    x_axis = np.linspace(time_range[0], time_range[1], spec_shape[1])
    f_max = f_min + (sr / 2)
    y_axis = np.linspace(f_min, f_max, spec_shape[0])
    return x_axis, y_axis

def get_audio(path):
	y, sr = librosa.load(path, sr=44100)
	return y[:sr*30]

def load_audio(path):
	y, sr = librosa.load(path, sr=44100)
	return y

def get_spectrogram(y, sr=44100):
	t_final = len(y)/sr
	D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512)), ref=np.max)
	return D, get_axes_values(sr, 0, [0, t_final], D.shape)

def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`
    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
    endpoint=True)