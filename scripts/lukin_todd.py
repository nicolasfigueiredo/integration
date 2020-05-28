import scipy.signal
import librosa
import numpy as np
import PIL

def lukin_todd_main_no_shift(y, n, kernel):
    specs = get_n_spectrograms(y, n=n)
    entropies = get_entropies(specs, kernel)
    return lukin_todd(specs, kernel, entropies)

def lukin_todd_main_shift(y, n, kernel):
    specs = get_n_spectrograms_shifted(y, n=n)
    entropies = get_entropies(specs, kernel)
    return lukin_todd(specs, kernel, entropies)

def lukin_todd(specs, kernel, entropies):
    multires_spec = np.zeros(specs[-1][0].shape)
    multires_spec = PIL.Image.fromarray(multires_spec)

    # res_map = np.zeros(specs[-1][0].shape)
    # res_map = PIL.Image.fromarray(res_map)

    shape = [256 // kernel[0], 3446//kernel[1]]

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            entropies_local = []
            for en in entropies:
                entropies_local.append(en[i,j])
            chosen_res = np.array(entropies_local).argmin()

            [y_start_cs, y_stop_cs], [x_start_cs, x_stop_cs] = get_range([i,j], kernel, specs[chosen_res][1], specs[chosen_res][2])
            chosen_res_subregion = specs[chosen_res][0][y_start_cs:y_stop_cs, x_start_cs:x_stop_cs]

            [y_start_mr, y_stop_mr], [x_start_mr, x_stop_mr] = get_range([i,j], kernel, specs[-1][1], specs[-1][2])
            chosen_res_subregion = PIL.Image.fromarray(chosen_res_subregion).resize((x_stop_mr - x_start_mr,y_stop_mr - y_start_mr))

            # res_map_subregion = PIL.Image.fromarray(np.ones([y_stop_mr - y_start_mr, x_stop_mr - x_start_mr]) * (chosen_res+1))
            
            box = (x_start_mr, y_start_mr, x_stop_mr, y_stop_mr)
            multires_spec.paste(chosen_res_subregion, box)
            # res_map.paste(res_map_subregion, box)

    return np.asarray(multires_spec)#, np.asarray(res_map) 

def half_bin_shift(y, window_size, sr):
    freq_shift = sr / (window_size * 2)
    y_shift = np.zeros(len(y), dtype='complex')
    for n in range(len(y)):
        y_shift[n] = np.exp(2*np.pi * 1j * n * freq_shift / sr ) * y[n]
    return y_shift

def calc_smearing(bins):
    bins_sorted = np.sort(bins, axis=None)[::-1]
    mean = np.dot(bins_sorted, np.array(range(len(bins_sorted)))+1)
    normalize = np.sqrt(np.size(bins) * np.max(bins)) * ((np.size(bins) + 1) / 2)
#     return mean / (np.sqrt(np.sum(bins)) * normalize + 0.00001)
    return mean / (normalize + 0.00001)

def calc_entropy(spectrogram, kernel_dimensions, n_fft=2048, hop_size=512, sr=44100):
    # kernel given in [freq. bins of 512 hz window res, time frames of 128 hop size res]
    
    delta_t = int(128 / hop_size * kernel_dimensions[1]) 
    delta_f = int((n_fft / 512) * kernel_dimensions[0])

    mapping = np.zeros([256 // kernel_dimensions[0], 3446//kernel_dimensions[1]])

    j = 0
    j_map = 0
    
    i = 0
    i_map = 0

    while j_map < mapping.shape[1]:
        while i_map < mapping.shape[0]:
            subregion = spectrogram[i:i+delta_f, j:j+delta_t]
#             mapping[i_map, j_map] = mappings.shannon_entropy(subregion)
            mapping[i_map, j_map] = calc_smearing(subregion)

            i += delta_f
            i_map += 1
            
        j += delta_t
        j_map += 1
        i_map = 0
        i = 0
    
    return mapping

# Get n spectrograms with a half-bin frequency shift
def get_n_spectrograms_shifted(y, n=3):
    n_ffts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    spec_list = []
    for i in range(n):
        n_fft = n_ffts[i]
        hop_size = n_fft // 4
        y_shifted = half_bin_shift(y, n_fft, 44100)
        _, _, spec = scipy.signal.stft(y_shifted, fs=44100, nperseg=n_fft, noverlap=n_fft-hop_size)
        spec = spec[:spec.shape[0]//2+1,:]
        spec_list.append([np.abs(spec)**2, n_fft, hop_size])
        
    return spec_list

def get_n_spectrograms(y, n=3):
    n_ffts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    spec_list = []
    for i in range(n):
        n_fft = n_ffts[i]
        hop_size = n_fft // 4
        spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_size)
        spec_list.append([np.abs(spec)**2, n_fft, hop_size])
        
    return spec_list


def get_entropies(specs, kernel):
    entropies = []
    for spec in specs:
        entropies.append(calc_entropy(spec[0], kernel, n_fft=spec[1], hop_size=spec[2]))
    return entropies

def get_entropies_2(specs, kernel, idx):
    entropies = []
    for spec in specs:
        [y_start, y_stop], [x_start, x_stop] = get_range([idx[0],idx[1]], kernel, spec[1], spec[2])
        entropies.append(calc_smearing(spec[0][y_start:y_stop, x_start:x_stop]))
    return entropies

def get_range(idx, kernel, n_fft, hop_size):
    # transforms from feature idx to stft idx
    delta_t = int(128 / hop_size * kernel[1]) 
    delta_f = int((n_fft / 512) * kernel[0])
    
    i = idx[0]
    j = idx[1]
    return [i*delta_f, (i+1)*delta_f], [j*delta_t, (j+1)*delta_t]