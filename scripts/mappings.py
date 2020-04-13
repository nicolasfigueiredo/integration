import numpy as np
import scipy.stats

# This script contains functions that perform feature extraction from a spectrogram.
# The main function is calc_map_aug2(), that subdivides a spectrogram into subregions of
# dimensions given by cents x ms, and then applies a function to each subregion (mean, std deviation, entropy)

def renyi_entropy(tfp_region, alpha=3):  # without normalizaton factor
    hist = np.histogram(tfp_region, bins=10, density=True)[0]
    return (1/(1-alpha)) * np.log2(np.sum(hist ** alpha))
    
def shannon_entropy(tfp_region):
    hist = np.histogram(tfp_region, bins=10, density=True)[0]
    return scipy.stats.entropy(hist)

def find_freq_list(fft_freqs, delta_f_c):
    # Returns the frequency list that deetermines the musical interval in cents
    # Ex: between fft_freqs[idx_list[i]] and fft_freqs[idx_list[i+1]] there is an interval of delta_f_c cents
    idx_list = [0]
    freq_step = fft_freqs[1] - fft_freqs[0]
    
    if fft_freqs[0] < 0.0001:   # then fft_freqs[0] == 0, estamos analisando o espectrograma inteiro
        f1 = fft_freqs[1]
    else:
        f1 = fft_freqs[0]
    
    f2 = f1 * 2 ** (delta_f_c/1200) 
    idx_f1 = 0
    idx_f2 = int(np.ceil((f2 - f1)/freq_step))
        
    while idx_f2 < len(fft_freqs):
        idx_list.append(idx_f2)
        idx_f1 = idx_f2
        f1 = fft_freqs[idx_f1]
        f2 = f1 * 2 ** (delta_f_c/1200)  
        idx_f2 = idx_f1 + int(np.ceil((f2 - f1)/freq_step))
        
    return idx_list

def calc_map_aug2(spectrogram, kernel_dimensions, n_fft=2048, hop_size=512, sr=44100, type='dp', alpha=3, fft_freqs=None):
    # calc_map_aug2() divides a spectrogram into subregions of dimensions given by kernel_dimensions, where:
    #   kernel_dimensions[0] is given in ms;
    #   kernel_dimensions[1] is given in cents, and therefore is not linear in frequency.
    # After this division, a mapping is applied for each subregion according to the 'type' keyword, and the
    # resulting matrix is returned.

    if fft_freqs is None:
        idx_list = find_freq_list(fft_frequencies(sr=sr, n_fft=n_fft), kernel_dimensions[1]) # essa linha: 
    else:
        idx_list = find_freq_list(fft_freqs, kernel_dimensions[1]) # para calcular mapa de regiões refinadas, já passamos o eixo de frequências     

    delta_t_ms = kernel_dimensions[0]   # dimensão em ms

    # ms_per_frame = (n_fft+hop_size) * 1000 / (sr*2)   # discussão
    ms_per_frame = hop_size * 1000 / sr
    delta_t = int(np.round(delta_t_ms / ms_per_frame))

    mapping = np.zeros([len(idx_list)-1, spectrogram.shape[1]//delta_t])
    
    j = 0
    j_map = 0

    while j < spectrogram.shape[1] - delta_t:
        for i_map in range(len(idx_list)-1):
            subregion = spectrogram[idx_list[i_map]:idx_list[i_map+1], j:j+delta_t]
            if type=='std dev':
                mapping[i_map, j_map] = np.std(subregion)
            elif type=='var':
                mapping[i_map, j_map] = np.var(subregion)
            if type=='dp':
                mapping[i_map, j_map] = np.sqrt(np.var(subregion))
            elif type=='avg':
                mapping[i_map, j_map] = np.mean(subregion)
            elif type=='shannon':
                mapping[i_map, j_map] = shannon_entropy(subregion)
            elif type=='renyi':
                mapping[i_map, j_map] = renyi_entropy(subregion, alpha=alpha)
            elif type=='maxmin':
                mapping[i_map, j_map] = np.max(subregion) - np.min(subregion)
        j += delta_t
        j_map += 1
    
    return mapping

# helper function taken from librosa
def fft_frequencies(sr=44100, n_fft=2048):
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
    endpoint=True)



## DEPRECATED
# def calc_map_aug(spectrogram, kernel_dimensions, n_fft=2048, hop_size=512, sr=44100, type='dp', alpha=3):
#     mapping = np.zeros(spectrogram.shape)
#     delta_t_ms = kernel_dimensions[0]   # dimensão em ms
#     delta_f_c = kernel_dimensions[1]   # dimensão em cents

#     ms_per_frame = (n_fft+hop_size) * 1000 / (sr*2)   # discussão
#     delta_t = int(np.ceil(delta_t_ms / ms_per_frame))
    
#     fft_freq = fft_frequencies(sr=sr, n_fft=n_fft)
#     fft_freq_step = fft_freq[1] # intervalo em Hz entre bins do spec
   
    
#     j = 0

#     while j < spectrogram.shape[1] - delta_t:
#         f1 = fft_freq[1] # começar a contar intervalo musical do 1o bin (n podemos começar de 0 hz)
#         f2 = f1 * 2 ** (delta_f_c/1200) 
#         # print(f2)
#         idx_f1 = 0
#         idx_f2 = int(np.ceil((f2)/fft_freq_step))
 
#         while idx_f2 < spectrogram.shape[0]:
#             subregion = spectrogram[idx_f1:idx_f2, j:j+delta_t]
#             if type=='dp':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = np.sqrt(np.var(subregion))
#             elif type=='avg':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = np.mean(subregion)
#             elif type=='shannon':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = shannon_entropy(subregion)
#             elif type=='renyi':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = renyi_entropy(subregion, alpha=alpha, n_fft=n_fft, sr=sr, hop_size=hop_size)

#             idx_f1 = idx_f2
#             f1 = fft_freq[idx_f1]
#             f2 = f1 * 2 ** (delta_f_c/1200)  
#             idx_f2 = idx_f1 + int(np.ceil((f2 - f1)/fft_freq_step))
#             # print(f2)
#         j += delta_t
    
#     return mapping


# def calc_var_map(spectrogram, kernel_dimensions):
#     var_map = np.zeros(spectrogram.shape)
#     delta_x = kernel_dimensions[0]
#     delta_y = kernel_dimensions[1]

#     i = 0
#     j = 0

#     while i < spectrogram.shape[0] - delta_x:
#         while j < spectrogram.shape[1] - delta_y:
#             subregion = spectrogram[i:i+delta_x, j:j+delta_y] 
#             var_map[i:i+delta_x, j:j+delta_y] = np.var(subregion) 
#             j += delta_y
#         j = 0
#         i += delta_x
    
#     return var_map

# # Calcula densidade de eventos por região retangular do mapa de eventos
# def calc_event_density(tfp_pianoroll, kernel_dimensions):
#     # tfp_pianoroll = (tfp_pianoroll > 0).astype(int)  # transforma em matriz binária (pode ser que eventos 
#                                                      # tenham sido somados ao construir o mapa de eventos (queremos isso?))
#     density = np.zeros(tfp_pianoroll.shape)

#     delta_x = kernel_dimensions[0]
#     delta_y = kernel_dimensions[1]

#     size_subreg = delta_x * delta_y

#     i = 0
#     j = 0

#     while i < tfp_pianoroll.shape[0] - delta_x:
#         while j < tfp_pianoroll.shape[1] - delta_y:
#             subregion = tfp_pianoroll[i:i+delta_x, j:j+delta_y] 
#             density[i:i+delta_x, j:j+delta_y] = np.sum(subregion.flatten())
#             j += delta_y
#         j = 0
#         i += delta_x
    
#     return density / size_subreg

# def calc_map(spectrogram, kernel_dimensions, type='dp'):
#     mapping = np.zeros(spectrogram.shape)
#     delta_x = kernel_dimensions[0]
#     delta_y = kernel_dimensions[1]

#     i = 0
#     j = 0

#     while i < spectrogram.shape[0] - delta_x:
#         while j < spectrogram.shape[1] - delta_y:
#             subregion = spectrogram[i:i+delta_x, j:j+delta_y]
#             if type=='dp':
#                 mapping[i:i+delta_x, j:j+delta_y] = np.sqrt(np.var(subregion))
#             elif type=='avg':
#                 mapping[i:i+delta_x, j:j+delta_y] = np.mean(subregion)
#             j += delta_y
#         j = 0
#         i += delta_x
    
#     return mapping
