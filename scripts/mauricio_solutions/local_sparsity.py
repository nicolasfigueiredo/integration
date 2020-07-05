import numpy as np
import util_m
import sparsity
from hamming_2D import hamming_2D

def calc_energy_ratio(multires_spec, specs, window):
    kernel_size = window.shape
    energy_ratio = np.zeros(multires_spec.shape)
    
    guard_cols = kernel_size[1] - 1
    guard_rows = int(np.floor(kernel_size[0] / 2))
    
    for i in range(guard_rows, multires_spec.shape[0] - guard_rows):
        for j in range(guard_cols, multires_spec.shape[1] - guard_cols):
            subregions_specs = specs[:,i-guard_rows:i+guard_rows+1, j-guard_cols:j+1]
            subregions_specs_windowed = np.array([x * window for x in subregions_specs])
            subregion_mrspec = multires_spec[i-guard_rows:i+guard_rows+1, j-guard_cols:j+1] * window
            energy_ratio[i,j] = np.min(np.sum(subregions_specs_windowed, axis=0)) / np.sum(subregion_mrspec)
            
    return energy_ratio

def local_sparsity(specs, kernel_size_analysis, kernel_size_energy_comp):
    window_a = hamming_2D(kernel_size_analysis)
    sparsities = sparsity.get_sparsities(specs, window_a, mode='local sparsity')
    idx = np.argmax(sparsities, axis=0)
    multires_spec = np.zeros(specs[0].shape)

    for i in range(multires_spec.shape[0]):         # jeito mais bonito de fazer isso?
        for j in range(multires_spec.shape[1]):     # algo do tipo multires_spec = specs[idx]
            multires_spec[i,j] = specs[idx[i,j]][i,j]

    window_en = hamming_2D(kernel_size_energy_comp, asymm=True)
    energy_ratio = calc_energy_ratio(multires_spec, specs, window_en)

    return multires_spec * energy_ratio

def get_sparsity_tensor(sparsities, zeta=50):
    tensor = []
    i = 0
    for sparsity in sparsities:
        tensor.append((sparsity / (np.prod(util_m.get_other_specs(sparsities, i), axis=0)+0.00000000001)) ** zeta)
        i += 1
    return tensor

def get_energy_ratios(specs, window):
    energy_ratios = []
    for spec in specs:
        energy_ratios.append(calc_energy_ratio(spec, specs, window))
    return energy_ratios

def smoothed_local_sparsity(specs, kernel_size_analysis, kernel_size_energy_comp):
    window_a = hamming_2D(kernel_size_analysis)
    sparsities = sparsity.get_sparsities(specs, window_a, mode='local sparsity')

    sparsity_tensor = get_sparsity_tensor(sparsities)
    window_en = hamming_2D(kernel_size_energy_comp, asymm=True)
    energy_ratios = get_energy_ratios(specs, window_en)
    multires_spec_SLS = np.sum(specs * sparsity_tensor * energy_ratios, axis=0)/ (np.sum(sparsity_tensor, axis=0)+0.000000001)
    return multires_spec_SLS