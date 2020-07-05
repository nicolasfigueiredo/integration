import numpy as np
import sparsity

def lukin_todd(specs, kernel_size, eta=1):
    """
    Gets a list of spectrograms (normalized, interpolated) and combines them
    according to the Lukin & Todd method
    """
    window = np.ones(kernel_size)
    sparsities = sparsity.get_sparsities(specs, window, mode='lukin todd')
    multires_spec = np.sum(specs * (sparsities ** eta) / np.sum((sparsities ** eta)), axis=0)
    return multires_spec