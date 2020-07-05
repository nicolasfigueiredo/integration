from scipy.signal import hamming
import numpy as np

def hamming_2D(kernel_dimensions, asymm=False):
    """
    Build a symmetric or asymmetric (right half zeroed) 2D Hamming 
    window according to the kernel dimensions
    """
    f = hamming(kernel_dimensions[0])
    if asymm:
        t = hamming((kernel_dimensions[1]*2-1))
    else:
        t = hamming(kernel_dimensions[1])
    window = (np.outer(f,t))
    return window[:,:kernel_dimensions[1]]