# -*- coding: utf-8 -*-
"""
Fourier series test signals, 
use as 'import FourierTestSignal as FTS'
"""

import numpy as np

class FourierTestSignal:
    """Provides a fourier signal with a finite number of frequencies"""
    def __init__(self, ampl_array, freq_array, length=1024):
        self.ampl_array = ampl_array
        self.freq_array = freq_array
        self.length = length
        self.n = np.arange(self.length, dtype=np.int32)
    
    def compute_signal(self):
        """Computes the fourier series for current settings"""
        signal = np.zeros(self.length, dtype=np.complex128)
        for i in range(len(self.freq_array)):
            phase = np.exp(2*np.pi*1j * self.freq_array[i] * self.n)
            signal += self.ampl_array[i] * phase
        return signal
    
    def add_freq(self, freq, ampl):
        """Add a frequency with ampltiude"""
        self.freq_array.append(freq)
        self.ampl_array.append(ampl)
        
    def remove_freq(self, freq, n_min=1, n_max=1, thresh=1e-7):
        """Remove frequencies from the signal (n_harm = n_max - n_min)"""
        freq_array = np.array(self.freq_array)
        indx = np.ones(freq_array.shape[0])
        for i in range(n_min, n_max, 1):
            remove_indx = (np.abs(freq_array - i * freq) < thresh)
            indx = np.logical_and(indx, remove_indx)
        self.freq_array = self.freq_array[indx]
        
    def add_harmonics(self, freq_indx, ampl_array):
        """Add higher harmonics of a frequency with varying amplitudes"""
        n_harm = len(ampl_array)
        freq = self.freq_array[freq_indx]
        higher_harmonics = [i*freq for i in range(2, n_harm + 2, 1)]
        self.freq_array.extend(higher_harmonics)
        self.ampl_array.extend(ampl_array)
        
    
