#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the frequency space of a 4d standard map with NaffND-gauss
"""

from NaffND import NaffND
import std_map_4d
import numpy as np
import time

def computeFreqSpace(N_samples, bbox=[[-0.25, 0.25], [-0.25, 0.25], 
                                      [0.25, 0.75], [0.25, 0.75]], 
                     N=2**10, k1=2.25, k2=3.0, k=1.0,
                     alpha=140, J=23, component=0, peak_tolerance=1e-6,
                     Offset=True):
    """Compute the frequency space using the NaffND gauss-variant"""
    if N_samples > 50000:
        raise RuntimeWarning("Too many samples, avoiding runtime error...")
    
    t1 = time.perf_counter()
    p10 = np.random.uniform(bbox[0][0], bbox[0][1], size=N_samples)
    p20 = np.random.uniform(bbox[1][0], bbox[1][1], size=N_samples)
    q10 = np.random.uniform(bbox[2][0], bbox[2][1], size=N_samples)
    q20 = np.random.uniform(bbox[3][0], bbox[3][1], size=N_samples)
    
    t2 = time.perf_counter()
    print(f"Random initial conditions generated in {t2-t1} s.")
    mapping = std_map_4d.map4dCylParallel
    signals = mapping(p10, p20, q10, q20, N=2*N, k1=k1, k2=k2, k=k)
    
    t3o = time.perf_counter()
    print(f"Signals computed in {t3o-t2} s.")
    # signals[:, 2:, :] -= 0.5
    
    t3 = time.perf_counter()
    # print(f"Signal offset corrected in {t3-t3o} s.")
    # print(signals.shape)
    # print(np.mean(np.mean(signals, axis=0), axis=1))
    
    c_signals = np.zeros((N_samples, 2, 2*N), dtype=np.complex128)
    c_signals[:, 0, :] = signals[:, 0, :] + 1j*signals[:, 2, :]
    c_signals[:, 1, :] = signals[:, 1, :] + 1j*signals[:, 3, :]
    t4 = time.perf_counter()
    print(f"Signals transformed in {t4-t3} s.")
    
    naff_nd1 = NaffND(c_signals[:,:,:N], n_freq=2, component=component, 
                      Offset=Offset, alpha=alpha, J=J, maxComponents=2)
    naff_nd2 = NaffND(c_signals[:,:,N:], n_freq=2, component=component, 
                      Offset=Offset, alpha=alpha, J=J, maxComponents=2)
    t5 = time.perf_counter()
    print(f"NaffND and FFT setup in {t5-t4} s.")
    naff_nd1.signals = 0
    naff_nd2.signals = 0
    naff_nd1.compute(peak_tolerance=peak_tolerance)
    naff_nd2.compute(peak_tolerance=peak_tolerance)
    t6 = time.perf_counter()
    print(f"Frequency computation in {t6-t5} s.")
    return naff_nd1, naff_nd2
    
def computeFreqSpaceCMap(N_samples, bbox=[[-0.25, 0.25], [-0.25, 0.25], 
                                      [0.25, 0.75], [0.25, 0.75]], 
                         N=2**10, k1=2.25, k2=3.0, k=1.0,
                         alpha=140, J=23, component=0, peak_tolerance=1e-6,
                         Offset=True):
    """Uses a special complex 4d-standard-map variant"""
    if N_samples > 50000:
        raise RuntimeWarning("Too many samples, avoiding runtime error...")
    
    t3 = time.perf_counter()
    mapping = std_map_4d.map4dCylSamples
    c_signals = mapping(N_samples, bbox, N=2*N, k1=k1, k2=k2, k=k,
                        mapping=std_map_4d.cmap4dCylParallel)
    t4 = time.perf_counter()
    print(f"Signals computed in {t4-t3} s.")
    
    naff_nd1 = NaffND(c_signals[:,:,:N], n_freq=2, component=component, 
                      Offset=Offset, alpha=alpha, J=J, maxComponents=2)
    naff_nd2 = NaffND(c_signals[:,:,N:], n_freq=2, component=component, 
                      Offset=Offset, alpha=alpha, J=J, maxComponents=2)
    t5 = time.perf_counter()
    print(f"NaffND and FFT setup in {t5-t4} s.")
    
    naff_nd1.compute(peak_tolerance=peak_tolerance)
    naff_nd2.compute(peak_tolerance=peak_tolerance)
    t6 = time.perf_counter()
    print(f"Frequency computation in {t6-t5} s.")
    return naff_nd1, naff_nd2

if __name__ == "__main__":
    print(__doc__)
    N_samples = 50000   # 7.2 hours runtime estiamte for 1e8 samples
    naff_nd1, naff_nd2 = computeFreqSpaceCMap(N_samples, N=4096)
    nu1part1, nu2part1 = naff_nd1.freq[:, 0], naff_nd1.freq[:, 1]
    nu1part2, nu2part2 = naff_nd2.freq[:, 0], naff_nd2.freq[:, 1]
    abs_diff = np.abs(nu1part1 - nu1part2)
    indx_thresh = (abs_diff < 1e-5)
    print(f"There are {np.sum(indx_thresh)} of {N_samples} frequencies below "
          + "threshold...")
    nu1, nu2 = nu1part1[indx_thresh], nu2part1[indx_thresh]
    indx_swap = (nu2 > nu1)
    nu1[indx_swap], nu2[indx_swap] = nu2[indx_swap], nu1[indx_swap]
    
    import matplotlib.pyplot as plt
    plt.plot(nu1, nu2, ls='', marker='o', ms=1, mew=1)
    
    """
ixsum1 = (np.abs(naff_nd1.freq[:,0]+naff_nd1.freq[:,1]-1) < 1e-5)
sum1orbits = naff_nd1.signals[ixsum1]
orbit = sum1orbits[0]
from Naff_var import naffnd_gauss, _remove_peak
n_freq=2;alpha=140;J=24;ReturnCoeff=True;Offset=True
def plot(orbit):
    plt.plot(orbit[0].real, orbit[0].imag, ls='', marker='o', ms=2, mew=1)
def plotreal(orbit):
    plt.plot(orbit[0], orbit[2], ls='', marker='o', ms=2, mew=1)
# [plot(orb1) for orb1 in sum1orbits[[1,4,8]]]

sum1 = [np.sum(orb1) for orb1 in sum1orbits]
regsum1 = (np.abs(np.array(sum1).real) < 1)
regsum1orbits = sum1orbits[regsum1]

np.array([0.1029953412704615]), np.array([-0.14460334923023588]), 
np.array([0.23807701401574216]), np.array([0.04652520959549544])    

np.array([0.16449112131839355]), np.array([-0.13491117751077086]),
np.array([0.06448871959677521]), np.array([-0.06788514071934743])

np.array([-0.07292519441012574]), np.array([0.10659305037910616]),
np.array([0.06553640797722976]), np.array([-0.10658947988110001])

np.array([-0.0726653224253282]), np.array([0.07878552791590221]),
np.array([0.08945230297581214]), np.array([-0.11364589970434186])

np.array([0.04283221, 0.02016283, 0.43395967, 0.45931292])

k1, k2, k = 2.25, 3.0, 1.0
my2pi = 2*np.pi
k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
def _map4dCylSingle(initial, N, my2pi, k1_2pi, k2_2pi, k_2pi):
    orbit = np.zeros((4, N), dtype=np.float64)
    orbit[:, 0] = initial
    for i in range(1, N, 1):
        orbit[2, i] = (orbit[2, i-1] + orbit[0, i-1]) % 1.0
        orbit[3, i] = (orbit[3, i-1] + orbit[1, i-1]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (orbit[2, i] + orbit[3, i]))
        orbit[0, i] = (orbit[0, i-1] + k1_2pi * np.sin(my2pi * orbit[2, i]) 
                       + coupling)
        orbit[1, i] = (orbit[1, i-1] + k2_2pi * np.sin(my2pi * orbit[3, i]) 
                       + coupling)
    return orbit

orb3 = _map4dCylSingle(np.array([regsum1orbits[0,0,0].real, 
                                 regsum1orbits[0,1,0].real, 
                                 regsum1orbits[0,0,0].imag, 
                                 regsum1orbits[0,1,0].imag]), 
                       4096, my2pi, k1_2pi, k2_2pi, k_2pi)
    """