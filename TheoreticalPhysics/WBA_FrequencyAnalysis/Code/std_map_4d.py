#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides 4D standard maps (coupled kicked rotor) for different conventions.
"""

import numpy as np
from numba import njit, prange

@njit 
def map4dCyl(p10, p20, q10, q20, N, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    length = len(p10)
    points = np.zeros((length, 4, N), dtype=np.float64)
    points[:, 0, 0] = p10
    points[:, 1, 0] = p20
    points[:, 2, 0] = q10
    points[:, 3, 0] = q20
    for i in range(1, N, 1):
        points[:, 2, i] = (points[:, 2, i-1] + points[:, 0, i-1]) % 1.0
        points[:, 3, i] = (points[:, 3, i-1] + points[:, 1, i-1]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (points[:, 2, i] + points[:, 3, i]))
        points[:, 0, i] = (points[:, 0, i-1] + coupling + k1_2pi * 
                            np.sin(my2pi * points[:, 2, i]))
        points[:, 1, i] = (points[:, 1, i-1] + coupling + k2_2pi * 
                            np.sin(my2pi * points[:, 3, i]))
    return points


@njit
def map4d(initial, N, k1, k2, k):
    """
    initial == [p10, p20, q10, q20]
    orbit   == [p1, p2, q1, q2], ( shape == (4, N) )
    Warning :: my2pi == 2*np.pi, k1_2pi == k1 / (2*np.pi), etc. !!
    """
    my2pi = 2*np.pi
    k1_2pi = k1 / my2pi
    k2_2pi = k2 / my2pi
    k_2pi = k / my2pi

    orbit = np.zeros((4, N), dtype=np.float64)
    orbit[:, 0] = initial
    for i in range(1, N, 1):
        orbit[2, i] = (orbit[2, i-1] + orbit[0, i-1]) % 1.0
        orbit[3, i] = (orbit[3, i-1] + orbit[1, i-1]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (orbit[2, i] + orbit[3, i]))
        orbit[0, i] = ((orbit[0, i-1] + k1_2pi * np.sin(my2pi * orbit[2, i]) 
                        + coupling) + 0.5 ) % 1.0 - 0.5
        orbit[1, i] = ((orbit[1, i-1] + k2_2pi * np.sin(my2pi * orbit[3, i]) 
                        + coupling) + 0.5 ) % 1.0 - 0.5
    return orbit

@njit
def map4dCylSingle(initial, N, my2pi, k1_2pi, k2_2pi, k_2pi):
    """
    initial == [p10, p20, q10, q20]
    orbit   == [p1, p2, q1, q2]
    Warning :: my2pi == 2*np.pi, k1_2pi == k1 / (2*np.pi), etc. !!
    """
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

@njit(parallel=True)
def map4dCylParallel(p10, p20, q10, q20, N, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    N_samples = p10.shape[0]
    points = np.zeros((N_samples, 4, N), dtype=np.float64)
    points[:, 0, 0] = p10
    points[:, 1, 0] = p20
    points[:, 2, 0] = q10
    points[:, 3, 0] = q20
    for i in prange(N_samples):
        # print(points[i,:,0])
        points[i, :, :] = map4dCylSingle(points[i, :, 0], N, 
                                         my2pi, k1_2pi, k2_2pi, k_2pi)
    return points

@njit
def _cmap4dCylSingle(initial, N, my2pi, k1_2pi, k2_2pi, k_2pi):
    """Warning :: my2pi == 2*np.pi, k1_2pi == k1 / (2*np.pi), etc. !!"""
    orbit = np.zeros((2, N), dtype=np.complex128)
    orbit[:, 0] = initial
    for i in range(1, N, 1):
        orbit[0].imag[i] = (orbit[0].imag[i-1] + orbit[0].real[i-1]) % 1.0
        orbit[1].imag[i] = (orbit[1].imag[i-1] + orbit[1].real[i-1]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (orbit[0].imag[i] 
                                            + orbit[1].imag[i]))
        orbit[0].real[i] = (orbit[0].real[i-1] 
                            + k1_2pi * np.sin(my2pi * orbit[0].imag[i]) 
                            + coupling)
        orbit[1].real[i] = (orbit[1].real[i-1] 
                            + k2_2pi * np.sin(my2pi * orbit[1].imag[i]) 
                            + coupling)
    return orbit

@njit(parallel=True)
def cmap4dCylParallel(p10, p20, q10, q20, N, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    N_samples = p10.shape[0]
    points = np.zeros((N_samples, 2, N), dtype=np.complex128)
    points[:, 0, 0] = p10 + 1j*q10
    points[:, 1, 0] = p20 + 1j*q20
    for i in prange(N_samples):
        points[i, :, :] = _cmap4dCylSingle(points[i, :, 0], N, 
                                           my2pi, k1_2pi, k2_2pi, k_2pi)
    return points

def map4dCylSamples(N_samples=10, bbox=[[-0.25, 0.25], [-0.25, 0.25], 
                                        [0.25, 0.75], [0.25, 0.75]],
                    N=2**12, k1=2.25, k2=3.0, k=1.0,
                    mapping=map4dCylParallel):
    """
    'N_samples' signals with initial conditions uniformly smapled form 'bbox'
    """
    p10 = np.random.uniform(bbox[0][0], bbox[0][1], size=N_samples)
    p20 = np.random.uniform(bbox[1][0], bbox[1][1], size=N_samples)
    q10 = np.random.uniform(bbox[2][0], bbox[2][1], size=N_samples)
    q20 = np.random.uniform(bbox[3][0], bbox[3][1], size=N_samples)
    
    return mapping(p10, p20, q10, q20, N, k1, k2, k)

if __name__ == "__main__":
    print(__doc__)
    points = map4dCylSamples(N_samples=100, mapping=cmap4dCylParallel)
