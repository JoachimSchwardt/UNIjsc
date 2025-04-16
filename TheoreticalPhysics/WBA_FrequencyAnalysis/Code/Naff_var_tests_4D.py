#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:49:27 2022

@author: joachim
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup()

from NaffND_cos import NaffND_cos, naffnd_cos
from std_map_4d import map4d

def naff_Nval(z, Nval, n_freq=2, ak=14, Offset=True):
    freqs = np.zeros((n_freq, Nval.shape[0]))
    for i in range(Nval.shape[0]):
        freq = naffnd_cos(z[:Nval[i]], n_freq, ak=ak, 
                          Offset=Offset, ReturnCoeff=False)
        freqs[:, i] = freq[:n_freq]
    return freqs

def fix_mirror_freqs(freqs):
    if np.ndim(freqs) < 2:
        freqs = np.expand_dims(freqs, axis=0)
    
    for i in range(freqs.shape[0]):
        f = freqs[i, -1]
        indx = (np.abs(freqs[i] - f) > np.abs(freqs[i] - 1 + f))
        freqs[i, indx] = 1 - freqs[i, indx]
    return freqs

def lbound(array, val=1e-16):
    array[array < val] = val
    return array

def axis_lim(array, minimum=0.0, maximum=1.0, scale=0.05):
    eps = scale * (maximum - minimum)
    minval, maxval = np.min(array), np.max(array)
    return np.max([minimum, minval - eps]), np.min([maximum, maxval + eps])

def main():
    Nmin = 128
    Nmax = 2**14
    NN = 50
    Nplot = 1024    # number of orbit points to plot
    
    Nval = (2**np.linspace(np.log2(Nmin), np.log2(Nmax), NN)).astype(int)
    N = Nval[-1]
    
    
    """
    k1,k2,k,p10,p20,q10,q20,proj
    0.7,0.5,0.1,0.05,0.05,0.5,0.5,z2    # second frequency not always detected
    """
    k1 = 0.7
    k2 = 0.5
    k = 0.1
    
    # order of 'init' and 'orbit' is p1, p2, q1, q2
    init = np.array([0.05, 0.04, 0.5, 0.5])
    orbit = map4d(init, N, k1, k2, k)       
    z1 = orbit[2] + 1j*orbit[0]      
    z2 = orbit[3] + 1j*orbit[1]
    
    # f_hann14 = fix_mirror_freqs(naff_Nval(z1, Nval, ak=14))
    f_hann14 = naff_Nval(z1, Nval, ak=14)
    err_hann14 = lbound(np.abs(f_hann14[:, :-1].T - f_hann14[:, -1]).T)
    
    f_hann1 = naff_Nval(z1, Nval, ak=5)
    err_hann1 = lbound(np.abs(f_hann1[:, :-1].T - f_hann1[:, -1]).T)
    
    
    c = special.Colors()
    Nval = Nval[:-1]
    fig, ax = plt.subplots(2, 2)
    for i in [0, 1]:
        # ax[i, 0].axis([0.0, 1.0, -0.5, 0.5])
        ax[i, 0].axis([*axis_lim(orbit[2+i]), *axis_lim(orbit[i], -0.5, 0.5)])
        ax[i, 0].set_xlabel(f'$q_{i+1}$')
        ax[i, 0].set_ylabel(f'$p_{i+1}$')
        ax[i, 0].plot(orbit[2+i, :Nplot], orbit[i, :Nplot], 
                      ls='', c=c.get_color(inc=0),
                      marker='o', ms=2, mew=0)
        
        ax[i, 1].set_xscale('log')
        ax[i, 1].set_yscale('log')
        ax[i, 1].set_xlim(Nval[0], Nval[-1])
        ax[i, 1].set_xlabel('N')
        # ax[i, 1].set_ylabel(r'$|\nu_N - \nu_{N_\mathrm{max}}|$')
        ax[i, 1].set_ylabel(r'$|\nu_N - \nu|$')
        ax[i, 1].plot(Nval, err_hann1[i], c=c.get_color(inc=0), ls='--', 
                      label='hann1')
        ax[i, 1].plot(Nval, err_hann14[i], c=c.get_color(inc=i), 
                      label='hann14')
        ax[i, 1].legend()
        
    plt.subplots_adjust(left=0.061, right=0.994, top=0.99, bottom=0.1, 
                        hspace=0.25, wspace=0.27)
    special.polish(fig, ax)    
    return 0

if __name__ == "__main__":
    print(__doc__)
    main()