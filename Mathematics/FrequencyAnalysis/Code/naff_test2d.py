#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 12:15:19 2022

@author: joachim
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from std_map import std_map
from naff import naff_num_int, naffnd_gauss, naffnd_cos
from naff_tools import naff_testbench
import mpl_special
from window_functions import gauss_weights, hann_weights, flattop_weights


def test_2d(n_min=32, n_max=4096, n_n=50, n_plot=1024, k=0.7):
    """
    Test numerical integration NAFF in console --> setup:
    n_min=32; n_max=4096; n_n=50; n_plot=1024; k=0.7
    
    w_method = gauss_weights; z = np.copy(signal[:512])
    
    eps_range = np.linspace(eps - 1e-9, eps + 1e-9, 50)
    min_vals = [root_expression(eps_val) for eps_val in eps_range]
    plt.plot(eps_range, min_vals)
    """
    q0 = 0.5
    p0 = 0.11
    n_arr = np.unique(
        np.logspace(np.log2(n_min), np.log2(n_max), n_n, base=2)
        .astype(int)
        )

    q_vals, p_vals = std_map(q0, p0, n_arr[-1], k)
    signal = (q_vals - 0.5) + 1j * p_vals

    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlabel("$q$")
    ax[0].set_ylabel("$p$")
    ax[0].axis([0, 1, -0.5, 0.5])

    ax[1].set_xlabel("$N$")
    ax[1].set_ylabel(r"$|\Delta \nu_N|$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(n_arr[0], n_arr[-2])

    ax[0].plot(q_vals[:n_plot], p_vals[:n_plot], ls='', marker='o', c='k')

    # methods = [naffnd, naffnd_cos, naff_num, naff_num, naff_num]
    # args = [(1,), (1, 2), (hann_weights,), (gauss_weights,),
    #         (flattop_weights(hann_weights, fpar=0.9),)]
    
    hann_weights2 = functools.partial(hann_weights, a_k=2)
    methods = [naffnd_gauss, naffnd_cos, naff_num_int, naff_num_int,]
               #naff_num_int]
    args = [(1,), (1, 2), (hann_weights2,), (gauss_weights,),]
            #(flattop_weights(hann_weights2, fpar=0.2),)]

    for ctr in range(len(methods)):
        method, arg = methods[ctr], args[ctr]
        name = mpl_special.mathrm(method.__name__)
        freq, diff = naff_testbench(signal, n_arr, method, *arg)
        ax[1].plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                   label=fr"$\nu_{{{name}}} = {freq}$")

    ax[1].legend()
    mpl_special.polish(fig, ax)
    return fig, ax


def main():
    print(__doc__)
    test_2d()

    return 0

if __name__ == "__main__":
    main()
