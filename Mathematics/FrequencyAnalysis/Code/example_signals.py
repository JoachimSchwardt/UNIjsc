#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example signals for the NAFF article
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from window_functions import hann_weights, gauss_weights
from std_map import std_map
from naff_tools import (chaos_indicator, convert_console_to_array, naff_testbench,
                        fourier_signal)
from naff import (naffnd_cos, naffnd_gauss, naff_num, naff_num_int, 
                  naff_laskar, naff_laskar_approx, _remove_peak_gauss,
                  naffnd_num, _remove_peak_num)
from scipy.optimize import brentq
from scipy.integrate import quad
import mpl_special


PATH = "/home/joachim/Documents/UNI/Mathematik/FrequencyAnalysis/"


def plot_discontinuous_periodic_signal():
    """
    Smooth signal, that has discontinuities if repeated.
    """
    n_n = 50
    n = np.arange(n_n)
    z_n = np.cos(2*np.pi * n/n_n * 0.7)

    # col = mpl_special.Colors()
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    ax.set_ylabel("Re\,$\{z_n\}$")
    ax.set_xlim(-n_n, 2*n_n - 1)
    ax.plot(n - n_n, z_n, ls='', marker='o', c='k', alpha=0.5)
    ax.axvline(0, c='k', ls='--', lw=0.5, alpha=0.5)
    ax.plot(n, z_n, ls='', marker='o', c='k')
    ax.axvline(n_n, c='k', ls='--', lw=0.5, alpha=0.5)
    ax.plot(n + n_n, z_n, ls='', marker='o', c='k', alpha=0.5)

    mpl_special.polish(fig, ax)


def plot_hanning_window_first_order():
    """
    Hanning window of first order
    """
    n_n = 500
    x = np.linspace(0, 1, n_n)
    w_method = functools.partial(hann_weights, a_k=1)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$w(x)$")
    ax.plot(x, w_method(x))

    mpl_special.polish(fig, ax)


def plot_continuous_periodic_signal():
    """
    Smooth signal, discontinuities removed using first order hanning window
    """
    n_n = 50
    n = np.arange(n_n)
    z_n = np.cos(2*np.pi * n/n_n * 0.7)
    w_n = hann_weights(n / n_n, a_k=1)
    zw_n = z_n * w_n


    col = mpl_special.Colors()
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    ax.set_ylabel("Re\,$\{z_n\}$")
    ax.set_xlim(-n_n, 2*n_n - 1)

    # discont. signal
    ax.plot(n - n_n, z_n, ls='', marker='o', c='k', alpha=0.5)
    ax.axvline(0, c='k', ls='--', lw=0.5, alpha=0.5)
    ax.plot(n, z_n, ls='', marker='o', c='k', label='$z_n$')
    ax.axvline(n_n, c='k', ls='--', lw=0.5, alpha=0.5)
    ax.plot(n + n_n, z_n, ls='', marker='o', c='k', alpha=0.5)

    # cont. signal
    ax.plot(n - n_n, zw_n, ls='', marker='o', c=col.get_color(0), alpha=0.5)
    ax.plot(n, zw_n, ls='', marker='o', c=col.get_color(0), label='$z_nw_n$')
    ax.plot(n + n_n, zw_n, ls='', marker='o', c=col.get_color(0), alpha=0.5)

    ax.legend()
    mpl_special.polish(fig, ax)


def plot_sample_2d_orbit():
    q0 = 0.5
    p0 = 0.11
    n_max = 512
    k = 0.7

    q_vals, p_vals = std_map(q0, p0, n_max, k)

    col = mpl_special.Colors()

    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$")
    ax.axis([0, 1, -0.5, 0.5])

    # for [q0, p0] in [[0.5, 0.45], [0.5, 0.35], [0.5, 0.25],
    #                  [0.5, 0.15], [0.5, -0.35], [0.5, -0.45],
    #                  [0.2, 0.4], [0.2, 0.3], [0.2, 0.2], [0.2, 0.1],
    #                  [0.2, -0.4], [0.2, -0.3], [0.2, -0.2], [0.2, -0.1],
    #                  ]:
    for [q0, p0] in convert_console_to_array(
                """
                q0 = 0.4860, p0 = -0.0138, diff = 0.00e+00, False
                q0 = 0.4068, p0 = -0.0032, diff = 0.00e+00, True
                q0 = 0.3301, p0 = 0.0048, diff = 2.29e-06, True
                q0 = 0.2773, p0 = 0.0064, diff = 1.29e-13, True
                q0 = 0.2207, p0 = 0.0025, diff = 3.94e-09, True
                q0 = 0.1570, p0 = -0.0045, diff = 3.65e-08, True
                q0 = 0.1050, p0 = -0.0014, diff = 2.95e-07, True
                q0 = 0.0538, p0 = 0.0017, diff = 9.97e-07, True
                q0 = 0.0492, p0 = -0.1015, diff = 1.50e-07, True
                q0 = 0.0492, p0 = -0.1736, diff = 8.83e-06, True
                q0 = 0.0468, p0 = -0.2404, diff = 6.11e-07, True
                q0 = 0.0437, p0 = -0.3498, diff = 2.24e-07, True
                q0 = 0.0453, p0 = -0.2986, diff = 5.34e-07, True
                q0 = 0.0476, p0 = -0.4118, diff = 4.82e-06, True
                q0 = 0.0430, p0 = -0.4568, diff = 1.72e-11, True
                q0 = 0.2191, p0 = -0.4553, diff = 7.31e-06, True
                q0 = 0.2113, p0 = -0.4320, diff = 7.58e-06, True
                q0 = 0.1268, p0 = -0.4561, diff = 1.85e-08, True
                q0 = 0.0430, p0 = 0.1530, diff = 2.06e-07, True
                q0 = 0.0740, p0 = -0.1364, diff = 4.27e-06, True
                q0 = 0.0787, p0 = -0.1441, diff = 3.73e-06, True
                q0 = 0.0562, p0 = 0.2888, diff = 1.80e-10, True
                q0 = 0.0569, p0 = 0.3369, diff = 1.99e-06, True
                q0 = 0.0639, p0 = 0.4409, diff = 1.13e-07, True
                q0 = 0.0849, p0 = 0.4099, diff = 1.12e-06, True
                q0 = 0.0414, p0 = 0.4720, diff = 4.30e-10, True
                q0 = 0.0282, p0 = 0.4844, diff = 2.61e-10, True
                q0 = 0.7747, p0 = 0.4588, diff = 2.89e-07, True
                q0 = 0.7755, p0 = 0.4324, diff = 4.36e-07, True
                q0 = 0.8422, p0 = 0.3688, diff = 2.38e-08, True
                q0 = 0.7359, p0 = 0.3400, diff = 1.04e-10, True
                q0 = 0.9306, p0 = 0.1732, diff = 1.06e-06, True
                q0 = 0.9345, p0 = 0.2027, diff = 1.21e-08, True
                q0 = 0.9516, p0 = 0.0894, diff = 3.85e-08, True
                """ + # chaotic orbits
                """
                q0 = 0.0212, p0 = -0.0254, diff = 8.67e-01, False
                q0 = 0.2331, p0 = 0.4937, diff = 1.48e-03, False
                q0 = 0.7793, p0 = 0.4464, diff = 3.89e-02, True
                q0 = 0.7793, p0 = 0.4464, diff = 3.89e-02, False
                q0 = 0.2222, p0 = -0.4429, diff = 5.55e-03, True
                """):
        q_n, p_n = std_map(q0, p0, 8*n_max, k)
        diff = chaos_indicator(q_n, p_n, tol=1e-6, correct_offset=True)
        if diff > 1e-6:
            c = 'k'
            alpha = 0.2
        else:
            c = col.get_color()
            alpha = 0.2
        ax.plot(q_n[:n_max], p_n[:n_max], alpha=alpha, ls='', marker='o', ms=1, c=c)

    ax.plot(q_vals, p_vals, ls='', marker='o', c='k')

    mpl_special.polish(fig, ax)
    fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_surround.png")


# def plot_sample_2d_orbit_abs_fft():
#     q0 = 0.5
#     p0 = 0.11
#     n_max = 512
#     k = 0.7

#     q_vals, p_vals = std_map(q0, p0, n_max, k)
#     w_n = hann_weights(np.arange(n_max) / n_max)
#     fft = np.fft.fft(w_n * (q_vals - 0.5 + 1j * p_vals))
#     abs_fft = np.abs(fft)

#     fig, ax = plt.subplots()
#     ax.set_xlabel("$j$")
#     ax.set_ylabel("$|F_j|$")
    
#     ax.plot(abs_fft, c='k')
    
#     ax.set_xlim(0, n_max-1)
#     ax.set_ylim(-1, ax.get_ylim()[1])
#     mpl_special.polish(fig, ax)
#     fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_abs_fft.png")


# def plot_sample_2d_orbit_abs_fft_zoom():
#     q0 = 0.5
#     p0 = 0.11
#     n_max = 512
#     k = 0.7

#     q_vals, p_vals = std_map(q0, p0, n_max, k)
#     w_n = hann_weights(np.arange(n_max) / n_max)
#     zw_n = w_n * (q_vals - 0.5 + 1j * p_vals)
#     fft = np.fft.fft(zw_n)
#     abs_fft = np.abs(fft)
#     ind = np.argmax(abs_fft)
#     j_vals = np.arange(ind - 2, ind + 3, 1)
    
#     # compute intermediate frequency amplitudes for comparison
#     j_vals_bg = np.linspace(j_vals[0] - 0.2, j_vals[-1] + 0.2, 200)
#     fft_bg = np.sum([zw_n * np.exp(-2*np.pi*1j * jval / n_max * np.arange(n_max))
#                      for jval in j_vals_bg], axis=1)

#     fig, ax = plt.subplots()
#     ax.set_xlabel("$j$")
#     ax.set_ylabel("$|F_j|$")
    
#     ax.plot(j_vals, abs_fft[j_vals], c='k', marker='o', ls='', ms=3)
#     ax.plot(j_vals_bg, np.abs(fft_bg), c='k', ls='--', alpha=0.4)
    
#     ax.set_xlim(j_vals[0] - 0.2, j_vals[-1] + 0.2)
#     ax.set_ylim(-1, ax.get_ylim()[1])
#     mpl_special.polish(fig, ax)
#     fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_abs_fft_zoom.png")


def plot_sample_2d_orbit_abs_fft_inset():
    q0 = 0.5
    p0 = 0.11
    n_max = 512
    k = 0.7

    q_vals, p_vals = std_map(q0, p0, n_max, k)
    w_n = hann_weights(np.arange(n_max) / n_max)
    zw_n = w_n * (q_vals - 0.5 + 1j * p_vals)
    fft = np.fft.fft(zw_n)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)
    j_vals = np.arange(ind - 2, ind + 3, 1)
    
    # compute intermediate frequency amplitudes for comparison
    j_vals_bg = np.linspace(j_vals[0] - 0.2, j_vals[-1] + 0.2, 200)
    fft_bg = np.sum([zw_n * np.exp(-2*np.pi*1j * jval / n_max * np.arange(n_max))
                     for jval in j_vals_bg], axis=1)

    fig, ax = plt.subplots()
    ax.set_xlabel("$j$")
    ax.set_ylabel("$|F_j|$")
    
    ax_inset = ax.inset_axes([0.15, 0.35, 0.5, 0.5])
    ax_inset.set_xlabel("$j$")
    ax_inset.set_ylabel("$|F_j|$")
    
    ax.plot(abs_fft, c='k')
    
    ax.set_xlim(0, n_max-1)
    ax.set_ylim(-1, ax.get_ylim()[1])
    
    ax_inset.plot(j_vals, abs_fft[j_vals], c='k', marker='o', ls='', ms=3)
    ax_inset.plot(j_vals_bg, np.abs(fft_bg), c='k', ls='--', alpha=0.4)
    
    ax_inset_bounds = [j_vals[0] - 0.2, j_vals[-1] + 0.2, -0.1, ax_inset.get_ylim()[1]]
    ax_inset.axis(ax_inset_bounds)
    rect, conn = ax.indicate_inset([ax_inset_bounds[0] - 2, ax_inset_bounds[2], 
                                    ax_inset_bounds[1] - ax_inset_bounds[0] + 4,
                                    ax_inset_bounds[3] - ax_inset_bounds[2] - 2], 
                                   inset_ax=ax_inset, transform=ax.transData, 
                                   edgecolor='k', alpha=0.5, lw=0.5)
    for con in conn:
        con.set_lw(0.5)
        con.set_ls('--')
    mpl_special.polish(fig, [ax, ax_inset])
    fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_abs_fft_inset.png")
    
    
def plot_sample_2d_orbit_f_eps():
    q0 = 0.5
    p0 = 0.11
    n_max = 512
    k = 0.7

    q_vals, p_vals = std_map(q0, p0, n_max, k)
    z = (q_vals - 0.5) + 1j * p_vals
    w_method = functools.partial(hann_weights, a_k=1)
    w_n = w_method(np.arange(n_max) / n_max)
    zw_n = w_n * z
    fft = np.fft.fft(zw_n)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)
    
    if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
        ind -= 1
    
    ratio = fft[(ind + 1) % z.size] / fft[ind]
    
    def root_expression(eps):
        """Function of which we want to find a root."""
        def integral(eps):
            def f_real(x):
                return w_method(x) * np.cos(2*np.pi * x * z.size * eps)
            def f_imag(x):
                return w_method(x) * np.sin(2*np.pi * x * z.size * eps)
            real = quad(f_real, 0.0, 1.0)[0]
            imag = quad(f_imag, 0.0, 1.0)[0]
            return real + 1j * imag

        return np.abs(integral(eps - 1/z.size)) - np.abs(ratio * integral(eps))
    
    delta = 1 / z.size
    eps = brentq(root_expression, 0, delta, xtol=1e-15)

    # slope = 294.8845589124666 at zoom level, can also be done analytically ::
    # def fp(eps):
    #     def integ(eps):
    #         def f_real(x):
    #             return w_method(x) * np.cos(2*np.pi * x * z.size * eps)
    #         def f_imag(x):
    #             return w_method(x) * np.sin(2*np.pi * x * z.size * eps)
    #         real = quad(f_real, 0.0, 1.0)[0]
    #         imag = quad(f_imag, 0.0, 1.0)[0]
    #         return real + 1j * imag
    #     def integ_p(eps):
    #         def f_real(x):
    #             return -w_method(x) * 2*np.pi * x * z.size * np.sin(2*np.pi * x * z.size * eps)
    #         def f_imag(x):
    #             return w_method(x) * 2*np.pi * x * z.size * np.cos(2*np.pi * x * z.size * eps)
    #         real = quad(f_real, 0.0, 1.0)[0]
    #         imag = quad(f_imag, 0.0, 1.0)[0]
    #         return real + 1j * imag
    #     part1 = (integ(eps - 1/z.size).real * integ_p(eps - 1/z.size).real 
    #              + integ(eps - 1/z.size).imag * integ_p(eps - 1/z.size).imag) / np.abs(integ(eps - 1/z.size))
    #     part2 = (integ(eps).real * integ_p(eps).real 
    #              + integ(eps).imag * integ_p(eps).imag) / np.abs(integ(eps) / ratio)
    #     return part1 - part2
    
    # plot
    eps_range = np.linspace(0, delta, 50)
    f_vals = [root_expression(eps_val) for eps_val in eps_range]
    eps_range_zoom = np.linspace(eps - 1e-7, eps + 1e-7, 20)
    f_vals_zoom = [root_expression(eps_val) for eps_val in eps_range_zoom]

    fig, ax = plt.subplots()
    ax.set_xlabel("$\epsilon$")
    ax.set_ylabel("$f(\epsilon)$")
    ax.set_xlim(0, delta)
    
    ax.plot(eps_range, f_vals, c='k')
    
    ax_inset = ax.inset_axes([0.1, 0.55, 0.4, 0.4])
    ax_inset.set_xlabel("$\epsilon$")
    ax_inset.set_ylabel("$f(\epsilon)$")
    ax_inset.set_xlim(eps_range_zoom[0], eps_range_zoom[-1])

    ax_inset.plot(eps_range_zoom, f_vals_zoom, c='k')    

    rect, conn = ax.indicate_inset([eps - 1e-5, -5e-3, 2e-5, 1e-2], 
                                   inset_ax=ax_inset, transform=ax.transData, 
                                   edgecolor='k', alpha=0.5, lw=0.5)
    for con in conn:
        con.set_lw(0.5)
        con.set_ls('--')
    mpl_special.polish(fig, [ax, ax_inset])
    fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_f_eps.png")
    
    
def plot_sample_2d_orbit_roc():
    n_min = 32
    n_max = 4096
    n_n = 50
    k = 0.7
    q0 = 0.5
    p0 = 0.11
    n_arr = np.unique(
        np.logspace(np.log2(n_min), np.log2(n_max), n_n, base=2)
        .astype(int)
        )

    q_vals, p_vals = std_map(q0, p0, n_arr[-1], k)
    signal = (q_vals - 0.5) + 1j * p_vals

    fig, [ax, ax2] = plt.subplots(1, 2)

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\Delta \nu_N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_arr[0], n_arr[-2])
    
    ax2.set_xlabel("$N$")
    ax2.set_ylabel(r"$\Delta\nu_N$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(n_arr[0], n_arr[-2])
    
    w_method = functools.partial(hann_weights, a_k=1)
    methods = [naff_laskar, naff_laskar_approx, naff_num_int, naff_num,]
    args = [[], [], (w_method,), (w_method,),]
    names = ["\mathrm{Laskar}", "\mathrm{approx.}", "f", "f_\Sigma"]

    for ctr in range(len(methods)):
        method, arg = methods[ctr], args[ctr]
        # name = mpl_special.mathrm(names[ctr])
        name = names[ctr]
        freq, diff = naff_testbench(signal, n_arr, method, *arg, swap_freq=False)
        if ctr == 0:
            diff_laskar = np.copy(diff)
            ax2.plot(n_min, 1e-16)
        else:
            diff_diff = np.max([np.abs(diff - diff_laskar), 
                                np.full_like(diff, 1e-16)], axis=0)
            ax2.plot(n_arr[:-1], diff_diff, ls='--', lw=0.5, marker='o',
                     label=fr"$|\nu_{{{names[0]}}} - \nu_{{{name}}}|$")
        ax.plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                label=fr"$\nu_{{{name}}} = {freq}$")

    ax.legend()
    ax2.legend()
    mpl_special.polish(fig, [ax, ax2])
    plt.subplots_adjust(wspace=0.17)
    fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_roc.png")
    
    
def plot_sample_2d_orbit_roc_gauss_cos():
    n_min = 32
    n_max = 4096
    n_n = 50
    k = 0.7
    q0 = 0.5
    p0 = 0.11
    n_arr = np.unique(
        np.logspace(np.log2(n_min), np.log2(n_max), n_n, base=2)
        .astype(int)
        )

    q_vals, p_vals = std_map(q0, p0, n_arr[-1], k)
    signal = (q_vals - 0.5) + 1j * p_vals

    fig, ax = plt.subplots()

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\Delta \nu_N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_arr[0], n_arr[-2])
    
    w_method = functools.partial(hann_weights, a_k=2)
    methods = [naffnd_gauss, naffnd_cos, naff_num, naff_num, ]
    args = [(1,), (1, 2,), (gauss_weights,), (w_method,),]
    names = ["\mathrm{gauss}", "H_2", "f_{\Sigma, \mathrm{gauss}}", 
             "f_{\Sigma, H_2}"]

    for ctr in range(len(methods)):
        method, arg = methods[ctr], args[ctr]
        name = names[ctr]
        freq, diff = naff_testbench(signal, n_arr, method, *arg, swap_freq=False)
        ax.plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                label=fr"$\nu_{{{name}}} = {freq}$")

    ax.legend()
    mpl_special.polish(fig, ax)
    fig.subplots_adjust(top=0.994)
    fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_roc_gauss_cos.png")
    
    
def plot_sample_2d_orbit_roc_cos():
    n_min = 32
    n_max = 4096
    n_n = 50
    k = 0.7
    q0 = 0.5
    p0 = 0.11
    n_arr = np.unique(
        np.logspace(np.log2(n_min), np.log2(n_max), n_n, base=2)
        .astype(int)
        )

    q_vals, p_vals = std_map(q0, p0, n_arr[-1], k)
    signal = (q_vals - 0.5) + 1j * p_vals

    fig, ax = plt.subplots()

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\Delta \nu_N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_arr[0], n_arr[-2])
    
    # args = [(1, 2*k + 1,) for k in range(num_k)]
    # names = [f"H_{2*k + 1}" for k in range(num_k)]
    k_vals = [1, 2, 3, 5, 8]
    args = [(1, k,) for k in k_vals]
    names = [f"H_{k}" for k in k_vals]
    methods = [naffnd_cos, ] * len(args)

    for ctr in range(len(methods)):
        method, arg = methods[ctr], args[ctr]
        name = names[ctr]
        freq, diff = naff_testbench(signal, n_arr, method, *arg, swap_freq=False)
        ax.plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                label=fr"$\nu_{{{name}}} = {freq}$")

    ax.legend()
    mpl_special.polish(fig, ax)
    fig.savefig(PATH + "pictures/sample_2d_orbit_k07_q05_p01_roc_cos.png")


def plot_multi_freq_sample_signal():
    flist = [0.524, (3 - np.sqrt(5)) / 2, np.sqrt(2) - 1, 1 / np.pi, 0.2, 0.0]
    alist = [1.0, 0.9, 0.6, 0.3, 0.1, 0.4 + 0.1j]
    z = fourier_signal(flist, alist, N=512)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"Re\,$\{z\}$")
    ax.set_ylabel(r"Im\,$\{z\}$")
    ax.set_aspect(1.0)
    ax.plot(z.real, z.imag, ls='', marker='o', c='k')
    mpl_special.set_ticks_linear(ax, -3, 3, numticks=7, which='x', dtype=int)
    mpl_special.polish(fig, ax, xva='center')
    fig.savefig(PATH + "pictures/multi_freq_sample_signal.png")


def plot_multi_freq_remove_peak():
    flist = [0.524, (3 - np.sqrt(5)) / 2, np.sqrt(2) - 1, 1 / np.pi, 0.2, 0.0]
    alist = [1.0, 0.9, 0.6, 0.3, 0.1, 0.4 + 0.1j]
    z = fourier_signal(flist, alist, N=512)
    w_n = gauss_weights(np.arange(z.size) / z.size)
    zw_n = w_n * z
    fft = np.fft.fft(zw_n)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)
    
    num_j = 15
    j_vals = np.arange(ind - num_j, ind + num_j, 1)
    
    freq, coeff = naffnd_gauss(z, n_freq=5, return_coeff=True, num_j=num_j)

    fig, ax = plt.subplots()
    ax.set_xlabel("$j$")
    ax.set_ylabel("$|F_j|$")
    ax.set_xlim(0, z.size - 1)
    
    ax_inset = ax.inset_axes([0.63, 0.42, 0.34, 0.55])
    ax_inset.set_xlabel("$j$")
    ax_inset.set_ylabel("$|F_j|$")
    
    ax.plot(abs_fft, c='k', label='before')
    ax_inset.plot(j_vals, abs_fft[j_vals], c='k', marker='o', ls='', ms=2.5)
    
    # remove peak
    _remove_peak_gauss(abs_fft, ind, freq[0], num_j=num_j)
    ax.plot(abs_fft, label='after')
    ax_inset.plot(j_vals, abs_fft[j_vals], marker='o', ls='', ms=2.5)
    
    rect, conn = ax.indicate_inset_zoom(ax_inset, edgecolor='k', alpha=0.5, lw=0.5)
    for con in conn:
        con.set_lw(0.5)
        con.set_ls('--')
    ax.legend()
    mpl_special.polish(fig, [ax, ax_inset])
    fig.savefig(PATH + "pictures/multi_freq_remove_peak.png")


def plot_two_freq_sample_signal():
    flist = [0.224, 0.225]
    alist = [1.0, 1.0]
    z = fourier_signal(flist, alist, N=512)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"Re\,$\{z\}$")
    ax.set_ylabel(r"Im\,$\{z\}$")
    ax.set_aspect(1.0)
    ax.plot(z.real, z.imag, ls='', marker='o', c='k')
    mpl_special.set_ticks_linear(ax, -2, 2, 5, which='y', dtype=int)
    mpl_special.polish(fig, ax, xva='center')
    fig.savefig(PATH + "pictures/two_freq_sample_signal.png")


def plot_two_freq_remove_peak():
    """
    dig = 3
    indx = [0, 1]
    for k in range(freq.size):
        diff = (freq[k]) - flist[indx[k]]
        a_rec = np.abs(coeff[k])
        a_true = np.abs(alist[indx[k]])
        adiff = a_rec - a_true
        p_rec = np.arctan2(coeff[k].imag, coeff[k].real)
        p_true = np.arctan2(alist[indx[k]].imag, alist[indx[k]].real)
        pdiff = p_rec - p_true
        text = "$ & $".join([f"{k}", f"{flist[k]:.{dig}f}", f"{alist[k]:.1f}", 
                             fr"\num{{{diff:.2e}}}", fr"\num{{{adiff:.2e}}}", 
                             fr"\num{{{pdiff:.2e}}}"])
        text = "$" + text.replace('j', '\i') + r"$\\"
        print(text)
    """
    flist = [0.224, 0.225]
    alist = [1.0, 1.0]
    z = fourier_signal(flist, alist, N=512)
    w_method = functools.partial(hann_weights, a_k=1)   # gauss_weights
    w_n = w_method(np.arange(z.size) / z.size)
    zw_n = w_n * z
    fft = np.fft.fft(zw_n)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)
    
    num_j = 10
    j_vals = np.arange(ind - num_j, ind + num_j, 1)
    
    # freq, coeff = naffnd_gauss(z, n_freq=2, return_coeff=True, num_j=num_j)
    freq, coeff = naffnd_num(z, w_method, n_freq=2, return_coeff=True, num_j=num_j)
    print(freq[0], coeff[0])
    print(freq[1], coeff[1])

    fig, ax = plt.subplots()
    ax.set_xlabel("$j$")
    ax.set_ylabel("$|F_j|$")
    ax.set_xlim(0, z.size - 1)
    
    ax_inset = ax.inset_axes([0.4, 0.32, 0.57, 0.65])
    ax_inset.set_xlabel("$j$")
    ax_inset.set_ylabel("$|F_j|$")
    
    ax.plot(abs_fft, c='k', label='before')
    ax_inset.plot(j_vals, abs_fft[j_vals], c='k', marker='o', ls='', ms=2.5)
    
    # remove peak
    # _remove_peak_gauss(abs_fft, ind, freq[0], num_j=num_j)
    _remove_peak_num(abs_fft, ind, freq[0], w_n, num_j=num_j)
    ax.plot(abs_fft, label='after')
    ax_inset.plot(j_vals, abs_fft[j_vals], marker='o', ls='', ms=2.5)
    
    rect, conn = ax.indicate_inset_zoom(ax_inset, edgecolor='k', alpha=0.5, lw=0.5)
    for con in conn:
        con.set_lw(0.5)
        con.set_ls('--')
    ax.legend()
    mpl_special.polish(fig, [ax, ax_inset])
    fig.savefig(PATH + "pictures/two_freq_remove_peak.png")
    

def main():
    print(__doc__)
    # plot_discontinuous_periodic_signal()
    # plot_hanning_window_first_order()
    # plot_continuous_periodic_signal()
    # plot_sample_2d_orbit()
    # plot_sample_2d_orbit_abs_fft()
    # plot_sample_2d_orbit_abs_fft_zoom()
    # plot_sample_2d_orbit_abs_fft_inset()
    # plot_sample_2d_orbit_f_eps()
    # plot_sample_2d_orbit_roc()
    # plot_sample_2d_orbit_roc_gauss_cos()
    # plot_sample_2d_orbit_roc_cos()
    # plot_multi_freq_sample_signal()
    # plot_multi_freq_remove_peak()
    # plot_two_freq_sample_signal()
    # plot_two_freq_remove_peak()
    
    return 0


if __name__ == "__main__":
    main()
    # plt.close('all')
