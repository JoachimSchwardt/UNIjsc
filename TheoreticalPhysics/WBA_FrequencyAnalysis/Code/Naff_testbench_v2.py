#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 12:15:19 2022

@author: joachim
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from NaffND import naffnd
from NaffND_cos import naffnd_cos, hann_coeff
from std_map import std_map, std_map4d
import mpl_special
from scipy.optimize import minimize_scalar, minimize, brentq
from scipy.integrate import quad


def hann_weights(x, a_k=1):
    a_k = hann_coeff(a_k)
    vals = np.array([a_k[k] * np.cos(2*np.pi*x * k) for k in range(a_k.size)])
    if isinstance(x, (float, int)):
        return np.sum(vals)
    else:
        return np.sum(vals, axis=0)


def gauss_weights(x, alpha=140):
    return np.exp(-alpha * (x - 0.5)**2)


def flattop_weights(w_method, fpar=0.5):
    """Replace midsection of weights with a constant"""
    def new_w_method(x):
        if isinstance(x, (float, int)):
            if x < fpar / 2:
                return w_method(x / fpar)
            elif x > 1 - fpar / 2:
                return w_method(1 - x / fpar)
            else:
                return 1.0
        else:
            weights = np.ones(x.size, dtype=float)
            indx = (x < fpar / 2)
            weights[indx] = w_method(x[indx] / fpar)
            indx = (x > 1 - fpar / 2)
            weights[indx] = w_method(1 - x[indx] / fpar)
            return weights

    return new_w_method


def naff_num(z, w_method):
    weights = w_method(np.arange(z.size))
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)
    nu_init = np.argmax(abs_fft) / z.size    # initial guess
    n_range = np.arange(z.size)

    def minimizer(nu, w_z, n_range):
        # return 1 / np.abs(np.sum(w_z * np.exp(-2*np.pi*1j * nu * n_range)))
        c_seq = w_z * np.exp(-2*np.pi*1j * nu * n_range)
        real = np.math.fsum(c_seq.real)
        imag = np.math.fsum(c_seq.imag)
        return 1 / np.abs(real**2 + imag**2)

    delta = 1 / z.size
    bracket = nu_init + np.array([-delta, 0.0, delta])
    nu = minimize_scalar(minimizer, bracket=bracket, args=(w_z, n_range),
                         tol=1e-15).x
    return nu


def naff_num_int(z, w_method):
    """
    Numerical NAFF using the integral approximation and 'quad'.

    Compute exact Fourier coefficients of the signal (F_j, j=0,...,N-1)
    Compute model coefficients (F_j^M = const * int_0^1 w(x) * exp(2pi*i * eps * x))

    Assume that 'R := F_{j+1} / F_j' is roughly 'R^M := F_{j+1}^M / F_j^M'
    and numerically solve $R * F_j^M = F_{j+1}^M$ for 'eps'

    This is done by minimizing the absolute value of
        'int_0^1 w(x) exp(2pi*i * eps * x) * [exp(-2pi*i * x/N) - R]'
    """
    weights = w_method(np.arange(z.size) / z.size)
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)

    if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
        ind -= 1

    ratio = fft[(ind + 1) % z.size] / fft[ind]

    # def minimizer(eps):
    #     """Function which should be minimized."""
    #     def integral(eps):
    #         def f_real(x):
    #             return w_method(x) * np.cos(2*np.pi * x * z.size * eps)
    #         def f_imag(x):
    #             return w_method(x) * np.sin(2*np.pi * x * z.size * eps)
    #         real = quad(f_real, 0.0, 1.0)[0]
    #         imag = quad(f_imag, 0.0, 1.0)[0]
    #         return real + 1j * imag

    #     res = integral(eps - 1/z.size) - ratio * integral(eps)
    #     # return res.real**2 + res.imag**2
    #     return np.abs(res)
    

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

    nu_init = ind / z.size    # initial guess
    delta = 1 / z.size
    # bracket = np.array([-delta, 0.0, delta]) + delta
    # eps = minimize_scalar(minimizer, bracket=bracket, tol=1e-15).x
    
    # FIXME: minimize not stable enough, method could work MUCH better!! (especially large N)
    # eps = minimize(minimizer, x0=[0.0], bounds=[(0.0, delta)], tol=1e-15).x[0]
    
    # root finding approach
    eps = brentq(root_expression, 0, delta, xtol=1e-15)
    nu = nu_init + eps
    return nu


def naff_testbench(signal, lengths, method, *args):
    """
    Apply the given Naff-method to the signal for given signal
    lengths and return the estimated error and the frequency
    """

    freqs = np.squeeze(
        np.array([method(signal[:length], *args) for length in lengths])
        )

    freq = float(freqs[-1])
    if freq > 0.5:
        freq = 1 - freq
        freqs = 1 - freqs

    indx = (np.abs(freqs - freq) > np.abs(1 - freqs - freq))
    freqs[indx] = 1 - freqs[indx]
    count = np.sum(indx)
    if count > 0:
        print(f"Had to swap {count} frequencies for {method.__name__}!")

    abs_diff = np.abs(freqs[:-1] - freq)
    abs_diff[abs_diff < 1e-16] = 1e-16
    return freq, abs_diff


def test_2d(n_min=32, n_max=4096, n_n=50, n_plot=1024, k=0.7):
    """
    Test numerical integration NAFF in console --> setup:
    n_min=32; n_max=4096; n_n=50; n_plot=1024; k=0.7
    w_method = gauss_weights; z = np.copy(signal[:512])
    """
    q0 = 0.5
    p0 = 0.1
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
    methods = [naffnd, naffnd_cos, naff_num_int, naff_num_int,]
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
