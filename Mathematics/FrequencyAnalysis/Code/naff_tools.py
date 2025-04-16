#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for testing NAFF variants.
"""
import numpy as np
from naff import naffnd_gauss


def fourier_signal(flist, alist, N=300, return_string=False, dig=3):
    """Test signal with a list of frequencies"""
    flist = np.asarray(flist)
    alist = np.asarray(alist)
    if flist.ndim == 0:
        flist = np.expand_dims(flist, axis=0)
        alist = np.expand_dims(alist, axis=0)
    z = np.sum([alist[n] * np.exp(1j * 2*np.pi * flist[n] * np.arange(N))
                for n in range(flist.size)], axis=0)
    if return_string:
        text = ""
        for n in range(flist.size):
            ampl = np.abs(alist[n])
            phase = np.arctan2(alist[n].imag, alist[n].real)
            text += f"{ampl:.{dig}f} "
            if np.abs(phase) > 10**(-dig):
                 text += f"\cdot \mathrm{{e}}^{{{phase:.{dig}f}\mathrm{{i}}}} "
            text += f"\mathrm{{e}}^{{2\pi\mathrm{{i}} \cdot {flist[n]:.{dig}f}\,n}}"
            if n < flist.size - 1:
                text += " + "

        return z, text
    return z


def chaos_indicator(q, p, correct_offset=True, tol=1e-7):
    n = q.size//2
    z = q + 1j * p
    if correct_offset:
        z -= 0.5

    nu1 = naffnd_gauss(z[:n])[0]
    nu2 = naffnd_gauss(z[n:])[0]
    diff = np.abs(nu1 - nu2)
    if diff > tol:
        z = np.exp(2*np.pi*1j * q)
        nu1 = naffnd_gauss(z[:n])[0]
        nu2 = naffnd_gauss(z[n:])[0]
        diff = np.abs(nu1 - nu2)

    return diff


def convert_console_to_array(text):
    lines = text.split('\n')
    q_vals, p_vals = [], []
    for line in lines:
        if line.endswith("True"):
            line_split = line.split(',')
            q_vals.append(line_split[0].split(' ')[-1])
            p_vals.append(line_split[1].split(' ')[-1])

    return np.array([q_vals, p_vals], dtype=float).T


def naff_testbench(signal, lengths, method, *args, swap_freq=True):
    """
    Apply the given Naff-method to the signal for given signal
    lengths and return the estimated error and the frequency
    """

    freqs = np.squeeze(
        np.array([method(signal[:length], *args) for length in lengths])
        )

    freq = float(freqs[-1])
    if swap_freq:
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


def dft(z, ind):
    if isinstance(ind, (int, float)):
        ind = np.array([ind])
    z = np.asarray(z)
    ind = np.asarray(ind)
    return np.array([np.sum(z * np.exp(-2*np.pi*1j * np.arange(z.size)/z.size * ind_val)) 
                     for ind_val in ind])
                     

def abs_dft(nu, w_z):
    n_range = np.arange(w_z.size)
    return np.abs(np.sum(w_z * np.exp(-2*np.pi*1j * n_range * nu)))


def optimize_f(f, x0, err, xtol=1e-15, max_iter=50, verbose=False):
    xa = x0 - err
    xb = x0
    xc = x0 + err
    fxa = f(xa)
    fxb = f(xb)
    fxc = f(xc)
    
    def update(xa, xb, xc, fxa, fxb, fxc):
        xm1 = (xa + xb) / 2
        xm2 = (xb + xc) / 2
        fxm1 = f(xm1)
        fxm2 = f(xm2)
        
        if fxb > fxm1 and fxb > fxm2:
            return xm1, xb, xm2, fxm1, fxb, fxm2, True
        elif fxb > fxm1 and fxb <= fxm2:
            return xb, xm2, xc, fxb, fxm2, fxc, True
        elif fxb <= fxm1 and fxb > fxm2:
            return xa, xm1, xb, fxa, fxm1, fxb, True
        else:
            if verbose:
                print(f"Warning, no progress made at {xa = }, {xb = }, {xc = } "
                      f"for {f.__name__ = }!")
            return xa, xb, xc, fxa, fxb, fxc, False
            
            
    for ctr in range(max_iter):
        xa, xb, xc, fxa, fxb, fxc, flag = update(xa, xb, xc, fxa, fxb, fxc)
        if not flag:
            break
        err = xb - xa
        if err < xtol:
            return xb
    
    print(f"Warning, {xtol = } not reached in {ctr = } steps, final error was {err}")
    return xb


def get_beta(z, nu, ind):
    beta = np.zeros((z.size, z.size), dtype=np.complex128)
    for m in range(z.size):
        for n in range(z.size):
            value = np.exp(-2*np.pi*1j * ind/z.size * (n+m))
            value *= z[n] + z[m]
            value *= np.exp(-2*np.pi*1j * n/z.size) + np.exp(-2*np.pi*1j * n*m/z.size)
            beta[m, n] = value
    return beta