#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 12:15:19 2022

@author: joachim
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from std_map import std_map4d
import naff
from naff_tools import naff_testbench, get_n_arr
import mpl_special
from window_functions import gauss_weights, hann_weights, flattop_weights
from scipy.signal.windows import chebwin


def test_signal(signal, n_arr,
                args=((functools.partial(hann_weights, a_k=1),),
                      (functools.partial(hann_weights, a_k=2),),
                      (gauss_weights,),
                      (functools.partial(chebwin, at=250),),),
                names=("H_1", "H_2", r"\mathrm{gauss}", r"\mathrm{cheb},250"),
                extras=None):
    fig, ax = plt.subplots()

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$|\Delta \nu_N|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_arr[0], n_arr[-2])
    methods = [naff.naffnd_num] * len(args)
    names = list(names)
    args = list(args)
    if extras is not None:
        if extras == "expensive":
            methods.append(naff_expensive)
            names.append(r"\mathrm{expensive}")
            args.append((None,))
        else:
            print(f"Warning, {extras = } not known!")

    for ctr in range(len(methods)):
        method, arg = methods[ctr], args[ctr]
        # name = mpl_special.mathrm(method.__name__)
        name = names[ctr]
        freq, diff = naff_testbench(signal, n_arr, method, *arg)
        ax.plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                   label=fr"$\nu_{{{name}}} = {freq}$")

    ax.legend()
    mpl_special.polish(fig, ax)


def get_test_signal_4d(q10=0.5, q20=0.5, p10=0.05, p20=0.05, n_arr=get_n_arr(),
                       k1=2.25, k2=3.0, k=1.0):
    orbit = std_map4d(p10, p20, q10, q20, n_arr[-1], k1, k2, k)
    signal = orbit[2] - 0.5 + 1j * orbit[0]
    return signal


def get_rotation3d(theta, axis=(0, 0, 1)):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.math.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_rotation4d(theta, n_axis=(0, 0, 1, 0, 0, 0)):
    """
    n_axis = [n_yz, n_zx, n_xy, n_xw, n_yw, n_zw],
    theta counts clockwise rotation in radians
    """
    n_axis = np.asarray(n_axis)
    n_axis = n_axis / np.linalg.norm(n_axis, ord=2)
    matrix = np.array([[0, n_axis[2], -n_axis[1], n_axis[3]],
                       [-n_axis[2], 0, n_axis[0], -n_axis[4]],
                       [n_axis[1], -n_axis[0], 0, n_axis[5]],
                       [-n_axis[3], n_axis[4], -n_axis[5], 0],
                       ])
    from scipy.linalg import expm
    return expm(-theta * matrix)


def get_torus(nu1=0.86747690445587, nu2=0.70096462929942, npoints=2048, r=1, R=5):
    n = np.arange(npoints)
    phi1 = 2*np.pi*nu1*n
    phi2 = 2*np.pi*nu2*n
    orbit = np.array([(R + r*np.sin(phi2)) * np.cos(phi1),
                      (R + r*np.sin(phi2)) * np.sin(phi1),
                      r*np.cos(phi2),
                      np.zeros_like(phi1)])
    return orbit


def plot_torus(torus, rotation=None, s=2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(torus[0], torus[1], torus[2], s=s)
    if rotation is not None:
        rtorus = rotation @ torus
        ax.scatter(rtorus[0], rtorus[1], rtorus[2], s=s)


def add_noise(signal, eps=1e-7):
    noise = np.random.uniform(-eps, eps, size=(2, signal.size))
    signal += noise[0] + 1j * noise[1]


def test_4d(extras=None):
    n_arr = get_n_arr(n_min=256, n_max=2**14, n_n=50)
    signal = get_test_signal_4d(n_arr=n_arr)
    test_signal(signal, n_arr, extras=extras)
    
    
def test_4d_osc():
    n_arr = get_n_arr(n_min=128, n_max=2048, n_n=50)
    signal = get_test_signal_4d(n_arr=n_arr, q10=0.58)
    methods = [functools.partial(naff.naffnd_cos, a_k=1),
               functools.partial(naff.naffnd_cos, a_k=2),
               naff.naffnd_osc2,
               naff.naffnd_osc,]
    names = "H_1", "H_2", r"\mathrm{osc\_fast2}", r"\mathrm{osc\_fast}"
    
    fig, ax = plt.subplots()

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$|\Delta \nu_N|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_arr[0], n_arr[-2])
    for ctr in range(len(methods)):
        method = methods[ctr]
        name = names[ctr]
        freq, diff = naff_testbench(signal, n_arr, method)
        ax.plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                   label=fr"$\nu_{{{name}}} = {freq}$")
    ax.legend()
    mpl_special.polish(fig, ax)


def gauss_nu(abs_fft, ind, alpha=140, k1=(0,), k2=(1,)):
    """
    for k in range(100):
    print(k, gauss_nu(abs_fft, ind, k1=[0, 0]*k + [1], k2=[1, -1] * k + [-1]) - nu1)
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)
    offset = np.sum(k1**2 - k2**2)
    denom = np.sum(k1 - k2) * 2 * abs_fft.size
    if denom == 0:
        print("WARNING: Zero denominator!")
        return ind/abs_fft.size
    ratio = np.prod(abs_fft[(k1 + ind) % abs_fft.size] / abs_fft[(k2 + ind) % abs_fft.size])
    return ind/abs_fft.size + (offset + alpha/np.pi**2 * np.log(ratio)) / denom


def test_torus4d():
    n_arr = get_n_arr(n_min=32, n_max=2**14, n_n=500)
    torus = get_torus(npoints=n_arr[-1])
    rot = get_rotation4d(np.pi/3, (0.0, 0.0, 0.0, 1.0, 1.0, 0.0))
    rot = get_rotation4d(np.pi/2, (0, 0, 1, 0, 0, 0)) @ rot
    rtorus = rot @ torus
    signal = rtorus[0] + 1j * rtorus[1]
    # add_noise(signal, eps=1e-6)

    signal = get_test_signal_4d(n_arr=n_arr)
    # fig, ax = plt.subplots()
    # ax.set_aspect(1.0)
    # ax.scatter(signal.real, signal.imag, linewidths=0, s=1)
    test_signal(signal, n_arr)


def plot_abs_fft(zlist):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    for signal in zlist:
        abs_fft = np.abs(np.fft.fft(signal))
        ax.plot(abs_fft / abs_fft.max())


def get_error_oscillations(z, segment_lengths, first=0, inc=10, num_freqs=1001,
                           method=naff.naffnd_gauss):
    segment_lengths = np.asarray(segment_lengths)
    if segment_lengths.ndim == 0:
        segment_lengths = np.expand_dims(segment_lengths, axis=0)
    nu = naff.naffnd_gauss(z)[0]
    initial_indices = np.arange(first, inc * num_freqs + first, inc)
    diff_list = []
    for segment_length in segment_lengths:
        freqs = np.array([method(z[start:start + segment_length])[0]
                          for start in initial_indices])
        diff = freqs - nu
        diff_list.append(diff)
    return initial_indices, np.squeeze(diff_list)


def _plot_error_oscillations(ax, z, initial_indices, diff_list,
                            segment_lengths=(1024, 2048, 4096),
                            first=0, inc=10, num_freqs=1001):
    segment_lengths = np.asarray(segment_lengths)
    if segment_lengths.ndim == 0:
        segment_lengths = np.expand_dims(segment_lengths, axis=0)
    for i, segment_length in enumerate(segment_lengths):
        ax.plot(initial_indices, diff_list[i], marker='o', ls='--', lw=0.5,
                label=fr"segment length $= {segment_length}$")


def plot_error_oscillations():
    n_arr = get_n_arr(n_max=2**17)
    z = get_test_signal_4d(n_arr=n_arr)
    segment_lengths = (1024, 2048, 4096)
    first = 0
    inc = 10
    num_freqs = 1001
    initial_indices, diff_list = get_error_oscillations(z, segment_lengths, first, inc, num_freqs)
    fig, ax = plt.subplots()
    ax.set_xlim(initial_indices[0], initial_indices[-1])
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\Delta\nu_n$")
    ax.set_title(r"$\Delta\nu_n := \nu_{{(n,n+\mathrm{{segment\ length}})}} "
                  + fr"- \nu_{{(0,2^{{{int(np.log2(z.size))}}})}}$")

    _plot_error_oscillations(ax, z, initial_indices, diff_list, segment_lengths)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.2)
    ax.legend()
    mpl_special.polish(fig, ax)


def naff_expensive_old(z, PLACEHOLDER_ARGUMENT=None):
    length = z.size // 2
    freqs = np.array([naff.naffnd_gauss(z[start:start+length])[0] for start in range(0, length, 1)])
    # from scipy.signal import find_peaks
    # indx = find_peaks(freqs)[0]
    # indx = np.sort([np.argmax(freqs), np.argmin(freqs)])
    # new_freqs = freqs[indx[0]:indx[1]]
    # return np.mean(new_freqs)
    return (freqs.max() + freqs.min()) / 2

def naff_expensive(z, method=lambda x: naff.naffnd_cos(x, a_k=2)):
    length = (3 * z.size) // 4
    freqs = np.array([method(z[start:start+length])[0]
                      for start in range(0, z.size - length, 1)])
    return (freqs.max() + freqs.min()) / 2


def testing_new_feature():
    """
    def f(x, A=2, omega=0.05, C=0.5, B=0.7):
        return A * np.sin(omega*x + C) + B
    L=6
    myx=np.array([-3, -1, 1, 3]) * L
    A=2; omega=0.05; C=0.5; B=0.7;
    myf = f(myx, A, omega, C, B)
    f0,f1,f2,f3 = myf
    b = ((myf[2]**2 - myf[1]**2 + myf[0] * myf[2]- myf[1] * myf[3]) 
         / (3*myf[2] - 3*myf[1]+ myf[0]-myf[3]))
    
    # full scan
    method = functools.partial(naff.naffnd_cos, a_k=2)
    length = 1024
    ind = np.arange(0, z.size - length, 1)
    #freqs = np.squeeze([method(z[i:i+length]) for i in ind])
    L=40
    f0,f1,f2,f3 = freqs[:4*L:L]
    F = (f0-f3)/(f2-f1)+2
    B = (F*(f2+f1)+f0+f3) / (2*(F+1))
    # A = np.sqrt((4*(f1 - f2)**3*(f0*(f3 - f2) + f1**2 - f1*(f2 + f3) + f2**2)) 
    #             / ((f0 + f1 - f2 - f3)*(f0 - 3*f1 + 3*f2 - f3)**2))
    A = np.abs((2*np.abs(f1 - f2)**1.5*np.abs(f0*(f3 - f2) + f1**2 - f1*(f2 + f3) + f2**2)**0.5)
               / (np.abs(f0 + f1 - f2 - f3)**0.5 * (f0 - 3*f1 + 3*f2 - f3)))
    print(A, B-nu)
    """
    """
    method = functools.partial(naff.naffnd_cos, a_k=2)
    l=37
    remainder=l//4
    samples = np.arange(4) * (remainder // 4) + remainder // 8
    print((3*l)//4, remainder, samples)
    initial_indices, diff_list = get_error_oscillations(z, l-remainder, 
                                                        num_freqs=remainder, method=method, inc=1)
    ###27 9 [1 3 5 7]
    
    offset = naff.naffnd_osc(z[:l]) - nu
    mysin = functools.partial(sin, offset=offset)
    xdata = samples
    #ydata = diff_list[0][xdata]
    freqs = freqs = np.array([method(z[start:start + l - l//4])[0]
                              for start in samples])
    ydata = freqs-nu
    par, cov = curve_fit(mysin, xdata, ydata, p0=(-5.83288326e-05,  6.23185407e-01, -5.62606988e+00), xtol=1e-15, ftol=1e-100)
    
    fig, ax = plt.subplots()
    ax.plot(initial_indices, diff_list[0])
    ax.scatter(samples, diff_list[0][samples], c='k', marker='x', s=8)
    ax.axhline((diff_list[0].max() + diff_list[0].min()) / 2, c='c')
    ax.axhline(naff.naffnd_osc(z[:l]) - nu, c='k')
    ax.axhline(naff.naffnd_cos(z[:l], a_k=2) - nu, c='r')
    x=np.linspace(initial_indices[0], initial_indices[-1], 300)
    y=sin(x, *par, offset=naff.naffnd_osc(z[:l]) - nu)
    ax.plot(x, y, ls='--', c='k')
    """
    #double oscillation plot
    """
    n_arr = get_n_arr(n_min=256, n_max=2**17, n_n=50)
    z = get_test_signal_4d(n_arr=n_arr)
    method = functools.partial(naff.naffnd_cos, a_k=2)
    length = 64
    ind = np.arange(0, z.size // 4, 1)
    freqs_64 = np.squeeze([method(z[i:i+length]) for i in ind])
    ydata = np.copy(freqs_64[:200])
    xdata = np.arange(ydata.size)
    def model(x, ampl, offset, freq, phase, low_freq, low_phase):
        return offset + ampl * np.sin(freq * x + phase) * np.sin(low_freq * x + low_phase)
    par, cov = curve_fit(model, xdata, ydata, p0=(5e-6, nu, 0.23, 0, 0.036, 0), xtol=1e-14, ftol=0)
    plt.plot(xdata, ydata)
    plt.plot(xdata, model(xdata, *par))
    print(par)
    """

def main():
    print(__doc__)
    # test_4d(extras="expensive")
    # test_torus4d()
    # plot_error_oscillations()
    test_4d_osc()
    return 0

if __name__ == "__main__":
    main()
