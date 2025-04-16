#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:06:30 2022

@author: joachim
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_special


def compute_1d_band_empty(ka, n_0, a, m, hbar, e):
    scaling = hbar**2 / (2*m*e*a**2)
    expr = (ka - 2*np.pi * n_0)**2
    return expr * scaling


def compute_1d_band_periodic(ka, n_0, a, m, hbar, e, v_0):
    scaling = hbar**2 / (2*m*e*a**2)
    expr = (ka - 2*np.pi * n_0)**2
    v = (v_0 / (scaling * a**2))**2 / 4
    # print(v, np.max(np.abs(expr)))
    return (expr + 0.5 * v / (expr - np.pi**2)) * scaling


def compute_1d_band_periodic_cbrt(ka, n_0, a, m, hbar, e, v_0):
    n_vals = n_0 + np.array([-1, 0, 1])
    scaling = hbar**2 / (2*m*e*a**2)
    v = (v_0 / (scaling * a**2))**2 / 4
    e_vals = np.zeros((n_vals.size, ka.size), dtype=np.complex128)
    for i in range(n_vals.size):
        e_vals[i] = (ka - 2*np.pi * n_vals[i])**2
    b_val = -np.sum(e_vals, axis=0)
    # print("cbrt", v, np.max(np.abs(e_vals[1])))
    c_val = e_vals[0] * e_vals[1] + e_vals[1] * e_vals[2] + e_vals[2] * e_vals[0] - 2*v
    d_val = v * (e_vals[0] + e_vals[2]) - np.prod(e_vals, axis=0)
    p_val = c_val - b_val**2 / 3
    q_val = d_val - c_val * b_val / 3 + 2 * b_val**3 / 27

    sqrt = np.sqrt(q_val**2 / 4 + p_val**3 / 27)
    root = -b_val / 3 + (q_val / 2 - sqrt)**(1/3) - (q_val / 2 + sqrt)**(1/3)
    # print(root**3 + b_val * root**2 + c_val * root + d_val, np.max(np.abs(root)))
    # raise RuntimeError
    return np.real(root) * scaling


def compute_2d_band_empty(kax, kay, nx_0, ny_0, a, m, hbar, e):
    x_expr = (kax - 2 * np.pi * nx_0)**2
    y_expr = (kay - 2 * np.pi * ny_0)**2
    return hbar**2 / (2 * m * e * a**2) * (x_expr + y_expr)


def get_bands(method, ka, *args, n_bands=2):
    all_bands = [method(ka, n_0, *args)
                 for n_0 in range(-n_bands+1, n_bands)]
    bands = [all_bands[n_bands-1]]     # zero-th band
    for i in range(1, n_bands):
        band_left, band_right = all_bands[n_bands-1-i], all_bands[n_bands-1+i]
        idx = (band_left > band_right)
        lower = np.copy(band_left)
        upper = np.copy(band_right)
        lower[idx] = band_right[idx]
        upper[idx] = band_left[idx]
        bands.append(lower)
        bands.append(upper)

    return bands


def plot_1d_bands(m, hbar, e):
    a = 2e-9
    v_0 = 1.5e-19
    n_k = 200
    n_bands = 2

    # k = np.linspace(-np.pi/a, np.pi/a, n_k)
    # k = np.linspace(-2/a, 2/a, n_k)
    k = np.linspace(0/a, np.pi/a, n_k)
    ka = k * a

    bands_0 = get_bands(compute_1d_band_empty, ka, 
                        a, m, hbar, e, n_bands=n_bands)
    bands_1 = get_bands(compute_1d_band_periodic, ka, 
                        a, m, hbar, e, v_0, n_bands=n_bands)
    bands_2 = get_bands(compute_1d_band_periodic_cbrt, ka, 
                        a, m, hbar, e, v_0, n_bands=n_bands)

    fig, ax = plt.subplots()
    ax.set_xlim(ka[0], ka[-1])
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(r"$\epsilon_\nu(ka) / \mathrm{eV}$")
    mpl_special.format_ticklabels(ax)

    for i in range(len(bands_0)):
        col = ax._get_lines.get_next_color()
        ax.plot(ka, bands_0[i], c=col, ls='--', alpha=0.5)
        ax.plot(ka, bands_1[i], c=col)
        # ax.plot(ka, (bands_2[i] - 1.5) / 3, c=col, ls='--')
        ax.plot(ka, bands_2[i], c=col, ls='--')

    mpl_special.polish(fig, ax)


def plot_2d_bands(m, hbar, e):
    a = 1e-9
    v_0 = 0.0e-19
    n_k = 200

    eps_fermi = np.pi * hbar**2 / (m*e) * (2.3 / a**2)
    kx = np.linspace(0/a, np.pi/a, n_k)
    kax = kx * a

    fig, ax = plt.subplots()
    ax.set_xlim(kax[0], kax[-1])
    ax.set_xlabel(r"$k_xa$")
    ax.set_ylabel(r"$\epsilon_\nu(k_xa,0) / \mathrm{eV}$")
    mpl_special.format_ticklabels(ax)
    
    ax.axhline(eps_fermi, c='k', ls='--', label=r"$\epsilon_F$")
    for [nx, ny] in [(0, 0), (1, 0), (0, 1)]:
        col = ax._get_lines.get_next_color()
        band = compute_2d_band_empty(kax, np.zeros_like(kax), nx, ny, a, m, hbar, e)
        ax.plot(kax, band, c=col, label=f"$(n_x, n_y) = ({nx}, {ny})$")

    ax.legend()
    mpl_special.polish(fig, ax)


def plot_2d_bands_extended(m, hbar, e):
    a = 1e-9
    v_0 = 0.0e-19
    n_k = 200

    kx = np.linspace(0/a, 3*np.pi/a, n_k)
    kax = kx * a

    fig, ax = plt.subplots()
    ax.set_xlim(kax[0], kax[-1])
    ax.set_xlabel(r"$k_xa$")
    ax.set_ylabel(r"$\epsilon_\nu(k_xa,0) / \mathrm{eV}$")
    mpl_special.format_ticklabels(ax)

    for [nx, ny] in [(0, 0), (1, 0), (0, 1)]:
        col = ax._get_lines.get_next_color()
        band = compute_2d_band_empty(kax, np.zeros_like(kax), nx, ny, a, m, hbar, e)
        ax.plot(kax, band, c=col, label=f"$(n_x, n_y) = ({nx}, {ny})$")

    ax.legend()
    mpl_special.polish(fig, ax)


def main():
    """Hauptprogramm"""
    print(__doc__)
    m = 9.1093837015e-31
    hbar = 1.054571817e-34
    e = 1.6022e-19
    
    # plot_1d_bands(m, hbar, e)
    plot_2d_bands(m, hbar, e)
    # plot_2d_bands_extended(m, hbar, e)

    return 0

if __name__ == "__main__":
    main()
