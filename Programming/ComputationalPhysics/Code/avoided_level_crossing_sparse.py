#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avoided level crossings for an asymmetric double well potential
"""

import functools
import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=True)


def get_potential(mode='harmonic oscillator'):
    """Generate different potentials for a 'mode' specifier"""
    if mode == 'harmonic oscillator':
        def potential(x):
            """Harmonic oscialltor with fixed amplitude of 1/2"""
            return 0.5 * x**2
    elif mode == 'double well':
        def potential(x, a_asym):
            """Asymmetric double well potential"""
            return x**4 - x**2 - a_asym * x
    else:
        msg = f"Mode was {mode}, but should be one of ['harmonic oscillator', 'double well']"
        raise ValueError(msg)

    return potential


def get_e_val_e_vec(v_vals, heff, dx=1.0, min_e_val=-np.inf, max_e_val=np.inf, 
                    e_val_only=False):
    """Compute the eigenvalues 'e_val' and eigenvectors 'e_vec' of a discretized
    hamilton matrix.
    The Eigenvectors are normalized by multiplying with '1 / sqrt(dx)'.
    Note that the rows of the matrix 'e_vec' correspond to the eigenvectors.
    """
    num_x = v_vals.size
    zeff = heff**2 / (2 * dx**2)
    off_diag = np.full(num_x - 1, -zeff)
    diag = v_vals + 2 * zeff
    
    if e_val_only:
        e_val = eigh_tridiagonal(diag, off_diag, eigvals_only=True)
        e_val = e_val[(min_e_val <= e_val) & (e_val <= max_e_val)]
        return e_val
    
    e_val, e_vec = eigh_tridiagonal(diag, off_diag)
    indx = ((e_val <= max_e_val) & (e_val >= min_e_val))
    return e_val[indx], e_vec.T[indx] / np.sqrt(dx)


def get_e_val_matrix(heff, v_vals, dx=1.0, max_e_val=np.inf):
    """Compute the eigenvalues for all given 'heff'"""
    e_val_matrix = np.full((heff.size, v_vals.size), np.inf)
    for i in range(heff.size):
        e_val = get_e_val_e_vec(v_vals, heff[i], dx, 
                                max_e_val=max_e_val, e_val_only=True)
        e_val_matrix[i, :e_val.size] = e_val
    return e_val_matrix


def iterative_alc_search(heff_limits, v_vals, dx, levels: tuple[int, int],
                         num_heff=10, atol=1e-7):
    """Returns the 'heff' limits where an avoided level crossing was found.
    'levels' must be a tuple containing the eigenvalue-indices of the crossing
    """
    while heff_limits[1] - heff_limits[0] > atol:
        heff_min, heff_max = heff_limits
        heff = np.linspace(heff_min, heff_max, num_heff+1)    # first heff interval
        e_val_matrix = get_e_val_matrix(heff, v_vals, dx)
        e_val1, e_val2 = e_val_matrix[:, levels[0]], e_val_matrix[:, levels[1]]
        abs_diff = np.abs(e_val1 - e_val2)
        i_min = np.argmin(abs_diff)          # smallest energy gap index

        if i_min == abs_diff.size - 1:
            i_min -= 1
        elif (i_min != 0 and abs_diff[i_min-1] < abs_diff[i_min+1]):
            i_min -= 1

        heff_limits = heff[i_min:i_min+2]
    return heff_limits


def plot_e_val_e_vec(x, v_vals, e_val, e_vec, max_e_val=0.1, scale_factor=1.0,
                     ylim_padding=0.03):
    """Plot all eigenvectors 'e_vec' at a height corresponding to their eigenvalue
    'e_val' for energies below 'max_e_val' over the array 'x'.
    Also plot the array 'v_vals' corresponding to the discretized potential.
    """

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(r"$V(x)$, $\psi_n(x)$")

    ax.set_xlim(x[0], x[-1])
    ax.plot(x, v_vals, c='k')
    for i in range(e_val.size):
        e_n_val = e_val[i]
        e_n_vec = e_vec[i]
        line = ax.plot(x, np.abs(e_n_vec)**2 * scale_factor + e_n_val)
        ax.axhline(e_n_val, lw=0.7, ls='--', c=line[0].get_color())

    ylim = np.array([np.min(v_vals), e_val[-1] + np.max(e_vec[-1])**2 * scale_factor])
    ylim += (ylim[1] - ylim[0]) * ylim_padding * np.array([-1, 1])
    ax.set_ylim(ylim)
    special.polish(fig, ax)


def plot_alc(heff, e_val_matrix, max_e_val=0.1, ylim=None, levels=None):
    """Plot the avoided le_vecel crossing of the eigenvalues below 'max_e_val'
    for all given'heff'
    """
    if levels is not None:
        e_val_matrix = e_val_matrix[:, levels]

    fig, ax = plt.subplots()
    ax.set_xlim(heff[0], heff[-1])
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(r"$\hbar_{\mathrm{eff}}$")
    ax.set_ylabel(r"$E_n(\hbar_{\mathrm{eff}})$")
    for i in range(e_val_matrix.shape[1]):
        e_val = e_val_matrix[:, i]
        if e_val[-1] > max_e_val:
            break

        ax.plot(heff, e_val)

    special.polish(fig, ax)


def main():
    print(__doc__)
    modes = {0: 'harmonic oscillator', 1: 'double well'}

    mode = 1
    potential = get_potential(modes[mode])
    max_e_val = 0.1
    scale_factor = 0.015

    if mode == 0:
        heff = 1.0          # effective hbar
        xmin = -5
        xmax = 5
        num_x = 400

    elif mode == 1:
        heff = 0.07          # effective hbar
        a_asym = 0.06        # asymmetry parameter of the double well potential
        xmin = -1.5#-20.5
        xmax = 1.5#20.5
        num_x = 500#3500
        potential = functools.partial(potential, a_asym=a_asym)

    x, dx = np.linspace(xmin, xmax, num_x+1, endpoint=False, retstep=True)
    x = x[1:]       # remove the first entry
    v_vals = potential(x)


    # double potential well for fixed 'heff'
    e_val, e_vec = get_e_val_e_vec(v_vals, heff, dx, max_e_val=max_e_val)
    plot_e_val_e_vec(x, v_vals, e_val, e_vec, max_e_val=max_e_val, 
                      scale_factor=scale_factor)

    # # avoided level crossing for many 'heff'
    # # heff_arr = np.linspace(0.01, 0.025, 200)
    # heff_arr = np.linspace(0.01, 0.025, 200)
    # e_val_matrix = get_e_val_matrix(heff_arr, v_vals, dx, max_e_val=max_e_val)
    # plot_alc(heff_arr, e_val_matrix)

    # heff_limits = [0.014, 0.015]
    # levels = (3, 4)         # the two eigenvalue-indices for the crossing
    # # heff_limits = [0.04, 0.05]
    # # levels = (1, 2)         # the two eigenvalue-indices for the crossing
    # heff_lim = iterative_alc_search(heff_limits, v_vals, dx=dx, levels=levels,
    #                                 num_heff=5, atol=1e-13)
    # heff_arr = np.linspace(heff_lim[0], heff_lim[1], 50)
    # e_val_matrix = get_e_val_matrix(heff_arr, v_vals, dx, max_e_val=max_e_val)
    # plot_alc(heff_arr, e_val_matrix, levels=levels)

    return 0

if __name__ == "__main__":
    main()
