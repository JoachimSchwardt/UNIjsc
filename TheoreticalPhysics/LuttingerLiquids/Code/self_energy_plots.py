#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self energy visualization for different models
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_special
# from residue_integrals import g_func
# from ep_visual import hamilton_2nd_order


PATH = "../MA_Latex/figures/"


def self_energy_first_model_v4(k_val, omega=0, u_minus=0.13, beta=3.2, u_val=1.0):
    arg = beta * k_val * u_val / 2
    prefactor = -u_minus * np.cosh(arg)
    matrix_structure = np.array([(omega / (k_val * u_val) + 1) * np.exp(-arg),
                                 (omega / (k_val * u_val) - 1) * np.exp(arg)])
    return prefactor * matrix_structure


def self_energy_to_energies(self_energy_func, k_vals, omega, *args):
    self_energy = self_energy_func(k_vals, omega, *args)
    determinant = -k_vals**2 - self_energy[0] * self_energy[1]     # assumes diagonals are 0
    energy = (-determinant + 0j)**0.5     # need complex type to get complex sqrt results
    return energy, -energy


def plot_complex(axis, x_vals, y_vals, **kwargs):
    y_vals = np.asarray(y_vals)
    if y_vals.ndim == 1:
        y_vals = np.expand_dims(y_vals, axis=0)
    colors = mpl_special.Colors()
    col_real = colors.get_color()
    col_imag = colors.get_color()
    for label, color in zip(["real", "imag"], [col_real, col_imag]):
        axis.plot([], [], c=color, label=label, **kwargs)
    for y_val in y_vals:
        axis.plot(x_vals, y_val.real, c=col_real, **kwargs)
        axis.plot(x_vals, y_val.imag, c=col_imag, **kwargs)
    axis.legend()


def convert_to_contiguous_arrays(arr1, arr2):
    """Given two arrays attempts to make the values inside as contiguous as possible.
    arr1 = np.arange(15)
    arr2 = np.arange(-15, 0)
    arr1[6], arr2[6] = arr2[6], arr1[6]
    convert_to_contiguous_arrays(arr1, arr2) --> recovers original
    """
    val1, val2 = arr1[0], arr2[0]
    for j in range(1, arr1.size):
        if np.abs(arr1[j] - val1) > np.abs(arr1[j] - val2):
            arr1[j], arr2[j] = arr2[j], arr1[j]
        val1, val2 = arr1[j], arr2[j]
    return arr1, arr2


def eigenvalues_2x2(arr):
    """
    omega = 0.14; u_minus = 0.05; beta = 1.6; K=0.8
    k = np.linspace(-0.2, 0.2, 300)
    g = np.zeros((2, 2, k.size), dtype=complex)
    M = (K + 1/K - 2)/2
    g[0,0] = 2**M / (omega + k + 1j * np.pi/beta * M)
    g[1,1] = 2**M / (omega - k + 1j * np.pi/beta * M)
    g[1,0] = (u_minus * 2**(K/2 + 3/(2*K) - 2)
              / (np.pi**2/beta**2 * (-1j*beta/np.pi*omega + M)**2 + k**2)
              * (1j* omega - np.pi/beta * M) / (np.pi/beta * (1-K) +1j* omega))
    g[0,1] = g[1,0]
    prefactor_2nd = 2**(M + 2*K) * u_minus**2 * beta**3 / (8*np.pi**2*K**2)
    g[0,0] += prefactor_2nd * omega / (omega - k + 1j * np.pi/beta * M)
    g[1,1] += -prefactor_2nd * omega / (omega + k + 1j * np.pi/beta * M)
    energies = eigenvalues_2x2(g)
    """
    # det = arr[0, 0] * arr[1, 1] - arr[1, 0] * arr[0, 1]
    # offset = -(arr[0, 0] + arr[1, 1]) / (2 * det)
    # root = np.sqrt(((arr[0, 0] - arr[1, 1]) / 2)**2 + arr[1, 0] * arr[0, 1]) / (2 * det)
    # return convert_to_contiguous_arrays(offset - root, offset + root)
    offset = (arr[0, 0] + arr[1, 1]) / 2
    root = np.sqrt(((arr[0, 0] - arr[1, 1]) / 2)**2 + arr[1, 0] * arr[0, 1])
    return offset - root, offset + root


def savefig(fig, name, datatype=".pdf"):
    plt.pause(0.01)                 # TODO: Fix embed_labels to apply once on startup...
    fig.savefig(name + datatype)
    plt.close("all")
    print("Saved figure: ", name + datatype)


def plot_first_model_v4(save=False, u_minus=0.13, beta=4.5, u_val=1.0):
    k_vals = np.linspace(-1, 1, 500)
    omega = 0
    args = (u_minus, beta, u_val)
    energies = self_energy_to_energies(self_energy_first_model_v4, k_vals, omega, *args)

    fig, ax = plt.subplots()
    ax.set_xlabel("$k$")
    ax.set_ylabel("$E$")
    ax.set_xlim(k_vals[0], k_vals[-1])
    plot_complex(ax, k_vals, energies)
    mpl_special.embed_labels(fig, ax)
    if save:
        params = f"_beta{beta:.1f}"
        savefig(fig, PATH + self_energy_first_model_v4.__name__ + params)


def plot_first_model_v6(save=False, omega=0.14, u_minus=0.05, beta=1.6, K=0.8):
    """ OLD VERSION ::
    u_minus=0.13, beta=4.5, K=1.2
    k_vals = np.linspace(-0.05, 0.05, 51)
    K_0 = (K + 1/K - 2) / 4
    K_int = (K - 1) / 2
    g_ll = -np.array([g_func(0, k_val, K=K_0) for k_val in k_vals]) / (2*beta)
    g_rr = -g_ll.conj()
    g_int = np.array([g_func(0, k_val, K=K_int) for k_val in k_vals])
    g_rl = np.abs(g_int)**2 * u_minus / (4*beta**2)
    fixed_part = 0 - 0.5 / g_ll - 0.5 / g_rr
    root = np.sqrt(4*g_rl**2 + (g_rr - g_ll)**2) / (2 * g_rr * g_ll)
    energies = fixed_part - root, fixed_part + root
    """
    k_vals = np.linspace(-0.2, 0.2, 300)
    g_r = np.zeros((2, 2, k_vals.size), dtype=complex)
    M = (K + 1/K - 2)/2
    g_r[0,0] = 2**M / (omega + k_vals + 1j * np.pi/beta * M)
    g_r[1,1] = 2**M / (omega - k_vals + 1j * np.pi/beta * M)
    g_r[1,0] = (u_minus * 2**(K/2 + 3/(2*K) - 2)
              / (np.pi**2/beta**2 * (-1j*beta/np.pi*omega + M)**2 + k_vals**2)
              * (1j* omega - np.pi/beta * M) / (np.pi/beta * (1-K) +1j* omega))
    g_r[0,1] = g_r[1,0]
    prefactor_2nd = 2**(M + 2*K) * u_minus**2 * beta**3 / (8*np.pi**2*K**2)
    g_r[0,0] += prefactor_2nd * omega / (omega - k_vals + 1j * np.pi/beta * M)
    g_r[1,1] += -prefactor_2nd * omega / (omega + k_vals + 1j * np.pi/beta * M)
    energies = eigenvalues_2x2(g_r)

    fig, ax = plt.subplots()
    ax.set_xlabel("$k$")
    ax.set_ylabel("$E$")
    ax.set_xlim(k_vals[0], k_vals[-1])
    plot_complex(ax, k_vals, energies)
    mpl_special.embed_labels(fig, ax)
    if save:
        params = f"_beta{beta:.1f}_K{K:.2f}_Um{u_minus:.2f}_omega{omega:.2f}"
        savefig(fig, PATH + "complex_E_first_model_v6" + params)
        
    
# def plot_first_model_v6_energies_ep():
#     k_vals = np.linspace(-0.4, 0.4, 400, endpoint=False)
#     k_vals, _, hamilton = hamilton_2nd_order(k_vals, [-0.1046], beta=1.1, K=0.6, u_minus=0.25)
#     energies = np.array([eigenvalues_2x2(hamilton[i, 0]) for i in range(k_vals.size)]).T
#     energies = convert_to_contiguous_arrays(*energies)
#     fig, ax = plt.subplots()
#     ax.set_xlabel("$k$")
#     ax.set_ylabel("$E$")
#     ax.set_xlim(k_vals[0], k_vals[-1])
#     plot_complex(ax, k_vals, energies)
#     mpl_special.embed_labels(fig, ax)


def u_minus_zero_comparison():
    """
    Comparison ::
    plot_complex(plt.subplots()[1], k_values-np.pi, np.roll(h_eff_k_w[:,500,1,0], -40))
    energies_num = np.array([eigenvalues_2x2(h_eff_k_w[i,500]) for i in range(k_values.size)])
    plot_complex(plt.subplots()[1], k_values - np.pi, np.roll(energies_num.T, -40, axis=1))
    energies_num = np.array([eigenvalues_2x2(h_eff_k_w[i,600]) for i in range(k_values.size)])
    plot_complex(plt.subplots()[1], k_values - np.pi, np.roll(energies_num.T, -40, axis=1))
    ### f"complex_E_numeric_UA1.5_UB1.5_omega0.5"
    """
    ## u_minus == 0 --> comparison analytic vs. numeric
    k_values = np.linspace(0, 2*np.pi, 80, endpoint=False)
    U = np.array([[1,1],[1,-1]]) / np.sqrt(2)
    u_minus=0.05
    omega=0#0.1#0
    beta=1.1
    K=0.6
    M=(K+1/K-2)/2
    k_vals = k_values[1:] - np.pi
    # g_ll = np.array([-g_func(omega, k, beta, M) / (2*beta) for k in k_vals])
    g_ll = np.array([2**M / (omega + k + 1j * M * np.pi/beta) for k in k_vals])
    g_rr = np.copy(g_ll)[::-1]    # assumes symmetry in k_vals (-3, ..., +3, equidistant)
    g_rl = np.array([u_minus * 2**(K/2 + 3/(2*K) - 2)
                     / (np.pi**2/beta**2 * (-1j*beta/np.pi*omega + 1/(2*K) + K/2 - 1)**2 + k**2)
                     * (np.pi/beta * (1-K/2 - 1/(2*K)) + 1j * omega)
                     / (np.pi/beta * (1-K) + 1j * omega) for k in k_vals])
    g_lr = np.copy(g_rl)
    G_lr = np.array([np.array([[g_rr[i], g_rl[i]], [g_lr[i], g_ll[i]]])
                     for i in range(k_vals.size)])
    H_lr = np.array([omega - np.linalg.inv(G_lr[i]) for i in range(k_vals.size)])
    H_ab = U @ H_lr @ U
    energies = np.array([eigenvalues_2x2(H_ab[i]) for i in range(k_vals.size)])
    plot_complex(plt.subplots()[1], k_vals, energies.T)
    # params = f"_Um{u_minus:.2f}_K{K:.2f}_beta{beta:.1f}_omega{omega:.2f}_exact"
    # savefig(plt.gcf(), PATH + "complex_E" + params)


def plot_numeric_paper_initial():
    """Dispersion relation for omega=0 for free and interacting Hamiltonian from 2021 paper"""
    figsize = mpl_special.figsize.set_figsize(0.45 * 390.745)    # \textwidth in beamer slide
    for key in ["initial", "final"]:
        data = np.load(f"data_joachim/h_eff_{key}.npy")
        k_vals = np.linspace(-np.pi, np.pi, data.shape[0], endpoint=False)
        energies = np.array([eigenvalues_2x2(data[i]) for i in range(k_vals.size)])
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.set_ticks([]); ax.yaxis.set_ticks([])
        ax.set_xlabel("$k$")
        ax.set_ylabel("$E$", rotation=0, ha="right", va="center")
        ax.set_xlim(k_vals[0], k_vals[-1])
        plot_complex(ax, k_vals, energies.T)
        savefig(fig, PATH + "sketch_complex_E_" + key)
        
    data = np.load("data_joachim/h_eff_initial.npy")
    energies = np.array([eigenvalues_2x2(data[i]) for i in range(k_vals.size)])
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_ticks([]); ax.yaxis.set_ticks([])
    ax.set_xlabel("$k$")
    ax.set_ylabel("$E$", rotation=0, ha="right", va="center")
    ax.set_xlim(k_vals[0], k_vals[-1])
    colors = mpl_special.Colors()
    col_real = colors.get_color()
    col_imag = colors.get_color()
    for energy in energies.T:
        ax.plot(k_vals, energy.real, c=col_real, alpha=0.5)
        ax.plot(k_vals, energy.imag, c=col_imag)
    offset = 20
    v_f = ((energies[k_vals.size//2 + offset, 0] - energies[k_vals.size//2 - offset, 1])
           / (k_vals[k_vals.size//2 + offset] - k_vals[k_vals.size//2 - offset]))
    for sign in [+1, -1]:
        ax.plot(k_vals, sign * v_f.real * k_vals, ls='--', c=ax.lines[0].get_color())
    ax.plot(0, 0, c=ax.lines[0].get_color(), label="real")
    ax.plot(0, 0, c=ax.lines[1].get_color(), label="imag")
    ax.legend()
    savefig(fig, PATH + "sketch_complex_E_" + key + "_linear")

def main():
    print(__doc__)
    # plot_first_model_v4(1, beta=1.2)
    # plot_first_model_v4(1, beta=6.2)
    return 0


if __name__ == "__main__":
    main()
