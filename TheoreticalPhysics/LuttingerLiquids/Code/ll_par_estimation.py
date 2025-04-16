#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:21:40 2023

@author: joachim
"""

import numpy as np
import green_toolkit
from green_toolkit import sigma_x, sigma_y
import matplotlib.pyplot as plt
from gf_h1h3_v3 import j_bn
from thesis_gf import green_perturbative, green_eff_ua, get_free_K0
from ep_visual import get_complex_e, ep_size_iter, energies_to_ep_size, get_plot_setup, get_pars_string
from self_energy_plots import plot_complex, savefig
from green_functions import green_numeric
from scipy.optimize import curve_fit, minimize
import mpl_special


PATH = "../MA_Latex/figures/"
U = green_toolkit.U


def compare_num_analytic_free_gf():
    """
    Observations:
        full K <-> 1/K invariance in analytic result (sqrt(2) vs. 1/sqrt(2))
    """
    k_num, omega_num, green_num = green_numeric(1.5, 1.5)
    hamilton_num = green_toolkit.hamilton_from_green(omega_num, green_num)
    hamilton_num_LR = U@hamilton_num@U
    green_num_LR = green_toolkit.green_from_hamilton(omega_num, hamilton_num_LR)
    beta=1.5; K=1.3; v=0.5; w=3; num_params = {"order" : 0}
    green = green_perturbative(k_num, omega_num, beta, K, v, w=w, num_params=num_params)
    omega_ind=omega_num.size//2
    A=0; B=0
    # beta=1.5; K=1.3; v=0.5; w=3; num_params = {"order" : 0}
    def free_gf_fit_imag(k_vals, beta_v=1.1, K=0.6, C=3, omega=0):
        """Free SP Green's function; fit routine"""
        k_vals = np.asarray(k_vals)
        M = (K + 1/K - 2) / 4
        prefactor = C   # C = -beta * w**(2*M)/(2*np.pi**2)
        g_rr = prefactor * j_bn(-beta_v/np.pi * k_vals, omega, M, n=1)
        return np.abs(g_rr.imag)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    k_indx = (k_num > 0) & (k_num < 2.0)
    xdata = k_num[k_indx]
    ydata = np.abs(green_num_LR[k_indx,omega_ind,A,B].imag)
    ax.plot(xdata, ydata, label="numeric", marker="o")
    # for beta, K, v in zip([0.7, 0.7], [0.7, 0.6], [0.5, 0.5]):
    #     green = green_perturbative(k_num, [omega_num[omega_ind]], beta, K, v,
    #                                w=w, num_params=num_params)
    #     #ax.plot(k_num[k_indx], np.abs(green[k_indx,0,A,B].imag), label=fr"$\beta={beta}, K={K}, v={v}, w={w}$", marker="o")
    #     green = free_gf_fit_imag(k_num[k_indx], beta, K, v)
    #     ax.plot(k_num[k_indx], np.abs(green), label=fr"$\beta={beta}, K={K}, v={v}$", marker="o")
    par, cov = curve_fit(free_gf_fit_imag, xdata, ydata, p0=(0.35, 0.7, 1),
                         bounds=([0.01, 0.01, 0], [20, 1, np.inf]))
    beta_v, K, C = par
    ax.plot(xdata, free_gf_fit_imag(xdata, beta_v, K, C),
            label=fr"$\beta v={beta_v : .3f}, K={K : .3f}, C={C : .3f}$", marker="o")
    ax.legend()


def compare_num_analyic_self_energy():
    k_num, omega_num, green_num, hamilton_num = green_numeric(0.3, 0.0)
    # self_energy_num = np.round(self_energy_num, decimals=8)
    # fig, ax = plt.subplots(2,2)
    # for ai in [0,1]:
    #     for bi in [0,1]:
    #         ax[ai,bi].set_xlabel("$k$")
    #         plot_complex(ax[ai,bi], k_num, self_energy_num[:,ai,bi])
    hamilton_num0 = np.array([-(1-np.cos(k)) * sigma_x - 0.5*np.sin(k) * sigma_y for k in k_num])
    self_energy_num = np.array([hamilton_num[ki, omega_num.size//2] - hamilton_num0[ki]
                                for ki in range(k_num.size)])
    beta=15; K=2.5; v=0.5; w=2.4
    green = green_perturbative(k_num, [0], beta, K, v, w=w, num_params={"order" : 0})
    hamilton = green_toolkit.hamilton_from_green([0], U@green@U)[:,0]
    self_energy = np.array([hamilton[ki] for ki in range(k_num.size)]) + self_energy_num[0,0,0]
    fig, ax = plt.subplots()
    ax.set_xlim(k_num[0], k_num[-1])
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\Sigma_{\mathrm{AA}}(k)$")
    plot_complex(ax, k_num, self_energy_num[:,0,0])
    plot_complex(ax, k_num, self_energy[:,0,0], ls='', marker='o')
    mpl_special.embed_labels(fig, ax)


def ll_par_from_ep_size(ep_delta, ep_offset, beta=1, K=0.6, g=3, vf=0.5, a=1, w=np.pi,
                        num_params={"order" : 1, "mmax" : 5, "mmaxp" : 5, "lmax" : 5, "numkp" : 48}):
    """estimate the LL parameters for given (ep_delta, ep_offset) -> (K, g, v)"""
    def function(x):
        K, g = x
        args = (beta, [K], vf / K, g, a, w, num_params)
        [ep_delta_val], [ep_offset_val], _ = ep_size_iter(green_perturbative, args=args)
        return np.abs(ep_delta_val - ep_delta)**2 + np.abs(ep_offset_val - ep_offset)**2
    result = minimize(function, [K, g])
    K, g = result.x
    return K, vf / K, g


def get_w_mod(w, beta, v, alpha=None):
    if alpha is not None:
        return np.pi*alpha/beta/v
    return w


def ll_par_from_ua_eq_ub(u_a=0.8, beta=1, K=0.7, v=0.5, w=np.pi, alpha=None, decay_rate=5):
    """estimate the LL parameters for given (u_a==u_b) -> (K, v)
    if 'alpha' is 'None' use the given 'w', else 'alpha' takes precedence and 'w = w(alpha)'."""
    k_num, omega_num, green_num, h_num = green_numeric(u_a, u_a, beta=beta)
    U = green_toolkit.U
    green_num_LR = U @ green_num @ U.conj()
    def function(x):
        K, v = x
        w_mod = get_w_mod(w, beta, v, alpha)
        green = green_perturbative(k_num, [0], beta, K=K, v=v, w=w_mod, num_params={"order" : 0})
        error = np.abs(green[:, 0, 0, 0] - green_num_LR[:, omega_num.size//2, 0, 0])
        weight = 1 / (1 + decay_rate * (k_num / k_num[-1])**2)
        return np.sum(error * weight)
    result = minimize(function, [K, v], bounds=[(0.1, 1.0), (0.01, 100)])
    K, v = result.x
    return K, v, get_w_mod(w, beta, v, alpha)


def compare_na_complex_e(u_a=1.5, u_b=-1.1, beta=0.7404, K=0.7, vf=0.5, g=1, a=1.0, w=np.pi,
                         num_params={"order" : 1, "mmax" : 5, "mmaxp" : 5, "lmax" : 5, "numkp" : 48},
                         k_max=0.5, k_count=301, save=False):
    """Compare numerical data for complex eigenvalues to LL perturbation theory result.
    The LL parameters are estimated using the data for the 'ep_size' at k=0"""
    k_num, omega, green, hamilton = green_numeric(u_a, u_b, beta=beta)
    omega_indx = omega.size//2
    omega_val = omega[omega_indx]
    energies_num = get_complex_e(hamilton[:, omega_indx])
    ep_delta, ep_offset = energies_to_ep_size(np.array(energies_num).T[k_num.size//2])
    K, v, g = ll_par_from_ep_size(ep_delta, ep_offset, beta, K, g, vf, a, w, num_params)
    args = (beta, K, v, g, a, w, num_params)
    fig, [ax] = get_plot_setup(xscale="linear", yscale="linear", xlabel="$k$", ylabel=r"$E_\pm$")
    ax.set_xlim(-k_max, k_max)
    indx = ~((k_num > k_max) | (k_num < -k_max))
    k_vals = np.linspace(-k_max, k_max, k_count)
    green = green_perturbative(k_vals, [omega_val], *args, delta=1e-14)
    hamilton = green_toolkit.hamilton_from_green([omega_val], green)
    energies = get_complex_e(hamilton[:, 0])
    plot_complex(ax, k_vals, energies)
    k_num = k_num[indx]; e_num = np.array(energies_num).T[indx].T
    for e_band in e_num:
        for i, attr in enumerate(["real", "imag"]):
            ax.plot(k_num, getattr(e_band, attr), ls='', marker='o', c=ax.lines[i].get_color())
    ax.legend(loc="upper center")
    filename = "compare_na_complex_e_" + get_pars_string((u_a, u_b, beta, K, v, g, w, num_params),
                                                         ["Ua", "Ub", "beta", "K", "v", "g", "w", "mmln"])
    print("Created figure", filename)
    if save:
        savefig(fig, PATH + filename)


def compare_na_complex_e_eff_ua(u_a=0.6):
    beta = 0.2; vf = 0.5; a = 1; alpha = 1
    K = get_free_K0(u_a, vf, a, alpha**2/(a**2+alpha**2))
    beta = 0.6; K = get_free_K0(u_a, vf, a)
    k_max = 0.6; k_count = 601
    k_vals = np.linspace(-k_max, k_max, k_count)
    omega_vals = np.array([0.0])
    v = vf / K
    w = np.pi*alpha/beta/v
    args = (beta, K, v, w)
    labels = ["beta", "K", "v", "w"]
    green = green_eff_ua(k_vals, omega_vals, *args)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    fig, [ax] = get_plot_setup(xscale="linear", yscale="linear", xlabel="$k$", ylabel=r"$E_\pm$")
    energies = get_complex_e(hamilton[:, 0])
    k_num, omega, green, hamilton = green_numeric(u_a, 0)
    omega_indx = omega.size//2
    energies_num = get_complex_e(hamilton[:, omega_indx])
    indx = ~((k_num > k_max) | (k_num < -k_max))
    plot_complex(ax, k_vals, energies)
    k_num = k_num[indx]; e_num = np.array(energies_num).T[indx].T
    for e_band in e_num:
        for i, attr in enumerate(["real", "imag"]):
            ax.plot(k_num, getattr(e_band, attr), ls='', marker='o', c=ax.lines[i].get_color())
    ax.set_xlim(k_vals[0], k_vals[-1])
    mpl_special.embed_labels(fig, ax)
    filename = "compare_na_eff_ua_complex_e_" + get_pars_string((u_a, beta, K, v, w),
                                                                ["Ua", "beta", "K", "v", "w"])


def compare_na_gf_example():
    num_params={"order" : 1, "mmax" : 5, "mmaxp" : 5, "lmax" : 5, "numkp" : 48}
    u_a=1.5; u_b=-1.1; beta=0.7404; K=0.7; vf=0.5; g=1; a=1.0; w=np.pi; k_count=301
    k_num, omega, green_num, hamilton_num = green_numeric(u_a, u_b)
    omega_indx = omega.size//2
    omega_val = omega[omega_indx]
    energies_num = get_complex_e(hamilton_num[:, omega_indx])
    ep_delta, ep_offset = energies_to_ep_size(np.array(energies_num).T[k_num.size//2])
    K, v, g = ll_par_from_ep_size(ep_delta, ep_offset, beta, K, g, vf, a, w, num_params)
    args = (beta, K, v, g, a, w, num_params)
    k_vals = np.linspace(k_num[0], k_num[-1], k_count)
    green = green_perturbative(k_vals, [omega_val], *args, delta=1e-14)
    U2 = 1/np.sqrt(2) * np.array([[1,1j],[1j,1]])
    green_AB = U2.conj() @ green @ U2
    fig,ax=plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            plot_complex(ax[i,j], k_num, green_num[:,omega_indx,i,j], ls='', marker='o')
            plot_complex(ax[i,j], k_vals, green_AB[:,0,i,j])
            ax[i,j].set_xlim(k_num[0], k_num[-1])


def compare_na_gf_ua_eq_ub(u_a=0.8, w=np.pi, alpha=None):
    beta=1
    k_num, omega_num, green_num, h_num = green_numeric(u_a, u_a, beta=beta)
    omega_indx = omega_num.size//2
    K, v, w = ll_par_from_ua_eq_ub(u_a, beta, w=w, alpha=alpha, decay_rate=5)
    print(f"LL parameters for UA==UB: K={K:.3f} and v={v:.3f}")
    k_vals = np.linspace(k_num[0], k_num[-1], 300)
    green = green_perturbative(k_vals, [0], beta, K=K, v=v, w=w, num_params={"order" : 0})
    U = green_toolkit.U
    fig, ax = plt.subplots()
    plot_complex(ax, k_num, (U @ green_num @ U.conj())[:,omega_indx,0,0], ls='', marker='o')
    plot_complex(ax, k_vals, green[:,0,0,0])
    ax.set_xlim(k_num[0], k_num[-1])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$G_{\mathrm{RR}}(k)$")
    mpl_special.embed_labels(fig, ax)


def plot_compare_na_ll_par_ua_eq_ub(w=np.pi, alpha=None):
    beta = 1; vf = 0.5
    u_a_num = np.linspace(0.1, 1.5, 15)
    u_a = np.linspace(u_a_num[0], u_a_num[-1], 100)
    pars = []
    for u_a_val in u_a_num:
        K, v, w = ll_par_from_ua_eq_ub(u_a_val, beta, v=vf, w=w, alpha=alpha, decay_rate=5)
        print(f"LL parameters for UA==UB: K={K:.3f} and v={v:.3f}")
        pars.append([K, v, w])
    K, v, w = np.array(pars).T
    K0 = get_free_K0(u_a, vf)
    fig, ax = plt.subplots()
    ax.set_xlim(u_a[0], u_a[-1])
    ax.set_xlabel(r"$U_\mathrm{A}$")
    for label, num_data, analytic in zip(["$K$", "$v$"], [K, v], [K0, vf / K0]):
        line = ax.plot(u_a_num, num_data, ls='', marker='o', label=label)
        ax.plot(u_a, analytic, c=line[0].get_color())
    ax.legend()
    mpl_special.embed_labels(fig, ax)