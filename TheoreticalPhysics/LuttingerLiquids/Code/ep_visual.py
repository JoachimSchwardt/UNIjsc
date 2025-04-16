#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:00:51 2023

@author: ag_budich1
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_special
import green_toolkit
from green_functions import green_numeric
from self_energy_plots import plot_complex, savefig
from time import perf_counter as pc


PATH = "../MA_Latex/figures/"
ptr_line_color = 0.3
ptr_label_color = (0, 0, 0, 0.6)


def ep_contour_plot(ax, hamilton, k_vals, omega_vals):
    eqn1, eqn2 = green_toolkit.hamilton_2x2_ep_eqn(hamilton)
    color_gen = mpl_special.Colors()
    colors = [color_gen.get_color() for _ in range(2)]
    ax.contour(k_vals, omega_vals, eqn1.T, [0], colors=colors[0])
    ax.contour(k_vals, omega_vals, eqn2.T, [0], colors=colors[1])
    lines = [plt.Line2D([0], [0], color=color) for color in colors]
    labels = [r"$\textbf{d}_\text{r}^2 - \textbf{d}_\text{i}^2$",
              r"$\textbf{d}_\text{r}\cdot \textbf{d}_\text{i}$"]
    ax.legend(lines, labels)


def get_k_w_green(green_method=green_numeric, k_vals=None, omega_vals=None, args=()):
    """Plot Exceptional Points for a numeric dataset"""
    if green_method == green_numeric:
        k_vals, omega_vals, green = green_method(*args)
    else:
        if k_vals is None:
            k_vals = np.linspace(-np.pi, np.pi, 101)
        k_vals = np.asarray(k_vals)
        if omega_vals is None:
            omega_vals = np.linspace(-0.4, 0.4, 71)
        omega_vals = np.asarray(omega_vals)
        green = green_method(k_vals, omega_vals, *args)
    return k_vals, omega_vals, green


def plot_ep(k_vals, omega_vals, green):
    """Plot Exceptional Points for a numeric dataset"""
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega$")
    ep_contour_plot(ax, hamilton, k_vals, omega_vals)
    #mpl_special.format_ticklabels(ax)
    mpl_special.embed_labels(fig, ax)


def plot_energies():
    """Plot the complex energies"""
    beta=1.1; K=0.6; u=1.0; u_minus=0.05
    k_vals = np.linspace(-0.4, 0.4, 301)
    omega_vals = np.array([0.0])
    from green_functions_matsubara import green_perturbative
    green = green_perturbative(k_vals, omega_vals, beta, K, u, u_minus, order=1)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    energies = np.array([green_toolkit.eigenvalues_2x2(hamilton[i,0])
                         for i in range(k_vals.size)])
    energies = green_toolkit.convert_to_contiguous_arrays(energies.T[0], energies.T[1])
    plot_complex(plt.subplots()[1], k_vals, energies)

# from time import perf_counter as pc
def energy_function(green_function, k_val, omega=0, args=None, ratio_warning_threshold=0.8):
    # t1=pc()
    green = green_function([k_val], [omega], *args)
    # t2=pc()
    hamilton = green_toolkit.hamilton_from_green([omega], green)
    # t3=pc()
    energies = green_toolkit.eigenvalues_2x2(hamilton[0,0])
    # t4=pc()
    # print(f"GREEN: {t2-t1:.6f}, HAMILTON: {t3-t2:.6f}, E: {t4-t3:.6f}, K: {k_val:.9f}")
    ratio = np.abs(green[0][0][0,1] / green[0][0][0,0])
    if ratio_warning_threshold is not None and ratio > ratio_warning_threshold:
        print(f"WARNING: PTR = {ratio:.3f} exceeds threshold of {ratio_warning_threshold:.2f}")
    return energies, ratio

def ep_search(green_function, k_min=0, k_max=1, omega=0, args=None,
              ktol=1e-4, etol=1e-6, max_iter=30):
    """Search for the EP using interval splitting"""
    (e1_min, e2_min), r_min = energy_function(green_function, k_min, args=args)
    (e1_max, e2_max), r_max = energy_function(green_function, k_max, args=args)
    r_mean = r_max
    if np.abs((e1_min - e2_min).real) > etol or np.abs((e1_max - e2_max).imag) > etol:
        raise ValueError("EP is not within initial bounds!")
    for i in range(max_iter):
        k_mean = (k_min + k_max) / 2
        if (k_max - k_min) < ktol:
            print(f"Iteration {i} for {args=}")
            return k_mean, r_mean
        (e1_mean, e2_mean), r_mean = energy_function(green_function, k_mean, args=args)
        if np.abs((e1_mean - e2_mean).imag) > etol:
            k_min = k_mean
        else:
            k_max = k_mean
    print(f"Warning: Error tolerance {ktol:.2e} not achieved in {max_iter} iterations. "
          f"Final uncertainty approximately {k_max - k_min:.2e}.")
    return k_mean, r_mean

def find_iterable_argument(args):
    for indx, arg in enumerate(args):
        if isinstance(arg, (np.ndarray, list)):    # find the (assumed only) iterable argument
            iter_arg = arg                         # (e.g. 'beta' is array-like and others scalar)
            iter_indx = indx
            break
    return iter_arg, iter_indx

def get_modified_iterable_argument(args, indx):
    args_i = []
    for arg in args:
        if isinstance(arg, (np.ndarray, list)):
            args_i.append(arg[indx])
        else:
            args_i.append(arg)
    return args_i

def ep_search_iter(green_function, k_min=0, k_max=1, omega=0, args=None,
                   ktol=1e-5, etol=1e-6, max_iter=30, initial_buffer=0.05):
    iter_arg, _ = find_iterable_argument(args)
    k_epv = np.zeros(len(iter_arg))
    for k_epi in range(k_epv.size):
        args_i = get_modified_iterable_argument(args, k_epi)
        try:
            k_ep, r_ep = ep_search(green_function, k_min, k_max, omega, args_i, ktol, etol, max_iter)
        except ValueError:
            print(f"Warning: EP not in bounds at indx {k_epi}, retrying with larger bounds")
            k_ep, r_ep = ep_search(green_function, k_min/2, 2*k_max, omega, args_i, ktol, etol, max_iter)
        k_epv[k_epi] = k_ep
        if k_epi == 0:
            k_min = k_ep * (1 - initial_buffer)
            k_max = k_ep * (1 + initial_buffer)
        else:
            delta_k = k_epv[k_epi] - k_epv[k_epi-1]
            k_min, k_max = sorted([k_ep * (1 - np.sign(delta_k)*initial_buffer), k_ep + 2*delta_k])
        if k_epi % 50 == 0:
            print(f"Finished iteration {k_epi}")
    return k_epv

def energies_to_ep_size(energies):
    """(E_+, E_-) -> (ep_delta, ep_offset)"""
    ep_delta = np.abs((energies[1] - energies[0]).imag) / 2
    ep_offset = np.mean(energies)
    return ep_delta, ep_offset

def ep_size_iter(green_function, omega=0, args=None, ratio_warning=0.8):
    iter_arg, _ = find_iterable_argument(args)
    ep_deltav = np.zeros(len(iter_arg))                    # imaginary energy gap of EP at k=0
    ep_offsetv = np.zeros(len(iter_arg), dtype=complex)    # average value of the two branches
    ratiov = np.zeros(len(iter_arg))
    for epi in range(ep_deltav.size):
        args_i = get_modified_iterable_argument(args, epi)
        energies, ratio = energy_function(green_function, 0, omega, args_i, ratio_warning)
        ep_deltav[epi], ep_offsetv[epi] = energies_to_ep_size(energies)
        ratiov[epi] = ratio
        if epi % 50 == 10:
            print(f"Finished iteration {epi}")
    return ep_deltav, ep_offsetv, ratiov

def ep_size_num(u_a_num, u_b_num=None, N=None):
    """(u_a, u_b) -> (ep_delta, ep_offset)"""
    u_a_num = np.asarray(u_a_num)
    if u_b_num is None:
        u_b_num = np.zeros_like(u_a_num)
    u_b_num = np.asarray(u_b_num)
    ep_deltav = []
    ep_offsetv = []
    for (u_a_val, u_b_val) in zip(u_a_num, u_b_num):
        k_vals, omega_vals, green, h_eff = green_numeric(u_a_val, u_b_val, N)
        energies = green_toolkit.eigenvalues_2x2(h_eff[k_vals.size//2, omega_vals.size//2])
        ep_delta, ep_offset = energies_to_ep_size(energies)
        ep_deltav.append(ep_delta)
        ep_offsetv.append(ep_offset)
    return ep_deltav, ep_offsetv


def get_mmln(num_params):
    return f"{num_params['mmaxp']}.{num_params['mmax']}.{num_params['lmax']}.{num_params['numkp']}"
def get_ep_delta_string():
    return r"$\Delta E_{\mathrm{EP}}$"
def get_ep_offset_string():
    return r"$-\langle E_{\mathrm{EP}}\rangle$"
def get_label(arg):
    if arg is None:
        raise ValueError
    elif arg == "beta":
        return r"$\beta$"
    elif arg == "vf":
        return r"$v_{\mathrm{F}}$"
    elif arg == "u_a":
        return r"$U_{\mathrm{A}}$"
    else:
        return f"${arg}$"

def get_par_string(arg, mod=".3f"):
    if arg is None:
        raise ValueError
    elif isinstance(arg, (np.ndarray, list)):
        par = f"{arg[0]:{mod}}.{arg[-1]:{mod}}"
    elif isinstance(arg, dict):
        par = get_mmln(arg)
    else:
        par = f"{arg:{mod}}"
    return par

def get_pars_string(args, labels, mod=".3f"):
    pars = "_".join([label + get_par_string(arg, mod) for (label, arg) in zip(labels, args)])
    return pars


from gf_h1h3_v3_numba import green_perturbative, green_eff_ua, get_free_K0


# def plot_with_ptr():
#     fig, ax = plt.subplots()
#     ax2 = ax.twinx()
#     ax2.set_ylabel("$R$", color=ptr_label_color)
#     ax2.tick_params(axis='y', labelcolor=ptr_label_color)
#     return fig, [ax, ax2]

def get_plot_setup(ptr=False, embed=True, xscale="log", yscale="log", xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    if ptr:
        ax2 = ax.twinx()
        ax2.set_ylabel("$R$", color=ptr_label_color)
        ax2.tick_params(axis='y', labelcolor=ptr_label_color)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if embed:
        mpl_special.embed_labels(fig, fig.axes)
    return fig, fig.axes

def plot_ep_size_K(green_function, args, ratio_warning=0.8, **kwargs):
    """args = (beta, K, vf, g, a, w, num_params)"""
    K_0 = 1     # free LL
    _arg, indx = find_iterable_argument(args)
    arg = _arg - K_0
    fig, [ax, ax2] = get_plot_setup(True, xlabel=get_label("K-1"), ylabel=r"Im$\,E$")
    ep_delta, ep_offset, ratio = ep_size_iter(green_function, 0, args, ratio_warning)
    ep_offset = -ep_offset.imag
    ax2.set_yscale("log")
    ax2.set_ylim(ratio.min(), 1)
    if arg[0] < 0:
        turning_point_indx = 0
        while arg[turning_point_indx] < 0:
            turning_point_indx += 1
        ax.set_xscale("symlog", linthresh=arg[turning_point_indx])
        ax.set_yscale("log")
        # ax.set_yscale("symlog", linthresh=ep_delta[turning_point_indx])
        # ax2.set_yscale("symlog", linthresh=ratio[turning_point_indx])
        # ax2.set_ylim(0, 1)
        ratio = np.insert(ratio, turning_point_indx, np.nan)
        arg = np.insert(arg, turning_point_indx, 0)
        ep_delta = np.insert(ep_delta, turning_point_indx, np.nan)
        ep_offset = np.insert(ep_offset, turning_point_indx, np.nan)
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.plot(arg, ep_delta, label=get_ep_delta_string(), **kwargs)
    ax.plot(arg, ep_offset, label=get_ep_offset_string(), **kwargs)
    ax.set_xlim(arg[0], arg[-1])
    ax2.plot(arg, ratio, c="k", ls='--', alpha=ptr_line_color)
    ax.legend(loc="upper center")
    if arg[0] < 0:
        ax.xaxis.set_ticks(ax.xaxis.get_ticklocs()[::2])
        # ax.yaxis.set_ticks(ax.yaxis.get_ticklocs()[::3])
    return fig

def plot_ep_size_arg(green_function, args, ratio_warning=0.8, xlabel=None, xvals=None, **kwargs):
    """args = (beta, K, vf, g, a, w, num_params)"""
    if xvals is None:
        arg, _ = find_iterable_argument(args)
    else:
        arg = xvals
    fig, axes = get_plot_setup(ptr=(ratio_warning is not None))
    ax = axes[0]
    ax.set_xlabel(get_label(xlabel))
    ax.set_ylabel(r"Im$\,E$")
    ax.set_xlim(arg[0], arg[-1])
    ax.set_xscale("log")
    ep_delta, ep_offset, ratio = ep_size_iter(green_function, 0, args, ratio_warning)
    ax.plot(arg, ep_delta, label=get_ep_delta_string(), **kwargs)
    ax.plot(arg, -ep_offset.imag, label=get_ep_offset_string(), **kwargs)
    ax.set_yscale("log")
    if ratio_warning is not None:
        axes[1].set_ylim(ratio.min(), 1)
        axes[1].set_yscale("log")
        axes[1].plot(arg, ratio, c="k", ls='--', alpha=ptr_line_color)
    ax.legend()
    mpl_special.embed_labels(fig, fig.axes)
    return fig

def plot_ep_size(key, save=False):
    beta = 1; K = 1.5; vf = 1; g = 0.2; a = 1.0; w = np.pi; alpha = 1
    labels = ["beta", "K", "vf", "g", "a", "w", "mmln"]
    filename_prefix = "ep_size"
    if key == "beta":
        beta = np.geomspace(8e-2, 2e3, 200)
        w = (np.pi*alpha/beta/vf)
        num_params={"order" : 1, "mmax" : 30, "mmaxp" : 5, "lmax" : 30, "numkp" : 48}
        args = (beta, K, vf, g, a, w, num_params)
        fig = plot_ep_size_arg(green_perturbative, args, xlabel="beta")
        fig.axes[0].legend(loc="lower right")
        filename_char = "h1h3"
    elif key == "K":
        K_lower = 1 + np.geomspace(-0.7, -2e-6, 100)
        K_upper = 1 + np.geomspace(2e-6, 0.99, 100)
        K = np.concatenate((K_lower, K_upper))
        num_params = {"order" : 1, "mmax" : 3, "mmaxp" : 5, "lmax" : 3, "numkp" : 48}
        args = (beta, K, vf, g, a, w, num_params)
        fig = plot_ep_size_K(green_perturbative, args)
        filename_char = "h1h3"
    elif key == "g":
        g = np.geomspace(1e-8, 1e-0, 150)
        num_params={"order" : 1, "mmax" : 1, "mmaxp" : 1, "lmax" : 1, "numkp" : 48}
        args = (beta, K, vf, g, a, w, num_params)
        fig = plot_ep_size_arg(green_perturbative, args, xlabel="g")
        #~g^1 #fig.axes[0].plot(g, g / g[0] * fig.axes[0].lines[0].get_ydata()[0])
        filename_char = "h1h3"
    elif key == "v":
        vf = np.linspace(0.2, 3.0, 150)
        w = (np.pi*alpha/beta/vf)
        num_params={"order" : 1, "mmax" : 1, "mmaxp" : 1, "lmax" : 1, "numkp" : 48}
        args = (beta, K, vf, g, a, w, num_params)
        fig = plot_ep_size_arg(green_perturbative, args, xlabel="v")
        fig.axes[0].set_xscale("linear")
        filename_char = "h1h3"
    elif key == "u_a":
        u_a = np.linspace(9e-2, 1.6, 50)
        beta = 0.6; vf = 0.5; a = 1; alpha = 1    # numerical beta is only 0.2 though...
        K = get_free_K0(u_a, vf, a)
        #beta = 0.2; K = get_free_K0(u_a, vf, a, alpha**2/(a**2+alpha**2))
        v = vf / K
        w = np.pi*alpha/beta/v
        args = (beta, K, v, w)
        labels = ["beta", "K", "v", "w"]
        fig = plot_ep_size_arg(green_eff_ua, args, xlabel="u_a", xvals=u_a, ratio_warning=None)
        fig.axes[0].set_xscale("linear")
        u_a_num = np.linspace(0.0, 1.5, 16)
        ep_delta_num, _ = ep_size_num(u_a_num, N=80)
        for (u_a_val, ep_delta_val) in zip(u_a_num, ep_delta_num):
            fig.axes[0].plot(u_a_val, ep_delta_val, marker='o', c='k')
        filename_char = "eff_ua"
    else:
        raise NotImplementedError(f"Key {key} not supported")
    filename = filename_prefix + "_" + filename_char + "_" + get_pars_string(args, labels)
    if save:
        savefig(fig, PATH + filename)
    return fig


def get_complex_e(hamilton):
    energies = np.array([green_toolkit.eigenvalues_2x2(h_val) for h_val in hamilton])
    energies = green_toolkit.convert_to_contiguous_arrays(energies.T[0], energies.T[1])
    return energies


def plot_complex_e(key="demo", save=False):
    labels = ["beta", "K", "vf", "g", "a", "w", "mmln"]
    if key == "demo":
        K=1.5; g=0.2; beta=1; kwm=0.3; w=np.pi; vf=1.0; a=1.0   #manual pars
        num_params = {"order" : 1, "mmax" : 1, "mmaxp" : 1, "lmax" : 1, "numkp" : 48}
        args = (beta, K, vf, g, a, w, num_params)
        k_vals = np.linspace(-kwm, kwm, 601)
        omega_vals = np.array([0.0])
        t_start = pc()
        green = green_perturbative(k_vals, omega_vals, beta, K, vf, g, w=w, num_params=num_params)
        t_end = pc()
        print(f"Computation done in {t_end-t_start:.3f} seconds")
        hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    else:
        raise NotImplementedError(f"Key {key} not supported")

    ratio = np.abs(green[:,0,0,1]/green[:,0,0,0])
    fig, [ax, ax2] = get_plot_setup(ptr=True, xscale="linear", yscale="linear")
    energies = get_complex_e(hamilton[:, 0])
    plot_complex(ax, k_vals, energies)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E_\pm$")
    ax.set_xlim(k_vals[0], k_vals[-1])
    ax2.set_ylim(0, 1)
    ax2.plot(k_vals, ratio, c="k", ls='--', alpha=ptr_line_color)
    mpl_special.embed_labels(fig, [ax, ax2])
    filename = "complex_E_" + get_pars_string(args, labels)
    if save:
        savefig(fig, PATH + filename)
    return fig


def plot_complex_e_ua():
    beta=1; alpha=1; a=1; kwm=0.5; vf=0.5; u_a = 1.0
    k_vals = np.linspace(-kwm, kwm, 301)
    omega_vals = np.array([0.0])
    green = green_eff_ua(k_vals, omega_vals, u_a, beta, vf, alpha, a)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    fig, ax = get_plot_setup(xscale="linear", yscale="linear")
    energies = get_complex_e(hamilton[:, 0])
    plot_complex(ax[0], k_vals, energies)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E_\pm$")
    ax.set_xlim(k_vals[0], k_vals[-1])


def main():
    """
    beta=1.1; K=0.6; u=1.0; u_minus=0.05
    k_vals, omega_vals, green = get_k_w_green(green_perturbative, args=(beta, K, u, u_minus, 1))
    hamilton = hamilton_from_green(omega_vals, green)
    plot_ep(k_vals, omega_vals, green)

    #complex E
    beta=1.1; K=0.9; u=1.0; u_minus=0.01; kwm = 0.04
    k_vals = np.linspace(-kwm, kwm, 301)
    omega_vals = np.array([0.4535])
    green = green_perturbative(k_vals, omega_vals, beta, K, u, u_minus, order=1)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    energies = np.array([green_toolkit.eigenvalues_2x2(hamilton[i,0])
                         for i in range(k_vals.size)])
    energies = green_toolkit.convert_to_contiguous_arrays(energies.T[0], energies.T[1])
    plot_complex(plt.subplots()[1], k_vals, energies)

    #plot ep
    beta=1.1; K=0.9; u=1.0; u_minus=0.01; kwm = 1.04
    k_vals = np.linspace(-kwm, kwm, 101)
    omega_vals = np.linspace(-kwm, kwm, 71)
    green = green_perturbative(k_vals, omega_vals, beta, K, u, u_minus, order=1)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega$")
    ep_contour_plot(ax, hamilton, k_vals, omega_vals)
    mpl_special.embed_labels(fig, ax)

    #both plots, refined (WARNING: PRE-L_0(beta,u)-fix!)
    #paper similarity:
    beta=0.2; vf=0.005; ua=2.0; ub=0; u_minus=(ua-ub)/2; u_plus=(ua+ub)/2;
    alpha=1e-3; kwm = 0.05; wm=0.001
    beta=0.05; vf=0.05; ua=1.9; ub=-1.1; u_minus=(ua-ub)/2; u_plus=(ua+ub)/2
    alpha=5e-3; kwm = 0.005; wm=0.001
    #other params:
    beta=0.1; vf=0.2; ua=2.0; ub=-1; u_minus=(ua-ub)/2; u_plus=(ua+ub)/2
    alpha=5e-2; kwm = 0.005; wm=0.001

    #quantiative matching, d == vf
    beta=0.5; vf=0.5; ua=0.2; ub=0; u_minus=(ua-ub)/2; u_plus=(ua+ub)/2
    alpha=1; kwm = 0.01; wm=0.01   # (g==c digits for some reason :) )

    K = 1/np.sqrt(1 + 4*u_plus*alpha/(2*np.pi*vf)); u=vf/K; g=-u_minus*alpha/(2*np.pi*u)
    k_vals = np.linspace(-kwm, kwm, 101)
    omega_vals = np.array([0.0])
    green = green_perturbative(k_vals, omega_vals, beta, K, u, g, order=1, model="density var",
                               alpha=alpha)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    energies = np.array([green_toolkit.eigenvalues_2x2(hamilton[i,0])
                         for i in range(k_vals.size)])
    energies = green_toolkit.convert_to_contiguous_arrays(energies.T[0], energies.T[1])
    fig, ax = plt.subplots()
    plot_complex(ax, k_vals, energies)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E_\pm$")
    ax.set_xlim(k_vals[0], k_vals[-1])
    mpl_special.embed_labels(fig, ax)
    print(energies[0].imag.max() - energies[1].imag.min(), energies[0][0], g, K)
    k_vals = np.linspace(-kwm, kwm, 31)
    omega_vals = np.linspace(-wm, wm, 20)
    green = green_perturbative(k_vals, omega_vals, beta, K, u, g, order=1, model="density var",
                               alpha=alpha)
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega$")
    ep_contour_plot(ax, hamilton, k_vals, omega_vals)
    eqn1, eqn2 = green_toolkit.hamilton_2x2_ep_eqn(hamilton)
    phase = np.sqrt(eqn1 + 2j*eqn2)
    arg = np.arctan2(phase.imag, phase.real)
    img = ax.imshow(arg.T, cmap="RdBu", aspect="auto", extent=[-kwm, kwm, -wm, wm],
                    origin="lower", vmin=-np.pi/2, vmax=np.pi/2, alpha=0.85)
    cbar = fig.colorbar(img, ax=ax, label=r"$\arg(\Delta E)$")
    mpl_special.format_ticklabels(cbar.ax, which="y")
    mpl_special.embed_labels(fig, [ax, cbar.ax])

    #complex E (10.01.2024)
    beta=20; vf=0.5; ua=1.5; ub=0; u_minus=(ua-ub)/2; u_plus=(ua+ub)/2
    w=1e3; kwm = 0.001; omega=0.0
    from rg_h1h3_model import get_rg_flow, rg_beta_h1h3, plot_rg_flow
    from time import perf_counter as pc
    couplings0 = np.array([u_minus/(4*np.pi**2*vf), -u_plus/(4*np.pi**2*vf), 1])
    l_max = np.log(beta*vf/np.pi * w)
    l_vals = np.linspace(0, l_max, 200)
    couplings, _ = get_rg_flow(rg_beta_h1h3, couplings0, l_vals)
    labels = ["$g_1$", "$g_3$", "$K$"]
    #plot_rg_flow(np.abs(couplings), l_vals, labels)
    g, g3, K = couplings[-1]
    #beta=10; kwm=0.001; w=1e3; ua=1.5; ub=0; vf=0.5      #RG pars
    K=2.5; g=0.0015; beta=20; kwm=1.2; w=np.pi; vf=0.5   #manual pars
    num_params = {"order" : 1, "mmax" : 8, "mmaxp" : 8, "lmax" : 3, "numkp" : 31}
    K=1.9; g=0.4; beta=1; kwm=1.2; w=np.pi; vf=0.8   #manual pars
    num_params = {"order" : 1, "mmax" : 1, "mmaxp" : 1, "lmax" : 1, "numkp" : 50}
    k_vals = np.linspace(-kwm, kwm, 101)
    omega_vals = np.array([omega])
    t_start = pc()
    green = green_perturbative(k_vals, omega_vals, beta, K, vf, g, w=w, num_params=num_params)
    t_end = pc()
    print(f"Computation done in {t_end-t_start:.3f} seconds")
    hamilton = green_toolkit.hamilton_from_green(omega_vals, green)
    energies = np.array([green_toolkit.eigenvalues_2x2(hamilton[i,0])
                         for i in range(k_vals.size)])
    energies = green_toolkit.convert_to_contiguous_arrays(energies.T[0], energies.T[1])
    ratio=np.abs(green[:,0,0,1]/green[:,0,0,0])
    fig, ax = plt.subplots()
    plot_complex(ax, k_vals, energies)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E_\pm$")
    ax.set_xlim(k_vals[0], k_vals[-1])
    color = (0,0,0,0.6)
    ax2=ax.twinx()
    ax2.set_ylabel("PTR", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    ax2.plot(k_vals, ratio, c="k", ls='--', alpha=0.3)
    mpl_special.embed_labels(fig, [ax,ax2])
    filename = f"complex_E_h1h3_Ua{ua:.2f}_Ub{ub:.2f}_beta{beta:.2f}_vf{vf:.1f}_w{w:.3f}_omega{omega:.2f}_mmlkn{num_params['mmaxp']}.{num_params['mmax']}.{num_params['lmax']}.{num_params['numkp']}_g{g:.3f}_K{K:.2f}"
    #fig.savefig("../MA_Latex/figures/" + filename + ".pdf")
    print(f"RG: l_max={l_max:.2f}, g1/g10={g/couplings0[0]:.2f}, g3/g30={g3/couplings0[1]:.2f}")
    print(f"PT ratio: mean={np.mean(ratio):.3f}, max={np.max(ratio):.3f}, min={np.min(ratio):.3f}")
    print(f"Delta Im E={energies[0].imag.max() - energies[1].imag.min():.4f}, g={g:.2e}, g3={g3:.2e}, K={K:.3f}")
    print("num_params : ", num_params)
    print(fr"$U_\A={ua:.1f}$, $U_\B={ub:.1f}$, $\beta={beta:.1f}$, $\vf={vf:.1f}$, $w=\pi$, $\omega=0$, $m_\text{{max}}'={num_params['mmaxp']}$, $m_\text{{max}}={num_params['mmax']}$, $l_\text{{max}}={num_params['lmax']}$, $n_{{k'}}={num_params['numkp']}$, $g={g:.3f}$, $K={K:.2f}$, $\text{{PTR}}={np.min(ratio):.2f}\dots {np.max(ratio):.2f}$")
    """
    # plot_ep(*get_k_w_green(green_numeric, args=(1.5, -1.1)))
    # plot_ep(*get_k_w_green(green_perturbative, args=(1.1, 0.6, 1.0, 0.05, 1)))
    
    # plot_complex_e(key="demo")
    # plot_ep_size(key="g")
    # plot_ep_size(key="K")
    # plot_ep_size(key="beta")
    return 0


if __name__ == "__main__":
    main()
