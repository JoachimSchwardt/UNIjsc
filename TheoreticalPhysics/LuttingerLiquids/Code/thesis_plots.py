#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:57:34 2024

@author: ag_budich1
"""

import numpy as np
import matplotlib.pyplot as plt
import thesis_toolkit as tlk
import thesis_gf as gf
import thesis_algorithm as algo
import mpl_special
FIGSIZE_PT = 438.17247
plt.rcParams["figure.figsize"] = mpl_special.set_figsize(FIGSIZE_PT)  # 362.77269
plt.rcParams["figure.dpi"] = 340
plt.rcParams["font.size"] = 11


PATH = "../MA_Latex/thesis/figures/"
DATAPATH = "data_joachim/"

LINEWIDTH_THIN = 0.5


def plot_test_figure():
    fig, ax = plt.subplots()
    xv = np.linspace(0,1,100)
    tlk.plot_complex(ax, xv, xv**2 - 2j*xv + 0.5j - 0.3)
    tlk.autoformat(ax, xlabel="$x$", ylabel="$y$")
    ax.legend()
    mpl_special.embed_labels(fig, ax)
    tlk.savefig(fig, PATH + "test")

def plot_ll_ep(ax, beta=15, K=0.4, v=1.25, g=0.2, k_max=0.505, k_count=593, alpha=2, w=np.pi,
               num_params={"mmax" : 9, "mmaxp" : 9, "lmax" : 9, "numkp" : 48}, store=False):
    w = tlk.get_w(w, alpha, beta, v)
    labels = ["kmax", "kcount", "beta", "K", "v", "g", "a", "alpha", "mmln"]
    pars_string = tlk.get_pars_string((k_max, k_count, beta, K, v, g, 1, alpha, num_params), labels)
    dataname = "data_joachim/green_" + pars_string + ".npy"
    filename = "ll_ep_" + pars_string
    k_vals = np.linspace(-k_max, k_max, k_count)
    try:
        green = np.load(dataname)
        print("Loaded GF from disk")
    except FileNotFoundError:
        green = tlk.timer(gf.green_perturbative, k_vals, [0], beta, K, v, w=w, g=g,
                          num_params=num_params)
        if store:
            np.save(dataname, green)
            print("Stored GF to disk:", dataname)
    energies = algo.get_energies(green, [0])
    energies = tlk.contiguous_arrays(energies)

    tlk.plot_complex(ax, k_vals, energies.T)
    tlk.autoformat(ax, xlabel="$k$", ylabel="$E$")
    ratio = gf.get_ptr(green)
    # ax2 = tlk.plot_ptr_axis(ax)
    # ax2.plot(k_vals, ratio, c="k", ls='--', alpha=tlk.ptr_line_color)
    print(f"MAX PTR: {np.max(ratio) : .2f}")
    ax.legend()
    print(f"Created figure {filename}")
    return filename

def plot_ll_example_ep(beta=15, K=0.4, v=1.25, g=0.2, k_max=0.505, k_count=593, alpha=2, w=np.pi,
                       num_params={"mmax" : 9, "mmaxp" : 9, "lmax" : 9, "numkp" : 48}, save=False):
    w = tlk.get_w(w, alpha, beta, v)
    k_vals = np.linspace(-k_max, k_max, k_count)
    green = gf.green_perturbative(k_vals, [0], beta, K, v, w=w, g=g, num_params=num_params)
    energies = algo.get_energies(green, [0])
    energies = tlk.contiguous_arrays(energies)

    fig, ax = plt.subplots()
    filename = plot_ll_ep(ax, beta, K, v, g, k_max, k_count, alpha, w, num_params)
    tlk.mpl_special.embed_labels(fig, fig.axes)
    if save:
        tlk.savefig(fig, PATH + filename)


def plot_ll_par_fit(axis, u_a=1.5, u_b=1.1, beta=1, alpha=2, w=np.pi, parameters=None):
    """Plot the complex energy eigenvalues for a fit of LL parameters to numerical data"""
    k_max = 0.705
    k_num, omega_num, green_num, _ = gf.green_numeric(u_a, u_b, beta=beta)
    omega_indx = omega_num.size // 2
    omega = omega_num[omega_indx]
    indx = tlk.arg_restrict(k_num, -k_max, k_max)
    k_num_vals = k_num[indx]
    green_fit = green_num[indx, omega_indx]
    energies_fit = algo.get_energies(green_fit, omega)

    num_params={"mmax" : 3, "mmaxp" : 3, "lmax" : 3, "numkp" : 48}
    if parameters is None:
        K, v, g, beta, cov = tlk.timer(algo.ll_par_fit, energies_fit, k_num_vals, beta=beta,
                                       alpha=alpha, num_params=num_params, mute_ptr_warnings=True)
    else:
        K, v, g, beta = parameters
        cov = None
        print("Parameters", parameters, "given, fit calculation skipped")
    w = tlk.get_w(w, alpha, beta, v)
    k_vals = np.linspace(-k_max, k_max, 301)
    green_num_vals = green_num[indx, omega_indx]
    energies_num = algo.get_energies(green_num_vals, omega)
    green = gf.green_perturbative(k_vals, omega, beta, K, v, g=g, w=w, num_params=num_params)
    energies = algo.get_energies(green, omega)
    energies = tlk.contiguous_arrays(energies)

    tlk.plot_complex(axis, k_num_vals, energies_num.T, ls='', marker='o', set_label=False)
    tlk.plot_complex(axis, k_vals, energies.T)
    tlk.autoformat(axis, xlabel="$k$", ylabel="$E$")
    axis.legend()
    return K, v, g, beta, cov


def plot_gf_ll_par_fit(save=False, manip=True, AB=False):
    u_a=1.5; u_b=-1.1; beta=1; alpha=2; vf=0.5; a=1; w=np.pi
    num_params = {"mmax" : 3, "mmaxp" : 3, "lmax" : 3, "numkp" : 48}
    k_num, omega_num, green_num, _ = gf.green_numeric(u_a, u_b, beta=beta)
    omega_indx = omega_num.size // 2
    omega = omega_num[omega_indx]
    k_max = 2.05
    indx = tlk.arg_restrict(k_num, -k_max, k_max)
    k_num = k_num[indx]
    green_num = green_num[indx, omega_indx]
    green_num_lr = tlk.basis_ab_to_lr(green_num)
    grl, glr = green_num_lr[:, 0, 1], green_num_lr[:, 1, 0]
    gminus = (grl - glr) / 2
    goff = [gminus, -gminus]
    green_num_lr_manip = np.zeros_like(green_num_lr)
    for i in range(2):
        green_num_lr_manip[:, i, i] = green_num_lr[:, i, i]
        green_num_lr_manip[:, i, (i+1) % 2] = goff[i]
    green_num_ab_manip = tlk.basis_lr_to_ab(green_num_lr_manip)
    if manip:
        green_num_fit = green_num_lr_manip
    else:
        green_num_fit = green_num_lr
    K, v, g, cov = algo.ll_par_fit_gf(green_num_fit, k_num, beta=beta, alpha=alpha, vf=vf, w=w, a=a,
                                      num_params=num_params, mute_ptr_warnings=True)
    w = tlk.get_w(w, alpha, beta, v)
    green = gf.green_perturbative(k_num, omega, beta, K, v, g, a, w, num_params)
    fig, ax = plt.subplots(2, 2)
    if AB:
        RL = [r"\mathrm{A}", r"\mathrm{B}"]
    else:
        RL = [r"\mathrm{R}", r"\mathrm{L}"]
    for i in range(2):
        for j in range(2):
            tlk.plot_complex(ax[i,j], k_num, green[:,0,i,j])
            if manip and AB:
                g_num = green_num_ab_manip
            elif manip and not AB:
                g_num = green_num_lr_manip
            elif not manip and AB:
                g_num = green_num
            else:
                g_num = green_num_lr
            tlk.plot_complex(ax[i,j], k_num, g_num[:,i,j], set_label=False, ls='', marker='o')
            ax[i,j].legend()
            tlk.autoformat(ax[i,j], xlabel="$k$", ylabel=rf"$G_{{{RL[i]}{RL[j]}}}$")
    mpl_special.embed_labels(fig, ax)
    labels = ["UA", "UB", "beta", "K", "vf", "g", "a", "alpha", "mmln"]
    args = (u_a, u_b, beta, K, vf, g, a, alpha, num_params)
    figure_type = "gf_ll_par_fit"
    if manip:
        figure_type += "_manip"
    filename = figure_type + "_" + tlk.get_pars_string(args, labels)
    print(f"Created figure {filename}")
    if save:
        tlk.savefig(fig, PATH + filename)


def plot_ll_par_fit_all(save=False, store=False, save_all=False):
    u_a = 1.5; beta = 1
    u_bv = np.arange(-1.5, 1.6, 0.1)
    u_bv = u_bv[np.abs(u_bv) > 0.65]
    parameters = np.zeros((len(u_bv), 4))
    covs = np.zeros((len(u_bv), 3, 3))
    labels = ["UA", "UB", "beta"]
    args = (u_a, u_bv, beta)
    dataname_par = DATAPATH + "ep_ll_par_fit_par_" + tlk.get_pars_string(args, labels) + ".npy"
    dataname_cov = DATAPATH + "ep_ll_par_fit_cov_" + tlk.get_pars_string(args, labels) + ".npy"
    try:
        parameters = np.load(dataname_par)
        covs = np.load(dataname_cov)
    except FileNotFoundError:
        print("Datafile", dataname_par, "does not exist, computing values...")
        for i, u_b in enumerate(u_bv):
            fig, axis = plt.subplots()
            K, v, g, _, cov = plot_ll_par_fit(axis, u_a, u_b, beta=beta)
            mpl_special.embed_labels(fig, axis)
            parameters[i] = [K, v, g, beta]
            covs[i] = cov
            labels = ["UA", "UB", "K", "v", "beta", "g"]
            args = (u_a, u_b, K, v, beta, g)
            filename = "ep_ll_par_fit_" + tlk.get_pars_string(args, labels)
            if save_all:
                tlk.savefig(plt.gcf(), PATH + filename)
        if store:
            np.save(dataname_par, parameters)
            np.save(dataname_cov, covs)
            print("Created datafile", dataname_par, "and", dataname_cov)
    fig, ax = plt.subplots()
    ms = 2.5
    ax.plot(u_bv, parameters[:, 0], label="$K$", ls='', marker='o', ms=ms)
    ax.plot(u_bv, parameters[:, 1], label="$v$", ls='', marker='o', ms=ms)
    color_g = mpl_special.Colors().colors[2]
    ax.plot([], [], label="$g$", c=color_g, ls='', marker='o', ms=ms)
    ax2 = ax.twinx()
    ax2.plot(u_bv, parameters[:, 2], c=color_g, ls='', marker='o', ms=ms)
    ax2.set_yscale("log")
    ax2.set_ylim(1.2e-1, 2.5e2)
    ax2.set_ylabel("$g$")
    tlk.autoformat(ax, xlabel=r"$U_{\mathrm{B}}$", ylabel=r"$K, v$")
    ax.legend()
    mpl_special.embed_labels(fig, [ax, ax2])
    if save or save_all:
        fig.savefig(PATH + "ll_par_fit_UB-1.5.1.5_alpha2.pdf")

def plot_ll_par_fit_example(save=False):
    u_a = 1.5; beta = 1
    u_bv = np.arange(-1.5, 1.6, 0.1)
    u_bv = u_bv[np.abs(u_bv) > 0.65]
    parameters = np.zeros((len(u_bv), 4))
    labels = ["UA", "UB", "beta"]
    args = (u_a, u_bv, beta)
    dataname_par = DATAPATH + "ep_ll_par_fit_par_" + tlk.get_pars_string(args, labels) + ".npy"
    dataname_cov = DATAPATH + "ep_ll_par_fit_cov_" + tlk.get_pars_string(args, labels) + ".npy"
    try:
        parameters = np.load(dataname_par)
        covs = np.load(dataname_cov)
    except FileNotFoundError:
        print("Datafile", dataname_par, "does not exist!")
        return
    figure_dpi_orig = plt.rcParams["figure.dpi"]
    plt.rcParams["figure.dpi"] = 330
    figsize = mpl_special.set_figsize(FIGSIZE_PT)
    fig = plt.figure(figsize=(figsize[0], figsize[0] / 1.5))
    ncols = 3; colspan = 2
    ax1 = plt.subplot2grid((2, ncols), (0, 0), colspan=colspan, rowspan=2)
    ax2 = plt.subplot2grid((2, ncols), (0, colspan), colspan=1)
    ax3 = plt.subplot2grid((2, ncols), (1, colspan), colspan=1)
    for i, axis in enumerate([ax1, ax2, ax3]):
        xoffset = 0.03
        yoffset = 0.025
        if i == 0:
            xoffset /= 2
            yoffset /= 2
        axis.text(xoffset, 1 - yoffset, tlk.panels[i], transform=axis.transAxes, ha="left", va="top")

    ms = 2
    errors = np.sqrt([[covs[i, j, j] for j in range(3)] for i in range(len(covs))])
    errors[np.isnan(errors)] = 0
    ax1.errorbar(u_bv, parameters[:, 0], yerr=errors[:, 0], ls='', marker='o', ms=ms)
    ax1.errorbar(u_bv, parameters[:, 1], yerr=errors[:, 1], ls='', marker='o', ms=ms)
    ax12 = ax1.twinx()
    for color, label in zip(mpl_special.Colors().colors[:3], ["$K$", "$v$", "$g$"]):
        ax1.plot([], [], label=label, c=color, ls='', marker='o', ms=ms)
    ax12.errorbar(u_bv, parameters[:, 2], yerr=errors[:, 2], c=color, ls='', marker='o', ms=ms)
    ax12.set_yscale("log")
    ax12.set_ylim(1.4e-1, 4.7e2)
    ax12.set_ylabel("$g$")
    plot_ll_par_fit(ax2, u_b=u_bv[14], beta=beta, parameters=parameters[14])
    plot_ll_par_fit(ax3, u_b=u_bv[6], beta=beta, parameters=parameters[6])
    ax2.set_yticks([]); ax3.set_yticks([])
    ax2.set_xticks([]); ax2.set_xlabel(None)
    ax2.set_ylabel(None); ax3.set_ylabel(None)
    ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * 1.15)
    ax3.set_ylim(ax3.get_ylim()[0], ax3.get_ylim()[1] * 1.07)
    tlk.autoformat(ax1, xlabel=r"$U_{\mathrm{B}}$", ylabel=r"$K, v$")
    ax1.legend()
    mpl_special.embed_labels(fig, [ax1, ax12, ax2, ax3])
    filename = "ep_ll_par_fit_cov_triple_UBI14.6"   #u_bv[[14,6]] == array([1.2, -0.9])
    plt.rcParams["figure.dpi"] = figure_dpi_orig
    if save:
        tlk.savefig(fig, PATH + filename)


def plot_gf_ua_eq_ub_par_fit(axis, u_a=1.0, beta=1, w=np.pi, alpha=2, vf=0.5, a=1):
    k_num, omega_num, green_num, _ = gf.green_numeric(u_a, u_a, beta=beta)
    k_max = 2.05
    indx = tlk.arg_restrict(k_num, -k_max, k_max)
    k_num = k_num[indx]
    green_num = green_num[indx]
    omega_indx = omega_num.size // 2
    K, v, w, _ = algo.ll_par_ua_eq_ub_gf(u_a, beta, w=w, alpha=alpha)
    K_ana = gf.get_free_K0(u_a, vf, a, factor=alpha**2/(a**2+alpha**2))
    v_ana = vf / K_ana
    w = tlk.get_w(w, alpha, beta, v)
    w_ana = tlk.get_w(w, alpha, beta, v_ana)
    k_vals = np.linspace(k_num[0], k_num[-1], 501)
    green = gf.green_perturbative(k_vals, [0], beta, K=K, v=v, w=w, order=0)
    green_ana = gf.green_perturbative(k_vals, [0], beta, K=K_ana, v=v_ana, w=w_ana, order=0)
    green_num_lr = tlk.basis_ab_to_lr(green_num)[:, omega_indx]
    tlk.plot_complex(axis, k_num, green_num_lr[:, 0, 0], ls='', marker='o', set_label=False)
    tlk.plot_complex(axis, k_vals, green[:, 0, 0, 0], ls='--', alpha=0.9, set_label=False)
    tlk.plot_complex(axis, k_vals, green_ana[:, 0, 0, 0], label_specifier=r"G_{\mathrm{RR}}")
    tlk.autoformat(axis, xlabel="$k$")#, ylabel=r"$G_{\mathrm{RR}}$")
    axis.legend()
    args = (u_a, beta, alpha, w, vf, a)
    labels = ["UA", "beta", "alpha", "w", "vf", "a"]
    filename = "gf_ua_eq_ub_par_fit_" + tlk.get_pars_string(args, labels)
    print(f"Created figure {filename}")

def plot_compare_na_ll_par_ua_eq_ub(axis, beta=1, w=np.pi, alpha=2, vf=0.5, a=1):
    u_a_num = np.linspace(0.0, 1.5, 16)
    u_a = np.linspace(u_a_num[0], u_a_num[-1] + 0.02, 150)
    pars = [[1, vf, tlk.get_w(w, alpha, beta, vf)]]
    covs = [np.zeros((2,2))]
    for u_a_val in u_a_num[1:]:
        K, v, w, cov = algo.ll_par_ua_eq_ub_gf(u_a_val, beta, w=w, alpha=alpha)
        pars.append([K, v, w])
        covs.append(cov)
    K, v, w = np.array(pars).T
    errors = np.sqrt(np.diag(cov))
    K0 = gf.get_free_K0(u_a, vf, a, factor=alpha**2/(a**2 + alpha**2))
    for label, num_data, analytic, error in zip(["$K$", "$v$"], [K, v], [K0, vf / K0], errors):
        line = axis.errorbar(u_a_num, num_data, yerr=error, ls='', marker='o')
        axis.plot(u_a, analytic, c=line[0].get_color(), label=label)
    tlk.autoformat(axis, xlabel=r"$U_\mathrm{A}$")
    axis.legend()
    args = (u_a, beta, alpha, w, vf, a)
    labels = ["UA", "beta", "alpha", "w", "vf", "a"]
    filename = "gf_ua_eq_ub_LL_Kv_" + tlk.get_pars_string(args, labels)
    print(f"Created figure {filename}")

def plot_ua_eq_ub_dual_example(save=False):
    figsize = mpl_special.set_figsize(FIGSIZE_PT)
    fig, ax = plt.subplots(1, 2, figsize=(figsize[0], figsize[0]/2))
    plot_gf_ua_eq_ub_par_fit(ax[0])
    plot_compare_na_ll_par_ua_eq_ub(ax[1])
    ax[1].axhline(0.5, ls='--', c=mpl_special.Colors().colors[1], alpha=0.8)
    for i, axis in enumerate(ax):
        axis.text(0.03, 1-0.025, tlk.panels[i], transform=axis.transAxes, ha="left", va="top")
    mpl_special.embed_labels(fig, ax)
    filename = "gf_ua_eq_ub_cov_dual_example"
    if save:
        tlk.savefig(fig, PATH + filename)


def plot_numeric_paper_initial(save=False):
    """Dispersion relation for omega=0 for free and interacting Hamiltonian from 2021 paper"""
    h_effs = [np.load(DATAPATH + f"h_eff_{key}.npy") for key in ["initial", "final"]]
    k_vals = np.linspace(-np.pi, np.pi, h_effs[0].shape[0], endpoint=False)
    energies = [algo.get_energies_from_heff(h_eff) for h_eff in h_effs]
    figsize = mpl_special.set_figsize(FIGSIZE_PT)
    fig = plt.figure(figsize=(figsize[0], figsize[1]/1.618))
    colspan = 8
    ncols = 2*colspan + 1
    ax1 = plt.subplot2grid((1, ncols), (0, 0), colspan=colspan)
    ax2 = plt.subplot2grid((1, ncols), (0, colspan+1), colspan=colspan)
    ax3 = plt.subplot2grid((1, ncols), (0, colspan), colspan=1)
    ax3.axis("off")
    for i, axis in enumerate([ax1, ax2]):
        axis.yaxis.set_ticks([])
        axis.set_xlim(k_vals[0], k_vals[-1])
        axis.set_xlabel("$k$")
        #axis.set_ylabel("$E$", rotation=0, ha="right", va="center")
        #coords = axis.yaxis.get_label().get_position()
        #axis.yaxis.set_label_coords(-0.03, 0.9, transform=axis.transAxes)
        colors = mpl_special.Colors()
        col_real = colors.get_color()
        col_imag = colors.get_color()
        for ctr, energy in enumerate(energies[i].T):
            axis.plot(k_vals, energy.real, c=col_real, label=tlk.get_label("real E") if ctr else None)
            axis.plot(k_vals, energy.imag, c=col_imag, label=tlk.get_label("imag E") if ctr else None)
        axis.legend()
        mpl_special.set_ticks_linear(axis, k_vals[0], -k_vals[0], 5)
        mpl_special.format_ticklabels(axis)
    offset = 20
    v_f = ((energies[0][k_vals.size//2 + offset, 0] - energies[0][k_vals.size//2 - offset, 1])
           / (k_vals[k_vals.size//2 + offset] - k_vals[k_vals.size//2 - offset]))
    for sign in [+1, -1]:
        ax1.plot(k_vals, sign * v_f.real * k_vals, ls='--', c=ax1.lines[0].get_color(), alpha=0.6)
    ax3.arrow(0.15, 0.5, 1.2, 0, transform=ax3.transAxes, clip_on=False, length_includes_head=True,
              head_length=0.1, head_width=0.03, fc="k", ec="k", capstyle="round", joinstyle="round")
    ax3.text(0.35, 0.55, r"$H_\mathrm{int}$", transform=ax3.transAxes)
    mpl_special.embed_labels(fig, [ax1, ax2], embed_ylabels=False)
    if save:
        tlk.savefig(fig, PATH + "sketch_complex_E")


def plot_ep_contour(axis, k_vals, omega_vals, beta=15, K=0.4, v=0.5, g=0.2, alpha=2, w=np.pi,
                    num_params={"mmax" : 5, "mmaxp" : 5, "lmax" : 5, "numkp" : 48}, save=False):
    """EP contour plot of dr^2==di^2 and dr.di==0; (541, 360)
    k_max=0.45; g=0.25; beta=15; K=0.4; v=1.0; k_count_high=301; alpha=2; w=np.pi; omega_max=0.2
    num_params={"mmax" : 5, "mmaxp" : 5, "lmax" : 5, "numkp" : 48}; k_count=53; omega_count=30
    plot_ep_contour(-k_max, k_max, k_count, omega_max, omega_count, beta, K, v, g, alpha, w, num_params)
    plot_ll_example_ep(beta, K, v, g, k_max, k_count_high, alpha=alpha, w=w, num_params=num_params)
    """
    w = tlk.get_w(w, alpha, beta, v)
    args = ((k_vals.size, omega_vals.size), beta, K, v, g, alpha, w, num_params)
    labels = ["contour", "beta", "K", "v", "g", "alpha", "w", "mmln"]
    pars_string = tlk.get_pars_string(args, labels)
    dataname = DATAPATH + "green_" + pars_string + ".npy"
    filename = "ep_" + pars_string
    try:
        green = np.load(dataname)
        print("Loaded GF from disk")
    except FileNotFoundError:
        green = tlk.timer(gf.green_perturbative, k_vals, omega_vals, beta, K, v, g,
                          w=w, num_params=num_params, use_omega_symmetry=True)
        np.save(dataname, green)
        print("Stored GF to disk:", dataname)
    contour1, contour2 = tlk.plot_ep(axis, k_vals, omega_vals, green)
    tlk.plot_fermi_arcs(axis, contour1, contour2, c=(0.7, 0.7, 0.7))
    tlk.plot_contour_intersection(axis, contour1, contour2, c='k', ms=2.5)
    print(f"Created figure {filename}")

def plot_ep_contour_example(save=False):
    #k_max=0.0505; g=0.1; beta=50; K=0.4; v=1; k_count_high=301; alpha=2; w=np.pi; omega_max=0.05
    k_max=0.21; g=0.1; beta=20; K=0.4; v=0.8; k_count_high=1501; alpha=2; w=np.pi; omega_max=0.1
    num_params={"mmax" : 8, "mmaxp" : 5, "lmax" : 8, "numkp" : 48}; k_count=641; omega_count=800
    k_vals = np.linspace(-k_max, k_max, k_count)
    omega_scale = 1.1
    omega_vals = np.linspace(-omega_max*omega_scale, omega_max*omega_scale, omega_count)

    fig = plt.figure()
    colspan = 2
    ax1 = plt.subplot2grid((1, 2*colspan+1), (0, 0), colspan=colspan)
    ax2 = plt.subplot2grid((1, 2*colspan+1), (0, colspan), colspan=colspan+1)
    plot_ll_ep(ax1, beta, K, v, g, k_max/1.6, k_count_high, alpha=alpha, w=w, num_params=num_params)
    plot_ep_contour(ax2, k_vals, omega_vals, beta, K, v, g, alpha, w, num_params)
    ax1.legend(loc="upper center")
    ax1_max = 0.2
    ax1.set_ylim(-ax1_max, ax1_max)
    tlk.set_ticks_linear(ax1, -ax1_max, ax1_max, 3, which="y")
    tlk.set_ticks_linear(ax2, -omega_max, omega_max, 3, which="y")
    for i, axis in enumerate([ax1, ax2]):
        axis.text(0.03, 1-0.025, tlk.panels[i], transform=axis.transAxes, ha="left", va="top")
    mpl_special.embed_labels(fig, fig.axes)
    filename = "ep_contour_dual_example"
    if save:
        tlk.savefig(fig, PATH + filename)


def get_compute_ez_size_pars(key="g"):
    """Compute ep_delta, ep_offset and k_ep for predetermined parameters and store the result"""
    g=0.1; beta=20; K=0.4; v=0.8; a=1; alpha=2
    num_params = {"mmax" : 10, "mmaxp" : 10, "lmax" : 10, "numkp" : 48}
    if key == "g":
        g_lower = np.geomspace(1e-8, 5e-2, 11, endpoint=False)
        g_upper = np.geomspace(5e-2, 3e-1, 51)
        g = np.concatenate((g_lower, g_upper))
    elif key == "K":
        K_lower_low = 1 + np.geomspace(-0.6, -0.1, 31, endpoint=False)
        K_lower_high = 1 + np.geomspace(-0.1, -2e-6, 7)
        K_upper_low = 1 + np.geomspace(2e-6, 0.1, 7, endpoint=False)
        K_upper_high = 1 + np.geomspace(0.1, 0.8, 21)
        K = np.concatenate((K_lower_low, K_lower_high, K_upper_low, K_upper_high))
    elif key == "v":
        v_lower = np.geomspace(4e-2, 1, 51, endpoint=False)
        v_upper = np.geomspace(1, 5, 5, endpoint=False)
        v_high = np.geomspace(5, 10, 3)
        v = np.concatenate((v_lower, v_upper, v_high))
        num_params_upper = {"mmax" : 30, "mmaxp" : 10, "lmax" : 30, "numkp" : 48}
        num_params_high = {"mmax" : 200, "mmaxp" : 10, "lmax" : 200, "numkp" : 48}
        num_params_lower = [num_params] * v_lower.size
        num_params_upper = [num_params_upper] * v_upper.size
        num_params_high = [num_params_high] * v_high.size
        num_params = np.concatenate((num_params_lower, num_params_upper, num_params_high))
    elif key == "beta":
        beta_lower = np.geomspace(3e-1, 2e1, 101, endpoint=False)
        beta_upper = np.geomspace(2e1, 1e2, 8)
        beta_high = np.geomspace(1e2, 1e3, 8)[1:-1]
        #beta=1e3 : ep_delta, ep_offset, k_ep = 0.010996237, -5.63e-16-0.018912556j, 0.001429161
        # beta_peak = np.array([1e3]) # data unreliable? (maybe R too big)
        #beta=9e2 : ep_delta, ep_offset, k_ep = 0.012693067, -2.99e-16-0.021428685j, 0.001661631
        # beta_peak = np.array([9e2]) # same here, still weird spike
        beta_peak = np.array([750])
        beta = np.concatenate((beta_lower, beta_upper, beta_high, beta_peak))
        num_params_upper = {"mmax" : 50, "mmaxp" : 10, "lmax" : 50, "numkp" : 48}
        num_params_high = {"mmax" : 500, "mmaxp" : 10, "lmax" : 500, "numkp" : 48}
        num_params_peak = {"mmax" : 500, "mmaxp" : 10, "lmax" : 500, "numkp" : 48}
        num_params_lower = [num_params] * beta_lower.size
        num_params_upper = [num_params_upper] * beta_upper.size
        num_params_high = [num_params_high] * beta_high.size
        num_params_peak = [num_params_peak] * beta_peak.size
        num_params = np.concatenate((num_params_lower, num_params_upper, num_params_high, num_params_peak))
    w = tlk.get_w(None, alpha, beta, v)
    args = (beta, K, v, g, a, w, num_params)
    return args

def compute_ep_size(args, labels, ktol=1e-9, etol=1e-9, load_only=False):
    filename = "ep_data_" + tlk.get_pars_string(args, labels)
    filename_full = DATAPATH + filename + ".npy"
    try:
        ep_delta, ep_offset, k_ep = np.load(filename_full)
        if not load_only:
            print("File", filename_full, "already exists, computation skipped")
    except FileNotFoundError:
        if load_only:
            raise FileNotFoundError("File", filename_full, "not found and load_only = True")
        else:
            ep_delta, ep_offset = tlk.timer(algo.ep_size_iter, gf.green_perturbative, args=args)
            k_ep = tlk.timer(algo.ep_search_iter, gf.green_perturbative, k_max=0.05, args=args,
                             ktol=ktol, etol=etol)
            np.save(filename_full, np.array([ep_delta, ep_offset, k_ep]))
            print("Saved data to:", filename_full)
    return ep_delta, ep_offset, k_ep

def plot_ep_size(axis, args, labels, vf=0.2):
    ep_delta, ep_offset, k_ep = compute_ep_size(args, labels, load_only=True)
    argi, indx = algo.find_iterable_argument(args)
    if indx == 1:
        arg = argi-1
        turning_point_indx = 0
        while arg[turning_point_indx] < 0:
            turning_point_indx += 1
        linthresh = arg[turning_point_indx]
        axis.set_xscale("symlog", linthresh=linthresh)
        arg = np.insert(arg, turning_point_indx, 0)
        ep_delta = np.insert(ep_delta, turning_point_indx, np.nan)
        ep_offset = np.insert(ep_offset, turning_point_indx, np.nan*0j)
        k_ep = np.insert(k_ep, turning_point_indx, np.nan)
        # xticks = [arg[0], -1e-3, -1e-5, 0, 1e-5, 1e-3, arg[-1]]
        # xticklabels = [arg[0]+1,"$1-10^{-3}$","$1-10^{-5}$", 1, "$1+10^{-5}$", "$1+10^{-3}$",arg[-1]+1]
        xticks = [arg[0], -1e-4, 0, 1e-4, arg[-1]]
        xticklabels = [arg[0]+1, "$1-10^{-4}$", 1, "$1+10^{-4}$", arg[-1]+1]
        axis.set_xticks(xticks, xticklabels)
    else:
        arg = argi
        axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(arg[0], arg[-1])
    axis.set_xlabel(tlk.get_label(labels[indx]))
    axis.plot(arg, ep_delta.real, label=tlk.get_ep_delta_string())
    axis.plot(arg, -ep_offset.imag, label=tlk.get_ep_offset_string())
    axis.plot(arg, k_ep.real * vf, label="$v_0$" + tlk.get_label("k ep"))

def _plot_inset_ep_size_example(axis, axi_size=(0.2, 0.05, 0.6, 0.45), store=False):
    axi = axis.inset_axes(axi_size, transform=axis.transAxes)
    axi.set_xticks([]); axi.set_yticks([])
    k_max = 0.11; k_count = 301
    k_vals = np.linspace(-k_max, k_max, k_count)
    axi.set_xlim(-k_max, k_max)
    axi.set_ylim(-0.125, 0.01)
    beta=10; v=0.4; w=tlk.get_w(None, 2, beta, v)
    dataname = DATAPATH + "green_ep_size_inset_example.npy"
    try:
        green = np.load(dataname)
        print("Loaded GF from disk")
    except FileNotFoundError:
        green = tlk.timer(gf.green_perturbative, k_vals, [0], beta, K=0.5, v=v, g=0.3, w=w,
                          num_params={"mmax" : 10, "mmaxp" : 5, "lmax" : 10, "numkp" : 48})
        if store:
            np.save(dataname, green)
            print("Stored GF to disk:", dataname)
    energies = algo.get_energies(green, [0])
    energies = tlk.contiguous_arrays(energies)
    for y_val in energies.T:
        axi.plot(k_vals, y_val.real, c='k')
        axi.plot(k_vals, y_val.imag, c='k')
    k_ep = 0.098604
    ep_delta, ep_offset = algo.energies_to_ep_size(energies[k_vals.size // 2])
    cv = mpl_special.Colors().colors
    scale = 8
    ArrowP = plt.matplotlib.patches.FancyArrowPatch
    p1 = ArrowP((0, ep_offset.imag), (0, ep_offset.imag - ep_delta), arrowstyle='<->', shrinkA=0,
                shrinkB=0, fc=cv[0], ec=cv[0], mutation_scale=scale, joinstyle="round", capstyle="round")
    p2 = ArrowP((0, 0), (0, ep_offset.imag), arrowstyle='<->', shrinkA=0, shrinkB=0, fc=cv[1],
                ec=cv[1], mutation_scale=scale, joinstyle="round", capstyle="round")
    p3 = ArrowP((0, ep_offset.imag), (k_ep, ep_offset.imag), arrowstyle='<->', shrinkA=0, shrinkB=0,
                fc=cv[2], ec=cv[2], mutation_scale=scale, joinstyle="round", capstyle="round")
    axi.add_patch(p1)
    axi.add_patch(p2)
    axi.add_patch(p3)
    axi.text(-k_ep/20, ep_offset.imag - ep_delta/2, r"$\Delta E_{\mathrm{EP}}$",
             transform=axi.transData, c=cv[0], ha="right", va="center")
    axi.text(-k_ep/20, ep_offset.imag/2.6, r"$\langle E_{\mathrm{EP}}\rangle$", transform=axi.transData,
             c=cv[1], ha="right", va="center")
    axi.text(k_ep/2, ep_offset.imag - ep_delta/3, tlk.get_label("k ep"),
             transform=axi.transData, c=cv[2], ha="center", va="center")
    return axi

def plot_ep_size_example(save=False):
    labels = ["beta", "K", "v", "g", "a", "w", "mmln"]
    figsize = mpl_special.set_figsize(FIGSIZE_PT)
    figure_dpi_orig = plt.rcParams["figure.dpi"]
    plt.rcParams["figure.dpi"] = 280
    fig, ax = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 1.3))
    keys = ["g", "K", "v", "beta"]
    for i in range(2):
        for j in range(2):
            key = keys[2*i + j]
            args = get_compute_ez_size_pars(key)
            plot_ep_size(ax[i, j], args, labels)
            ax[i,j].text(0.5, 1-0.03, tlk.panels[2*i+j], transform=ax[i,j].transAxes, ha="center", va="top")
    ax[0, 0].legend()
    ax[0, 0].set_ylim(ax[0, 0].get_ylim()[0], ax[0, 0].get_ylim()[1]*4)
    ax[1, 1].set_ylim(ax[1, 1].get_ylim()[0], ax[1, 1].get_ylim()[1]*0.6)
    _plot_inset_ep_size_example(ax[1, 1])
    mpl_special.embed_labels(fig, ax)
    # fig.get_layout_engine().set(wspace=0, w_pad=0)
    fig.tight_layout()
    fig.subplots_adjust(left=0.07, bottom=0.048, right=1-0.025, top=0.99, wspace=0.2, hspace=0.13)
    filename = "ep_size_quad_example"
    plt.rcParams["figure.dpi"] = figure_dpi_orig
    if save:
        tlk.savefig(fig, PATH + filename)

def plot_gf_ll_na_example(save=False, AB=False):
    u_a=1.5; u_b=1.2; beta=1; alpha=2; vf=0.5; a=1; w=np.pi
    K, v, g = 0.636, 0.537, 13.261
    num_params = {"mmax" : 5, "mmaxp" : 5, "lmax" : 5, "numkp" : 48}
    k_num, omega_num, green_num, _ = gf.green_numeric(u_a, u_b, beta=beta)
    omega_indx = omega_num.size // 2
    omega = omega_num[omega_indx]
    k_max = 2.05
    indx = tlk.arg_restrict(k_num, -k_max, k_max)
    k_num = k_num[indx]
    green_num = green_num[indx, omega_indx]
    green_num_lr = tlk.basis_ab_to_lr(green_num)
    grl, glr = green_num_lr[:, 0, 1], green_num_lr[:, 1, 0]
    gminus = (grl - glr) / 2
    goff = [gminus, -gminus]
    green_num_lr_manip = np.zeros_like(green_num_lr)
    for i in range(2):
        green_num_lr_manip[:, i, i] = green_num_lr[:, i, i]
        green_num_lr_manip[:, i, (i+1) % 2] = goff[i]
    green_num_ab_manip = tlk.basis_lr_to_ab(green_num_lr_manip)
    w = tlk.get_w(w, alpha, beta, v)
    k_vals = tlk.increase_density(k_num, factor=4)
    green = gf.green_perturbative(k_vals, omega, beta, K, v, g, a, w, num_params)
    fig, ax = plt.subplots(2, 2)
    if AB:
        RL = [r"\mathrm{A}", r"\mathrm{B}"]
    else:
        RL = [r"\mathrm{R}", r"\mathrm{L}"]
    for i in range(2):
        for j in range(2):
            tlk.plot_complex(ax[i,j], k_vals, green[:,0,i,j])
            if AB:
                g_num_manip = green_num_ab_manip
                g_num = green_num
            else:
                g_num_manip = green_num_lr_manip
                g_num = green_num_lr
            tlk.plot_complex(ax[i,j], k_num, g_num[:,i,j], set_label=False, ls='', marker='o')
            if i+j==1:
                ax[i,j].plot(k_num, g_num_manip[:,i,j].real,
                             ls='', mew=1, marker='x', c='b', alpha=0.7)
            ax[i,j].legend()
            tlk.autoformat(ax[i,j], xlabel="$k$", ylabel=rf"$G_{{{RL[i]}{RL[j]}}}$")
    mpl_special.embed_labels(fig, ax)
    labels = ["UA", "UB", "beta", "K", "v", "g", "a", "alpha", "mmln"]
    args = (u_a, u_b, beta, K, v, g, a, alpha, num_params)
    figure_type = "gf_ll_na_example"
    figure_type += ("_AB" if AB else "_LR")
    filename = figure_type + "_" + tlk.get_pars_string(args, labels)
    print(f"Created figure {filename}")
    if save:
        tlk.savefig(fig, PATH + filename)

def plot_rg_flow(axes, l_vals, vf=0.5, u_a=1.5, u_b=-1.0, a=1, K_0=1, methods="ope", magnitude=True):
    methods = tlk.asarray(methods)
    labels = ["$g_2$", "$g_4$", "$K$", "$v$"]
    couplings = np.array([algo.get_rg_flow_h2h4(l_vals, vf, u_a, u_b, a, K_0, method, magnitude)
                          for method in methods])
    for i, axis in enumerate(axes):
        axis.set_xlim(l_vals[0], l_vals[-1])
        axis.set_xlabel("$l$")
        colors = mpl_special.Colors().colors
        for i_m, method in enumerate(methods):
            coupling = couplings[i_m].T[i]
            label = labels[i] if i_m==0 else None
            ls = '-' if i_m==0 else '--'
            axis.plot(l_vals, coupling, label=label, ls=ls, c=colors[i])
        if magnitude and (i == 0 or i == 1):
            axis.set_yscale("log")
        axis.legend()

def plot_rg_example():
    vf = 0.5; K_0 = 0.8; a = 1
    u_a = 0.05; u_b = 0.0
    fig, ax = plt.subplots(2, 2)
    l_vals = algo.get_l_vals(2)
    plot_rg_flow(ax.flat, l_vals, vf, u_a, u_b, a, K_0, methods=["ms", "ope"])
    mpl_special.embed_labels(fig, ax)

def get_rg_hypothetical_example():
    """If K>1 the dynamic equilibrium assumption works is satisfied for sufficiently weak interactions"""
    vf = 0.5; K_0 = 1.5; a = 1
    u_av = np.linspace(1e-4, 0.2, 200); u_b = 0.0
    couplings_final = np.ones((4, u_av.size))
    for i, u_a in enumerate(u_av):
        l_init = 20
        l_vals = algo.get_l_vals(l_init, l_count=10)
        couplings = algo.get_rg_flow_h2h4(l_vals, vf, u_a, u_b, a, K_0, "ms", rtol=1e-13, atol=1e-13)
        couplings_final[:, i] = couplings[-1]
    return couplings_final


def plot_rel_err_sb1s(axis, mode="m", k=0.4, omega=0.1, b=0.6, a=1.0, s=1, max_val=18, step=6):
    mmaxv = np.arange(0, max_val + 1, 1 if mode == "m" else step)
    lmaxv = np.arange(0, max_val + 1, 1 if mode == "l" else step)
    labels = {"m" : r"m_\mathrm{max}", "l" : r"l_\mathrm{max}"}
    arrvs = {"m" : mmaxv, "l" : lmaxv}
    axis.set_yscale("log")
    def get_other_label(mode):
        if mode == "m":
            return "l"
        return "m"
    for arg_scalar in arrvs[get_other_label(mode)]:
        if mode == "m":
            args = (k, omega, b, a, s, mmaxv, arg_scalar)
        elif mode == "l":
            args = (k, omega, b, a, s, arg_scalar, lmaxv)
        arg, _ = algo.find_iterable_argument(args)
        rel_err = algo.err_sb1s_arg(args)
        axis.set_xlabel(f"${labels[mode]}$")
        label = f"${labels[get_other_label(mode)]} = {arg_scalar}$"
        axis.plot(arg, rel_err, marker='o', ls='-', lw=LINEWIDTH_THIN, label=label)
    axis.set_xlim(arg[0], arg[-1])
    axis.set_xticks(arg[::4])
    axis.legend()

def plot_rel_err_sb1s_example(save=False):
    fig, ax = plt.subplots(1, 2)
    k=1.3; omega=0.1; b=0.4; s=1; av=[0.5, 1.0]
    ax[0].set_ylabel(r"$|\Delta S|$")
    lw = ax[0].spines["top"].get_linewidth()
    bbox_props = {"boxstyle" : "square", "facecolor" : "white", "linewidth" : lw}
    for i_a, a in enumerate(av):
        ax[i_a].text(0.5, 0.97, rf"$a={a}$", ha="center", va="top",
                     transform=ax[i_a].transAxes, bbox=bbox_props)
        plot_rel_err_sb1s(ax[i_a], "l", k, omega, b, a, s)
    for i_a, axis in enumerate(ax):
        axis.set_ylim(9e-16, 8)
        if i_a > 0:
            axis.set_yticks([])
    mpl_special.embed_labels(fig, ax)
    labels = ["k", "omega", "b", "a", "s"]
    filename = "rel_err_sb1s_example_" + tlk.get_pars_string((k, omega, b, av, s), labels)
    print(f"Created figure {filename}")
    if save:
        tlk.savefig(fig, PATH + filename)

def get_k_vals_num_example():
    k_vals = np.linspace(-30, 30, 1000)
    k_vals_num = np.linspace(k_vals[0], k_vals[-1], 35 + 1, endpoint=False)
    k_vals_num += (k_vals_num[1] - k_vals_num[0]) / 2
    return k_vals, k_vals_num

def get_sb1s_example_data(omega=1j, b=0.6, a=1.0, s=1, store=False):
    args = (omega, b, a, s)
    labels = ["omega", "b", "a", "s"]
    filename = "sb1s_example_data_" + tlk.get_pars_string(args, labels)
    try:
        vals_num = np.load(DATAPATH + filename + ".npy")
    except FileNotFoundError:
        def get_L(k):
            if np.abs(k) < 0.1: return 30
            if np.abs(k) < 1: return 20
            if np.abs(k) < 10: return 15
            return 10
        _, k_vals_num = get_k_vals_num_example()
        vals_num = []
        def integrand(x, tau):
            return (np.exp(-1j*k*x)*(1/np.tan(tau+1j*s*(x+a)) + 1/np.tan(tau+1j*s*(x-a)))
                    / np.sin(tau+1j*x)**(1+b) / np.sin(tau-1j*x)**b)
        for k in k_vals_num:
            L = get_L(k)
            vals_num.append(tlk.c_quad(lambda tau: np.exp(omega*tau)
                                       * tlk.c_quad(lambda x: integrand(x, tau), -L, L)[0], 0, np.pi))
    if store:
        np.save(DATAPATH + filename + ".npy", np.array(vals_num))
    return np.array(vals_num)

def get_jbn_example_data(omega=1j, b=0.6, a=1.0, n=1, store=False):
    args = (omega, b, a, n)
    labels = ["omega", "b", "a", "n"]
    filename = "jbn_example_data_" + tlk.get_pars_string(args, labels)
    try:
        vals_num = np.load(DATAPATH + filename + ".npy")
    except FileNotFoundError:
        def get_L(k):
            if np.abs(k) < 0.1: return 30
            if np.abs(k) < 1: return 20
            if np.abs(k) < 10: return 15
            return 10
        _, k_vals_num = get_k_vals_num_example()
        vals_num = []
        def integrand(x, tau):
            return np.exp(-1j*k*x)/ np.sin(tau+1j*x)**(n+b) / np.sin(tau-1j*x)**b
        for k in k_vals_num:
            L = get_L(k)
            vals_num.append(tlk.c_quad(lambda tau: np.exp(omega*tau)
                                       * tlk.c_quad(lambda x: integrand(x, tau), -L, L)[0], 0, np.pi))
    if store:
        np.save(DATAPATH + filename + ".npy", np.array(vals_num))
    return np.array(vals_num)

def plot_special_func_na_example(save=False):
    omega=1j; b=0.6; a=1.0; n=1; s=1
    fl = [f"J_{{b,{n}}}", f"S_{{b,1,{s}}}"]
    k_vals, k_vals_num = get_k_vals_num_example()
    k_vals = tlk.increase_density(k_vals, factor=4)
    jbn_vals_num = get_jbn_example_data()
    sb1s_vals_num = get_sb1s_example_data()
    jbn_vals = gf.j_bn(k_vals, omega, b, n)
    sb1s_vals = np.array([gf.s_b1s(k, omega, b, a, s, mmax=20, lmax=20) for k in k_vals])
    vals = [jbn_vals, sb1s_vals]
    vals_num = [jbn_vals_num, sb1s_vals_num]
    figsize = mpl_special.set_figsize(FIGSIZE_PT)
    fig, ax = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]/1.2))
    for i in range(2):
        ax[i].set_xlim(k_vals[0], k_vals[-1])
        ax[i].set_xlabel("$k$")
        ax[i].set_yscale("log")
        labels = {"real" : fr"$|\,\mathrm{{Re}}\,{fl[i]}|$", "imag" : fr"$|\,\mathrm{{Im}}\,{fl[i]}|$"}
        for attr in ["real", "imag"]:
            ax[i].plot(k_vals, np.abs(getattr(vals[i], attr)), label=labels[attr])
            ax[i].plot(k_vals_num, np.abs(getattr(vals_num[i][:,0], attr)), ls='', marker='o', c='k')
        ax[i].legend()
        ax[i].text(0.03, 1-0.025, tlk.panels[i], transform=ax[i].transAxes, ha="left", va="top")
        if i == 0:
            ax[i].set_ylim(3e-1, 3e1)
        if i == 1:
            ax[i].set_ylim(2e-3, 2e1)
    mpl_special.embed_labels(fig, ax)
    filename = "appendix_special_func_na_example"
    print(f"Created figure {filename}")
    if save:
        tlk.savefig(fig, PATH + filename)

def plot_special_func_example(save=False):
    omega=0; omega2=1; b=0.6; a=1.0; nv=[0,1,2]
    fl = [f"J_{{b,{n}}}" for n in nv]
    k_vals = np.linspace(-8.5, 8.5, 300)
    vals = [gf.j_bn(k_vals, omega, b, n) for n in nv]
    vals2 = [gf.j_bn(k_vals, omega2, b, n) for n in nv]
    for s in [+1,-1]:
        vals.append(np.array([gf.s_b1s(k, omega, b, a, s, mmax=20, lmax=20) for k in k_vals]))
        vals2.append(np.array([gf.s_b1s(k, omega2, b, a, s, mmax=20, lmax=20) for k in k_vals]))
        fl.append(fr"$S_{{b,1,{s}}}$")

    figsize = mpl_special.set_figsize(FIGSIZE_PT)
    figure_dpi_orig = plt.rcParams["figure.dpi"]
    plt.rcParams["figure.dpi"] = 280
    fig, ax = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 1.3))
    for i in range(4):
        axis = ax[i // 2, i % 2]
        axis.set_xlim(k_vals[0], k_vals[-1])
        axis.set_xlabel("$k$")
        labels = {"real" : fr"$\,\mathrm{{Re}}\,{fl[i]}$", "imag" : fr"$\,\mathrm{{Im}}\,{fl[i]}$"}
        if i < 3:
            for attr in ["real", "imag"]:
                line = axis.plot(k_vals, getattr(vals[i], attr), label=labels[attr])
                axis.plot(k_vals, getattr(vals2[i], attr), ls='--', c=line[0].get_color())
        if i == 3:
            for j in range(2):
                line = axis.plot(k_vals, vals[i+j].real, label=fl[i+j])
                axis.plot(k_vals, vals2[i+j].real, ls='--', c=line[0].get_color())
        axis.legend()
        axis.text(0.03, 1-0.025, tlk.panels[i], transform=axis.transAxes, ha="left", va="top")
        mpl_special.set_ticks_linear(axis, -8, 8, 5, dtype=int)
    mpl_special.embed_labels(fig, ax)
    plt.rcParams["figure.dpi"] = figure_dpi_orig
    filename = "appendix_special_func_example"
    print(f"Created figure {filename}")
    if save:
        tlk.savefig(fig, PATH + filename)


def main():
    """Run all Plots"""
    # plot_test_figure()
    # plot_ll_par_fit()
    # plot_ll_example_ep()
    # plot_gf_ua_eq_ub_par_fit()
    # plot_compare_na_ll_par_ua_eq_ub()
    # plot_numeric_paper_initial()
    # plot_ep_contour_example()
    # plot_ep_size_example()
    # plot_rg_example()

    return 0


if __name__ == "__main__":
    main()
