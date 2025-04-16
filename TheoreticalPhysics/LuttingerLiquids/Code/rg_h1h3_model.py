#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:19:24 2023

@author: ag_budich1
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import mpl_special

V0 = 0.5
A0 = 1
BETA0 = 10
K0 = 1
U_A = 2.5
U_B = 0.6
L0 = 100
U_MINUS = (U_A - U_B) / 2
U_PLUS = (U_A + U_B) / 2
#LAMBDA0 = V0 / A0**2
LAMBDA0 = np.pi / A0
LAMBDAF = 1 / BETA0 / V0
G10 = -U_MINUS*A0 / (2*np.pi)
G20 = U_PLUS*A0 / (8*np.pi**2)
G30 = -U_PLUS*A0 * 2*np.pi**2 / (2*np.pi)**4
G40 = U_PLUS*A0 / (np.pi * V0)
LMAX = np.log(LAMBDA0 / LAMBDAF)
if LMAX < 0:
    raise ValueError(f"RG flow not possible for beta = {BETA0:.4f}, a = {A0:.4f} and v = {V0:.4f}")

def rg_beta_h1h3(couplings, l_val):
    c1, c3, K = couplings
    c1p = (1-K)*c1
    c3p = (2 - 4*K) * (c3 - 2*np.pi*K*c1**2)
    Kp = -K**2* (-16*np.pi*c3**2 + 8*np.pi*(K-1) * (2*K-1)*c1**2)
#     c3p = (2 - 4*K) * (c3 - np.pi*c1**2)
#     Kp = -K**2* (-16*np.pi*K*c3**2 + 8*np.pi*(K-1) * (2*K-1)*c1**2)
    return np.array([c1p,c3p,Kp])

def rg_beta_h1h3_var(couplings, l_val):
    c1, c3, K1, K2 = couplings      # K=1/sqrt(K1K2), v=sqrt(K1/K2)
    K = 1 / np.sqrt(K1 * K2)
    v = np.sqrt(K1 / K2)
    c1p = (1 - K) * c1 - 2*np.pi*c1*c3
    c3p = (2 - 4*K) * c3 + np.pi*c1**2
    K1p = -16*np.pi**2*v*c3**2 + 2*np.pi**2*v*(1/K - 1) * (3/K - 4) * c1**2
    K2p = -16*np.pi**2 / v * c3**2 + 2*np.pi**2 / v * (1/K - 1) * c1**2
    return np.array([c1p, c3p, K1p, K2p])

def rg_beta_h1h3_var2(couplings, l_val):
    c1, c3, K, v = couplings
    c1p = (1 - K) * c1 - 2*np.pi*c1*c3
    c3p = (2 - 4*K) * c3 + np.pi*c1**2
    Kp = 16*np.pi**2*K**2 * c3**2 + 4*np.pi**2 * (1-K)**2 * c1**2
    vp = -2*np.pi**2 * v * (1-K) * (2-K) / K * c1**2
    return np.array([c1p, c3p, Kp, vp])

def rg_beta_h1h3_var3(couplings, l_val):
    c1, c3, K1, K2 = couplings      # K=1/sqrt(K1K2), v=sqrt(K1/K2)
    K = 1 / np.sqrt(K1 * K2)
    v = np.sqrt(K1 / K2)
    c1p = (1 - K) * c1 + 2*np.pi*c1*c3
    c3p = (2 - 4*K) * c3 - np.pi*c1**2
    K1p = 16*np.pi**2*v*c3**2 - 2*np.pi**2*v*(1/K - 1) * (3 - 4/K) * c1**2
    K2p = 16*np.pi**2 / v * c3**2 - 2*np.pi**2 / v * (1/K - 1) * c1**2
    return np.array([c1p, c3p, K1p, K2p])

def rg_beta_h1h3_var4(couplings, l_val):
    c1, c3, K, v = couplings
    c1p = (1 - K) * c1 + 2*np.pi*c1*c3
    c3p = (2 - 4*K) * c3 - np.pi*c1**2
    Kp = -16*np.pi**2*K**2*c3**2 - 4*np.pi**2*(1-K)**2/K * c1**2
    vp = v*2*np.pi**2*(2-K)*(1/K - 1) * c1**2
    return np.array([c1p, c3p, Kp, vp])

def get_rg_flow(rg_beta, couplings0, l_vals=None):
    if l_vals is None:
        l_vals = np.linspace(0, LMAX, 200)
    couplings = odeint(rg_beta, couplings0, l_vals)
    return couplings, l_vals

def plot_rg_flow_example(var, vf=0.5, u_a=1.5, u_b=-1.0, a=1, lmax=5, K_0=1,
                         model="h1h3", magnitude=True):
    l_vals = np.linspace(0, lmax, 200)
    labels = ["$g_1$", "$g_3$", "$K$", "$v$"]
    method = "rg_beta_" + model + "_" + var
    v0 = vf / K_0
    u_minus = (u_a - u_b) / 2
    u_plus = (u_a + u_b) / 2
    g10 = -u_minus*a/(4*np.pi**2*v0) * K_0 * np.sqrt(2)
    g30 = -u_plus*a/(4*np.pi**2*v0)
    if var == "var3":
        K_10 = v0/K_0; K_20 = 1/(v0*K_0)
        couplings, l_vals = get_rg_flow(globals()[method], [g10, g30, K_10, K_20], l_vals)
        c1, c3, K1, K2 = couplings.T
        K = 1 / np.sqrt(K1 * K2); v = np.sqrt(K1 / K2)
        couplings = np.array([c1, c3, K, v]).T
    elif var == "var4":
        couplings, l_vals = get_rg_flow(globals()[method], [g10, g30, K_0, v0], l_vals)

    if magnitude:
        couplings = np.abs(couplings)
    plot_rg_flow(couplings, l_vals, labels)


def plot_rg_flow_single(ax, coupling, l_vals, label, **kwargs):
    ax.plot(l_vals, coupling, label=label, **kwargs)

def plot_rg_flow(couplings, l_vals, labels, style_kwargs: dict=None):
    if style_kwargs is None:
        style_kwargs = [{}] * len(labels)
    fig, ax = plt.subplots()
    for coupling, label, style_kwarg in zip(couplings.T, labels, style_kwargs):
        plot_rg_flow_single(ax, coupling, l_vals, label, **style_kwarg)
    ax.set_xlim(l_vals[0], l_vals[-1])
    #ax.set_yscale("log")
    ax.set_xlabel("$l$")
    ax.legend()
    mpl_special.embed_labels(fig, ax)


def get_couplings(rg_beta, u_a, u_b, vf=0.5, beta=1, a=1, alpha=None, l_max=20, n_l=2):
    """(rg_beta, u_a, u_b, vf, beta, a, alpha, l_max, n_l) -> (g1, g3, K, v, w) [RG]"""
    if alpha is None:
        alpha = 1 / vf
    u_minus = (u_a - u_b) / 2
    u_plus = (u_a + u_b) / 2
    K_0 = 1 / np.sqrt(1 + 2 * u_plus * a / (np.pi * vf) * alpha**2 / (a**2 + alpha**2))
    v0 = vf / K_0
    g10 = -u_minus*a / (4*np.pi**2 * v0)
    g30 = -u_plus*a / (4*np.pi**2 * v0)
    if l_max is None:
        lambda0 = 1 / alpha
        lambdaf = 1 / beta / v0
        l_max = np.log(lambda0 / lambdaf)
        if l_max <= 0:
            raise ValueError(f"l_max must be positive! ({beta=:.2e}, {vf=:.2e}, {alpha=:.2e}")
    l_vals = np.linspace(0, l_max, n_l)
    if rg_beta == rg_beta_h1h3_var:
        K_10 = V0/K_0; K_20 = 1/(V0*K_0)
        couplings, _ = get_rg_flow(rg_beta_h1h3_var, [g10, g30, K_10, K_20], l_vals)
        c1, c3, K1, K2 = couplings.T
        K = 1 / np.sqrt(K1 * K2); v = np.sqrt(K1 / K2)
        couplings = np.array([c1, c3, K, v]).T
    else:
        couplings, _ = get_rg_flow(rg_beta, [g10, g30, K_0, v0], l_vals)
    return *couplings[-1], np.pi*alpha / beta / vf * np.exp(l_max)


def main():
    print(__doc__)

    labels = ["$g_1$", "$g_3$", "$K$"]
    couplings0 = np.array([G10, G30, K0])
    l_vals = np.linspace(0, LMAX, 200)
    couplings, l_vals = get_rg_flow(rg_beta_h1h3, couplings0, l_vals)
    # plot_rg_flow(np.abs(couplings), l_vals, labels)
    # plot_rg_flow(couplings, l_vals, labels)
    l_vals = np.linspace(0, 5, 200)
    K10 = V0/K0; K20 = 1/(V0*K0)
    couplings, l_vals = get_rg_flow(rg_beta_h1h3_var, [G10, G30, K10, K20], l_vals)
    c1, c3, K1, K2 = couplings.T
    K = 1 / np.sqrt(K1 * K2); v = np.sqrt(K1 / K2)
    # plot_rg_flow(np.array([c1, c3, K, v]).T, l_vals, ["$g_1$", "$g_3$", "$K$", "$v$"])

    vf = 0.5; u_a = -1.5; u_b = 1.0
    beta = 1; a = 1; alpha = 1
    u_minus = (u_a - u_b) / 2
    u_plus = (u_a + u_b) / 2
    K_0 = 1 / np.sqrt(1 + 2 * u_plus * a / (np.pi * vf) * alpha**2 / (a**2 + alpha**2))
    v0 = vf / K_0
    g10 = -u_minus*a / (4*np.pi**2 * v0)
    g30 = -u_plus*a / (4*np.pi**2 * v0)
    lambda0 = np.pi / A0
    lambdaf = 1 / beta / V0
    l_max = np.log(lambda0 / lambdaf)
    l_vals = np.linspace(0, l_max, 200)
    couplings, _ = get_rg_flow(rg_beta_h1h3_var2, [g10, g30, K_0, v0], l_vals)
    # plot_rg_flow(np.abs(couplings), l_vals, ["$g_1$", "$g_3$", "$K$", "$v$"])
    # plt.gca().set_yscale("log")
    print(couplings[-1])

if __name__ == "__main__":
    main()
