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

PRIM_SCALING = 0    # primitive scaling of any action (presumably dimensionless --> no scaling)
W = 1               # RG cut-off for C_nm correlators --> hope that influence for W ~ 1 is small
V0 = 0.5
A0 = 1
BETA0 = 10
K0 = 1
U_A = 1.5
U_B = 0.6
L0 = 100
U_MINUS = (U_A - U_B) / 2
U_PLUS = (U_A + U_B) / 2
LAMBDA0 = V0 / A0**2
LAMBDAF = 1 / BETA0 / V0
G10 = -U_MINUS*A0 / (2*np.pi*L0)
G20 = U_PLUS*A0 / (8*np.pi**2*A0**2)
G30 = -U_PLUS*A0 * 2*np.pi**2*A0**2 / L0**4
G40 = U_PLUS*A0 / (np.pi * V0)
LMAX = np.log(LAMBDA0 / LAMBDAF)
if LMAX < 0:
    raise ValueError(f"RG flow not possible for beta = {BETA0:.4f}, a = {A0:.4f} and v = {V0:.4f}")
    
def get_g(c4, K):
    return 2*np.pi * K * c4 * G40

def phi_f_correlator(x, tau, c4=1, K=1, beta=1, v=1, Lambda=1, a=1):
    arg = v*Lambda*np.sqrt(1 + get_g(c4, K) * np.cos(Lambda*a))
    return 2*np.cos(Lambda*x) / np.sinh(beta/2 * arg) * np.cosh((beta/2 - tau) * arg)

def rg_beta_tree(couplings, l_val):
    """Beta function at tree level"""
    c1, c2, c3, c4, a, beta, Lambda, v, K = couplings
    phi_f_val0 = phi_f_correlator(0, 0, c4, K, beta, v, Lambda, a)
    phi_f_valx = phi_f_correlator(a, 0, c4, K, beta, v, Lambda, a)
    c1p = c1 * (PRIM_SCALING - K* phi_f_val0)
    c2p = c2 * (PRIM_SCALING - K* 2*phi_f_val0 + 2*K*phi_f_valx)
    c3p = c3 * (PRIM_SCALING - K* 2*phi_f_val0 - 2*K*phi_f_valx)
    c4p = PRIM_SCALING
    ap = -a
    betap = -beta
    Lambdap = -Lambda
    vp = 0
    Kp = 0
    return np.array([c1p, c2p, c3p, c4p, ap, betap, Lambdap, vp, Kp])

def C_00(s, c4, a, beta, Lambda, v, K):
    """S2 S2 correlator with x^0 y^0"""
    kappa = np.sqrt(1 + 2*get_g(c4, K) * np.cos(Lambda*a))
    bvl = beta * v * Lambda
    hyperbolic = 2 * np.sinh(W*kappa) / np.tanh(bvl*kappa) / kappa
    sinusoidal = 2 * np.sin(W) * np.cos(Lambda*s*a)
    return 2/Lambda**2 * sinusoidal * hyperbolic

def C_20(s, c4, a, beta, Lambda, v, K):
    """S2 S2 correlator with x^2 y^0"""
    kappa = np.sqrt(1 + 2*get_g(c4, K) * np.cos(Lambda*a))
    bvl = beta * v * Lambda
    hyperbolic = 2 * np.sinh(W*kappa) / np.tanh(bvl*kappa) / kappa
    sinusoidal = 2 * (2*W*np.cos(W) + (W**2-2)*np.sin(W)) * np.cos(Lambda*s*a)
    return 2/Lambda**2 * sinusoidal * hyperbolic

def C_02(s, c4, a, beta, Lambda, v, K):
    """S2 S2 correlator with x^0 y^2"""
    kappa = np.sqrt(1 + 2*get_g(c4, K) * np.cos(Lambda*a))
    bvl = beta * v * Lambda
    hyperbolic = (2 * ((W**2*kappa**2 + 2)*np.sinh(W*kappa) - 2*W*kappa * np.cosh(W*kappa))
                  / np.tanh(bvl*kappa) / kappa)
    sinusoidal = 2 * np.sin(W) * np.cos(Lambda*s*a)
    return 2/Lambda**2 * sinusoidal * hyperbolic

def rg_beta_1loop(couplings, l_val):
    """Beta function with 1-loop corrections"""
    c1, c2, c3, c4, a, beta, Lambda, v, K = couplings
    phi_f_val0 = phi_f_correlator(0, 0, c4, K, beta, v, Lambda, a)
    phi_f_valx = phi_f_correlator(a, 0, c4, K, beta, v, Lambda, a)
    c1p = c1 * (PRIM_SCALING - K* phi_f_val0)
    c2p = c2 * (PRIM_SCALING - K* 2*phi_f_val0 + 2*K*phi_f_valx)
    c3p = c3 * (PRIM_SCALING - K* 2*phi_f_val0 - 2*K*phi_f_valx)
    c4p = PRIM_SCALING
    ap = -a
    betap = -beta
    Lambdap = -Lambda
    # S2S2_mod = (c2**2 * G20**2 * K * 8*a**2/v**2
    #             * (2*C_00(0, c4, a, beta, Lambda, v, K)
    #                 - 4*C_00(1, c4, a, beta, Lambda, v, K)
    #                 + 2*C_00(2, c4, a, beta, Lambda, v, K)))
    S2S2_mod = (c2**2 * G20**2 * K * 8*a**2/v**2
                * (-2*Lambda**2 * a**2 * C_00(1, c4, a, beta, Lambda, v, K)))
    S3S3_mod = (c3**2 * G30**2 * K * 16/v**2
                * (C_20(0, c4, a, beta, Lambda, v, K) + C_02(0, c4, a, beta, Lambda, v, K)))
    S_sqr_mod = S2S2_mod + S3S3_mod
    vp = S_sqr_mod / 2 * np.pi*K*v**2
    Kp = S_sqr_mod / 2 * np.pi*K**2*v
    return np.array([c1p, c2p, c3p, c4p, ap, betap, Lambdap, vp, Kp])

def plot_rg_flow():
    lv = np.linspace(0, LMAX, 200)
    couplings0 = np.array([1, 1, 1, 1, A0, BETA0, LAMBDA0, V0, K0])
    yv_tree = odeint(rg_beta_tree, couplings0, lv)
    yv = odeint(rg_beta_1loop, couplings0, lv)
    fig, ax = plt.subplots()
    labels = ["c_1", "c_2", "c_3", "c_4", "a", r"\beta", r"\Lambda", "v", "K"]
    for i in [0, 1, 2, 8]:
        line = ax.plot(lv, yv_tree[:,i], label=f"${labels[i]}$")
        ax.plot(lv, yv[:,i], c=line[0].get_color(), ls='--')
    ax.set_xlim(lv[0], lv[-1])
    ax.set_ylim(1e-6, 1.5)
    ax.set_yscale("log")
    ax.set_xlabel("$l$")
    ax.legend()
    mpl_special.embed_labels(fig, ax)
    
# plot_rg_flow()