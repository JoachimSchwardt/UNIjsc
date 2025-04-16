#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:26:26 2023

@author: joachim
"""

import numpy as np
from scipy import special
from special_functions import j_0, j_1
from greens_j import get_data
import green_toolkit


def pochhammer_var(val, index, variant="approx"):
    if index == 0:
        return 1

    if variant == "exact":
        return np.abs(special.gamma(1 + val + index) / special.gamma(1 + val))**2
    elif variant == "approx":
        return np.pi * index / np.sin(np.pi * index)
    else:
        raise NotImplementedError(f"{variant = } does not exist for 'pochhammer_var'")


def F_1(xi_p, xi_m, beta_u=2.6, alpha=1e-3, variant="approx"):
    """F_1 function with 'beta_u = beta * u'"""
    gamma_xi_m = pochhammer_var(alpha/beta_u, 1j * xi_m/beta_u, variant)
    gamma_xi_p = pochhammer_var(alpha/beta_u, 1j * xi_p/beta_u, variant)
    regularization = (alpha + 1j*xi_m) * (alpha + 1j*xi_p) / alpha**2
    return np.log(regularization) / 2 - np.log(gamma_xi_m * gamma_xi_p) / 2


def F_2(xi_p, xi_m, beta_u=2.6, alpha=1e-3, variant="approx"):
    """F_2 function with 'beta_u = beta * u'"""
    gamma_xi_m = pochhammer_var(alpha/beta_u, 1j * xi_m/beta_u, variant)
    gamma_xi_p = pochhammer_var(alpha/beta_u, 1j * xi_p/beta_u, variant)
    regularization = (alpha + 1j*xi_m) / (alpha + 1j*xi_p)
    return np.log(regularization) / 2 - np.log(gamma_xi_m / gamma_xi_p) / 2


def F_ell(xi_ell, beta_u=2.6, alpha=1e-3, variant="approx"):
    """F_ell == F_1 - ell * F_2 == F(xi_ell) with 'beta_u = beta * u'"""
    gamma_xi_ell = pochhammer_var(alpha/beta_u, 1j * xi_ell/beta_u, variant)
    regularization = (alpha + 1j*xi_ell) / alpha
    return np.log(regularization) - np.log(gamma_xi_ell)


def exp_expval(a_vals, b_vals, xi_vals, K=1, beta_u=2.6, alpha=1e-3, variant="approx"):
    """Exponential expectation value identity ::
        <prod_j exp(i * A_j * phi(xi_j) + i * B_j * theta(xi_j))> ==
            == exp(1/2 * sum_{j<k} (K * A_j * A_k + 1/K * B_j * B_k) * F_1(xi_j - xi_k)
                   -1/2 * sum_{j<k} (A_j * B_k + A_k * B_j) * F_2(xi_j - xi_k))
    """
    a_vals = np.asarray(a_vals)
    b_vals = np.asarray(b_vals)
    xi_vals = np.asarray(xi_vals)
    if not np.allclose(np.sum(a_vals), 0) or not np.allclose(np.sum(b_vals), 0):
        return 0.0
    exponent = 0.0
    for k in range(1, a_vals.size):
        for j in range(k):
            xi_p = xi_vals[j,0] - xi_vals[k,0]
            xi_m = xi_vals[j,1] - xi_vals[k,1]
            exponent += ((K * a_vals[j] * a_vals[k] + 1/K * b_vals[j] * b_vals[k])
                         * F_1(xi_p, xi_m, beta_u, alpha, variant)
                         - ((a_vals[j] * b_vals[k] + a_vals[k] * b_vals[j])
                            * F_2(xi_p, xi_m, beta_u, alpha, variant)))
    return np.exp(exponent / 2)


def green_numeric(u_a=1.5, u_b=1.5, N=None, beta=None):
    """(u_a, u_b) -> (k, omega, green, h_eff)"""
    if N is None:
        path = f"data_joachim/N*SBA_UA{u_a:.6f}_UB{u_b:.6f}_h0.030000"
    else:
        path = f"data_joachim/N{N}*SBA_UA{u_a:.6f}_UB{u_b:.6f}_h0.030000"
    if beta is not None:
        if beta == 1:
            path += f"_beta{beta}"
    path += "/"
    data = get_data(path)
    k_vals = data["k_values"] - np.pi
    omega_vals = data["omega_values"]
    green = data["ret_green_k_w"]
    h_eff = data["h_eff_k_w"]
    if k_vals.size == 80:
        green = np.roll(green, -40, axis=0)
        h_eff = np.roll(h_eff, -40, axis=0)
    return k_vals, omega_vals, green, h_eff


def green_order_0(k_vals, omega=0.0, beta=1.1, K=0.6, u=1.0, alpha=1e-3):
    """Zeroth order perturbation theory for SP Green's function"""
    k_vals = np.asarray(k_vals)
    M = (K + 1/K - 2) / 2
    prefactor = 1j*np.sin(np.pi*M/2) * (np.pi * alpha /(beta*u))**M * beta / (2*np.pi**2)
    k_p = (omega + u*k_vals) / (2*u)
    k_m = (omega - u*k_vals) / (2*u)
    g_rr = prefactor * j_0(beta*u * k_p / np.pi, 1 + M/2) * j_0(beta*u * k_m / np.pi, M/2)
    g_ll = prefactor * j_0(beta*u * k_m / np.pi, 1 + M/2) * j_0(beta*u * k_p / np.pi, M/2)
    green = np.array([np.array([[g_rr[i], 0], [0, g_ll[i]]])
                      for i in range(k_vals.size)])
    return green


def green_order_1(k_vals, omega=0.0, beta=1.1, K=0.6, u=1.0, alpha=1e-3):
    """First order perturbation theory for SP Green's function"""
    k_vals = np.asarray(k_vals)
    N = (1/K - K) / 2
    k_p = (omega + u*k_vals) / (2*u)
    k_m = (omega - u*k_vals) / (2*u)
    print(f"PARAMS: {beta=}, {K=}, {u=}, {alpha=}, {omega=}")
    j_1_k_p = np.array([j_1(beta*u/np.pi * k_p[i], K) for i in range(k_vals.size)])
    if np.allclose(k_m, k_p[::-1]):
        j_1_k_m = j_1_k_p[::-1]
    else:
        print("Asymmetric k-values in 'green_order_1', comp. cost *= 2!")
        j_1_k_m = np.array([j_1(beta*u/np.pi * k_m[i], K) for i in range(k_vals.size)])
    # g_rl = (-1j * 8**(K-1) * beta**2 / np.pi**4 * np.cos(np.pi*N/2)
    #         * (np.pi*alpha / (beta*u))**(N+2*K-2) * j_1_k_p * j_1_k_m)
    g_rl = (8**(K-1) * beta**2 / np.pi**4 * np.sin(np.pi*N/2)
            * (np.pi*alpha / (beta*u))**(N+2*K-2) * j_1_k_p * j_1_k_m)
    g_lr = np.copy(g_rl)
    green = np.array([np.array([[0, g_rl[i]], [g_lr[i], 0]])
                      for i in range(k_vals.size)])
    return green


# def green_order_2(k_vals, omega=0.0, beta=1.1, K=0.6, variant="approx"):
#     """Second order perturbation theory for SP Green's function"""
#     k_vals = np.asarray(k_vals)
#     M = (K + 1/K - 2) / 2
#     if variant == "approx":
#         g_ll = np.array([-2**(M+2*K) * beta**3/(8*np.pi*K**2)
#                          * omega / (omega + k + 1j * M * np.pi/beta)
#                          for k in k_vals])
#     else:
#         raise NotImplementedError(f"Variant {variant} does not exist (yet)!")
#     g_rr = -np.copy(g_ll)[::-1]    # assumes symmetry in k_vals (-3, ..., +3, equidistant)
#     green = np.array([np.array([[g_rr[i], 0], [0, g_ll[i]]])
#                       for i in range(k_vals.size)])
#     return green


def green_perturbative(k_vals, omega_vals=0, beta=1.1, K=0.6, u=1.0, u_minus=0.25, order=1):
    """Perturbative calculation of the SP Green's function"""
    omega_vals = np.asarray(omega_vals)
    green = np.zeros((k_vals.size, omega_vals.size, 2, 2), dtype=complex)
    for i_omega, omega in enumerate(omega_vals):
        green_vals = green_order_0(k_vals, omega, beta, K, u)
        if order >= 1:
            green_vals += 1j*u_minus * green_order_1(k_vals, omega, beta, K, u)
        # if order >= 2:
        #     green_vals += (1j*u_minus)**2/2 * green_order_2(k_vals, omega, beta, K, u)
        green[:, i_omega] = green_vals
    return green