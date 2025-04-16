#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:26:12 2023

@author: joachim
"""

import numpy as np
import mpmath as mp
from scipy import special, integrate


def c_quad(func, xmin, xmax, args=(), **kwargs):
    def f_real(x, *args):
        return np.real(func(x, *args))
    def f_imag(x, *args):
        return np.imag(func(x, *args))
    r_int = integrate.quad(f_real, xmin, xmax, args, **kwargs)
    i_int = integrate.quad(f_imag, xmin, xmax, args, **kwargs)
    return (r_int[0] + 1j*i_int[0], np.abs(r_int[1] + 1j*i_int[1]))


def g_func(omega, k, beta=2.6, K=0.2, digits=8):
    """
    integrate E^(I omega_n^F tau - I k x) csc(pi/beta (tau + I x)) (sin^2 + sinh^2)**(-K)
    """
    mp.mp.dps = digits
    n_val = (beta*omega / (np.pi*1j) - 1) / 2
    z_val = beta / (4*np.pi*1j) * (omega - k)
    z_bar_val = beta / (4*np.pi*1j) * (omega + k)
    prefactor = 4**K * beta**2/(2*np.pi*1j) * special.gamma(K + n_val + 1)
    part1 = complex(mp.hyper((K+1, K+1 + n_val, (K+1)/2 + z_val),
                             (n_val + 2, (K+3)/2 + z_val), 1))
    factor1 = special.gamma(K) * special.gamma(n_val + 2) * ((K+1)/2 + z_val)
    part2 = complex(mp.hyper((K, K+1 + n_val, K/2 + z_bar_val),
                             (n_val + 1, K/2 + 1 + z_bar_val), 1))
    factor2 = special.gamma(K+1) * special.gamma(n_val + 1) * (K/2 + z_bar_val)
    return prefactor * (part1 / factor1 - part2 / factor2)


def i_1(z, K):
    """integral csch(z + x - I 0^+)^((K+1) / 2) csch(x - I 0^+)^((K-1) / 2) from x=-inf to inf"""
    return ((1 + np.exp(1j*np.pi*K)) * 8**((K-1) / 2) * special.beta(K/2, 1-K)
            / np.cosh(z)**((K-1) / 2) / (1 + (2**((K-1) / 2) - 1) / np.cosh(z)**((K+7) / 10)))


def j_0(z, K):
    """integral e^(I zx) csch(x)^K from x=0 to inf"""
    return (2**(K-1) * special.gamma(1-K) * special.gamma(K/2 - 1j * z/2)
            / special.gamma(1-K/2 - 1j * z/2))


def h_1(z, A, B, C, r, limit=200):
    """integral e^(I zx) csch(x)^A sech(x)^B / (1 + r * sech(x)^C) from x=0 to inf"""
    integrand = lambda x: (x**((A+B)/2 - 1j*z/2 - 1) * (1-x)**(-A) * (1+x)**(C-B)
                           / (r*x**(C/2) + 2**(-C) * (1+x)**C))
    integral = (c_quad(integrand, 0, 0.1, limit=limit)[0]
                + c_quad(integrand, 0.1, 0.9)[0]
                + c_quad(integrand, 0.9, 1, limit=limit)[0])
    return 2**(A+B-C-1) * integral


def j_1(z, K):
    """cos(pi K/2) B(K/2, 1-K) * H_1(z, (1/K - K) / 2, (K-1)/2, (K+7)/10, 2^((K-1)/2) - 1)"""
    return (np.cos(np.pi * K/2) * special.beta(K/2, 1-K)
            * h_1(z, (1/K - K) / 2, (K-1) / 2, (K+7) / 10, 2**((K-1)/2) - 1))


def correlator_phi_ell_main(x=0.43, tau=0.78, ell=1, u=1.9, beta=2.6, alpha=1e-3):
    """int 0 to inf 2/k * exp(-alpha*k) * n_B(beta*u*k) * (cos(kx + ell*1j*k*u*tau) - 1)
        = log(Gamma(1 + (alpha + 1j*x - ell*u*tau) / (beta*u)) * ... / Gamma(...)**2)
    """
    beta_u = beta * u
    z_ell = u*tau - 1j*ell*x
    result = np.log(special.gamma(1 + (alpha + z_ell) / beta_u)
                     * special.gamma(1 + (alpha - z_ell) / beta_u)
                     / special.gamma(1 + alpha / beta_u)**2)
    return result

def correlator_phi_ell_reg(x=0.43, tau=0.78, ell=1, u=1.9, alpha=1e-3):
    """int 0 to inf 1/k * exp(-alpha*k) * (exp(-k*u*tau + 1j*ell*k*x) - 1)"""
    return -np.log(1 + (u*tau - 1j*ell*x) / alpha)

def correlator_phi_ell(x=0.43, tau=0.78, ell=1, u=1.9, beta=2.6, alpha=1e-3):
    """<phi_ell(r) phi_ell(0) - phi_ell(0)**2>"""
    beta_u = beta * u
    result = np.log(np.pi*alpha/beta_u / np.sin(np.pi/beta_u * (u*tau - 1j*ell*x)))
    return result