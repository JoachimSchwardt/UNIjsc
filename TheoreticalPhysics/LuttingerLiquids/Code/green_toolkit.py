#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:27:53 2023

@author: joachim
"""

import numpy as np



sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
U = 1/np.sqrt(2) * np.array([[1, 1j], [1j, 1]])   # basis transform AB <--> LR


def xt_to_xi_ell(x, t, u=1.0, ell=1):
    """Convert 'x' and 't' to relativistic coordinate r'xi_ell' with velocity 'u'."""
    return u*t + ell * x


def xt_to_xi(x, t, u=1.0):
    """Convert 'x' and 't' to relativistic coordinates r'xi_\pm' with velocity 'u'."""
    return xt_to_xi_ell(x, t, u, 1), xt_to_xi_ell(x, t, u, -1)


def xi_to_xt(xi_p, xi_m, u=1.0):
    """Convert relativistic coordinates r'xi_\pm' to 'x' and 't' with velocity 'u'."""
    x = (xi_p - xi_m) / 2
    t = (xi_p + xi_m) / (2*u)
    return x, t    


def bose_einstein(value):
    """Bose-Einstein distribution function"""
    if value < -50:
        return -1
    elif value > 300:
        return 0.0
    return 1 / (np.exp(value) - 1)


def hamilton_from_green(omega_vals, green):
    """Convert Green's function to Hamiltonian for given frequencies"""
    hamilton = np.array([[omega * np.eye(2) - np.linalg.inv(green[i_k, i_omega])
                          for i_omega, omega in enumerate(omega_vals)]
                         for i_k in range(green.shape[0])])
    return hamilton


def green_from_hamilton(omega_vals, hamilton):
    """Convert Hamiltonian to Green's function for given frequencies"""
    green = np.zeros_like(hamilton)
    for i_omega, omega in enumerate(omega_vals):
        for i_k in range(hamilton.shape[0]):
            try:
                arr = np.linalg.inv(omega * np.eye(2) - hamilton[i_k, i_omega])
            except np.linalg.LinAlgError:
                arr = np.linalg.inv(omega * np.eye(2) - hamilton[i_k, i_omega] + 1e-15)
            green[i_k, i_omega] = arr
    return green


def hamilton_2x2_decomp(hamilton):
    shape = hamilton.shape
    if shape[-2] != 2 or shape[-1] != 2:
        msg = f"Shape should be 2 by 2 in the last two axes but was {shape}"
        raise IndexError(msg)

    d_0 = np.trace(sigma_0 @ hamilton, axis1=-1, axis2=-2)
    d_x = np.trace(sigma_x @ hamilton, axis1=-1, axis2=-2)
    d_y = np.trace(sigma_y @ hamilton, axis1=-1, axis2=-2)
    d_z = np.trace(sigma_z @ hamilton, axis1=-1, axis2=-2)
    return d_0, d_x, d_y, d_z


def hamilton_2x2_ep_eqn(hamilton):
    """Return the equations d_r^2 - d_i^2 and d_r.d_i governing EP's for a given 2x2-Hamilton"""
    _, d_x, d_y, d_z = hamilton_2x2_decomp(hamilton)
    dr2_di2 = d_x.real**2 + d_y.real**2 + d_z.real**2 - d_x.imag**2 - d_y.imag**2 - d_z.imag**2
    dr_di = d_x.real * d_x.imag + d_y.real * d_y.imag + d_z.real * d_z.imag
    return dr2_di2, dr_di


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
    """Eigenvalues for a 2x2 matrix"""
    offset = (arr[0, 0] + arr[1, 1]) / 2
    root = np.sqrt(((arr[0, 0] - arr[1, 1]) / 2)**2 + arr[1, 0] * arr[0, 1])
    return offset - root, offset + root
