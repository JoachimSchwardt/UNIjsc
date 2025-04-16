#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:05:34 2023

@author: ag_budich1
"""

import numpy as np
import sympy as sy
from scipy.integrate import quad

def phi_sympy(x, amplitude, mu, sigma):
    return amplitude * sy.exp(-(x - mu)**2 / (2*sigma**2)) * (x**3 + mu)

def get_phi(amplitude_val=1.0, mu_val=1.0, sigma_val=1.0, derivative=0):
    x, amplitude, mu, sigma = sy.symbols("x, amplitude, mu, sigma")
    f_sympy = phi_sympy(x, amplitude, mu, sigma)
    for _ in range(derivative):
        f_sympy = f_sympy.diff(x)
        
    f_numpy = sy.lambdify(x, 
                          f_sympy.evalf(subs={amplitude : amplitude_val,
                                              mu : mu_val,
                                              sigma : sigma_val,
                                              }), 
                          "numpy")
    return f_numpy
    
# def phi_prime(x, amplitude=1.0, mu=1.0, sigma=1.0):
#     return -(x - mu) / sigma**2 * phi(x, amplitude, mu, sigma)

def test_cos_nabla_phi_integral():
    m = 2.0
    lower = -5.0#-np.inf
    upper = 5.0#np.inf
    phi = get_phi()
    phi_prime = get_phi(derivative=1)
    original, original_err = quad(lambda x: np.cos(m*phi(x)) * phi_prime(x), lower, upper)
    print(f"Original integral: {original:.5f} +- {original_err:.3e}")
    

def lie_hadamard_derivative_test():
    x = sy.Symbol('x')
    f = sy.symbols('f', cls=sy.Function)
    n = 3
    lie_h_n = sy.exp(-f(x)) * sy.diff(sy.exp(f(x)), x, n)
    print(lie_h_n)
    
# test_cos_nabla_phi_integral()
lie_hadamard_derivative_test()