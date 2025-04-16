#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical confirmation of the analytic solution to the scattering problem
see 'Scattering.pdf' for the derivation.

main2 simulates the scattering of x-ray photons off water, assuming
gaussian distributions of the peaks and adding random noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup()


def root(x, theta=0.0, E=200.0, m=0.5, M=1000.0):
    """ Units of mass and energy in MeV """
    p = np.sqrt(E**2 - m**2) * np.cos(theta)
    a = (E+M)**2 - p**2
    b = (E+M) * (E*M + m**2)
    c = (E*M + m**2)**2 + m**2*p**2
    return a * x**2  - 2*b * x + c
    # return  (M*(E-x) - E*x + m**2)**2 - (x**2 - m**2) * p**2

def root_analytic(theta=0.0, E=200.0, m=0.5, M=1000.0):
    p = np.sqrt(E**2 - m**2) * np.cos(theta)
    a = (E+M)**2 - p**2
    b = (E+M) * (E*M + m**2)
    r = p * np.sqrt((E**2 - m**2) * (M**2 - m**2) + m**2*p**2)
    return np.array([b + r, b - r]) / a
    
    # p = np.sqrt(E**2 - m**2) * np.cos(theta)
    # a = (E+M)**2 - p**2
    # b = (E+M) * (E*M + m**2)
    # c = (E*M + m**2)**2 + m**2*p**2
    # r = np.sqrt(b**2 - a*c)
    # return np.array([b - r, b + r]) / a
    
    # return [E / (1 + E/M * (1 - np.cos(theta)))] * 2
    

def main():
    E = 20.0
    m = 0.5
    M = 1e3
    theta = np.pi / 6
    
    N = 500
    r1, r2 = root_analytic(theta, E, m, M)
    if r1 > r2: 
        r1, r2 = r2, r1
    delta = r2 - r1
    
    x = np.linspace(r1 - 0.05 * delta, r2 + 0.05 * delta, N)
    val = root(x, theta, E, m, M)
    
    
    fig, ax = plt.subplots()
    colors = special.Colors()
    ax.plot(x, val, c=colors.get_color())
    ax.axvline(r1, c='k')
    ax.axvline(r2, c='k')
    ax.axhline(0.0, c='k')
    special.polish(fig, ax)
    
    return 0

def main2():
    
    return 0

if __name__ == "__main__":
    print(__doc__)
    main2()