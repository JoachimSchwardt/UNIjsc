#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots and calculations for the neutrino problem from sheet 06.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=1, figsize=special.set_figsize())

hc = 1.97 * 1e-7            # eV * m

GF = (246 * 1e9)**(-2)      # eV**2
n0 = 6 * 1e31               # m**(-3)
Rs = 7e8                    # m
r0 = 0.1 * Rs               # m

theta_deg = 34
theta = theta_deg * np.pi/180
delta_m2 = 7.5 * 1e-5        # eV**(-2)

def get_xy(x, y, r):
    assert(r <= 1.0 and r >= 0.0)   
    xn = np.sort(x)
    yn = np.sort(y)
    xmax, xmin = xn[-1], xn[0]
    xcrit = xmin + r * (xmax - xmin)
    i = np.searchsorted(xn, xcrit)
    return xcrit, yn[i]


def plot_maser():
    """
    See https://stackoverflow.com/questions/30081846/set-matplotlib-
    rectangle-edge-to-outside-of-specified-width for Rectangle code.
    """    
    A = 7.9e-6     # eV
    
    def ew(eps, A=A):
        return np.sqrt(eps**2 + A**2)
    
    N = 300
    eps = np.linspace(-5*A, 5*A, N)
    ew_val = ew(eps, A)
    
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\frac{\vec{d}\cdot\vec{\epsilon}}{\mathrm{eV}}$")
    ax.set_ylabel(r"$\frac{E-E_0}{\mathrm{eV}}$")
    ax.set_title(f"$\\mathrm{{NH}}_3$-Maser with $A={A*1e6:.1f}\,\\mu$eV")
    
    ax.axhline(0.0, c='k', lw=0.5)
    ax.axvline(0.0, c='k', lw=0.5)
    
    ax.plot(eps, eps, ls='--', c='k', lw=0.8)
    ax.plot(eps, -eps, ls='--', c='k', lw=0.8)
    
    ax.plot(eps, ew_val, c='b')
    ax.plot(eps, -ew_val, c='b')
    
    offset = 0.14 * np.max(ew_val)
    xpos, ypos = get_xy(eps, eps, 0.01)
    ax.text(xpos, -ypos - offset, r"$| 1 \rangle$", va='center', ha='center')
    ax.text(xpos, ypos + offset, r"$| 2 \rangle$", va='center', ha='center')
    
    xpos, ypos = get_xy(eps, eps, 0.99)
    ax.text(xpos, -ypos + offset, r"$| 2 \rangle$", va='center', ha='center')
    ax.text(xpos, ypos - offset, r"$| 1 \rangle$", va='center', ha='center')
    
    props = dict(fc='w', ec='k', lw=0.5)
    
    ax.text(eps[0], 0.1 * offset, r"$\theta = 0^\circ$", bbox=props,
            va='center')
    ax.text(0.0, eps[N//4], r"$\theta = 45^\circ$", bbox=props, ha='center')
    ax.text(eps[-1], 0.1 * offset, r"$\theta = 90^\circ$", bbox=props, 
            ha='right', va='center')
    
    special.polish(fig, ax)


def delta_mn2(r, E=9e6, r0=r0, n0=n0):
    return 2*E * np.sqrt(2)*GF * ne(r, r0, n0) * hc**3

def delta_m2m(r, E=9e6, r0=r0, n0=n0):
    return np.sqrt((delta_m2 * np.cos(2*theta) - delta_mn2(r, E, r0, n0))**2
                   + (delta_m2 * np.sin(2*theta))**2)


def ne(r, r0=r0, n0=n0):
    return n0 * np.exp(-r/r0)

def main():
    N = 300
    r = np.linspace(0.0, Rs, N)
    # r = np.logspace(-8, np.log10(Rs), N)
    n = ne(r)
    E = 9*1e6
    
    ratio = 2*E * np.sqrt(2) * GF*n0*hc**3 / delta_m2
    print(r"The mass contribution from the scattering amplitude $\Delta m_N^2$"
          + f" is {ratio:.2f} times larger than the vacuum mass splitting.")
    
    
    mmu2 = np.cos(theta)**2 * np.full(N, delta_m2)
    A = delta_mn2(r, E)
    me2 = A + np.sin(theta)**2 * delta_m2
    delta_m2m_val = delta_m2m(r, E)
    
    m12 = (delta_m2 + A - delta_m2m_val) / 2
    m22 = (delta_m2 + A + delta_m2m_val) / 2
    
    # m12pure = np.cos(theta)**2 * A
    # m22pure = np.sin(theta)**2 * A + delta_m2
    
    x = n / n0 * E * 1e-6
    
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\frac{n_e(r)}{n_0}\frac{E}{\mathrm{MeV}}$")
    # ax.set_xlabel(r"$\frac{n_e(r)}{n_0}$")
    ax.set_ylabel(r"$\frac{m^2 - m_{\nu_1}^2}{\mathrm{eV}^2}$")
    ax.set_title(f"$E={E*1e-6:.1f}$\,MeV, "
                 + f"$\\theta_{{12}} = {theta_deg}^\\circ$ and "
                 + f"$\\Delta m_{{12}}^2 = {delta_m2*1e5:.1f}$"
                 + r"$\times 10^{-5}\,\mathrm{eV}^2$")
    
    dy = np.max(m22) * 1e3 * 1.1       # 1e-3 * eV^2
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0.0, dy)
    
    ax.plot(x, mmu2 * 1e3, c='k', ls='--', label=r"$m_{\nu_{\mu}}^2$")
    ax.plot(x, me2 * 1e3, c='k', ls='--', label=r"$m_{\nu_{e}}^2$")
    
    ax.plot(x, m12 * 1e3, c='b', label=r"$\tilde{m}_{\nu_{1}}^2$")
    ax.plot(x, m22 * 1e3, c='b', label=r"$\tilde{m}_{\nu_{2}}^2$")
    
    # ax.plot(x, m12pure * 1e3, c='g', label=r"$m_{\nu_{1}}^2$")
    # ax.plot(x, m22pure * 1e3, c='g', label=r"$m_{\nu_{2}}^2$")
    
    
    offset = 0.05 * dy
    xpos, ypos = get_xy(x, mmu2, 0.9)
    ax.text(xpos, ypos * 1e3 + offset, r"$m_{\nu_{\mu}}^2$")
    xpos, ypos = get_xy(x, me2, 0.75)
    ax.text(xpos, ypos * 1e3 - offset, r"$m_{\nu_{e}}^2$")
    
    xpos, ypos = get_xy(x, m12, 0.5)
    ax.text(xpos, ypos * 1e3 - offset, r"$\tilde{m}_{\nu_{1}}^2$", c='b')
    xpos, ypos = get_xy(x, m22, 0.5)
    ax.text(xpos, ypos * 1e3 + offset, r"$\tilde{m}_{\nu_{2}}^2$", c='b')
    
    ax.text(0.0, 1.0, r"$\times 10^{-3}$", 
            ha='left', va='bottom', transform=ax.transAxes)
    
    # ax.legend()
    special.polish(fig, ax, xva='bottom')
    
    return 0

if __name__ == "__main__":
    print(__doc__)
    main()
    # plot_maser()