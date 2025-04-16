#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dispersionsrelationen und Zustandsdichten berechnen und visualisieren
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=True)


def dispersion_chain(q, M=1, k1=1, k2=10, a=1):
    root = np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(q*a)) / M
    val = (k1 + k2) / M
    return np.sqrt([val - root, val + root])


def dos_chain(omega, M, k1, k2):
    """Computes the density of states for a given phonon branch."""
    x = np.abs(k1 + k2 - M*omega**2)
    den = np.sqrt(np.abs((2*k1*k2)**2 - (x**2 - k1**2 - k2**2)**2))
    return 2/np.pi * M*omega * x / den


def plot_dispersion(q, omega, labels=None, title=None):
    q = np.array(q)
    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)
        
    omega = np.array(omega)
    if omega.ndim == 1:
        omega = np.expand_dims(omega, axis=0)
        
    if isinstance(labels, type(None)):
        labels = np.array([None] * q.shape[0])
        
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$q / a$")
    ax.set_ylabel(r"$\omega(q)$")
    ax.set_xlim(np.min(q[:, 0]), np.max(q[:, -1]))
    ax.set_title(title)
    for i in range(q.shape[0]):
        ax.plot(q[i], omega[i], label=labels[i])
    special.polish(fig, ax)
    return fig, ax

def plot_dos(omega, dos, labels=None, title=None, dos_max=10):
    if isinstance(labels, type(None)):
        labels = np.array([None] * omega.shape[0])
        
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\omega / a$")
    ax.set_ylabel(r"$n(\omega)$")
    ax.set_xlim(0.0, np.max(omega) * 1.05)
    ax.set_ylim(0.0, dos_max)
    ax.set_title(title)
    for i in range(omega.shape[0]):
        ax.plot(omega[i], dos[i], label=labels[i])
    special.polish(fig, ax)
    return fig, ax
    

def main():
    print(__doc__)
    M = 1
    a = 1
    k1 = 1
    k2 = 4*k1
    
    N = 300
    q = np.linspace(-np.pi / a, np.pi / a, N)
    
    omega = dispersion_chain(q, M, k1, k2, a)
    dos = dos_chain(omega, M, k1, k2)
    
    title = fr"$M={M}, \kappa={k1}, K={k2}, a={a}$"
    # fig, ax = plot_dispersion([q, q], omega, title=title)
    
    fig, ax = plot_dos(omega, dos, title=title)
    
    return 0

if __name__ == "__main__":
    main()
    