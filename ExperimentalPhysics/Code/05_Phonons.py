# -*- coding: utf-8 -*-
"""
Phonon dispersion relations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=True)

def omega_sqr(k, m=1.0, M=2.0, C=5.0, a=0.2):
    """[a] == nm"""
    mu = m*M / (m+M)
    root = np.sqrt(1 - 4*mu**2 / (m*M) * np.sin(k*a/2)**2)
    return C/mu * (1 + root), C/mu * (1 - root)
    
def omega(k, m=1.0, M=2.0, C=5.0, a=0.2):
    omega_p, omega_m = omega_sqr(k, m, M, C, a)
    return np.sqrt(omega_p), np.sqrt(omega_m)


def main():
    m = 1.0
    M = 2*m
    mu = m*M / (m+M)
    a = 0.2
    C = 5.0
    
    omega0 = np.sqrt(C / mu)
    vs = a*mu*omega0 / np.sqrt(2*m*M)    # speed of sound
    
    pival = np.linspace(-np.pi, np.pi, 300)
    kval = pival / a
    omega_p, omega_m = omega(kval, m, M, C, a) / omega0
    
    colors = special.Colors()
    fig, ax = plt.subplots()
    ax.set_xlim(pival[0], pival[-1])
    ax.set_ylim(0.0, 1.05 * np.max(omega_p))
    ax.set_xlabel(r"$k\ /\ a$")
    ax.set_ylabel(r"$\omega(k)\ /\ \omega_0$")
    ax.set_title(f"$m\,/\,M = 1 / {int(M/m)}$")
    
    ax.plot(pival, omega_m, c=colors.get_color(), label="acoustic")
    ax.plot(pival, omega_p, c=colors.get_color(), label="optical")
    ax.legend()
    special.format_ticklabels(ax)
    special.polish(fig, ax)
    
    return 0

if __name__ == "__main__":
    print(__doc__)
    main()