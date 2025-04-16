#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bose Einstein Condensation Plots (Problem Sheet 1 of Superconductivity)
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_special
from scipy.constants import c


def main():
    print(__doc__)

    k = np.linspace(0, 20, 1000)
    omega_p = c / 200e-9       # typical Lond penetration depth is 50...500nm
    omega = np.sqrt(omega_p**2 + (c * k * 1e6)**2)

    fig, ax = plt.subplots()
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0, np.max(omega))
    # ax.axhline(0.0, c='k', lw=0.7)
    # ax.axvline(0.0, c='k', lw=0.7)
    ax.set_xlabel(r"$k / \SI{}{\frac{1}{\micro m}}$")
    ax.set_ylabel(r"$\omega(k) / \SI{}{Hz}$")

    ax.plot(k, omega)
    mpl_special.polish(fig, ax)
    # fig.savefig("../EE_Pictures/ev_distributions_simple.png")
    return 0


if __name__ == "__main__":
    main()
