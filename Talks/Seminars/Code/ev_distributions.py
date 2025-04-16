#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extreme Value Distributions and simple Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt


def gumball(x):
    """Gumball EV Distribution"""
    return np.exp(-np.exp(-x))


def frechet(x, alpha=1.0):
    """Gumball EV Distribution"""
    res = np.zeros(x.size, dtype=float)
    indx = (x > 0)
    res[indx] = np.exp(-x[indx]**(-alpha))
    return res


def weibull(x, alpha=1.0):
    """Gumball EV Distribution"""
    res = np.ones(x.size, dtype=float)
    indx = (x < 0)
    res[indx] = np.exp(-(-x[indx])**alpha)
    return res


def dist_to_string(dist, alpha=1.0):
    string = ""
    if dist == frechet:
        string += r"$\mathrm{Fr\'echet}$"
    else:
        string += dist.__name__.capitalize()
        
    string += " : "

    if dist == frechet:
        string += r"$\mathrm{e}^{-x^{-\alpha}}$"
    elif dist == gumball:
        string += r"$\mathrm{e}^{-\mathrm{e}^{-x}}$"
    elif dist == weibull:
        string += r"$\mathrm{e}^{-(-x)^\alpha}$"

    if dist != gumball:
        string += fr" ($\alpha = {alpha}$)"

    return string


def main():
    print(__doc__)
    import mpl_special
    mpl_special.setup(UseTex=False)

    x = np.linspace(-5, 5, 1000)
    alpha = 1.0

    fig, ax = plt.subplots()
    ax.set_xlim(x[0], x[-1])
    # ax.set_ylim(-0.01, 1.01)
    ax.axhline(0.0, c='k', lw=0.7)
    ax.axvline(0.0, c='k', lw=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$G(x)$")

    for dist in [gumball, frechet, weibull]:
        if dist == gumball:
            data = dist(x)
        else:
            data = dist(x, alpha)
        ax.plot(x, data, label=dist_to_string(dist, alpha))

    ax.legend()
    mpl_special.polish(fig, ax)
    plt.subplots_adjust(top=0.996,
                        bottom=0.053,
                        left=0.058,
                        right=0.998,)
    fig.savefig("../EE_Pictures/ev_distributions_simple.png")
    return 0


if __name__ == "__main__":
    main()
