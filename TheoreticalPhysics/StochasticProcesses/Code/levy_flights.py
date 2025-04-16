# -*- coding: utf-8 -*-
"""
Untersuchung Levy Distributionen und Levy fligths
"""

import  scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import special
special.setup(UseTex=True, xmajorpad=15.0)


def X(phi, y , alpha): 
    """
    The magic function f(phi,y) responsable for generating levy-stable
    distributions
    """
    return ( (np.sin(alpha * phi) / np.cos(phi)**(1/alpha)) 
            * (np.cos((1-alpha) * phi) / y)**((1-alpha) / alpha) )


def show_dist(alpha, N):

    y = stats.expon.rvs(size = N)
    phi = (stats.uniform.rvs(size = N) - 0.5) * np.pi
    
    fig, ax = plt.subplots()
    ax.set_title(f"$\\rho_X(x)$-distribution for $N$ = {N}") 
    ax.set_xlabel(r"$|x|$")
    ax.set_ylabel("count")
    colors = special.Colors()
    
    for a in alpha: 
        x = np.abs(X(phi, y, a))

        hist, bins = np.histogram(x, bins=100)

        # histogram on log scale. 
        # Use non-equal bin sizes, such that they look equal on log scale.
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    
        ax.hist(x, bins = logbins, label = f"$\\alpha = {a}$", 
                color=colors.get_color(), histtype='step', lw=2.5)
        
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend()
    special.polish(fig, ax) 
    return fig, ax


def levy_flight(steps, alpha, nrows=1, ncols=1): 
    alpha = np.array(alpha)
    fig, ax = plt.subplots(nrows, ncols)
    plt.suptitle(f'Levy-flights with $N$ = {steps} steps')
    colors = special.Colors()
    ax = ax.flatten()
    
    for i in range(alpha.shape[0]):
        ax[i].set_xlabel(r"$x$")
        ax[i].set_ylabel(r"$y$")
        y1 = stats.expon.rvs(size = steps)
        phi1 = (stats.uniform.rvs(size = steps) - 0.5) * np.pi
        
        y2 = stats.expon.rvs(size = steps)
        phi2 = (stats.uniform.rvs(size = steps) - 0.5) * np.pi    
        
        posx = X(phi1, y1, alpha[i]).cumsum()
        posy = X(phi2, y2, alpha[i]).cumsum()
        
        ax[i].plot(posx, posy, c=colors.get_color(), 
                   label=f"$\\alpha = {alpha[i]}$")
        
        ax[i].legend()
    special.polish(fig, ax, SetCaptions=True, xva='center', yha='center')
    return fig, ax



def main():
    N = 10000
    # alpha = [0.2, 0.5, 0.75, 1,  1.5, 2]
    # fig, ax = show_dist(alpha, N)
    # plt.savefig("Levy_stable_distribution.png")
    
    # alpha = [0.1, 0.5, 0.75, 1,  1.5, 1.75, 1.85, 2]
    alpha = [2.0, 1.75, 1.5, 1.25]
    fig, ax = levy_flight(N, alpha, nrows=2, ncols=2)
    plt.subplots_adjust(top=0.93, right=0.99, left=0.06, bottom=0.1)
    # plt.savefig("Levy_flights_large_alpha.png")
    
    # alpha = [0.8, 0.9, 1.0]
    # fig, ax = levy_flight(N, alpha)
    # plt.savefig("Levy_stable_distribution.png")
    return 0
    
  
    

if __name__ == '__main__':
    print(__doc__)
    main()