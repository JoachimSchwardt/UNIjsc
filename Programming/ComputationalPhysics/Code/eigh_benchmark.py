#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark of eigh_tridiagonal from scipy
"""

from time import perf_counter
import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

def benchmark_eigh(nvals):
    times = np.zeros(nvals.size)
    for i in range(nvals.size):
        diag = np.full(nvals[i], 2)
        off_diag = np.full(nvals[i]-1, -1)
        t_start = perf_counter()
        _ = eigh_tridiagonal(diag, off_diag, eigvals_only=True, 
                             lapack_driver='sterf')
        t_end = perf_counter()
        times[i] = t_end - t_start
    return times
        
def main():
    print(__doc__)
    nvals = np.unique(np.logspace(3.3, 3.6, 15, base=10.0, dtype=int))
    
    times = benchmark_eigh(nvals)
    
    slope, offset = np.polyfit(np.log(nvals), np.log(times), deg=1)
    
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("$N$")
    ax.set_ylabel("Time / s")
    ax.plot(nvals, times, ls='', marker='o', ms=2, mew=0, c='b')
    ax.plot(nvals, np.exp(offset) * nvals**slope, lw=1, c='orange', 
            label=fr"Fit: $T={np.exp(offset):.2e}\cdot N^{{{slope:.2f}}}$")
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
        