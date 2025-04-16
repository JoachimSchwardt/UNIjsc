#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:05:27 2022

@author: joachim
"""

import numpy as np
from sympy.ntheory import factorint
from latex_tables import tables
from functools import lru_cache
from numba import njit
from scipy.integrate import quad
import matplotlib.pyplot as plt
import mpl_special

EULER_MASCHERONI = float("0.577 215 664 901 532 860 606 512".replace(' ', ''))


def lie_integrand(x):
    return (1 - 0.833/x) / np.log(x)

def li_integrand(x):
    return 1 / np.log(x)

def lie(x):
    return [quad(lie_integrand, 2, xval+1)[0] for xval in x]

def li(x):
    return [quad(li_integrand, 2, xval+1)[0] for xval in x]


def count_pk(n, p, k):
    """Count the number of times the number p**k fits into n."""
    return n // p**k


@njit
def primes_small(n):
    """Compute the primes below n using a simple erastothenes-sieve
    https://stackoverflow.com/questions/3939660/
    sieve-of-eratosthenes-finding-primes-python"""
    m = n+1
    #numbers = [True for i in range(m)]
    numbers = [True] * m #EDIT: faster
    for i in range(2, int(n**0.5 + 1)):
        if numbers[i]:
            for j in range(i*i, m, i):
                numbers[j] = False
    primes = []
    for i in range(2, m):
        if numbers[i]:
            primes.append(i)
    return primes


LIMIT = 10**5
PRIMES = primes_small(LIMIT)


# @njit
def pi_small(x):
    """Count number of primes below x for relatively small x"""
    return len(primes_small(x))


@lru_cache(maxsize=None)
# @njit
def phi(x, a):
    if a == 0:
        return x
    if a == 1:
        return (x + 1) // 2
    if x < PRIMES[a]:
        if x > 0:
            return 1
        else:
            return 0
        
    t = phi(x, a-1) - phi(x // PRIMES[a-1], a-1)
    return t


@lru_cache(maxsize=None)
# @njit
def pi(x, limit=LIMIT):
    """https://stackoverflow.com/questions/19070911/
    feasible-implementation-of-a-prime-counting-function"""
    if x < limit:
        return pi_small(int(x))

    z = int((x + 0.5)**(0.5))
    a = pi(int(z**(1/2) + 0.5))     # fourth root of x
    b = pi(z)                       # square root of x
    c = pi(int(x**(1/3) + 0.5))     # cube root of x
    sum_ = phi(x, a) + (b + a - 2) * (b - a + 1) / 2
    for i in range(a+1, b+1):
        w = int(x / PRIMES[i-1])
        sum_ -= pi(int(w))
        lim = pi(int(w**(1/2) + 0.5))
        if i <= c:
            for j in range(i, lim+1):
                sum_ -= pi(int(w / PRIMES[j-1])) + 1 - j
    return int(sum_)


def table_count_pk(n=30):
    nfac = np.math.factorial(n)
    nfac_dict = factorint(nfac)
    primes = nfac_dict.keys()
    max_k = int(np.log2(n))

    header = [["$p$"]
              + [fr"$N_{{p,{k}}}$" for k in range(1, max_k+1)]
              + ["$N_p$"]]

    cells = [[p] + [count_pk(n, p, k) for k in range(1, max_k+1)]
             for p in primes]
    sum_cells = [[np.sum(cells[pi][1:])] for pi in range(len(primes))]
    table = tables.Table(header)
    table.add_cells(cells)
    table.add_cells(sum_cells, 'right')
    table.create_table()
    print(table)


def test_pi():
    for n in np.logspace(1, 7, 7):
        print(f"{n = }, pi(n) = {pi(n)}")
        
        
def pi_lut(nmin=1, nmax=5, npoints=100):
    filename = f"pi_{nmin}_{nmax}_{npoints}.gz"
    try:
        nvals, pi_vals = np.loadtxt(filename)
    except OSError:
        print(f"Cannot find {filename}, computing values...")
        nvals = np.unique(np.int32(np.logspace(nmin, nmax, npoints)))
        pi_vals = np.array([pi(n) for n in nvals])
        if nmax > 6:
            np.savetxt(filename, np.array([nvals, pi_vals]))
    return nvals, pi_vals


def plot_pi():
    nvals, pi_vals = pi_lut(1, 9, 100)
    li_vals = li(nvals)
    lie_vals = lie(nvals)
    
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    # ax.set_ylabel("$\pi(n)$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(nvals[0], nvals[-1])
    # ax.set_ylim(0, pi_vals[-1])
    
    ax.plot(nvals, pi_vals, ls='', c='k', marker='o', label='$\pi(n)$')
    ax.plot(nvals, li_vals, label='$\mathrm{Li}(n)$')
    ax.plot(nvals, lie_vals, label='$\mathrm{Li}_\mathrm{corr.}(n)$')
    ax.legend()
    mpl_special.polish(fig, ax)
    return fig, ax
       

def plot_pi_err():
    nvals, pi_vals = pi_lut(1, 9, 100)
    li_vals = li(nvals)
    lie_vals = lie(nvals)
    
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    ax.set_ylabel("$\Delta\pi(n)$")
    ax.set_xscale('log')
    # ax.set_xlim(nvals[0], nvals[-1])
    # ax.set_ylim(0, pi_vals[-1])
    
    ax.plot(nvals, li_vals - pi_vals, label='$\mathrm{Li}(n)$')
    ax.plot(nvals, lie_vals - pi_vals, label='$\mathrm{Li}_\mathrm{corr.}(n)$')
    ax.legend()
    mpl_special.polish(fig, ax)
    return fig, ax
       

def errf(x):
    p = np.array(PRIMES)
    p = p[p <= x]
    k = np.outer(np.arange(1, 30), np.ones(p.size))
    vals = (x / p**k) % 1.0
    return np.sum(vals) / x


def errf2(x, p, max_iter=200):
    vals = (x / p**np.arange(1, max_iter, dtype=float)) % 1.0
    return np.sum(vals)


def errf2v(x, p, max_iter=10):
    k = np.outer(np.arange(1, max_iter), np.ones(p.size))
    vals = (x / p**k) % 1.0
    return np.sum(vals, axis=0)


def test_sum_fractional_weighted():
    x = 10**8
    p = primes_small(x)
    p2 = p[p > np.sqrt(x)]
    
    # ### this is NOT just 0.5 * sum(log(p)) !!
    # np.sum( np.log(p) * ((x / p) % 1.0)) 
    
    x3 = 8005000
    p3 = p[p < x3]
    np.sum( np.log(p3) * ((x3 / p3) % 1.0)) / x3    # approaches 0.4226... (??)
    
    # first fractional term, below sqrt(x), much much smaller
    # x3 = 10**8
    # p3 = p[p < x3]
    # p3 = p3[p3 < np.sqrt(x3)]
    # np.sum( np.log(p3) * ((x3 / p3**6) % 1.0)) / x3
    
    fig, ax = plt.subplots()
    step = 100
    ax.scatter(p2[::step], (x / p2[::step]) % 1.0)
        

def test_log_prime_approx(x=10**8):
    p = primes_small(x)
    p2 = p[p > np.sqrt(x)]
    p3 = p[p <= np.sqrt(x)]
    
    # log1 = np.sum(np.log(p2) * np.floor(x / p2))
    log11 = np.sum(np.log(p2) * x / p2)
    log12 = -np.sum(np.log(p2) * ((x / p2) % 1.0))
    log2 = np.sum(np.log(p3) * x / (p3 - 1))
    log3 = -np.sum(np.log(p3) * errf2v(x, p3, 50))
    log = x * np.log(x) - x
    



def main():
    print(__doc__)
    # test_pi()
    plot_pi()
    plot_pi_err()
    return 0

if __name__ == "__main__":
    main()
