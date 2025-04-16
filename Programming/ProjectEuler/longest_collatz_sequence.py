#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the longest Collatz sequence with s starting seed below 'n'.
"""

from numba import njit

@njit
def collatz(n):
    ctr = 1
    while n > 1:
        if n % 2 == 1:
            n = 3 * n + 1
            ctr += 1
        n //= 2
        ctr += 1
    return ctr


@njit
def max_sequence(nmax):
    max_ctr = 1
    max_n = 1
    for n in range(1, nmax):
        ctr = collatz(n)
        if ctr > max_ctr:
            max_ctr = ctr
            max_n = n
    return max_n, max_ctr


def main():
    print(__doc__)
    
    n = 1000000
    max_n, max_ctr = max_sequence(n)
    print(f"Solution for {n = } is {max_n = }, {max_ctr = }")
    return 0


if __name__ == "__main__":
    main()
