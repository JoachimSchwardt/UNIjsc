#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the smallest number that is divisible by all numbers <= 'n'
"""

import numpy as np
from largest_prime_factor import factorize


def cumulate_factorize(n):    
    all_primes = {}
    for val in range(1, n+1):
        primes = factorize(val)
        
        primes_dict = {}
        
        # we want to extend the all_primes dictionary by new "excess" primes
        for prime in primes:
            try:
                primes_dict[prime] += 1
            except KeyError:
                primes_dict[prime] = 1
                
        for prime in primes_dict.keys():
            try:
                if all_primes[prime] < primes_dict[prime]:
                    all_primes[prime] = primes_dict[prime]
            except KeyError:
                all_primes[prime] = primes_dict[prime]
                
    return all_primes


def main():
    print(__doc__)
    n = 20
    primes = cumulate_factorize(n)
    val = np.prod([prime**fac for prime, fac in primes.items()])
    print(f"Solution for {n = } is {val}.")
    return 0


if __name__ == "__main__":
    main()
