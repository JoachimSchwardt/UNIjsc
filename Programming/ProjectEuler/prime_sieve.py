#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prime sieve
"""

from numba import njit

@njit
def prime_sieve(n):
    primes = [2]
    val = 3
    while val <= n:
        
        # check if prime
        isprime = True
        for prime in primes:
            if prime**2 <= val and val % prime == 0:
                isprime = False
                break
        
        if isprime:
            primes.append(val)
        
        val += 2
    return primes


def main():
    print(__doc__)
    
    n = 50000
    primes = prime_sieve(n)
    
    print(f"Solution for {n = } is {primes[-1]}.")
    return 0


if __name__ == "__main__":
    main()
