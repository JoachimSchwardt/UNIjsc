#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factorize a given number
"""
from numba import njit

@njit
def factorize(n):
    prime = 2
    primes = []
    counts = []
    
    ctr = 0
    while n % prime == 0:
        n //= prime
        ctr += 1
        
    if ctr > 0:
        primes.append(prime)
        counts.append(ctr)
    
    prime = 3
    while prime**2 <= n:
        
        ctr = 0
        while n % prime == 0:
            n //= prime
            ctr += 1
            
        if ctr > 0:
            primes.append(prime)
            counts.append(ctr)
            
        prime += 2
    
    if n > 1:
        primes.append(n)    # remainding prime factor
        counts.append(1)
        
    return primes, counts


def main():
    print(__doc__)
    
    n = 600851475143
    primes, counts = factorize(n)
    print(f"Solution for {n = } is {primes}, {counts = }.")
    return 0


if __name__ == "__main__":
    main()
