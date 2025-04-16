#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the largest prime factor of a given number
"""

def factorize(n):
    prime = 2
    primes = []
    while n % prime == 0:
        primes.append(prime)
        n //= prime
    
    prime = 3
    while prime**2 <= n:
        while n % prime == 0:
            primes.append(prime)
            n //= prime
        prime += 2
    
    if n > 1:
        primes.append(n)    # remainding prime factor
        
    return primes


def main():
    print(__doc__)
    
    n = 600851475143
    primes = factorize(n)
    print(f"Solution for {n = } is {primes}.")
    return 0


if __name__ == "__main__":
    main()
