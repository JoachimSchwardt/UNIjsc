#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sum of all primes below a given amount.
"""

import numpy as np
from prime_sieve import prime_sieve


def main():
    print(__doc__)
    
    n = 2000000
    primes = prime_sieve(n)
    
    print(f"Solution for {n = } is {np.sum(primes)}.")
    return 0


if __name__ == "__main__":
    main()
