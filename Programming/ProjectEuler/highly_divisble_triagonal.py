#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find highly divisble triagonal numbers
"""

import numpy as np
from factorize import factorize


def get_triangular_numbers(count):
    return np.cumsum(np.arange(1, count + 1, 1))


def divisors(num):
    """
    https://en.wikipedia.org/wiki/Divisor_function

    Divisors function:
        sigma_0(n) = prod_{j=0}^{r} (a_j + 1)
        where n = prod_{j=0}^{r} p_j^{a_j}
    """
    _primes, counts = factorize(num)
    return np.prod(np.array(counts) + 1, dtype=int)


def main():
    print(__doc__)

    count = 15000
    values = get_triangular_numbers(count)
    div = [divisors(val) for val in values]
    
    for ind, dval in enumerate(div):
        if dval >= 500:
            break
        
    # ind = np.argmax(div)
    max_div = div[ind]
    val = values[ind]

    # print(f"Solution for {count = } is {values = }, {div = }")
    print(f"Solution for {count = } is {max_div = }, {val = }")
    return 0


if __name__ == "__main__":
    main()
