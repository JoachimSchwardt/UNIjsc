#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 01:11:55 2022

@author: joachim
"""

import numpy as np
# from sympy.ntheory import factorint
from numba import njit


@njit
def proper_divisors(num):

    # factors = factorint(num)
    # primes = factors.keys()
    # counts = factors.values()

    # divs = [1]
    ### slow version...
    # divs = [i for i in range(1, num+1) if num % i == 0]
    divs = [1]
    for i in range(2, (num // 2) + 1):
        if num % i == 0:
            divs.append(i)
    return np.array(divs)


# @njit
def divisor_sums(max_val):
    """Compute all proper divisor sums for inputs below a given value"""
    div_sums = np.zeros(max_val, dtype=int)
    for i in range(max_val):
        new_div_sum = np.sum(proper_divisors(i))
        div_sums[i] = new_div_sum

    return div_sums


def amicable_numbers(max_val):
    """Compute all amicable numbers below a given value"""
    div_sums = divisor_sums(max_val)
    amicable = []
    for i in range(max_val):
        div_sum = div_sums[i]
        if div_sum < max_val:
            if div_sums[div_sum] == i and div_sum != i:
                amicable.append(i)

    return amicable


def main():
    print(__doc__)
    max_val = 10000
    
    amicable = amicable_numbers(max_val)
    
    print(f"Amicable below {max_val = } are :: {amicable}")
    print(f"Their sum is {np.sum(amicable) = }")
    return 0

if __name__ == "__main__":
    main()
