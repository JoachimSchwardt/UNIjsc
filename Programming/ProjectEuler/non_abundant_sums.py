#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A perfect number is a number for which the sum of its proper divisors is 
exactly equal to the number. For example, the sum of the proper divisors of 28 
would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.

A number n is called deficient if the sum of its proper divisors is less than 
n and it is called abundant if this sum exceeds n.

As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest 
number that can be written as the sum of two abundant numbers is 24. 
By mathematical analysis, it can be shown that all integers greater than 28123 
can be written as the sum of two abundant numbers. However, this upper limit 
cannot be reduced any further by analysis even though it is known that the 
greatest number that cannot be expressed as the sum of two abundant numbers 
is less than this limit.

Find the sum of all the positive integers which cannot be written as the sum 
of two abundant numbers.
"""

import numpy as np
from numba import njit
from amicable_numbers import proper_divisors


@njit
def abundant_numbers(max_val):
    """Find all abundant numbers below the given value"""
    abundant = []
    for val in range(2, max_val):
        div_sum = np.sum(proper_divisors(val))
        if div_sum > val:
            abundant.append(val)
            
    return np.array(abundant)


@njit
def search_sorted(val, array):
    """Binary search for a value in a sorted array"""
    lower_i = 0
    upper_i = array.size - 1
    midpoint_i = (upper_i + lower_i) // 2
    if val > array[upper_i] or val < array[lower_i]:
        return -1
    
    while True:
        if val == array[lower_i]:
            return lower_i
        if val == array[upper_i]:
            return upper_i
        if val == array[midpoint_i]:
            return midpoint_i
        
        if val < array[midpoint_i]:
            upper_i = midpoint_i
        else:
            lower_i = midpoint_i
            
        midpoint_i = (upper_i + lower_i) // 2
        if midpoint_i == lower_i:
            return -1


@njit
def check_sum(val, array):
    """Check if the given value can be written as a sum of 
    two values in the given sorted array"""
    for i in range(array.size):
        first = array[i]
        res = val - first
        if res < first:
            return (-1, -1)
        other_i = search_sorted(res, array)
        if other_i >= 0:
            return (i, other_i)
    


def main():
    print(__doc__)
    max_val = 28123
    abundant = abundant_numbers(max_val)
    remaining = []
    for val in range(1, max_val+1):
        indx = check_sum(val, abundant)
        if indx[0] < 0:
            remaining.append(val)
        
    # print(remaining)
    print(f"Sum of non-abundant sums: {np.sum(remaining) = }")
        
    return 0


if __name__ == "__main__":
    main()
