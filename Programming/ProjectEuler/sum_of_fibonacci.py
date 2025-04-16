#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sum of all even fibonacci numbers below 'n'
"""

import numpy as np


def fibo_list(n):
    """Return all fibonacci numbers below 'n'"""
    vals = [1, 1]
    size = 2
    while True:
        newval = vals[size-1] + vals[size-2]
        if newval < n:
            vals.append(newval)
            size += 1
        else:
            break
    return np.array(vals)


def main():
    print(__doc__)
    
    n = 4*10**6
    fibo_vals = fibo_list(n)
    total = np.sum([val for val in fibo_vals if val % 2 == 0])
    print(f"Solution for {n = } is {total}.")
    return 0


if __name__ == "__main__":
    main()
