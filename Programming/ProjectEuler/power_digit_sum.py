#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sum of all digits of the number 2**N.
"""

import numpy as np


def digit_sum(number):
    return np.sum([int(val) for val in str(number)])

def main():
    print(__doc__)
    
    n = 1000
    dsum = digit_sum(2**n)
    print(f"Solution for {n = } is {dsum = }")
    return 0


if __name__ == "__main__":
    main()
