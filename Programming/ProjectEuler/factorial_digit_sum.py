#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 01:11:55 2022

@author: joachim
"""

import numpy as np


def main():
    print(__doc__)
    
    n = np.math.factorial(100)
    digit_sum = np.sum([int(val) for val in str(n)])
    print(f"Sum of digits in {n = } is {digit_sum = }")
    
    return 0

if __name__ == "__main__":
    main()
