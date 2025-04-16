#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Number of ways to navigate an NxN-grid from top left to bottom right.
Note that we limit the movement to 'down' and 'right'.
Thus we need 'N' steps of 'down' and 'right' respectively, 
resulting in permutations of the list [D, D, R, D, ..., R, R, D]

There are exactly 2N entries, and N of these are 'right'.
The number of ways is given by the binomial coefficient (2N, N).
"""

import numpy as np

def main():
    print(__doc__)
    
    n = 20
    number_of_paths = np.math.comb(2*n, n)
    print(f"Solution for {n = } is {number_of_paths = }")
    return 0


if __name__ == "__main__":
    main()
