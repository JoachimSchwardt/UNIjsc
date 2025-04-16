#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Special Pythonagorean triplet (a < b < c, a**2 + b**2 = c**2)

a + b + c = psum
a**2 + b**2 = (psum - a - b)**2,
0 = psum**2 - 2*psum * (a + b) + 2*a*b 
  = psum**2 - 2*psum * a + 2*b * (a - psum)
b = psum * (psum - 2*a) / (2 * (psum - a))

"""

import numpy as np

def search_triplet(psum):
    triplet = [0, 0, 0]
    for a in range(1, psum // 3):
        if (psum * (psum - 2*a)) % (2 * (psum - a)) == 0:
            b = psum * (psum - 2*a) // (2 * (psum - a))
            triplet = [a, b, psum - a - b]
            # print(triplet)
            # assert(a**2 + b**2 == triplet[-1]**2)
    
    return triplet


def main():
    print(__doc__)
    psum = 1000     # a+b+c == psum
    triplet = search_triplet(psum)

    product = np.prod(triplet)
    print(f"Solution for {psum = } is {triplet = }, {product = }.")
    return 0


if __name__ == "__main__":
    main()
