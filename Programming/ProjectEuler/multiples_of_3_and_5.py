#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect all integer multiplies of a factor 'x' smaller than a given number 'n'
Sum them all up.
"""

def compute(n, multiples):
    total = 0
    # for i, mul in enumerate(multiples):
    #     n_muls = n // mul       # number of multiples below n
    #     if (n_muls * mul == n):
    #         n_muls -= 1     # we do exclude 'n' itself from the sum
        
    #     # sum :: mul + 2*mul + 3*mul + ... + n_muls * mul
    #     total += mul * ((n_muls + 1) * n_muls) // 2
    for nval in range(1, n):
        for mul in multiples:
            if nval % mul == 0:
                total += nval
                break
            
    return total


def main():
    print(__doc__)
    
    n = 1000
    multiples = [3, 5]
    total = compute(n, multiples)
    print(f"Solution for {n = } and {multiples = } is {total}.")
    return 0


if __name__ == "__main__":
    main()
