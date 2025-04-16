#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the naive gauss eleimination procedure for solving 'Ax = b'.
"""

import numpy as np
from numba import njit


def solve(A, b):
    N = b.shape[0]    
    
    # gauss elimination
    for col in range(N-1):
        if A[col, col] == 0:
            for row in range(col+1, N-1, 1):
                if A[row, col] != 0:
                    A[col, col:], A[row, col:] = A[row, col:], A[row, col:]
                    b[col], b[row] = b[row], b[col]
            
        for row in range(col+1, N, 1):
            A[row, col:] -= A[col, col:] * A[row, col] / A[col, col]
    
    return solve_triangular(A, b)

@njit
def solve_triangular(R, z):
    N = z.shape[0]
    x = np.zeros(N)    
    for i in range(N-1, -1, -1):
        if R[i, i] != 0:
            x[i] = (z[i] - np.sum(x[i+1:] * R[i, i+1:])) / R[i, i]
        else:
            raise RuntimeError("Singular matrix 'R' in 'solve_triangular()'!")
    
    return x

def main():
    N = 4

    A = (np.linspace(0.1, 1, N**2)**2).reshape((N, N))
    A[0, 0] = 50
    b = np.linspace(0, 1, N)
    
    R = np.copy(A)
    for i in range(N):
        for j in range(i):
            R[i,j] = 0
    
    xrt = np.linalg.solve(R, b)
    xr = solve(R, b)  
    
    return 0

if __name__ == "__main__":
    print(__doc__)
    main()