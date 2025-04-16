#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:06:30 2023

@author: joachim
"""

import numpy as np
from numba import njit

def get_hamilton(spins, j_horizontal=1, j_vertical=1j):
    # return get_hamilton_sublattice(spins, j_horizontal, j_vertical, lattice=0, step=1)
    res1 = get_hamilton_sublattice(spins, j_horizontal, j_vertical, lattice=0)
    res2 = get_hamilton_sublattice(spins, j_horizontal, j_vertical, lattice=1)
    return res1 + res2


@njit
def get_hamilton_sublattice(spins, j_horizontal=1, j_vertical=1j, lattice=0, step=2):
    """
    Only works for even number of rows and columns (uses block-evaluation).
    """
    nrows, ncols = spins.shape
    res = 0j
    for row in range(lattice, nrows + lattice, step):
        for col in range(lattice, ncols + lattice, step):
            # res += spins[row, col] * spins[row, (col + 1) % ncols] * j_horizontal
            # res += spins[row, col] * spins[(row + 1) % nrows, col] * j_vertical
            # res += ((spins[row % nrows, col % ncols] 
            #          + spins[(row + 1) % nrows, (col + 1) % ncols])
            #         * (j_horizontal * spins[row % nrows, (col + 1) % ncols]
            #            + j_vertical * spins[(row + 1) % nrows, col % ncols])
            #         )
            res += (spins[row % nrows, col % ncols] 
                    * (j_horizontal * spins[row % nrows, (col + 1) % ncols]
                       + j_vertical * spins[(row + 1) % nrows, col % ncols]))
            res += (spins[(row + 1) % nrows, (col + 1) % ncols] 
                    * (j_vertical * spins[row % nrows, (col + 1) % ncols]
                       + j_horizontal * spins[(row + 1) % nrows, col % ncols]))
            # print(row, col, res, "spins", 
            #       spins[row % nrows, col % ncols], 
            #       spins[(row + 1) % nrows, (col + 1) % ncols],
            #       spins[row % nrows, (col + 1) % ncols],
            #       spins[(row + 1) % nrows, col % ncols])
    return res


def get_spins(nrows, ncols):
    return np.random.choice([1, -1], size=(nrows, ncols))


def main():
    print(__doc__)
    nrows, ncols = 2, 4
    spins = get_spins(nrows, ncols)
    hamilton = get_hamilton(spins)
    print(f"{spins} with energy {hamilton}")
    for lattice in [0, 1]:
        print(get_hamilton_sublattice(spins, lattice=lattice))
    
    return 0


if __name__ == "__main__":
    main()
