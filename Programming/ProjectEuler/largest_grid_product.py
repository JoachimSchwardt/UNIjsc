#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Largest product along any direction in a given grid
"""

import numpy as np


# def max_adjacent_product(grid, n_adj=4):
#     max_prod = 0
#     max_ind = np.array([0, 0])
#     directions = {'right' : np.array([0, 1]),
#                   'down' : np.array([1, 0]),
#                   'diag' : np.array([1, 1])}
#     ind = max_ind
#     prod = max_prod
#     ctr = 0
#     for direction, di in directions.items():
#         1

#     while ind[0] < grid.shape[0]:
#         while ind[1] < grid.shape[1]:
#             # print(f"{prod = }, {ind = }, {digits[ind] = }, {ctr = }")
#             if prod == 0:
#                 # print("Product was zero, reset counter ... ")
#                 ctr = 1
#                 prod = grid[ind]
#             elif ctr < n_adj:
#                 # print("Nonzero product, less than n_adj elements")
#                 ctr += 1
#                 prod *= grid[ind]
#             else:
#                 # print("n_adj elements, remove first and add last")
#                 prod = (prod // grid[ind - n_adj]) * grid[ind]

#             if prod > max_prod:
#                 max_prod = prod
#                 max_ind = ind - ctr + 1

#             ind[1] += 1
#         ind[0] += 1

#     return max_prod, max_ind


def check_inbounds(grid, ind):
    inbounds = True
    for ctr, ival in enumerate(ind):
        if ival >= grid.shape[ctr] or ival < 0:
            inbounds = False
    return inbounds

def max_adjacent_product(grid, direction, n_adj=4):
    dx, dy = direction
    prod = 0
    max_prod = prod
    max_ind = [None, None]
    for ix in range(grid.shape[0]):
        indx = np.array([ix + dx * ctr for ctr in range(n_adj)])
        for iy in range(grid.shape[1]):
            indy = np.array([iy + dy * ctr for ctr in range(n_adj)])
            if check_inbounds(grid, [indx[-1], indy[-1]]):
                prod = np.prod(grid[indx, indy])
                if prod > max_prod:
                    max_prod = prod
                    max_ind = [ix, iy]
    return max_prod, max_ind



def main():
    print(__doc__)

    grid_str = """08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48"""

    grid = np.array([[int(val) for val in row.split(' ')]
                     for row in grid_str.split('\n')])
    n_adj = 4
    d_list = ['right', 'down', 'diagdown', 'diagup']
    directions = {'right' : np.array([0, 1]),
                  'down' : np.array([1, 0]),
                  'diagdown' : np.array([1, 1]),
                  'diagup' : np.array([-1, 1])}
    data = [max_adjacent_product(grid, directions[dval], n_adj=n_adj)
            for dval in d_list]
    products = [value[0] for value in data]
    indices = [value[1] for value in data]

    i_max = np.argmax(products)
    max_prod = products[i_max]
    max_ind = indices[i_max]

    dx, dy = directions[d_list[i_max]]
    print(products, indices, i_max)
    vals = grid[[max_ind[0] + dx * ctr for ctr in range(n_adj)], 
                [max_ind[1] + dy * ctr for ctr in range(n_adj)]]

    print(f"Solution for {grid = } is {max_prod = }, {vals = }.")
    return 0


if __name__ == "__main__":
    main()
