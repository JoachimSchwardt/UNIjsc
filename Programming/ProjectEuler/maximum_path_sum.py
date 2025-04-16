#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 23:24:31 2022

@author: joachim
"""

import numpy as np


def cumsum_triangle(triangle):
    t_sum = np.copy(triangle)
    for ri, row in enumerate(triangle):
        if ri == 0:
            continue
        
        for ci, val in enumerate(row):
            max_parent = 0
            if ci > 0:
                max_parent = triangle[ri-1][ci-1]
            if ci < ri:
                parent2 = triangle[ri-1][ci]
                if parent2 > max_parent:
                    max_parent = parent2
                
            t_sum[ri][ci] = max_parent + val
    
    return t_sum


def max_path_sum(triangle):
    t_sum = cumsum_triangle(triangle)
    # max_sum = t_sum[0][0]
    
    # for ri in range(len(t_sum) - 1, -1, -1):
    #     ci_max = np.argmax(t_sum[ri])
    #     max_sum += t_sum[ri][ci_max]
    
    # return max_sum
    return np.max(t_sum[-1])



def main():
    print(__doc__)
    
    triangle_str = """75
95 64
17 47 82
18 35 87 10
20 04 82 47 65
19 01 23 75 03 34
88 02 77 73 07 63 67
99 65 04 28 06 16 70 92
41 41 26 56 83 40 80 70 33
41 48 72 33 47 32 37 16 94 29
53 71 44 65 25 43 91 52 97 51 14
70 11 33 28 77 73 17 78 39 68 17 57
91 71 52 38 17 14 91 43 58 50 27 29 48
63 66 04 68 89 53 67 30 73 16 69 87 40 31
04 62 98 27 23 09 70 98 73 93 38 53 60 04 23"""
    
    triangle = np.array([[int(val) for val in row.split(' ')]
                         for row in triangle_str.split('\n')], dtype=object)
    
    # for ri, row in enumerate(triangle):
    #     for ci, val in enumerate(row):
    #         if ri > 0:
    #             string = f"{ri = }, {ci = }, {val = }, "
    #             if ci > 0:
    #                 string += f"{triangle[ri-1][ci-1]}, "
    #             if ci < ri:
    #                 string += f"{triangle[ri-1][ci]}"
    #             print(string)
    
    max_sum = max_path_sum(triangle)
                
    
    print(f"Solution for {triangle.tolist() = } is {max_sum = }")
    return 0


if __name__ == "__main__":
    main()
