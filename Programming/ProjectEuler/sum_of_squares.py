#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 23:07:33 2022

@author: joachim
"""


def square_of_sum(n):
    return (n * (n+1) // 2)**2


def sum_of_squares(n):
    return n * (n+1) * (2*n + 1) // 6


def main():
    print(__doc__)
    
    n = 10
    sum_of_sq = sum_of_squares(n)
    sq_of_sum = square_of_sum(n)
    
    print(f"Solution for {n = } is {sq_of_sum - sum_of_sq}.")
    return 0


if __name__ == "__main__":
    main()
