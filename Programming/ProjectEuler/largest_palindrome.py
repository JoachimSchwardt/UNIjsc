#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the largest palindrome constructed from a product of two 
'd'-digit numbers
"""


def is_palindrome(val):
    string = str(val)
    return string == string[::-1]


def max_palindrome(digits):
    maxval = 10**digits
    pal = 0
    for x in range(1, maxval):
        for y in range(x, maxval):
            val = x * y
            if val > pal and is_palindrome(val):
                pal = val            
    
    return pal


def main():
    print(__doc__)
    digits = 3
    pal = max_palindrome(digits)
    print(f"Solution for {digits = } is {pal}.")
    return 0


if __name__ == "__main__":
    main()
