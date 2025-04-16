#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 20:53:07 2022

@author: joachim
"""

import numpy as np


ALPHABET = {'a' : 1,
            'b' : 2,
            'c' : 3,
            'd' : 4,
            'e' : 5,
            'f' : 6,
            'g' : 7,
            'h' : 8,
            'i' : 9,
            'j' : 10,
            'k' : 11,
            'l' : 12,
            'm' : 13,
            'n' : 14,
            'o' : 15,
            'p' : 16,
            'q' : 17,
            'r' : 18,
            's' : 19,
            't' : 20,
            'u' : 21,
            'v' : 22,
            'w' : 23,
            'x' : 24,
            'y' : 25,
            'z' : 26,}

def value(name):
    return np.sum([ALPHABET[letter] for letter in name.lower()])


def main():
    print(__doc__)
    names = np.loadtxt("names.txt", delimiter=',', dtype=str)
    names = [name.replace('"', '') for name in names]
    names_sort = np.sort(names)
    names_values = np.array([value(name) for name in names_sort])
    
    score = np.sum(np.arange(1, names_values.size+1) * names_values)
    
    print(f"Name score is {score = }")
    
    return 0

if __name__ == "__main__":
    main()
