# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:04:17 2019

@author: Joachim
"""
import numpy as np
import random as rd
from itertools import product, combinations
from functools import reduce 

def factor(n):
    f, fs = 3, []
    while n % 2 == 0:
        fs.append(2)
        n /= 2
    while f * f <= n:
        while n % f == 0:
            fs.append(f)
            n /= f
        f += 2
    if n > 1: fs.append(n)
    return fs

def cut(lst, indexes):
    last = 0
    for i in indexes:
        yield lst[last:i]
        last = i
    yield lst[last:]

def generate(lst, n):
    for indexes in combinations(list(range(1,len(lst))), n - 1):
        yield list(cut(lst, indexes))

def sublists(lst):
    for doslice in product([True, False], repeat=len(lst) - 1):
        slices = []
        start = 0
        for i, slicehere in enumerate(doslice, 1):
            if slicehere:
                slices.append(lst[start:i])
                start = i
        slices.append(lst[start:])
        yield slices
        
def order_sublists(lst, dim=4):
    sublst = list(sublists(lst))
    del_lst =[]
    for i in range(len(sublst)-1):
        if len(sublst[i]) != dim:
            del_lst.append(i)
    for index in sorted(del_lst, reverse=True):
        del sublst[index]
    return sublst

def reduce_list(lst, n, dim=4):
    del_lst = []
    for i in range(len(lst)-1):
        for j in range(dim):
            new_entry = reduce(lambda x, y: x*y, lst[i][j])
            lst[i][j] = new_entry
            if new_entry >= n:
                del_lst.append(i)  
    for index in sorted(np.unique(del_lst), reverse=True):
        del lst[index]
    return lst[:-1]

def permutate(n, dim=4, iterations=20):
    lst = factor(n*10**(2*dim))
    permutated_lst = []
    for i in range(iterations):
        x = rd.sample(lst, len(lst))
        permutated_lst.append(x)
        lst = x
    return permutated_lst
    
def main2(n, iterations=20, dim=4):
    lst = list(permutate(n, dim, iterations))
    newlst = []
    result = []
    erg = []
    for sublst in lst:
        newlst.append(reduce_list(order_sublists(sublst, dim), n*100, dim))
    for sets in newlst:
        for i in range(len(sets)):
            erg.append(reduce(lambda x, y: x+y, sets[i]))
    erg = [int(x) for x in erg]
    newlst = list(filter(None, newlst))
    for elem in np.where(np.array(erg) == n):
        result.append(np.concatenate(newlst)[elem])
    return result

def main(n, iterations=20, dim=4):
    if len(factor(n*100)) > 1:
        lst = list(permutate(n, dim, iterations))
        newlst = []
        result = []
        erg = []
        for sublst in lst:
            newlst.append(reduce_list(list(generate(sublst, dim)), n*100, dim))
        for sets in newlst:
            for i in range(len(sets)):
                erg.append(reduce(lambda x, y: x+y, sets[i]))
        erg = [int(x) for x in erg]
        newlst = list(filter(None, newlst))
        try:
            for elem in np.where(np.array(erg) == n*100):
                result.append(np.concatenate(newlst)[elem])
        except ValueError:
            return
        try:
            return result[0]
        except IndexError:
            return
#x = []
#for i in range(800, 1001):
#    erg = np.unique(main(i/100, 1000, 4))
#    print("Zahl: ", i/100, erg)
#    x.append(erg)
#print(x)
# for i in np.linspace(0, 1, 101):
#     print(main(109 + i, 1000, 3))
    
# print(np.linspace(0, 1, 101))
print(main(7.47, 1000, 4))