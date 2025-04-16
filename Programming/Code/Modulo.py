# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:49:18 2019

@author: Joachim
"""
import time as time

"""
(a*b) mod c = (a mod c * b mod c) mod c
(a+b) mod c = (a mod c + b mod c) mod c
n = sum_0^k a_k*10**(9*k)
n mod d = (sum_1^k mod d + a_0*10**(9*0) mod d) mod d
b_k mod d = a_k*10**(9*k) mod d = (a_k mod d * mod_10_9(k, d)) mod d
n mod d = sum b_k mod d = (sum_0^k-1 b_k mod d + b_k mod d) mod d
"""

def mod_sub(n, d): # for n in range 1e9
    return n - d*int(n/d)

def mod_10_9(power, d): # calculates 10**(9*power) mod d
    x = mod_sub(1e9, d)
    if power == 0:
        return 1
    elif power == 1:
        return x
    else:
        return mod_sub(mod_10_9(power-1, d) * x, d)
    
def mod_sum(array, d): # calculates sum(array) mod d
    l = len(array)
    x = array[0]
    if l==1:
        return x
    else:
        return mod_sub(mod_sum(array[1:l], d) + x, d)
    
def mod_array(n, d):
    num = str(n)
    l = len(num)
    x = 9*int(l/9)
    array = []
    res = []
    counter = 0
    for i in range(0, x, 9):
        array.append(int(num[l-9-i:l-i]))
    try:
        array.append(int(num[0:l-x]))
    except ValueError:
        None
    for elem in array:
        res.append(mod_sub(mod_sub(elem, d)*mod_10_9(counter, d), d))
        counter += 1
    return res

def mod(n, d):
    array = mod_array(n, d)       
    return mod_sum(array, d)

number = 10**1000 + 1
d = 470

#print(mod_sum([1, 2, 3], 5))

#t1 = time.clock()
#erg = sc.floor(number + sc.sqrt(3))
#t2 = time.clock()
#print("Python: {} with {}s".format(erg, t2-t1))

t1 = time.clock()
erg = number%d
t2 = time.clock()
print("Python: {} with {}s".format(erg, t2-t1))

t1 = time.clock()
erg = mod(number, d)
t2 = time.clock()
print("Approx: {} with {}s".format(erg, t2-t1))