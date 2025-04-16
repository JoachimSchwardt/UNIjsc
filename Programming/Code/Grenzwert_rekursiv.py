# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 20:50:35 2019

@author: Joachim
"""
import sympy as sy
def func_exact(s, x=sy.Symbol('x')):
    f = (1+1/x)**x
    for i in range(s):
        f_lim = sy.limit(f, x, sy.oo)
        f = x*(f_lim - f) 
    return f_lim
sy.pprint(func_exact(3))