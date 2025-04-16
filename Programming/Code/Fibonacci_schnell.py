# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:12:10 2018

@author: Joachim
"""
import math
def Fibo_schnell(n):
    p=(1+math.sqrt(5))/2
    x=((p**n-(1-p)**n)/math.sqrt(5))
    return math.floor(x)
def Fibo_Reihe(n):
    for i in range(n+1):
        print(Fibo_schnell(i))
def Fibo_Ratio(n):
    for i in range(2,n+1):
        x=Fibo_schnell(i)/Fibo_schnell(i-1)
        print("{} ".format(i),"{:.40f}".format(x))
    return " "
print(Fibo_Ratio(50))