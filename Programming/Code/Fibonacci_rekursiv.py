# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:07:45 2018

@author: Joachim
"""
#import timeit
#tic=timeit.default_timer()
def Fibo_rekursiv(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return Fibo_rekursiv(n-1)+Fibo_rekursiv(n-2)
print(Fibo_rekursiv(6))
#toc=timeit.default_timer()
#print(toc - tic) 
##exponentieller Anstieg der Laufzeit 
##55.8s, 0.35s, 0.006s, 0,00016s