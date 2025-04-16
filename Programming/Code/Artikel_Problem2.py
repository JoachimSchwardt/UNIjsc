# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:29:11 2018

@author: Joachim
"""
import numpy as np
#def Artikel_Preise(n):
#    a = b = c = d = 0
#    while True:
#        for i in range(int(2*n / 3)):
#            b=0
#            a+=25
#            for j in range(int(n / 2)):
#                c=0
#                b+=10
#                for k in range(int(n / 4)):
#                    d=0
#                    c+=2
#                    for l in range(int(n / 4)):
#                        d+=2
#                        if (a+b+c+d)==n:
#                            if int(a*b*c*d)==int(n*1e6):
#                               return a, b, c, d
#    
#if __name__ == "__main__":
#    print("Die Werte für {} sind:".format(737), Artikel_Preise(800))
    
#
#def test(n):
#    a = b = 0
#    while True:
#        for i in range(10):
#            a+=1
#            b=0
#            for j in range(10):
#                b+=1
#                if a+b==4:
#                    if a*b==4:
#                        return a, b         

#q+w+e+r=x
#q*w*e*r=x
# -> q=x/(wer)
# -> x/(wer)+w+e+r=x
# -> x/(er) +ww+we+wr=wx
# -> ww+w(e+r-x)+x/(er)=0
# -> w1 = -(e+r-x)/2+sqrt((e+r-x)**2-4*x/(er))
# -> w2 = -(e+r-x)/2-sqrt((e+r-x)**2-4*x/(er))
# -> q = x/(er*(-(e+r-x)/2+sqrt((e+r-x)**2-4*x/(er))))
from itertools import product, permutations
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

def Klassen(n):
    Faktoren = np.array(factor(n*1e6)).astype(int)
    K = np.array([])
    search = 0
    search_recent = 0
    for i in range(int(Faktoren[len(Faktoren)-1])):
        search += np.searchsorted(Faktoren, i+3) - search_recent
        if search == 0:
            i -= 1
        else:
            K = np.append(K, search)            
            search_recent += search
            search = 0
    K[0] -= 1
    for j in range(len(K)):
        K[j] += K[j-1]
    Set = np.array(np.split(Faktoren, K.astype(int)))
    return Set[:-1]

def Kombinationen(n):
    Set = Klassen(n)
    samples = np.array([]).astype(int)
    for i in range(len(Set)-1):
        samples = np.append(samples, Set[i][0])
    K = np.array([])
    for i in range(len(Set)-1):
        for j in range(len(Set)-1):
            K = np.append(K, np.outer(samples[i], samples[j]))
        Excess = np.where(K >= 2*n / 3)
        K = np.unique(np.sort(np.delete(K, Excess)).astype(int))
    return K

def Classes(n):
    Faktoren = np.array(factor(n*1e6)).astype(int)
    K = np.array([])
    search = 0
    search_recent = 0
    for i in range(int(Faktoren[len(Faktoren)-1])):
        search += np.searchsorted(Faktoren, i+3) - search_recent
        if search == 0:
            i -= 1
        else:
            K = np.append(K, search)            
            search_recent += search
            search = 0
    return K

""" How to create multidimensional list:
    classes = [([0] * 5) for i in range(4)]"""
    
def Klassen2(n):
    factors = factor(n*1e6)
    uniques = np.unique(factors)
    indices = ([0] * len(uniques))
    for i in range(len(uniques)):
        indices[i] = np.where(factors == uniques[i])[0][-1] + 1
    classes = np.split(factors, indices)
    return [*uniques], classes[:-1]

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
        
def order_sublists(lst, dim):
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

def main(n, dim=4):
    lst = list(permutations(Klassen2(n)[1]))
    newlst = []
    result = []
    erg = []
    for sublst in lst:
        newlst.append(reduce_list(list(order_sublists(np.concatenate(sublst), dim)), n))
    for sets in newlst:
        for i in range(len(sets)):
            erg.append(reduce(lambda x, y: x+y, sets[i]))
    erg = [int(x) for x in erg]
    for elem in np.where(np.array(erg) == n):
        result.append(np.concatenate(newlst)[elem])
    return result

print(main(747))
#print(factor(134), factor(125), factor(125), factor(352))
#x = [[1,2],[3,4],[5,6]]
#x = (([2., 2., 2., 2., 2., 2.]), ([5., 5., 5., 5., 5., 5.]), ([11.]), ([67.]))
#x = sum(x, [])
#print(x)

#print(list(permutations(Klassen2(737)[1])))
#mylist = factor(737*1e6)
#print(*Klassen(737))
#print(*list(sublists([*Klassen(737)])))
#newlist = list(order_sublists(mylist, 4))

#print(reduce_list(newlist, 737))