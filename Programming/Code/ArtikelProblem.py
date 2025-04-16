"""
Kombinatorische Loesung des Artikel-Problems:
    Gegeben ist ein Preis, also eine Zahl 'c (cost)', sodass '100*c' eine 
    natuerliche Zahl wird. Gesucht sind nun n-Tupel an Preisen (a_j), sodass
        'sum_j a_j = c'    und    'prod_j a_j = c' ist. 
        
Dem Programm wird dafuer der Preis als natuerliche Zahl uebergeben, also 
entsprechen 7.37 Euro gerade dem Input 737. Das zweite Argument ist die Anzahl
an 'Dimensionen n', also die Laenge des 'a_j'-Vektors.

Das Ergebnis ist '100*a_j', also ein Vektor natuerlicher Zahlen.
"""

import numpy as np
from itertools import product, combinations
from functools import reduce 
import time

def factor(n):
    """
    Primfaktorzerlegung einer Zahl 'n'.
    """
    f, fs = 3, []
    while n % 2 == 0:             # Primzahl 2 finden
        fs.append(2)
        n /= 2
    while f * f <= n:             # Primzahlen groesser 2 finden
        while n % f == 0:
            fs.append(f)
            n /= f
        f += 2                    # naechstgroessere moegliche Primzahl
    if n > 1:                     # genau dann, wenn n eine Primzahl ist
        fs.append(int(n))  
    return fs

def p_factor(n, dim=4):
    """
    Fuer den Algorithmus muss noch (2n-2)-mal der Faktor 2 und der Faktor 5
    hinzugefugt werden.
    Die Funktion gibt zudem eine Liste aller vorhandenen Primzahlen zurueck.
    """
    factor_n = factor(n)                  # Primfaktoren von 'n'
    for i in range(2*dim - 2):  
        factor_n.append(2)                # Fuegt 2n-2 mal den Faktor 2 hinzu
        factor_n.append(5)                # ... Faktor 5 ...
        
    primes = sorted(set(factor_n))        # Liste vorhandener Primzahlen
    powers = []          # zugehoerige Liste der Faktoren jeder Primzahl
    for p in primes:     # Iteration ueber Primzahlen
        powers.append(factor_n.count(p))  # Anzahl des Faktors 'p' 
        
    return primes, powers

# def partition(n, dim=4):
#     """
#     Gibt alle moeglichen Partitionen der Zahl 'n' mit Laenge 'd' zurueck.
#     # https://stackoverflow.com/questions/10035752/elegant-python-code-for
#     #-integer-partitioning          (Link zur Quelle)
#     """
#     def partition_subroutine(n, dim, depth=0):
#         """Sub-Routine fuer den Algorithmus."""
#         if depth == dim:
#             return [[]]
#         lst = []
#         for i in range(n+1):
#             for item in partition_subroutine(n-i, dim, depth=depth+1):
#                 lst.append(item + [i])
#         return lst
    
#     lst = []
#     for p in partition_subroutine(n, dim-1):
#         lst.append([n - sum(p)] + p)
#     np.random.shuffle(lst)
#     return lst

def partition(n, k, l=0):
    """
    n is the integer to partition, k is the length of partitions, 
    l is the min partition element size.
    #https://stackoverflow.com/questions/18503096/python-integer-partitioning
    #-with-given-k-partitions
    """
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield [n,]
        return
    for i in range(l, n//k+1):
        for result in partition(n-i,k-1,i):
            yield [i,] + result

def expand_partition(intgr, dim=4, n=1000):
    """
    Erweitert die Laenge der Partition auf 'n * dim'. 
    Dazu werden die Elemente mit 'g_i = 1/var_i' gewichtet. 
    Sei 'p_i' ein Element der Partition 'p_intgr'. Dann definiert man
        'var_i := sum_j (p_i[j] - p_i_avg)**2', mit 'p_i_avg = intgr / dim'.
    Es soll 'sum_i g_i = n * dim' gelten, also muessen die Gewichte mit
        'g_i *= n * dim / (sum_i g_i)' mmultipliziert werden.
    
    Gibt ein Numpy-Array mit Shape (dim, n * dim) zurueck. 
    """
    n = n * dim                           # Um n % dim == 0 sicherzustellen
    p_intgr = [*partition(intgr, dim)]    # Partition von 'intgr' erstellen
    var = np.sum((np.array(p_intgr) - intgr / dim)**2, axis=1)
    var[var == 0] = min(var[var != 0])
    g = (n / sum(1 / var) * 1 / var).astype(int)      # Gewichte normieren
    for i in range(n - sum(g)):    # Durch Abrunden muessen noch 'n - sum(g)'
        g[i % len(g)] += 1         # Elemente angefuegt werden
    
    expand_p = []
    for i in range(len(p_intgr)):
        expand_p.append(p_intgr[i] * g[i])
    expand_p = [item for sublst in expand_p for item in sublst]
    return np.array(expand_p).reshape((n, 4))
            
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

def aj_expand(cost, dim=4, n=1000):
    primes, powers = p_factor(cost, dim)
    
    powers_aj = np.zeros((len(primes), n * dim, dim))
    for i in range(len(primes)):
        p_i = expand_partition(powers[i], dim, n)
        powers_aj[i, :, :] = p_i
        # Spalten von powers_aj[i] shuffeln
        powers_aj[i] = shuffle_along_axis(powers_aj[i], axis=1)
        # Zeilen shuffeln
        # np.random.shuffle(powers_aj[i])
        
    # Primzahlen potenzieren mit 'powers_aj' Array
    primes_aj = (primes**powers_aj.T).T
    # Produkt der Primfaktoren ergibt die 'aj'
    aj = np.prod(primes_aj, axis=0, dtype=int)
    # Summe fuer Test auf 'aj_sum == n'
    aj_sum = np.sum(aj, axis=1)
    print(primes_aj)
    print(aj_sum)
    return aj_sum, aj

def calculate_aj(cost, dim=4, n=1000):
    """
    Berechnet moegliche Loesungen fuer a_j und deren Summen a_j_sum.
    """
    primes, powers = p_factor(cost, dim)
    
    p_lst = []
    p_length = []
    for i in range(len(primes)):
        p_i = [*partition(powers[i], dim)]
        p_lst.append(p_i)
        p_length.append(len(p_i))
        
    total_length = np.prod(p_length) * n
    powers_aj = np.zeros((len(primes), dim, total_length))
    for i in range(len(primes)):
        iter_length = int(total_length / p_length[i])
        p_lst_multiple = np.array(p_lst[i] * iter_length)
        powers_aj[i] = p_lst_multiple.reshape((total_length, dim)).T
        
        # Spalten von powers_aj[i] shuffeln
        powers_aj[i] = shuffle_along_axis(powers_aj[i], axis=0)
        # Zeilen shuffeln
        np.random.shuffle(powers_aj[i].T)
    
    # Primzahlen potenzieren mit 'powers_aj' Array
    primes_aj = (primes**powers_aj.T).T
    # Produkt der Primfaktoren ergibt die 'aj'
    aj = np.prod(primes_aj, axis=0, dtype=int)
    # Summe fuer Test auf 'aj_sum == n'
    aj_sum = np.sum(aj, axis=0)
    return aj_sum, aj


def Solution(cost, dim=4, n=1000):
    aj_sum, aj = calculate_aj(cost, dim, n)
    return aj[:, (aj_sum == cost)].T

def Solution_expand(cost, dim=4, n=1000):
    aj_sum, aj = aj_expand(cost, dim, n)
    return aj[(aj_sum == cost), :]

def main():
    cost = 747          # cost * 100
    dim = 4             # number of dimensions
    n = 10000            # number of iterations
    
    t1=time.time()
    # result = Solution(cost, dim, n)
    # print(result)
    # result = aj_expand(cost, dim, n)
    # print(result)
    result = Solution_expand(cost, dim, n)
    print(result)
    t2=time.time()
    print(t2-t1)
    # result = main2(7.47, n, dim)
    # print(result)
    # t3=time.time()
    # print(t3-t2)
    
    
import random as rd

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
    
def main3(n, iterations=20, dim=4):
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

def main2(n, iterations=20, dim=4):
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
# x = []
# for i in range(800, 1001):
#     erg = np.unique(main2(i/100, 1000, 4))
#     print("Zahl: ", i/100, erg)
#     x.append(erg)
# print(x)
# for i in np.linspace(0, 1, 101):
#     print(main2(109 + i, 1000, 3))
    
# print(np.linspace(0, 1, 101))    
if __name__ == "__main__":
    main()

