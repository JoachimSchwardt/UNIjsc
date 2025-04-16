# -*- coding: utf-8 -*-
"""
Standard map Ahead-of-Time Compilation with numba.
"""

import numpy as np
from numba import njit, guvectorize, float64, vectorize
# from numba.pycc import CC
# from iterator.imports.maps.map_standard_fast2 import Map
from iterator.imports.maps.map_standard_fast_tv_cyl import Map as Map2d
from explorator.imports.maps.map_standard4d_v3_cyl import Mapping as Map4d

# @njit('UniTuple(f8[:, :], 2)(f8[:], f8[:], u4, f8)', cache=True)
# def _std_map_multi(q0, p0, Npoints, K):
#     q = np.zeros((Npoints, len(q0)), dtype=np.float64)
#     p = np.zeros((Npoints, len(p0)), dtype=np.float64)
#     q[0, :] = q0
#     p[0, :] = p0
#     for i in range(1, Npoints, 1):
#         p[i, :] = p[i-1, :] + K * np.sin(2*np.pi * q[i-1, :]) / (2*np.pi)
#         q[i, :] = q[i-1, :] + p[i, :]
#     return q % 1.0, p

# @njit('UniTuple(f8[:], 2)(f8, f8, u4, f8)', cache=True)
# def _std_map_njit(q0, p0, Npoints, K):
#     q = np.zeros(Npoints, dtype=np.float64)
#     p = np.zeros(Npoints, dtype=np.float64)
#     q[0] = q0
#     p[0] = p0
#     for i in range(1, Npoints, 1):
#         p[i] = p[i-1] + K * np.sin(2*np.pi * q[i-1]) / (2*np.pi)
#         q[i] = q[i-1] + p[i]
#     return q % 1.0, p

@njit#('UniTuple(f8[:], 2)(f8, f8, u4, f8)', cache=True)
def _std_map(q0, p0, Npoints, K):
    # mapping = Map2d(K)
    # return mapping.mapN(q0, p0, Npoints, return_orbit=True)
    q = np.zeros(Npoints, dtype=np.float64)
    p = np.zeros(Npoints, dtype=np.float64)
    q[0] = q0
    p[0] = p0
    for i in range(1, Npoints, 1):
        q[i] = (q[i-1] + p[i-1]) % 1.0
        p[i] = p[i-1] + K * np.sin(2*np.pi * q[i]) / (2*np.pi)
    return q, p

@njit#('UniTuple(f8[:,:], 2)(f8[:], f8[:], u4, f8)', cache=True)
def _std_map_multi(q0, p0, Npoints, K):
    length = len(q0)
    q = np.zeros((Npoints, length), dtype=np.float64)
    p = np.zeros((Npoints, length), dtype=np.float64)
    for i in range(0, length, 1):
        q[:, i], p[:, i] = _std_map(q0[i], p0[i], Npoints, K)
    return q, p

def _std_map4d(p1, p2, q1, q2, Npoints, k1=0.0, k2=0.0, k=0.0):
    mapping = Map4d(k1, k2, k)
    points = mapping.mapN([p1, p2, q1, q2], Npoints).points
    p1, p2, q1, q2 = points[:,0], points[:,1], points[:,2], points[:,3]
    return p1, p2, q1, q2
    
def _std_map_multi4d(p1, p2, q1, q2, Npoints, k1=0.0, k2=0.0, k=0.0):
    length = len(q1)
    q1arr = np.zeros((Npoints, length), dtype=np.float64)
    q2arr = np.zeros((Npoints, length), dtype=np.float64)
    p1arr = np.zeros((Npoints, length), dtype=np.float64)
    p2arr = np.zeros((Npoints, length), dtype=np.float64)
    for i in range(length):
        points = _std_map4d(p1[i], p2[i], q1[i], q2[i], Npoints, k1, k2, k)
        q1arr[:, i], q2arr[:, i], p1arr[:, i], p2arr[:, i] = points
    return p1arr, p2arr, q1arr, q2arr

def std_map(q0, p0, Npoints, K):
    """
    Standard map for any number of initial value pairs with orbit length 'N'.
    """
    # if type(q0) == int or type(p0) == int:
    #     print("q0 and p0 have to of type 'float' but were " + 
    #           f"type(q0)={type(q0)} and type(p0)={type(p0)}")
    #     raise TypeError
    # mapping = Map(K)
        
    if isinstance(q0, (float, np.float64)) and type(q0) == type(p0):
        return _std_map(q0, p0, Npoints, K)
        # return mapping.mapN(q0, p0, Npoints, return_orbit=True)
    
    if isinstance(q0[0], (float, np.float64)) and type(p0[0]) == type(q0[0]):
        return _std_map_multi(q0, p0, Npoints, K)
        # return mapping.mapNarray(q0, p0, Npoints)
    
    print("No method for given input types" + 
          f"type(q0)={type(q0)} and type(p0)={type(p0)}")
    raise TypeError

# cc = CC('std_map')
# # Uncomment the following line to print out the compilation steps
# cc.verbose = True

# @cc.export('_std_map', 'f8[:, :](f8, f8, f8, i4)')
# def _std_map(x0, y0, K, itrtns):
#     p = np.zeros((2, itrtns + 1), dtype=np.float64)
#     p[:, 0] = [x0, y0]
#     for i in range(1, itrtns + 1, 1):
        # p[1, i] = (p[1, i-1] - K * np.sin(2*np.pi*p[0, i-1]) 
        #            / (2*np.pi)) % 1.0
#         p[0, i] = (p[0, i-1] + p[1, i]) % 1.0
#     return p

# from numba import float64, int32, njit
# @njit(float64[:, :](float64, float64, float64, int32), cache=True)
# def _std_map2(x0, y0, K, itrtns):
#     p = np.zeros((2, itrtns + 1), dtype=np.float64)
#     p[:, 0] = [x0, y0]
#     for i in range(1, itrtns + 1, 1):
#         p[1, i] = (p[1, i-1] - K * np.sin(2*np.pi*p[0, i-1]) / (2*np.pi)) % 1.0
#         p[0, i] = (p[0, i-1] + p[1, i]) % 1.0
#     return p
    
###############################################################################
# Benchmarking different std maps
###############################################################################
    
def std_map4d(p10, p20, q10, q20, Npoints, k1, k2, k):
    # choose coorect mapping for the dimensions of the initial values
    if isinstance(p10, (float, np.float64)):
        mapping = map4dnjit
    elif isinstance(p10[0], (float, np.float64)):
        if len(p10) < 2*Npoints:
            mapping = map4dnjitmulti
        else:
            mapping = map4dnjit2multi
    return mapping(p10, p20, q10, q20, Npoints, k1, k2, k)

class Mapping4dCyl(object):
    def __init__(self, k1=2.25, k2=3.0, k=1.0):
        self.k1, self.k2, self.k = k1, k2, k
        
    def mapN(self, p10, p20, q10, q20, Npoints):
        # p10, p20, q10, q20 = init
        return map4dnjit(p10, p20, q10, q20, Npoints, 
                         self.k1, self.k2, self.k)
        
    def mapNarray(self, p10, p20, q10, q20, Npoints):
        # p10, p20, q10, q20 = init[:, 0], init[:, 1], init[:, 2], init[:, 3]
        if 2*Npoints > len(p10):
            return map4dnjitmulti(p10, p20, q10, q20, Npoints, 
                                  self.k1, self.k2, self.k)
        else:
            return map4dnjit2multi(p10, p20, q10, q20, Npoints, 
                                   self.k1, self.k2, self.k)
    
@njit
def map4dnjit(p10, p20, q10, q20, Npoints, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    p1 = np.zeros(Npoints, dtype=np.float64)
    p2 = np.zeros(Npoints, dtype=np.float64)
    q1 = np.zeros(Npoints, dtype=np.float64)
    q2 = np.zeros(Npoints, dtype=np.float64)
    
    p1[0] = p10
    p2[0] = p20
    q1[0] = q10
    q2[0] = q20
    
    for i in range(1, Npoints, 1):
        q1[i] = (q1[i-1] + p1[i-1]) % 1.0
        q2[i] = (q2[i-1] + p2[i-1]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (q1[i] + q2[i]))
        p1[i] = p1[i-1] + k1_2pi * np.sin(my2pi * q1[i]) + coupling
        p2[i] = p2[i-1] + k2_2pi * np.sin(my2pi * q2[i]) + coupling
    return p1, p2, q1, q2

def map4dc(p10, p20, q10, q20, Npoints, k1, k2, k):
    mapN = Map4d(k1, k2, k).mapN
    return mapN(np.array([p10, p20, q10, q20]), Npoints)

@njit
def map4dnjitmulti(p10, p20, q10, q20, Npoints, k1, k2, k):
    length = len(p10)
    p1 = np.zeros((Npoints, length), dtype=np.float64)
    p2 = np.zeros((Npoints, length), dtype=np.float64)
    q1 = np.zeros((Npoints, length), dtype=np.float64)
    q2 = np.zeros((Npoints, length), dtype=np.float64)
    for i in range(length):
        p1[:, i], p2[:, i], q1[:, i], q2[:, i] = map4dnjit(p10[i], p20[i], 
                                                           q10[i], q20[i],
                                                           Npoints, k1, k2, k)
    return p1, p2, q1, q2

@njit
def map4dnjit2multi(p10, p20, q10, q20, Npoints, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    length = len(p10)
    p1 = np.zeros((Npoints, length), dtype=np.float64)
    p2 = np.zeros((Npoints, length), dtype=np.float64)
    q1 = np.zeros((Npoints, length), dtype=np.float64)
    q2 = np.zeros((Npoints, length), dtype=np.float64)
    p1[0, :] = p10
    p2[0, :] = p20
    q1[0, :] = q10
    q2[0, :] = q20
    coupling = np.zeros(length, dtype=np.float64)
    for i in range(1, Npoints, 1):
        q1[i, :] = (q1[i-1, :] + p1[i-1, :]) % 1.0
        q2[i, :] = (q2[i-1, :] + p2[i-1, :]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (q1[i, :] + q2[i, :]))
        # for j in range(length):
        #     coupling[j] = k_2pi * np.sin(my2pi * (q1[i, j] + q2[i, j]))
        p1[i, :] = p1[i-1, :] + k1_2pi * np.sin(my2pi * q1[i, :]) + coupling
        p2[i, :] = p2[i-1, :] + k2_2pi * np.sin(my2pi * q2[i, :]) + coupling
    return p1,p2,q1,q2

# @guvectorize([(float64[:,:], float64, float64, float64, 
#                float64, float64[:,:])], '(n,n),(),(),(),()->(n,n)',
#              target='parallel')
# def map4d_step(arr, my2pi, k1_2pi, k2_2pi, k_2pi, new):
#     new[2, :] = (arr[2, :] + arr[0, :]) % 1.0
#     new[3, :] = (arr[3, :] + arr[1, :]) % 1.0
#     coup = k_2pi * np.sin(my2pi * (new[2, :] + new[3, :]))
#     new[0, :] = arr[0, :] + coup + k1_2pi * np.sin(my2pi * new[2, :])
#     new[1, :] = arr[1, :] + coup + k2_2pi * np.sin(my2pi * new[3, :])
    

# @njit
# def map4dnjit_parallel(p10, p20, q10, q20, Npoints, k1, k2, k):
#     my2pi = 2*np.pi
#     k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
#     length = len(p10)
#     points = np.zeros((4, Npoints, length), dtype=np.float64)
#     points[0, 0, :] = p10
#     points[1, 0, :] = p20
#     points[2, 0, :] = q10
#     points[3, 0, :] = q20
#     for i in range(1, Npoints, 1):
#         # points[2, i, :] = (points[2, i-1, :] + points[0, i-1, :]) % 1.0
#         # points[3, i, :] = (points[3, i-1, :] + points[1, i-1, :]) % 1.0
#         # coupling = k_2pi * np.sin(my2pi * (points[2, i, :] + points[3, i, :]))
#         # points[0, i, :] = (points[0, i-1, :] + coupling + k1_2pi * 
#         #                    np.sin(my2pi * points[2, i, :]))
#         # points[1, i, :] = (points[1, i-1, :] + coupling + k2_2pi * 
#         #                    np.sin(my2pi * points[3, i, :]))
#         map4d_step(points[:, i-1, :], my2pi, k1_2pi, k2_2pi, 
#                    k_2pi, points[:, i, :])
#     return points


# @vectorize
# def map4dnjit_vectorized(p10, p20, q10, q20, Npoints, k1, k2, k):
#     my2pi = 2*np.pi
#     k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
#     length = len(p10)
#     points = np.zeros((4, Npoints, length), dtype=np.float64)
#     points[0, 0, :] = p10
#     points[1, 0, :] = p20
#     points[2, 0, :] = q10
#     points[3, 0, :] = q20
#     for i in range(1, Npoints, 1):
#         map4d_step(points[:, i-1, :], my2pi, k1_2pi, k2_2pi, 
#                    k_2pi, points[:, i, :])
#     return points
    

def map4dcmulti(p10, p20, q10, q20, Npoints, k1, k2, k):
    length = len(p10)
    p1 = np.zeros((Npoints, length), dtype=np.float64)
    p2 = np.zeros((Npoints, length), dtype=np.float64)
    q1 = np.zeros((Npoints, length), dtype=np.float64)
    q2 = np.zeros((Npoints, length), dtype=np.float64)
    for i in range(length):
        p1[:, i], p2[:, i], q1[:, i], q2[:, i] = map4dc(p10[i], p20[i], 
                                                        q10[i], q20[i],
                                                        Npoints, k1, k2,
                                                        k).points.T
    return p1, p2, q1, q2

if __name__ == "__main__":
    """
    N = 2**10
    %timeit -n 1000 map4dpy(p10,p20,q10,q20,N,k1,k2,k)
9.14 ms ± 235 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    %timeit -n 1000 map4dc(p10,p20,q10,q20,N,k1,k2,k)
1.47 ms ± 76.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    %timeit -n 1000 map4dnjit(p10,p20,q10,q20,N,k1,k2,k)
71.5 µs ± 7.72 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    N = 2**16
    %timeit -n 100 map4dnjit(p10,p20,q10,q20,N,k1,k2,k)
4.39 ms ± 76.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    %timeit -n 100 map4dc(p10,p20,q10,q20,N,k1,k2,k)
55.5 ms ± 1.44 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

    multi maps
    p10arr=np.linspace(0.0,0.3,N0)
    p20arr=np.linspace(-0.3,0.1,N0)
    q10arr=np.linspace(0.3,0.7,N0)
    q20arr=np.linspace(0.4,0.6,N0)
    N = 2**16
    N0 = 50
    %timeit -n 1 -r 1 map4dnjitmulti(p10arr, p20arr, q10arr, q20arr, N,k1,k2,k)
478 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    %timeit -n 1 -r 1 map4dcmulti(p10arr, p20arr, q10arr, q20arr, N,k1,k2,k)
4.59 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

    N = 2**10
    N0 = 2**11
    %timeit -n 1 -r 1 map4dcmulti(p10arr, p20arr, q10arr, q20arr, N,k1,k2,k)
4.16 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    %timeit -n 1 -r 1 map4dnjitmulti(p10arr, p20arr, q10arr, q20arr, N,k1,k2,k)
247 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    
    """
    # cc.compile()
    print(__doc__)
    """
@vectorize(nopython=True)
def map4d_step(arr, length, my2pi, k1_2pi, k2_2pi, k_2pi):
    new = np.zeros((4, length), dtype=np.float64)
    new[2, :] = (arr[2, :] + arr[0, :]) % 1.0
    new[3, :] = (arr[3, :] + arr[1, :]) % 1.0
    coup = k_2pi * np.sin(my2pi * (new[2, :] + new[3, :]))
    new[0, :] = (arr[0, :] + coup + k1_2pi * np.sin(my2pi * new[2, :]))
    new[1, :] = (arr[1, :] + coup + k2_2pi * np.sin(my2pi * new[3, :]))
    return new
    
from numba import guvectorize, float64
@guvectorize([(float64[:,:], float64, float64, float64, float64, float64[:,:])], '(n,n),(),(),(),()->(n,n)', target='parallel')
def map4d_step(arr, my2pi, k1_2pi, k2_2pi, k_2pi, new):
    new[2, :] = (arr[2, :] + arr[0, :]) % 1.0
    new[3, :] = (arr[3, :] + arr[1, :]) % 1.0
    coup = k_2pi * np.sin(my2pi * (new[2, :] + new[3, :]))
    new[0, :] = (arr[0, :] + coup + k1_2pi * np.sin(my2pi * new[2, :]))
    new[1, :] = (arr[1, :] + coup + k2_2pi * np.sin(my2pi * new[3, :]))
    """