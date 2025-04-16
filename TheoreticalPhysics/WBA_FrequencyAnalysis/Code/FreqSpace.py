# -*- coding: utf-8 -*-
"""
High performance frequency grid calculations using numba.
"""

import numpy as np
from numba import njit, prange#, vectorize, guvectorize, float64, int32
import WBA_core_for_FreqSpace as WBA_core
import time
import matplotlib.pyplot as plt

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

@njit
def map4dnjit_sequential(p10, p20, q10, q20, Npoints, k1, k2, k):
    length = len(p10)
    points = np.zeros((4, Npoints, length), dtype=np.float64)
    for i in range(length):
        points[:, :, i] = map4dnjit(p10[i], p20[i], q10[i], q20[i], 
                                    Npoints, k1, k2, k)
    points[2:, :, :] -= 0.5
    return points

@njit(parallel=True)
def map4dnjitmulti(p10, p20, q10, q20, Npoints, k1, k2, k):
    length = len(p10)
    p1 = np.zeros((Npoints, length), dtype=np.float64)
    p2 = np.zeros((Npoints, length), dtype=np.float64)
    q1 = np.zeros((Npoints, length), dtype=np.float64)
    q2 = np.zeros((Npoints, length), dtype=np.float64)
    for i in prange(length):
        p1[:, i], p2[:, i], q1[:, i], q2[:, i] = map4dnjit(p10[i], p20[i], 
                                                           q10[i], q20[i],
                                                           Npoints, k1, k2, k)
    return p1, p2, q1, q2

# @guvectorize(['void(f8[:,:],f8,f8,f8,f8,f8[:,:])'], 
#               '(m,n),(),(),(),()->(m,n)', nopython=True, target='parallel')
# def map4d_step(old, my2pi, k1_2pi, k2_2pi, k_2pi, new):
#     new[2, :] = (old[2, :] + old[0, :]) % 1.0
#     new[3, :] = (old[3, :] + old[1, :]) % 1.0
#     coupling = k_2pi * np.sin(my2pi * (new[2, :] + new[3, :]))
#     new[0, :] = old[0, :] + coupling + k1_2pi * np.sin(my2pi * new[2, :])
#     new[1, :] = old[1, :] + coupling + k2_2pi * np.sin(my2pi * new[3, :])


# @njit
# def map4d_step(points, my2pi, k1_2pi, k2_2pi, k_2pi):
#     newPoints = np.array((4, len(points[0, :])))
#     newPoints[2, :] = (points[2, :] + points[0, :]) % 1.0
#     newPoints[3, :] = (points[3, :] + points[1, :]) % 1.0
#     coupling = k_2pi * np.sin(my2pi * (newPoints[2, :] + newPoints[3, :]))
#     newPoints[0, :] = points[0, :] + coupling + k1_2pi * np.sin(my2pi * newPoints[2, :])
#     newPoints[1, :] = points[1, :] + coupling + k2_2pi * np.sin(my2pi * newPoints[3, :])
#     return newPoints

# @njit
# def map4dnjit3multi(p10, p20, q10, q20, Npoints, k1, k2, k):
#     my2pi = 2*np.pi
#     k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
#     length = len(p10)
#     points = np.zeros((4, Npoints, length), dtype=np.float64)
#     points[0, 0, :] = p10
#     points[1, 0, :] = p20
#     points[2, 0, :] = q10
#     points[3, 0, :] = q20
#     for i in range(1, Npoints, 1):
#         points[:, i, :] = map4d_step(points[:, i-1, :], my2pi, k1_2pi, 
#                                       k2_2pi, k_2pi)
#         # points[:, i, :] = points[:, i-1, :]
#     return points

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

@njit 
def map4dnjit3multi(p10, p20, q10, q20, Npoints, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    length = len(p10)
    points = np.zeros((4, Npoints, length), dtype=np.float64)
    points[0, 0, :] = p10
    points[1, 0, :] = p20
    points[2, 0, :] = q10
    points[3, 0, :] = q20
    for i in range(1, Npoints, 1):
        points[2, i, :] = (points[2, i-1, :] + points[0, i-1, :]) % 1.0
        points[3, i, :] = (points[3, i-1, :] + points[1, i-1, :]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (points[2, i, :] + points[3, i, :]))
        points[0, i, :] = (points[0, i-1, :] + coupling + k1_2pi * 
                            np.sin(my2pi * points[2, i, :]))
        points[1, i, :] = (points[1, i-1, :] + coupling + k2_2pi * 
                            np.sin(my2pi * points[3, i, :]))
    points[2:, :, :] -= 0.5
    return points

@njit(parallel=True)
def map4d_step_njit(old, length, my2pi, k1_2pi, k2_2pi, k_2pi, new):
    for i in prange(length):
        new[2, i] = (old[2, i] + old[0, i]) % 1.0
        new[3, i] = (old[3, i] + old[1, i]) % 1.0
        coupling = k_2pi * np.sin(my2pi * (new[2, i] + new[3, i]))
        new[0, i] = old[0, i] + coupling + k1_2pi * np.sin(my2pi * new[2, i])
        new[1, i] = old[1, i] + coupling + k2_2pi * np.sin(my2pi * new[3, i])

@njit 
def map4dnjit_parallel(p10, p20, q10, q20, Npoints, k1, k2, k):
    my2pi = 2*np.pi
    k1_2pi, k2_2pi, k_2pi = k1 / my2pi, k2 / my2pi, k / my2pi
    length = len(p10)
    points = np.zeros((4, Npoints, length), dtype=np.float64)
    points[0, 0, :] = p10
    points[1, 0, :] = p20
    points[2, 0, :] = q10
    points[3, 0, :] = q20
    for i in range(1, Npoints, 1):
        map4d_step_njit(points[:, i-1, :], length, my2pi, k1_2pi, 
                        k2_2pi, k_2pi, points[:, i, :])
    return points

###############################################################################
# High performance WBA frequency calculations for a 4d grid of initials
###############################################################################

@njit
def _WBA_torus4d_single_parallel(points, N):
    # if WBA_core.r_squared_ratio_test(points, thresh): #FIXME: this is meh..
    points = WBA_core.transform_nd_torus(points) 
    x, y, z, w = WBA_core.sort_by_extent(points)
    phi1 = np.arctan2(x, z) / (2*np.pi)
    phi1Diff = phi1[1:] - phi1[:-1]
    phi1Diff = (phi1Diff + 0.5) % 1.0 - 0.5
    phi2 = np.arctan2(y, w) / (2*np.pi)
    phi2Diff = phi2[1:] - phi2[:-1]
    phi2Diff = (phi2Diff + 0.5) % 1.0 - 0.5
    weights = WBA_core._weights(N + 1)
    freq1p1 = np.sum(weights * phi1Diff[:N], dtype=np.float64)
    freq2p1 = np.sum(weights * phi2Diff[:N], dtype=np.float64)
    freq1p2 = np.sum(weights * phi1Diff[N-1:], dtype=np.float64)
    freq2p2 = np.sum(weights * phi2Diff[N-1:], dtype=np.float64)
    return freq1p1, freq2p1, freq1p2, freq2p2

@njit
def _WBA_torus4d_single_parallel_v2(points, N):
    points = WBA_core.transform_nd_torus(points) 
    x, y, z, w = WBA_core.sort_by_extent(points)
    phi1 = np.arctan2(x, z) / (2*np.pi)
    phi1Diff = phi1[1:] - phi1[:-1]
    WBA_core.embedding(phi1Diff)
    # phi1Diff = ((phi1[1:] - phi1[:-1]) + 0.5) % 1.0 - 0.5
    # if np.any(phi1Diff < -0.25) and np.any(phi1Diff > 0.25):
    #     phi1Diff %= 1.0
    phi2 = np.arctan2(y, w) / (2*np.pi)
    phi2Diff = phi2[1:] - phi2[:-1]
    WBA_core.embedding(phi2Diff)
    # phi2Diff = ((phi2[1:] - phi2[:-1]) + 0.5) % 1.0 - 0.5
    # if np.any(phi2Diff < -0.25) and np.any(phi2Diff > 0.25):
    #     phi2Diff %= 1.0
    weights = WBA_core._weights(N + 1)
    freq1p1 = np.sum(weights * phi1Diff[:N], dtype=np.float64)
    freq2p1 = np.sum(weights * phi2Diff[:N], dtype=np.float64)
    freq1p2 = np.sum(weights * phi1Diff[N-1:], dtype=np.float64)
    freq2p2 = np.sum(weights * phi2Diff[N-1:], dtype=np.float64)
    return freq1p1, freq2p1, freq1p2, freq2p2
    
@njit
def _WBA_torus4d_fast(points, N):
    length = len(points[0, 0, :])
    freq = np.zeros((4, length), dtype=np.float64)
    for i in range(0, length, 1):
        freq[:, i] = _WBA_torus4d_single_parallel_v2(points[:, :, i], N)
    return np.abs(freq)

@njit(parallel=True)
def _WBA_torus4d_parallel_fast(points, N, thresh):
    length = len(points[0, 0, :])
    freq = np.zeros((4, length), dtype=np.float64)
    for i in prange(0, length, 1):
        freq[:, i] = _WBA_torus4d_single_parallel(points[:, :, i], N, thresh)
    return np.abs(freq)

@njit(parallel=True)
def freq4d_grid_random_parallel(minval, maxval, counts, Npoints, 
                                k1, k2, k):
    p1min, p2min, q1min, q2min = minval
    p1max, p2max, q1max, q2max = maxval
    chunk, steps = counts
    totalSize = chunk * steps
    freq = np.zeros((4, totalSize), dtype=np.float64)
    print("Torus4d transform with random initial conditions ...")
    _freq = _WBA_torus4d_fast
    _map = map4dnjit3multi
    
    for i in prange(steps):
        p10 = np.random.uniform(p1min, p1max, chunk)
        p20 = np.random.uniform(p2min, p2max, chunk)
        q10 = np.random.uniform(q1min, q1max, chunk)
        q20 = np.random.uniform(q2min, q2max, chunk)
        print("Step ", i, " of ", steps)
        freq[:, i*chunk:(i+1)*chunk] = \
            _freq(_map(p10, p20, q10, q20, 2*Npoints, k1, k2, k), 
                  Npoints)
    return freq

# @njit
def freq4d_grid_random(minval, maxval, counts, Npoints, 
                       k1, k2, k, thresh=0.005):
    p1min, p2min, q1min, q2min = minval
    p1max, p2max, q1max, q2max = maxval
    p1count, p2count, q1count, q2count = counts
    chunk = p1count * p2count
    totalSize = np.prod(counts)
    freq = np.zeros((4, totalSize), dtype=np.float64)
    print("Torus4d transform with random initial conditions ...")
    _freq = _WBA_torus4d_parallel_fast
    # p10 = np.random.uniform(p1min, p1max, totalSize)
    # p20 = np.random.uniform(p2min, p2max, totalSize)
    # q10 = np.random.uniform(q1min, q1max, totalSize)
    # q20 = np.random.uniform(q2min, q2max, totalSize)
    # _freq = _Naff_multi   #does this even work? (2 instead of 4 freqs..)
    
    for ctr in range(0, totalSize, chunk):
        p10 = np.random.uniform(p1min, p1max, chunk)
        p20 = np.random.uniform(p2min, p2max, chunk)
        q10 = np.random.uniform(q1min, q1max, chunk)
        q20 = np.random.uniform(q2min, q2max, chunk)
        print("Step ", ctr, " of ", totalSize)
        # points = map4dnjit_parallel(p10[ctr:ctr+chunk], p20[ctr:ctr+chunk], 
        #                             q10[ctr:ctr+chunk], q20[ctr:ctr+chunk], 
        #                             2*Npoints, k1, k2, k)
        t2 = time.perf_counter()
        points = map4dnjit_parallel(p10, p20, q10, q20, 2*Npoints, k1, k2, k)
        t3 = time.perf_counter()
        print("Orbit in ", t3 - t2)
        points[2:, :, :] -= 0.5
        t4 = time.perf_counter()
        #print("Shift in ", t4 - t3)
        freq[:, ctr:ctr+chunk] = _freq(points, Npoints, thresh)
        print("Frequencies in ", time.perf_counter() - t4)
        points = None   # clear memory to avoid overflow performance loss
    return freq#, p10, p20, q10, q20

def compute_freq4d_grid(minval, maxval, counts, Npoints, k1, k2, k, 
                        _freq=freq4d_grid_random_parallel):
    t1 = time.time()
    f1p1, f2p1, f1p2, f2p2 = _freq(minval, maxval, counts, Npoints, 
                                   k1, k2, k)
    print("Frequency grid computed in ", time.time() - t1)
    return f1p1, f2p1, f1p2, f2p2

def plot_freq4d_grid(fVals, order=5, RetVal=0):
    f1p1, f2p1, f1p2, f2p2 = fVals
    diff1, diff2 = np.abs(f1p1 - f1p2), np.abs(f2p1 - f2p2)
    diff1[diff1 < 1e-16] = 1e-16
    diff2[diff2 < 1e-16] = 1e-16
    abl1, abl2 = -np.log10(diff1), -np.log10(diff2)
    indx = ((abl1 < order) | (abl2 < order) | (f1p1 < 1e-3) | (f2p1 < 1e-3))
    fig, ax = plt.subplots(figsize=(16, 10))
    # ax.axis([0.0, 0.5, 0.0, 0.5])
    f1plot, f2plot = f1p1[~indx], f2p1[~indx]
    # indx1, indx2 = (f1plot > 0.5), (f2plot > 0.5)
    indx12 = (f2plot > f1plot)
    f1plot[indx12], f2plot[indx12] = f2plot[indx12], f1plot[indx12]
    marker, ms, mew = '.', 1, 0
    if len(f1plot) < 10000:
        marker, ms, mew = 'o', 2, 0.5
    ax.plot(f1plot, f2plot, c='k', ls='', marker=marker, ms=ms, mew=mew)
    print(f"Frequency space with {len(f1plot)} points.")
    plt.show()
    if RetVal:
        return np.array([f1p1, f2p1, f1p2, f2p2])[:, ~indx]
    pass

if __name__ == "__main__":
    # PATHDATA = "C:\\WBA_Python\\WBA_Python\\DataFiles\\"
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    PATHDATA = PATH_TP + "CP_Bachelor\\WBA_Python\\FreqSpace\\"
    # fname = "FreqSpaceN10P_0_25to0_25P_0_25to0_25Q0_25to0_75Q0_25to0_75K2_25K3_0K1_0WBA_8000x8000random_order3.gz"
    
    
    print(__doc__)
    minval = np.array([-0.25, -0.25, 0.25, 0.25])
    maxval = np.array([0.25, 0.25, 0.75, 0.75])
    # minval = np.array([-0.15, -0.15, 0.35, 0.35])
    # maxval = np.array([0.15, 0.15, 0.65, 0.65])
    counts = np.array([1000, 8])
    # counts = np.array([1000, 25000])
    Npoints = 2048
    k1, k2, k = 2.25, 3.0, 1.0
    fname = (f"FreqSpaceN{int(np.log2(Npoints))}p{maxval[0]}p{maxval[1]}" + 
             f"q{maxval[2]}q{maxval[3]}k1{k1}k2{k2}k{k}" + 
             f"_{counts[0]}x{counts[1]}_")
    fname = fname.replace(".", "")
    fname += str(int(time.time())) + ".gz"
    ### CHANGE DATA PATH ON DIFFERENT PC !!!!
    # counts = np.array([200, 200, 20, 20])    # runtime estimate  seconds
    # counts = np.array([1000, 8])
    fVals = compute_freq4d_grid(minval, maxval, counts, Npoints, k1, k2, k,
                                _freq=freq4d_grid_random_parallel)
    frequencies = plot_freq4d_grid(fVals, order=3, RetVal=1)
    # np.savetxt(PATHDATA + fname, frequencies)