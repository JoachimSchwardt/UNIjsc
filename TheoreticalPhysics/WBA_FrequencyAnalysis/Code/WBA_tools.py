"""
Tools for analysis of the WBA method and comparison to naff.
Contains some methods for visualization of grids and convergence behavior.
"""

import numpy as np
import functools
import time
import matplotlib.pyplot as plt
from std_map import _std_map, _std_map_multi, map4dnjit2multi
import WBA_core
from numba import njit
from matplotlib import rcParams
rcParams["figure.dpi"] = 100



###############################################################################
# General tools for analysis of convergence
###############################################################################

@njit('u4[:](f8, f8, u4)', cache=True)
def N_arr(Nmin, Nmax, NN):
    """
    Logarithmic distribution of 'NN' integers in [2**Nmin ... 2**Nmax]
    """
    res = 2**np.linspace(Nmin, Nmax, NN)
    return np.unique(res.astype(np.uint32))

###############################################################################
# Functions for easier comparison to the naff method
###############################################################################

from CPG.naff.examples.std_map_frequencies import compute_freq
from explorator.comp.naff_call import naff_4d

def _Naff(N, q, p, MapToCircle=1):
    freq = np.zeros(len(N), dtype=np.float64)
    for i in range(len(N)):
        freq[i] = compute_freq(q[:N[i]], p[:N[i]], MapToCircle)
    return freq

def _Naff4d(N, points, proj=0):
    #projection 0 or 1 for q1p1, or q2,p2
    shape = np.shape(points)
    if shape[0] < shape[1]:
        points = points.T
    freq = np.zeros((2, len(N)))
    for i in range(len(N)):
        freq[:, i] = naff_4d(points[:N[i], :], proj=proj)
    return freq

def _Naff_multi(q0, p0, N=200, K=0.0, MapToCircle=1, Continuous=1):
    FlagShape = (np.shape(np.shape(q0))[0] == 1)
    if FlagShape:
        length = len(q0)
        freqNaff = np.zeros(length)
        q, p = _std_map(q0[0], p0[0], N, K)
    else:
        length = len(q0[0])
        freqNaff = np.zeros(length)
        q, p = q0[:, 0], p0[:, 0]
    
    freqNaff[0] = compute_freq(q, p, MapToCircle)
    if freqNaff[0] > 0.5 and Continuous == 1:
        freqNaff[0] = 1 - freqNaff[0]
    for i in range(1, length, 1):
        if FlagShape:
            q, p = _std_map(q0[i], p0[i], N, K)
        else:
            q, p = q0[:, i], p0[:, i]
        freqNaff[i] = compute_freq(q, p, MapToCircle)
        if Continuous == 1:
            if (abs(freqNaff[i] - freqNaff[i-1]) > 
                abs(1 - freqNaff[i] - freqNaff[i-1])):
                freqNaff[i] = 1 - freqNaff[i]
    return freqNaff

###############################################################################
# WBA as chaos indicator by comparison of convergence for the two halves of a
# given set of orbits
###############################################################################

# @njit('f8(u4, f8[:])', cache=True)   
def _absdiff_N2N_single(N, p):
    """
    Computes the difference in frequency for a 2N-orbit in phase-space.
    Frequencies are based off of the first 'N' and latter 'N' points.
    Returns the difference of both frequencies. 
    
    Call this function with 'p = cos(2*pi*q)' for use as a chaos indicator.
    """
    N_div2 = N // 2
    WBA_N = WBA_core._WBA_single(p[:N_div2])
    WBA_2N = WBA_core._WBA_single(p[N_div2 + N%2:])
    return np.abs(WBA_N - WBA_2N)

# @njit('f8[:](u4[:], f8[:])', cache=True)
def _absdiff_N2N(Narr, p):
    """
    Computes the difference in frequency for a 2N-orbit in phase-space 
        for all 'N' in 'Narr'.
    Frequencies are based off of the first 'N' and latter 'N' points.
    Returns an array containing the differences between those frequencies.
    
    Call this function with 'p = cos(2*pi*q)' for use as a chaos indicator.
    """
    Narr_div2 = (Narr // 2).astype(np.uint32)
    WBA_N = WBA_core._WBA(Narr_div2, p[:np.max(Narr_div2)])
    WBA_2N = WBA_core._WBA(Narr_div2, p[len(p) - np.max(Narr_div2):])
    return np.abs(WBA_N - WBA_2N)

@njit('f8[:](u4, f8[:, :])')
def _absdiff_N2N_multi(N, p):
    """
    Computes the difference in frequency for a 2N-orbit in phase-space 
        for all 'N' in 'Narr'.
    Frequencies are based off of the first 'N' and latter 'N' points.
    Returns an array containing the differences between those frequencies.
    
    Call this function with 'p = cos(2*pi*q)' for use as a chaos indicator.
    -> not necessary as of 27.03.2021, fixed issue with 0.5-boundary.
    """
    N_div2 = N // 2
    WBA_N = WBA_core._WBA_multi(p[:N_div2, :])
    WBA_2N = WBA_core._WBA_multi(p[N_div2 + N%2:, :])
    return np.abs(WBA_N - WBA_2N)

@njit('f8[:,:](f8[:], f8[:], u4, f8, b1)')
def _grid_absdiff_N2N(q0, p0, N, K, UseCos):
    Nq, Np = len(q0), len(p0)
    
    absDiffGrid = np.zeros((Nq, Np), dtype=np.float64)
    for i in range(Nq):
        q, p = _std_map_multi(np.full(Np, q0[i]), p0, N, K)
        if UseCos:
            absDiffGrid[i, :] = _absdiff_N2N_multi(N, np.cos(2*np.pi*q))
        else:
            absDiffGrid[i, :] = _absdiff_N2N_multi(N, p)
    return absDiffGrid

def _Naff_arr(q, p, MapToCircle):
    freq = np.zeros(len(q[0]), dtype=np.float64)
    for i in range(len(q[0])):
        freq[i] = compute_freq(q[:, i], p[:, i], MapToCircle)
    return freq

def _absdiff_N2N_multi_Naff(N, q, p, MapToCircle):
    N_div2 = N // 2
    Naff_N = _Naff_arr(q[:N_div2, :], p[:N_div2, :], MapToCircle)
    Naff_2N = _Naff_arr(q[N_div2 + N%2:, :], p[N_div2 + N%2:, :], MapToCircle)
    indx = (np.abs(Naff_N - Naff_2N) > np.abs(Naff_N - 1 + Naff_2N))
    Naff_2N[indx] = 1 - Naff_2N[indx]
    return np.abs(Naff_N - Naff_2N)

def _grid_absdiff_N2N_Naff(q0, p0, N, K, MapToCircle=1):
    Nq, Np = len(q0), len(p0)
    
    absDiffGrid = np.zeros((Nq, Np), dtype=np.float64)
    for i in range(Nq):
        q, p = _std_map_multi(np.full(Np, q0[i]), p0, N, K)
        absDiffGrid[i, :] = _absdiff_N2N_multi_Naff(N, q, p, MapToCircle)
    return absDiffGrid

###############################################################################
# WBA and Naff applied to two frequency signals on a grid
###############################################################################

#experimental
def WBA_arctan2_shift(q, p):
    phi = np.arctan2(q, p) / (2*np.pi)
    phiDiff = phi[1:] - phi[:-1]
    WBA_core.embedding(phiDiff)
    # shift = np.mean(np.abs(phiDiff))
    # phiDiff = (phiDiff + shift) % 1.0 - shift
    return WBA_core._WBA_single(phiDiff)

# @njit('f8[:,:](f8[:], f8[:], f8, u4)', cache=True)
def _2f_grid(freq, freq2, ampl, N):
    lenFreq = len(freq)
    lenFreq2 = len(freq2)
    Narr = np.arange(0.0, N, 1.0, dtype=np.float64)
    _freq = WBA_core._WBA_single_arctan2
    _freq = WBA_arctan2_shift
    
    freqGrid = np.zeros((lenFreq, lenFreq2), dtype=np.float64)
    for i in range(lenFreq):
        z1 = np.exp(2*np.pi*1j * freq[i] * Narr)
        for j in range(lenFreq2):
            z2 = ampl * np.exp(2*np.pi*1j * freq2[j] * Narr)
            z = z1 + z2
            freqGrid[i, j] = _freq(z.real, z.imag)
    
    return freqGrid % 1.0

def _2f_grid_Naff(freq, freq2, ampl, N, MapToCircle=1):
    lenFreq = len(freq)
    lenFreq2 = len(freq2)
    Narr = np.arange(0.0, N, 1.0, dtype=np.float64)
    
    freqGrid = np.zeros((lenFreq, lenFreq2), dtype=np.float64)
    for i in range(lenFreq):
        z1 = np.exp(2*np.pi*1j * freq[i] * Narr)
        for j in range(lenFreq2):
            z2 = ampl * np.exp(2*np.pi*1j * freq2[j] * Narr)
            z = z1 + z2
            q, p = WBA_core.map_arctan2(z.real, z.imag)
            freqGrid[i, j] = compute_freq(q, p, MapToCircle)
    
    return freqGrid % 1.0


###############################################################################
# High performance WBA frequency calculations for a 4d grid of initials
###############################################################################

@njit
def _WBA_torus4d_single_parallel(points, N, thresh):
    points = WBA_core.transform_nd_torus(points) #FIXME: this is not ideal!
    x, y, z, w = WBA_core.sort_by_extent(points, thresh)
    phi1 = np.arctan2(x, z) / (2*np.pi)
    phi1Diff = phi1[1:] - phi1[:-1]
    phi1Diff = (phi1Diff + 0.5) % 1.0 - 0.5
    if np.any(phi1Diff > 0.25) and np.any(phi1Diff < -0.25):
        phi1Diff %= 1.0
    phi2 = np.arctan2(y, w) / (2*np.pi)
    phi2Diff = phi2[1:] - phi2[:-1]
    phi2Diff = (phi2Diff + 0.5) % 1.0 - 0.5
    if np.any(phi2Diff > 0.25) and np.any(phi2Diff < -0.25):
        phi2Diff %= 1.0
    weights = WBA_core._weights(N + 1)
    freq1p1 = np.sum(weights * phi1Diff[:N], dtype=np.float64)
    freq2p1 = np.sum(weights * phi2Diff[:N], dtype=np.float64)
    freq1p2 = np.sum(weights * phi1Diff[N-1:], dtype=np.float64)
    freq2p2 = np.sum(weights * phi2Diff[N-1:], dtype=np.float64)
    return freq1p1, freq2p1, freq1p2, freq2p2

@njit
def _WBA_torus4d_parallel_fast(points, N, thresh):
    length = len(points[0, 0, :])
    freq = np.zeros((4, length), dtype=np.float64)
    for i in range(length):
        freq[:, i] = _WBA_torus4d_single_parallel(points[:, :, i], N, thresh)
    return np.abs(freq)

# from std_map import map4dnjit2multi
@njit(nogil=True)
def freq4d_grid_random(minval, maxval, counts, Npoints, 
                       k1, k2, k, thresh=1e-5):
    p1min, p2min, q1min, q2min = minval
    p1max, p2max, q1max, q2max = maxval
    p1count, p2count, q1count, q2count = counts
    chunk = p1count * p2count
    totalSize = np.prod(counts)
    freq = np.zeros((4, totalSize), dtype=np.float64)
    print("Torus4d transform with random initial conditions ...")
    _freq = _WBA_torus4d_parallel_fast
    # _freq = _Naff_multi   #does this even work? (2 instead of 4 freqs..)
    
    for ctr in range(0, totalSize, chunk):
        p10 = np.random.uniform(p1min, p1max, chunk)
        p20 = np.random.uniform(p2min, p2max, chunk)
        q10 = np.random.uniform(q1min, q1max, chunk)
        q20 = np.random.uniform(q2min, q2max, chunk)
        print("Step ", ctr, " of ", totalSize)
        # tMap = time.perf_counter()
        # p1, p2, q1, q2 = map4dnjit2multi(p10, p20, q10, q20,
        #                                  2*Npoints, k1, k2, k)
        points = map4dnjit2multi(p10, p20, q10, q20, 2*Npoints, k1, k2, k)
        points[2:, :, :] -= 0.5
        # tFreq = time.perf_counter()
        # print("Orbits in ", tFreq - tMap)
        # q1 -= 0.5
        # q2 -= 0.5
        # freq[:, ctr:ctr+chunk] = _freq(np.array([p1, p2, q1, q2]), 
        #                                 Npoints, thresh)
        freq[:, ctr:ctr+chunk] = _freq(points, Npoints, thresh)
        # print("Frequencies in ", time.perf_counter() - tFreq)
    return freq

from FreqSpace import map4dnjit3multi, _WBA_torus4d_fast
def freq4d_grid_given_initials(inits, Npoints, k1, k2, k, thresh=0.005):
    chunk = 1000
    length = len(inits[0])
    freq = np.zeros((4, length))
    for ctr in range(0, length, chunk):
        new_idx = min(ctr+chunk, length)
        points = map4dnjit3multi(inits[0, ctr:new_idx], 
                                 inits[1, ctr:new_idx],
                                 inits[2, ctr:new_idx],
                                 inits[3, ctr:new_idx], 
                                 2*Npoints, k1, k2, k)
        # freq[:, ctr:new_idx] = \
        #     _WBA_torus4d_parallel_fast(points, Npoints, thresh)
        freq[:, ctr:new_idx] = _WBA_torus4d_fast(points, Npoints)
        points = None
    return freq

# @njit
def freq4d_grid(minval, maxval, counts, Npoints, k1, k2, k, thresh=1e-5):
    """
    Calculates nu1 and nu2 for evenly spaced initial conditions in a 4d grid.
    To test in IPython console:
        minval=np.array([-0.5,-0.5,0.0,0.0])
        maxval=np.array([0.5,0.5,1.0,1.0])
        counts=np.array([10,12,3,4])
        Npoints=1024
        k1,k2,k=0.5,0.7,0.1
        
        freq4d_grid(minval, maxval, counts, Npoints, k1, k2, k)
    """
    p1min, p2min, q1min, q2min = minval
    p1max, p2max, q1max, q2max = maxval
    p1count, p2count, q1count, q2count = counts
    
    p10 = np.linspace(p1min, p1max, p1count)
    p20 = np.linspace(p2min, p2max, p2count)
    q10 = np.linspace(q1min, q1max, q1count)
    q20 = np.linspace(q2min, q2max, q2count)
    
    p10mesh = np.outer(p10, np.ones(p2count)).flatten()
    p20mesh = np.outer(np.ones(p1count), p20).flatten()
    # q10mesh = np.outer(q10, np.ones(q2count)).flatten()
    # q20mesh = np.outer(np.ones(q1count), q20).flatten()
    
    # # determine chunk size from available memory; FIXME: experimental!
    # maxMemory = 10**8
    # chunkSize = maxMemory // Npoints
    chunk = p1count * p2count
    freq1part1 = np.zeros(np.prod(counts), dtype=np.float64)
    freq1part2 = np.zeros(np.prod(counts), dtype=np.float64)
    freq2part1 = np.zeros(np.prod(counts), dtype=np.float64)
    freq2part2 = np.zeros(np.prod(counts), dtype=np.float64)
    
    if thresh > 0:
        print("Torus4d transform ...")
        _freq = WBA_core._WBA_torus4d_multi
        # _freq = _Naff_multi
        
        ctr = 0
        for i in range(q1count):
            q1val = np.full(chunk, q10[i])
            for j in range(q2count):
                q2val = np.full(chunk, q20[j])
                print("Step ", j + i*q2count + 1, " of ", q1count * q2count)
                tMap=time.perf_counter()
                p1, p2, q1, q2 = map4dnjit2multi(p10mesh, p20mesh, 
                                                 q1val, q2val,
                                                 2*Npoints, k1, k2, k)
                tFreq = time.perf_counter()
                print("Orbits in ", tFreq - tMap)
                q1 -= 0.5
                q2 -= 0.5
                
                f1 = _freq(np.array([p1[:Npoints], p2[:Npoints],
                                      q1[:Npoints], q2[:Npoints]]), thresh)
                f11, f21 = f1[0], f1[1]
                f2 = _freq(np.array([p1[Npoints:], p2[Npoints:],
                                      q1[Npoints:], q2[Npoints:]]), thresh)
                f12, f22 = f2[0], f2[1]
                # fig,ax=plt.subplots(1,2)
                # ax[0].scatter(q1,p1,s=2)
                # ax[1].scatter(q2,p2,s=2)
                # print(_freq(np.array([p1,p2,q1,q2]), 0.005))
                # f11 = _freq(q1[:Npoints], p1[:Npoints], MapToCircle=0)
                # f21 = _freq(q2[:Npoints], p2[:Npoints], MapToCircle=0)
                # f12 = _freq(q1[Npoints:], p1[Npoints:], MapToCircle=0)
                # f22 = _freq(q2[Npoints:], p2[Npoints:], MapToCircle=0)
                print("Frequencies in ", time.perf_counter() - tFreq)
                freq1part1[ctr:ctr+chunk] = f11
                freq2part1[ctr:ctr+chunk] = f21
                freq1part2[ctr:ctr+chunk] = f12
                freq2part2[ctr:ctr+chunk] = f22
                ctr += chunk
    # if q1count + q2count == 2:
    #     q1val, q2val = np.full(chunk, q10[0]), np.full(chunk, q20[0])
    #     # print(q10[0], q20[0], p10, p20)
    #     p1, p2, q1, q2 = map4dnjit2multi(p10mesh, p20mesh, q1val, q2val,
    #                                         2*Npoints, k1, k2, k)
    #     q1 -= 0.5
    #     q2 -= 0.5
    #     freq1 = _freq_WBA(np.array([p1[:Npoints], p2[:Npoints],
    #                                 q1[:Npoints], q2[:Npoints]]), thresh)
    #     freq2 = _freq_WBA(np.array([p1[Npoints:], p2[Npoints:],
    #                                 q1[Npoints:], q2[Npoints:]]), thresh)
    #     return freq1, freq2#, p1, p2, q1, q2
    
    else:
        _freq = WBA_core._WBA_multi_parallel
        ctr = 0
        for i in range(q1count):
            q1val = np.full(chunk, q10[i])
            for j in range(q2count):
                q2val = np.full(chunk, q20[j])
                p1, p2, q1, q2 = map4dnjit2multi(p10mesh, p20mesh, 
                                                 q1val, q2val,
                                                 2*Npoints, k1, k2, k)
                freq1part1[ctr:ctr+chunk] = _freq(p1[:Npoints])
                freq2part1[ctr:ctr+chunk] = _freq(p2[:Npoints])
                freq1part2[ctr:ctr+chunk] = _freq(p1[Npoints:])
                freq2part2[ctr:ctr+chunk] = _freq(p2[Npoints:])
                ctr += chunk
    return freq1part1, freq2part1, freq1part2, freq2part2

###############################################################################
# Tools for comparison and visualization
###############################################################################

def imshow_grid(ax, x, y, grid, Nxlabels=5, Nylabels=5, cmap='viridis_r', 
                interpolation='None', origin='lower', dig=3, norm=None):
    img = ax.imshow(grid, cmap=cmap, interpolation=interpolation, 
                    origin=origin, norm=norm)
    #Nxlabels=10; Nylabels=15
    nx = x.shape[0]
    x_step = int(nx / (Nxlabels - 1))
    x_positions = np.round(np.arange(0, nx, x_step), dig)
    x_labels = np.round(x[::x_step], dig)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    
    ny = y.shape[0]
    y_step = int(ny / (Nylabels - 1)) 
    y_positions = np.round(np.arange(0, ny, y_step), dig)
    y_labels = np.round(y[::y_step], dig)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    return img

def single_test(q0, p0, N, K, MapToCircle=1):
    t1 = time.perf_counter()
    q, p = _std_map(q0, p0, N, K)
    t2 = time.perf_counter()
    nu_WBA = WBA_core.WBA(q, p) % 1.0
    t3 = time.perf_counter()
    nu_Naff = compute_freq(q, p, MapToCircle)
    t4 = time.perf_counter()
    dt_map_ms = (t2 - t1) * 1e3
    dt_frq_WBA_ms = (t3 - t2) * 1e3
    dt_frq_Naff_ms = (t4 - t3) * 1e3
    print(f"Standard map with N={N} points in {round(dt_map_ms, 3)} ms")
    print(f"Parameters (x,y)={(q0,p0)} with K={K}:")
    print(f"    WBA: nu={nu_WBA} in {round(dt_frq_WBA_ms, 3)} ms")
    print(f"    Naff: nu={nu_Naff} in {round(dt_frq_Naff_ms, 3)} ms")
    return

# @njit
def _search_orbit_do_iteration(f, f0, p0, q0arr, N, K, stepSize):
    p0arr = np.linspace(p0 - f0 + f, p0, stepSize + 1)[:-1]
    q, p = _std_map_multi(q0arr, p0arr, N, K)
    f0arr = WBA_core._WBA_multi(p) % 1.0
        
    indx = np.argmin(np.abs(f0arr - f))
    f0 = f0arr[indx]
    p0 = p0arr[indx]
    absDiff = np.abs(f - f0)
    return f0, p0, absDiff

# @njit
def _search_orbit_from_freq(f, q0, p0, eps, Npow, maxNpow, K, 
                            stepSize, maxDepth, maxAbsDiffCounter):
    N = 2**Npow
    q, p = _std_map(q0, p0, N, K)
    f0 = WBA_core._WBA_single(p) % 1.0
    absDiff = np.abs(f - f0)
    q0arr = np.full(stepSize, q0, dtype=np.float64)
    absDiffCounter = 0
    for counter in range(maxDepth):
        f0, p0, absDiff = _search_orbit_do_iteration(f, f0, p0, q0arr, 
                                                     N, K, stepSize)
        if absDiff < eps:
            absDiffCounter += 1
            if absDiffCounter == maxAbsDiffCounter:
                break
        
        # if Npow <= maxNpow and Npow < -np.log10(absDiff):
        #     Npow += 1
        #     N *= 2
        
    if absDiffCounter < maxAbsDiffCounter:
        print("Warning, maximum search depth ", maxDepth, "reached!\n",  
              "final Orbit length was ", N)
        
    N = 2**maxNpow
    for i in range(maxAbsDiffCounter):
        f0, p0, absDiff = _search_orbit_do_iteration(f, f0, p0, q0arr, 
                                                     N, K, stepSize)
    return p0, absDiff

def search_orbit_from_freq(f, K, eps=1e-12, Npow=14, q0=None, p0=None,
                           maxDepth=100, stepSize=10, 
                           RetEps=0, maxNpow=16, maxAbsDiffCounter=5):
    if q0 == None: q0 = f
    if p0 == None: p0 = f - 0.5
        
    p0, absDiff = _search_orbit_from_freq(f, q0, p0, eps, Npow, maxNpow, K,
                                          stepSize, maxDepth, 
                                          maxAbsDiffCounter)
    if RetEps:
        return q0, p0, absDiff
    return q0, p0

def conv(ax, q, p, Narr, _dig=3, freq=None, thresh=1e-5, mapMode='none'):
    t1 = time.perf_counter()
    freq_WBA = WBA_core.WBA(q, p, Narr, thresh=thresh, mapMode=mapMode)
    t2 = time.perf_counter()
        
    if freq == None:
        freq = freq_WBA[-1]
        
    indx = (np.abs(freq - freq_WBA) > np.abs(1 - freq - freq_WBA))
    freq_WBA[indx] = 1 - freq_WBA[indx]
        
    WBA_diff = np.abs(freq - freq_WBA)
    WBA_diff[WBA_diff < 1e-16] = 1e-16
    
    ax.plot(Narr, WBA_diff, lw=1.5)
    ax.set_title(f"WBA in {round((t2-t1) * 1e3, 3)} ms \n" +
                  f"$\\nu_{{WBA}} = {round(freq_WBA[-1], _dig)}$", 
                  fontsize=14)
    return freq_WBA[-1]

def compare_conv(ax, q, p, Narr, _dig=16, freq=None, c=None, ShowLegend=1,
                  MapToCircle=1, thresh=1e-5, mapMode='none', NaffLimit=1, 
                  lw=1.5, lfs=16, SetTitle=1, freqLabel=None, 
                  UseMarkers=None, alphaNaff=1.0, AssertNaffEqualWBA=0):
    # print("MapToCircle", MapToCircle)
    t1 = time.perf_counter()
    freq_Naff = _Naff(Narr, q, p, MapToCircle)
    t2 = time.perf_counter()
    freq_WBA = WBA_core.WBA(q, p, Narr, thresh=thresh, mapMode=mapMode)
    t3 = time.perf_counter()
    if mapMode == 'none':
        freq_WBA %= 1.0
    elif mapMode == 'arctan2':
        freq_WBA = np.abs(freq_WBA)
    
    if Narr[-1] > 1.5*Narr[-2] and len(Narr) > 49:
        if NaffLimit:
            freq = freq_Naff[-1]
            print(f"Using Naff for {Narr[-1]} as true frequency {freq}.")
        else:
            freq = freq_WBA[-1]
            print(f"Using WBA for {Narr[-1]} as true frequency {freq}.")
    if freq == None:
        freq_WBA_limit = freq_WBA[-1]
        freq_Naff_limit = freq_Naff[-1]
    else:
        freq_WBA_limit = freq
        freq_Naff_limit = freq
        
    indxWBA = (np.abs(freq_WBA_limit - freq_WBA) > 
                np.abs(1 - freq_WBA - freq_WBA_limit))
    indxNaff = (np.abs(freq_Naff_limit - freq_Naff) >
                np.abs(1 - freq_Naff - freq_Naff_limit))
    freq_WBA[indxWBA] = 1 - freq_WBA[indxWBA]
    freq_Naff[indxNaff] = 1 - freq_Naff[indxNaff]
    if AssertNaffEqualWBA:
        indx = (np.abs(freq_Naff - freq_WBA) > 
                np.abs(1 - freq_Naff - freq_WBA))
        freq_Naff[indx] = 1 - freq_Naff[indx]
        if indx[-1]:
            freq_Naff_limit = 1 - freq_Naff_limit
        
    WBA_diff = np.abs(freq_WBA_limit - freq_WBA)
    Naff_diff = np.abs(freq_Naff_limit - freq_Naff)
    
    WBA_diff[WBA_diff < 1e-16] = 1e-16
    Naff_diff[Naff_diff < 1e-16] = 1e-16
    
    title = (f"Latest WBA in {round((t3-t2) * 1e3, 3)} ms and " +
             f"Latest Naff in {round((t2-t1) * 1e3, 3)} ms\n")
    labelWBA = r'$\nu_{\mathrm{WBA}}$ = ' + str(round(freq_WBA[-1], _dig))
    labelNaff = r'$\nu_{\mathrm{Naff}}$ = ' + str(round(freq_Naff[-1], _dig))
    title += labelWBA + " and " + labelNaff
    
    if freq != None:
        labelNaff = None
        if type(freqLabel) == type(None):
            labelWBA = f'$\\nu = {round(freq, _dig)}$'
        else:
            labelWBA = f'$\\nu = {freqLabel}$'
        
    if type(c) != type(None):
        if type(UseMarkers) != type(None):
            marker, ms, mew = UseMarkers
            line = ax.plot(Narr[:-1], WBA_diff[:-1], 
                           marker=marker, ms=ms, mew=mew, 
                           lw=lw, label=labelWBA, c=c)
        else:
            line = ax.plot(Narr[:-1], WBA_diff[:-1], 
                           lw=lw, label=labelWBA, c=c)
    else:
        line = ax.plot(Narr[:-1], WBA_diff[:-1],
                       lw=lw, label=labelWBA)
    ax.plot(Narr[:-1], Naff_diff[:-1], alpha=alphaNaff, 
            lw=lw, ls='--', c=line[0].get_color(), label=labelNaff)
    if SetTitle:
        ax.set_title(title, fontsize=14)
    
    if ShowLegend:
        ax.legend(fontsize=lfs)
    return freq_WBA[-1], freq_Naff[-1]

def _action_cmpr_conv(ax, qpos, ppos, K, Npoints, Narr, args):
    thresh, MapToCircle, mapMode, NaffLimit = args
    q, p = _std_map(qpos, ppos, Npoints, K)
    ax[0].plot(q, (p + 0.5) % 1.0 - 0.5, marker='o', ls='', ms=2, mew=1)
    
    q, p = _std_map(qpos, ppos, np.max(Narr), K)
    q -= 0.5    # for arctan2 mapping
    if K == 0.0:    # use theoretical frequency
        compare_conv(ax[1], q, p, Narr, freq=ppos % 1.0, thresh=thresh,
                     mapMode=mapMode, MapToCircle=MapToCircle)
    else:           # use estimate for true frequency
        compare_conv(ax[1], q, p, Narr, thresh=thresh,
                     mapMode=mapMode, MapToCircle=MapToCircle,
                     NaffLimit=NaffLimit)
    pass

def _mouse_click(event, ax, K, Npoints, Narr, _action, args=()):
    """
    _mouse_click takes arguments (event, ax, K, Npoints, Narr, _action, args)
    _action takes arguments (ax, qpos, ppos, K, Npoints, Narr, *args)
    """
    mode = event.canvas.toolbar.mode
    if event.inaxes == ax[0] and mode == '':
        if event.button == 1:
            qpos, ppos = event.xdata, event.ydata
            
            _action(ax, qpos, ppos, K, Npoints, Narr, args)
        elif event.button == 3:
            ax[0].lines = []
            
        event.canvas.draw() 
        
    if event.button == 3 and event.inaxes == ax[1] and mode == '':
        ax[1].lines = []
        event.canvas.draw()
    pass

def interactive_plot(K=1.0, Npoints=200, Nmin=5.0, Nmax=16.0, NN=100,
                     z0arr=None, thresh=1e-5, NmaxLimit=None, 
                     mapMode='none', MapToCircle=1, NaffLimit=1, 
                     axis=None):
    Narr = N_arr(Nmin, Nmax, NN)
    if NmaxLimit != None:
        Narr = np.append(Narr, np.uint32(2**NmaxLimit))
    
    _dict = {0: {'xlabel' : 'q_n', 'ylabel' : 'p_n', 
                  'title' : f"K={K} and {Npoints} points"},
              1: {'xlabel' : 'N', 'ylabel' : r"$|\nu_{N_{max}} - \nu_{N}|$", 
                  'title' : ""}}
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].axis([0, 1, -0.5, 0.5])
    if type(axis) != type(None):
        ax[0].axis(axis)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    for i in range(len(ax)):
        ax[i].set_title(_dict[i]['title'], fontsize=14)
        ax[i].set_xlabel(_dict[i]['xlabel'], fontsize=14)
        ax[i].set_ylabel(_dict[i]['ylabel'], fontsize=14)
        
    if type(z0arr) != type(None):
        for z0 in z0arr:
            q0arr, p0arr = _std_map(*z0, Npoints, K)
            ax[0].plot(q0arr, (p0arr + 0.5) % 1.0 - 0.5, 
                       marker='o', ls='', ms=2, mew=1)
            
            q0arr, p0arr = _std_map(*z0, np.max(Narr), K)
            if K == 0.0:    # use theoretical frequency
                compare_conv(ax[1], q0arr-0.5, p0arr, Narr, freq=z0[1] % 1.0, 
                             mapMode=mapMode)
            else:           # use estimate for true frequency
                compare_conv(ax[1], q0arr-0.5, p0arr, Narr, mapMode=mapMode,
                             NaffLimit=NaffLimit)
            
    mouse_click = functools.partial(_mouse_click, ax=ax, K=K, 
                                    args=(thresh, MapToCircle, mapMode,
                                          NaffLimit), 
                                    Npoints=Npoints, Narr=Narr, 
                                    _action=_action_cmpr_conv)
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    plt.show()    
    return


if __name__ == "__main__":
    """
    Interesting values: (x, y, K)
    (0.0, 0.645, -0.5) -> 2/3
    (0.0, 0.215, 0.5) -> 1/4
    22.03: 3x speed performance, seemingly fast conv.
    23.03: Convergence comparison, WBA for T-array.
    24.03: Unfolding closed orbits, conv. for noised freq.
    25.03: Interactive Plot for 2f-signal, 
            chaos indicator (cos works much better, reason FIXME)
    26.03: Fix: Naff 2f-signal analysis (disabled mapping)
    27.03: Fix: with wrong frequency WBA for wrapped signals in 'p'
    28.03: Fix: WBA 2f-signal analysis (again signal wrapping in 'p')
    29.03: Documentation for test functions
    30.03: 
    31.03: Heatmap for chaos indication and 2f-signals. 
            Arctan2 test for oscillatory circles and stretched 2f-signals.
    """
    print(__doc__)
    # N = 2**14
    # Narr = N_arr(5.0, 14.0, 200)
    q0 = 0.1
    p0 = [1.5-np.sqrt(5/4), np.sqrt(2)-1, np.sqrt(13/4)-2.5, 1/3, 3/10, -0.25]
    # p0 = [0.3, 0.49, 0.41, -0.35, -0.2]
    
    # # rot. circles for 0.7, 0.387 for period 7 orbit
    # K = 0.7
    # q0 = 1-0.06     # 1-x for changed map
    # p0golden = search_orbit_from_freq(1.5-np.sqrt(1.25), K,q0=q0)[1]
    # p0 = [0.18, 0.28, 0.45, -0.189, p0golden]  
    # z0arr = [[q0, p] for p in p0]
    # interactive_plot(K, z0arr=z0arr, Nmin=5.0, Nmax=14.0, Npoints=1000, 
    #                   mapMode='none', NmaxLimit=16, NaffLimit=0)
     
    # osc. circles for K=0.7
    K = 0.7
    q0 = 1-0.4                                    
    p0 = [-0.23, -0.21, -0.17, -0.1, 0.19]  
    z0arr = [[q0, p] for p in p0]
    # interactive_plot(K, z0arr=z0arr, Nmax=14.0, Npoints=1000, NmaxLimit=20,
    #                   MapToCircle=0, mapMode='arctan2', NaffLimit=0)
    # z0 = [[0.04874372, -0.18734367]]
    # z0 = [[0.04874371859296483, -0.187343671835918]]
    # interactive_plot(2.32, 10**3, Nmax=6.0,NN=2,z0arr=z0)
    
    # print(search_orbit_from_freq(1.5-np.sqrt(1.25), K,q0=q0,RetEps=1))
    
    def fconv(q0,p0,K=0.7,nmin=5.0,nmax=18.0,nn=200,key='naff'):
        fig, ax = plt.subplots(1,2,figsize=(15,10))
        Narr=N_arr(nmin,nmax,nn)
        q,p=_std_map(q0,p0,np.max(Narr),K)
        if key == 'naff': fn = _Naff(Narr,q,p)
        if key == 'wba': fn = WBA_core.WBA(q,p,Narr)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].plot(Narr, np.abs(fn - fn[-1]))
        ax[1].plot(q,p,ls='',marker='o',ms=2,mew=1)
        plt.show()
        return
    
    # phi = 1 + np.sqrt(5)
    # f = np.arange(phi/600, 1.0, phi/300)
    # precompile numba functions to reduce overhead for proper benchmarks
    # q, p = _std_map(q0, p0, N, K)      
    # WBA_core._WBA_single(np.array([0.0, 0.3]))
    # single_test(q0, p0, N, K, UseQdiff=0)
    
