"""
Implementations of Weigthed Birkhoff Averages (WBA) 
for various dimensionalities and usecases.
"""

import numpy as np
from std_map import _std_map, _std_map_multi
from numba import njit

@njit('Tuple((f8[:], f8))(u4)', cache=True)
def _weights(N):
    n = np.arange(1, N, 1, dtype=np.float64) / N
    # _gnN = np.exp( -(n * (1 - n))**(-1))
    _gnN = np.exp( -140 * (n - 0.5)**2)
    _sum = np.sum(_gnN, dtype=np.float64)
    return _gnN, _sum

@njit#('f8(f8[:])', cache=True)
def _WBA_single(arr):
    N = len(arr)
    weigths, _sum = _weights(N + 1)
    return np.sum(weigths * arr) / _sum

# @njit#('f8[:](u4[:], f8[:])', cache=True)
def _WBA(N, arr):
    freq = np.zeros(len(N), dtype=np.float64)    
    for i in range(len(N)):
        freq[i] = _WBA_single(arr[:N[i]])
    return freq

@njit('f8[:](f8[:,:])', cache=True)
def _WBA_multi(arr):
    freq = np.zeros(len(arr[0]), dtype=np.float64)    
    for i in range(len(arr[0])):
        freq[i] = _WBA_single(arr[:, i])
    return freq

# @njit
def inertia_tensor_nd(r, ndim=4):
    return (np.sum(r*r) * np.eye(ndim) - r @ r.T) / len(r[0])

@njit
def transform_nd_torus(r, ndim=4):
    r = np.ascontiguousarray(r)
    iTensor = (np.sum(r*r) * np.eye(ndim) - np.dot(r, r.T)) / len(r[0]) 
    eigVal, eigVec = np.linalg.eigh(iTensor)
    return np.dot(eigVec.T, r)  # eigVec.T @ r
# @njit
# def transform_nd_torus(rOld, ndim=4):
#     r = np.ascontiguousarray(rOld[:, ::10])#, dtype=np.float64)
#     #iTensor = inertia_tensor_nd(r, ndim)
#     iTensor = (np.sum(r*r) * np.eye(ndim) - np.dot(r, r.T)) / len(r[0]) 
#     eigVal, eigVec = np.linalg.eigh(iTensor)
#     # print(np.linalg.det(eigVec))    #TODO: remove this later
#     return np.dot(eigVec.T, rOld)  # eigVec.T @ r
    
@njit
def special_dyad_product(a, ndim):
    aLen = a.shape[1]
    weights, _sum = _weights(aLen + 1)
    res = np.zeros((ndim, ndim), dtype=np.float64)
    for i in range(0, ndim, 1):
        for j in range(i, ndim, 1):
            res[i, j] = np.sum(a[i, :] * a[j, :] * weights) / _sum
            res[j, i] = res[i, j]
    return res

@njit
def WBA_transform_nd_torus(r, ndim=4):
    r = np.ascontiguousarray(r)
    # iTensor = (np.sum(_WBA_multi(r*r)) * np.eye(ndim) / ndim
    #             - special_dyad_product(r, ndim) )
    # iTensor = special_dyad_product(r, ndim)
    # iTensor = np.eye(ndim)
    iTensor = (np.sum(r*r) * np.eye(ndim) - np.dot(r, r.T)) / len(r[0]) 
    eigVal, eigVec = np.linalg.eigh(iTensor)   
    return np.dot(eigVec.T, r)

# @njit#('UniTuple(f8[:], 2)(f8[:], f8[:])', cache=True)
def map_arctan2(q, p):
    """
    Maps a given set of (q, p) points to polar coordinates (phi, r).
    """    
    rSqr = 1#q**2 + p**2  # FIXME: performance issues
    phi = np.arctan2(q, p) / (2*np.pi)
    return phi, rSqr

# @njit
def _WBA_multi_parallel(arr):
    N = len(arr[:, 0])
    weigths, _sum = _weights(N)
    return np.sum(weigths * arr[1:, :].T, axis=1, dtype=np.float64) / _sum

# @njit
def _WBA_multi_arctan2_parallel(q, p):
    N = len(p[:, 0])
    weigths, _sum = _weights(N)
    phi = np.arctan2(q, p) / (2*np.pi)
    phiDiff = (phi[1:, :] - phi[:-1, :] + 0.5) % 1.0 - 0.5
    return np.sum(weigths * phiDiff.T, axis=1, dtype=np.float64) / _sum

# @njit
def _WBA_multi_parallel_wrapper(q, p, mapMode):
    if mapMode == 'none': return _WBA_multi_parallel(p)
    if mapMode == 'arctan2': return _WBA_multi_arctan2_parallel(q, p)
    
### EXPERIMENTAL
@njit
def embedding2(arr):
    N = arr.shape[0]
    for i in range(1, N, 1):
        delta = arr[i] - arr[i-1]
        if np.abs(delta) > np.abs(delta + 1):
            arr[i] += 1
    delta = arr[0] - arr[-1]
    if np.abs(delta) > np.abs(delta + 1):
        arr[0] += 1
        
@njit
def embedding3(arr):
    N = arr.shape[0]
    arr = np.sort(arr)
    # shift = np.zeros(N)
    # deltaArr = arr[1:] - arr[:-1]
    for i in range(1, N, 1):
        delta = arr[i] - arr[i-1]
        if delta > 0.5: arr[i] -= 1
        elif delta < -0.5: arr[i] += 1
      
@njit
def embedding(arr):
    # seg = np.round(arr - arr[0], 0)
    # return arr - seg
    for i in range(1, arr.shape[0], 1):
        delta = arr[i] - arr[0]
        if delta > 0.5: arr[i] -= 1
        elif delta < -0.5: arr[i] += 1

@njit#('f8(f8[:], f8[:])', cache=True)
def _WBA_single_arctan2(q, p):
    # phi, r = map_arctan2(q, p)
    phi = np.arctan2(p, q) / (2*np.pi)
    phiDiff = phi[1:] - phi[:-1]
    # #this is all the same problem, just shifts the problematic region
    # #0.5 does not matter here, tried anything from 0.0 to 0.5
    # #however, the 0.25 below is very important! But how to explain this?
    embedding(phiDiff)
    # following three lines are a good solution
    # phiDiff = (phiDiff + 0.5) % 1.0 - 0.5
    # if np.any(phiDiff > 0.25) and np.any(phiDiff < -0.25):
    #     phiDiff %= 1.0
    # alpha = 0.5
    
    # phiDiff = (phiDiff + alpha) % 1.0 - alpha
    # if np.any(phiDiff > 0.75-alpha) and np.any(phiDiff < 0.25-alpha):
    #     phiDiff[phiDiff < 0.5 - alpha] += 1.0
    # #variant with 0.5 -> 0.25 but only one 'any' condition has small region
    # if np.any(phiDiff < 0.25) and np.any(phiDiff > 0.75):
    #     phiDiff[phiDiff < 0.5] += 1
    return np.abs(_WBA_single(phiDiff)) #FIXME: added np.abs on 11.07.2021

# @njit#('f8[:](u4[:], f8[:], f8[:])', cache=True)
def _WBA_arctan2(N, q, p):
    freq = np.zeros(len(N), dtype=np.float64)    
    for i in range(len(N)):
        freq[i] = _WBA_single_arctan2(q[:N[i]], p[:N[i]])
    return freq

# @njit#('f8[:](f8[:,:], f8[:,:])', cache=True)
def _WBA_multi_arctan2(q, p):
    freq = np.zeros(len(q[0]), dtype=np.float64)
    for i in range(len(q[0])):
        freq[i] = _WBA_single_arctan2(q[:, i], p[:, i])
    return freq

##### EXPERIMENTAL START
def _WBA_fourier_single(q, p):
    freq1 = _WBA_single_arctan2(q, p)
    n_arr = np.arange(q.shape[0])
    qf1 = q - np.cos(2*np.pi * freq1 * n_arr)
    pf1 = p - np.sin(2*np.pi * freq1 * n_arr)
    freq2 = _WBA_single_arctan2(qf1, pf1)
    return freq1, freq2

def fourier_coeff(freq, z_n, n_coef):
    size = z_n.real.shape[0]
    n_arr = np.arange(size)
    weights, _sum = _weights(size + 1)
    a_k = np.zeros(n_coef, dtype=np.complex128)
    for k in range(1, n_coef + 1, 1):
        a_k[k-1] = np.sum(z_n * np.exp(-2*np.pi*1j*k*freq*n_arr) * weights)
    return a_k / _sum

def fourier_series(freq, a_k, N):
    n_arr = np.arange(N)
    result = np.zeros(N, dtype=np.complex128)
    for k in range(a_k.shape[0]):
        result += a_k[k] * np.exp(2*np.pi*1j*freq*k*n_arr)
        # result += a_k[k].conjugate()
    return result
##### EXPERIMENTAL END

# @njit#('f8[:](u4[:], f8[:], f8[:], f8)', cache=True)
def _WBA_decision(N, q, p, thresh):
    freq = np.zeros(len(N), dtype=np.float64)    
    for i in range(len(N)):
        freq[i] = _WBA_single(p[:N[i]])
        if np.abs(freq[i]) < thresh:
            freq[i] = _WBA_single_arctan2(q[:N[i]], p[:N[i]])
    return freq

# @njit#('f8[:](f8[:,:], f8[:,:], f8)', cache=True)
def _WBA_multi_decision(q, p, thresh):
    freq = np.zeros(len(q[0]), dtype=np.float64)
    for i in range(len(q[0])):
        freq[i] = _WBA_single(p[:, i])
        if np.abs(freq[i]) < thresh:
            freq[i] = _WBA_single_arctan2(q[:, i], p[:, i])
    return freq

# @njit#(cache=True)
def _WBA_single_wrapper(q, p, mapMode):
    if mapMode == 'none': freq = _WBA_single(p)
    if mapMode == 'arctan2': freq = _WBA_single_arctan2(q, p)
    if mapMode == 'decision': raise TypeError
    return freq

# @njit(cache=True)
def _WBA_wrapper(N, q, p, thresh, mapMode):
    if mapMode == 'none': freq = _WBA(N, p)
    if mapMode == 'arctan2': freq = _WBA_arctan2(N, q, p)
    if mapMode == 'decision': freq = _WBA_decision(N, q, p, thresh)
    return freq

# @njit(cache=True)
def _WBA_multi_wrapper(q, p, thresh, mapMode):
    if mapMode == 'none': freq = _WBA_multi(p)
    if mapMode == 'arctan2': freq = _WBA_multi_arctan2(q, p)
    if mapMode == 'decision': freq = _WBA_multi_decision(q, p, thresh)
    return freq

def WBA(q0, p0, N=None, K=None, thresh=None, mapMode='none'):
    """
    WBA for the following input types:
    If 'K' is given, caluculate (q, p) signals from the standard map.
        Accepts a or an array pair of initial values. If 'N' is an array, 
        (q0, p0) should not be arrays (not implemented, will raise Error!).
        
    If 'K' is not given, calculate the WBA from the given array of 'p0'.
        If 'N' is an array, the WBA is calculate for each element of 'N'.    
    """
    mapMode_list = ['none', 'arctan2', 'decision']
    
    MaxNErrorMsg = "max(N) has to be less or equal to the length of 'p0'!"
    TypeErrorMsg = "No method for given types found!"
    modeListErrorMsg = ("'mapMode' has to be either one of ", mapMode_list,  
                        " but was '", mapMode, "'!")
    qpTypeErrorMsg = ("q0 and p0 have to of type 'float' (or array) but were " 
                      + "type(q0)=", type(q0), " and type(p0)=", type(p0))
    NTypeErrorMsg = ("N has to be a positive integer or an array of positive "
                     + "integers, but was of type ", type(N)) 
    
    # error handling due to incorrect input types
    if mapMode not in mapMode_list: 
        print(modeListErrorMsg)
        raise TypeError
        
    if mapMode != 'none' and thresh == None:
        thresh = 1e-5
        
    if (isinstance(q0, (np.float64, float)) and 
        isinstance(p0, (np.float64, float))): 
        qpMode = 'single'
    elif type(q0[0]) == np.float64 and type(p0[0]) == np.float64: 
        qpMode = 'multi'
    elif type(q0[0, 0]) == np.float64 and type(p0[0, 0]) == np.float64: 
        qpMode = 'multi 2d'
    else: 
        print(qpTypeErrorMsg)
        raise TypeError
        
    if type(N) == type(None): Ntype = 'none'
    elif isinstance(N, (np.uint32, int)) and N > 0: Ntype = 'int'
    elif type(N[0]) == np.uint32 and N[0] > 0: Ntype = 'array int'
    else: 
        print(NTypeErrorMsg)
        raise TypeError
    
    # case structure for calling methods corresponding to correct modes
    if type(K) == float:
        if qpMode == 'single':
            if Ntype == 'int':
                q, p = _std_map(q0, p0, N, K)
                return _WBA_single_wrapper(q - 0.5, p, mapMode)
            if Ntype == 'array int':
                q, p = _std_map(q0, p0, np.max(N), K)
                return _WBA_wrapper(N, q - 0.5, p, thresh, mapMode)

        if qpMode == 'multi':
            if Ntype == 'int':
                q, p = _std_map_multi(q0, p0, N, K)
                return _WBA_multi_wrapper(q - 0.5, p, thresh, mapMode)
            if Ntype == 'array int': 
                print(TypeErrorMsg)
                raise TypeError
    
    if type(K) == type(None):
        if qpMode == 'multi':
            if Ntype == 'none':
                return _WBA_single_wrapper(q0, p0, mapMode)
            if Ntype == 'array int':
                if np.max(N) > len(q0): 
                    print(MaxNErrorMsg)
                    raise IndexError
                return _WBA_wrapper(N, q0, p0, thresh, mapMode)
        if qpMode == 'multi 2d': 
            return _WBA_multi_wrapper(q0, p0, thresh, mapMode)           
            
    print(TypeErrorMsg)
    print("No Output produced. Raising error to help isolate the issue.")
    raise TypeError
    
# # @njit#deprecated
# def find_elliptic_sections(points, thresh=1e-3):
#     # r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
#     # ratio1 = np.min(r1sqr) / np.max(r1sqr)
#     # ratio2 = np.min(r2sqr) / np.max(r2sqr)
#     # # print("Ratios of squared radii are ", ratio1, ratio2)
#     # if ratio1 < thresh or ratio2 < thresh:
#     #     # print("Wrong mapping detected, reprojecting orbit...")
#     #     q1, p2 = p2, q1
#     #     r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
#     #     ratio1 = np.min(r1sqr) / np.max(r1sqr)
#     #     ratio2 = np.min(r2sqr) / np.max(r2sqr)
#     #     # print("Corrected ratios are ", ratio1, ratio2)
#     #     if ratio1 < thresh or ratio2 < thresh:
#     #         # print("Final attempt of reprojecting...")
#     #         q2, q1 = q1, q2
#     #         r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
#     #         ratio1 = np.min(r1sqr) / np.max(r1sqr)
#     #         ratio2 = np.min(r2sqr) / np.max(r2sqr)
#     # #         print("Final corrected ratios are ", ratio1, ratio2)
#     # # print()
#     p1,p2,q1,q2=points
#     if r_squared_ratio_test(points, thresh):
#         q1, p2 = p2, q1
#         if r_squared_ratio_test(points, thresh):
#             q2, q1 = q1, q2
#     return p1, p2, q1, q2

@njit
def r_squared_ratio_test(points, thresh):
    p1, p2, q1, q2 = points
    r1sqr = p1*p1 + q1*q1
    r1sqrMax = np.max(r1sqr)
    ratio1 = np.min(r1sqr) / r1sqrMax
    r2sqr = p2*p2 + q2*q2
    r2sqrMax = np.max(r2sqr)
    ratio2 = np.min(r2sqr) / r2sqrMax
    # print("Ratios of squared radii are ", ratio1, ratio2, thresh)
    return (ratio1 < thresh and ratio2 < thresh)# and r1sqrMax > thresh 
            #and r2sqrMax > thresh)

@njit
def swap_rows(arr, x, y):
    arr[np.array([x, y])] = arr[np.array([y, x])]
    pass

@njit
def sort_by_extent2(points, thresh):
    # pointsSort = np.ascontiguousarray(points[:, ::10])
    # # diff = np.max(points, axis=1) - np.min(points, axis=1)
    # length = len(points)
    # diff = np.zeros(length, dtype=np.float64)
    # for i in range(length):
    #     diff[i] = np.max(points[i]) - np.min(points[i])
    # indx = np.argsort(diff)
    # return points[indx]
    if r_squared_ratio_test(points, thresh):
        # q1, p2 = p2, q1
        # points[[2, 1], :] = points[[1, 2], :] 
        swap_rows(points, 2, 1)
        # swap_rows(pointsSort, 2, 1)
        if r_squared_ratio_test(points, thresh):
            # q2, q1 = q1, q2
            # points[[3, 2], :] = points[[2, 3], :]
            swap_rows(points, 3, 2)
    # p1, p2, q1, q2 = points
    # r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
    # ratio1 = np.min(r1sqr) / np.max(r1sqr)
    # ratio2 = np.min(r2sqr) / np.max(r2sqr)
    # print("Ratios of squared radii are ", ratio1, ratio2)
    # if ratio1 < thresh or ratio2 < thresh:
    #     print("Wrong mapping detected, reprojecting orbit...")
    #     q1, p2 = p2, q1
    #     r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
    #     ratio1 = np.min(r1sqr) / np.max(r1sqr)
    #     ratio2 = np.min(r2sqr) / np.max(r2sqr)
    #     print("Corrected ratios are ", ratio1, ratio2)
    #     if ratio1 < thresh or ratio2 < thresh:
    #         print("Final attempt of reprojecting...")
    #         q2, q1 = q1, q2
    #         r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
    #         ratio1 = np.min(r1sqr) / np.max(r1sqr)
    #         ratio2 = np.min(r2sqr) / np.max(r2sqr)
    #         print("Final corrected ratios are ", ratio1, ratio2)
    # print()
    return points#p1, p2, q1, q2

@njit 
def min_max_ratio(arr):
    return np.min(arr) / np.max(arr)

@njit
def sort_by_extent(points, thresh=-1):
    # if thresh > 0: print("using thresh is deprecated!")
    p1sqr, q1sqr = points[0, :100]**2, points[2, :100]**2
    p2sqr, q2sqr = points[1, :100]**2, points[3, :100]**2
    r11 = min_max_ratio(p1sqr + q1sqr) #proj 1
    r12 = min_max_ratio(p2sqr + q2sqr)
    r21 = min_max_ratio(p1sqr + p2sqr) #proj 2
    r22 = min_max_ratio(q1sqr + q2sqr)
    r31 = min_max_ratio(p1sqr + q2sqr) #proj 3
    r32 = min_max_ratio(q1sqr + p2sqr)
    ratios = np.array([r11, r12, r21, r22, r31, r32])
    argmax = np.argmax(ratios)
    if argmax > 3: swap_rows(points, 3, 2)
    elif argmax > 1: swap_rows(points, 2, 1)
    return points
    
@njit
def _WBA_torus4d_single(points, thresh):
    # print()
    # print("standard, ", points[:, 0])
    if r_squared_ratio_test(points, 1.0): #FIXME: is this good
        # print("transform, ", points[:, 0])
        points = transform_nd_torus(points)
    # p1, p2, q1, q2 = find_elliptic_sections(*points, thresh)
        x, y, z, w = sort_by_extent(points, thresh)
    else:
        y, w, x, z = points
    # points = transform_nd_torus(points)
    # x, y, z, w = sort_by_extent(points)
    # import matplotlib.pyplot as plt
    # fig,ax=plt.subplots(1,2)
    # ax[0].scatter(x,z,s=2)
    # ax[1].scatter(y,w,s=2)
    freq1 = _WBA_single_arctan2(x, z)
    freq2 = _WBA_single_arctan2(y, w)
    return freq1, freq2

@njit
def _WBA_torus4d_single_var(points, thresh=0.01):
    _freq = _WBA_single_arctan2
    freq1a = _freq(points[2, :], points[0, :])
    freq2a = _freq(points[3, :], points[1, :])
    x, y, z, w = transform_nd_torus(points)
    freq1b, freq2b = _freq(x, y), _freq(z, w)
    freq1c, freq2c = _freq(x, z), _freq(y, w)
    freq1d, freq2d = _freq(x, w), _freq(z, y)
    freq1 = np.abs(np.array([freq1a, freq1b, freq1c, freq1d]))
    freq2 = np.abs(np.array([freq2a, freq2b, freq2c, freq2d]))
    indx = ((freq1 > 0.5) & (freq2 > 0.5))
    freq1[indx], freq2[indx] = 1 - freq1[indx], 1 - freq2[indx]
    return np.max(freq1), np.max(freq2)

# @njit
def _WBA_torus4d(points, N, thresh):
    """Mean of all input arrays should be as close to zero as possible."""
    if r_squared_ratio_test(points, 1.0): #FIXME: is this good
        print("Transforming torus ...")
        points = transform_nd_torus(points)
        x, y, z, w = sort_by_extent(points, thresh)
    else:
        y, w, x, z = points
    # x, y, z, w = find_elliptic_sections(p1, p2, q1, q2, thresh)
    freq1 = _WBA_arctan2(N, z, x)
    freq2 = _WBA_arctan2(N, w, y)
    return freq1, freq2

@njit
def _WBA_torus4d_multi(points, thresh):
    length = len(points[0, 0, :])
    freq = np.zeros((2, length), dtype=np.float64)
    for i in range(length):
        freq[:, i] = _WBA_torus4d_single(points[:, :, i], thresh)
        # freq[:, i] = _WBA_torus4d_single_var(points[:, :, i], thresh)
    return np.abs(freq)

def WBA_torus4d(points, N=None, thresh=0.01):
    if type(N) == type(None):
        if len(np.shape(points)) == 2:
            return _WBA_torus4d_single(points, thresh)
        return _WBA_torus4d_multi(points, thresh)
    return _WBA_torus4d(points, N, thresh)

###############################################################################
# Fourier Series Method :: iteratively remove frequencies from a signal
###############################################################################

# @njit
def remove_freq(signal, freq, n_harm=1, n_min=1):
    n = np.arange(0, signal.shape[0], 1, dtype=np.int32)
    for i in range(n_min, n_harm+1, 1):
        phase = np.exp(-2*np.pi*1j * freq * i * n)
        ampl = _WBA_single(signal * phase)
        signal -= ampl * np.conjugate(phase)
    return signal

# @njit
def find_elliptic_projection(points, ndim=4):
    """Finds the elliptic projection
    Step 1: want to find polar radius for all combinations -> square points
    Step 2: there exist '1+2+...+(ndim-1) = ndim*(ndim-1)/2' projections
    """
    points_sqr = points**2                      # Step 1
    n_combinations = ndim * (ndim - 1) // 2     # Step 2
    ratios = np.zeros(n_combinations)    
    indxs = np.zeros((2, n_combinations), dtype=np.int32)
    ctr = 0
    for i in range(0, ndim, 1):
        q_sqr = points_sqr[i, :]
        for j in range(i+1, ndim, 1):
            p_sqr = points_sqr[j, :]
            ratios[ctr] = min_max_ratio(q_sqr + p_sqr)
            indxs[:, ctr] = [i, j]
            ctr += 1
    argmax = np.argmax(ratios)
    indx = indxs[:, argmax]
    return points[indx, :]
            
import matplotlib.pyplot as plt #FIXME
# @njit
def WBA_fsm_single(points, n_freq=2, n_harm=1, ndim=4):
    """WBA using the FourierSeriesMethod for a single ND-input signal"""
    points = transform_nd_torus(points, ndim)
    x, y = find_elliptic_projection(points, ndim)
    fig, ax = plt.subplots(figsize=(10,9))
    ax.scatter(x, y, s=3)
    freq = np.zeros(n_freq)
    freq[0] = _WBA_single_arctan2(x, y)
    z = x - 1j * y #FIXME
    for i in range(1, n_freq, 1):
        z = remove_freq(z, freq[i-1], n_harm) #FIXME
        ax.scatter(z.real, z.imag, s=2)
        freq[i] = _WBA_single_arctan2(z.real, z.imag)
    print(freq) #FIXME
    return freq

# @njit
def WBA_fsm(Narr, points, n_freq=2, n_harm=1, ndim=4):
    NN = Narr.shape[0]
    freq = np.zeros((NN, n_freq))
    for i in range(NN):
        freq[i, :] = WBA_fsm_single(points[:, :Narr[i]], n_freq, n_harm, ndim)
    return freq#[:, :2]

if __name__ == "__main__":
    print(__doc__)