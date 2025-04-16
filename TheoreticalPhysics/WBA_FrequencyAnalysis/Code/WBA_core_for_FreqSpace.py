"""
Implementations of Weigthed Birkhoff Averages (WBA) 
for various dimensionalities and usecases.
"""

import numpy as np
# from std_map import _std_map, _std_map_multi
from numba import njit

@njit('f8[:](u4)', cache=True)
def _weights(N):
    n = np.arange(1, N, 1, dtype=np.float64) / N
    _gnN = np.exp( -(n * (1 - n))**(-1))
    _sum = np.sum(_gnN, dtype=np.float64)
    return _gnN / _sum

@njit#('f8(f8[:])', cache=True)
def _WBA_single(arr):
    N = len(arr)
    weights = _weights(N + 1)
    return np.sum(weights * arr, dtype=np.float64)

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
    # r = np.ascontiguousarray(rOld[:, ::10])#, dtype=np.float64)
    #iTensor = inertia_tensor_nd(r, ndim)
    r = np.ascontiguousarray(r)
    iTensor = (np.sum(r*r) * np.eye(ndim) - np.dot(r, r.T)) / r.shape[1]
    eigVal, eigVec = np.linalg.eigh(iTensor)
    return np.dot(eigVec.T, r)  # eigVec.T @ r

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
    weights = _weights(N)
    return np.sum(weights * arr[1:, :].T, axis=1, dtype=np.float64)

# @njit
def _WBA_multi_arctan2_parallel(q, p):
    N = len(p[:, 0])
    weights = _weights(N)
    phi = np.arctan2(q, p) / (2*np.pi)
    phiDiff = (phi[1:, :] - phi[:-1, :] + 0.5) % 1.0 - 0.5
    return np.sum(weights * phiDiff.T, axis=1, dtype=np.float64)

# @njit
def _WBA_multi_parallel_wrapper(q, p, mapMode):
    if mapMode == 'none': return _WBA_multi_parallel(p)
    if mapMode == 'arctan2': return _WBA_multi_arctan2_parallel(q, p)
    
@njit
def embedding(arr):
    for i in range(1, arr.shape[0], 1):
        delta = arr[i] - arr[0]
        if delta > 0.5: arr[i] -= 1
        elif delta < -0.5: arr[i] += 1

@njit#('f8(f8[:], f8[:])', cache=True)
def _WBA_single_arctan2(q, p):
    # phi, r = map_arctan2(q, p)
    phi = np.arctan2(q, p) / (2*np.pi)
    phiDiff = phi[1:] - phi[:-1]
    embedding(phiDiff)
    # phiDiff = phiDiff % 1.0
    # if np.any(phiDiff < 0.25) and np.any(phiDiff > 0.75):
    #     phiDiff[phiDiff < 0.5] += 1
    return _WBA_single(phiDiff)

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

# def WBA(q0, p0, N=None, K=None, thresh=None, mapMode='none'):
#     """
#     WBA for the following input types:
#     If 'K' is given, caluculate (q, p) signals from the standard map.
#         Accepts a or an array pair of initial values. If 'N' is an array, 
#         (q0, p0) should not be arrays (not implemented, will raise Error!).
        
#     If 'K' is not given, calculate the WBA from the given array of 'p0'.
#         If 'N' is an array, the WBA is calculate for each element of 'N'.    
#     """
#     mapMode_list = ['none', 'arctan2', 'decision']
    
#     MaxNErrorMsg = "max(N) has to be less or equal to the length of 'p0'!"
#     TypeErrorMsg = "No method for given types found!"
#     modeListErrorMsg = ("'mapMode' has to be either one of ", mapMode_list,  
#                         " but was '", mapMode, "'!")
#     qpTypeErrorMsg = ("q0 and p0 must be type 'float' (or array) but were " 
#                       + "type(q0)=", type(q0), " and type(p0)=", type(p0))
#     NTypeErrorMsg = ("N must be a positive integer or an array of positive "
#                      + "integers, but was of type ", type(N)) 
    
#     # error handling due to incorrect input types
#     if mapMode not in mapMode_list: 
#         print(modeListErrorMsg)
#         raise TypeError
        
#     if mapMode != 'none' and thresh == None:
#         thresh = 1e-5
        
#     if (isinstance(q0, (np.float64, float)) and 
#         isinstance(p0, (np.float64, float))): 
#         qpMode = 'single'
#     elif type(q0[0]) == np.float64 and type(p0[0]) == np.float64: 
#         qpMode = 'multi'
#     elif type(q0[0, 0]) == np.float64 and type(p0[0, 0]) == np.float64: 
#         qpMode = 'multi 2d'
#     else: 
#         print(qpTypeErrorMsg)
#         raise TypeError
        
#     if type(N) == type(None): Ntype = 'none'
#     elif isinstance(N, (np.uint32, int)) and N > 0: Ntype = 'int'
#     elif type(N[0]) == np.uint32 and N[0] > 0: Ntype = 'array int'
#     else: 
#         print(NTypeErrorMsg)
#         raise TypeError
    
#     # case structure for calling methods corresponding to correct modes
#     if type(K) == float:
#         if qpMode == 'single':
#             if Ntype == 'int':
#                 q, p = _std_map(q0, p0, N, K)
#                 return _WBA_single_wrapper(q - 0.5, p, mapMode)
#             if Ntype == 'array int':
#                 q, p = _std_map(q0, p0, np.max(N), K)
#                 return _WBA_wrapper(N, q - 0.5, p, thresh, mapMode)

#         if qpMode == 'multi':
#             if Ntype == 'int':
#                 q, p = _std_map_multi(q0, p0, N, K)
#                 return _WBA_multi_wrapper(q - 0.5, p, thresh, mapMode)
#             if Ntype == 'array int': 
#                 print(TypeErrorMsg)
#                 raise TypeError
    
#     if type(K) == type(None):
#         if qpMode == 'multi':
#             if Ntype == 'none':
#                 return _WBA_single_wrapper(q0, p0, mapMode)
#             if Ntype == 'array int':
#                 if np.max(N) > len(q0): 
#                     print(MaxNErrorMsg)
#                     raise IndexError
#                 return _WBA_wrapper(N, q0, p0, thresh, mapMode)
#         if qpMode == 'multi 2d': 
#             return _WBA_multi_wrapper(q0, p0, thresh, mapMode)           
            
#     print(TypeErrorMsg)
#     print("No Output produced. Raising error to help isolate the issue.")
#     raise TypeError
    

@njit
def swap_rows(arr, x, y):
    arr[np.array([x, y])] = arr[np.array([y, x])]

@njit 
def min_max_ratio(arr):
    return np.min(arr) / np.max(arr)

@njit
def sort_by_extent(points):
    # p1sqr, p2sqr = points[0, :]**2, points[2, :]**2 #100 p may be enough
    # q1sqr, q2sqr = points[1, :]**2, points[3, :]**2
    p1sqr, q1sqr = points[0, :256]**2, points[2, :256]**2
    p2sqr, q2sqr = points[1, :256]**2, points[3, :256]**2
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
def _WBA_torus4d_single(points):
    points = transform_nd_torus(points)
    x, y, z, w = sort_by_extent(points)
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
def _WBA_torus4d(points, N):
    """Mean of all input arrays should be as close to zero as possible."""
    points = transform_nd_torus(points)
    x, y, z, w = sort_by_extent(points)
    # x, y, z, w = find_elliptic_sections(p1, p2, q1, q2, thresh)
    freq1 = _WBA_arctan2(N, z, x)
    freq2 = _WBA_arctan2(N, w, y)
    return freq1, freq2

@njit
def _WBA_torus4d_multi(points):
    length = len(points[0, 0, :])
    freq = np.zeros((2, length), dtype=np.float64)
    for i in range(length):
        freq[:, i] = _WBA_torus4d_single(points[:, :, i])
        # freq[:, i] = _WBA_torus4d_single_var(points[:, :, i], thresh)
    return np.abs(freq)

def WBA_torus4d(points, N=None, thresh=0.005):
    if type(N) == type(None):
        if len(np.shape(points)) == 2:
            return _WBA_torus4d_single(points)
        return _WBA_torus4d_multi(points)
    return _WBA_torus4d(points, N)

if __name__ == "__main__":
    print(__doc__)