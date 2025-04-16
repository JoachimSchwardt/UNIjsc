"""
Distribute array values to regular intervals.
Best use is 'gen = indx_array_generator()' with next(gen).
"""

import numpy as np
from numba import njit

def array_to_intervals(arr, bucketLength):
    totalBucketsCount = int(np.max(arr) // bucketLength) + 1
    indxList = [[] for _ in range(totalBucketsCount)]
    _iterator(indxList, arr, bucketLength)
    
    return indxList

def _iterator(indxList, arr, bucketLength):
    for i in range(len(arr)):
        bucketIndx = int(arr[i] // bucketLength)
        indxList[bucketIndx].append(i)
    return indxList

@njit('Tuple((u4[:,:], uint16[:]))(f8[:], f8, i4)')
def _array_to_intervals_njit(arr, bLen, bSizeEst):
    totalBCount = int(np.max(arr) // bLen) + 1
    indxArray = np.full((totalBCount, bSizeEst), len(arr), dtype=np.uint32)
    indxTable = np.zeros(totalBCount, dtype=np.uint16)
    for i in range(len(arr)):
        bIndx = int(arr[i] // bLen)
        indxArray[bIndx, indxTable[bIndx]] = i
        indxTable[bIndx] += 1
    return indxArray, indxTable

def indx_array(arr, bLen, bSizeEst=-1):
    if bSizeEst < 0:         # catch negative default estimate
        bSizeEst = 100       # general predicition, might be very large
    indxArray, indxTable = _array_to_intervals_njit(arr, bLen, bSizeEst)
    indxList = [indxArray[i, :indxTable[i]] for i in range(len(indxArray))]
    return indxList

@njit
def indx_array_generator(arr, bLen, bSizeEst=-1):
    if bSizeEst < 0:         # catch negative default estimate
        bSizeEst = 100       # general predicition, might be very large
    indxArray, indxTable = _array_to_intervals_njit(arr, bLen, bSizeEst)
    for i in range(len(indxArray)):
        yield indxArray[i, :indxTable[i]]


if __name__ == "__main__":
    arr = np.random.randint(0, 100, 10).astype(np.float64)
    # arr = np.arange(100)
    bucketLength = 9.0
    indxListTest = array_to_intervals(arr, bucketLength)
    print(indxListTest)
    """
    Test environment:
    N = 10**5
    bucketLength = 10.0
    bucketSizeEstimate = 20
    arr = np.arange(0, 20*N, 20) + np.random.uniform(10.0, 1000.0, N)
    indx = indx_array(arr, bucketLength, bucketSizeEstimate)
    indx = array_to_intervals(arr, bucketLength)
    indxLen = [len(i) for i in indx]
    len(indx) / len(arr), max(indxLen)
    """