import numpy as np
from time import perf_counter as pc
from numba import njit

@njit('Tuple((u4[:,:], uint16[:]))(f8[:], f8, i4)')
def _array_to_intervals_njit(arr, bLen, bSizeEst):
    """Internal function for 'indx_array_generator'. """
    totalBCount = int(np.max(arr) // bLen) + 1
    indxArray = np.full((totalBCount, bSizeEst), len(arr), dtype=np.uint32)
    indxTable = np.zeros(totalBCount, dtype=np.uint16)
    for i in range(len(arr)):
        bIndx = int(arr[i] // bLen)
        indxArray[bIndx, indxTable[bIndx]] = i
        indxTable[bIndx] += 1
    return indxArray, indxTable

@njit
def indx_array_generator(arr, bLen, bSizeEst=-1):
    """
    Creates a generator object by 'indx_gen = indx_array_generator(...)'
    The generator is used by calling 'indx = next(gen)' to get the next list
    of indices.
    """
    if bSizeEst < 0:         # catch negative default estimate
        bSizeEst = 100       # general predicition, might be very large
    indxArray, indxTable = _array_to_intervals_njit(arr, bLen, bSizeEst)
    for i in range(len(indxArray)):
        yield indxArray[i, :indxTable[i]]

def simulate_data(data, linkRTO, thres):
    linkUtilization = np.mean(linkRTO)      # mean value is percentage link on

    indxOffset = len(linkRTO) - 20 * len(data)
    dataFilled = np.zeros(20 * len(data) + indxOffset, dtype=np.float64)
    dataFilled[:-indxOffset:20] = data

    dataLatency = dataFilled[linkRTO == 1]
    underThresh = np.sum((dataLatency > 0.0) & (dataLatency < thres)) / np.sum(dataLatency > 0.0)
    return linkUtilization, underThresh

@njit('UniTuple(f8[:], 2)(f8[:], f8, f8)')
def get_RTO(data, alpha, beta):
    SRTT = np.zeros(len(data), dtype=np.float64)
    RTO = np.zeros(len(data), dtype=np.float64)
    SRTT[0] = data[0]
    RTTvar = data[0] / 2
    RTO[0] = SRTT[0] + 4 * RTTvar

    for i in range(1, len(data), 1):
        RTTvar = (1 - beta) * RTTvar + beta * abs(SRTT[i - 1] - data[i])
        SRTT[i] = (1 - alpha) * SRTT[i-1] + alpha * data[i]
        RTO[i] = SRTT[i] + 4 * RTTvar
    return SRTT, RTO

@njit('uint8[:](f8[:], u4, u4, f8, f8, u4, f8[:])')
def _RTO_fix(data, timeval, thres, alpha, beta, downtime, absRecTime):
    SRTTs, RTOs = get_RTO(data, alpha, beta)
    linkRTO = np.ones(len(data) * 20 + 2 * timeval, dtype=np.uint8)
    #print(timeval, absRecTime[-1] + timeval, timeval)
    indx_gen = indx_array_generator(absRecTime, timeval)
    for timer in range(timeval, int(absRecTime[-1]) + timeval, timeval):
        indx = next(indx_gen)
        # indx = ((absRecTime < timer) & (absRecTime >= timer - timeval))
        # if np.any(indx):
        # indx = np.where((absRecTime < timer) & 
        #                 (absRecTime >= timer - timeval))[0]
        # indxStart = max(int(timer / timeval) - 5, 0)
        # absRecTimeSlice = absRecTime[indxStart:indxStart + 5]
        # indx = np.where((absRecTimeSlice < timer) & (absRecTimeSlice >= timer - timeval))[0] + indxStart
        # indx2 = np.where((absRecTime < timer) & (absRecTime >= timer - timeval))[0]
        # counter += 1
        # if np.any(indx != indx2):
        #     ecounter +=1
        #     print("Error: ", indx, indx2)

        # if len(indx) > 0:
        #     print(indx[0], min(indx), max(indx), indx[-1])
        if len(indx) > 0:
            r = data[indx]
            RTOval = RTOs[indx]
            RTO = RTOval[-1]
            # RTO = np.min(RTOs[indx])

            if np.all(r > thres):
                linkRTO[timer:timer + downtime] = 0

            if RTO < thres:
                linkRTO[timer:timer + downtime] = 1

        else:
            linkRTO[timer:timer + downtime] = 0
    return linkRTO

def main():
    seq, tb1, tb2 = np.genfromtxt("201217_data.csv", delimiter=',')[3:-1].T
    # print(tb1[:10], tb1[-5:])

    alpha = 0.125
    beta = 0.25
    timeval = np.array([19, 39, 59, 109])
    downtime = np.array([19, 39, 59, 109])*2
    thres = 50
    decision = ["RTO_fix", "SRTT_fix", "RTO_new" ,"SRTT_new", "none"]
    decision_for_title = ["RTO_fix", "SRTT_fix", "RTO_new" ,"SRTT_new", "none"]

    data = tb1 #[:51000]
    # print(data[0:5], data[-1])
    # matrix = np.zeros((5, len(data)))
    # underTh = np.zeros(len(data))
    # linkU = np.zeros(len(data))
    # c = 0
    k = 0
    j = 0

    timescala = np.arange(0, len(data) * 20 + 2 * timeval[k], 1,
                          dtype=np.uint32)
    timervals = np.arange(0, timescala[-1], timeval[k], dtype=np.uint32)
    absSentTime = np.arange(0, len(data)*20, 20, dtype=np.uint32)
    absRecTime = absSentTime + data

    t1 = pc()
    linkRTO = _RTO_fix(data, timeval[k], thres, alpha, beta, 
                       downtime[j], absRecTime)
    t2 = pc()
    linkUtilization, underThresh = simulate_data(data, linkRTO, thres)
    t3 = pc()
    print(f"timer: {t2-t1} and simulate data {t3 - t2}")
    print(linkUtilization, underThresh)

    # for t in range(len(downtime)):
    #     for k in range(len(timeval)):
    #         #under_thres, linkusage = _RTO_fix(data, timeval[k], thres, alpha, beta, downtime[t])
    #         #underTh[c] = under_thres
    #         #linkU[c] = linkusage
    #         c += 1
    
if __name__ == "__main__":
    main()
    """
    timer: 154.17677936697146 and simulate data 0.04922366500250064
    0.8655828267603963 0.9983546943608983
    matlab linkutilization: 9.075093e-01, underThres: 9.981523e-01
    """