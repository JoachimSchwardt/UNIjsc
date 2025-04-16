"""Frequency space computation using numpy and numba"""

import numpy as np
from numba import njit, prange, objmode
import time

@njit(parallel=True)
def computeFreqSpaceParallel(numThreads, numFreq, bbox, N, 
                             my2pi, k1_2pi, k2_2pi, k_2pi, alpha, J):
    for thread in prange(numThreads):
        computeFreqSpaceThread(thread, numFreq, bbox, N, 
                               my2pi, k1_2pi, k2_2pi, k_2pi, alpha, J)

@njit
def computeFreqSpaceThread(thread, numFreq, bbox, N, 
                           my2pi, k1_2pi, k2_2pi, k_2pi, alpha, J):
    weights = np.exp(-alpha * (np.arange(N) / N - 0.5)**2)
    freq = np.zeros((4, numFreq))
    z10 = np.random.uniform(bbox[0, 0], bbox[1, 0], size=numFreq) \
        + 1j * np.random.uniform(bbox[0, 2], bbox[1, 2], size=numFreq)
    q20 = np.random.uniform(bbox[0, 1], bbox[1, 1], size=numFreq) 
    p20 = np.random.uniform(bbox[0, 3], bbox[1, 3], size=numFreq)
    
    for i in range(numFreq):
        orbit, q2, p2 = compute_orbit(z10[i], q20[i], p20[i], N, 
                                      my2pi, k1_2pi, k2_2pi, k_2pi)
        with objmode(abs_fft='f8[:]'):
            abs_fft = np.abs(np.fft.fft(orbit * weights))
        remove_peak(abs_fft, 0, 0.0, N, alpha, J)
        freq[0, i], ind = compute_freq(abs_fft, N, alpha)
        remove_peak(abs_fft, ind, freq[0, i], N, alpha, J)
        remove_peak(abs_fft, N - ind, 1.0 - freq[0, i], N, alpha, J)
        freq[2, i], ind = compute_freq(abs_fft, N, alpha)
        
        orbit, q2, p2 = compute_orbit(orbit[-1], q2, p2, N, 
                                      my2pi, k1_2pi, k2_2pi, k_2pi)
        with objmode(abs_fft='f8[:]'):
            abs_fft = np.abs(np.fft.fft(orbit * weights))
        remove_peak(abs_fft, 0, 0.0, N, alpha, J)
        freq[1, i], ind = compute_freq(abs_fft, N, alpha)
        remove_peak(abs_fft, ind, freq[1, i], N, alpha, J)
        remove_peak(abs_fft, N - ind, 1.0 - freq[1, i], N, alpha, J)
        freq[3, i], ind = compute_freq(abs_fft, N, alpha)
        
    save_freq(thread, freq)
    
@njit
def save_freq(thread, freq):
    with objmode:
        np.savetxt("pyresults_threadID" + str(thread) + ".gz", freq)
        
@njit
def compute_freq(abs_fft, N, alpha):
    ind = np.argmax(abs_fft)
    ind -= (abs_fft[ind-1] > (abs_fft[(ind+1) % N] + 1e-6))
    R = abs_fft[ind] / abs_fft[(ind+1) % N]
    nu = (2 * ind + 1 - alpha / np.pi**2 * np.log(R)) / (2 * N)
    return nu, ind

@njit
def remove_peak(abs_fft, ind, nu, N, alpha=140, J=23):
    """Remove a gaussian peak from a FFT at a given frequency"""
    ampl = abs_fft[ind]
    phase = 2 * (nu * N - ind)
    for j in range(-J+1, J, 1):
        abs_fft[ind+j] -= ampl * np.exp(-np.pi**2 / alpha * (j * (j - phase)))

@njit
def compute_orbit(z10, q2, p2, N, my2pi, k1_2pi, k2_2pi, k_2pi):
    """Warning :: my2pi == 2*np.pi, k1_2pi == k1 / (2*np.pi), etc. !!"""
    orbit = np.zeros(N, dtype=np.complex128)
    orbit[0] = z10
    
    for i in range(1, N, 1):
        orbit.real[i] = (orbit.real[i-1] + orbit.imag[i-1]) % 1.0
        q2 = (q2 + p2) % 1.0
        coupling = k_2pi * np.sin(my2pi * (orbit.real[i] + q2))
        orbit.imag[i] = ((orbit.imag[i-1] 
                            + k1_2pi * np.sin(my2pi * orbit.real[i]) 
                            + coupling) + 0.5) % 1.0 - 0.5
        p2 = ((p2 
                            + k2_2pi * np.sin(my2pi * q2) 
                            + coupling) + 0.5) % 1.0 - 0.5
    return orbit, q2, p2

def main():
    N = 4096
    numThreads = 16
    numFreq = 10    # 16 1e4 (nwithout compile-time) --> 9.4s (about 87% cpu)
    
    bbox = np.array([[0.45, 0.45, -0.05, -0.05], 
                     [0.55, 0.55, 0.05, 0.05]])
    
    k1 = 2.25
    k2 = 3.0
    k = 1.0
    
    my2pi = 2 * np.pi
    k1_2pi = k1 / my2pi
    k2_2pi = k2 / my2pi
    k_2pi = k / my2pi
    
    alpha = 140.0
    J = 23
    
    start = time.perf_counter()
    computeFreqSpaceParallel(numThreads, numFreq, bbox, N, 
                             my2pi, k1_2pi, k2_2pi, k_2pi, alpha, J)
    end = time.perf_counter()
    print(f"Total execution time was {(end - start) * 1000} ms.")
    
    
if __name__ == "__main__":
    print(__doc__)
    main()
    """
    import matplotlib.pyplot as plt
    from matplotlib import special
    special.setup(dpi=100)
def get_freq(prefix="pyresults_threadID", suffix=".gz", numThreads=4, 
             thresh=1e-6, usecols=None, skiprows=0, delimiter=' '):
    freq = [np.loadtxt(prefix + str(i) + suffix, usecols=usecols,
                       skiprows=skiprows, delimiter=delimiter) 
            for i in range(numThreads)]
    if freq[0].shape[0] > freq[0].shape[1]:
        freq = [f.T for f in freq]
    numFreq = freq[0].shape[1]
    freq_total = np.zeros((4, numThreads * numFreq))
    for i in range(4):
        for j in range(numThreads):
            freq_total[i, (j*numFreq):((j+1)*numFreq)] = freq[j][i,:]
    indx05 = (freq_total > 0.5)
    freq_total[indx05] = 1 - freq_total[indx05]
    nu11, nu12, nu21, nu22 = freq_total
    indx12 = (nu11 < nu21)
    nu11[indx12], nu21[indx12] = nu21[indx12], nu11[indx12]
    indx12 = (nu12 < nu22)
    nu12[indx12], nu22[indx12] = nu22[indx12], nu12[indx12]
    indx = (np.abs(nu11 - nu12) < thresh)
    nu1, nu2 = nu11[indx], nu21[indx]
    print(f"Total of {nu1.shape[0]} frequencies out of {numFreq * numThreads}"
          + " with chaos indicator below", thresh)
    return nu1, nu2

old Naff for compariso::
PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\"
PATHFREQ = PATH_TP + "CP_Bachelor\\WBA_Python\\FreqSpace\\"
fn1, fn2 = np.loadtxt(PATHFREQ + "Naff2021_freqs.gz")
indxr = ((fn1 > 0.27) & (fn1 < 0.31))
fn1r, fn2r = fn1[indxr], fn2[indxr]

plt.plot(fn1, fn2, ls='', marker='o', ms=1, mew=1, c='r')
plt.plot(nu1, nu2, ls='', marker='o', ms=1, mew=1, c='k')
plt.plot(f1, f2, ls='', marker='o', ms=1, mew=1, c='b')
    """
    #f1, f2 = get_freq(prefix=r"C:\users\joachim\documents\visual studio 2019\c++\naff\freqspace4d_singlefile\noneresults_threadID", suffix=".txt", skiprows=1, usecols=[4,5,6,7], delimiter=',')