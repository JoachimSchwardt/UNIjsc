#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-dimensional numerical analysis of fundamental frequencies (NaffND)

Uses gaussian weights to arrive at a frequency estimation of
    nu == j/N + 1/2N - alpha/(2N*pi**2) * log(R)
    
    j == index of a local maximum in the FFT
    N == signal length
    alpha == factor of gaussian weights (exp(-alpha * (...)**2))
    R == F_j / F_{j+1} (F_0, ... F_{N-1} are the FFT points)
    
This script provides a class which can be instantiated with a number of
    'signalCount' different signals, provided they are all of length 'N'.
"""

import numpy as np

def _remove_peak(fft, ind, nu, alpha=140, J=30):
    """Remove a gaussian peak from a FFT at a given frequency"""
    N = fft.shape[0]
    ampl = fft[ind % N]
    phase = nu * N - ind
    for j in range(-J+1, J, 1):
        corr_exp = -np.pi**2 / alpha * (j**2 - 2 * phase * j)
        fft[(ind+j) % N] -= ampl * np.exp(corr_exp)

def naffnd(z, n_freq=1, alpha=140, J=30, ReturnCoeff=False, Offset=False):
    """
    NaffND with gaussian weights.
    Computes the first 'n_freq' fundamental frequencies of the complex-valued
        input signal 'z', as well as their respective complex amplitudes.
    Parameters:
        n_freq == number of frequenices to be computed until max(fft) < 1e-15
                  (iterative process is aborted if the fft becomes 'flat')
        alpha == gaussian weights given by exp(-alpha * (t / N - 0.5)**2)
                 where t = 0,...,N-1 with 'N' being the signal length
                 (alpha = 140 leads to weights being about 1e-16 at t = 0)
        J == for the '2*J - 1' points closest to the relevant peak in the fft,
             the respective fft-value is subtracted based on an approximate
             analytical formula
             (J = 23 would be enough for machine precision at the outer parts)
        ReturnCoeff == whether to return the computed frequency amplitudes
        Offset == whether to compute, return and adjust a signal offset
    """
    N = z.shape[0]
    t = np.arange(N)    # t = 0,...,N-1
    weights = np.exp(-alpha * (t / N - 0.5)**2)
    fft = np.fft.fft(weights * z)   # need complex-valued fft for coeff.
    abs_fft = np.abs(fft)           # only need absolute values for freqs.
    
    # frequencies and their respective amplitudes
    nu_arr = np.zeros(n_freq + Offset)
    c_arr = np.zeros(n_freq + Offset, dtype=np.complex128)     
    
    if J > N//2:
        J = N//2
    if Offset:    # compute constant offset and adjust fft spectrum
        c_arr[n_freq] = np.sqrt(alpha / np.pi) * fft[0] / N    
        _remove_peak(abs_fft, ind=0, nu=0.0, J=J)
    
    for i in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-15:   
            break   # abort if fft-peak too small -> nu == 0
            
        # choose 'ind' closer to true peak (with 1e-6 tolerance)
        if abs_fft[ind - 1] > abs_fft[(ind + 1) % N] + 1e-6: 
            ind -= 1
        R = abs_fft[ind] / abs_fft[(ind + 1) % N]    # avoid index error
            
        # Ratio 'R' of largest neighboring fft-values to compute 'nu'
        R = abs_fft[ind] / abs_fft[ind + 1]
        nu = ind / N + 1 / (2*N) - alpha / (2 * np.pi**2 * N) * np.log(R)
        nu_arr[i] = nu
        
        # compute the complex value of the frequency amplitude
        phase = nu * N - ind
        c = np.sqrt(alpha / np.pi) * fft[ind] / N * np.exp(-1j*np.pi * phase)
        c_arr[i] = c * np.exp(np.pi**2 / alpha * phase**2)
        
        # remove the previous peak from the fft spectrum
        _remove_peak(abs_fft, ind, nu, alpha=alpha, J=J)
        
        # maybe also remove the mirrored frequency? <-- see NaffND tests..
        _remove_peak(abs_fft, N-ind, 1-nu, alpha=alpha, J=J)
    if ReturnCoeff:
        return nu_arr, c_arr
    return nu_arr

def naffnd_narr(z, Narr, n_freq=1, alpha=140, J=30, 
                ReturnCoeff=False, Offset=False):
    """
    Given an array with signal lengths (e.g. Narr = np.logspace(...)),
    computes the frequencies of the first 'Narr[i]' points
    of the given signal.
    """
    freq = np.zeros((Narr.shape[0], n_freq + Offset))
    if ReturnCoeff:
        coeff = np.zeros((Narr.shape[0], n_freq + Offset), 
                         dtype=np.complex128)
    for row in range(Narr.shape[0]):
        temp = naffnd(z[:Narr[row]], n_freq=n_freq, alpha=alpha, J=J, 
                      ReturnCoeff=ReturnCoeff, Offset=Offset)
        if ReturnCoeff:
            freq[row], coeff[row] = temp
        else:
            freq[row] = temp
    if ReturnCoeff:
        return freq, coeff
    return freq
    

class NaffND(object):
    """
    Input parameters :: 
        signals == complex ndarray with 1, 2 or 3 dimensions
                   Innermost axis has to be the signal length 'N'.
                   First axis should ideally correspond to the 
                       different signals.
                   Second axis should correspond to the components 
                       of each signal.
                        
        proj    == index of the component
                   Before the computation, one of those components 
                   is chosen via the 'proj' parameter.
                       --> computation only with 1d or 2d arrays
            
        n_freq  == number of frequencies to extract from every given signal
        
        Offset  == boolean, whether to compensate a nonzero signal-offset
                   Behavior beyond a certain signal length (N >~ 1e2...1e3)
                       this has no effect on the precision, as compared to 
                       manually removing the offset 
                       (possible in special cases) 
                        
        alpha   == decay parameter of the gaussian weights used
                   A value of 140 leads to values around 1e-16 at the edges.
                   
        J       == number of points to remove from each peak in the FFT
                   A value of 23 includes points down to 1e-16
                   (this parameter depends on alpha! relevant factor is
                    exp(-pi**2 / alpha * J**2))
                   
        maxComponents == [OPTIONAL] maximal number of components
                   Use this in the ambiguous case of a 2d-signal-array, 
                   as this could refer to ::
                       1. many signals with one component (default)
                       2. a single signal with multiple components
                   Specify the 'maxComponents' to ensure case (2)
    """
    def __init__(self, signals, n_freq=2, component=0, 
                 Offset=True, alpha=140, J=23, maxComponents=1):
        """Only accepts complex valued input signals."""
        if signals.dtype != complex:
            print("Signals should be of type 'complex' but is of type "
                  + f"{signals.dtype} !")
            raise TypeError
        
        self.n_freq = n_freq
        self.Offset = Offset
        self.alpha = alpha
        self.J = J
        self.maxComponents = maxComponents
        self.component = component
        
        self.signals = signals
        
        self.signals_shape = np.shape(signals)
        self.N = self.signals_shape[-1]    # signal length should be innermost
        self.M = self.signals_shape[0]     # number of different signals first
        
        # catch the 1d and 2d case
        self.z = self._convertValidateSignal()
        self.M = self.z.shape[0]
        
        # used to extract the np.argmax result ::
        # let 'a' be a 2d-array and 'ind = np.argmax(a, axis=1)'
        # then to access the maxima of 'a' use the syntax 'a[row_ind, ind]'
        # see 'https://stackoverflow.com/questions/14222110/extracting-
        # numpy-array-slice-from-numpy-argmax-results' for reference
        self.row_ind = np.arange(self.M) 
        
        # setup frequencies and corresponding coefficients
        self.freq = np.zeros((self.M, n_freq + Offset), dtype=np.float64)
        self.coeff = np.zeros((self.M, n_freq + Offset), dtype=np.complex128) 
        
        # if self.M == 1:
        #     self.freq = np.expand_dims(self.freq, axis=0)
        #     self.coeff = np.expand_dims(self.coeff, axis=0)
        
        self._FFTsetup()
        
        # catch potential IndexError by adapting the peak size
        if self.J > self.N//2: 
            self.J = self.N//2
        
        if self.Offset:
            self._correctOffset()
        
    def _convertValidateSignal(self):
        ndim = np.ndim(self.signals)
        if (ndim > 3) or (ndim < 1):
            print(f"Error: number of signal dimensions was {ndim}, " 
                  + "but should instead be one of (1, 2, 3) !")
            raise IndexError
        elif ndim == 3:
            return self.signals[:, self.component, :]
        elif ndim == 1:
            return np.expand_dims(self.signals, axis=0)
        elif ndim == 2:     # handle ambiguous 2d case
            if self.M == self.maxComponents:
                self.M = 1    # input is a single signal with many components
                return np.expand_dims(self.signals[self.component, :], axis=0)
            else:
                if self.M <= self.n_freq:
                    print("Warning, 2d-input signal understood as a list of "
                          + "1d signals. Use 'maxComponents' to indicate "
                          + "a single signal with multiple components!")
            return self.signals
        
    def _correctOffset(self):
        """Correct a non-zero constant offset in the FFT"""
        self.coeff[:, self.n_freq] = \
            np.sqrt(self.alpha / np.pi) * self.fft[:, 0] / self.N    
        self._removePeak(ind=0, nu=0.0)

    def _removePeak(self, ind, nu):
        """Remove a gaussian peak from a FFT at a given frequency"""
        ampl = self.abs_fft[self.row_ind, ind]
        phase = nu * self.N - ind
        for j in range(-self.J+1, self.J, 1):
            corr_exp = -np.pi**2 / self.alpha * (j**2 - 2 * phase * j)
            self.abs_fft[self.row_ind, (ind+j) % self.N] -= \
                ampl * np.exp(corr_exp)
            
    def _FFTsetup(self):
        """Set up the FFT"""
        self.t = np.arange(self.N)
        self.weights = np.exp(-self.alpha * (self.t / self.N - 0.5)**2)
        self.fft = np.fft.fft(self.weights * self.z, axis=1)
        self.abs_fft = np.abs(self.fft)

    def compute(self, peak_tolerance=1e-6, abort_peak_height=1e-15):
        """
        Execute the iterative computation of all 'n_freq' frequencies, 
        as well as their corresponding coefficients
        """
        for i in range(self.n_freq):
            # find global maximum of the FFT
            ind = np.argmax(self.abs_fft, axis=1) 
            
            # abort if fft-peak too small  -->  nu == 0
            abort_peak_height_indx = (self.abs_fft[self.row_ind, ind]
                                       < abort_peak_height)
            self.row_ind = self.row_ind[~abort_peak_height_indx]
            if np.all(abort_peak_height_indx):
                break
                
            # compare the FFT-value left and right of the peak at 'ind'
            # the ratio 'R' is computed using the peak and the larger of these
            ind-= (self.abs_fft[self.row_ind, ind - 1] > 
                   (self.abs_fft[self.row_ind, (ind + 1) % self.N] 
                    + peak_tolerance))            
                
            # avoid index error in the case of 'ind == N-1'
            R = (self.abs_fft[self.row_ind, ind] 
                 / self.abs_fft[self.row_ind, (ind + 1) % self.N])    
            
            # formula for the frequency estimate using 'R'
            nu = (ind / self.N 
                  + 1 / (2*self.N) 
                  - self.alpha / (2 * np.pi**2 * self.N) * np.log(R))
            self.freq[:, i] = nu
            
            # compute the complex value of the frequency amplitude
            phase = nu * self.N - ind
            c = (np.sqrt(self.alpha / np.pi) 
                 * self.fft[self.row_ind, ind] / self.N 
                 * np.exp(-1j*np.pi * phase)
                 * np.exp(np.pi**2 / self.alpha * phase**2))
            self.coeff[:, i] = c
            
            # remove the previous peak from the fft spectrum
            self._removePeak(ind, nu)
            self._removePeak(self.N - ind, 1 - nu)
            
            
def test_NaffND():
    from std_map_4d import map4dCyl
    N = 3000       # signal length
    k1, k2, k = 2.25, 3.0, 1.0
    
    M = 5          # number of signals
    p10, p20, q10, q20 = \
        np.random.uniform(0.45, 0.55, size=4*M).reshape((4, M))
    p10 -= 0.5
    p20 -= 0.5
    signals = map4dCyl(p10, p20, q10, q20, N, k1, k2, k)
    cSignals = np.zeros((M, 2, N), dtype=np.complex128)
    cSignals[:, 0, :] = signals[:, 0, :] + 1j*signals[:, 2, :]
    cSignals[:, 1, :] = signals[:, 1, :] + 1j*signals[:, 3, :]
    
    s = cSignals[0, 0, :]
    # s2 = cSignals[0, :, :]
    # s2var = cSignals[:, 0, :]
    
    naff_nd = NaffND(cSignals, n_freq=2, component=0)
    naff_1d = NaffND(s)
    # naff_2d = NaffND(s2, maxComponents=2)
    # naff_2d_var = NaffND(s2var)
    
    naff_nd.compute()
    naff_1d.compute()
    # naff_2d.compute()
    # naff_2d_var.compute()
    
    import matplotlib.pyplot as plt
    from Naff_var import signal_list, naffnd_gauss
    # z1d = signal_list(naff_1d.freq[0], naff_1d.coeff[0], N=N)
    # print(naffnd_gauss(s,2,ReturnCoeff=1, Offset=1))
    # print(naff_1d.freq, naff_1d.coeff)
    # z2d = signal_list(naff_2d.freq[0], naff_2d.coeff[0], N=N)
    # z2d_var = signal_list(naff_2d_var.freq[0], naff_2d_var.coeff[0], N=N)
    
    def plot(orig, recon):
        fig, ax = plt.subplots()
        p = ax.plot(orig.real, orig.imag, ls='', marker='o', ms=2, mew=1)
        ax.plot(recon.real, recon.imag, ls='', marker='x', ms=4, mew=1,
                c=p[0].get_color())
        
    # p1,p2,q1,q2 = (0.06190071726227242, 0.09866037504383973, 
    #                0.5870820952643117, 0.4106125964592201)
    for i in range(M):
        znd = signal_list(naff_nd.freq[i], naff_nd.coeff[i], N=N)
        plot(cSignals[i, 0, :], znd)
        
def test_naffnd_narr():
    from std_map_4d import map4dCyl
    Narr = (2**np.log10(np.logspace(5.0, 14.0, 25))).astype(int)
    N = Narr[-1]       # signal length
    k1, k2, k = 2.25, 3.0, 0.2
    
    p10 = np.array([0.05]) 
    p20 = np.array([0.03])
    q10 = np.array([0.51])
    q20 = np.array([0.5])
    
    M = p10.shape[0]         # number of signals
    signals = map4dCyl(p10, p20, q10, q20, 2*N, k1, k2, k)
    cSignals = np.zeros((M, 2, 2*N), dtype=np.complex128)
    cSignals[:, 0, :] = signals[:, 0, :] + 1j*signals[:, 2, :]
    cSignals[:, 1, :] = signals[:, 1, :] + 1j*signals[:, 3, :]
    
    z1 = cSignals[0, 0, :N]   # first initial condition, first component
    z1 -= 0.5j
    z2 = cSignals[0, 1, :N]   # first initial condition, second component
    z2 -= 0.5j
    z12 = cSignals[0, 0, N:]   # first initial condition, first component
    z12 -= 0.5j
    z22 = cSignals[0, 1, N:]   # first initial condition, second component
    z22 -= 0.5j
    freq = naffnd_narr(z1, Narr, n_freq=2, Offset=False)
    freq2 = naffnd(z12, n_freq=2, Offset=False)
    
    from Naff_var import naff1d_approx
    
    freq_approx = np.array([[naff1d_approx(z1[:Nval]), 
                             naff1d_approx(z2[:Nval])] 
                            for Nval in Narr])
    freq_approx2 = np.array([naff1d_approx(z12), naff1d_approx(z22)])
    
    import matplotlib.pyplot as plt
    from matplotlib import special
    special.setup(UseTex=True)
    fig, ax = plt.subplots(2, 2)
    colors = special.Colors()
    ax[0, 0].set_title(f"$k_1={k1}$, $k_2={k2}$, $k={k}$, $N={N}$")
    
    colors.get_color()
    z = [z1, z2]
    for i in range(2 - (k < 1e-2)):
        ax[i, 0].set_xlabel(f"$q_{i+1}$")
        ax[i, 0].set_ylabel(f"$p_{i+1}$")
        ax[i, 0].plot(z[i].real, z[i].imag, ls='', marker='o', 
                      c=colors.prev_color())
        
        ax[i, 1].set_xlabel(r"$N$")
        ax[i, 1].set_ylabel(r"$|\nu_N-\nu_{N_\mathrm{max}}|$")
        ax[i, 1].set_xscale('log')
        ax[i, 1].set_yscale('log')
        ax[i, 1].set_xlim(Narr[0], Narr[-1])
        delta_freq = np.abs(freq[-1, i] - freq2[i])
        delta_freq_approx = np.abs(freq_approx[-1, i] - freq_approx2[i])
        
        title = (f"$\\nu_{{\\mathrm{{gauss}}}}={freq[-1, i]}$, " 
                 + f"$\\Delta\\nu={delta_freq:.2e}$ \n" 
                 + f"$\\nu_{{\\mathrm{{naff}}}}={freq_approx[-1, i]}$, "
                 + f"$\\Delta\\nu={delta_freq_approx:.2e}$")
        ax[i, 1].set_title(title)
        
        abs_diff = np.abs(freq[:-1, i] - freq[-1, i])
        abs_diff[abs_diff < 1e-16] = 1e-16
        abs_diff_approx = np.abs(freq_approx[:-1, i] - freq_approx[-1, i])
        abs_diff_approx[abs_diff_approx < 1e-16] = 1e-16
        ax[i, 1].plot(Narr[:-1], abs_diff, c=colors.prev_color())
        ax[i, 1].plot(Narr[:-1], abs_diff_approx, ls='--', 
                      c=colors.prev_color())
    
    special.polish(fig, ax.flatten(), SetCaptions=False)
    
            
if __name__ == "__main__":
    print(__doc__)
    # test_NaffND()
    test_naffnd_narr()
        
    
    """
def richardson(func, n, K=4):
    return np.sum([(-1)**(K-j) * np.math.comb(K, j) * func(j+n) 
                   * (j+n)**K for j in range(K+1)]) / np.math.factorial(K)
    """
        
    """
def _std_map(q0, p0, K, N):
    q = np.zeros(N)
    p = np.zeros(N)
    q[0] = q0
    p[0] = p0
    for i in range(1, N, 1):
        q[i] = (q[i-1] + p[i-1]) % 1.0
        p[i] = p[i-1] + K * np.sin(2*np.pi * q[i]) / (2*np.pi)
    return q, p
    
q,p=_std_map(0.7, 0.3, K=0.1, N=65536)
z=q+1j*p
np.fft.fft(z)



def bitReverse(x, N):
    n = 0
    for i in range(int(np.log2(N))):
        n <<= 1
        n |= (x & 1)
        x >>= 1
    return n

def computeFFT(inp):
    N = inp.shape[0]
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        out[i] = inp[bitReverse(i, N)]
    
    for s in range(1, int(np.log2(N)) + 1, 1):
        m = 1 << s 
        m2 = m >> 1
        
        w = 1.0 + 0.0j
        if m2 == 0:
            wm = 1.0 + 0.0j
        else:
            wm = np.exp(-1j * np.pi / m2)
        for j in range(m2):
            for k in range(j, N, m):
                t = w * out[k + m2]
                out[k + m2] = out[k] - t
                out[k] += t
            w *= wm
    return out

def testfft(inp):
    N = inp.shape[0]
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        print(f"{i:08b} | {bitReverse(i, N):08b}")
        out[i] = inp[bitReverse(i, N)]

    for s in range(1, int(np.log2(N)) + 1, 1):
        m = 1 << s 
        m2 = m >> 1
        print(f"{s = }, {m = }, {m2 = }")

        w = 1.0 + 0.0j
        wm = np.exp(-1j * np.pi / m2)
        print(f"{wm = }")
        for j in range(m2):
            for k in range(j, N, m):
                print(f"{k = }, {k+m2 = }")
                t = w * out[k + m2]
                out[k + m2] = out[k] - t
                out[k] += t
                print(f"{t = }, {out = }")
            w *= wm
    return out
    """

