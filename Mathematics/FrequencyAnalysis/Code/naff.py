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
from scipy.optimize import brentq
from scipy.integrate import quad
from window_functions import hann_coeff, hann_weights, gauss_weights


#########################################################################################
# GAUSS
#########################################################################################


def _remove_peak_gauss(fft, ind, nu, alpha=140, num_j=20):
    """Remove a gaussian peak from a FFT at a given frequency"""
    size = fft.shape[0]
    ampl = fft[ind % size]
    phase = nu * size - ind
    for j in range(-num_j+1, num_j, 1):
        corr_exp = -np.pi**2 / alpha * (j**2 - 2 * phase * j)
        fft[(ind+j) % size] -= ampl * np.exp(corr_exp)


def naffnd_gauss(z, n_freq=1, alpha=140, num_j=20, return_coeff=False):
    """
    NaffND with gaussian weights.
    Computes the first 'n_freq' fundamental frequencies of the complex-valued
        input signal 'z', as well as their respective complex amplitudes.
    Parameters:
        n_freq == number of frequenices to be computed until max(fft) < 1e-15
                  (iterative process is aborted if the fft becomes 'flat')
        alpha == gaussian weights given by exp(-alpha * (n / size - 0.5)**2)
                 where n = 0,...,N-1 with 'N' being the signal length
                 (alpha = 140 leads to weights being about 1e-16 at n = 0)
        num_j == for the '2*num_j - 1' points closest to the relevant peak in the fft,
             the respective fft-value is subtracted based on an approximate
             analytical formula
             (num_j = 23 would be enough for machine precision at the outer parts)
        return_coeff == whether to return the computed frequency amplitudes
        Offset == whether to compute, return and adjust a signal offset
    """
    size = z.shape[0]
    n = np.arange(size)    # n = 0,...,size-1
    weights = gauss_weights(n / size)
    fft = np.fft.fft(weights * z)   # need complex-valued fft for coeff.
    abs_fft = np.abs(fft)           # only need absolute values for freqs.

    # frequencies and their respective amplitudes
    nu_arr = np.zeros(n_freq)
    c_arr = np.zeros(n_freq, dtype=np.complex128)

    if num_j > size//2:
        num_j = size//2

    for i in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-15:
            break   # abort if fft-peak too small -> nu == 0

        # choose 'ind' closer to true peak (with 1e-6 tolerance)
        if abs_fft[ind - 1] > abs_fft[(ind + 1) % size] + 1e-6:
            ind -= 1
            
        ratio = abs_fft[ind] / abs_fft[(ind + 1) % size]    # avoid index error
        nu = ind / size + 1 / (2*size) - alpha / (2 * np.pi**2 * size) * np.log(ratio)
        nu_arr[i] = nu

        # compute the complex value of the frequency amplitude
        phase = nu * size - ind
        c = np.sqrt(alpha / np.pi) * fft[ind] / size * np.exp(-1j*np.pi * phase)
        c_arr[i] = c * np.exp(np.pi**2 / alpha * phase**2)

        # remove the previous peak from the fft spectrum
        _remove_peak_gauss(abs_fft, ind, nu, alpha=alpha, num_j=num_j)

        # maybe also remove the mirrored frequency? <-- see NaffND tests..
        _remove_peak_gauss(abs_fft, size - ind, 1 - nu, alpha=alpha, num_j=num_j)
    if return_coeff:
        return nu_arr, c_arr
    return nu_arr


def naffnd_gauss_narr(z, n_arr, n_freq=1, alpha=140, num_j=30,
                      return_coeff=False):
    """
    Given an array with signal lengths (e.g. n_arr = np.logspace(...)),
    computes the frequencies of the first 'n_arr[i]' points
    of the given signal.
    """
    freq = np.zeros((n_arr.shape[0], n_freq))
    if return_coeff:
        coeff = np.zeros((n_arr.shape[0], n_freq),
                         dtype=np.complex128)
    for row in range(n_arr.shape[0]):
        temp = naffnd_gauss(z[:n_arr[row]], n_freq=n_freq, alpha=alpha, num_j=num_j,
                            return_coeff=return_coeff)
        if return_coeff:
            freq[row], coeff[row] = temp
        else:
            freq[row] = temp
    if return_coeff:
        return freq, coeff
    return freq


class NaffND_gauss:
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

        alpha   == decay parameter of the gaussian weights used
                   A value of 140 leads to values around 1e-16 at the edges.

        num_j   == number of points to remove from each peak in the FFT
                   A value of 23 includes points down to 1e-16
                   (this parameter depends on alpha! relevant factor is
                    exp(-pi**2 / alpha * num_j**2))

        max_components == [OPTIONAL] maximal number of components
                   Use this in the ambiguous case of a 2d-signal-array,
                   as this could refer to ::
                       1. many signals with one component (default)
                       2. a single signal with multiple components
                   Specify the 'max_components' to ensure case (2)
    """
    def __init__(self, signals, n_freq=2, component=0,
                 alpha=140, num_j=23, max_components=1):
        """Only accepts complex valued input signals."""
        if signals.dtype != complex:
            print("Signals should be of type 'complex' but is of type "
                  + f"{signals.dtype} !")
            raise TypeError

        self.n_freq = n_freq
        self.alpha = alpha
        self.num_j = num_j
        self.max_components = max_components
        self.component = component

        self.signals = signals

        self.signals_shape = np.shape(signals)
        self.size = self.signals_shape[-1]    # signal length should be innermost
        self.num_signals = self.signals_shape[0]     # number of different signals first

        # catch the 1d and 2d case
        self.z = self._convert_validate_signal()
        self.num_signals = self.z.shape[0]

        # used to extract the np.argmax result ::
        # let 'a' be a 2d-array and 'ind = np.argmax(a, axis=1)'
        # then to access the maxima of 'a' use the syntax 'a[row_ind, ind]'
        # see 'https://stackoverflow.com/questions/14222110/extracting-
        # numpy-array-slice-from-numpy-argmax-results' for reference
        self.row_ind = np.arange(self.num_signals)

        # setup frequencies and corresponding coefficients
        self.freq = np.zeros((self.num_signals, n_freq), dtype=np.float64)
        self.coeff = np.zeros((self.num_signals, n_freq), dtype=np.complex128)

        self._fft_setup()

        # catch potential IndexError by adapting the peak size
        if self.num_j > self.size//2:
            self.num_j = self.size//2

    def _convert_validate_signal(self):
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
            if self.num_signals == self.max_components:
                self.num_signals = 1    # input is a single signal with many components
                return np.expand_dims(self.signals[self.component, :], axis=0)
            else:
                if self.num_signals <= self.n_freq:
                    print("Warning, 2d-input signal understood as a list of "
                          + "1d signals. Use 'max_components' to indicate "
                          + "a single signal with multiple components!")
            return self.signals

    def _remove_peak(self, ind, nu):
        """Remove a gaussian peak from a FFT at a given frequency"""
        ampl = self.abs_fft[self.row_ind, ind]
        phase = nu * self.size - ind
        for j in range(-self.num_j+1, self.num_j, 1):
            corr_exp = -np.pi**2 / self.alpha * (j**2 - 2 * phase * j)
            self.abs_fft[self.row_ind, (ind+j) % self.size] -= \
                ampl * np.exp(corr_exp)

    def _fft_setup(self):
        """Set up the FFT"""
        self.t = np.arange(self.size)
        self.weights = np.exp(-self.alpha * (self.t / self.size - 0.5)**2)
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
            ind -= (self.abs_fft[self.row_ind, ind - 1] >
                    (self.abs_fft[self.row_ind, (ind + 1) % self.size]
                     + peak_tolerance))

            # avoid index error in the case of 'ind == N-1'
            ratio = (self.abs_fft[self.row_ind, ind]
                 / self.abs_fft[self.row_ind, (ind + 1) % self.size])

            # formula for the frequency estimate using 'R'
            nu = (ind / self.size
                  + 1 / (2*self.size)
                  - self.alpha / (2 * np.pi**2 * self.size) * np.log(ratio))
            self.freq[:, i] = nu

            # compute the complex value of the frequency amplitude
            phase = nu * self.size - ind
            c = (np.sqrt(self.alpha / np.pi)
                 * self.fft[self.row_ind, ind] / self.size
                 * np.exp(-1j*np.pi * phase)
                 * np.exp(np.pi**2 / self.alpha * phase**2))
            self.coeff[:, i] = c

            # remove the previous peak from the fft spectrum
            self._remove_peak(ind, nu)
            self._remove_peak(self.size - ind, 1 - nu)


#########################################################################################
# COSINE
#########################################################################################


def naff_cos(z, a_k, return_coeff=False):
    size = z.shape[0]
    x = np.arange(size) / size
    weights = np.sum([a_k[k] * np.cos(2*np.pi*x*k)
                      for k in range(a_k.shape[0])], axis=0)
    fft = np.fft.fft(weights * z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)

    # choose 'ind' closer to true peak (with 1e-6 tolerance)
    flag = False
    if abs_fft[ind - 1] > abs_fft[(ind + 1) % size] + 1e-6:
        flag = True

    if flag:
        ratio = abs_fft[ind] / abs_fft[(ind - 1) % size]
    else:
        ratio = abs_fft[ind] / abs_fft[(ind + 1) % size]

    def fj_cos(eps, a_k=a_k, size=size):
        res = 2.0 * (a_k[0] * (eps**2 - 1/size**2) + a_k[1] * eps**2)
        fac = eps * (eps - 1/size) * (eps + 1/size)
        corr = 0.0
        for k in range(2, a_k.shape[0], 1):
            corr += a_k[k] * (1 / (eps - k/size) + 1 / (eps + k/size))

        return res + fac * corr

    def root(eps, ratio, a_k=a_k, size=size):
        fj = fj_cos(eps, a_k, size)
        fj_plus1 = fj_cos(eps - 1/size, a_k, size)
        fac = (eps + 1/size) / (eps - 2/size)
        return ratio * np.abs(fac) - np.abs(fj / fj_plus1)

    delta = 1e-12 #1e-3/size
    a = delta
    b = 1/size - delta
    if root(a, ratio) * root(b, ratio) < 0.0:
        eps = brentq(root, a, b, args=(ratio), xtol=1e-15)
        # print(f"success positive eps = {eps:.2e}", flag)
        # print("Success: ratio = ", ratio, size)
    elif root(-a, ratio) * root(-b, ratio) < 0.0:
        eps = brentq(root, -b, -a, args=(ratio), xtol=1e-15)
        # print(f"success negative eps = {eps:.2e}", flag)
        # print("Success: ratio = ", ratio, size)
    else:
        eps = 0.0
        print("failure warning, eps set to zero")
        print("Failure: ratio = ", ratio, size)
    # print()

    # if flag:
    #     eps = brentq(root, -b, -a, args=(ratio))
    #     print(f"success negative eps = {eps:.2e}", flag)
    #     print("Success: ratio = ", ratio, size)
    # else:
    #     eps = brentq(root, a, b, args=(ratio))
    #     print(f"success positive eps = {eps:.2e}", flag)
    #     print("Success: ratio = ", ratio, size)
    # eps = brentq(root, delta, 1/size - delta, args=(ratio))

    if flag:
        nu = ind/size - eps
    else:
        nu = ind/size + eps

    if return_coeff:
        coeff = 2 * fft[ind] / size
        if np.abs(eps) > 1e-12:
            coeff *= 2j*np.pi * size * eps / (np.exp(2j*np.pi * eps * size) - 1)

        corr = np.sum([a_k[k] * (1 / (eps - k/size) + 1 / (eps + k/size))
                       for k in range(1, a_k.shape[0], 1)])
        coeff /= (2*a_k[0] + eps * corr)

    return nu

def remove_peak_cos(abs_fft, eps, ind, a_k):
    size = abs_fft.shape[0]
    ampl = abs_fft[ind]
    corr = np.sum([a_k[k] * (1 / (eps - k/size) + 1 / (eps + k/size))
                   for k in range(1, a_k.shape[0], 1)])
    ampl /= 2*a_k[0] + eps * corr

    for j in range(-a_k.shape[0]+1, a_k.shape[0], 1):
        if j == 0:
            continue
        corr = np.sum([a_k[np.abs(k)] / (eps - (k+j)/size)
                        for k in range(-a_k.shape[0]+1, a_k.shape[0], 1)
                        if k != -j])
        corr = eps * corr + a_k[np.abs(j)] + a_k[0] * eps / (eps - j/size)
        # corr = np.sum([a_k[np.abs(k)] / (eps - (k+j)/size)
        #                for k in range(1-a_k.shape[0], a_k.shape[0], 1) ])
        # corr = np.sum([a_k[k] * (1 / (eps - (k+j)/size) + 1 / (eps + (k-j)/size) )
        #                for k in range(0, a_k.shape[0], 1) ])
        # corr = eps * corr
        abs_fft[(ind + j) % size] -= ampl * np.abs(corr)
    abs_fft[ind] = 0.0


def naffnd_cos(z, n_freq=1, a_k=1, return_coeff=False):
    """
    NaffND with an arbitrary cosine-series window filter.
    Computes the first 'n_freq' fundamental frequencies of the complex-valued
        input signal 'z', as well as their respective complex amplitudes.
    Parameters:
        n_freq == number of frequenices to be computed until max(fft) < 1e-15
                  (iterative process is aborted if the fft becomes 'flat')
        a_k    == set of weights for a cosine-series acting as the window.
                  May instead provide a positive integer to generate
                  the 'a_k'-th order Hanning-Window
        return_coeff == whether to return the computed frequency amplitudes
        Offset == whether to compute, return and adjust a signal offset
    """
    size = z.shape[0]
    if isinstance(a_k, int):
        a_k = hann_coeff(a_k)

    x = np.arange(size) / size    # x = 0,...,1 - 1/size
    weights = np.sum([a_k[k] * np.cos(2*np.pi*x*k)
                      for k in range(a_k.shape[0])], axis=0)
    fft = np.fft.fft(weights * z)   # need complex-valued fft for coeff.
    abs_fft = np.abs(fft)           # only need absolute values for freqs.

    # frequencies and their respective amplitudes
    nu_arr = np.zeros(n_freq)
    c_arr = np.zeros(n_freq, dtype=np.complex128)

    def fj_cos(eps, a_k=a_k, size=size):
        res = 2.0 * (a_k[0] * (eps**2 - 1/size**2) + a_k[1] * eps**2)
        fac = eps * (eps - 1/size) * (eps + 1/size)
        corr = 0.0
        for k in range(2, a_k.shape[0], 1):
            corr += a_k[k] * (1 / (eps - k/size) + 1 / (eps + k/size))

        return res + fac * corr

    def root(eps, ratio, a_k=a_k, size=size):
        fj = fj_cos(eps, a_k, size)
        fj_plus1 = fj_cos(eps - 1/size, a_k, size)
        fac = (eps + 1/size) / (eps - 2/size)
        return ratio * np.abs(fac) - np.abs(fj / fj_plus1)

    delta = 1e-12 #1e-3/size
    # a = delta
    b = 1/size - delta

    for i in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-15:
            break   # abort if fft-peak too small -> nu == 0

        # choose 'ind' closer to true peak (with 1e-6 tolerance)
        flag = False
        if abs_fft[ind - 1] > abs_fft[(ind + 1) % size] + 1e-6:
            flag = True

        if flag:
            ratio = abs_fft[ind] / abs_fft[(ind - 1) % size]
        else:
            ratio = abs_fft[ind] / abs_fft[(ind + 1) % size]

        if root(-b, ratio) * root(b, ratio) < 0.0:
            eps = brentq(root, -b, b, args=(ratio), xtol=1e-15)
        # if root(a, ratio) * root(b, ratio) < 0.0:
        #     eps = brentq(root, a, b, args=(ratio), xtol=1e-15)
        #     print("1:", eps, flag)
        # elif root(-a, ratio) * root(-b, ratio) < 0.0:
        #     eps = brentq(root, -b, -a, args=(ratio), xtol=1e-15)
        #     print("2:", eps, flag)
        else:
            eps = 0.0
            print("failure warning, eps set to zero")
            print("Failure: ratio = ", ratio, size)

        if flag:
            nu = ind/size - eps
        else:
            nu = ind/size + eps

        nu_arr[i] = nu

        # compute the complex value of the frequency amplitude
        if return_coeff:
            coeff = 2 * fft[ind] / size
            if np.abs(eps) > 1e-12:
                coeff *= 2j*np.pi * size * eps / (np.exp(2j*np.pi * eps * size) - 1)

            corr = np.sum([a_k[k] * (1 / (eps - k/size) + 1 / (eps + k/size))
                            for k in range(1, a_k.shape[0], 1)])

            c_arr[i] = coeff / (2*a_k[0] + eps * corr)

        # remove the previous peak from the fft spectrum
        remove_peak_cos(abs_fft, ind=ind, eps=eps, a_k=a_k)

        # also remove the mirrored frequency?
        remove_peak_cos(abs_fft, ind=size-ind, eps=(-1) * eps, a_k=a_k)
    if return_coeff:
        return nu_arr, c_arr
    return nu_arr


class NaffND_cos:
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

        a_k      == set of weights for a cosine-series acting as the window.
                   May instead provide a positive integer to generate
                   the 'a_k'-th order Hanning-Window

        Offset  == boolean, whether to compensate a nonzero signal-offset
                   Behavior beyond a certain signal length (size >~ 1e2...1e3)
                       this has no effect on the precision, as compared to
                       manually removing the offset
                       (possible in special cases)

        max_components == [OPTIONAL] maximal number of components
                   Use this in the ambiguous case of a 2d-signal-array,
                   as this could refer to ::
                       1. many signals with one component (default)
                       2. a single signal with multiple components
                   Specify the 'max_components' to ensure case (2)
    """
    def __init__(self, signals, n_freq=2, a_k=3, component=0, max_components=1):
        """Only accepts complex valued input signals."""
        if signals.dtype != complex:
            msg = ("Input 'signals' should be of type 'complex' "
                  + f"but is of type {signals.dtype} !")
            raise TypeError(msg)

        self.n_freq = n_freq
        if isinstance(a_k, int):
            self.a_k = hann_coeff(a_k)
        else:
            self.a_k = a_k
        self.max_components = max_components
        self.component = component

        self.signals = signals

        self.signals_shape = np.shape(signals)
        self.size = self.signals_shape[-1]          # signal length should be innermost
        self.num_signals = self.signals_shape[0]    # number of different signals first

        # catch the 1d and 2d case
        self.z = self._convert_validate_signal()
        self.num_signals = self.z.shape[0]

        # used to extract the np.argmax result ::
        # let 'a' be a 2d-array and 'ind = np.argmax(a, axis=1)'
        # then to access the maxima of 'a' use the syntax 'a[row_ind, ind]'
        # see 'https://stackoverflow.com/questions/14222110/extracting-
        # numpy-array-slice-from-numpy-argmax-results' for reference
        self.row_ind = np.arange(self.num_signals)

        # setup frequencies and corresponding coefficients
        self.freq = np.zeros((self.num_signals, n_freq), dtype=np.float64)
        self.coeff = np.zeros((self.num_signals, n_freq), dtype=np.complex128)

        self._fft_setup()

    def _convert_validate_signal(self):
        ndim = np.ndim(self.signals)
        if (ndim > 3) or (ndim < 1):
            msg = (f"Error: number of signal dimensions was {ndim}, "
                   + "but should instead be one of (1, 2, 3) !")
            raise IndexError(msg)
        elif ndim == 3:
            return self.signals[:, self.component, :]
        elif ndim == 1:
            return np.expand_dims(self.signals, axis=0)
        elif ndim == 2:     # handle ambiguous 2d case
            if self.num_signals == self.max_components:
                self.num_signals = 1    # input is a single signal with many components
                return np.expand_dims(self.signals[self.component, :], axis=0)
            
            if self.num_signals <= self.n_freq:
                print("Warning, 2d-input signal understood as a list of "
                      + "1d signals. Use 'max_components' to indicate "
                      + "a single signal with multiple components!")
            return self.signals

    def _remove_peak(self, ind, eps):
        """Remove a gaussian peak from a FFT at a given frequency"""
        ampl = self.abs_fft[self.row_ind, ind]
        corr = np.sum([self.a_k[k] * (1 / (eps - k/self.size)
                                     + 1 / (eps + k/self.size))
                       for k in range(1, self.a_k.shape[0], 1)], axis=0)
        ampl /= 2*self.a_k[0] + eps * corr

        for j in range(-self.a_k.shape[0]+1, self.a_k.shape[0], 1):
            if j == 0:
                continue
            corr = np.sum([self.a_k[np.abs(k)] / (eps - (k+j)/self.size)
                            for k in range(-self.a_k.shape[0]+1,
                                           self.a_k.shape[0], 1)
                            if k != -j], axis=0)
            corr = (eps * corr + self.a_k[np.abs(j)]
                    + self.a_k[0] * eps / (eps - j/self.size))

            self.abs_fft[self.row_ind, (ind+j) % self.size] -= ampl * np.abs(corr)
        self.abs_fft[self.row_ind, ind] = 0.0

    def _fj_cos(self, eps):
        res = 2.0 * (self.a_k[0] * (eps**2 - 1/self.size**2) + self.a_k[1] * eps**2)
        fac = eps * (eps - 1/self.size) * (eps + 1/self.size)
        corr = 0.0
        for k in range(2, self.a_k.shape[0], 1):
            corr += self.a_k[k] * (1 / (eps - k/self.size) + 1 / (eps + k/self.size))

        return res + fac * corr

    def _root(self, eps, ratio):
        fj = self._fj_cos(eps)
        fj_plus1 = self._fj_cos(eps - 1/self.size)
        fac = (eps + 1/self.size) / (eps - 2/self.size)
        return ratio * np.abs(fac) - np.abs(fj / fj_plus1)

    def _fft_setup(self):
        """Set up the FFT"""
        self.t = np.arange(self.size)
        self.weights = np.sum([self.a_k[k] * np.cos(2*np.pi*k * self.t / self.size)
                               for k in range(self.a_k.size)], axis=0)
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
            flag = (self.abs_fft[self.row_ind, ind - 1] >
                    (self.abs_fft[self.row_ind, (ind + 1) % self.size]
                     + peak_tolerance))

            # avoid index error in the case of 'ind == size-1'
            ratio = (self.abs_fft[self.row_ind, ind]
                     / self.abs_fft[self.row_ind, (ind + (-1)**flag) % self.size])

            # computing the frequency using root-finding
            # vectorization of this step is very inefficient, refer to c++
            delta = 1e-12 #1e-3/size
            a = delta
            b = 1/self.size - delta
            # indx = (self._root(a, ratio) * self._root(b, ratio) < 0.0)
            # indx2 = (self._root(-a, ratio) * self._root(-b, ratio) < 0.0)

            eps = np.zeros(ratio.shape[0])
            for l in range(ratio.shape[0]):
                if self._root(a, ratio[l]) * self._root(b, ratio[l]) < 0.0:
                    eps[l] = brentq(self._root, a, b, args=ratio[l], xtol=1e-15)
                elif self._root(-a, ratio) * self._root(-b, ratio[l]) < 0.0:
                    eps[l] = brentq(self._root, -b, -a, args=ratio[l], xtol=1e-15)
                else:
                    eps[l] = 0.0
                    print(f"Failure warning, eps[{l = }] set to zero")
                    print(f"Failure: {ratio = }", self.size)

            nu = ind/self.size + (-1)**flag * eps
            self.freq[:, i] = nu


            # FIXME: amplitudes of the coefficients are always correct,
            #        but phase is incompatible with gauss-naff results
            # compute the complex value of the frequency amplitude
            coeff = 2 * self.fft[self.row_ind, ind] / self.size
            coeff[np.abs(eps) > 1e-12] *= \
                2j*np.pi * self.size * eps / (np.exp(2j*np.pi * eps * self.size) - 1)

            corr = np.sum([self.a_k[k] * (1 / (eps - k/self.size)
                                         + 1 / (eps + k/self.size))
                           for k in range(1, self.a_k.shape[0], 1)], axis=0)
            coeff /= (2*self.a_k[0] + eps * corr)
            self.coeff[:, i] = coeff

            # remove the previous peak from the fft spectrum
            self._remove_peak(ind, eps)
            self._remove_peak(self.size - ind, -eps)


#########################################################################################
# NUMERICAL
#########################################################################################


# def naff_num(z, w_method):
#     weights = w_method(np.arange(z.size))
#     w_z = weights * z
#     fft = np.fft.fft(w_z)
#     abs_fft = np.abs(fft)
#     nu_init = np.argmax(abs_fft) / z.size    # initial guess
#     n_range = np.arange(z.size)

#     def minimizer(nu, w_z, n_range):
#         # return 1 / np.abs(np.sum(w_z * np.exp(-2*np.pi*1j * nu * n_range)))
#         c_seq = w_z * np.exp(-2*np.pi*1j * nu * n_range)
#         real = np.math.fsum(c_seq.real)
#         imag = np.math.fsum(c_seq.imag)
#         return 1 / np.abs(real**2 + imag**2)

#     delta = 1 / z.size
#     bracket = nu_init + np.array([-delta, 0.0, delta])
#     nu = minimize_scalar(minimizer, bracket=bracket, args=(w_z, n_range),
#                          tol=1e-15).x
#     return nu


def _fj_eps_num(eps, weights):
    """Model Fourier coefficients"""
    n_range = np.arange(weights.size)
    return np.sum(weights * np.exp(2*np.pi*1j * n_range * eps))

def _f_eps_num(eps, weights, ratio):
    """Function of which we want to find a root."""
    return (np.abs(_fj_eps_num(eps - 1/weights.size, weights))
            - np.abs(ratio * _fj_eps_num(eps, weights)))


def naff_num(z, w_method):
    """
    sizeumerical NAFF using the integral approximation and 'quad'.

    Compute exact Fourier coefficients of the signal (F_j, j=0,...,N-1)
    Compute model coefficients (F_j^M = const * int_0^1 w(x) * exp(2pi*i * eps * x))

    Assume that 'R := F_{j+1} / F_j' is roughly 'R^M := F_{j+1}^M / F_j^M'
    and numerically solve $R * F_j^M = F_{j+1}^M$ for 'eps'

    This is done by minimizing the absolute value of
        'int_0^1 w(x) exp(2pi*i * eps * x) * [exp(-2pi*i * x/N) - R]'
    """
    weights = w_method(np.arange(z.size) / z.size)
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)

    if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
        ind -= 1

    ratio = fft[(ind + 1) % z.size] / fft[ind]
    # def root_expression(eps):
    #     """Function of which we want to find a root."""
    #     def integral(eps):
    #         n_range = np.arange(z.size)
    #         return np.sum(w_method(n_range / z.size) * np.exp(2*np.pi*1j * n_range * eps))

    #     return np.abs(integral(eps - 1/z.size)) - np.abs(ratio * integral(eps))

    nu_init = ind / z.size    # initial guess
    delta = 1 / z.size
    eps = brentq(_f_eps_num, 0, delta, xtol=1e-15, args=(weights, ratio))
    nu = nu_init + eps
    return nu


def _remove_peak_num(abs_fft, ind, nu, weights, num_j=10):
    """Remove a gaussian peak from a FFT at a given frequency"""
    size = abs_fft.size
    ampl = abs_fft[(ind) % size] / _fj_eps_num(nu - ind/size, weights)
    for j in range(ind - num_j + 1, ind + num_j, 1):
        abs_fft[j % size] -= np.abs(ampl * _fj_eps_num(nu - j/size, weights))


def naffnd_num(z, w_method, n_freq=1, num_j=10, return_coeff=False):
    """
    NaffND with arbitrary weights.
    Computes the first 'n_freq' fundamental frequencies of the complex-valued
        input signal 'z', as well as their respective complex amplitudes.
    Parameters:
        n_freq == number of frequenices to be computed until max(fft) < 1e-15
                  (iterative process is aborted if the fft becomes 'flat')
        w_method == arbitrary function returning weights for inputs in [0, 1]
        num_j == for the '2*num_j - 1' points closest to the relevant peak in the fft,
                 the respective fft-value is subtracted based on an approximate
                 analytical formula
        return_coeff == whether to return the computed frequency amplitudes
    """
    weights = w_method(np.arange(z.size) / z.size)
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)

    # frequencies and their respective amplitudes
    nu_arr = np.zeros(n_freq)
    c_arr = np.zeros(n_freq, dtype=np.complex128)

    for ctr in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-15:
            break   # abort if fft-peak too small -> nu == 0

        if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
            ind -= 1

        ratio = fft[(ind + 1) % z.size] / fft[ind]

        nu_init = ind / z.size    # initial guess
        delta = 1 / z.size
        eps = brentq(_f_eps_num, -delta, delta,
                     xtol=1e-15, args=(weights, ratio))
        nu_k = nu_init + eps
        nu_arr[ctr] = nu_k

        # compute the complex value of the frequency amplitude
        c_k = fft[ind] / _fj_eps_num(eps, weights)
        c_arr[ctr] = c_k

        # remove the previous peak from the fft spectrum
        _remove_peak_num(abs_fft, ind, nu_k, weights, num_j=num_j)
        _remove_peak_num(abs_fft, z.size - ind, 1 - nu_k, weights, num_j=num_j)

    if return_coeff:
        return nu_arr, c_arr
    return nu_arr


def naff_num_int(z, w_method):
    """
    Numerical NAFF using the integral approximation and 'quad'.

    Compute exact Fourier coefficients of the signal (F_j, j=0,...,N-1)
    Compute model coefficients (F_j^M = const * int_0^1 w(x) * exp(2pi*i * eps * x))

    Assume that 'R := F_{j+1} / F_j' is roughly 'R^M := F_{j+1}^M / F_j^M'
    and numerically solve $R * F_j^M = F_{j+1}^M$ for 'eps'

    This is done by minimizing the absolute value of
        'int_0^1 w(x) exp(2pi*i * eps * x) * [exp(-2pi*i * x/N) - R]'
    """
    weights = w_method(np.arange(z.size) / z.size)
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)

    if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
        ind -= 1

    ratio = fft[(ind + 1) % z.size] / fft[ind]

    def root_expression(eps):
        """Function of which we want to find a root."""
        def integral(eps):
            def f_real(x):
                return w_method(x) * np.cos(2*np.pi * x * z.size * eps)
            def f_imag(x):
                return w_method(x) * np.sin(2*np.pi * x * z.size * eps)
            real = quad(f_real, 0.0, 1.0)[0]
            imag = quad(f_imag, 0.0, 1.0)[0]
            return real + 1j * imag

        return np.abs(integral(eps - 1/z.size)) - np.abs(ratio * integral(eps))

    nu_init = ind / z.size    # initial guess
    delta = 1 / z.size

    # root finding approach
    # eps = brentq(root_expression, -delta, delta, xtol=1e-15)
    eps = brentq(root_expression, 0, delta, xtol=1e-15)
    nu = nu_init + eps
    return nu


#########################################################################################
# LASKAR (old method and integral approximation)
#########################################################################################


def naff_laskar(z):
    """Old method for NAFF due to Laskar"""
    weights = hann_weights(np.arange(z.size) / z.size, a_k=1)
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)

    if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
        ind -= 1

    c = np.cos(2*np.pi / z.size)

    # ratio = abs_fft[ind] / abs_fft[(ind + 1) % z.size]
    # root = np.sqrt(c**2 * (ratio + 1)**2 - 2 * ratio * (2*c**2 - c - 1))
    # num = (ratio + c) * (1 - ratio) + np.sign(ratio + c) * root
    ratio = abs_fft[(ind + 1) % z.size] / abs_fft[ind]
    root = np.sqrt(ratio**2 * c**2 * (ratio + 1)**2 - 2 * ratio**3 * (2*c**2 - c - 1))
    num = (1 + ratio * c) * (ratio - 1) + np.sign(1 + ratio * c) * root
    y_par = num / (ratio**2 + 1 + 2 * ratio * c)

    nu = ind / z.size + np.arcsin(y_par * np.sin(2 * np.pi/z.size)) / (2*np.pi)
    return nu


def naff_laskar_approx(z):
    """Naff 1d with hanning-window weights; approximation"""
    weights = hann_weights(np.arange(z.size) / z.size, a_k=1)
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)
    ind = np.argmax(abs_fft)

    if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
        ind -= 1

    ratio = abs_fft[(ind + 1) % z.size] / abs_fft[ind]
    nu = ind / z.size + 1 / z.size * (2 * ratio - 1) / (ratio + 1)
    return nu


#########################################################################################
# ITERATIVE METHODS
#########################################################################################

"""
ratio = fft[(ind + 1) % z.size] / fft[ind]
eps = brentq(_g_model_num, -delta, delta, xtol=1e-15, 
             args=(delta, 0.0, weights, ratio))

ratio1 = dft_f(w_z, nu_init + eps)[0] / fft[ind]
eps1 = brentq(_g_model_num, -delta, delta, xtol=1e-15, 
              args=(eps, 0.0, weights, ratio1))

ratio2 = dft_f(w_z, nu_init + eps1)[0] / dft_f(w_z, nu_init + eps)[0]
eps2 = brentq(_g_model_num, -delta, delta, xtol=1e-15, 
              args=(eps1, eps, weights, ratio2))

eps_range = np.linspace(-delta, delta, 50)
g_vals = [_g_model_num(eps_val, delta, weights, ratio) for eps_val in eps_range]
plt.plot(eps_range, g_vals)
"""


def dft_f(z, nu):
    z = np.asarray(z)
    if isinstance(nu, (int, float)):
        nu = np.array([nu])
    nu = np.asarray(nu)
    return np.array([np.sum(z * np.exp(-2*np.pi*1j * np.arange(z.size) * nu_val)) 
                     for nu_val in nu])


def _f_model_num(eps, weights):
    """Model Fourier coefficients"""
    n_range = np.arange(weights.size)
    return np.sum(weights * np.exp(2*np.pi*1j * n_range * eps))


def _g_model_num(eps, offset_upper, offset_lower, weights, ratio):
    """eps := nu - j/N, --> F^M_{j+1}(eps) == F_j^M(eps - 1/N)"""
    f_offset = _f_model_num(eps - offset_upper, weights)
    f_main = _f_model_num(eps - offset_lower, weights)
    return np.abs(f_offset) - np.abs(ratio * f_main)


# def abs_dft(w_z, nu):
#     n_range = np.arange(w_z.size)
#     phase = np.exp(-n_range * 2j * np.pi * nu)
#     dft = np.sum(w_z * phase)
#     return np.sqrt(dft.real * dft.real + dft.imag * dft.imag)


# def abs_dft_prime(w_z, nu):
#     n_range = np.arange(w_z.size)
#     phase = n_range * 2 * np.pi * nu
#     cos = np.cos(phase)
#     sin = np.sin(phase)
#     dft_real = np.sum(w_z.real * cos + w_z.imag * sin)
#     dft_real_prime = np.sum(n_range * (-w_z.real * sin + w_z.imag * cos))
#     dft_imag = np.sum(w_z.imag * cos - w_z.real * sin)
#     dft_imag_prime = np.sum(n_range * (-w_z.imag * sin - w_z.real * cos))
#     return dft_real * dft_real_prime + dft_imag * dft_imag_prime


def optimize_dft(w_z, nu_est, err_est=1e-6, xtol=1e-15, max_iter=50):
    phase = -2*np.pi*1j * np.arange(w_z.size)
    def abs_dft(nu):
        return np.abs(np.sum(w_z * np.exp(phase * nu)))
        
    left = abs_dft(nu_est - err_est)
    middle = abs_dft(nu_est)
    right = abs_dft(nu_est + err_est)
    
    if middle < left and middle < right:
        print("Optimal frequency does not lie within given error estimate")
        return nu_est
        
    
