"""
Numerical analysis of fundamental frequencies -- high order hanning windows
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
from scipy.optimize import brentq
special.setup(UseTex=False)

def hann_coeff(M):
    """Returns the cosine series coefficients for the M-th hanning window"""
    ak = np.zeros(M+1)
    ak[0] = np.math.comb(2*M, M)
    for k in range(1, M+1, 1):
        ak[k] = 2 * (-1)**k * np.math.comb(2*M, M-k)
    return ak / (4**M)


ak = np.array([0.21557895, -0.41653158, 0.277263158,
               -0.083578947, 0.006947368])
ak = np.array([0.5, -0.5])
ak = np.array([3, -4, 1]) / 8.0
ak = hann_coeff(14)
# ak = np.array([1, -1.942604, 1.340318, -0.440811, 0.043097])
def naff_cos(z, ak=ak, RemoveOffset=True, ComputeCoeff=False):
    N = z.shape[0]
    #ak /= N
    x = np.arange(N) / N
    weights = np.sum([ak[k] * np.cos(2*np.pi*x*k)
                      for k in range(ak.shape[0])], axis=0)
    fft = np.fft.fft(weights * z)
    abs_fft = np.abs(fft)

    if RemoveOffset:
        #coeff = fft[0] / (N*ak[0])
        abs_fft[1:ak.shape[0]] -= abs_fft[0] * np.abs(ak[1:]) / (2*ak[0])
        abs_fft[-ak.shape[0]+1:] -= abs_fft[0] * np.abs(ak[:0:-1]) / (2*ak[0])
        abs_fft[0] = 0.0


    ind = np.argmax(abs_fft)

    # choose 'ind' closer to true peak (with 1e-6 tolerance)
    flag = False
    if abs_fft[ind - 1] > abs_fft[(ind + 1) % N] + 1e-6:
        flag = True

    if flag:
        R = abs_fft[ind] / abs_fft[(ind - 1) % N]
    else:
        R = abs_fft[ind] / abs_fft[(ind + 1) % N]

    def Fj_cos(eps, ak=ak, N=N):
        res = 2.0 * (ak[0] * (eps**2 - 1/N**2) + ak[1] * eps**2)
        fac = eps * (eps - 1/N) * (eps + 1/N)
        corr = 0.0
        for k in range(2, ak.shape[0], 1):
            corr += ak[k] * (1 / (eps - k/N) + 1 / (eps + k/N))

        return res + fac * corr

    def root(eps, R, ak=ak, N=N):
        Fj = Fj_cos(eps, ak, N)
        Fj_plus1 = Fj_cos(eps - 1/N, ak, N)
        fac = (eps + 1/N) / (eps - 2/N)
        return R * np.abs(fac) - np.abs(Fj / Fj_plus1)

    delta = 1e-12 #1e-3/N
    a = delta
    b = 1/N - delta
    if root(a, R) * root(b, R) < 0.0:
        eps = brentq(root, a, b, args=(R), xtol=1e-15)
        # print(f"success positive eps = {eps:.2e}", flag)
        # print("Success: R = ", R, N)
    elif root(-a, R) * root(-b, R) < 0.0:
        eps = brentq(root, -b, -a, args=(R), xtol=1e-15)
        # print(f"success negative eps = {eps:.2e}", flag)
        # print("Success: R = ", R, N)
    else:
        eps = 0.0
        print("failure warning, eps set to zero")
        print("Failure: R = ", R, N)
    # print()

    # if flag:
    #     eps = brentq(root, -b, -a, args=(R))
    #     print(f"success negative eps = {eps:.2e}", flag)
    #     print("Success: R = ", R, N)
    # else:
    #     eps = brentq(root, a, b, args=(R))
    #     print(f"success positive eps = {eps:.2e}", flag)
    #     print("Success: R = ", R, N)
    # eps = brentq(root, delta, 1/N - delta, args=(R))

    if flag:
        nu = ind/N - eps
    else:
        nu = ind/N + eps

    if ComputeCoeff:
        coeff = 2 * fft[ind] / N
        if np.abs(eps) > 1e-12:
            coeff *= 2j*np.pi * N * eps / (np.exp(2j*np.pi * eps * N) - 1)

        corr = np.sum([ak[k] * (1 / (eps - k/N) + 1 / (eps + k/N))
                       for k in range(1, ak.shape[0], 1)])
        coeff /= (2*ak[0] + eps * corr)

    return nu

def remove_peak(abs_fft, eps, ind, ak=ak):
    N = abs_fft.shape[0]
    ampl = abs_fft[ind]
    corr = np.sum([ak[k] * (1 / (eps - k/N) + 1 / (eps + k/N))
                   for k in range(1, ak.shape[0], 1)])
    ampl /= 2*ak[0] + eps * corr

    for j in range(-ak.shape[0]+1, ak.shape[0], 1):
        if j == 0:
            continue
        corr = np.sum([ak[np.abs(k)] / (eps - (k+j)/N)
                        for k in range(-ak.shape[0]+1, ak.shape[0], 1)
                        if k != -j])
        corr = eps * corr + ak[np.abs(j)] + ak[0] * eps / (eps - j/N)
        # corr = np.sum([ak[np.abs(k)] / (eps - (k+j)/N)
        #                for k in range(1-ak.shape[0], ak.shape[0], 1) ])
        # corr = np.sum([ak[k] * (1 / (eps - (k+j)/N) + 1 / (eps + (k-j)/N) )
        #                for k in range(0, ak.shape[0], 1) ])
        # corr = eps * corr
        abs_fft[(ind + j) % N] -= ampl * np.abs(corr)
    abs_fft[ind] = 0.0


def naffnd_cos(z, n_freq=1, ak=ak, ReturnCoeff=False, Offset=False):
    """
    NaffND with an arbitrary cosine-series window filter.
    Computes the first 'n_freq' fundamental frequencies of the complex-valued
        input signal 'z', as well as their respective complex amplitudes.
    Parameters:
        n_freq == number of frequenices to be computed until max(fft) < 1e-15
                  (iterative process is aborted if the fft becomes 'flat')
        ak     == set of weights for a cosine-series acting as the window.
                  May instead provide a positive integer to generate
                  the 'ak'-th order Hanning-Window
        ReturnCoeff == whether to return the computed frequency amplitudes
        Offset == whether to compute, return and adjust a signal offset
    """
    N = z.shape[0]
    if type(ak) == int:
        ak = hann_coeff(ak)

    x = np.arange(N) / N    # x = 0,...,1-1/N
    weights = np.sum([ak[k] * np.cos(2*np.pi*x*k)
                      for k in range(ak.shape[0])], axis=0)
    fft = np.fft.fft(weights * z)   # need complex-valued fft for coeff.
    abs_fft = np.abs(fft)           # only need absolute values for freqs.

    # frequencies and their respective amplitudes
    nu_arr = np.zeros(n_freq + Offset)
    c_arr = np.zeros(n_freq + Offset, dtype=np.complex128)

    if Offset:    # compute constant offset and adjust fft spectrum
        c_arr[n_freq] = fft[0] / (N*ak[0])
        remove_peak(abs_fft, ind=0, eps=0.0, ak=ak)

    def Fj_cos(eps, ak=ak, N=N):
        res = 2.0 * (ak[0] * (eps**2 - 1/N**2) + ak[1] * eps**2)
        fac = eps * (eps - 1/N) * (eps + 1/N)
        corr = 0.0
        for k in range(2, ak.shape[0], 1):
            corr += ak[k] * (1 / (eps - k/N) + 1 / (eps + k/N))

        return res + fac * corr

    def root(eps, R, ak=ak, N=N):
        Fj = Fj_cos(eps, ak, N)
        Fj_plus1 = Fj_cos(eps - 1/N, ak, N)
        fac = (eps + 1/N) / (eps - 2/N)
        return R * np.abs(fac) - np.abs(Fj / Fj_plus1)

    delta = 1e-12 #1e-3/N
    a = delta
    b = 1/N - delta

    for i in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-15:
            break   # abort if fft-peak too small -> nu == 0

        # choose 'ind' closer to true peak (with 1e-6 tolerance)
        flag = False
        if abs_fft[ind - 1] > abs_fft[(ind + 1) % N] + 1e-6:
            flag = True

        if flag:
            R = abs_fft[ind] / abs_fft[(ind - 1) % N]
        else:
            R = abs_fft[ind] / abs_fft[(ind + 1) % N]

        if root(-b, R) * root(b, R) < 0.0:
            eps = brentq(root, -b, b, args=(R), xtol=1e-15)
        # if root(a, R) * root(b, R) < 0.0:
        #     eps = brentq(root, a, b, args=(R), xtol=1e-15)
        #     print("1:", eps, flag)
        # elif root(-a, R) * root(-b, R) < 0.0:
        #     eps = brentq(root, -b, -a, args=(R), xtol=1e-15)
        #     print("2:", eps, flag)
        else:
            eps = 0.0
            print("failure warning, eps set to zero")
            print("Failure: R = ", R, N)

        if flag:
            nu = ind/N - eps
        else:
            nu = ind/N + eps

        nu_arr[i] = nu

        # compute the complex value of the frequency amplitude
        if ReturnCoeff:
            coeff = 2 * fft[ind] / N
            if np.abs(eps) > 1e-12:
                coeff *= 2j*np.pi * N * eps / (np.exp(2j*np.pi * eps * N) - 1)

            corr = np.sum([ak[k] * (1 / (eps - k/N) + 1 / (eps + k/N))
                            for k in range(1, ak.shape[0], 1)])

            c_arr[i] = coeff / (2*ak[0] + eps * corr)

        # remove the previous peak from the fft spectrum
        remove_peak(abs_fft, ind=ind, eps=eps, ak=ak)

        # also remove the mirrored frequency?
        remove_peak(abs_fft, ind=N-ind, eps=-eps, ak=ak)
    if ReturnCoeff:
        return nu_arr, c_arr
    return nu_arr


class NaffND_cos(object):
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

        ak      == set of weights for a cosine-series acting as the window.
                   May instead provide a positive integer to generate
                   the 'ak'-th order Hanning-Window

        Offset  == boolean, whether to compensate a nonzero signal-offset
                   Behavior beyond a certain signal length (N >~ 1e2...1e3)
                       this has no effect on the precision, as compared to
                       manually removing the offset
                       (possible in special cases)

        maxComponents == [OPTIONAL] maximal number of components
                   Use this in the ambiguous case of a 2d-signal-array,
                   as this could refer to ::
                       1. many signals with one component (default)
                       2. a single signal with multiple components
                   Specify the 'maxComponents' to ensure case (2)
    """
    def __init__(self, signals, n_freq=2, ak=ak, component=0,
                 Offset=True, maxComponents=1):
        """Only accepts complex valued input signals."""
        if signals.dtype != complex:
            msg = ("Input 'signals' should be of type 'complex' "
                  + f"but is of type {signals.dtype} !")
            raise TypeError(msg)

        self.n_freq = n_freq
        if type(ak) == int:
            self.ak = hann_coeff(ak)
        else:
            self.ak = ak
        self.Offset = Offset
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

        if self.Offset:
            self._correctOffset()

    def _convertValidateSignal(self):
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
        self.coeff[:, self.n_freq] = self.fft[:, 0] / (self.N * self.ak[0])
        self._removePeak(ind=0, eps=0.0)

    def _removePeak(self, ind, eps):
        """Remove a gaussian peak from a FFT at a given frequency"""
        ampl = self.abs_fft[self.row_ind, ind]
        corr = np.sum([self.ak[k] * (1 / (eps - k/self.N)
                                     + 1 / (eps + k/self.N))
                       for k in range(1, self.ak.shape[0], 1)], axis=0)
        ampl /= 2*self.ak[0] + eps * corr

        for j in range(-self.ak.shape[0]+1, self.ak.shape[0], 1):
            if j == 0:
                continue
            corr = np.sum([self.ak[np.abs(k)] / (eps - (k+j)/self.N)
                            for k in range(-self.ak.shape[0]+1,
                                           self.ak.shape[0], 1)
                            if k != -j], axis=0)
            corr = (eps * corr + self.ak[np.abs(j)]
                    + self.ak[0] * eps / (eps - j/self.N))

            self.abs_fft[self.row_ind, (ind+j) % self.N] -= ampl * np.abs(corr)
        self.abs_fft[self.row_ind, ind] = 0.0

    def _Fj_cos(self, eps):
        res = 2.0 * (self.ak[0] * (eps**2 - 1/self.N**2) + self.ak[1] * eps**2)
        fac = eps * (eps - 1/self.N) * (eps + 1/self.N)
        corr = 0.0
        for k in range(2, self.ak.shape[0], 1):
            corr += self.ak[k] * (1 / (eps - k/self.N) + 1 / (eps + k/self.N))

        return res + fac * corr

    def _root(self, eps, R):
        Fj = self._Fj_cos(eps)
        Fj_plus1 = self._Fj_cos(eps - 1/self.N)
        fac = (eps + 1/self.N) / (eps - 2/self.N)
        return R * np.abs(fac) - np.abs(Fj / Fj_plus1)

    def _FFTsetup(self):
        """Set up the FFT"""
        self.t = np.arange(self.N)
        self.weights = np.sum([self.ak[k] * np.cos(2*np.pi*k * self.t / self.N)
                               for k in range(self.ak.size)], axis=0)
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
                    (self.abs_fft[self.row_ind, (ind + 1) % self.N]
                     + peak_tolerance))

            # avoid index error in the case of 'ind == N-1'
            R = (self.abs_fft[self.row_ind, ind]
                 / self.abs_fft[self.row_ind, (ind + (-1)**flag) % self.N])

            # computing the frequency using root-finding
            # vectorization of this step is very inefficient, refer to c++
            delta = 1e-12 #1e-3/N
            a = delta
            b = 1/self.N - delta
            # indx = (self._root(a, R) * self._root(b, R) < 0.0)
            # indx2 = (self._root(-a, R) * self._root(-b, R) < 0.0)

            eps = np.zeros(R.shape[0])
            for l in range(R.shape[0]):
                if self._root(a, R[l]) * self._root(b, R[l]) < 0.0:
                    eps[l] = brentq(self._root, a, b, args=R[l], xtol=1e-15)
                elif self._root(-a, R) * self._root(-b, R[l]) < 0.0:
                    eps[l] = brentq(self._root, -b, -a, args=R[l], xtol=1e-15)
                else:
                    eps[l] = 0.0
                    print(f"Failure warning, eps[{l = }] set to zero")
                    print(f"Failure: {R = }", self.N)

            nu = ind/self.N + (-1)**flag * eps
            self.freq[:, i] = nu


            # FIXME: amplitudes of the coefficients are always correct,
            #        but phase is incompatible with gauss-naff results
            # compute the complex value of the frequency amplitude
            coeff = 2 * self.fft[self.row_ind, ind] / self.N
            coeff[np.abs(eps) > 1e-12] *= \
                2j*np.pi * self.N * eps / (np.exp(2j*np.pi * eps * self.N) - 1)

            corr = np.sum([self.ak[k] * (1 / (eps - k/self.N)
                                         + 1 / (eps + k/self.N))
                           for k in range(1, self.ak.shape[0], 1)], axis=0)
            coeff /= (2*self.ak[0] + eps * corr)
            self.coeff[:, i] = coeff

            # remove the previous peak from the fft spectrum
            self._removePeak(ind, eps)
            self._removePeak(self.N - ind, -eps)

def main():
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

if __name__ == "__main__":
    print(__doc__)

    """
for j in range(ind - ak.shape[0]+1, ind + ak.shape[0], 1):
    print(f"\n {j = }:")
    print(abs_fft[j])
    val = 0.0
    for k in range(1, ak.shape[0], 1):
        val += ak[k] * (1 / (eps - (k+j)/N) + 1 / (eps + (k-j)/N))
    val *= abs_fft[ind] * np.sin(np.pi*nu*N)
    print(val)
    """
