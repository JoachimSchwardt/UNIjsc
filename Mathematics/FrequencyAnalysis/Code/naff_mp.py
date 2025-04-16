import numpy as np
import mpmath as mp
import window_functions_mp as win_mp
import naff


def dft_mp(signal, nu):
    res = signal[0]
    kappa = mp.exp(-2*mp.pi*1j * nu)
    exponential = kappa
    for i in range(1, len(signal)):
        res += signal[i] * exponential
        exponential *= kappa
    return res

def _fj_eps_num(eps, weights):
    """Model Fourier coefficients"""
    n_range = np.arange(weights.size)
    return np.sum(weights * np.exp(2*np.pi*1j * n_range * eps))

def _fj_eps_num_mp(eps, weights):
    """Model Fourier coefficients"""
    res = mp.mpc(weights[0], 0)
    kappa = mp.exp(2*mp.pi*1j * eps)
    exponential = kappa
    for i in range(1, len(weights)):
        res += weights[i] * exponential
        exponential *= kappa
    return res

def _f_eps_num_mp(eps, weights, ratio):
    """Function of which we want to find a root."""
    return (abs(ratio * _fj_eps_num_mp(eps - mp.mpf(1) / len(weights), weights))
            - abs(_fj_eps_num_mp(eps, weights)))

def _remove_peak_num(abs_fft, ind, nu, weights, num_j=10):
    """Remove a gaussian peak from a FFT at a given frequency"""
    size = abs_fft.size
    ampl = abs_fft[ind % size] / _fj_eps_num(nu - ind/size, weights)
    for j in range(ind - num_j + 1, ind + num_j, 1):
        abs_fft[j % size] -= np.abs(ampl * _fj_eps_num(nu - j/size, weights))

def naffnd_num_mp(z_mp, weights_mp, n_freq=1, num_j=10, return_coeff=False):
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
    weights = np.array(weights_mp, dtype=np.float64)
    z = np.array(z_mp, dtype=np.complex128)
    w_z_mp = [weights_mp[i] * z_mp[i] for i in range(z.size)]
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)

    # frequencies and their respective amplitudes
    nu_arr = [mp.mpf(0)] * n_freq
    c_arr = [mp.mpc(0j)] * n_freq

    for ctr in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-10:
            break   # abort if fft-peak too small -> nu == 0

        if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
            ind -= 1

        #ratio = fft[(ind + 1) % z.size] / fft[ind]
        ind_mp = mp.mpf(float(ind))
        dft_j_plus1 = dft_mp(w_z_mp, ((ind_mp+1) % z.size) / z.size)
        dft_j = dft_mp(w_z_mp, ind_mp / z.size)
        ratio = dft_j / dft_j_plus1
        nu_init = ind_mp / z.size    # initial guess
        delta = 1 / z.size
        # eps = mp.findroot(lambda x: _f_eps_num_mp(x, weights_mp, ratio), 
        #                   (-delta, delta), solver='anderson', tol=1e-27)
        eps = newton(_f_eps_num_mp, delta / 2, weights_mp, ratio, xtol=1e-27)
        nu_k = nu_init + eps
        nu_arr[ctr] = nu_k

        # compute the complex value of the frequency amplitude
        c_arr[ctr] = dft_j / _fj_eps_num_mp(eps, weights_mp)

        # remove the previous peak from the fft spectrum
        _remove_peak_num(abs_fft, ind, float(nu_k), weights, num_j=num_j)
        _remove_peak_num(abs_fft, z.size - ind, 1 - float(nu_k), weights, num_j=num_j)

    if return_coeff:
        return nu_arr, c_arr
    return nu_arr



###############################################################################
# NAFFND GAUSS
###############################################################################


def _fj_eps_gauss_mp(eps, alpha, n_points):
    """Model Fourier coefficients"""
    # phase = mp.mpc(n_points * 1j) * eps * mp.pi
    # res = mp.exp(phase * (1 + phase / alpha)) * mp.sqrt(mp.pi / alpha) * n_points
    phase = n_points * eps * mp.pi
    res = mp.exp(1j * phase) * mp.exp(-phase**2 / alpha) * mp.sqrt(mp.pi / alpha) * n_points
    return res

def _f_eps_gauss_mp(eps, alpha, n_points, ratio):
    """Function of which we want to find a root."""
    return (abs(ratio * _fj_eps_gauss_mp(eps - mp.mpf(1) / n_points, alpha, n_points))
            - abs(_fj_eps_gauss_mp(eps, alpha, n_points)))

def _remove_peak_gauss(abs_fft, ind, nu, alpha, n_points, num_j=10):
    """Remove a gaussian peak from a FFT at a given frequency"""
    size = abs_fft.size
    ampl = abs_fft[ind % size] / naff._fj_eps_gauss(nu - ind/size, alpha, n_points)
    for j in range(ind - num_j + 1, ind + num_j, 1):
        abs_fft[j % size] -= np.abs(ampl * naff._fj_eps_gauss(nu - j/size, alpha, n_points))

def naffnd_gauss_mp(z_mp, n_freq=1, alpha=280, num_j=10, return_coeff=False):
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
    weights_mp = win_mp.get_window(len(z_mp), win_mp.gauss_weights, alpha)
    weights = np.array(weights_mp, dtype=np.float64)
    z = np.array(z_mp, dtype=np.complex128)
    w_z_mp = [weights_mp[i] * z_mp[i] for i in range(z.size)]
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)

    # frequencies and their respective amplitudes
    nu_arr = [mp.mpf(0)] * n_freq
    c_arr = [mp.mpc(0j)] * n_freq

    for ctr in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-10:
            break   # abort if fft-peak too small -> nu == 0

        if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
            ind -= 1

        #ratio = fft[(ind + 1) % z.size] / fft[ind]
        ind_mp = mp.mpf(float(ind))
        dft_j_plus1 = dft_mp(w_z_mp, ((ind_mp+1) % z.size) / z.size)
        dft_j = dft_mp(w_z_mp, ind_mp / z.size)
        ratio = dft_j / dft_j_plus1
        eps = 1 / mp.mpf(2 * z.size) * (1 - alpha / mp.pi**2 * mp.log(abs(ratio)))
        nu_k = ind_mp / z.size + eps
        nu_arr[ctr] = nu_k

        # compute the complex value of the frequency amplitude
        c_arr[ctr] = dft_j / _fj_eps_gauss_mp(eps, alpha, z.size)

        # remove the previous peak from the fft spectrum
        _remove_peak_gauss(abs_fft, ind, float(nu_k), alpha, z.size, num_j=num_j)
        _remove_peak_gauss(abs_fft, z.size - ind, 1 - float(nu_k), alpha, z.size, num_j=num_j)

    if return_coeff:
        return nu_arr, c_arr
    return nu_arr



###############################################################################
# NAFFND COSINE
###############################################################################


def _fj_eps_cos_mp(eps, a_k, size):
    """Model Fourier coefficients"""
    res = mp.mpf(2) * (a_k[0] * (eps**2 - 1/size**2) + a_k[1] * eps**2)
    fac = eps * (eps - 1/size) * (eps + 1/size)
    corr = mp.mpf(0)
    for k in range(2, len(a_k)):
        corr += a_k[k] * (1 / (eps - k/size) + 1 / (eps + k/size))
    return res + fac * corr

def _f_eps_cos_mp(eps, ratio, a_k, size):
    """Function of which we want to find a root."""
    f_j = _fj_eps_cos_mp(eps, a_k, size)
    f_j_plus1 = _fj_eps_cos_mp(eps - mp.mpf(1) / size, a_k, size)
    fac = (eps + 1/size) / (eps - 2/size)
    return abs(ratio * fac * f_j_plus1) - abs(f_j)

def _remove_peak_cos(abs_fft, eps, ind, a_k):
    """Remove a cosine-series peak from a FFT at a given frequency"""
    size = abs_fft.size
    num_a_k = len(a_k)
    ampl = abs_fft[ind]
    corr = sum([a_k[k] * (1 / (eps - k / size) + 1 / (eps + k / size))
                for k in range(1, num_a_k)])
    ampl /= 2*a_k[0] + eps * corr
    for j in range(-num_a_k+1, num_a_k, 1):
        if j == 0:
            continue
        corr = sum([a_k[abs(k)] / (eps - (k+j)/size)
                    for k in range(-num_a_k+1, num_a_k, 1) if k != -j])
        corr = eps * corr + a_k[abs(j)] + a_k[0] * eps / (eps - j/size)
        abs_fft[(ind + j) % size] -= ampl * abs(corr)
    abs_fft[ind] = 0.0
    
def naffnd_cos_mp(z_mp, n_freq=1, a_k=1, return_coeff=False):
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
    if isinstance(a_k, int):
        a_k = win_mp.hann_coeff(a_k)
        
    weights_mp = win_mp.get_window(len(z_mp), win_mp.cos_weights, a_k)
    weights = np.array(weights_mp, dtype=np.float64)
    z = np.array(z_mp, dtype=np.complex128)
    w_z_mp = [weights_mp[i] * z_mp[i] for i in range(z.size)]
    w_z = weights * z
    fft = np.fft.fft(w_z)
    abs_fft = np.abs(fft)

    # frequencies and their respective amplitudes
    nu_arr = [mp.mpf(0)] * n_freq
    c_arr = [mp.mpc(0j)] * n_freq
    
    for ctr in range(n_freq):
        ind = np.argmax(abs_fft)    # find the global maximum of the FFT
        if abs_fft[ind] < 1e-10:
            break   # abort if fft-peak too small -> nu == 0

        if abs_fft[(ind + 1) % z.size] < abs_fft[(ind - 1) % z.size]:
            ind -= 1

        ind_mp = mp.mpf(float(ind))
        dft_j_plus1 = dft_mp(w_z_mp, ((ind_mp+1) % z.size) / z.size)
        dft_j = dft_mp(w_z_mp, ind_mp / z.size)
        ratio = dft_j / dft_j_plus1
        eps = newton(_f_eps_cos_mp, 1 / (2 * z.size), ratio, a_k, z.size)
        nu = ind_mp/ z.size + eps
        nu_arr[ctr] = nu

        # compute the complex value of the frequency amplitude
        if return_coeff:
            coeff = mp.mpf(2) * dft_j / z.size
            if abs(eps) > 1e-12:
                coeff *= 2j*mp.pi * z.size * eps / (mp.exp(2j*mp.pi * eps * z.size) - 1)

            corr = sum([a_k[k] * (1 / (eps - k / z.size) + 1 / (eps + k / z.size))
                        for k in range(1, len(a_k))])
            c_arr[ctr] = coeff / (2*a_k[0] + eps * corr)
        
        if ctr < n_freq - 1:
            # remove the previous peak from the fft spectrum
            _remove_peak_cos(abs_fft, ind=ind, eps=eps, a_k=a_k)
            _remove_peak_cos(abs_fft, ind=z.size - ind, eps=-eps, a_k=a_k)
    if return_coeff:
        return nu_arr, c_arr
    return nu_arr



###############################################################################
# ROOT FINDING
###############################################################################



def newton(func, x0, *args, xtol=1e-27, dx=1e-10, max_steps=100):
    for step in range(max_steps):
        fx0 = func(x0, *args)
        fx1 = func(x0 + dx, *args)
        delta_x0 = fx0 * dx / (fx1 - fx0)
        x0 -= delta_x0
        if abs(delta_x0) < dx:
            dx = delta_x0
        if abs(delta_x0) < xtol:
            break
    return x0
