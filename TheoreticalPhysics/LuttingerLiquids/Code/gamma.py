#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gammaln function using numba
https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.f
"""

import numpy as np
from numba import njit, prange

def gammaln(z):
    if isinstance(z, np.ndarray):
        if z.ndim == 1:
            return _gammaln_1d(z.real, z.imag)
        elif z.ndim == 2:
            return _gammaln_2d(z.real, z.imag)
        else:
            raise NotImplementedError(f"Input dimension {z.ndim} not supported!")
    return _gammaln(z.real, z.imag)

@njit(fastmath=True, error_model="numpy", nogil=True, parallel=True)
def _gammaln_1d(zr, zi):
    size = zr.shape[0]
    gz = np.zeros(size, dtype=np.complex128)
    for i in prange(size):
        gz[i] = _gammaln(zr[i], zi[i])
    return gz

@njit(fastmath=True, error_model="numpy", nogil=True, parallel=True)
def _gammaln_2d(zr, zi):
    gz = np.zeros(zr.shape, dtype=np.complex128)
    for i in prange(zr.shape[0]):
        for j in range(zr.shape[1]):
            gz[i,j] = _gammaln(zr[i,j], zi[i,j])
    return gz

def beta(z1, z2):
    if isinstance(z1, np.ndarray):
        if z1.ndim == 1:
            return _beta_1d(z1.real, z1.imag, z2.real, z2.imag)
        else:
            raise NotImplementedError()
    return _beta(z1.real, z1.imag, z2.real, z2.imag)

@njit(fastmath=True, error_model="numpy", nogil=True, parallel=True)
def _beta_1d(z1r, z1i, z2r, z2i):
    size = z1r.shape[0]
    gz = np.zeros(size, dtype=np.complex128)
    for i in prange(size):
        gz[i] = _beta(z1r[i], z1i[i], z2r[i], z2i[i])
    return gz

@njit(fastmath=True)
def _beta(z1r, z1i, z2r, z2i):
    return np.exp(_gammaln(z1r, z1i) + _gammaln(z2r, z2i) - _gammaln(z1r+z2r, z1i+z2i))

@njit(fastmath=True)
def _gammaln(zr, zi):
    a = np.array([8.333333333333333e-02,-2.777777777777778e-03,
                  7.936507936507937e-04,-5.952380952380952e-04,
                  8.417508417508418e-04,-1.917526917526918e-03,
                  6.410256410256410e-03,-2.955065359477124e-02,
                  1.796443723688307e-01,-1.39243221690590e+00])
    x1 = 0
    if ( zi == 0.0e+00 and zr == np.int32(zr) and zr <= 0.0e+00 ) :
        gr = 1.0e+300
        gi = 0.0e+00
        return gr + 1j*gi
    elif ( zr < 0.0e+00 ) :
        x1 = zr
        y1 = zi
        zr = -zr
        zi = -zi

    x0 = zr

    if ( zr <= 7.0e+00 ) :
        na = int ( 7 - zr )
        x0 = zr + na

    z1 = np.sqrt ( x0 * x0 + zi * zi )
    th = np.arctan ( zi / x0 )
    gr = ( x0 - 0.5e+00 ) * np.log ( z1 ) - th * zi - x0 + 0.5e+00 * np.log ( 2.0e+00 * np.pi )
    gi = th * ( x0 - 0.5e+00 ) + zi * np.log ( z1 ) - zi

    for k in range(10):
        t = z1 ** ( -1 - 2 * k )
        gr = gr + a[k] * t * np.cos ( ( 2.0e+00 * k + 1.0e+00 ) * th )
        gi = gi - a[k] * t * np.sin ( ( 2.0e+00 * k + 1.0e+00 ) * th )

    if ( zr <= 7.0e+00 ) :
        gr1 = 0.0e+00
        gi1 = 0.0e+00
        for j in range(na):
            gr1 = gr1 + 0.5e+00 * np.log ( ( zr + j ) ** 2 + zi * zi )
            gi1 = gi1 + np.arctan ( zi / ( zr + j ) )
        gr = gr - gr1
        gi = gi - gi1

    if ( x1 < 0.0e+00 ) :
        z1 = np.sqrt ( zr * zr + zi * zi )
        th1 = np.arctan ( zi / zr )
        sr = - np.sin ( np.pi * zr ) * np.cosh ( np.pi * zi )
        si = - np.cos ( np.pi * zr ) * np.sinh ( np.pi * zi )
        z2 = np.sqrt ( sr * sr + si * si )
        th2 = np.arctan ( si / sr )
        if ( sr < 0.0e+00 ) :
            th2 = np.pi + th2
        gr = np.log ( np.pi / ( z1 * z2 ) ) - gr
        gi = - th1 - th2 - gi
        zr = x1
        zi = y1

    # if ( kf == 1 ) : ## convert to gamma(z) instead of gammaln(z)
    #     g0 = np.exp ( gr )
    #     gr = g0 * np.cos ( gi )
    #     gi = g0 * np.sin ( gi )

    return gr + 1j*gi

def gamma(z):
    if isinstance(z, np.ndarray):
        if z.ndim == 1:
            return _gamma_1d(z)
        else:
            raise NotImplementedError(f"Input dimension {z.ndim} not supported!")
    return _gamma(z)

@njit(fastmath=True, nogil=True, error_model="numpy")
def _gamma_1d(z):
    size = z.shape[0]
    gz = np.zeros(size, dtype=np.complex128)
    for i in range(size):
        gz[i] = _gamma(z[i])
    return gz

@njit(fastmath=True, error_model="numpy")
def _gamma(z):
    """https://github.com/Bobingstern/Lanczos-Approximation/blob/main/gamma.hpp"""
    gamma_p = (676.5203681218851,
              -1259.1392167224028
              ,771.32342877765313
              ,-176.61502916214059
              ,12.507343278686905
              ,-0.13857109526572012
              ,9.9843695780195716e-6
              ,1.5056327351493116e-7
    )
    sqrt_2pi = 2.5066282746310002
    gamma_len = 8
    if (z.real < 0.5):
        return np.pi / np.sin(np.pi*z) / _gamma(1-z)
    x=1
    for i in range(gamma_len):
        x += gamma_p[i] / (z+i)
    t = z + gamma_len - 1.5
    return x * sqrt_2pi * np.exp(-t + (z-0.5) * np.log(t))

@njit(fastmath=True, error_model="numpy")
def _gamma_real(x):
    """https://github.com/Bobingstern/Lanczos-Approximation/blob/main/gamma.hpp"""
    gamma_p = (676.5203681218851,
              -1259.1392167224028
              ,771.32342877765313
              ,-176.61502916214059
              ,12.507343278686905
              ,-0.13857109526572012
              ,9.9843695780195716e-6
              ,1.5056327351493116e-7
    )
    sqrt_2pi = 2.5066282746310002
    gamma_len = 8
    if (x < 0.5):
        return np.pi / np.sin(np.pi*x) / _gamma_real(1-x)
    y=1
    for i in range(gamma_len):
        y += gamma_p[i] / (x+i)
    t = x + gamma_len - 1.5
    return y * sqrt_2pi * np.exp(-t + (x-0.5) * np.log(t))


def gamma_rel(z1, z2):
    if isinstance(z1, np.ndarray):
        if z1.ndim == 1:
            return _gamma_rel_1d(z1, z2)
        elif z1.ndim == 2:
            return _gamma_rel_2d(z1, z2)
        else:
            raise NotImplementedError()
    return _gamma_rel(z1, z2)

@njit(fastmath=True, error_model="numpy", nogil=True, parallel=True)
def _gamma_rel_1d(z1, z2):
    size = z1.shape[0]
    gz = np.zeros(size, dtype=np.complex128)
    for i in prange(size):
        gz[i] = _gamma_rel(z1[i], z2[i])
    return gz

@njit(fastmath=True, error_model="numpy", nogil=True, parallel=True)
def _gamma_rel_2d(z1, z2):
    gz = np.zeros(z1.shape, dtype=np.complex128)
    for i in prange(z1.shape[0]):
        for j in range(z1.shape[1]):
            gz[i,j] = _gamma_rel(z1[i,j], z2[i,j])
    return gz

@njit(fastmath=True)
def _gamma_rel(z1, z2):
    """Compute the relative Gamma-function 'Gamma(z1) / Gamma(z2)'
    for s1 in [+1,-1]:
        for s2 in [+1,-1]:
            z1=s1*3.214+10.331j
            z2=s2*0.5155531+0.1331j
            print(special.gamma(z1)/special.gamma(z2))
            print(np.exp(gammaln(z1) - gammaln(z2)))
            print(gamma_rel(z1,z2))
    size=10**6; xmin=-10; xmax=10
    rnd = np.random.uniform(xmin,xmax, size=(4,size))
    z=rnd[0]+1j*rnd[1]; z2=rnd[2]+1j*rnd[3]
    %timeit -n 1 res1 = special.gamma(z)/special.gamma(z2)
    %timeit -n 1 res2 = np.exp(gammaln(z) - gammaln(z2))
    %timeit -n 1 res3 = gamma_rel(z, z2)
    print(np.abs((res2-res3)/res1).max())
    """
    gamma_p = (676.5203681218851,
              -1259.1392167224028
              ,771.32342877765313
              ,-176.61502916214059
              ,12.507343278686905
              ,-0.13857109526572012
              ,9.9843695780195716e-6
              ,1.5056327351493116e-7
    )
    gamma_len = 8
    mirror_bound = 0.4
    x1 = 1
    x2 = 1
    if (z1.real > mirror_bound) and (z2.real > mirror_bound):
        for i in range(gamma_len):
            x1 += gamma_p[i] / (z1+i)
            x2 += gamma_p[i] / (z2+i)
        t1 = z1 + gamma_len - 1.5
        t2 = z2 + gamma_len - 1.5
        return x1 / x2 * np.exp(t2-t1 + (z1 - 0.5) * np.log(t1) - (z2 - 0.5) * np.log(t2))
    elif (z1.real > mirror_bound) and (z2.real <= mirror_bound):
        for i in range(gamma_len):
            x1 += gamma_p[i] / (z1+i)
            x2 += gamma_p[i] / (1-z2+i)
        t1 = z1 + gamma_len - 1.5
        t2 = 1-z2 + gamma_len - 1.5
        return x1 * x2 * 2 * np.sin(np.pi*z2) * np.exp(-t1-t2 + (z1 - 0.5) * np.log(t1) + (1-z2 - 0.5) * np.log(t2))
    elif (z1.real <= mirror_bound) and (z2.real > mirror_bound):
        for i in range(gamma_len):
            x1 += gamma_p[i] / (1-z1+i)
            x2 += gamma_p[i] / (z2+i)
        t1 = 1-z1 + gamma_len - 1.5
        t2 = z2 + gamma_len - 1.5
        return 1 / x1 / x2 / 2 / np.sin(np.pi*z1) * np.exp(t1+t2 - (1-z1 - 0.5) * np.log(t1) - (z2 - 0.5) * np.log(t2))
    else:
        for i in range(gamma_len):
            x1 += gamma_p[i] / (1-z1+i)
            x2 += gamma_p[i] / (1-z2+i)
        t1 = 1-z1 + gamma_len - 1.5
        t2 = 1-z2 + gamma_len - 1.5
        return x2 / x1 * np.sin(np.pi*z2)/np.sin(np.pi*z1) * np.exp(t1-t2 - (1-z1 - 0.5) * np.log(t1) + (1-z2 - 0.5) * np.log(t2))


def main():
    print(__doc__)
    from scipy import special
    size=10**6
    z=np.random.uniform(size=size)+1j*np.random.uniform(size=size)
    z2=np.random.uniform(size=size)+1j*np.random.uniform(size=size)
    mybeta = lambda x,y: special.gamma(x)*special.gamma(y)/special.gamma(x+y)
    res1 = mybeta(z, z2)
    res2 = beta(z, z2)
    print(np.max(np.abs(res1 - res2)), np.sum(np.abs(res1 - res2)))
    #%timeit -n 1 res1 = mybeta(z, z2)
    #%timeit -n 1 res2 = np.exp(gammaln(z) + gammaln(z2) - gammaln(z+z2))


if __name__ == "__main__":
    main()
