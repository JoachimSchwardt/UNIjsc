#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window functions for NAFF
"""

import numpy as np
from scipy.signal import bspline


def hann_coeff(M):
    """Returns the cosine series coefficients for the M-th hanning window"""
    ak = np.zeros(M+1)
    ak[0] = np.math.comb(2*M, M)
    for k in range(1, M+1, 1):
        ak[k] = 2 * (-1)**k * np.math.comb(2*M, M-k)
    return ak / (4**M)


def hann_weights(x, a_k=1):
    a_k = hann_coeff(a_k)
    return cos_weights(x, a_k)


# Flat-Top: [0.21557895, -0.41663158, 0.277263158, -0.083578947, 0.006947368]
# Blackman-Harris: [0.35875, -0.48829, 0.14128, -0.01168]
def cos_weights(x, a_k):
    a_k = np.asarray(a_k)
    vals = np.array([a_k[k] * np.cos(2*np.pi*x * k) for k in range(a_k.size)])
    if isinstance(x, (float, int)):
        return np.sum(vals)
    else:
        return np.sum(vals, axis=0)


def sinc_weights(x):
    return np.sinc(2 * (x - 0.5))


def planck_weights(x, alpha=0.1):
    res = np.ones_like(x)
    indx = ((x > 0) & (x < alpha))
    res[indx] = (1 + np.exp(alpha / x[indx] - 1 / (1 - x[indx] / alpha)))**(-1)
    indx = ((x < 1) & (x > 1 - alpha))
    res[indx] = (1 + np.exp(alpha / (1 - x[indx]) - 1 / (1 - (1 - x[indx]) / alpha)))**(-1)
    indx = ((x == 0.0) | (x == 1.0))
    res[indx] = 0.0
    return res


# from scipy.signal.windows import chebwin, at=250 gives very sharp drop


def gauss_weights(x, alpha=140):
    return np.exp(-alpha * (x - 0.5)**2)


def flattop_weights(w_method, fpar=0.5):
    """Replace midsection of weights with a constant"""
    def new_w_method(x):
        if isinstance(x, (float, int)):
            if x < fpar / 2:
                return w_method(x / fpar)
            elif x > 1 - fpar / 2:
                return w_method(1 - x / fpar)
            else:
                return 1.0
        else:
            weights = np.ones(x.size, dtype=float)
            indx = (x < fpar / 2)
            weights[indx] = w_method(x[indx] / fpar)
            indx = (x > 1 - fpar / 2)
            weights[indx] = w_method(1 - x[indx] / fpar)
            return weights

    return new_w_method


def bspline_weights(x, order=3):
    """scipy.signal.bspline, order==3 gives Parzen window"""
    return bspline((order + 1) * (x - 0.5), order)

