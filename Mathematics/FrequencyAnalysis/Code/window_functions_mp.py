#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window functions for NAFF
"""

# import numpy as np
import mpmath as mp
# from scipy.signal import bspline

def mp_comb(upper, lower):
    val = mp.factorial(upper) / (mp.factorial(lower) * mp.factorial(upper - lower))
    return val

def hann_coeff(order):
    """Returns the cosine series coefficients for the 'order'-th hanning window"""
    ak = [mp.mpf(0)] * (order + 1)
    ak[0] = mp_comb(2*order, order) / (4**order)
    for k in range(1, order+1, 1):
        ak[k] = mp.mpf(2 * (-1)**k) * mp_comb(2*order, order - k) / (4**order)
    return ak


def cos_weights(x, a_k):
    try:
        res = [mp.mpf(0)] * len(x)
        for i in range(len(x)):
            for k in range(len(a_k)):
                res[i] += a_k[k] * mp.cos(2 * mp.pi * x[i] * k)
    except TypeError:
        res = mp.mpf(0)
        for k in range(len(a_k)):
            res += a_k[k] * mp.cos(2 * mp.pi * x * k)
    return res


def hann_weights(x, a_k=1):
    a_k = hann_coeff(a_k)
    return cos_weights(x, a_k)


# def planck_weights(x, alpha=0.1):
#     res = np.ones_like(x)
#     indx = ((x > 0) & (x < alpha))
#     res[indx] = (1 + np.exp(alpha / x[indx] - 1 / (1 - x[indx] / alpha)))**(-1)
#     indx = ((x < 1) & (x > 1 - alpha))
#     res[indx] = (1 + np.exp(alpha / (1 - x[indx]) - 1 / (1 - (1 - x[indx]) / alpha)))**(-1)
#     indx = ((x == 0.0) | (x == 1.0))
#     res[indx] = 0.0
#     return res


# from scipy.signal.windows import chebwin, at=250 gives very sharp drop


def gauss_weights(x, alpha=140):
    weights = [mp.mpf(0)] * len(x)
    for i in range(len(x)):
        weights[i] = mp.exp(-alpha * (x[i] - 0.5)**2)
    return weights

# def flattop_weights(w_method, fpar=0.5):
#     """Replace midsection of weights with a constant"""
#     def new_w_method(x):
#         if isinstance(x, (float, int)):
#             if x < fpar / 2:
#                 return w_method(x / fpar)
#             elif x > 1 - fpar / 2:
#                 return w_method(1 - x / fpar)
#             else:
#                 return 1.0
#         else:
#             weights = np.ones(x.size, dtype=float)
#             indx = (x < fpar / 2)
#             weights[indx] = w_method(x[indx] / fpar)
#             indx = (x > 1 - fpar / 2)
#             weights[indx] = w_method(1 - x[indx] / fpar)
#             return weights

#     return new_w_method


# def bspline_weights(x, order=3):
#     """scipy.signal.bspline, order==3 gives Parzen window"""
#     return bspline((order + 1) * (x - 0.5), order)


def get_window(n_points, window=gauss_weights, *args):
    try:
        arr = [mp.mpf(i) / n_points for i in range(n_points)]
        weights = window(arr, *args)
    except TypeError:
        weights = window(n_points, *args)
    return weights
