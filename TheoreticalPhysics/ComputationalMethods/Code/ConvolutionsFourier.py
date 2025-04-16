# -*- coding: utf-8 -*-
"""
Numerical results and visualization of convolutions, Fourier transformations
and related integral problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numba import njit

@njit
def sinc_product(x, beta):
    """Product of Sinc functions"""
    res = np.sin(x) / x
    for i in range(beta.shape[0]):
        res *= np.sin(beta[i] * x) / (beta[i] * x)
    return res

@njit
def sinc_product_log(x, beta):
    """Product of Sinc functions, x -> exp(-x) substitution"""
    z = np.log(x)
    res = sinc_product(z, beta) / x
    return res

@njit
def sinc_product_inv(x, beta):
    """Product of Sinc functions, x -> 1 / (1+x) substitution"""
    z = 1 / x - 1
    res = sinc_product(z, beta) / x**2
    return res

@njit
def sinc_product_tan(x, beta):
    """Product of Sinc functions, z = tan(x) substitution"""
    z = np.tan(x)
    res = sinc_product(z, beta) * (1 + z**2)
    return res

def sinc_product_integral_numeric(beta):
    """Numerical estimate of the sinc-product-integral"""
    # res, err = quad(sinc_product, 0, np.inf, beta, limit=1000)
    # print(quad(sinc_product_log, 0, 1, beta, limit=1000))
    # print(quad(sinc_product_inv, 0, 1, beta, limit=1000))
    res, err = quad(sinc_product_tan, 0, np.pi/2, beta, limit=1000)
    return 2 * res, err    
    # return quad(sinc_product, 0, np.inf, beta)

def sinc_product_integral_theory(beta):
    """Theoretical prediction for the sinc-product-integral
    UPDATE: This formula appears to be only correct for N <= 3"""
    res = np.pi
    sigma = np.sum(beta) - 1
    if sigma < 0:
        return res
    
    if sigma > 2:
        sigma = 2
        
    N = beta.shape[0]
    eps = 2 / np.math.factorial(N) * sigma**N / np.prod(beta)
    return res * (1 - eps / (2**N))
    
def main():
    print(__doc__)
    beta = np.array([1, 1])
    theo = sinc_product_integral_theory(beta)
    num, num_err = sinc_product_integral_numeric(beta)
    # num, num_err = np.pi, 1e-13
    # print(sinc_product_integral_numeric(beta))
    print(f"Beta = {beta}")
    print(f"Theoretical predicition is {theo}")
    print(f"Numerical estimate is {num} +- {num_err}")
    abs_diff = np.abs(theo - num)
    def tolerance(abs_diff, num_err):
        if abs_diff > num_err: 
            return "larger"
        return "less"
    print(f"Difference is {abs_diff} and {tolerance(abs_diff, num_err)}" + 
          " than numerical error")
    
if __name__ == "__main__":
    main()