#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS sample path generation and noise reduction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def log_spiral(t, tau=1, alpha=1):
    r_t = np.exp(-alpha * t / tau)
    phi_t = 2 * np.pi * t / tau
    return r_t * np.cos(phi_t), r_t * np.sin(phi_t)


def generate_noisy_path(t, tau=1, alpha=1, sigma=0.1):
    x, y = log_spiral(t, tau, alpha)
    noise2d = np.random.normal(0, sigma, size=(2, t.size))
    return x + noise2d[0], y + noise2d[1]


def plot_path_and_filter():
    size = 100
    t = np.linspace(0, 0.6, size)
    tau = 1
    alpha = 1
    sigma = 0.01

    x, y = log_spiral(t, tau, alpha)
    xn, yn = generate_noisy_path(t, tau, alpha, sigma)

    # size == window_size of filter
    xy_fil = uniform_filter1d(np.array([xn, yn]), size=3)
    xf, yf = xy_fil[0], xy_fil[1]

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.plot(x, y, ls='--', lw=0.5, marker='o', ms=1, label="True Path")
    ax.plot(xn, yn, ls='--', lw=0.5, marker='o', ms=1, label="Noisy Path")
    ax.plot(xf, yf, ls='--', lw=0.5, marker='o', ms=1, label="Filtered Path")
    ax.legend()


def uniform_filter(arr, size=3):
    """
    size == length of the window filter
    
    Example: arr = [2., 8., 0., 4., 1., 9., 9., 0.]
    uniform_filter(arr, size=1) = [2.   , 8.   , 0.   , 4.   , 1.   , 9.   , 9.   , 0.]
    uniform_filter(arr, size=2) = [2.   , 5.   , 4.   , 2.   , 2.5  , 5.   , 9.   , 4.5]
    uniform_filter(arr, size=3) = [4.   , 3.333, 4.   , 1.666, 4.666, 6.333, 6.   , 3.]
    uniform_filter(arr, size=4) = [5.   , 3.   , 3.5  , 3.25 , 3.5  , 5.75 , 4.75 , 4.5]
    uniform_filter(arr, size=5) = [4.   , 3.2  , 3.   , 4.4  , 4.6  , 4.6  , 3.8  , 5.4]
    """
    arr = np.asarray(arr)
    w_left = size // 2              # length of the window to the left
    w_right = size - w_left         # length of the window to the right
    new_arr = np.zeros_like(arr)    # new values
    
    def index(ind):
        """Example: arr = [5, 6, 3, 2, 2, 1, 7] --> arr.size = 7
        index(5) = 5                -->  arr[index] = 1
        index(-1) = 0               -->  arr[index] = 5
        index(-2) = 1               -->  arr[index] = 6
        index(7) = 7 - (8 % 7) = 6  -->  arr[index] = 7
        index(8) = 7 - (9 % 7) = 5  -->  arr[index] = 1
        """
        if ind < 0:
            return -(ind+1)
        elif ind >= arr.size:
            return arr.size - ((ind+1) % arr.size)
        else:
            return ind
    
    # initialize running sum
    running_sum = 0
    for i in range(size):
        running_sum += arr[index(i - w_left)]   
        
    #print("ARRAY: ", arr, arr.size)
    #for i in range(- w_left, arr.size + w_right+1):
    #    print(i, index(i), arr[index(i)]) 
        
    new_arr[0] = running_sum / size
    for i in range(arr.size-1):
        running_sum += arr[index(i + w_right)] - arr[index(i - w_left)]
        new_arr[i+1] = running_sum / size
    return new_arr


def main():
    print(__doc__)

    plot_path_and_filter()

    return 0


if __name__ == "__main__":
    main()
