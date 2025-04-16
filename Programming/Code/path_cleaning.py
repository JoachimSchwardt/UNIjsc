#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS sample path generation and noise reduction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import mpl_special


def log_spiral(t, tau=1, alpha=1):
    r_t = np.exp(-alpha * t / tau)
    phi_t = 2 * np.pi * t / tau
    return r_t * np.cos(phi_t), r_t * np.sin(phi_t)


def generate_noisy_path(t, tau=1, alpha=1, sigma=0.1):
    x, y = log_spiral(t, tau, alpha)
    noise2d = np.random.normal(0, sigma, size=(2, t.size))
    return x + noise2d[0], y + noise2d[1]


def clear_noise(positions, window_size=10):
    steps = positions.shape[1]
    # delta_t = window_size / steps
    # velocity = np.diff(positions, n=1) / delta_t
    # acceleration = np.diff(positions, n=2) / delta_t**2
    
    # acc_filter = uniform_filter1d(acceleration, axis=1, size=window_size)
    # vel_filter = np.cumsum(acc_filter, axis=1) * delta_t
    # pos_filter = np.cumsum(vel_filter, axis=1) * delta_t
    
    # # fig, ax = plt.subplots()
    # # ax.plot(acceleration[0])
    # # ax.plot(acc_filter[0])
    # print(np.mean(acceleration[0]), np.mean(acc_filter[0]))
    # # ax.plot(velocity[0])
    # # ax.plot(vel_filter[0])
    # print(np.mean(velocity[0]), np.mean(vel_filter[0]))
    # # ax.plot(positions[0])
    # # ax.plot(pos_filter[0])
    # print(np.mean(positions[0]), np.mean(pos_filter[0]))
    # raise RuntimeError
    
    # pos_filter = np.zeros_like(positions)
    # for dim in range(positions.shape[0]):
    #     pos_filter
    
    pos_filter = uniform_filter1d(positions, size=window_size)
    
    return pos_filter
    
    
    
    
# def kalman_xy(x, P, measurement, R,
#               motion = np.zeros(4),
#               Q = np.eye(4)):
#     """
#     Parameters:    
#     x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
#     P: initial uncertainty convariance matrix
#     measurement: observed position
#     R: measurement noise 
#     motion: external motion added to state vector x
#     Q: motion noise (same shape as P)
#     """
#     return kalman(x, P, measurement, R, motion, Q,
#                   F = np.array([[1, 0, 1, 0], 
#                                 [0, 1, 0, 1], 
#                                 [0, 0, 1, 0], 
#                                 [0, 0, 0, 1]]),
#                   H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
                      

# def kalman(x, P, measurement, R, motion, Q, F, H):
#     '''
#     Parameters:
#     x: initial state
#     P: initial uncertainty convariance matrix
#     measurement: observed position (same shape as H*x)
#     R: measurement noise (same shape as H)
#     motion: external motion added to state vector x
#     Q: motion noise (same shape as P)
#     F: next state function: x_prime = F*x
#     H: measurement function: position = H*x

#     Return: the updated and predicted new values for (x, P)

#     See also http://en.wikipedia.org/wiki/Kalman_filter

#     This version of kalman can be applied to many different situations by
#     appropriately defining F and H 
#     '''
#     # UPDATE x, P based on measurement m    
#     # distance between measured and current position-belief
#     y = np.array(measurement).T - H @ x
#     S = H @ P @ H.T + R  # residual convariance
#     K = P @ H.T @ np.linalg.inv(S)    # Kalman gain
#     x = x + K @ y
#     I = np.eye(F.shape[0])   # identity matrix
#     P = (I - K@H) @ P

#     # PREDICT x, P based on motion
#     x = F@x + motion
#     P = F@P@F.T + Q

#     return x, P


# def apply_kalman_2d(x, y):
#     X = np.zeros(4)
#     P = np.eye(4) * 1e7 # initial uncertainty

#     R = 0.01**2
#     new_xy = np.zeros((2, x.size))
#     for i in range(x.size):
#         X, P = kalman_xy(X, P, (x[i], y[i]), R)
#         new_xy[:, i] = X[:2]
#     return new_xy[0], new_xy[1]


def plot_path_and_filter():
    size = 500
    t = np.linspace(0, 1.3, size)
    tau = 1
    alpha = 1
    sigma = 0.01
    
    x, y = log_spiral(t, tau, alpha)
    xn, yn = generate_noisy_path(t, tau, alpha, sigma)
    # xkal, ykal = apply_kalman_2d(xn, yn)
    xy_fil = clear_noise(np.array([xn, yn]), window_size=3)
    xf, yf = xy_fil[0], xy_fil[1]
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.plot(x, y, ls='--', lw=0.5, marker='o', ms=1, label="True Path")
    ax.plot(xn, yn, ls='--', lw=0.5, marker='o', ms=1, label="Noisy Path")
    # ax.plot(xkal, ykal, ls='--', lw=0.5, marker='o')
    ax.plot(xf, yf, ls='--', lw=0.5, marker='o', ms=1, label="Filtered Path")
    ax.legend()
    mpl_special.polish(fig, ax)


def main():
    print(__doc__)
    
    plot_path_and_filter()
    
    return 0


if __name__ == "__main__":
    main()
