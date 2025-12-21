#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lyapunov Fractals
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import toolkit as tlk
# import tqdm
# import mpl_special

INCHES_PER_PT = 1 / 72.27

PAPERWIDTH_PT = 597
PAPERHEIGHT_PT = 845

FIGSIZE = [PT * INCHES_PER_PT for PT in (PAPERWIDTH_PT, PAPERHEIGHT_PT)]
FGS2 = [FIGSIZE[0], FIGSIZE[1]/2]

EPSILON = 1e-4

@njit
def iterator(a_b, n_iterations=2048, n_startup=512):
    x_0 = 0.5
    exponent = 0
    length = len(a_b)
    for n in range(n_iterations):
        r_n = a_b[n % length]
        # print("PRE ITERATION:", r_n, x_0)
        x_0 = r_n * x_0 * (1-x_0)
        # print("POST ITERATION:", r_n, x_0)
        # print("LOG", np.log2(np.abs(x_0)))
        if n >= n_startup:
            ### interstingly, this incorrect version gives very similar structure
            # exponent += np.log2(np.abs(x_0) + EPSILON)
            exponent += np.log2(np.abs(r_n * (1 - 2*x_0)) + EPSILON)
    return exponent / (n_iterations - n_startup)

@njit(parallel=True)
def run_iterator_2d(a, b, sequence="AB", n_iterations=1<<10):
    results = np.zeros(a.shape, dtype=np.float64)
    a_b = np.zeros(len(sequence), dtype=np.float64)
    for i_a in prange(a.shape[0]):
        for i_b in range(a.shape[1]):
            for i in range(len(sequence)):
                a_b[i] = a[i_a, i_b] if sequence[i] == "A" else b[i_a, i_b]
            result = iterator(a_b, n_iterations)
            results[i_a, i_b] = result
    return results

def get_extent(x, y):
    """Compute the correct extent for an imshow plot with given (x,y)-values. MWE:
    x=np.linspace(-3, 3, 8); y=np.linspace(-1, 1, 4)
    x2, y2 = np.meshgrid(x,y)
    a = np.sign(y2)/(1+x2**2)*y2**2
    plt.imshow(a, extent=get_extent(x,y), cmap="inferno", interpolation="none")
    plt.scatter(x2, y2)
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    extent = [x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
    return extent

def test():
    # a_1 = tlk.linspace_center(2, 4, 1200)
    # b_1 = tlk.linspace_center(2, 4, 900)
    a_1 = tlk.linspace_center(3.4, 4, 1200)
    b_1 = tlk.linspace_center(2.5, 3.4, 900)
    a_2, b_2 = np.meshgrid(a_1, b_1)
    result = run_iterator_2d(a_2, b_2, "BBBBBBAAAAAA")
    fig, ax = plt.subplots()# cmaps: twilight, RdBu
    ax.imshow(result.T, origin="lower", extent=tlk.get_extent(b_1, a_1), cmap="RdBu",
               vmin=-1, vmax=result.max())

def plot_titlepage_background():
    a_1 = tlk.linspace_center(3.4, 4, int(FGS2[0]*400))
    b_1 = tlk.linspace_center(2.5, 3.4, int(FGS2[1]*400))
    a_2, b_2 = np.meshgrid(a_1, b_1)
    result = run_iterator_2d(a_2, b_2, "BBBBBBAAAAAA")
    fig, ax = plt.subplots(figsize=FGS2)
    ax.axis("off")
    ax.imshow(result.T, origin="lower", extent=get_extent(b_1, a_1),
              cmap="twilight", vmin=-1, vmax=result.max())
    #fig.savefig("titlepage_bg.pdf")
    #fig.savefig("titlepage_bg.png", dpi=300)
    

def get_color(exponent):
    """https://www.shadertoy.com/view/fldBWr"""
    #color goes from black to yellow as exponent goes from -inf to 0
    negScale = 1.0 #this controls how quickly the color goes from black to yellow (bigger = faster)
    
    #color goes from blue to black as exponent goes from 0 to inf
    posScale = 2.0; #this controls how quickly the color goes from blue to black (bigger = faster)
    
    if(exponent<=0.0): #stable color
        exponent = np.exp(np.abs(negScale) * exponent);
        cutoff = 0.98;                
        col1 = (0.0,0.0,0.0);
        col2 = (1.0,0.76,0.0);
        col3 = (1.0,0.8,0.7);
        if(exponent <= cutoff):
            return col1 + (col2-col1) * exponent/cutoff;
        else:
            return col2 + (col3-col2) * (cutoff-exponent)/(cutoff-1.0);
    else: #chaotic color
        exponent = np.exp(-np.abs(posScale) * exponent);
        return (0.0, 0.0, exponent)

def main():
    # iterator = get_iterator("AB")
    # n_iterations = np.unique(np.geomspace(1, 3000, 100, dtype=int))
    # a = 2.5
    # b = 2.8
    # final = iterator(a, b, 10000)
    # final_v = np.array([iterator(a, b, n) for n in n_iterations])
    # error = np.abs(final_v - final)
    return 0

if __name__ == "__main__":
    main()
