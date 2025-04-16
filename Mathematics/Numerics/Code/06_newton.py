# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np


def f(x):
    return x**3 - 3*x**2 + 1
def fprime(x):
    return 3*x**2 - 6*x
    

def newton(f, fprime, x0, eps=1e-14, max_iter=50):
    x = x0
    for n in range(max_iter):
        f_val = f(x)
        if np.abs(f_val) < eps:
            print("Found root!\n")
            return x
        
        fp_val = fprime(x)
        if fp_val == 0.0:
            print(f"Warning, derivative zero at x = {x}!")
            return x
        
        print(f"n={n:<3}: x = {x}, f(x) = {f_val:.3e}, f'(x) = {fp_val:.3e}")
        x -= f_val / fp_val
        
    
    print(f"Warning, accuracy not reached after max_iter = {max_iter} steps!")
    return x

def x0_from_interval(interval):
    """ Simply take the midpoint -- suffices for the given example. """
    a, b = interval
    return (a + b) / 2.0


def main():
    intervals = [[-1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]
    init = [x0_from_interval(interval) for interval in intervals]
    
    zeros = [newton(f, fprime, x0) for x0 in init]
    
    for zero in zeros:
        print(f"Found root at x = {zero} with f(x) = {f(zero):.3e}")
        
    return 0

if __name__ == "__main__":
    main()