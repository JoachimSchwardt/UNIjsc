"""
Richardson Extrapolation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup()

# def richardson(sn, N=1, K=2):
#     SN = np.sum(sn[:N])
#     corr = np.zeros(K)
#     for n in range(1, K+1, 1):
#         factor = np.sum([(-1)**j * np.math.comb(K, j) * (N+K-j)**K 
#                          for j in range(0, K-n+1, 1)])
#         corr[n-1] = sn[N+n-1] * factor
#     return SN + np.sum(corr) / np.math.factorial(K)
def richardson(sn, N=1, K=2):
    SN = np.sum(sn[:N])
    corr = np.zeros(K)
    for n in range(1, K+1, 1):
        factor = np.math.fsum([(-1)**j * (N+K-j)**K 
                             / np.math.factorial(j) / np.math.factorial(K-j)
                             for j in range(0, K-n+1, 1)])
        corr[n-1] = sn[N+n-1] * factor
    return SN + np.math.fsum(corr)

def richardson2(Sn, N=1, K=2):
    Rn = np.math.fsum([(-1)**(K-j) * (N + j)**K * Sn[N+j-1]
                       / np.math.factorial(j) / np.math.factorial(K-j) 
                       for j in range(0, K+1, 1)])
    return Rn

def richardson3(Sn, N=1):
    S4n = Sn[4*N - 1]
    S2n = Sn[2*N - 1]
    S1n = Sn[N - 1]
    r = S4n + 2*S1n - 3*S2n
    p = 3*S2n**2 - S1n**2 - 2*S4n*S1n
    q = S4n*S1n**2 - S2n**3
    root = np.sqrt(p**2 - 4*r*q)
    Rn = np.array([-(p + root), root - p])/ (2*r)
    return Rn
    

def SN(sn, step=50):
    array = np.zeros(sn.shape[0] // step)
    array[0] = np.sum(sn[0:step])
    for k in range(1, array.shape[0], 1):
        array[k] = array[k-1] + np.sum(sn[k*step:(k+1)*step])
    return array
    
def RN(sn, step=50, K=5):
    array = np.zeros(sn.shape[0] // step)
    for k in range(0, array.shape[0], 1):
        array[k] = richardson(sn, N=(k+1)*step - K - 1, K=K)
    return array
   
    
def RN2(sn, step=50, K=5):
    array = np.zeros(sn.shape[0] // step)
    for k in range(0, array.shape[0], 1):
        array[k] = richardson2(sn, N=(k+1)*step - K - 1, K=K)
    return array

def RN3(sn, step=50):
    array = np.zeros(sn.shape[0] // step)
    for k in range(0, array.shape[0], 1):
        array[k] = richardson3(sn, N=(k+1)*step // 4)[0]
    return array
    
def main():
    n = 100
    
    key = 'basel'
    
    if key == 'basel':
        nval = np.arange(1, n+1, 1)
        sn = 1 / nval**2
        S = np.pi**2 / 6
    
    if key == 'exp':
        sn = np.zeros(n)
        sn[0] = 1
        for k in range(1, n, 1):
            sn[k] = sn[k-1] / k
        S = np.e
    
    Sn = np.cumsum(sn)
    
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$|R_N^{(K)} - S|$')
    
    
    colors = special.Colors()
    
    step = 5
    ax.plot(S - SN(sn, step=step), ls='', marker='o', c='k')
    
    r3 = np.abs(S - RN3(Sn, step=step))
    ax.plot(r3, ls='', marker='o', c=colors.prev_color(), 
            label="R3")
    
    for K in range(1, 5, 1):
        r1 = np.abs(S - RN(sn, step=step, K=K))
        r2 = np.abs(S - RN2(Sn, step=step, K=K))
        ax.plot(r1, ls='', marker='o', c=colors.get_color(), 
                label=f"{K = }, R1")
        ax.plot(r2, ls='', marker='x', mew=1, ms=3, c=colors.prev_color(), 
                label=f"{K = }, R2")
        
        ax.plot(np.abs(r1 - r2), c='k', ls=':')
        print(f"Maximum difference: {np.max(np.abs(r1 - r2))}")
    ax.legend()
    special.polish(fig, ax)
    return 0

# def main2():
#     n = 100
#     nval = np.arange(1, n+1, 1)
#     sn = 1 / nval**2
#     Sn = np.cumsum(sn)
    
#     S = np.pi**2 / 6
    
#     N = 10
#     K = 1
    
#     r1 = richardson(sn, N=N, K=K)
#     r2 = richardson2(sn, N=N, K=K)
#     print(r1, r2)

if __name__ == "__main__":
    print(__doc__)
    main()
    
    """
n = 1000
nval = np.arange(1, n+1, 1)
sn = 1 / nval**2
S = np.pi**2 / 6
Sn = np.cumsum(sn)
    """