# -*- coding: utf-8 -*-
"""
Transforming two dimensional tori i a 4d phase space
"""

from std_map import Mapping4dCyl as Mapping
import numpy as np
import matplotlib.pyplot as plt
from WBA_4D_tests import Colors

def eig_inertia_tensor(r, ndim=4, RetEigVal=0):
    iTensor = (np.sum(r*r) * np.eye(ndim) - np.dot(r, r.T)) / len(r[0]) 
    eigVal, eigVec = np.linalg.eigh(iTensor)
    if RetEigVal:
        return eigVal, eigVec
    else:
        return eigVec

def transform_nd_torus(r, ndim=4):
    eigVec = eig_inertia_tensor(r, ndim, RetEigVal=0)
    return np.dot(eigVec.T, r)

def plot_torus4d_transform(init=None, k1=2.25, k2=3.0, k=1.0, N=2**10):
    colors = Colors()
    colors = [colors.get_color() for _ in range(8)]
    if type(init) == type(None):
        init = [0.08090725947182, -0.07995087108008,
                0.44041467529610, 0.59718510271600]
    points = np.array(Mapping(k1, k2, k).mapN(*init, N))
    pTrans = transform_nd_torus(points)
    indxList = [None, [0,1,2,3], [0,2,3,1], [0,3,1,2], 
                [0,1,1,2], [1,2,2,3], [2,3,3,0], [3,0,0,1],]
    fig, ax = plt.subplots(4, 4, figsize=(10, 14))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            k = j + 4*(i//2)
            indx = indxList[k]
            c = colors[k]
            if type(indx) == type(None):
                ax[i,j].plot(points[2+i%2, :], points[0+i%2, :], ls='',
                           marker='o', ms=2, mew=0, c=c)
            else:
                print(indx, indx[(0+2*(i%2)) % 4], indx[(1+2*(i%2)) % 4])
                ax[i,j].plot(pTrans[indx[(0+2*(i%2)) % 4], :], 
                             pTrans[indx[(1+2*(i%2)) % 4], :],
                             ls='', marker='o', ms=2, mew=0, c=c)
    plt.tight_layout()
    return 

if __name__ == "__main__":
    plot_torus4d_transform()
