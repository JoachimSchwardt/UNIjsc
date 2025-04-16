#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a class for simple construction of arbitrary lattices.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup()


class Lattice():
    def __init__(self, a, R, N=None):
        """
        The number of atoms per cell is 'p'
        The dimension of the lattice is 'd'
        
        a == contains the 'p-1' additional basis vectors 
        R == contains the 'd' primitive lattice vectors
        N == contains the number of cells to create along each primitive vector
        """
        
        # convert to np-array
        if type(a) == list: a = np.array(a)
        if type(R) == list: R = np.array(R)
            
        self.a = a
        self.R = R
        
        self.p = a.shape[0] + 1         # number of basis atoms
        self.d = R.shape[1]             # dimension of the lattice
        
        # add cells if 'N' was already given to 'init' 
        if type(N) != type(None):
            self.add_cells(N)
        
        
    def add_cells(self, N):
        """Add cells to the lattice (N == [n_1, ..., n_d])"""
        # convert to np-array
        if type(N) == list:
            N = np.array(N)
        
        if N.shape[0] != self.d:
            msg = f"Dimension of {N} does not match with that of R ({self.d})"
            raise IndexError(msg)
            
        self.N = N
        self.ctr = 0                    # counts the filled number of cells
        Ncells = np.prod(N)             # total number of cells
        points = np.zeros((Ncells, self.p, self.d))
        
        # recursively add cells along all dimensions
        self.__add_cell_axis(points, dim=self.d-1)
        
        self.points = points
    
    
    def __add_cell_axis(self, points, dim, offset=0):
        if dim == 0:
            for i in range(self.N[dim]):
                points[self.ctr, 1:] = self.a
                points[self.ctr] += i * self.R[dim] + offset
                self.ctr += 1
                
        else:
            for i in range(self.N[dim]):
                self.__add_cell_axis(points, dim - 1, offset)
                offset += self.R[dim]
            offset[:dim] = 0
                
    def plot(self, **kwargs):
        if self.d == 2:
            self.plot2d(**kwargs)
            
        elif self.d == 3:
            self.plot3d(**kwargs)
            
            
    def plot2d(self, **kwargs):
        x = self.points[:, :, 0]
        y = self.points[:, :, 1]
        
        fig, ax = plt.subplots()
        ax.set_aspect(1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        
        ax.plot(x, y, **kwargs)
        
        # plot basis vectors
        for offset in self.points[:, 0]:
            for a in self.a:
                xval = offset[0] + np.array([0, a[0]])
                yval = offset[1] + np.array([0, a[1]])
                ax.plot(xval, yval, ls='-', lw=0.5, c='b')
        
        # plot lattice grid lines
        Nx, Ny = self.N
        for i in range(Ny):
            xval, yval = self.points[[i*Nx, (i+1) * Nx - 1], 0].T
            ax.plot(xval, yval, c='k', alpha=0.3, ls='-', lw=0.5)
            
        for i in range(Nx):
            xval, yval = self.points[[i, (Ny-1) * Nx + i], 0].T
            ax.plot(xval, yval, c='k', alpha=0.3, ls='-', lw=0.5)
        
        special.polish(fig, ax)
        
        
    def plot3d(self, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        
        x = self.points[:, :, 0]
        y = self.points[:, :, 1]
        z = self.points[:, :, 2]
        ax.scatter(x, y, z, **kwargs)
        fig.tight_layout()
            

def main():
    key = 'Graphene'
    
    if key == 'Graphene':
        a = 1
        a_arr = [[0, a]]
        R_arr = np.array([[np.sqrt(3), 0], [np.sqrt(3)/2, 1.5]]) * a
        l = Lattice(a_arr, R_arr)
        
        N = [14, 6]
        l.add_cells(N)
        
        l.plot(c='b', ls='', marker='o')
    
    if key == 'complex 2d':
        a = 1
        a_arr = [[0, 0.7*a], [0.25*a, 0.25*a], [-0.3*a, -0.05*a]]
        R_arr = np.array([[np.sqrt(3), 0], [np.sqrt(3)/2, 1.5]]) * a
        l = Lattice(a_arr, R_arr)
        
        N = [8, 6]
        l.add_cells(N)
        
        l.plot(c='b', ls='', marker='o')
        
    if key == 'diamond':
        a = 1
        a_arr = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * a
        R_arr = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]) * a
        l = Lattice(a_arr, R_arr)
        
        N = [4, 3, 2]
        l.add_cells(N)
        
        l.plot(c='b', marker='o')
        
    return 0


if __name__ == "__main__":
    print(__doc__)
    main()