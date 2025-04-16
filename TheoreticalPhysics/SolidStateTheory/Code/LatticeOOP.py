#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object oriented way of creating lattices
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=True)


def point_in_parallelogram(a1, a2, b1, b2, x, y):
    #https://stackoverflow.com/questions/44399749/get-all-lattice-points-lying-inside-a-shapely-polygon
    """
    a1, a2, b1, b2 = 1, 2, 3, -1
    n = 100
    points = 6 * np.random.rand(2, n) - 1
    x = np.linspace(0, 4, 100)
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.grid(True)
    ax.plot(x, a2*x, c='k')
    ax.plot(x, -x/b1, c='k')
    ax.plot(x, a2 - (x-a1) / b1, c='k')
    ax.plot(x, a2*(x-b1) + b2, c='k')
    c = ['r'] * n
    for i in range(n):
        if point_in_parallelogram(a1, a2, b1, b2, points[0, i], points[1, i]):
            c[i] = 'g'
    ax.scatter(points[0], points[1], s=1, c=c)
    """
    return (0 < a2*x - a1*y < b1*a2 - b2*a1
            and 0 < b1*y - b2*x < b1*a2 - b2*a1)

        
class Node:
    def __init__(self, xy):
        self.xy = xy                # coordiantes in real space
        self.NN = []                # nearest neighbours
        
    def _set_NN(self, n, mu):
        """Set the nearest neighbours by lattice and basis coordinates."""            
        self.NN = np.array([[n[i][0], n[i][1], mu[i]] 
                            for i in range(len(mu))])
        
        
class LatticeNode(Node):
    def __init__(self, n, xy, a=[[0, 0]]):
        if type(n) == list:
            n = np.array(n)
            
        self.n = n                  # lattice coordinates
        Node.__init__(self, xy)     # initialize 'xy' and 'NN' for this node
        
        # basis elements of this lattice node
        self.basis = np.array([Node(aval + xy) for aval in a])
        
    def __getitem__(self, mu):
        return self.basis[mu]
        
    def _set_NN(self, n, mu):
        """Set the nearest neighbours by lattice and basis coordinates."""         
        for i in range(self.basis.shape[0]): 
            self.basis[i]._set_NN(n[i] + self.n, mu[i])
            
    def _set_lattice_NN(self, n):
        """Set the nearest lattice neighbours by lattice coordinates."""  
        self.NN = np.array([nval + self.n for nval in n])
        
    def _get_xy(self):
        """Get all basis coordiantes in real space"""
        return np.array([node.xy for node in self.basis])
        
        
        
class Lattice2D():
    def __init__(self, Rx, Ry, n=None, a=None):
        self.Rx = Rx
        self.Ry = Ry
        
        if type(n) != type(None):
            self.add_nodes(n, a)
        
        
    def add_nodes(self, n, a=None):
        """
        Add a grid of 'n = [nx, ny]' cells to the lattice.
        May also create basis nodes in 'a' for each lattice node.
        """
        self.n = n
        self.lattice = np.array([[
                LatticeNode([nx, ny], 
                            nx * self.Rx + ny * self.Ry, 
                            a)
                for ny in range(n[1])] 
            for nx in range(n[0])])
        
        
    def __valid_indices(self, arr, i=0):
        indx = ((arr[:, i] < 0) | (arr[:, i] >= self.n[i]))
        return ~indx

        
    def set_lattice_NN(self, n=None):
        """
        Set the nearest neighbours in the lattice for this node.
        Note: This method ignore the basis, only the lattice is relevant.
        """
        if type(n) == type(None):
            n = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        elif type(n) == list:
            n = np.array(n)
            
        for nx in range(self.n[0]):
            for ny in range(self.n[1]):
                self[nx, ny]._set_lattice_NN(n)
        
                # validate cell indices
                for i in range(2):     # loop over 'x, y' 
                    indx = self.__valid_indices(self[nx, ny].NN, i)
                    self[nx, ny].NN = self[nx, ny].NN[indx]
                        
        
    def set_NN(self, n, mu):
        """
        Set the nearest neighbours by lattice and basis coordinates relative
        to the first basis starting at at [0, 0]
        
        n == cell indices for each NN for each basis atom
             (structure : [[basis_atom1_cell_indx1, 
                            basis_atom1_cell_indx2, ...], 
                           [basis_atom2_cell_indx1, 
                            basis_atom2_cell_indx2, ...],
                           ...])
             
        mu == basis indices for each NN for each basis atom
            (structure : [basis_atom1_basis_indx1, 
                          basis_atom1_basis_indx2, 
                          ...])
        
        
        Example (basis with two atoms)::
            n = [[[0, 0], [0, -1], [1, -1]],     # first basis atom
                 [[0, 0], [0, 1], [-1, 1]]]      # second basis atom
            
            mu = [[1, 1, 1],                     # first 
                  [0, 0, 0]]                     # second
        """
        if type(n) == list: n = np.array(n)
        if type(mu) == list: mu = np.array(mu)
            
        if n.shape[0] != mu.shape[0]:
            msg = ("Incompatible lattice and basis coordinates!\n"
                    + f"n has length {len(n)} and mu has length {len(mu)}.")
            raise IndexError(msg)
        
        for nx in range(self.n[0]):
            for ny in range(self.n[1]):
                self[nx, ny]._set_NN(n, mu)
                
                # validate cell indices
                for m in range(mu.shape[0]):
                    for i in range(2):     # loop over 'x, y' 
                        indx = self.__valid_indices(self[nx, ny][m].NN, i)
                        self[nx, ny][m].NN = self[nx, ny][m].NN[indx]
        
    
    def get_xy(self):
        """Get all node coordiantes in real space."""
        x, y = np.array(
            [xy for node in self.lattice.flatten() for xy in node._get_xy()]
            ).T
        return x, y
                
                
    def plot_grid(self, ax, **kwargs):
        """Plot grid lines representing the unit cell structure."""
        for key, value in [['c', 'k'], ['lw', 0.5], ['alpha', 0.3]]:
            kwargs.setdefault(key, value)
        
        for nx in range(self.n[0]):
            for ny in range(self.n[1]):
                x0, y0 = self[nx, ny].xy
                for indx in self[nx, ny].NN:
                    x1, y1 = self[indx[0], indx[1]].xy
                    if x1 > x0 or y1 > y0:
                        ax.plot([x0, x1], [y0, y1], **kwargs)
        
        
    def plot_NN(self, ax, **kwargs):
        """Plot lines connecting the nearest neighbours for all nodes."""
        for key, value in [['c', 'blue'], ['lw', 0.5]]:
            kwargs.setdefault(key, value)
            
        for nx in range(self.n[0]):
            for ny in range(self.n[1]):
                for mu in range(self[0, 0].basis.shape[0]):
                    x0, y0 = self[nx, ny][mu].xy
                    for indx in self[nx, ny][mu].NN:
                        x1, y1 = self[indx[0], indx[1]][indx[2]].xy
                        if x1 > x0 or y1 > y0:
                            ax.plot([x0, x1], [y0, y1], **kwargs)
        
    
    def plot(self, ax, **kwargs):
        """Plot the entire lattice with primitive grid and basis structure."""
        for key, value in [['c', 'blue'], ['marker', 'o']]:
            kwargs.setdefault(key, value)
            
        x, y = self.get_xy()
        ax.scatter(x, y, **kwargs)
        
        
    def __getitem__(self, n):
        return self.lattice[n[0], n[1]]
    

def main():
    key = 'hexagonal'
    
    if key == 'hexagonal':
        a = 1
        n = [14, 6]
        a_arr = np.array([[0, 0]]) * a
        Rx = np.array([np.sqrt(3), 0]) * a
        Ry = np.array([np.sqrt(3)/2, 1.5]) * a
        
        l = Lattice2D(Rx, Ry, n, a_arr)
        # l.set_NN([[[0, 0], [0, -1], [1, -1]]], 
        #           [[1, 1, 1]])
        l.set_lattice_NN()
    
    if key == 'Graphene':
        a = 1
        n = [14, 6]
        a_arr = np.array([[0, 0], [0, 1]]) * a
        Rx = np.array([np.sqrt(3), 0]) * a
        Ry = np.array([np.sqrt(3)/2, 1.5]) * a
        
        l = Lattice2D(Rx, Ry, n, a_arr)
        l.set_NN([[[0, 0], [0, -1], [1, -1]], [[0, 0], [0, 1], [-1, 1]]], 
                  [[1, 1, 1], [0, 0, 0]])
        l.set_lattice_NN()
        
    if key == 'complex 2d':        
        a = 1
        n = [8, 6]
        a_arr = np.array([[0, 0], [0, 0.7], [0.25, 0.25], [-0.3, -0.05]]) * a
        Rx = np.array([np.sqrt(3), 0]) * a
        Ry = np.array([np.sqrt(3)/2, 1.5]) * a
        
        l = Lattice2D(Rx, Ry, n, a_arr)
        l.set_NN([[[0, 0], [0, 0], [1, -1]], 
                  [[0, 0], [0, 1], [-1, 1]], 
                  [[0, 0], [0, 0], [0, 0]], 
                  [[0, 0], [0, 0], [0, 0]]], 
                 [[2, 3, 1], 
                  [1, 3, 0], 
                  [1, 2, 2], 
                  [0, 3, 3]])
        l.set_lattice_NN()
        
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    l.plot_grid(ax)
    l.plot_NN(ax)
    l.plot(ax)
    special.polish(fig, ax)
        
    return l


if __name__ == "__main__":
    print(__doc__)
    l = main()