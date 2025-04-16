#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object oriented way of creating lattices
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import special
special.setup(UseTex=True)


def point_in_parallelogram(a, b, point):
    #https://stackoverflow.com/questions/44399749/
    #  get-all-lattice-points-lying-inside-a-shapely-polygon
    """
    a = np.array([1, 2])
    b = np.array([3, -1])
    n = 100
    cell = np.array([np.zeros_like(a), a, b, a+b])
    points = 4 * np.random.rand(n, 2) - [0, 1]
    c = ['r'] * n
    for i in range(n):
        if point_in_parallelogram(a, b, points[i]):
            c[i] = 'g'
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.scatter(points[:,0], points[:,1], s=1, c=c)
    ax.scatter(*cell.T, c='k', s=4)
    """
    D = np.linalg.det([a, b])
    if D < 0:
        a, b = b, a
        D *= -1
    return (0 <= np.cross(a, point) < D
            and 0 <= np.cross(point, b) < D)


def get_parallelogram(ab):
    """Returns the parallelepiped formed by the three vectors in 'abc'."""
    a, b = ab
    return np.array([np.zeros_like(a), a, b, a+b])


def get_lattice_points_in_cell(cell):
    """
    cell = get_parallelepiped(lcbv)
    indices = get_lattice_points_in_cell(cell)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.scatter(indices[:,0], indices[:,1], s=1, c='b')
    ax.scatter(*cell.T, c='k')
    """
    rmin, rmax = np.min(cell, axis=0), np.max(cell, axis=0)
    a, b = cell[1:3]
    indices = []
    for x in range(rmin[0], rmax[0] + 1):
        for y in range(rmin[1], rmax[1] + 1):
            point = np.array([x, y])
            if point_in_parallelogram(a, b, point):
                indices.append(point)
    return np.array(indices)


def get_shifted_range(n):
    """n must be a tuple [a, b] with 'a <= 0 < b'."""
    return np.roll(np.arange(n[0], n[1]), n[0])

class Node:
    def __init__(self, xy):
        self.xy = xy              # coordiantes in real space
        self.NN = []                # nearest neighbours

    def _set_NN(self, n, mu):
        """Set the nearest neighbours by lattice and basis coordinates."""
        self.NN = np.array([[*n[i], mu[i]] for i in range(len(mu))])


class LatticeNode(Node):
    def __init__(self, n, xy, a=[[0, 0]]):
        if type(n) == list:
            n = np.array(n)

        self.n = n                  # lattice coordinates
        Node.__init__(self, xy)    # initialize 'xy' and 'NN' for this node

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

    def get_xy(self):
        """Get all basis coordiantes in real space"""
        return np.array([node.xy for node in self.basis])



class Lattice2D():
    def __init__(self, Rx, Ry, n=None, a=None, lcbv=None):
        """
        Rx, Ry   == primitive lattice vectors

        n        == [nx, ny], number of unit cells along each axis
                    Note that the unit cells do not have to correspond to
                    the primitive lattice vectors, see 'lcbv'.

        a        == 2D-array containing the basis coordinates in real space.
                    Note that the origin is NOT included by default!
                    This allows to shift the basis within the lattice cells.

        lcbv     == Acronym for 'linear combination of basis vectors'.
                    This is a '2x2'-integral matrix.
                    Each row contains the coefficients for a linear combination
                    of the primitive lattice vectors.
                    The resulting two vectors will be used as a unit cell,
                    effectively allowing the use of arbitrary structures.
                    The basis will be extended, such that the lattice is unchanged.
        """
        self.dim = 2    # dimension

        if type(lcbv) == type(None):
            self.lcbv = np.eye(self.dim, dtype=int)
        elif len(lcbv) == self.dim:
            if type(lcbv) == list:
                lcbv = np.array(lcbv)
            self.lcbv = lcbv
        else:
            msg = "Incorrect format for parameter 'lcbv'."
            raise TypeError(msg)

        # primitive lattice vectors
        self.pRx = Rx
        self.pRy = Ry

        # non-primitive lattice vectors
        # self.Rx = Rx @ self.lcbv
        # self.Ry = Ry @ self.lcbv
        self.Rx = self.lcbv[0, 0] * Rx + self.lcbv[0, 1] * Ry
        self.Ry = self.lcbv[1, 0] * Rx + self.lcbv[1, 1] * Ry

        # convex hull of the cell in 'primitive lattice coordinates'
        self.cell = get_parallelogram(self.lcbv)

        if type(a) != type(None):
            if type(a) == list:
                a = np.array(a)
            self.a = a
        else:
            self.a = np.array([[0, 0]])    # default basis if none was given

        if np.any(self.lcbv != np.eye(self.dim, dtype=int)):
            self.__extend_basis()

        if type(n) != type(None):
            self.add_nodes(n, self.a)


    def __extend_basis(self):
        """
        Extend the basis with respect to the transformed basis vectors.

        Step 1: Get the primitive lattice indices of every node,
                that lies within the larger 'lcbv-cell'

        Step 2: Add all original basis atoms at all 'new' basis atoms
                This creates 'old_basis_size * number_of_nodes_in_lcbv'
                nodes, which form the new basis.
                (Example: self.a = [[0, 0], [1, 1]] and 3 nodes in
                          the 'lcbv-cell' --> 2 * 3 = 6 'new' basis atoms)
        """
        indices = get_lattice_points_in_cell(self.cell)
        self.a = np.array([a + (ind @ [self.pRx, self.pRy])
                           for a in self.a
                           for ind in indices])


    def __convert_n(self, n):
        """
        Returns an array of the form
            [[nx_min, nx_max], [...], [...]]
        where 'nx_min' is the lowest cell index along the 'x'-axis.
        """
        # start and end cell along each axis
        n_new = np.zeros((self.dim, self.dim), dtype=int)
        if n.ndim == 2:
            n_new = n
        elif n.ndim == 1:
            n_new[:, 1] = n
        return n_new


    def add_nodes(self, n, a=None):
        """
        Add a grid of 'n' cells to the lattice, where
            n = [[nx_min, nx_max],
                 [ny_min, ny_max]]
        May also create basis nodes in 'a' for each lattice node.
        """
        if type(n) == list:
            n = np.array(n)

        n = self.__convert_n(n)
        self.n = n

        self.nx_range = get_shifted_range(n[0])
        self.ny_range = get_shifted_range(n[1])

        self.lattice = np.array([[
                    LatticeNode([nx, ny], nx * self.Rx + ny * self.Ry, a)
                for ny in self.ny_range]
            for nx in self.nx_range])


    def __valid_indices(self, arr, i=0):
        """
        Returns the indices at which 'arr' does not violate the valid lattice
        along axis 'i'.
        """
        indx = ((arr[:, i] < self.n[i, 0]) | (arr[:, i] >= self.n[i, 1]))
        return ~indx


    def set_lattice_NN(self, n=None):
        """
        Set the nearest neighbours in the lattice for this node.
        Note: This method ignores the basis, only the lattice is relevant.
        """
        if type(n) == type(None):
            n = np.array([[1, 0], [-1, 0],
                          [0, 1], [0, -1]])
        elif type(n) == list:
            n = np.array(n)

        for nx in self.nx_range:
            for ny in self.ny_range:
                self[nx, ny]._set_lattice_NN(n)

                # validate cell indices
                for i in range(self.dim):
                    indx = self.__valid_indices(self[nx, ny].NN, i)
                    self[nx, ny].NN = self[nx, ny].NN[indx]


    def set_NN(self, n_mu=None):
        """
        Set the nearest neighbours by lattice and basis coordinates relative
        to the first basis starting at [0, 0]

        You may instead provide an integer corresponding to the number of
        nearest neighbours for each basis atom.
        In that case the corresponding indices will be computed automatically.

        n_mu == cell indices for each NN for each basis atom
             (structure : [[basis_atom1_cell_indx1, basis_atom1_basis_indx1,
                            basis_atom1_cell_indx2, basis_atom1_basis_indx2,
                            ...],
                           [basis_atom2_cell_indx1, basis_atom2_basis_indx1,
                            basis_atom2_cell_indx2, basis_atom2_basis_indx2,
                            ...],
                           ...])
        """
        if type(n_mu) == list:
            n_mu = np.array(n_mu)

        bsize = self.a.shape[0]
        if type(n_mu) == int:
            # Initialize array for the distance to all surrounding nodes
            # Indices for those nodes will be generate in 'indx_all'
            dist_arr = np.zeros(bsize * 3**self.dim - 1)
            indx_all = np.zeros((dist_arr.shape[0], self.dim+1), dtype=int)
            n_mu = np.zeros((bsize, n_mu, self.dim+1), dtype=int)

            ctr = 0
            for mu in range(bsize):
                pos = self[0, 0].basis[mu].xy
                for nu in range(bsize):
                    for nx in [-1, 0, 1]:
                        for ny in [-1, 0, 1]:
                            if nu == mu and nx == ny == 0:
                                continue

                            pos2 = (nx * self.Rx + ny * self.Ry + self.a[nu])

                            dist = np.linalg.norm(pos - pos2)
                            dist_arr[ctr] = dist
                            indx_all[ctr] = [nx, ny, nu]
                            ctr += 1

                # only keep indices corresponding the nearest 'n_mu' neighbours
                n_mu[mu] = indx_all[np.argsort(dist_arr)][:n_mu.shape[1]]
                ctr = 0

        # set indices for nearest neighbours
        for nx in self.nx_range:
            for ny in self.ny_range:
                self[nx, ny]._set_NN(n_mu[:, :, :self.dim], n_mu[:, :, -1])

                # validate cell indices
                for m in range(n_mu.shape[0]):
                    for i in range(self.dim):
                        indx = self.__valid_indices(self[nx, ny][m].NN, i)
                        self[nx, ny][m].NN = self[nx, ny][m].NN[indx]


    def get_xy(self):
        """Get all node coordinates in real space."""
        x, y = np.array(
            [xy for node in self.lattice.flatten() for xy in node.get_xy()]
            ).T
        return x, y


    def plot_grid(self, ax, **kwargs):
        """Plot grid lines representing the unit cell structure."""
        for key, value in [['c', 'k'], ['lw', 0.5], ['alpha', 0.3]]:
            kwargs.setdefault(key, value)

        for nx in self.nx_range:
            for ny in self.ny_range:
                x0, y0 = self[nx, ny].xy
                for indx in self[nx, ny].NN:
                    x1, y1 = self[indx].xy
                    if x1 > x0 or y1 > y0:
                        ax.plot([x0, x1], [y0, y1], **kwargs)


    def plot_NN(self, ax, **kwargs):
        """Plot lines connecting the nearest neighbours for all nodes."""
        for key, value in [['c', 'blue'], ['lw', 0.5]]:
            kwargs.setdefault(key, value)

        for nx in self.nx_range:
            for ny in self.ny_range:
                for mu in range(self[0, 0].basis.shape[0]):
                    x0, y0 = self[nx, ny][mu].xy
                    for indx in self[nx, ny][mu].NN:
                        x1, y1 = self[indx[:self.dim]][indx[-1]].xy
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


def main(key = 'graphene'):
    if key == 'graphene':
        a = 1
        n = [15, 6]
        a_arr = np.array([[0, 0], [0, 1]]) * a
        Rx = np.array([np.sqrt(3), 0]) * a
        Ry = np.array([np.sqrt(3)/2, 1.5]) * a

        lcbv = np.array([[1, 0], [-1, 2]])
        # lcbv = None
        # a_arr = None

        l = Lattice2D(Rx, Ry, n, a_arr, lcbv)
        l.set_NN(3)     # set number of nearest neighbours for every basis atom
        l.set_lattice_NN([[1, 0], [0, -1], [1, 1], [-1, -1]])

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
