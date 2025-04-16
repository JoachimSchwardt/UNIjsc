# -*- coding: utf-8 -*-
"""
Numerically computes averages in high dimensions.
"""

import numpy as np

def uniform_dist(xmin=0, xmax=1, N=1000, dim=2):
    """Generates a uniform distribution in 'dim' dimensions."""
    dist = np.random.uniform(xmin, xmax, size=N*dim)
    return dist.reshape((N, dim))

def uniform_circular_dist(rad=1, N=1000, dim=2):
    """Generates a uniform distribution on a circle"""
    dist = uniform_dist(-rad, rad, N, dim)
    rad_sq_dist = np.sum(dist**2, axis=1)
    indx = np.where(rad_sq_dist <= rad**2)
    return dist[indx]

def alt_uniform_circular_dist(rad, N=1000):
    """Uses transformed probability densities"""
    r_dist = np.sqrt(np.random.uniform(0, rad**2, size=N))  
    phi_dist = np.random.uniform(0, 2*np.pi, size=N)
    return np.array([r_dist, phi_dist])

def distance_polar_coord(p1, p2):
    """Computes the norm of the difference between sets of polar coordiantes"""
    r1, phi1 = p1
    r2, phi2 = p2
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi1 - phi2))

def main():
    print(__doc__)
    rad = 10
    N = 100000
    diff_theo = 128 * rad / (45 * np.pi)
    print(f"True value is {diff_theo}")
    
    dist = uniform_circular_dist(rad, 2*N)
    length = dist.shape[0]
    dist = dist[:2 * (length // 2)]
    dist1, dist2 = dist[::2], dist[1::2]
    print("Result from square distribution (reduced to circle):")
    difference = np.sqrt(np.sum((dist1 - dist2)**2, axis=1))
    print(np.mean(difference))
    
    print("Result from circular distribution of same size")
    rad_dist = alt_uniform_circular_dist(rad, 2*N)
    p1, p2 = rad_dist[:, :N], rad_dist[:, N:]
    difference = distance_polar_coord(p1, p2)
    print(np.mean(difference))
    
    

if __name__ == "__main__":
    main()