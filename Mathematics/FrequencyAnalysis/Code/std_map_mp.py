#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides standard maps in multiplce precision arithmetic
"""

# import numpy as np
import mpmath as mp
 

class OrbitParameters:
    def __init__(self, k1=2.25, k2=3.0, k=1.0, n_points=8192):
        self.k1 = k1
        self.k2 = k2
        self.k = k
        self.n_points = n_points

def std_map(q0, p0, k, n_points):
    pi2 = 2 * mp.pi
    q = [q0] * n_points
    p = [p0] * n_points
    for i in range(1, n_points):
        q[i] = (q[i-1] + p[i-1]) % 1.0
        p[i] = (p[i-1] + k / pi2 * mp.sin(pi2 * q[i]) + 0.5) % 1.0 - 0.5
    return q, p

def std_map_4d(q10, p10, q20, p20, orb_par):
    pi2 = 2 * mp.pi
    q1 = [q10] * orb_par.n_points
    p1 = [p10] * orb_par.n_points
    q2 = [q20] * orb_par.n_points
    p2 = [p20] * orb_par.n_points
    for i in range(1, orb_par.n_points):
        q1[i] = (q1[i-1] + p1[i-1]) % 1.0
        q2[i] = (q2[i-1] + p2[i-1]) % 1.0
        coupling = orb_par.k / pi2 * mp.sin(pi2 * (q1[i] + q2[i]))
        p1[i] = (p1[i-1] + coupling + orb_par.k1 / pi2 * mp.sin(pi2 * q1[i]) + 0.5) % 1.0 - 0.5
        p2[i] = (p2[i-1] + coupling + orb_par.k2 / pi2 * mp.sin(pi2 * q2[i]) + 0.5) % 1.0 - 0.5
    return q1, p1, q2, p2


if __name__ == "__main__":
    print(__doc__)
