"""
Understand what is happening with the frequencies
 at the horseshoe structure.

see [Lan2012, Lan2016, RicLanBaeKet2014, LanRicBaeKet2014]
for more details.

[Lan2012, p.31]:
For FSC the numerically obtained frequencies
ω = (ω1 , ω2 ) of many regular orbits lie on resonances such as
1 : −1 : 0,    1 : 1 : 1    or      m : 1 : n,
although resonances are not possible for regular tori in a perturbed system,
see Section 2.3.

FIXME: MF: I do not understand this right now..

Commont for frequency analysis with respect to  -1 : 3 : 0 !
[LanRicBaeKet2014, p. 6]:
For a torus around Mres additionally to the frequency XXX
independent frequency XXX is computed.

"""

from __future__ import (division, print_function)

import numpy as np

from explorator.imports.maps.map_standard4d_v3 import Mapping


from CPG.naff import NaffND, Naff1D


def example_horseshoe():
    """
    determine frequency of the horseshow region
    """
    mapping = Mapping(2.25, 3.0, 1.0)
    N = 8192

    # --- Point 2
    # Result= [ 0.2962509   0.28652886] [ 0.2962509   0.10152525]
    # [ 0.2962509   0.10152525] 0.185003607533 5.43902367589e-10

    init_point = np.array([0.08090725947182,
                           -0.07995087108008,
                           0.44041467529610,
                           0.59718510271600])

    torus = mapping.mapN(init_point, N)

    points = torus.points

    # defining signal using projections onto canonical planes
    z1 = points[:, 2] - 1j*points[:, 0]
    z2 = points[:, 3] - 1j*points[:, 1]

    # determines frequency using Naff1D for each signal
    naff1d_nu_1_comp = Naff1D(z1)
    naff1d_nu_2_comp = Naff1D(z2)

    naff1d_nu_1 = naff1d_nu_1_comp.compute_frequency()
    naff1d_nu_2 = naff1d_nu_2_comp.compute_frequency()
    print(80*"*")
    print("Naff1D - z1 signal:", naff1d_nu_1)
    print("Naff1D - z2 signal:", naff1d_nu_2)
    print("Both frequencies are the same!")
    print(80*"*")

    # now using NaffND
    naff_nd_comp = NaffND(points, N_order=2, N_harmonics=1)

    # compute with projection 0: q1 p1 signal!
    # this is the same as .compute_frequncey([2, 0])
    naffnd_nu_1 = naff_nd_comp.compute_frequency(0)

    # compute with projection 1: q2 p2 signal!
    # this is the same as .compute_frequncey([3, 1])
    naffnd_nu_2 = naff_nd_comp.compute_frequency(1)

    print(80*"*")
    print("NaffND with N_order =", naff_nd_comp.N_order,
          "and N_harmonics =", naff_nd_comp.N_harmonics)
    print(naffnd_nu_1)
    print(naffnd_nu_2)
    print(80*"*")

    naff_nd_comp = NaffND(points, N_order=2, N_harmonics=3)

    # compute with projection 0: q1 p1 signal!
    # this is the same as .compute_frequncey([2, 0])
    naffnd_nu_1 = naff_nd_comp.compute_frequency(0)

    # compute with projection 1: q2 p2 signal!
    # this is the same as .compute_frequncey([3, 1])
    naffnd_nu_2 = naff_nd_comp.compute_frequency(1)

    print(80*"*")
    print("NaffND with N_order =", naff_nd_comp.N_order,
          "and N_harmonics =", naff_nd_comp.N_harmonics)
    print(naffnd_nu_1)
    print(naffnd_nu_2)
    print(80*"*")

    print("Substracting higher harmnonics of the first frequency"
          "from the signal results in a different second frequency.")

    print("Wild speculation: Does the second frequency correspond "
          "to a frequency analysis with respect to the resonance line?")


if __name__ == "__main__":
    example_horseshoe()
###############################################################################
