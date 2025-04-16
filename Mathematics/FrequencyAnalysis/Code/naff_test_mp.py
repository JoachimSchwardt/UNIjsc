"""
NAFF test in multiple precision math
"""

import functools
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import std_map_mp as mp_maps
import naff_mp
import naff
from naff_tools import naff_testbench_mp, get_n_arr, naff_testbench
import mpl_special
import window_functions_mp as win_mp
mp.mp.dps = 30

def get_signal(n_points=8192):
    orb_par = mp_maps.OrbitParameters(n_points=n_points)
    q10 = mp.mpf("0.5")
    p10 = mp.mpf("0.05")
    q20 = mp.mpf("0.5")
    p20 = mp.mpf("0.05")
    q1, p1, q2, p2 = mp_maps.std_map_4d(q10, p10, q20, p20, orb_par)
    signal = [q1[i] - 0.5 + 1j * p1[i] for i in range(len(q1))]
    return signal


def test_signal(signal, n_arr, n_freq=1, num_j=10,
                methods = (naff_mp.naffnd_gauss_mp,),
                # windows=(functools.partial(win_mp.hann_weights, a_k=1),
                #          functools.partial(win_mp.hann_weights, a_k=2),
                #          win_mp.gauss_weights,),
                # names=("H_1", "H_2", r"\mathrm{gauss}",)
                args=((280.0,),),
                names=(r"\mathrm{gauss}",)
                ):
    assert(len(signal) == n_arr[-1])
    fig, ax = plt.subplots()
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$|\Delta \nu_N|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_arr[0], n_arr[-2])

    for ctr in range(len(methods)):
        method, arg = methods[ctr], args[ctr]
        # name = mpl_special.mathrm(window.__name__)
        name = names[ctr]
        freq, diff = naff_testbench_mp(signal, n_arr, method, n_freq, *arg, max_prec=1e-30)
        ax.plot(n_arr[:-1], diff, ls='--', lw=0.5, marker='o',
                label=fr"$\nu_{{{name}}} = {freq}$")

    ax.legend()
    mpl_special.polish(fig, ax)


def main():
    print(__doc__)
    # n_max = 2**20
    # signal = get_signal(n_max)
    # n_arr = get_n_arr(n_n=20, n_max=n_max, n_min=256)
    # test_signal(signal, n_arr, args=((140.0,), (280.0,), (1,), (2,), ),
    #             names=(r"\mathrm{gauss},\alpha=140", r"\mathrm{gauss},\alpha=280", 
    #                    "H_1", "H_2"),
    #             methods=((naff_mp.naffnd_gauss_mp, naff_mp.naffnd_gauss_mp,
    #                       naff_mp.naffnd_cos_mp, naff_mp.naffnd_cos_mp, ) ),
    #             )

    # z = np.array(signal, dtype=np.complex128)
    # freq, diff = naff_testbench(z, n_arr, naff.naffnd_gauss)
    # plt.plot(n_arr[:-1], diff, ls='--', marker='o', ms=2, mew=0, lw=0.5,
    #          label=r'gauss, $\alpha=140$, double-prec.')
    
    # from scipy.signal.windows import chebwin
    # freq, diff = naff_testbench(z, n_arr, naff.naffnd_num, 
    #                             lambda x: chebwin(x, at=250))
    # plt.plot(n_arr[:-1], diff, ls='--', marker='o', ms=2, mew=0, lw=0.5, 
    #          label='Dolph-Chebyshev at 250, double-prec.')
    # plt.legend()
    
    # weights_mp = win_mp.get_window(len(signal), win_mp.gauss_weights)
    # print(naff_mp.naffnd_num_mp(signal, weights_mp, 2, return_coeff=True))
    return 0


if __name__ == "__main__":
    main()
