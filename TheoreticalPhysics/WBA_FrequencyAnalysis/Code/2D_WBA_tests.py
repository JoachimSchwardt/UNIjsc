# -*- coding: utf-8 -*-
"""
Test signals for WBA frequency analysis.
"""
import numpy as np
import WBA_tools
import WBA_core
from std_map import _std_map, _std_map_multi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import functools
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams["figure.dpi"] = 50

# # for comparison
# from CPG.naff.examples.std_map_frequencies import compute_freq
# from CPG.naff.utils.examples.show_2d_torus_in_4d import 


def _2f_signal(freq, freq2, ampl, N, qOffset=0.0, pOffset=0.0, 
               qFactor=None, pFactor=None):
    """
    Creates a pure signal with a main frequency of 'freq' and amplitude 1.
    Adds a second signal with ampltitude 'ampl' and frequency 'freq2'
    May add offsets and scaling in both coordinates.
    """
    Narr = np.arange(0.0, N, 1.0, dtype=np.float64)
    z = (np.exp(2.0*1j*np.pi * freq * Narr) + 
         ampl * np.exp(2.0*1j*np.pi * freq2 * Narr))
    if qFactor == None or pFactor == None:
        return z.real + qOffset, z.imag + pOffset
    return (z.real + qOffset) * qFactor, (z.imag + pOffset) * pFactor

def plot_2f_signal(nrows, ncols, freq, freq2, ampl, N, qOffset=0.0,
                   pOffset=0.0, qFactor=1.0, pFactor=1.0):
    fig, ax = plt.subplots(nrows, 2*ncols, figsize=(15, 10))
    ax1 = ax[::2].flatten()
    ax2 = ax[1::2].flatten()
    for i in range(len(freq)):
        q, p = _2f_signal(freq[i], freq2[i], ampl[i], N, qOffset,
                          pOffset, qFactor[i], pFactor[i])
        ax1[i].plot(q, p, ls='', marker='o', ms=3, mew=1)
        ax1[i].set_aspect(1.0)
        phi, r = WBA_core.map_arctan2(q, p)
        ax2[i].plot(phi, r, ls='', marker='o', ms=3, mew=1)
        ax1[i].set_title(f"$\\nu={freq[i]}, \\nu_{{2}}={freq2[i]}$, " 
                         + f"$A={ampl[i]}$", fontsize=12)
    return
    
            
def art_signal(freq, freq2, ampl=0.1, Nmin=5.0, Nmax=14.0, NN=200,
               WBAOnly=1, _dig=3, qOffset=0.0, pOffset=0.0, 
               qFactor=1.0, pFactor=1.0, thresh=1e-5):
    """
    Test of frequency analysis for artificial signals with arctan2-mapping.
    """
    Narr = WBA_core.N_arr(Nmin, Nmax, NN)
    N2f = np.max(Narr)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
        
    q, p = _2f_signal(freq, freq2, ampl, N2f, qOffset, pOffset,
                      qFactor, pFactor)
    q, p = WBA_core.map_arctan2(q, p)
    ax[0].plot(q[:NN], p[:NN], c='b', ls='', marker='o', ms=2, mew=1,
               label='MapArctan2')
    if WBAOnly:
        freq_arc = WBA_tools.conv(ax[1], q, p, Narr, _dig=_dig, thresh=thresh,
                                  mapMode='arctan2')
        print(f"Arctan2: WBA={freq_arc}")
    else:
        freq_arc = WBA_tools.compare_conv(ax[1], q, p, Narr, freq, c='b',
                                          _dig=_dig, MapToCircle=0, 
                                          thresh=thresh, mapMode='arctan2')
        print(f"Arctan2: WBA={freq_arc[0]}, Naff={freq_arc[1]}")
        
    ax[0].legend(fontsize=14)
    plt.show()
    return

def art_signal_grid(Nf=100, Nf2=100, ampl=0.2, N=1000, step=None,
                    UseNaff=0):
    phi3 = (1 + np.sqrt(5)) / 3
    if step != None:
        phi3 = step * Nf
        
    max_f = 1.0
        
    freq = np.arange(phi3 / (2*Nf), max_f, phi3 / Nf)
    freq2 = np.arange(phi3 / (2*Nf2), max_f, phi3 / Nf2)
    title = f"Signal length {N} and amplitude {ampl} of secondary frequency"
    
    if UseNaff:
        freqGrid = WBA_tools._2f_grid_Naff(freq, freq2, ampl, N)
        title += " using Naff"
    else:
        freqGrid = WBA_tools._2f_grid(freq, freq2, ampl, N)
        
    freqTrue = np.outer(freq, np.ones(len(freq)))
    # print(freqGrid, freq)
    indx = (np.abs(freqGrid - freqTrue) > np.abs(1 - freqGrid - freqTrue))
    freqGrid[indx] = 1 - freqGrid[indx]
    # print(freqGrid)
    
    absDiffGrid = np.abs(freqGrid - freqTrue)
    absDiffGrid[absDiffGrid < 1e-16] = 1e-16
    absDiffLog = -np.log10(absDiffGrid)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xlabel("secondary frequency", fontsize=14)
    ax.set_ylabel("main frequency", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_aspect(1.0)
    
    img = WBA_tools.imshow_grid(ax, freq, freq2, absDiffLog)
    fig.colorbar(img)
    fig.tight_layout()
    plt.show()
    return

def art_signal_interactive(freq=0.37, freq2=0.6083, ampl=0.9, 
                           Nmin=5.0, Nmax=14.0, NN=200, _dig=3, step=None, 
                           qOffset=0.0, pOffset=0.0, CompareToNaff=0, 
                           qFactor=1.0, pFactor=1.0, ConfineSignal=0, 
                           thresh=2.0):
    """
    Interactive plot to visualize the convergence of the frequency analysis of
    an artificial signal with two frequencies and offset (see '_2f_signal'). 
    For this, the absolute difference of the calculate frequency to the 'true'
    value of 'freq' is shown as a function of the signal length.
      
    Parameters:
        freq = frequency of main signal
        freq2 = frequency of secondary signal added to the main signal
        ampl = amplitude of the secondary signal
        Nmin, Nmax = range of signal lengths is [2**Nmin ... 2**Nmax]
        NN = total number of different signal lengths used
        _dig = number of digits displayed for frequency values
        qOffset, pOffset = Offsets for creating the '_2f_signal'
        CompareToNaff = comparison to frequency analysis using Naff if 'True'
        
    """
    Narr = WBA_tools.N_arr(Nmin, Nmax, NN)    # different signal lengths
    N2f = np.max(Narr)                  # maximal signal length
    q, p = _2f_signal(freq, freq2, ampl, N2f, qOffset, pOffset, 
                      qFactor, pFactor)
    
    # step sizes for sliders
    a_step = 0.01
    f_step = (1 + np.sqrt(5)) / 300    # irrational step size (golden mean)
    if step != None:
        f_step = step

    # fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    fig = plt.figure(figsize=(15, 10))
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot(2, 3, 3)      
    ax2 = plt.subplot(2, 3, 6)
    
    ax = [ax0, ax1, ax2]
    
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    # ax[0].margins(x=0)
    if ConfineSignal:
        ax[1].axis([-2 + qOffset, 2 + qOffset, -2 + pOffset, 2 + pOffset])
    ax[1].plot(q[:NN], p[:NN], ls='', marker='o', ms=2, mew=1)
    
    plt.subplots_adjust(bottom=0.25)
    
    if CompareToNaff:
        WBA_tools.compare_conv(ax[0], q, p, Narr, _dig, freq, ShowLegend=0, 
                               thresh=thresh, MapToCircle=0, mapMode='arctan2')
    else:
        WBA_tools.conv(ax[0], q, p, Narr, _dig, freq, thresh=thresh,
                       mapMode='arctan2')
    phi, r = WBA_core.map_arctan2(q, p)
    ax[2].plot(phi[:NN], r[:NN], ls='', marker='o', ms=2, mew=1)
        
    # plt.axes :: [x0, y0, length, height] for slider size and position
    axColor = 'lightgoldenrodyellow'
    axFreq = plt.axes([0.2, 0.17, 0.65, 0.03], facecolor=axColor)
    axFreq2 = plt.axes([0.2, 0.12, 0.65, 0.03], facecolor=axColor)
    axAmp = plt.axes([0.2, 0.07, 0.65, 0.03], facecolor=axColor)
    
    sFreq = Slider(axFreq, 'FreqSignal', 0.005, 1.0, 
                   valinit=freq, valstep=f_step)
    sFreq2 = Slider(axFreq2, 'Freq2', 0.005, 1.0, 
                        valinit=freq2, valstep=f_step)
    sAmp = Slider(axAmp, 'Ampl', 0.0, 1.0, 
                  valinit=ampl, valstep=a_step)
    
    def update(val):
        """
        Update method is called on every change of a slider.
        """
        # get current slider values
        ampl = sAmp.val
        freq = sFreq.val
        freq2 = sFreq2.val
        
        # create new two frequency signal
        q, p = _2f_signal(freq, freq2, ampl, N2f, qOffset, pOffset, 
                          qFactor, pFactor)
        ax[1].plot(q[:NN], p[:NN], ls='', marker='o', ms=2, mew=1)
        
        # visualize convergence of frequency analysis for the updated signal
        if CompareToNaff:
            WBA_tools.compare_conv(ax[0], q, p, Narr, _dig, freq, ShowLegend=0,
                                   thresh=thresh, MapToCircle=0,
                                   mapMode='arctan2')
        else:
            WBA_tools.conv(ax[0], q, p, Narr, _dig, freq, thresh=thresh,
                           mapMode='arctan2')
            
        # map using arctan2 method
        phi, r = WBA_core.map_arctan2(q, p)
        ax[2].plot(phi[:NN], r[:NN], ls='', marker='o', ms=2, mew=1)
        
        ########### analysis of wrapped phi-values
        # phiDiff = phi[1:]-phi[:-1]
        # print("data info phi diff\n", max(phiDiff), min(phiDiff), 
        #       np.mean(phiDiff), np.std(phiDiff))
        # abc = phiDiff % 1.0
        # a2 = ((phiDiff - 0.5) % 1.0) + 0.5
        # print("3 freq, abc,wba,a2\n",
        #       WBA_core._WBA_single(abc),WBA_core._WBA_single(a2))
        # print("data info abc\n", max(abc), min(abc), 
        #       np.mean(abc), np.std(abc))
        # print("data info abc\n", max(a2), min(a2), 
        #       np.mean(a2), np.std(a2))
        
        # if np.any(abc < 0.25) and np.any(abc > 0.75):
        #     abc[abc < 0.5] += 1
        #     print(WBA_core._WBA_single(abc))
        #     a2[a2 < 0.5] += 1
        #     print(WBA_core._WBA_single(a2))
        # print("##################################################")
        fig.canvas.draw_idle()
    
    sFreq.on_changed(update)
    sFreq2.on_changed(update)
    sAmp.on_changed(update)
    
    def _reset_click(event, ax):
        """
        Resets a subplot on right mouse button click.
        """
        mode = event.canvas.toolbar.mode
        for i in range(len(ax)):
            if event.inaxes == ax[i] and mode == '' and event.button == 3:
                ax[i].lines = []
                event.canvas.draw() 
        pass
    
    reset_click = functools.partial(_reset_click, ax=ax)
    fig.canvas.mpl_connect('button_press_event', reset_click)
    
    plt.show()
    return

def _action_abs_diff(ax, qpos, ppos, K, Npoints, Narr, ShowCos=0):
    """Documentation in 'chaos_indicator'."""
    q, p = _std_map(qpos, ppos, np.max(Narr), K)
    absDiff = WBA_tools._absdiff_N2N(Narr, p)
    
    # color = cm.hot()
    # limit of 5.5 from SanMei2020 page 5 left side middle
    # ChaosIndicatorFlag = True if absDiff[-1] > 10**(-5.5) else False
    
    # c = 'r' if ChaosIndicatorFlag else 'b'
    
    # cm.'colormap_name_as_string', '_r' uses reversed colors
    c = cm.viridis_r(-np.log10(absDiff[-1]) / 16)   
    
    ax[1].plot(Narr, absDiff, lw=1.5, c=c)
    if ShowCos:
        absDiffCos = WBA_tools._absdiff_N2N(Narr, np.cos(2*np.pi*q))
        ax[1].plot(Narr, absDiffCos, lw=1.5, ls='--', c=c)
    ax[0].plot(q[:Npoints], p[:Npoints], marker='o', ls='', ms=2, mew=1, c=c)
    return absDiff[-1]

def chaos_indicator(K=1.0, Npoints=200, Nmin=5.0, Nmax=16.0, NN=100, 
                    ShowCos=0):
    """
    Uses '_action_abs_diff' for the computation (called on mouse click).
    
    Calculates an orbit of the standard map using the mouse position on click 
    as initial values (qpos, ppos). Visualizes the convergence of the ferquency
    analysis for different orbit lengths in a loglog plot. 
    
    For this, the WBA is calculated for the first and second halves of each
    orbit and their difference (see 'WBA._abs_diff_N2N') is used as an 
    indicator of chaos (larger differences correspond to more chaotic orbits). 
    
    The color corresponds to the number of digits of precision reached at the
    maximum orbit length (see colorbar).
    
    Parameters:
        K = kicking strength for standard map
        Npoints = number of iterations displayed for each orbit 
        Nmin, Nmax = range of orbit lengths is [2**Nmin ... 2**Nmax]
        NN = total number of different orbit lengths used
        ShowCos = comparison to WBA with 'cos(2*np.pi*q)' instead of 'p'
    """
    Narr = WBA_tools.N_arr(Nmin, Nmax, NN)
    Narr_max = np.max(Narr)
    
    if Npoints > Narr_max:
        WarningMsg = (f"Warning: Npoints={Npoints} has to be smaller than " + 
                      f"max(Narr)={Narr_max} !" + 
                      "Npoints was set equal to max(Narr).")
        print(WarningMsg)
        Npoints = Narr_max
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].axis([0, 1, -0.5, 0.5])
    ax[0].set_title(f"K={K}, N={Npoints}")
    ax[1].axis([Narr[0], Narr[-1], 1e-16, 1])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    mouse_click = functools.partial(WBA_tools._mouse_click, ax=ax, K=K, 
                                    Npoints=Npoints, Narr=Narr, 
                                    _action=_action_abs_diff, args=(ShowCos))
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    plt.show()
    return

def _indicator_map(data, q, p, title, cmap=None, axis=[0.0, 1.0, -0.5, 0.5]):    
    p0count = len(p[0])    
    N = len(p[:, 0])
    if N * p0count > 4*10**7:
        marker = '.'
        ms, mew = 1, 0
    elif N * p0count > 10**6:
        marker = '.'
        ms, mew = 2, 0
    elif N * p0count > 10**4:
        marker = '.'
        ms, mew = 2, 1
    else:
        marker = 'o'
        ms, mew = 2, 1
    print(f"Marker is {marker} of size {ms} and edge width {mew}")
     
    if cmap == None:
        cmap = 'viridis_r'
    colormap = cm.get_cmap(cmap)
    dataForCmap = data - np.min(data)
    dataForCmap /= np.max(dataForCmap)
    cMap = colormap(dataForCmap)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(title, fontsize=14)
    ax.axis(axis)
    
    p = (p + 0.5) % 1.0 - 0.5
    for i in range(p0count):
        ax.plot(q[:, i], p[:, i], ls='', marker=marker, 
                ms=ms, mew=mew, c=cMap[i])
        
    # create cmap colorbar with correct limits
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(np.min(data), np.max(data))
    fig.colorbar(cmap, ax=ax)
    # plt.show()
    pass

def _q0p0_array(q0, p0min, p0max, p0count):
    p0 = np.linspace(p0min, p0max, p0count, dtype=np.float64)
    if type(q0) == float:
        q0 = np.full(p0count, q0, dtype=float)
    elif type(q0) == type(None):
        q0 = np.random.uniform(0.0, 1.0, p0count)
    return q0, p0

def _freq_map(q, p, N=1024, K=0.7, mode='WBA', 
              mapMode='decision', thresh=1e-3, cmap=None):  
    p0count = len(p[0])
    title = (f"Frequency map for K={K} and {N} iterations " 
             + f"\nof {p0count} initial conditions")
    if mode == 'WBA': 
        title += " using WBA"
        freq = WBA_core.WBA(q - 0.5, p, thresh=thresh, mapMode=mapMode) % 1.0
    elif mode == 'Naff':
        title += " using Naff"
        freq = WBA_tools._Naff_multi(q, p, N, K, MapToCircle=1, 
                                     Continuous=0) % 1.0
    elif mode == 'absDiff':
        title += (" using WBA and Naff" 
                 + "\nColor value corresponds to number of equal digits")
        freqWBA = WBA_core.WBA(q - 0.5, p, thresh=thresh, 
                               mapMode=mapMode) % 1.0
        freqNaff = WBA_tools._Naff_multi(q, p, N, K, MapToCircle=1,
                                         Continuous=0) % 1.0
        absDiff = np.abs(freqWBA - freqNaff)
        absDiff2 = np.abs(1 - freqWBA - freqNaff)
        indx = (absDiff > absDiff2)
        absDiff[indx] = absDiff2[indx]
        absDiff[absDiff < 1e-16] = 1e-16
        freq = -np.log10(absDiff)
    
    _indicator_map(freq, q, p, title, cmap=cmap)
    pass

def freq_maps(q0=None, p0min=-0.5, p0max=0.5, p0count=200, N=1024, K=0.7, 
              ShowWBA=1, ShowNaff=0, ShowDiff=0, mapMode='decision',
              thresh=1e-3, cmap=None):
    q0, p0 = _q0p0_array(q0, p0min, p0max, p0count)
    q, p = _std_map_multi(q0, p0, N, K)
    
    if ShowWBA:
        _freq_map(q, p, N, K, mode='WBA', mapMode=mapMode, thresh=thresh,
                  cmap=cmap)
    if ShowNaff:
        _freq_map(q, p, N, K, mode='Naff', cmap=cmap)
    if ShowDiff:
        _freq_map(q, p, N, K, mode='absDiff', mapMode=mapMode, thresh=thresh)
    pass

def chaos_indicator_map(q0=None, p0min=-0.5, p0max=0.5, p0count=100, 
                        K=0.7, N=200):
    """
    Calculates several orbits of the standard map. 
    
    The color corresponds to the number of digits of precision reached at the
    maximum orbit length (see colorbar). For this, the WBA is calculated for 
    the first and second halves of each orbit and their differences 
    (see 'WBA._abs_diff_N2N_multi') are used as an indicator of chaos 
    (larger differences correspond to more chaotic orbits).
    
    Parameters:
        K = kicking strength for standard map
        N = number of iterations for each orbit 
        q0 = initial 'q' values as an array of points. Defaults to a random
             uniform distribution. If only one value is given, its used for
             every initial '(q, p)' pair
        p0min, p0max = minimum and maximum initial 'p' values
        p0count = total number of inital 'p' values (equidistant)
        ShowCos = comparison to WBA with 'cos(2*np.pi*q)' instead of 'p'
    """
    q0, p0 = _q0p0_array(q0, p0min, p0max, p0count)
        
    title = (f"Convergence of WBA as chaos indicator for {N} iterations\n" 
             + r"Colormap value gives the $\log_{10}$ of " 
             + r"$|WB_{[1,N]}(q,p)-WB_{[N+1,2N]}(q,p)|$")
    
    q, p = _std_map_multi(q0, p0, N, K)
    absDiff = WBA_tools._absdiff_N2N_multi(N, p)
    
    absDiff[absDiff < 1e-16] = 1e-16        # avoid zeros for 'log'
    absDiffLog = -np.log10(absDiff)  
    _indicator_map(absDiffLog, q, p, title)
    return


def chaos_indicator_grid(Nq=100, Np=None, N=10**3, K=0.0, UseCos=0, UseNaff=0,
                         UseImshow=1, ReturnValues=0, absDiffLog=None):
    if Np == None:
        Np = Nq
    
    eps = 0#1 / Nq
    q0 = np.linspace(eps, 1.0 - eps, Nq)
    p0 = np.linspace(-0.5 + eps, 0.5 - eps, Np)
    title = (f"Orbit length {N} and K={K} on a ({Nq}$\\times ${Np}) grid")
    
    ComputeData = True
    if type(absDiffLog) != type(None):
        print("Using given data as absolute difference")
        ComputeData = False
    if UseCos:
        title += r" using WBA with $\cos(2\pi q)$"
        if ComputeData:
            absDiffGrid = WBA_tools._grid_absdiff_N2N(q0, p0, N, K, UseCos).T
    elif UseNaff:
        title += " using Naff"
        if ComputeData:
            absDiffGrid = WBA_tools._grid_absdiff_N2N_Naff(q0, p0, N, K, 0).T
    else:
        title += " using WBA with $p$"
        if ComputeData:
            absDiffGrid = WBA_tools._grid_absdiff_N2N(q0, p0, N, K, UseCos).T
        
    if ComputeData:
        absDiffGrid[absDiffGrid < 1e-16] = 1e-16
        absDiffLog = -np.log10(absDiffGrid)
        if ReturnValues:
            return absDiffLog, q0, p0
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(title, fontsize=14)
    if UseImshow:
        ax.set_xlabel("q", fontsize=14)
        ax.set_ylabel("p", fontsize=14)
        
        img = WBA_tools.imshow_grid(ax, q0, p0, absDiffLog)
        fig.colorbar(img)
        
    else:
        ax.hist(absDiffLog.flatten(), bins=Nq // 3, color='k', lw=1, 
                histtype='step', density=True)
        
    fig.tight_layout()    
    plt.show()
    if ComputeData:
        return absDiffLog, q0, p0

def chaos_along_vector(initial, Nq=100, N=1000, K=0.0, UseCos=0,
                       ReturnValues=0, ShowHist=0):
    [q1, p1, q2, p2] = initial
    q0 = np.linspace(q1, q2, Nq, dtype=np.float64)
    p0 = np.linspace(p1, p2, Nq, dtype=np.float64)
    
    q, p = _std_map_multi(q0, p0, N, K)
    title = (f"Chaos indicator for orbit lengths {N} and K={K} " + 
             f"from {q1, p1} to {q2, p2}")
    
    if UseCos:
        absDiff = WBA_tools._absdiff_N2N_multi(N, np.cos(2*np.pi*q))
        title += r" using $\cos(2\pi q)$"
    else:
        absDiff = WBA_tools._absdiff_N2N_multi(N, p)
        
    absDiff[absDiff < 1e-16] = 1e-16
    absDiffLog = -np.log10(absDiff)
    if ReturnValues:
        return absDiffLog, q0, p0
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(title, fontsize=14)
    if ShowHist:
        ax.hist(absDiffLog, bins=50, color='k', lw=1, 
                histtype='step', density=True)
        
    else:
        t = np.linspace(0, 1, Nq, dtype=np.float64)
        ax.plot(t, absDiffLog, c='k', lw=1.5)
        
    plt.show()
    
    return absDiffLog, q0, p0


def freq_along_vector(initial, Nq=100, N=1000, K=0.0, 
                      CompareToNaff=0, thresh=1e-5, mapMode='decision',
                      minDiffBoundary=2.0, NaffCont=1, ShowDiff=0):
    [q1, p1, q2, p2] = initial
    q0 = np.linspace(q1, q2, Nq, dtype=np.float64)
    p0 = np.linspace(p1, p2, Nq, dtype=np.float64)
    freqWBA = WBA_core.WBA(q0, p0, N, K, thresh, mapMode=mapMode) % 1.0
        
    t = np.linspace(0, 1, Nq)
    fig, ax = plt.subplots(1, 1 + (CompareToNaff and ShowDiff), 
                           figsize=(15, 10))
    if type(ax) != np.ndarray:
        ax = np.array([ax], dtype=object)
    
    ax[0].set_title(f"Frequency along vector from {q1,p1} to {q2,p2}\n" +
                    f"with K={K}, N={N}", fontsize=14)
    ax[0].set_ylabel("Frequency", fontsize=14)
    
    if CompareToNaff:
        freqNaff = WBA_tools._Naff_multi(q0, p0, N=N, K=K, 
                                         Continuous=NaffCont)
        absDiff = np.abs(freqWBA - freqNaff)
        freqNaff2 = 1 - freqNaff
        absDiff2 = np.abs(freqWBA - freqNaff2)
        indx = (absDiff > minDiffBoundary * absDiff2)
        freqNaff[indx] = freqNaff2[indx]
        absDiff[indx] = absDiff2[indx]
            
        ax[0].plot(t, freqNaff, ls='', marker='o', 
                   ms=2, mew=1, c='r', label='Naff')
            
        if ShowDiff:
            ax[1].plot(t, absDiff, c='k', ls='', marker='o', ms=2, mew=1)
            ax[1].set_ylabel("Frequency difference", fontsize=14)
            ax[1].set_yscale('log')
            ax[1].set_title("Absolute difference in frequency\n" +
                            "between WBA and Naff", fontsize=14)
        
    ax[0].plot(t, freqWBA, ls='', marker='o', 
               ms=2, mew=1, c='b', label='WBA')
    ax[0].legend(fontsize=12)
    plt.show()
    return

def freq_along_vector_grid(initial, Nq, Nrows, Ncols, K=0.0, N=1000, 
                           thresh=1e-5):
    [q1, p1, q2, p2] = initial
    q0 = np.linspace(q1, q2, Nq, dtype=np.float64)
    p0 = np.linspace(p1, p2, Nq, dtype=np.float64)
    
    fig, ax = plt.subplots(Nrows, 2*Ncols, figsize=(15, 10))
    ax = ax.flatten()
    ax1 = ax[::2]
    ax2 = ax[1::2]
    t = np.linspace(0, 1, Nq)
    
    plt.suptitle(f"Frequency along vector from {q1,p1} to {q2,p2}\n" +
                 f"for $K\in{K}$, N={N}", fontsize=14)
    for i in range(0, Nrows*Ncols, Ncols):
        ax1[i].set_ylabel("Frequency", fontsize=14)
        
    for i, axs in enumerate(ax1):
        freqWBA = WBA_core.WBA(q0, p0, N, K[i], thresh, mapMode='decision')
        freqNaff = WBA_tools._Naff_multi(q0, p0, N=N, K=K[i])
        absDiff = np.abs(freqWBA - freqNaff)
        freqNaff2 = 1 - freqNaff
        absDiff2 = np.abs(freqWBA - freqNaff2)
        if np.mean(absDiff) > np.mean(absDiff2):
            freqNaff = freqNaff2
            absDiff = absDiff2
            
        axs.plot(t, freqWBA, lw=1.5, c='b', label='WBA')
        axs.plot(t, freqNaff, lw=1.4, c='r', label='Naff')
        axs.set_title(f"K={K[i]}")
        axs.legend()
        ax2[i].plot(t, absDiff, lw=1.5, c='k')
        ax2[i].set_yscale('log')
        
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    """
    f1, f2, ampl = 0.4954, 0.668, any  ->  freq not found, interesting shape
    explanation for wrapped orbits in p direction after arctan2:
        we need the phi-differences mod 1.0 to perform WBA for frequency
        However, the maximum of this array will approch 1.0 for certain 
        f and f2 combinations. Then, the phiDiff wraps around to 0.0 and 
        results in a wrong result. 
    """
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    PATHDATA = PATH_TP + "CP_Bachelor\\WBA_Python\\DataFiles\\"
    """
    https://stackoverflow.com/questions/9266150/matplotlib-generating-
    vector-plot  --> vector images, maybe alternative to pyxgraph later.
    plt.savefig(PATH + "ChaosMap1000_200x200_K01", .eps, format='...',
                dpi=100...300)
    plt.savefig(PATH + "name", dpi=150)
    np.savetxt(PATHDATA)
    
    """
    
    q0, p0, K = 0.2, 0.5, 0.92
    Npoints = 30
    phi = (1 + np.sqrt(5)) / 2
    # freq_along_vector_grid([0.5, 0.01, 0.5, 0.5], 300, 3, 2, 
    #                         K=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], N=1000)
    # freq_along_vector_grid([0.5, 0.01, 0.5, 0.5], 300, 3, 2, 
    #                         K=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1], N=1000)
    #### freq_along_vector([0.1,0.4,0.1,0.45],1000,1000,0.7,1)
    # art_signal_interactive(2 - phi, np.sqrt(2) - 1, 0.5, 
    #                         NN=200, Nmin=5.0, Nmax=14.0,
    #                         _dig=16, CompareToNaff=1, 
    #                         ConfineSignal=1)
    # freq_along_vector([0.76, 0.25, 0.76, 0.3], 1000, 
    #                   2**10, 0.9, 1, 1.0, 'none')
    # freq_along_vector([0.71, 0.25, 0.71, 0.3], 1000, 
    #                   2**10, 0.9, 1, 1.0, 'none')
    #different map: q -> 1-q
    chaos_along_vector([0.5,0.001,0.5,0.2], 1000, 2**10, 0.5)
    # chaos_along_vector([0.321,-0.5,0.321,0.5],1000,10000,-1.0,1,0,1)
    # freq_along_vector([1-0.2, 0.43, 1-0.2, 0.47], 
    #                   1000, 2**10, 0.7, 1, 1.0, 'none')
    # freq_along_vector([1-0.224, 0.43, 1-0.224, 0.47], 
    #                   1000, 2**10, 0.7, 1, 1.0, 'none')
    # freq_along_vector([0.9, 0.18, 0.9, 0.25], 
    #                   1000, 2**10, 1.0, 1, 1.0, 'none')
    # freq_along_vector([0.9, 0.18, 0.9, 0.25], 
    #                   1000, 2**12, 1.0, 1, 1.0, 'none')

    # absDiff, q0, p0 = chaos_along_vector([0.321, -0.5, 0.321, 0.5], 1000, 
    #                                       10000, -1.0, ShowHist=1, UseCos=1)
    # art_signal_grid(Nf=100, Nf2=100, ampl=0.5, N=1024,
    #                 UseNaff=0)
    # art_signal_grid(Nf=100, Nf2=100, ampl=0.5, N=1024,
    #                 UseNaff=1)
    # art_signal_grid(Nf=100, Nf2=100, ampl=0.1, N=1024, step=0.01)
    # art_signal_grid(Nf=100, Nf2=100, ampl=0.1, N=1024, step=0.01, UseNaff=1)
    # data = chaos_indicator_grid(Nq=400, N=2**12, K=0.9, UseNaff=1,
    #                             ReturnValues=1, UseImshow=0) 
    # val = data[0]
    # chaos_indicator_grid(Nq=500, N=10**4, K=0.1) # RUNTIMEWARNING (minutes)
    # freq_maps(ShowWBA=0, ShowNaff=0, ShowDiff=1, 
    #           p0count=4000, N=4096, K=0.9)
        