# -*- coding: utf-8 -*-
"""
Plotting for the WBA thesis.
"""

import WBA_core, WBA_tools, WBA_2D_tests, WBA_4D_tests
from std_map import _std_map, Mapping4dCyl
from explorator.comp.naff_call import naff_4d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
rcParams["figure.dpi"] = 100

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

global fs, lfs, lw, lwnaff, ms, mew, _dig, tls, phiGold, phiSilver
global phiGoldLabel, phiSilverLabel, alphaNaff, colors, labels, fscale
# global rotation, yxOffset, yyOffset, xxOffset, xyOffset
fs = 21     # fontsize (titles and labels)
lfs = 16    # legend fontsize
lw = 1.0
lwnaff = 1.3  # lw for naff
ms = 4
mew = 1     
_dig = 16   # digits for nu
tls = 18    # tick label size
alphaNaff = 0.6
phiGold = (np.sqrt(5) - 1) / 2
phiSilver = np.sqrt(2) - 1
phiGoldLabel = "\\varphi_\\mathrm{golden}"
phiSilverLabel = "\sqrt{2} - 1"
labels = [phiGoldLabel, phiSilverLabel, "\\frac{4}{5}"]
colors = 10*['blue', 'darkorange', 'green', 'darkred', 
             'cyan', 'orangered', 'purple', 'lime']
fscale = 1.0

class Colors():
    def __init__(self, colors=None, ctr=0):
        self.ctr = ctr
        if type(colors) == type(None):
            self.colors = list(plt.get_cmap("tab10").colors)
        else:
            self.colors = colors
        self.clength = len(self.colors)
    def get_color(self):
        cval = self.colors[self.ctr % self.clength]
        self.ctr += 1
        return cval
    
cColors = Colors(colors)

def set_xticks_linear(ax, vmin, vmax, numticks, decimals=7):
    # xlim = ax.get_xlim()
    xticks = np.round(np.linspace(vmin, vmax, numticks), decimals)
    #xloc = (xticks - xlim[0]) / (xlim[1] - xlim[0])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    pass

def set_yticks_linear(ax, vmin, vmax, numticks, decimals=7):
    yticks = np.round(np.linspace(vmin, vmax, numticks), decimals)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    pass

def xticks_in_limits(xticks, xlimits):
    """Subroutine for 'embed_labels'. Returns subset of ticks inside limits"""
    new_ticks = []
    for tick in xticks:
        tick_we = tick.get_window_extent()
        if tick_we.x1 > xlimits[0] and tick_we.x0 < xlimits[1]:
            new_ticks.append(tick)
    return new_ticks

def yticks_in_limits(yticks, ylimits):
    """Subroutine for 'embed_labels'. Returns subset of ticks inside limits"""
    new_ticks = []
    for tick in yticks:
        tick_we = tick.get_window_extent()
        if tick_we.y1 > ylimits[0] and tick_we.y0 < ylimits[1]:
            new_ticks.append(tick)
    return new_ticks

class AlphabeticalLabels():
    def __init__(self, abc_labels=None, ctr=0):
        if type(abc_labels) == type(None):
            self.abc_labels = [r"(a)", r"(b)", r"(c)", r"(d)", 
                               r"(e)", r"(f)", r"(g)", r"(h)"]
        else:
            self.abc_labels = abc_labels
        self.ctr = ctr
        self.length = len(self.abc_labels)
    
    def get_label(self):
        next_label = self.abc_labels[self.ctr % self.length]
        self.ctr += 1
        return next_label
    
def embed_labels(axes, SetCaptions=True, order=None, labelAxis=None,
                 fontsize=None, alignment=None):
    if type(fontsize) == type(None):
        fontsize=fs
    if type(alignment) == type(None):
        xva, yha = 'center', 'center'
    else:
        xva, yha = alignment
        
    try:
        length = len(axes)
    except TypeError:
        axes = [axes]
        length = len(axes)
    try:
        if SetCaptions == True:
            SetCaptions = [1] * length
        if SetCaptions == False:
            SetCaptions = [0] * length
    except:
        assert len(SetCaptions) == length
    
    Labels = AlphabeticalLabels()
    if type(order) != type(None):
        axes = [axes[i] for i in order]
        
    if type(labelAxis) == type(None):
        labelAxis = ['both'] * length
    else:
        assert len(labelAxis) == length
        
    for i, axis in enumerate(axes):
        ax0 = axis.get_window_extent().x0
        ay0 = axis.get_window_extent().y0
        ax1 = axis.get_window_extent().x1
        ay1 = axis.get_window_extent().y1
        # print()
        # print(ax0, ax1, ay0, ay1)
        width = ax1 - ax0
        height = ay1 - ay0
        xticks = xticks_in_limits(axis.get_xticklabels(), [ax0, ax1])
        # print([tick.get_window_extent() for tick in xticks])
        if len(xticks) > 1 and (labelAxis[i] in ['both', 'x']):
            xxpos = ((xticks[-1].get_window_extent().x0 
                      + xticks[-2].get_window_extent().x1)/2 - ax0) / width
            xypos = (xticks[-1].get_window_extent().y0 - ay0) / height
            if xva == 'bottom':
                xypos = (((xticks[-1].get_window_extent().y0 
                          + xticks[-1].get_window_extent().y1)/2 - ay0) 
                         / height )
                # xypos = (xticks[-1].get_window_extent().y1 - ay0) / height
                xva = 'center'
            xlabel = axis.get_xlabel()
            axis.set_xlabel(xlabel, fontsize=fontsize, rotation=0, 
                            va=xva, ha='center')
            axis.xaxis.set_label_coords(xxpos, xypos)
        else:
            SetCaptions[i] = 0
        
        yticks = yticks_in_limits(axis.get_yticklabels(), [ay0, ay1])
        if len(yticks) > 1 and (labelAxis[i] in ['both', 'y']):
            yypos = ((yticks[-1].get_window_extent().y0 
                      + yticks[-2].get_window_extent().y1)/2 - ay0) / height 
            yxpos = (yticks[-1].get_window_extent().x0 - ax0) / width 
            if yha == 'left':
                yxpos += (yticks[-1].get_window_extent().x1
                          - yticks[-1].get_window_extent().x0) / (2*width) 
                yha = 'center'
            ylabel = axis.get_ylabel()
            axis.set_ylabel(ylabel, fontsize=fontsize, rotation=0, 
                            ha=yha, va='center')
            axis.yaxis.set_label_coords(yxpos, yypos)
        # print(xxpos, xypos, yxpos, yypos)
        if SetCaptions[i] != 0:
            # print(labels[i], xypos)
            # ydiff = (xticks[-1].get_window_extent().y1 
            #          - xticks[-1].get_window_extent().y0) / height
            # print(ydiff)
            va = 'top'
            if SetCaptions[i] == 2: va = 'bottom'
            # print(xypos, ydiff, xypos - ydiff)
            # # axis.text(0.5, 0.5, labels[i], c='cyan', fontsize=fs)
            # axis.text(0.5, xypos - ydiff, labels[i], fontsize=fs,
            #           ha='center', va='top', transform=axis.transAxes)
            axis.text(0.5, xypos, Labels.get_label(), fontsize=fontsize,
                      ha='center', va=va, transform=axis.transAxes)
    pass

###############################################################################
# Artificial signals
###############################################################################

def simple_periodic_plot(q0=0.1, N=100, init=None, OnlySignal=0):
    if type(init) == type(None):
        # init = np.array([(np.sqrt(n**2 + 4) - n) / 2 for n in range(1, 4)])
        # init[0] = 1/3
        # init[2] = 4/5
        init = [phiGold, phiSilver, 0.8]
    
    def mapping(nu=1/3, q0=0.0, N=100):
        return (q0 + np.arange(N) * nu) % 1.0
    
    Nmin, Nmax = 1.5, 14.0
    Narr = WBA_tools.N_arr(Nmin, Nmax, 40)
        
    xscale, yscale = 16*fscale, 9*fscale
    if OnlySignal: xscale /= 2#; yscale /= 1.2
    fig, ax = plt.subplots(1, 2 - OnlySignal, figsize=(xscale, yscale))
    if OnlySignal: ax = [ax]
    ax[0].set_xlabel(r"$q_n$", fontsize=fs)
    ax[0].set_ylabel(r"$\nu$", fontsize=fs)
    ax[0].axis([0.0, 1.0, 0.0, 1.0])
    if not OnlySignal:
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].axis([2**Nmin, 2**Nmax, 0.5*1e-16, 1e-13])
        ax[1].set_xlabel(r"$N$", fontsize=fs)
        ax[1].set_ylabel(r"$|\nu-\nu_{N}|$", fontsize=fs)
    for axis in ax:
        axis.tick_params(labelsize=tls)
        
    for i, nu in enumerate(init):
        qNu = mapping(nu, q0=q0, N=Narr[-1])
        pNu = np.full(N, nu)
        label = f"$\\nu={labels[i]}$"
        ax[0].plot(qNu[:N], pNu[:N], ls='', 
                   marker="o", ms=ms, mew=mew, c=colors[i], 
                   label=label)
        nuWBA = WBA_core._WBA(Narr, (qNu[1:] - qNu[:-1]) % 1.0)
        nuNaff = WBA_tools._Naff(Narr, qNu, pNu, 1)
        diffWBA = np.abs(nu - nuWBA)
        diffNaff = np.abs(nu - nuNaff)
        diffWBA[diffWBA < 1e-16] = 1e-16
        diffNaff[diffNaff < 1e-16] = 1e-16
        
        if not OnlySignal:
            ax[1].plot(Narr, diffWBA, ls='-', lw=lw, c=colors[i],
                              marker='o', ms=ms, mew=mew, 
                              label=label)
            ax[1].plot(Narr, diffNaff, ls='--', lw=lwnaff, alpha=alphaNaff,
                       #marker='o', ms=ms, mew=mew,
                       c=colors[i])
    if OnlySignal:
        ax[0].legend(numpoints=3, fontsize=lfs)
    else:
        ax[1].legend(fontsize=lfs)
    fig.tight_layout()
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=(not OnlySignal))
    return fig, ax

def plot_art_signal(freq=None, freq2=0.5, ampl=0.0, Nmin=5.0, Nmax=14.0,
                    NN=200, _dig=16, step=None, qOffset=0.0, pOffset=0.0,
                    CompareToNaff=1, qFactor=1.0, pFactor=1.0, rad=None, 
                    ConfineSignal=1, thresh=2.0, Nplot=1000, SetYlimits=0,
                    OnlySignal=0):
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
    if type(freq) == type(None):
        freq = np.array([(np.sqrt(n**2 + 4) - n) / 2 for n in range(1, 4)])
        freq[2] = 4/5
    if type(rad) == type(None):
        rad = [1.0, 0.7, 1.3]
        
    Narr = WBA_tools.N_arr(Nmin, Nmax, NN)    # different signal lengths
    N2f = np.max(Narr)                  # maximal signal length

    # fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    if OnlySignal: 
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 9))
        ax = [None, ax1, ax2]
    else:
        fig = plt.figure(figsize=(16*fscale, 10*fscale))
        ax0 = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2)
        ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=2, rowspan=1)      
        ax2 = plt.subplot2grid((2, 5), (1, 0), colspan=2, rowspan=1)
        
    
        ax = [ax0, ax1, ax2]
    
    if not OnlySignal:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        if SetYlimits:
            ax[0].axis([2**Nmin, 2**Nmax, 0.5*1e-16, 1e-13])
        else:
            ax[0].set_xlim((2**Nmin, 2**Nmax))
        ax[0].set_xlabel(r"$N$", fontsize=fs)
        ax[0].set_ylabel(r"$|\nu-\nu_{N}|$", fontsize=fs)
    ax[1].set_xlabel(r"$q_n$", fontsize=fs)
    ax[1].set_ylabel(r"$p_n$", fontsize=fs)
    ax[2].set_xlabel(r"$\phi_n$", fontsize=fs)
    ax[2].set_ylabel(r"$r_n$", fontsize=fs)
    if ConfineSignal:
        ax[1].axis([-2 + qOffset, 2 + qOffset + OnlySignal/2, 
                    -2 + pOffset + OnlySignal/2, 
                    2 + pOffset - OnlySignal/2])
        ax[2].set_xlim(-0.5, 0.5)
        
    for i in range(len(freq)):
        q, p = WBA_2D_tests._2f_signal(freq[i], freq2, ampl, N2f, qOffset, 
                                       pOffset, qFactor, pFactor, rad=rad[i])
        if not OnlySignal:
            WBA_tools.compare_conv(ax[0], q, p, Narr, _dig, freq[i], 
                                   ShowLegend=1, 
                                   thresh=thresh, MapToCircle=0, c=colors[i], 
                                   mapMode='arctan2', lw=lw, SetTitle=0,
                                   freqLabel=labels[i], 
                                   UseMarkers=['o', ms, mew],
                                   alphaNaff=alphaNaff)
            label = None
        else: label = f"$\\nu={labels[i]}$"
        ax[1].plot(q[:Nplot], p[:Nplot], ls='', 
                   marker='o', ms=3, mew=1, c=colors[i], label=label)
        r, phi = np.sqrt(q*q + p*p), np.arctan2(q, p) / (2*np.pi)
        ax[2].plot(phi[:Nplot], r[:Nplot], ls='', 
                   marker='o', ms=3, mew=1, c=colors[i])
    if not OnlySignal: ax[0].legend(fontsize=lfs)
    else: ax[1].legend(numpoints=3, fontsize=lfs)
    ax[1].locator_params(nbins=5)
    ax[2].locator_params(nbins=5)
    for axis in ax[OnlySignal:]:
        axis.tick_params(labelsize=tls)
    fig.tight_layout()
    fig.canvas.draw()
    if OnlySignal: embed_labels(ax[OnlySignal:], SetCaptions=False)
    else: embed_labels(ax, order=[1, 0, 2])
    return fig, ax

def _debug_art_signal(nu1=np.sqrt(2)-1, nu2=1/8, ampl=0.5, N=2**11,  
                      alpha=0.0, ModAlpha=0, UseSolution=0):
    q, p = WBA_2D_tests._2f_signal(nu1, nu2, ampl=ampl, N=N, rad=1.0)
    phi = np.arctan2(q, p) / (2*np.pi)
    phiDiff = phi[1:] - phi[:-1]
    if ModAlpha:
        phiDiff = (phiDiff + alpha) % 1.0 - alpha
        if UseSolution:
            if np.any(phiDiff > 0.75-alpha) and np.any(phiDiff < 0.25-alpha):
                phiDiff[phiDiff < 0.5 - alpha] += 1.0
    print(WBA_core.WBA(phi[1:], phiDiff))
    return phi, phiDiff

def plot_debug_art_signal(nu1=np.sqrt(2)-1, nu2=1/8, ampl=0.5, N=2**11,  
                          alpha=0.0, ModAlpha=0, UseSolution=0):
    fig, ax = plt.subplots(1, 1, figsize=(16*fscale, 9*fscale))
    ax.set_xlabel(r"$\phi_n$", fontsize=fs)
    ax.set_ylabel(r"$\Delta\phi_n$", fontsize=fs+4)
    phi1, phiDiff1 = _debug_art_signal(nu1, nu2, ampl, N, 
                                       alpha, ModAlpha, UseSolution)
    phi2, phiDiff2 = _debug_art_signal(nu2, nu1, ampl, N, 
                                       alpha, ModAlpha, UseSolution)
    ax.set_xlim(-0.5, 0.5)
    if ModAlpha:
        ax.set_ylim(- alpha, 1.0 - alpha)
    ax.plot(phi1[1:], phiDiff1, ms=ms, mew=mew, ls='', marker='o',
               c=colors[0])
    ax.plot(phi2[1:], phiDiff2, ms=ms, mew=mew, ls='', marker='o',
               c=colors[1])
    ax.tick_params(labelsize=tls+3)
    ax.locator_params(axis='y', nbins=6)
    fig.tight_layout()
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=0, fontsize=fs+4)
    return fig

def _plot_art_signal_grid(fig, ax, ctr, ctr2, Nf=300, Nf2=300, 
                          ampl=0.5, N=1024, alpha=None,
                          step=None, UseNaff=0, PrintAvgLogDiff=1, RetVal=0, 
                          absDiffGrid=None, cNorm=None, SetTitle=1):
    phi3 = phiGold / 0.6
    if step != None:
        phi3 = step * Nf
        
    max_f = 1.0
        
    freq = np.arange(phi3 / (2*Nf), max_f, phi3 / Nf)
    freq2 = np.arange(phi3 / (2*Nf2), max_f, phi3 / Nf2)
    # title = f"$N={N}$ and $A={ampl}$" 
    title = f"$N={N}$" 
    if UseNaff:
        title += " using Naff"
    else:
        title += " using WBA"
    if type(alpha) != type(None):
        title = f"$\\alpha={float(alpha)}$"
    if type(absDiffGrid) == type(None):
        if UseNaff:
            freqGrid = WBA_tools._2f_grid_Naff(freq, freq2, ampl, N).T
        else:
            freqGrid = WBA_tools._2f_grid(freq, freq2, ampl, N).T
        
        freqTrue = np.outer(freq, np.ones(len(freq2))).T
        # print(freqGrid, freq)
        indx = (np.abs(freqGrid - freqTrue) > np.abs(1 - freqGrid - freqTrue))
        freqGrid[indx] = 1 - freqGrid[indx]
        # print(freqGrid)
        
        absDiffGrid = np.abs(freqGrid - freqTrue)
        absDiffGrid[absDiffGrid < 1e-16] = 1e-16
        
    if PrintAvgLogDiff:
        absDiffLog = -np.log10(absDiffGrid)
        print("Average digits of precision are ", np.mean(absDiffLog))
    if RetVal:
        return absDiffGrid
    # fig, ax = plt.subplots(figsize=(11, 10))
    # plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.08)
    ax.locator_params(nbins=5)
    ax.tick_params(labelsize=tls)
    ax.set_xlabel(r"$\nu_1$", fontsize=fs)
    ax.set_ylabel(r"$\nu_2$", fontsize=fs)
    if SetTitle: ax.set_title(title, fontsize=fs)
    if type(cNorm) == type(None):
        cNorm = plt.cm.colors.LogNorm(vmin=np.min(absDiffGrid), 
                                      vmax=np.max(absDiffGrid))
    
    # img = WBA_tools.imshow_grid(ax, freq, freq2, absDiffGrid, 
    #                             cmap='viridis', norm=cNorm)
    img = ax.imshow(absDiffGrid, cmap='viridis', norm=cNorm,
                    extent=[np.min(freq), np.max(freq), 
                            np.min(freq2), np.max(freq2)],
                    origin='lower')
    ax_pos = ax.get_position()
    if ctr2:
        cbar_axim = fig.add_axes([ax_pos.x1 + 0.015, ax_pos.y0 + 0.04,
                                  0.02, (ax_pos.y1 - ax_pos.y0)*0.8])
        cbar = fig.colorbar(img, cax=cbar_axim,) 
        cbar.ax.tick_params(labelsize=tls)
    return fig

def plot_art_signal_grid(Alpha=0):
    ampl = 0.5; N = [10, 12]; Np = 300; Nq = 300
    mode = ["WBA",  "Naff"]
    if Alpha:
        alpha = [["000", "015"], ["035", "050"]]
        alphaFloat = [[0.0, 0.15], [0.35, 0.5]]
    else:
        alphaFloat = [[None, None], [None, None]]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 10))
    plt.subplots_adjust(left=0.045, right=0.92, top=0.965, 
                        bottom=0.075, hspace=0.12, wspace=0.0)
    for i in range(2):
        ax[i, 0].set_ylabel(r"$\nu_2$", fontsize=fs)
        
        for j in range(2):
            ax[1, j].set_xlabel(r"$\nu_1$", fontsize=fs)
            ax[i, j].locator_params(nbins=5)
            if Alpha:
                fname = f"OscPeriodic2fA05GridAlpha{alpha[i][j]}WBA"
            else:
                fname = f"OscPeriodic2fA05GridN{N[i]}" + mode[j]
            absDiffGrid = np.loadtxt(PATHDATA + fname + ".gz")
            if j == 0:
                cNorm = plt.cm.colors.LogNorm(vmin=np.min(absDiffGrid), 
                                              vmax=np.max(absDiffGrid))
            _plot_art_signal_grid(fig, ax[i, j], i, j, Np, Nq, ampl, 2**N[i], 
                                  absDiffGrid=absDiffGrid, 
                                  SetTitle=1,#(i or Alpha), 
                                  alpha=alphaFloat[i][j], 
                                  UseNaff=(j and not Alpha), cNorm=cNorm)
    axes = ax.flatten()
    fig.canvas.draw()
    embed_labels(axes, SetCaptions=False)
    for i in range(2):
        ax[i, 1].set_yticklabels([])
        ax[i, 1].get_yaxis().set_visible(False)
        for j in range(2):
            ax[0, j].set_xticklabels([])
            ax[0, j].get_xaxis().set_visible(False)
    return fig

###############################################################################
# 2D standard map
###############################################################################
    
def plot_2d_orbits(K=0.7, Npoints=1024, Nmin=5.0, Nmax=14.0, NN=100, 
                   init=None, thresh=1e-5, NmaxLimit=None, 
                   mapMode='none', MapToCircle=0, NaffLimit=1):
    if type(init) == type(None):
        # q0 = 0.1     # 1-x for changed map
        # p0golden = WBA_tools.search_orbit_from_freq(1.5-np.sqrt(1.25), 
        #                                             K, q0=q0)[1]
        # q0ratio, p0ratio = WBA_tools.search_orbit_from_freq(0.8, K, q0=q0)
        # p0 = [p0ratio, -0.19, p0golden]  
        # init = [[q, p] for q, p in zip([q0ratio, q0, q0], p0)]
        if mapMode == 'none':
            init = [WBA_tools.search_orbit_from_freq(nu, K)
                    for nu in [phiGold, phiSilver, 0.8]]
        if mapMode == 'arctan2':
            init = [[0.6, p0] for p0 in [-0.1, -0.17, -0.231] ]
        
    # WBA_tools.interactive_plot(K, Npoints, Nmin, Nmax, NN, init, thresh,
    #                            NmaxLimit, mapMode, MapToCircle, NaffLimit)
    
    Narr = WBA_tools.N_arr(Nmin, Nmax, NN)
    if NmaxLimit != None:
        Narr = np.append(Narr, np.uint32(2**NmaxLimit))
    
    _dict = {0: {'xlabel' : r'$q_n$', 'ylabel' : r'$p_n$', 
                  'title' : f"$K={K}$ and {Npoints} points"},
              1: {'xlabel' : '$N$', 
                  'ylabel' : r"$|\nu - \nu_{N}|$",#\nu_{N_\mathrm{max}} 
                  'title' : ""}}
    
    fig, ax = plt.subplots(1, 2, figsize=(16*fscale, 9*fscale))
    ax[0].axis([0, 1, -0.5, 0.5])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
        
    for i, z0 in enumerate(init):
        q0arr, p0arr = _std_map(*z0, Npoints, K)
        ax[0].plot(q0arr, (p0arr + 0.5) % 1.0 - 0.5, 
                   marker='o', ls='', ms=2, mew=1, c=colors[i])
        
        q, p = _std_map(*z0, Narr[-1], K)
        WBA_tools.compare_conv(ax[1], q - 0.5, p, Narr, _dig, ShowLegend=1, 
                               thresh=thresh, MapToCircle=MapToCircle,
                               c=colors[i], lfs=lfs-2, 
                               mapMode=mapMode, lw=lw, SetTitle=0,
                               UseMarkers=['o', ms, mew],
                               alphaNaff=alphaNaff, NaffLimit=NaffLimit,
                               AssertNaffEqualWBA=1)
    ax[1].set_xlim(2**Nmin, 2**Nmax)
    for i in range(len(ax)):
        ax[i].set_title(_dict[i]['title'], fontsize=fs)
        ax[i].set_xlabel(_dict[i]['xlabel'], fontsize=fs)
        ax[i].set_ylabel(_dict[i]['ylabel'], fontsize=fs)
        ax[i].tick_params(labelsize=tls)
    fig.tight_layout()
    fig.canvas.draw()
    embed_labels(ax)
    return fig, ax

def _inset_standard_map(ax, init, K, N, insetAxis, 
                        insetAxisExtent=None, Norbits=20):
    axisInset = ax.inset_axes(insetAxis)  # x0, y0, width, height
    # q0 = np.linspace(init[0], init[2], Norbits)
    # p0 = np.linspace(init[1], init[3], Norbits)
    axisInset.tick_params(labelsize=10)
    initRegular = np.loadtxt(PATHDATA + "InitsFor2DMapAxisInsetRegular.gz")
    initChaos = np.loadtxt(PATHDATA + "InitsFor2DMapAxisInsetChaos.gz")
    # for i in range(Norbits):
    #     q, p = _std_map(q0[i], p0[i], N, K)
    #     axisInset.plot(q, p, ls='', marker='o', ms=1, mew=0.5, c=colors[i])
    for i in range(len(initRegular[0, :])):
        q, p = _std_map(initRegular[0, i], initRegular[1, i], N, K)
        axisInset.plot(q, p, ls='', marker='o', ms=2, mew=0.0, c=colors[i])
    for i in range(len(initChaos[0, :])):
        q, p = _std_map(initChaos[0, i], initChaos[1, i], N, K)
        axisInset.plot(q, p, ls='', marker='o', ms=1, mew=0.0, c='k', 
                       alpha=0.5)
    if type(insetAxisExtent) == type(None):
        axisInset.set_xticklabels([])
        axisInset.set_yticklabels([])
        axisInset.get_xaxis().set_visible(False)
        axisInset.get_yaxis().set_visible(False)
    else:
        axisInset.axis(insetAxisExtent)
        axisInset.locator_params(nbins=4)
    return axisInset

def plot_freq_along_vector(init=None, K=0.9, N=2**12, Nt=500):
    if type(init) == type(None):
        init1 = [0.71, 0.25, 0.71, 0.3]
        init2 = [0.76, 0.25, 0.76, 0.3]
        init = [init1, init2]
        # x0, y0, width, height in relative coordinates to axes
        insetAxes = [0.075, 0.5, 0.4, 0.45]
        insetExtent = [0.67, 0.8, 0.25, 0.3]#0.23, 0.32]
        
    fig, ax = plt.subplots(2, len(init), 
                           figsize=(15*fscale*16/15, 10*fscale*16/15))
    for i in range(len(ax)):
        print("Frequency along vector ", init[i])
        WBA_2D_tests.freq_along_vector(
            init[i], ax=ax.T[i], Nq=Nt, N=N, K=K, CompareToNaff=1, 
            thresh=1e-5, mapMode='none', minDiffBoundary=2.0, NaffCont=1, 
            ShowDiff=1, fs=fs, tls=tls, lfs=lfs, Show=0, SetTitle=0, 
            SetYlabel=(i in [0, 1]), loc='lower right')
    
    ax = ax.flatten()
    try:
        axisInset = _inset_standard_map(ax[0], init[0], K, N, insetAxes,
                                        insetAxisExtent=insetExtent)
    except NameError or IndexError:
        print("Can't inset axis without axis parameters")
    ax[3].set_ylabel("")
    for j in [0, 1]:    
        ax[j].set_ylim(0.28, 0.37)
        ax[j].locator_params(nbins=5)
    fig.canvas.draw()
    embed_labels(ax, labelAxis=['both', 'both', 'x', 'x'], 
                 SetCaptions=[2, 2, 1, 1])
    axisInset.set_xlabel(r"$q$", fontsize=12)
    axisInset.set_ylabel(r"$p$", fontsize=12)
    embed_labels(axisInset, SetCaptions=False, fontsize=12,
                 alignment=['bottom', 'left'])
    # for qval in [0.71, 0.76]:
    #     axisInset.axvline(qval, ls='--', c='k', alpha=0.5, lw=0.8)
    for j in [0, 1]:
        ax[j].set_xticklabels([])
        ax[j].get_xaxis().set_visible(False)
    fig.tight_layout()
    return fig

def plot_freq_along_vector_v2(K=0.9, N=2**12, Nt=1000, RemoveChaos=0):
    init = [0.71, 0.25, 0.71, 0.3]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    q0 = np.random.uniform(0.67, 0.75, size=400)
    p0 = np.random.uniform(0.25, 0.3, size=200)
    p0 = np.append(p0, np.linspace(0.28, 0.3, 200))
    
    cColors.ctr = 0
    for i in range(len(q0)):
        cval = cColors.get_color()
        q, p = _std_map(q0[i], p0[i], N, K)
        nu1 = WBA_core._WBA_single(p[:N//2])
        nu2 = WBA_core._WBA_single(p[N//2:])
        if np.abs(nu1 - nu2) > 1e-7: 
            ax[1].plot(q[:128], p[:128], ls='', marker='o', 
                    ms=1.5, mew=0, c='k', alpha=0.5)
        else:
            ax[1].plot(q, p, ls='', marker='o', 
                    ms=2, mew=0, c=cval, alpha=1.0)
    ax[1].axis([0.67, 0.75, 0.25, 0.3])
    
    print("Frequency along vector ", init)
    WBA_2D_tests.freq_along_vector(
        init, ax=ax[0], Nq=Nt, N=N, K=K, CompareToNaff=1, 
        thresh=1e-5, mapMode='none', minDiffBoundary=2.0, NaffCont=1, 
        ShowDiff=0, fs=fs, tls=tls, lfs=lfs, Show=0, SetTitle=0, 
        SetYlabel=1, RemoveChaos=RemoveChaos)
        
    ax[0].set_ylim(0.28, 0.37)
    ax[0].locator_params(nbins=5)
    ax[1].locator_params(nbins=5)
    ax[0].set_xlabel(r"$t$", fontsize=fs)
    ax[1].set_xlabel(r"$q_n$", fontsize=fs)
    ax[1].set_ylabel(r"$p_n$", fontsize=fs)
    for axis in ax:
        axis.tick_params(labelsize=tls)
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=False, fontsize=fs)
    ax[1].axvline(0.71, ls='--', c='k', alpha=0.8, lw=0.8)
    fig.tight_layout()
    return fig, ax

def plot_2d_std_map(count, K=0.9, N=2**12, q0min=0.0, q0max=1.0, 
                    p0min=-0.5, p0max=0.5):
    q0 = np.random.uniform(q0min, q0max, size=count)
    p0 = np.random.uniform(p0min, p0max, size=count)
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))
    ax.locator_params(nbins=5)
    ax.set_xlabel(r"$q_n$", fontsize=fs)
    ax.set_ylabel(r"$p_n$", fontsize=fs)
    ax.axis([q0min, q0max, p0min, p0max])
    cColors.ctr = 0
    for i in range(len(q0)):
        cval = cColors.get_color()
        q, p = _std_map(q0[i], p0[i], N, K)
        nu1 = WBA_core._WBA_single(p[:N//2])
        nu2 = WBA_core._WBA_single(p[N//2:])
        if np.abs(nu1 - nu2) > 1e-7: 
            ax.plot(q[:128], p[:128], ls='', marker='o', 
                    ms=1.5, mew=0, c='k', alpha=0.5)
        else:
            ax.plot(q, p, ls='', marker='o', 
                    ms=1.5, mew=0, c=cval, alpha=1.0)
    ax.tick_params(labelsize=tls)
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=False, fontsize=fs)
    fig.tight_layout()
    return fig, ax

def _plot_chaos_maps(fig, ax, ctr, K=0.9, N=2**10, 
                     Np=50, Nq=None, mode='WBA', RetVal=0, 
                     absDiffGrid=None, SetTitle=1, UseImshow=1):
    assert mode in ['WBA', 'cos', 'Naff', 'NaffMTC']
    if Nq == None:
        Nq = Np
    mode_dict = {'WBA' : 'WBA with $p_n$', 
                 'cos' : 'WBA with $\cos(2\pi q_n)$',
                 'Naff' : 'Naff',
                 'NaffMTC' : 'Naff'}
    
    q0 = np.linspace(0.0, 1.0, Nq)
    p0 = np.linspace(-0.5, 0.5, Np)
    # title = (f"Orbit length {N} and K={K} on a ({Nq}$\\times ${Np}) grid")  
    title = (f"K={K} using {mode_dict[mode]}")   
    
    if type(absDiffGrid) == type(None):
        if mode == 'WBA' or mode == 'cos':
            _func = WBA_tools._grid_absdiff_N2N
            absDiffGrid = _func(q0, p0, N, K, mode == 'cos').T
        if mode == 'Naff' or mode == 'NaffMTC':
            _func = WBA_tools._grid_absdiff_N2N_Naff
            absDiffGrid = _func(q0, p0, N, K, mode == 'NaffMTC').T
        
        absDiffGrid[absDiffGrid < 1e-16] = 1e-16
        
    if RetVal:
        return absDiffGrid
    if UseImshow:
        # plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.08)
        vmin = np.min(absDiffGrid)
        vmax = np.max(absDiffGrid)
        print(mode, ctr, vmin, vmax)
        cNorm = plt.cm.colors.LogNorm(vmin=vmin, vmax=vmax)
        
        # img = WBA_tools.imshow_grid(ax, q0, p0, absDiffGrid, cmap='viridis',
        #                             norm=cNorm)
        img = ax.imshow(absDiffGrid, cmap='viridis', norm=cNorm,
                        extent=[np.min(q0), np.max(q0), 
                                np.min(p0), np.max(p0)],
                        origin='lower')
        # cbar_axim = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # cbar_axim = fig.add_axes([0.46, 0.175 + ctr*0.45, 0.02, 0.35])
        ax_pos = ax.get_position()
        cbar_axim = fig.add_axes([ax_pos.x1 + 0.015, ax_pos.y0 + 0.04,
                                  0.02, (ax_pos.y1 - ax_pos.y0)*0.8])
        cbar = fig.colorbar(img, cax=cbar_axim,) 
        cbar.ax.tick_params(labelsize=tls)
    else:
        title = "Distribution of the chaos indicator"
        # ax.set_xlabel(r"$\Delta\nu(N)$", fontsize=fs)
        # ax.set_ylabel(r"counts", fontsize=fs)
        ax.set_xscale('log')
        ax.get_yaxis().set_visible(False)
        bins = np.logspace(-16, 0, Nq // 4)
        ax.hist(absDiffGrid.flatten(), bins=bins, color='k', lw=lw, 
                histtype='step', label=f"$N={N}$")
        loc = 'upper left'
        if N > 4095:
            if mode == 'WBA': loc = 'upper right'
            if mode == 'cos': loc = 'upper center'
        ax.legend(fontsize=lfs, loc=loc)
        # plt.subplots_adjust(left=0.085, right=0.98, top=0.985, bottom=0.085)
    ax.tick_params(labelsize=tls)
    if SetTitle: ax.set_title(title, fontsize=fs)
    return ax

def plot_chaos_maps(mode='WBA'):
    K = 0.9; N = [10, 12]; Np = 500; Nq = None
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(left=0.035, right=0.985, top=0.965, 
                        bottom=0.085, hspace=0.125, wspace=0.165)
    ax[1, 0].set_xlabel(r"$q_n$", fontsize=fs)
    ax[1, 1].set_xlabel(r"$\Delta\nu(N)$", fontsize=fs)
    for i in range(2):
        ax[i, 0].set_ylabel(r"$p_n$", fontsize=fs)
        fname = f"ChaosGrid2D_K{str(K).replace('.','')}_N{N[i]}" + mode 
        absDiffGrid = np.loadtxt(PATHDATA + fname + ".gz")
        
        for j in range(2):
            _plot_chaos_maps(fig, ax[i, j], i, K, 2**N[i], Np, Nq, mode, 
                             absDiffGrid=absDiffGrid, 
                             SetTitle=(i == 0), UseImshow=(j == 0))
    
    from matplotlib.ticker import LogLocator
    
    ax[1, 1].xaxis.set_major_locator(LogLocator(numticks=4))
    for j in [0, 1]:
        ax[0, j].set_xticklabels([])
        ax[0, j].get_xaxis().set_visible(False)
    fig.canvas.draw()
    embed_labels(ax.flat)
    return fig

###############################################################################
# 4D standard map
###############################################################################

def plot_4d_orbits(k1=2.25, k2=3.0, k=1.0, Nplot=4096, Nmin=5.0, Nmax=14.0,
                   NN=100, init=None, mapMode='none', MapToCircle=1,
                   ShowMap=0, ShowTransform=0, SetTitle=1, WBAOnly=0):
    FlagInitNone = 0
    PresentationFlag = 0
    if type(init) == type(None):
        FlagInitNone = 1
        if abs(k - 1.0) < 1e-3:
            k1=2.25; k2=3.0; k=1.0
            print("Initials for strong coupling")
            initArray = [[0.028, 0.0, 0.5, 0.5],  
                         [0.02, -0.02, 0.5, 0.49], 
                         [-0.047, -0.02, 0.52, 0.49],]
        if mapMode == 'none' and MapToCircle == 1:
            if abs(k - 0.01) < 1e-3:
                k1=0.5; k2=0.7; k=0.01
                # initArray = [[0.39, -0.38, 0.0, 0.0], 
                #              [0.25, 0.322, 0.0, 0.0], 
                #              [0.35, 0.264, 0.0, 0.0], 
                #              [-0.355, 0.36, 0.145, 0.15],  
                #              [0.29, 0.235, 0.0, 0.0]]
                initArray = [[0.3636320489192, 0.35280259676167, 0.0, 0.0],
                             [-0.2589, -0.2623, 0.0, 0.0],
                             [0.288, -0.17113, 0.0, 0.0]]
            if abs(k - 0.04) < 1e-3:
                k1=0.5; k2=0.7; k=0.04
                initArray = [[0.3928, 0.2602, 0.0, 0.0], 
                             [-0.4, 0.131, 0.0, 0.0],
                             [0.42, 0.24, 0.0, 0.0], 
                             [0.333, -0.246, 0.0, 0.0],]
        if mapMode == 'arctan2' and MapToCircle == 0:
            if abs(k - 0.01) < 1e-3:
                k1=0.5; k2=0.7; k=0.01
                # initArray = [[0.38, 0.14, 0.5, 0.5], 
                #              [0.27, 0.08, 0.06, 0.5],
                #              [0.355, 0.11, 0.5, 0.5], 
                #              [0.44, 0.16, 0.5, 0.5], 
                #              [0.27, 0.04, 0.5, 0.5], 
                #              [0.25, 0.06, 0.5, 0.5]]
                initArray = [[0.08, 0.05, 0.5, 0.5], 
                             #[0.12, -0.05, 0.5, 0.5],
                             [0.1, 0.0, 0.6, 0.4], 
                             [-0.04, -0.1, 0.56, 0.671]]
                
        if mapMode == 'arctan2_p' and MapToCircle == 0:
            mapMode = 'arctan2'
            PresentationFlag = 1
            if abs(k - 0.01) < 1e-3:
                k1=0.5; k2=0.7; k=0.01
                initArray = [[0.08, 0.05, 0.5, 0.5], 
                             [-0.04, -0.1, 0.56, 0.671]]
    else:
        if init == 'failure':
            print("Initials for partial failure of frequency anlysis")
            k1=2.25; k2=3.0; k=1.0
            initArray = [[0.028, 0.0, 0.5, 0.5], 
                         [0.0007595, -0.02, 0.498507, 0.49], 
                         #[0.02, -0.02, 0.5, 0.49], 
                         [-0.047, -0.02, 0.52, 0.49],]
        if init == 'failure_p':
            PresentationFlag = 1
            k1=2.25; k2=3.0; k=1.0
            initArray = [[0.0007595, -0.02, 0.498507, 0.49], 
                         [-0.047, -0.02, 0.52, 0.49],]
            
        elif init == 'horseshoe':
            inits = np.loadtxt(PATHDATA + "InitHorseshoeResonance.gz")
            initArray = inits[:, 0]
        else:
            print("Using custom initial values")
            initArray = init
            
    _func = WBA_4D_tests.interactive_plot4d
    fig, ax = _func(k1, k2, k, Nplot, Nmin, Nmax, NN, initArray,
                    mapMode=mapMode, MapToCircle=MapToCircle, 
                    ShowMap=ShowMap, ShowTransform=ShowTransform, 
                    fs=fs, lfs=lfs-2, colors=colors, _dig=_dig, 
                    UseMarkers=['o', ms-1, mew-0.5], SetTitle=0,
                    AssertNaffEqualWBA=1, lw=lw, lwnaff=lwnaff,
                    WBAOnly=WBAOnly, fscale=fscale, 
                    AssertNu1Nu2Order=(ShowMap == 1), 
                    AssertNuLess05=(mapMode != 'none'), 
                    PresentationFlag=PresentationFlag)
    if SetTitle:
        ax[0].set_title(f"$K_1={round(k1, 3)}$, $K_2={round(k2, 3)}$, \
$K={round(k, 3)}$ and {Nplot} points", fontsize=fs)
        if PresentationFlag:
            rcParams["legend.loc"] = 'lower left'
            ax[0].set_title(f"$K_1={k1}, K_2={k2}$ and $K={k}$", fontsize=fs)
    
    ax[1].set_xlim(2**Nmin, 2**Nmax)
    ax[3].set_xlim(2**Nmin, 2**Nmax)
    if MapToCircle == 0 and FlagInitNone == 1:
        shift = 0
        if PresentationFlag:
            shift = 0.05
        axisBbox = [0.25 + shift, 0.75 - shift, -0.25 + shift, 0.25 - shift]
        for i in [0, 2]:
            ax[i].axis(axisBbox)
            set_xticks_linear(ax[i], axisBbox[0], axisBbox[1], 5)
            set_yticks_linear(ax[i], axisBbox[2], axisBbox[3], 5)
            
    from matplotlib.ticker import MaxNLocator
    if init.startswith('failure'):
        # axisBbox = [0.41, 0.59, -0.09, 0.09]
        axisBbox = [0.4, 0.6, -0.1, 0.1]
        for i in [0, 2]:
            ax[i].axis(axisBbox)
            set_xticks_linear(ax[i], axisBbox[0], axisBbox[1], 5)
            set_yticks_linear(ax[i], axisBbox[2], axisBbox[3], 5)
        if mapMode == 'arctan2':
            ax[1].set_ylim(1e-20, 0.1)
            ax[3].set_ylim(1e-20, 1.0)
            # ax[1].set_ylim(1e-17, 0.01)
            # ax[3].set_ylim(1e-17, 1.0)
        if mapMode == 'torus4d':
            ax[5].set_xlim(-0.01605, 0.01605)
            
        if PresentationFlag:
            for axis in ax[np.array([4, 5])]:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
    for i, axis in enumerate(ax):
        axis.tick_params(labelsize=tls+1 + ShowMap)
        if i in np.array([0, 2, 4, 5]):
            if not PresentationFlag:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.canvas.draw()
    labelAxis = ['both', 'x', 'both', 'x', 'both', 'both']
    if PresentationFlag:
        embed_labels(ax, SetCaptions=False, labelAxis=labelAxis[:len(ax)],
                     fontsize=fs+1 + ShowMap)
    else:
        embed_labels(ax, SetCaptions=[0,0,1,1,0,1], 
                     labelAxis=labelAxis[:len(ax)], fontsize=fs+1 + ShowMap)
    fig.tight_layout()
    return fig, ax

def plot_torus4d_transform(init=None, N=2**14, k1=2.25, k2=3.0, k=1.0,
                           thresh=0.01, ShowConv=0, SetTitle=0):
    if type(init) == type(None):
        initArray = [0.0007595, -0.02, 0.498507, 0.49]
    if init == 'failure':
        initArray = [0.08090725947182, -0.07995087108008,
                     0.44041467529610, 0.59718510271600]
    if init == 'horseshoe resonance':
        inits = np.loadtxt(PATHDATA + "InitHorseshoeResonance.gz")
        initArray = inits[:, 0]
    if init == 'lower edge':
        inits = np.loadtxt(PATHDATA + "InitsLowEdgeFailure.gz")
        initArray = inits[:, 0]
    if init == 'flat transform':
        inits = np.loadtxt(PATHDATA + "InitsEdgyFailure.gz")
        initArray = inits[:, 0]
        
    _func = WBA_4D_tests.frequency_transformed_torus4d
    fig, ax = _func(*initArray, N, k1, k2, k, thresh, ShowConv, RetVal=0,
                    SetTitle=SetTitle, SetLabels=1, fs=fs, TightLayout=0,
                    colors=colors, fscale=fscale)
    from matplotlib.ticker import MaxNLocator
    for axis in ax:
        axis.tick_params(labelsize=tls)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for axis in ax[np.array([0, 4])]:
        xlim = axis.get_xlim()
        set_xticks_linear(axis, np.round(xlim[0], 2), 
                          np.round(xlim[1], 2), numticks=3)
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=[0,0,0,0,1,1,1,1])
    fig.tight_layout()
    return fig

def _plot_torus4d_transform(ax, points, ms=1, fs=fs):
    labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    ax[0, 0].set_xlabel(r"$q_1$", fontsize=fs)
    ax[0, 0].set_ylabel(r"$p_1$", fontsize=fs)
    ax[1, 0].set_xlabel(r"$q_2$", fontsize=fs)
    ax[1, 0].set_ylabel(r"$p_2$", fontsize=fs)
    ax[0, 1].set_xlabel(labels[0], fontsize=fs)
    ax[0, 1].set_ylabel(labels[1], fontsize=fs)
    ax[1, 1].set_xlabel(labels[2], fontsize=fs)
    ax[1, 1].set_ylabel(labels[3], fontsize=fs)
    ax[0, 0].scatter(points[2, :], points[0, :], s=ms, color='k')
    ax[1, 0].scatter(points[3, :], points[1, :], s=ms, color='k')
    points[2:, :] -= 0.5
    transp = WBA_core.transform_nd_torus(points)
    transp = WBA_core.sort_by_extent(transp)
    ax[0, 1].scatter(transp[2, :], transp[0, :], s=ms, color='blue')
    ax[1, 1].scatter(transp[3, :], transp[1, :], s=ms, color='blue')
    pass

def plot_torus4d_transform_p(N=2**14, k1=2.25, k2=3.0, k=1.0):
    initArray = [0.0007595, -0.02, 0.498507, 0.49]
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    points = np.array(Mapping4dCyl(k1, k2, k).mapN(*initArray, N))
    _plot_torus4d_transform(ax, points)
    ax = ax.flatten()
    from matplotlib.ticker import MaxNLocator
    for axis in ax:
        axis.tick_params(labelsize=tls)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for axis in ax[np.array([0, 2])]:
        set_xticks_linear(axis, 0.46, 0.54, 5)
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=False, fontsize=fs)
    fig.tight_layout()
    return fig, ax

def plot_torus4d_transform2(N=2**14, k1=2.25, k2=3.0, k=1.0):
    _map = Mapping4dCyl(k1, k2, k).mapN
    fig, ax = plt.subplots(2, 4, figsize=(16, 9))
    ax = np.array([ax[:, :2], ax[:, 2:]])
    for i, fname in enumerate(["InitsEdgyFailure.gz", 
                               "InitsLowEdgeFailure.gz",]):
        inits = np.loadtxt(PATHDATA + fname)
        initArray = inits[:, 0]
        points = np.array(_map(*initArray, N))
        _plot_torus4d_transform(ax[i], points)
    
    from matplotlib.ticker import MaxNLocator
    ax[1, 0, 1].set_ylim(-0.405, 0.405)
    ax[1, 1, 0].set_ylim(-0.305, 0.305)
    ax[0, 0, 1].set_ylim(-0.305, 0.305)
    
    for axis in ax.flat:
        axis.tick_params(labelsize=tls)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
    for axis in ax.flat[::2]:
        xlim = axis.get_xlim()
        set_xticks_linear(axis, np.round(xlim[0], 2), 
                          np.round(xlim[1], 2), numticks=3)
        
    for axis in ax.flat:
        axis.tick_params(labelsize=tls)
    fig.canvas.draw()
    embed_labels(ax.flat, SetCaptions=[0,0,1,1,0,0,1,1])
    fig.tight_layout()
    return fig

def plot_freqspace_deltaphi(inits=None, CompFreq=1, 
                            initsIndx=np.array([0, 10])):
    if type(inits) == type(None):
        inits = np.loadtxt(PATHFREQ + "InitsNu1_0285_0085_sorted.gz")
        inits = inits[:, initsIndx]
    _plot = WBA_4D_tests.plot_projection_deltaphi
    fig, ax = _plot(inits=inits, Embedding=1, CompFreq=CompFreq)
    ax[1, 0].set_ylim(-0.283, -0.0483)
    from matplotlib.ticker import MaxNLocator
    for axis in ax.flat:
        axis.tick_params(labelsize=tls+1)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # ylim = axis.get_ylim()
        # set_yticks_linear(axis, np.round(ylim[0], 2), 
        #                   np.round(ylim[1], 2), numticks=5)
    fig.canvas.draw()
    embed_labels(ax.flat, SetCaptions=True, fontsize=fs+1)
    fig.tight_layout()
    return fig, ax

def _points_and_sorted_points(init, k1=2.25, k2=3.0, k=1.0, N=2**14):
    points = np.array(Mapping4dCyl(k1, k2, k).mapN(*init, N))
    points[2:, :] -= 0.5
    sortp = WBA_core.sort_by_extent(WBA_core.transform_nd_torus(points))
    return points, sortp

def plot_freqspace_deltaphi_v2(ms=1):
    inits = np.loadtxt(PATHFREQ + "InitsNu1_0285_0085_sorted.gz")
    inits = inits[:, np.array([0, 10])]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for i in range(2):
        points, sortp = _points_and_sorted_points(inits[:, i])
        p2, q2 = points[np.array([1, 3]), :]
        x3, x4 = sortp[np.array([1, 3]), :]
        ax[i, 0].scatter(q2 + 0.5, p2, c='b', s=ms)
        ax[i, 1].scatter(x3, x4, c='b', s=ms)
        phi = np.arctan2(x3, x4) / (2*np.pi)
        deltaPhi = phi[1:] - phi[:-1]
        WBA_core.embedding(deltaPhi)
        print(WBA_core._WBA_single(deltaPhi))
        print(naff_4d(points.T), end="\n")
        ax[i, 2].scatter(phi[1:], deltaPhi, c='b', s=ms)
        ax[i, 2].set_xlim(-0.5, 0.5)
        ax[i, 0].set_xlabel(r"$q_2$", fontsize=fs)
        ax[i, 0].set_ylabel(r"$p_2$", fontsize=fs)
        ax[i, 1].set_xlabel(r"$x_3$", fontsize=fs)
        ax[i, 1].set_ylabel(r"$x_4$", fontsize=fs)
        ax[i, 2].set_xlabel(r"$\phi_2$", fontsize=fs)
        ax[i, 2].set_ylabel(r"$\Delta\phi_2$", fontsize=fs)
    from matplotlib.ticker import MaxNLocator
    for i, axis in enumerate(ax.flat):
        axis.tick_params(labelsize=tls+1)
        nbins = 3
        if i in [0, 3]: nbins = 4
        axis.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        if i == 1: 
            set_xticks_linear(axis, -0.01, 0.01, 3)
            continue
        axis.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    fig.canvas.draw()
    embed_labels(ax.flat, SetCaptions=False, fontsize=fs+1)
    embed_labels(ax[0, 2], SetCaptions=False, fontsize=fs+1, 
                 alignment=("center", "left"))
    fig.tight_layout()
    return fig, ax

def plot_3d_mpl():
    inits = np.loadtxt(PATHDATA + "InitsLowEdgeFailure.gz")
    points = np.array(Mapping4dCyl().mapN( *inits[:,0],4096))
    points[2:, :] -= 0.5
    transp = WBA_core.transform_nd_torus( points)
    x,y,z,w = transp
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(x, y, z, c=w, cmap='viridis')
    return fig

def plot_freq_space(fnames, axis=None, order=4, RetVal=0, 
                    cmap='viridis'):
    fList = [np.loadtxt(fname) for fname in fnames]
    # fList = [np.random.uniform(size=(4, val)) for val in [250, 500, 250]]
    fLengths = [max(np.shape(fElem)) for fElem in fList]
    fVals = np.zeros((4, np.sum(fLengths)))
    ctr = 0
    for i, fElem in enumerate(fList):
        fVals[:, ctr:ctr+fLengths[i]] = fElem
        ctr += fLengths[i]
        
    print(f"Total number of frequency pairs is {ctr}.")
        
    f1p1, f2p1, f1p2, f2p2 = fVals
    diff1, diff2 = np.abs(f1p1 - f1p2), np.abs(f2p1 - f2p2)
    diff1[diff1 < 1e-16] = 1e-16
    diff2[diff2 < 1e-16] = 1e-16
    abl1, abl2 = -np.log10(diff1), -np.log10(diff2)
    indx = ((abl1 < order) | (abl2 < order) | (f1p1 < 1e-3) | (f2p1 < 1e-3))
    if RetVal:
        return np.array([f1p1, f2p1, f1p2, f2p2])[:, ~indx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(right=0.83)
    ax.tick_params(labelsize=tls)
    ax.set_xlabel(r"$\nu_1$", fontsize=fs)
    ax.set_ylabel(r"$\nu_2$", fontsize=fs)
    if type(axis) == type(None):
        ax.axis([0.27, 0.31, 0.06, 0.17])
    else:
        ax.axis(axis)
    
    f1plot, f2plot = f1p1[~indx], f2p1[~indx]
    indx12 = (f2plot > f1plot)
    f1plot[indx12], f2plot[indx12] = f2plot[indx12], f1plot[indx12]
    
    print(f"Frequency space with {len(f1plot)} points of order {order}.")
    # ax.plot(f1plot, f2plot, c='k', ls='', marker='.', ms=1, mew=0)
    # colormap = plt.cm.get_cmap(cmap)
    # dataForCmap = abl1[~indx] - np.min(abl1[~indx])
    # dataForCmap /= np.max(dataForCmap)
    # cMap = colormap(dataForCmap)
    cNorm = plt.cm.colors.LogNorm(vmin=1e-15, vmax=1e-3)
    
    im = ax.scatter(f1plot, f2plot, c=diff1[~indx], 
                    marker='.', s=1, linewidths=0,
                    cmap=cmap, norm=cNorm)
    # ticks = np.array([-4, -7, -10, -13, -16])
    # cmap = plt.cm.ScalarMappable(cmap=cmap)
    # cmap.set_clim(np.min(abl1), np.max(abl1))
    # cmap.set_clim(min(np.min(abl1), np.min(abl2)), 
    #               max(np.max(abl1), np.max(abl2)))
    cbar_axim = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_axim,) 
                        #ticks=10**ticks.astype(np.float64))
    # cbar = fig.colorbar(cmap, cax=cbar_axim, fontsize=fs)
    # cbar.set_ticklabels([f"$10^{{{tick}}}$" for tick in ticks])
    cbar.ax.tick_params(labelsize=tls)
    # ticklabs = cbar.ax.get_yticklabels()
    # cbar.ax.set_yticklabels(ticklabs, fontsize=10)
    # cbar.set_label(r'$|\Delta\nu|$', fontsize=fs)
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=False)
    # plt.tight_layout()
    return fig

# def test_plot_cmap():
#     fig, ax = plt.subplots(figsize=(12, 10))
#     plt.subplots_adjust(right=0.83)
#     ax.tick_params(labelsize=tls)
#     cNorm = plt.cm.colors.LogNorm(vmin=1e-5, vmax=1e-2)
#     x=np.linspace(0.0, 1.0, 200)
#     y=0.5*x**2 + 0.3
#     c=np.logspace(-5,-2,200)
#     im = ax.scatter(x, y, c=c, 
#                     marker='o', s=4, linewidths=0,
#                     cmap='viridis', norm=cNorm)
#     ticks = np.array([-2, -3, -4, -5, -6])
#     cbar_axim = fig.add_axes([0.87, 0.15, 0.03, 0.7])
#     cbar = fig.colorbar(im, cax=cbar_axim,)
#                         ticks=10**ticks.astype(np.float64))
#     # cbar = fig.colorbar(cmap, cax=cbar_axim, fontsize=fs)
#     # cbar.set_ticklabels([f"$10^{{{tick}}}$" for tick in ticks])
#     cbar.ax.tick_params(labelsize=tls)
#     return fig

if __name__ == "__main__":
    print(__doc__)
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    PATHDATA = PATH_TP + "CP_Bachelor\\WBA_Python\\DataFiles\\"
    PATHFREQ = PATH_TP + "CP_Bachelor\\WBA_Python\\FreqSpace\\"
    ##### horizontal y labels with horizontal alignment 'ha'
    # set_ylabel(ylabel, rotation='horizontal', ha='right', fontsize=fs) 
    
    # fig, ax = simple_periodic_plot(OnlySignal=1, N=128)
    # plt.savefig(PATH + "PureRotation.png", dpi=150)
    # plt.close(fig)
    
    
    # ##### artificial signals for different frequencies
    # fig, ax = plot_art_signal(Nmin=1.5, Nmax=14.0, NN=40, SetYlimits=1,
    #                           rad=[0.7, 1.0, 1.3], OnlySignal=1, Nplot=128)
    # plt.savefig(PATH + "OscPeriodicOnlySignal.png", dpi=150)
    # plt.close(fig)
    # fig, ax = plot_art_signal(freq2=abs(3 - np.sqrt(13))/2,
    #                       ampl=0.1, NN=100,
    #                       rad=[0.7, 1.0, 1.3])
    # plt.savefig(PATH + "OscPeriodic2fA01.png", dpi=150)
    # plt.close(fig)
    # fig, ax = plot_art_signal(freq2=abs(3 - np.sqrt(13))/2,
    #                       ampl=0.5, NN=100,
    #                       rad=[0.7, 1.0, 1.3], OnlySignal=1)
    # plt.savefig(PATH + "OscPeriodic2fA05.png", dpi=150)
    # plt.close(fig)
    
    # ##### delta phi of phi for artificial signal
    # fig = plot_debug_art_signal(ModAlpha=1)
    # plt.savefig(PATH + "OscPeriodic2fA05PhiDiff.png", dpi=150)
    # plt.close(fig)
    # fig = plot_debug_art_signal(ModAlpha=0)
    # plt.savefig(PATH + "OscPeriodic2fA05PhiDiffNoMod.png", dpi=150)
    # plt.close(fig)
    
    # ##### grid of frequencies for artificial signals
    # fig = plot_art_signal_grid(Alpha=0)
    # plt.savefig(PATH + "OscPeriodic2fA05Grid.png", dpi=150)
    # plt.close(fig)
    # fig = plot_art_signal_grid(Alpha=1)
    # plt.savefig(PATH + "OscPeriodic2fA05GridAlphaWBA.png", dpi=150)
    # plt.close(fig)
    
    # ### if False:
    # ###     adg = np.loadtxt(PATHDATA + "OscPeriodic2fA05GridN10WBA.gz")
    # ###     adg = plot_art_signal_grid(N=2**10, RetVal=1)
    # ###     np.savetxt("OscPeriodic2fA05GridN10Alpha050WBA.gz", adg)
    # ### fig = plot_art_signal_grid(N=2**10, ampl=0.5, Nf=300, Nf2=300,
    # ###                             absDiffGrid=adg, UseNaff=0)
    # ### plt.savefig(PATH + "OscPeriodic2fA05GridN10Alpha050WBA.png", dpi=150)
    
    # fig, ax = plot_2d_orbits(MapToCircle=1, mapMode='none')
    # plt.savefig(PATH + "Rot2D_K07.png", dpi=150)
    # plt.close(fig)
    # fig, ax = plot_2d_orbits(MapToCircle=0, mapMode='arctan2')
    # plt.savefig(PATH + "Osc2D_K07.png", dpi=150)
    # plt.close(fig)
    
    # ##### frequency along vector
    # fig = plot_freq_along_vector(Nt=1000)
    # plt.savefig(PATH + "FreqAlongVectorK09N12Nt1000_025to030.png", dpi=150)
    # plt.close(fig)
    # fig, ax = plot_freq_along_vector_v2(Nt=1000)
    # plt.savefig(PATH + "FreqAlongVector_v2.png", dpi=100)
    # fig, ax = plot_freq_along_vector_v2(Nt=1000, RemoveChaos=1)
    # plt.savefig(PATH + "FreqAlongVector_v2_noChaos.png", dpi=100)
    # fig, ax = plot_2d_std_map(count=150)
    # plt.savefig(PATH + "StdMap2D_K09_150rand.png", dpi=100)
    
    # ##### chaos indicator maps
    # # fig = plot_chaos_maps('WBA')
    # for mode in ['WBA', 'cos', 'Naff', 'NaffMTC']:
    #     fig = plot_chaos_maps(mode=mode)
    #     plt.savefig(PATH + "ChaosMaps2D_K09" + mode + ".png", dpi=150)
    #     plt.close(fig)
    
# debugging
# q, p = WBA_2D_tests._2f_signal( 0.4142,0.153,ampl=0.5, N=2**11, rad=1.0)
# phi=np.arctan2(q, p) / (2*np.pi)
# phiDiff = phi[1:] - phi[:-1]
# alpha=0.2
# phiDiff = (phiDiff + alpha) % 1.0 - alpha
# if np.any(phiDiff > 0.75-alpha) and np.any(phiDiff < 0.25-alpha):
#     phiDiff[phiDiff < 0.5 - alpha] += 1.0
# #if np.any(phiDiff > 0.25) and np.any(phiDiff < -0.25):
# #    phiDiff %= 1.0
# print(WBA_core.WBA(phi[1:], phiDiff))
# fig,ax=plt.subplots()
# ax.scatter(phi[1:],phiDiff,s=4)

    ##### 4d orbits
    # plot_4d_orbits(k1=0.5, k2=0.7, k=0.01, mapMode='none', MapToCircle=1)
    # plt.savefig(PATH + "RotRot4DK05K07K001.png", dpi=150)
    # plot_4d_orbits(k1=0.5, k2=0.7, k=0.01, mapMode='arctan2', MapToCircle=0)
    # plt.savefig(PATH + "OscOsc4DK05K07K001.png", dpi=150)
    # plot_4d_orbits(mapMode='arctan2', MapToCircle=0, init='failure')
    # plt.savefig(PATH + "OscOsc4DK225K3K1_failure.png", dpi=150)
    # plot_4d_orbits(mapMode='torus4d', MapToCircle=0, init='failure',
    #                 ShowMap=1, ShowTransform=1, SetTitle=0, WBAOnly=1)
    # plt.savefig(PATH + "OscOsc4DK225K3K1_success.png", dpi=150)
    ### plot_4d_orbits(mapMode='torus4d', MapToCircle=0, init='horseshoe',
    ###                 ShowMap=1, ShowTransform=1, SetTitle=0, WBAOnly=0)
    
    # fig, ax = plot_4d_orbits(k1=0.5, k2=0.7, k=0.01, mapMode='arctan2_p',
    #                          MapToCircle=0)
    # plt.savefig(PATH + "OscOsc4DK05K07K001_p.png", dpi=100)
    # fig, ax = plot_4d_orbits(mapMode='arctan2', init='failure_p',
    #                          MapToCircle=0)
    # plt.savefig(PATH + "OscOsc4DK225K3K1_failure_p.png", dpi=100)
    # fig, ax = plot_4d_orbits(mapMode='torus4d', MapToCircle=0,
    #                          init='failure_p', ShowMap=1, ShowTransform=1,
    #                          SetTitle=0, WBAOnly=0)
    # plt.savefig(PATH + "OscOsc4DK225K3K1_success_p.png", dpi=100)
    
    ##### torus 4d transform
    # plot_torus4d_transform()
    # plt.savefig(PATH + "Torus4dTransform.png", dpi=150)
    # plot_torus4d_transform(init='horseshoe resonance')
    # plt.savefig(PATH + "Torus4dTransformHorseshoe.png", dpi=150)
    # plot_torus4d_transform(init='lower edge')
    # plt.savefig(PATH + "Torus4dTransformLowerEdge.png", dpi=150)
    # plot_torus4d_transform(init='flat transform')
    # plt.savefig(PATH + "Torus4dTransformFlatTransform.png", dpi=150)
    # plot_torus4d_transform2()
    # plt.savefig(PATH + "Torus4dTransform2failures.png", dpi=150)
    
    # fig, ax = plot_torus4d_transform_p()
    # plt.savefig(PATH + "Torus4dTransform_p.png", dpi=100)
    
    ##### frequency space for 4D standard map
    # fNamePrefix = (PATHDATA + "FreqSpaceN11P_0_25to0_25P_0_25to0_25" + 
    #                 "Q0_25to0_75Q0_25to0_75K2_25K3_0K1_0")
    # fNames = [fNamePrefix + suffix for suffix in 
    #           ["WBA_1000x25000random_order3.gz",
    #             "WBA_1000x50000random_order3.gz", 
    #             "WBA_200x250x20x25random_order3.gz"]]
    # fig = plot_freq_space(fNames, RetVal=0)
    # fig4 = test_plot_cmap()
    
    # fig, ax = plot_freqspace_deltaphi()
    # plt.savefig(PATH + "Torus4dTransform_DeltaPhi.png", dpi=150)
    # fig, ax = plot_freqspace_deltaphi_v2()
    # plt.savefig(PATH + "Torus4dTransform_DeltaPhi_p.png", dpi=100)
    


