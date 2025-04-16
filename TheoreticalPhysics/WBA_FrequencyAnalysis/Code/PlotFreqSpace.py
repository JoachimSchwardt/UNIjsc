# -*- coding: utf-8 -*-
"""
Plot of frequency space with WBA:
    fNames = list of fNames to be loaded. Prefered format is .gz
             (each file should contain a (4,n) - np.array with the structure:
                  first row: nu1 from first half of an orbit
                  second row: nu2 from first half of an orbit
                  third row: nu1 from second half of an orbit
                  fourth row: nu2 from second half of an orbit)
    axis = region of interest for the plot 
           (default is 'axis = [0.27, 0.31, 0.06, 0.17]')
    order = -log10(DeltaNu) is the cutoff for the chaos indicator
            (frequency from two halves of an orbit, DeltaNu := difference)
    RetVal = Return the values instead of plotting
    UseTex = use latex fonts (interactive plot becomes very slow, default 0)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from WBA_Plotting import embed_labels
rcParams["figure.dpi"] = 100

# from explorator.common.orbit_manager import OrbitManager

fs = 21     # fontsize (titles and labels)
tls = 18    # tick label size

def _resonance(nu1, m1, m2, l):
    assert m2 != 0
    return l / m2 - m1 / m2 * nu1

def _plot_resonance(ax, m1, m2, l, c='k', 
                    ls=(0, (5, 10)), lw=1.0,
                    alpha=0.5, fontsize=fs-4, xpos=0.1):
    start, end = ax.get_xlim()
    yOffset = (end - start) * 0.01
    nu1 = np.array([start, end])
    if m2 != 0:
        nu2 = _resonance(nu1, m1, m2, l)
        line = ax.plot(nu1, nu2, c=c, ls=ls, lw=lw, alpha=alpha)
    else:
        line = ax.axvline(l / m1, c=c, ls=ls, lw=lw, alpha=alpha)
    
    
    nu1pos = start + xpos * (end - start)
    
    if m2 != 0:
        rotation = -180/np.pi * np.arctan2(m1, m2)
        nu2pos = _resonance(nu1pos, m1, m2, l)
        rotation = ax.transData.transform_angles([rotation], 
                                                 [[nu1pos, nu2pos]])[0]
        if abs(rotation) > 90:
            rotation += 180
        ha, va = 'center', 'bottom'
    else:
        ylim = ax.get_ylim()
        nu2pos = ylim[0] + xpos * (ylim[1] - ylim[0])
        nu1pos = l / m1 + yOffset
        ha, va = 'left', 'center'
        rotation = 90
    text = ax.text(nu1pos, nu2pos + yOffset, f"${m1}:{m2}:{l}$",
                   fontsize=fontsize,
                   rotation=rotation, ha=ha, va=va, rotation_mode='anchor')
    return line, text

def load_data(fNames=None, SpecialFile=None):
    if type(fNames) == type(None):
        fNames = []
        for r, d, f in os.walk(os.getcwd()):
            for file in f:
                if file.startswith("FreqSpace") and file.endswith(".gz"):
                    fNames.append(file)
                if type(SpecialFile) != None:
                    if file.startswith(SpecialFile):
                        fNames.append(file)
    
    # orb_man = OrbitManager.load_h5("RicLanBaeKet2014_freqs.h5")
    # freqNaff = [np.array(orb.frequencies.freqs).T 
    #         for grp in orb_man.groups 
    #         for orb in grp.orbits]
    # fn1, fn2 = np.concatenate(freqNaff, axis=1)
                    
    print("Reading files ", fNames, " ... ")
    fList = [np.loadtxt(fName) for fName in fNames]
    fLengths = [max(np.shape(fElem)) for fElem in fList]
    fVals = np.zeros((4, np.sum(fLengths)))
    ctr = 0
    for i, fElem in enumerate(fList):
        fVals[:, ctr:ctr+fLengths[i]] = fElem
        ctr += fLengths[i]
    print(f"Total number of frequency pairs is {ctr}.")
    return fVals

def plot_freq_space(fWBA, axis=None, order=4, RetVal=0, 
                    cmap='viridis', fs=fs, tls=tls, UseTex=False,
                    ShowResonance=False, ShowNewNaff=0, ShowInset=0,
                    LabelPlotInset=0):
    if UseTex:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=UseTex)
    
    fn1, fn2 = np.loadtxt(PATHFREQ + "Naff2014_freqs.gz")
    fVals = fWBA
    f1p1, f2p1, f1p2, f2p2 = fVals
    diff1, diff2 = np.abs(f1p1 - f1p2), np.abs(f2p1 - f2p2)
    diff1[diff1 < 1e-16] = 1e-16
    diff2[diff2 < 1e-16] = 1e-16
    abl1, abl2 = -np.log10(diff1), -np.log10(diff2)
    indx = ((abl1 < order) | (abl2 < order) | (f1p1 < 1e-3) | (f2p1 < 1e-3))
    if RetVal:
        return np.array([f1p1, f2p1, f1p2, f2p2])[:, ~indx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.07, right=0.88, top=0.98, bottom=0.08)
    ax.tick_params(labelsize=tls)
    ax.set_xlabel(r"$\nu_1$", fontsize=fs)
    ax.set_ylabel(r"$\nu_2$", fontsize=fs)
    if type(axis) == type(None):
        ax.axis([0.27, 0.31, 0.06, 0.17])
    else:
        ax.axis(axis)
        
    if ShowResonance:
        m1 = [-1, -1, 3, -2200, 0, 10, 7, 7, ]
        m2 = [2, 3, 1, 5000, 8, 2, -1, 1, ]
        l = [0, 0, 1, -17, 1, 3, 2, 2, ]
        xpos = [0.7, 0.165, 0.65, 0.15, 0.44, 0.25, 0.65, 0.12]
        for i in [0, 1, 2, 4, 5, 6, 7]:
            _plot_resonance(ax, m1[i], m2[i], l[i], xpos=xpos[i])
            
    
    f1plot, f2plot = f1p1[~indx], f2p1[~indx]
    indx12 = (f2plot > f1plot)
    f1plot[indx12], f2plot[indx12] = f2plot[indx12], f1plot[indx12]
    
    print(f"Frequency space with {len(f1plot)} points of order {order}.")
    cNorm = plt.cm.colors.LogNorm(vmin=np.min(diff1[~indx]), 
                                  vmax=np.max(diff1[~indx]))
    
    ax.scatter(fn1, fn2, marker='.', s=3, c='k', alpha=0.4, 
               linewidths=0)
    if ShowNewNaff:
        fn1new, fn2new = np.loadtxt(PATHFREQ + "Naff2021_freqs.gz")
        ax.scatter(fn1new, fn2new, marker='.', s=7, c='orangered', 
                   alpha=0.3, linewidths=0)
        if ShowInset:
            # x0, y0, width, height
            from matplotlib.patches import Rectangle
            axisInset = ax.inset_axes([0.025, 0.45, 0.29, 0.33])  
            for axs in ['top','bottom','left','right']:
                axisInset.spines[axs].set_linewidth(1.0)
                axisInset.spines[axs].set_alpha(0.5)
            axisInset.tick_params(labelsize=10)
            x0, x1, y0, y1 = 0.286, 0.288, 0.156, 0.15
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                             facecolor='none', edgecolor='k', linewidth=1.0,
                             alpha=0.5)
            axisInsetBbox = axisInset.bbox.transformed(
                ax.transData.inverted())
            ax.plot([axisInsetBbox.x0 + 0.00005, x0], 
                    [axisInsetBbox.y1 + 0.00005, y0 - 0.0001], c='k',
                    linewidth=1.0, alpha=0.5)
            ax.plot([axisInsetBbox.x1 + 0.00005, x1], 
                    [axisInsetBbox.y0, y1], c='k',
                    linewidth=1.0, alpha=0.5)
            ax.add_patch(rect)
            axisInset.axis([x0, x1, y1, y0])
            if LabelPlotInset:
                axisInset.locator_params(nbins=2)
                axisInset.set_xlabel(r"$\nu_1$", fontsize=12)
                axisInset.set_ylabel(r"$\nu_2$", fontsize=12)
                embed_labels(axisInset, SetCaptions=False, fontsize=12,
                              alignment=['bottom', 'left'])
            else:
                axisInset.set_xticklabels([])
                axisInset.set_yticklabels([])
                axisInset.get_xaxis().set_visible(False)
                axisInset.get_yaxis().set_visible(False)
            indxInset = ((fn1 > x0) & (fn1 < x1) & (fn2 > y1))
            fn1Inset, fn2Inset = fn1[indxInset], fn2[indxInset]
            axisInset.scatter(fn1Inset, fn2Inset, c='k', 
                              marker='.', s=12, linewidths=0, alpha=0.8)
            indxInset = ((fn1new > x0) & (fn1new < x1) & (fn2new > y1))
            fn1Inset, fn2Inset = fn1new[indxInset], fn2new[indxInset]
            axisInset.scatter(fn1Inset, fn2Inset, c='orangered', 
                              marker='.', s=16, linewidths=0, alpha=0.8)
            indxInset = ((f1plot > x0) & (f1plot < x1) & (f2plot > y1))
            cInset = diff1[~indx]
            axisInset.scatter(f1plot[indxInset], f2plot[indxInset],
                              c=cInset[indxInset], marker='.', s=5,
                              linewidths=0, cmap=cmap, norm=cNorm)
    
    s = 1
    if len(f1plot) < 100000:
        s = 3
    # only nu1 frequency difference as chaos indicator 
    im = ax.scatter(f1plot, f2plot, c=diff1[~indx], 
                    marker='.', s=s, linewidths=0,
                    cmap=cmap, norm=cNorm)
    cbar_axim = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_axim,) 
    cbar.ax.tick_params(labelsize=tls)
    fig.canvas.draw()
    embed_labels(ax, SetCaptions=False, fontsize=fs)
    return fig, ax

if __name__ == "__main__":
    print(__doc__)    
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    PATHDATA = PATH_TP + "CP_Bachelor\\WBA_Python\\DataFiles\\"
    PATHFREQ = PATH_TP + "CP_Bachelor\\WBA_Python\\FreqSpace\\"
    fNames = None
    fNames = []
    for r, d, f in os.walk(os.getcwd()):
        for file in f:
            if file.startswith("FreqSpace") and file.endswith(".gz"):
                fNames.append(file)
    fNames = fNames[:2]
    fWBA = load_data(fNames)
    fig, ax = plot_freq_space(fWBA, UseTex=True, order=5, ShowResonance=True,
                              ShowNewNaff=False)
    # plt.savefig(PATH + "FreqSpaceColoredWithNaff_p.png", dpi=100)
    
    """
from numba import njit

@njit('Tuple((u4[:,:], uint16[:]))(f8[:], f8, i4)')
def _array_to_intervals_njit(arr, bLen, bSizeEst):
    totalBCount = int(np.max(arr) // bLen) + 1
    indxArray = np.full((totalBCount, bSizeEst), len(arr), dtype=np.uint32)
    indxTable = np.zeros(totalBCount, dtype=np.uint16)
    for i in range(len(arr)):
        bIndx = int(arr[i] // bLen)
        indxArray[bIndx, indxTable[bIndx]] = i
        indxTable[bIndx] += 1
    return indxArray, indxTable

fn1, fn2 = np.loadtxt(PATHFREQ + "Naff2021_freqs.gz")
indxr = ((fn1 > 0.27) & (fn1 < 0.31))
fn1r, fn2r = fn1[indxr], fn2[indxr]

indxArray, indxTable = _array_to_intervals_njit(fn1r - 0.27, 0.0001, 100)
fmax, fmin = [], []
for i in range(len(indxArray)):
    indx = indxArray[i, :indxTable[i]]
    f1, f2 = fn1r[indx], fn2r[indx]
    try:
        idxmax = np.argmax(f2)
        idxmin = np.argmin(f2)
    except ValueError:
        continue
    fmax.append([f1[idxmax], f2[idxmax]])
    fmin.append([f1[idxmin], f2[idxmin]])
    
fn1max, fn2max = np.array(fmax).T
fn1min, fn2min = np.array(fmin).T
indxplot = ((fn1max > 0.285) & (fn1max < 0.29) & (fn2max > 0.146))
fn1plot, fn2plot = fn1max[indxplot], fn2max[indxplot]

fig2, ax2 = plt.subplots(figsize=(12, 10))
ax2.scatter(fn1, fn2, marker='.', s=6, c='blue', alpha=0.3, linewidths=0)
ax2.scatter(fn1plot, fn2plot, marker='.', s=3, alpha=1.0, linewidths=1, c='darkred')
    """