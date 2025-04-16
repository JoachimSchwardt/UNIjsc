# -*- coding: utf-8 -*-
"""
Testing WBA for 4D maps
"""
import numpy as np
import WBA_tools
import WBA_core
from CPG.naff.examples.std_map_frequencies import compute_freq
from explorator.comp.naff_call import naff_4d
import time
# from std_map import _std_map4d, _std_map_multi4d
from explorator.imports.maps.map_standard4d_v3_cyl import Mapping
from std_map import Mapping4dCyl
# from explorator.common.slice_info import SliceInfo
from explorator.common.orbit_manager import OrbitManager
# from explorator.common.orbit_manager_from_list import orbit_manager_from_list
# import h5py
from mayavi import mlab
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

import matplotlib.pyplot as plt
import functools
# import concurrent.futures as cft
from matplotlib import rcParams
rcParams["figure.dpi"] = 100

from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
#https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar

def list_from_orbit_manager(orbit_manager):        
    return [orbit for orbit in orbit_manager.all_orbits()]

# class Orbit4d(object):
#     def __init__(self, init=[], k1=0.0, k2=0.0, k=0.0, Npoints=100):
#         self.k1, self.k2, self.k = k1, k2, k
#         self.Npoints = Npoints
#         # if len(init) == 0:
#         #     init = [0.4, 0.6, -0.2, 0.3]
#         [self.p10, self.p20, self.q10, self.q20] = init
#         # self.q1 = np.zeros(Npoints)
#         # self.q2 = np.zeros(Npoints)
#         # self.p1 = np.zeros(Npoints)
#         # self.p2 = np.zeros(Npoints)
#         self.update()
        
#     def update(self):
#         p1, p2, q1, q2 = _std_map4d(self.q10, self.q20, self.p10, self.p20,
#                                     self.Npoints, self.k1, self.k2, self.k)
#         self.p1, self.p2, self.q1, self.q2 = p1, p2, q1, q2
#         pass

# class OrbMan(object):
#     def __init__(self, initArray=[], k1=0.0, k2=0.0, k=0.0,
#                  Nplot=100, Npoints=100):
#         self.orbits = []
#         self.k1, self.k2, self.k = k1, k2, k
#         self.Npoints = Npoints
#         self.Nplot = Nplot
#         shapeInitArray = np.shape(initArray)[0]
#         if shapeInitArray == 0:
#             print("No initial values given")
#         elif shapeInitArray == 1:
#             orbit = self.new_orbit(initArray)
#             self.append(orbit)
#         elif shapeInitArray == 2:
#             for init in initArray:
#                 orbit = self.new_orbit(init) 
#                 self.append(orbit)
#         else:
#             print("Invalid shape of the initial array!")
#             raise TypeError
        
#     def append(self, orbit):
#         self.orbits.append(orbit)
        
#     def new_orbit(self, init=[0.0, 0.0, 0.5, 0.5]):
#         orbit = Orbit4d(init, self.k1, self.k2, self.k, self.Npoints)
#         return orbit
        
#     def get_orbit(self, indx=-1):
#         if len(self.orbits) > 0:
#             return self.orbits[indx]
#         return self.new_orbit()
        
#     def clear_all(self):
#         self.orbits = []
        
#     def save_h5(self, fname):
#         """alternative from https://stackoverflow.com/questions/20928136/
#         input-and-output-numpy-arrays-to-h5py"""
#         # data = np.zeros((4*len(self.orbits), ))
#         orbit_manager = orbit_manager_from_list(self.orbits, Mapping)
        # if fname.endswith(".h5"):
        #     fname = fname[:-3]
#         orbit_manager.save_h5(fname)
#         # with h5py.File(fname + ".h5", 'w') as hf:
#         #     hf.create_dataset(fname,  data=self.orbits)
        
#     def load_h5(self, fname):
#         if fname.endswith(".h5"):
#             fname = fname[:-3]
#         orbit_manager = OrbitManager.load_h5(fname + ".h5")
#         self.orbits = list_from_orbit_manager(orbit_manager)
#         # with h5py.File(fname + ".h5", 'r') as hf:
#         #     data = hf[fname][:]   
    
def generate_orbit_manager_filename(orb_man):
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_Data = "CP_Bachelor\\WBA_Python\\DataFiles\\"
    PATH = PATH_TP + PATH_Data
    k1, k2, k = str(orb_man.k1), str(orb_man.k2), str(orb_man.k)
    k1, k2, k = k1.replace(".","_"), k2.replace(".","_"), k.replace(".","_")
    N = str(orb_man.Npoints)
    timestamp = str(int(time.time()))
    fname = PATH + "OrbMan" + "N" + N + "k" + k1 + "k" + k2 + "k" + k
    fname += "_T" + timestamp
    return fname

def mayavi_plot3d(orbit_list, Nplot, key=3):
    # mlab.options.backend = 'envisage'
    mfig = mlab.figure(size=(1680, 945), bgcolor=(1,1,1), fgcolor=(0,0,0))
    indx = (np.arange(4) + key) % 4
    for orbit in orbit_list:
        p1, p2 = orbit.points[:Nplot, 0], orbit.points[:Nplot, 1]
        q1, q2 = orbit.points[:Nplot, 2], orbit.points[:Nplot, 3]
        plist = [q1, p1, q2, p2]
        mlab.points3d(plist[indx[0]], plist[indx[1]], plist[indx[2]],
                      plist[indx[3]], colormap="rainbow", scale_factor=0.002,
                      figure=mfig, scale_mode='none')
    mlab.outline()
    labels = [r"$q_1$", r"$p_1$", r"$q_2$", r"$p_2$"]
    mlab.axes(xlabel=labels[indx[0]], ylabel=labels[indx[1]],
              zlabel=labels[indx[2]])
    mlab.colorbar(title=labels[indx[3]], orientation='vertical')
    mlab.show()
    return #mfig

def _keypress(event, ax, orb_man):
    Nplot = orb_man.Nplot
    if event.key in ['alt+1', 'alt+2', 'alt+3', 'alt+4']:
        orbit_list = list_from_orbit_manager(orb_man)
        key = int(event.key[-1])
        return mayavi_plot3d(orbit_list, Nplot, key)
    if event.key in ['ctrl+1', 'ctrl+2', 'ctrl+3', 'ctrl+4']:
        orbit_list = [orb_man.get_orbit()]
        key = int(event.key[-1])
        return mayavi_plot3d(orbit_list, Nplot, key)
    
    if event.key == ' ':
        count = 0
        for grp in orb_man.groups:
            count += len(grp.orbits)
        print("Current number of orbits: ", count)
    if event.key == 'h':
        print("Saving current orbits, please enter filename:")
        fname = generate_orbit_manager_filename(orb_man)
        if fname == "":
            print("Aborted saving...")
            return
        if not fname.endswith(".h5"):
            fname += ".h5"
        orb_man.save_h5(fname)
        print("Wrote:", fname)
    if event.key == 'd':
        print("Clear plots and orbit manager...")
        for i in range(len(ax)):
            ax[i].lines = []
        orb_man.remove_all()
    event.canvas.draw()
    
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

def _mouse_click4d(event, ax, orb_man, Narr, _action, colors, args=()):
    """
    _mouse_click takes arguments (event, ax, K, Npoints, Narr, _action, args)
    _action takes arguments (ax, qpos, ppos, K, Npoints, Narr, *args)
    """
    mode = event.canvas.toolbar.mode
    
    indx = -1
    for i in range(len(ax)):
        if event.inaxes == ax[i]:
            indx = i
            break
        
    if mode != '' or indx == -1:
        # print("Mouse click has to be inside a plot pane!")
        return
        
    if event.button == 1 and indx in [0, 2]:   
        prevOrbitInit = orb_man.get_orbit()._get_initial_point()
        if indx == 0:
            q10, p10 = event.xdata, event.ydata
            q20, p20 = prevOrbitInit[3], prevOrbitInit[1]
        elif indx == 2:
            q10, p10 = prevOrbitInit[2], prevOrbitInit[0]
            q20, p20 = event.xdata, event.ydata
            
        # print([p10, p20, q10, q20])
        
        orbit = orb_man.new_orbit([p10, p20, q10, q20])
        orb_man.add_orbit(orbit)
        c = colors.get_color()
        _action(ax, indx, orbit.points, orb_man.Nplot, Narr, args, c=c)
        
    if event.button == 2:
        print("Clear plots and orbit manager...")
        for i in range(len(ax)):
            ax[i].lines = []
        orb_man.remove_all()
        
    if event.button == 3:
        ax[indx].lines = []
        
    event.canvas.draw()

def compare_conv4d(ax, orbit, Narr, MapToCircle, thresh, mapMode, NaffLimit,
                   _dig=16, freq1=None, freq2=None, ShowLegend=1, 
                   SetTitle=1, fs=12, lfs=10, c=None, lw=1.0, alphaNaff=0.6,
                   UseMarkers=None, AssertNaffEqualWBA=0, lwnaff=1.5,
                   WBAOnly=0, AssertNu1Nu2Order=0, AssertNuLess05=0):
    p1, p2 = orbit[:, 0], orbit[:, 1]
    q1, q2 = orbit[:, 2] - 0.5, orbit[:, 3] - 0.5
    # print(np.mean(q1), np.mean(q2))
    t1 = time.perf_counter()
    if mapMode == 'torus4d':
        # orbit = WBA_core.transform_nd_torus(np.array([p1, p2, q1, q2]))
        # orbit = WBA_core.sort_by_extent(orbit, thresh).T
        freqNaff1, freqNaff2 = WBA_tools._Naff4d(Narr, orbit)
    else:
        freqNaff1 = WBA_tools._Naff(Narr, q1, p1, MapToCircle)
        freqNaff2 = WBA_tools._Naff(Narr, q2, p2, MapToCircle)
    t2 = time.perf_counter()
    if mapMode == 'torus4d':
        freqWBA1, freqWBA2 = \
            np.abs(WBA_core.WBA_torus4d(np.array([p1, p2, q1, q2]), Narr))
    else:
        freqWBA1 = np.abs(WBA_core.WBA(q1, p1, Narr, thresh=thresh,
                                       mapMode=mapMode))
        freqWBA2 = np.abs(WBA_core.WBA(q2, p2, Narr, thresh=thresh,
                                       mapMode=mapMode))
        
    if AssertNuLess05:
        indx1 = (freqWBA1 > 0.5)
        indx2 = (freqWBA2 > 0.5)
        freqWBA1[indx1] = 1 - freqWBA1[indx1]
        freqWBA2[indx2] = 1 - freqWBA2[indx2]
        
    t3 = time.perf_counter()
    if mapMode == 'none':
        freqWBA1 %= 1.0
        freqWBA2 %= 1.0
    
    if Narr[-1] > 1.5*Narr[-2] and len(Narr) > 49:
        if NaffLimit:
            freq1 = freqNaff1[-1]
            freq2 = freqNaff2[-1]
            print(f"Using Naff for {Narr[-1]} as true frequency " + 
                  f"f1={freq1} and f2={freq2}.")
        else:
            freq1 = freqWBA1[-1]
            freq2 = freqWBA2[-1]
            print(f"Using WBA for {Narr[-1]} as true frequency " + 
                  f"f1={freq1} and f2={freq2}.")
    if freq1 == None:
        freqWBAlimit1 = freqWBA1[-1]
        freqNafflimit1 = freqNaff1[-1]
    else:
        freqWBAlimit1 = freq1
        freqNafflimit1 = freq1
    if freq2 == None:
        freqWBAlimit2 = freqWBA2[-1]
        freqNafflimit2 = freqNaff2[-1]
    else:
        freqWBAlimit2 = freq2
        freqNafflimit2 = freq2
        
    indxWBA1 = (np.abs(freqWBAlimit1 - freqWBA1) > 
                np.abs(1 - freqWBA1 - freqWBAlimit1))
    indxWBA2 = (np.abs(freqWBAlimit2 - freqWBA2) > 
                np.abs(1 - freqWBA2 - freqWBAlimit2))
    indxNaff1 = (np.abs(freqNafflimit1 - freqNaff1) >
                np.abs(1 - freqNaff1 - freqNafflimit1))
    indxNaff2 = (np.abs(freqNafflimit2 - freqNaff2) >
                np.abs(1 - freqNaff2 - freqNafflimit2))
    freqWBA1[indxWBA1] = 1 - freqWBA1[indxWBA1]
    freqWBA2[indxWBA2] = 1 - freqWBA2[indxWBA2]
    freqNaff1[indxNaff1] = 1 - freqNaff1[indxNaff1]
    freqNaff2[indxNaff2] = 1 - freqNaff2[indxNaff2]
    
    if AssertNaffEqualWBA:
        indx = (np.abs(freqNaff1 - freqWBA1) > 
                np.abs(1 - freqNaff1 - freqWBA1))
        freqNaff1[indx] = 1 - freqNaff1[indx]
        if indx[-1]:
            freqNafflimit1 = 1 - freqNafflimit1
        indx = (np.abs(freqNaff2 - freqWBA2) > 
                np.abs(1 - freqNaff2 - freqWBA2))
        freqNaff2[indx] = 1 - freqNaff2[indx]
        if indx[-1]:
            freqNafflimit2 = 1 - freqNafflimit2
            
    # fix the order of nu1 and nu2
    if AssertNu1Nu2Order:
        if freqNafflimit2 > freqNafflimit1:
            print("Reordering Naff frequencies")
            freqNafflimit2, freqNafflimit1 = freqNafflimit1, freqNafflimit2
            freqNaff2, freqNaff1 = freqNaff1, freqNaff2
        # print(freqWBAlimit2, freqWBAlimit1)
        if freqWBAlimit2 > freqWBAlimit1:
            print("Reordering WBA frequencies")
            freqWBAlimit2, freqWBAlimit1 = freqWBAlimit1, freqWBAlimit2
            freqWBA2, freqWBA1 = freqWBA1, freqWBA2
        #     print(freqWBAlimit2, freqWBAlimit1)
        # print(freqWBA1[-1], freqWBA2[-1], freqNaff1[-1], freqNaff2[-1])
        
    WBAdiff1 = np.abs(freqWBAlimit1 - freqWBA1)
    WBAdiff2 = np.abs(freqWBAlimit2 - freqWBA2)
    Naffdiff1 = np.abs(freqNafflimit1 - freqNaff1)
    Naffdiff2 = np.abs(freqNafflimit2 - freqNaff2)
    
    WBAdiff1[WBAdiff1 < 1e-16] = 1e-16
    Naffdiff1[Naffdiff1 < 1e-16] = 1e-16
    WBAdiff2[WBAdiff2 < 1e-16] = 1e-16
    Naffdiff2[Naffdiff2 < 1e-16] = 1e-16
    
    title1 = (f"Latest WBA in {round((t3-t2) * 1e3, 3)} ms and " +
              f"Latest Naff in {round((t2-t1) * 1e3, 3)} ms\n")
    labelWBA1 = (r'$\nu_{1,\mathrm{WBA}}$ = ' 
                 + str(round(freqWBA1[-1], _dig)))
    labelNaff1 = (r'$\nu_{1,\mathrm{Naff}}$ = ' 
                  + str(round(freqNaff1[-1], _dig)))
    labelWBA2 = (r'$\nu_{2,\mathrm{WBA}}$ = ' 
                 + str(round(freqWBA2[-1], _dig)))
    labelNaff2 = (r'$\nu_{2,\mathrm{Naff}}$ = ' 
                  + str(round(freqNaff2[-1], _dig)))
    title1 += labelWBA1 + " and " + labelNaff1
    title2 = labelWBA2 + " and " + labelNaff2
    
    if freq1 != None:
        labelWBA1 = None
        labelNaff1 = f'$\\nu = {round(freq1, _dig)}$'
    if freq2 != None:
        labelWBA2 = None
        labelNaff2 = f'$\\nu = {round(freq1, _dig)}$'
        
    if type(UseMarkers) != type(None):
        marker, ms, mew = UseMarkers
        ax[0].plot(Narr[:-1], WBAdiff1[:-1], lw=lw, c=c, marker=marker, 
                   ms=ms, mew=mew, label=labelWBA1)
        ax[1].plot(Narr[:-1], WBAdiff2[:-1], lw=lw, c=c, marker=marker, 
                   ms=ms, mew=mew, label=labelWBA2)
    else:
        ax[0].plot(Narr[:-1], WBAdiff1[:-1], lw=lw, c=c, label=labelWBA1)
        ax[1].plot(Narr[:-1], WBAdiff2[:-1], lw=lw, c=c, label=labelWBA2)
    if WBAOnly: 
        labelNaff1, labelNaff2 = None, None
    # else:
    ax[0].plot(Narr[:-1], Naffdiff1[:-1], 
               lw=lwnaff, ls='--', c=c, alpha=alphaNaff, label=labelNaff1)
    ax[1].plot(Narr[:-1], Naffdiff2[:-1], 
               lw=lwnaff, ls='--', c=c, alpha=alphaNaff, label=labelNaff2)
    if SetTitle:
        ax[0].set_title(title1, fontsize=fs)
        ax[1].set_title(title2, fontsize=fs)
    if ShowLegend:
        ax[0].legend(fontsize=lfs)
        ax[1].legend(fontsize=lfs)
    return freqWBA1[-1], freqWBA2[-1], freqNaff1[-1], freqNaff2[-1]

def _freq_N_4d(points, N, mapMode, mode):
    p1, p2, q1, q2 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    if mode == 'WBA':
        freq1 = WBA_core._WBA_single_wrapper(q1, p1, mapMode)
        freq2 = WBA_core._WBA_single_wrapper(q2, p2, mapMode)
    elif mode == 'Naff':
        MapToCircle = (mapMode == 'none')
        freq1 = compute_freq(q1, p1, MapToCircle)
        freq2 = compute_freq(q2, p2, MapToCircle)
    return freq1, freq2

def _grid4d_N_looper(i, args):
    [p10, p20, q10, q20, Npoints, mapMode, mode, p2count, mapN] = args
    freqVals = np.zeros((2, p2count))
    for j in range(p2count):
        p20val = p20[j]
        points = mapN([p10[i], p20val, q10, q20], Npoints).points
        freqVals[:, j] = _freq_N_4d(points, Npoints, mapMode, mode)
    return freqVals, i

def freqgrid_4d(q10, q20, p1min, p1max, p2min, p2max, p1count=100,
                p2count=None, k1=0.5, k2=0.7, k=0.01, Npoints=2**10,
                mapMode='none', mode='WBA'):
    if p2count == None:
        p2count = p1count
    mapN = Mapping(k1, k2, k).mapN
    p10 = np.linspace(p1min, p1max, p1count)
    p20 = np.linspace(p2min, p2max, p2count)
    freqGrid = np.zeros((2, p1count, p2count))
    t1 = time.time()
    args = [p10, p20, q10, q20, Npoints, mapMode, mode, p2count, mapN]
    for i in range(p1count):
        freqGrid[:, i, :] = _grid4d_N_looper(i, args)[0]
    t2 = time.time()
    print(f"Frequency calculation done in {t2 - t1} seconds")
    return freqGrid

def _absdiff_N2N_4d(points, N, mapMode, mode):
    p1, p2, q1, q2 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    if mode == 'WBA':
        freq1_N = WBA_core._WBA_single_wrapper(q1[:N], p1[:N],
                                              mapMode=mapMode)
        freq1_2N = WBA_core._WBA_single_wrapper(q1[N:], p1[N:],
                                               mapMode=mapMode)
        freq2_N = WBA_core._WBA_single_wrapper(q2[:N], p2[:N],
                                               mapMode=mapMode)
        freq2_2N = WBA_core._WBA_single_wrapper(q2[N:], p2[N:],
                                                mapMode=mapMode)
    elif mode == 'Naff':
        MapToCircle = (mapMode != 'arctan2')
        freq1_N = compute_freq(q1[:N], p1[:N], MapToCircle)
        freq1_2N = compute_freq(q1[N:], p1[N:], MapToCircle)
        freq2_N = compute_freq(q2[:N], p2[:N], MapToCircle)
        freq2_2N = compute_freq(q2[N:], p2[N:], MapToCircle)
        
    return np.abs(freq1_N - freq1_2N), np.abs(freq2_N - freq2_2N)

def _absdiff_Nlimit_4d(points, N, mapMode, mode):        
    p1l, p2l, q1l, q2l = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    p1, p2, q1, q2 = points[:N, 0], points[:N, 1], points[:N, 2], points[:N, 3]
    if mode == 'WBA':
        freq1l = WBA_core._WBA_single_wrapper(q1l, p1l, mapMode=mapMode)
        freq2l = WBA_core._WBA_single_wrapper(q2l, p2l, mapMode=mapMode)
        freq1 = WBA_core._WBA_single_wrapper(q1, p1, mapMode=mapMode)
        freq2 = WBA_core._WBA_single_wrapper(q2, p2, mapMode=mapMode)
    elif mode == 'Naff':
        MapToCircle = (mapMode != 'arctan2')
        freq1l = compute_freq(q1l, p1l, MapToCircle)
        freq2l = compute_freq(q2l, p2l, MapToCircle)
        freq1 = compute_freq(q1, p1, MapToCircle)
        freq2 = compute_freq(q2, p2, MapToCircle)
    return np.abs(freq1 - freq1l), np.abs(freq2 - freq2l)

def _logImshowWrapper(data, k1, k2, k, Npoints, q10, q20, 
                      p1min, p1max, p2min, p2max, p1count, p2count=None, 
                      Nlimit=None, mode='WBA', cmap='viridis_r', 
                      mapType='AbsDiffGrid', climits=None, titlestr=''):
    if p2count == None:
        p2count = p1count
        
    p10 = np.linspace(p1min, p1max, p1count)
    p20 = np.linspace(p2min, p2max, p2count)
    absDiffLog1, absDiffLog2 = data[0], data[1]
    title = (f"$k_1={k1}$, $k_2={k2}$, $k={k}$, orbit length $N={Npoints}$ \
for $(q_1,q_2)={q10,q20}$ and $(p_1,p_2)$ on a (${p1count}\
\\times{p2count}$) grid\n")
    if titlestr != '':
        title += titlestr
    elif mapType == 'AbsDiffGrid':
        if Nlimit == None:
            title += ("Color corresponds to $-\log_{{10}}|\\nu_{{[0,N-1]}}\
-\\nu_{{[N,2N-1]}}|$")
        else:
            title += ("Color corresponds to $-\log_{10}|\\nu_N\
-\\nu_{N_{lim}}|$" + f" with $N_{{lim}}={Nlimit}$" )
    elif mapType == 'FreqGrid':
        title += "Color corresponds to frequency"
    title += " using " + mode
    fig, ax = plt.subplots(1, 3, figsize=(15, 10), 
                           gridspec_kw={"width_ratios":[1, 1, 0.07]})
    plt.suptitle(title, fontsize=14)
    if type(climits) == type(None):
        minval = min(np.min(absDiffLog1), np.min(absDiffLog2))
        maxval = max(np.max(absDiffLog1), np.max(absDiffLog2))
    else:
        [minval, maxval] = climits
    for i in range(2):
        ax[i].set_xlabel(r"$p_1$", fontsize=14)
        ax[i].set_xlabel(r"$p_2$", fontsize=14)
        ax[i].set_title(f"$\\nu_{i+1}$", fontsize=14)
    img1 = WBA_tools.imshow_grid(ax[0], p10, p20, absDiffLog1, cmap=cmap)
    img2 = WBA_tools.imshow_grid(ax[1], p10, p20, absDiffLog2, cmap=cmap)
    
    cmap = plt.cm.ScalarMappable(cmap=cmap)
    for img in [img1, img2, cmap]:
        img.set_clim(minval, maxval)
    plt.tight_layout()
    ipos = InsetPosition(ax[1], [1.07, 0, 0.07, 1]) 
    ax[2].set_axes_locator(ipos)
    fig.colorbar(img2, cax=ax[2], ax=ax[:1])
    return

def _grid4d_N2N_looper(i, args):
    [p10, p20, q10, q20, Npoints, mapMode, mode, p2count, mapN] = args
    absDiffVals = np.zeros((2, p2count))
    for j in range(p2count):
        p20val = p20[j]
        points = mapN([p10[i], p20val, q10, q20], 2*Npoints).points
        absDiffVals[:, j] = _absdiff_N2N_4d(points, Npoints, mapMode, mode)
    return absDiffVals, i

def _grid4d_Nlimit_looper(i, args):
    [p10, p20, q10, q20, Npoints, mapMode, mode, p2count, mapN, Nlimit] = args
    absDiffVals = np.zeros((2, p2count))
    for j in range(p2count):
        p20val = p20[j]
        points = mapN([p10[i], p20val, q10, q20], Nlimit).points
        absDiffVals[:, j] = _absdiff_Nlimit_4d(points, Npoints, mapMode, mode)
    return absDiffVals, i

def _freq_grid4d(points1, points2, thresh=1e-5, mapMode='none'):
    points1[2, :] -= 0.5; points1[3, :] -= 0.5
    points2[2, :] -= 0.5; points2[3, :] -= 0.5
    if mapMode == 'torus4d':
        print("Using 4d torus major axis transformation...")
        _freq_WBA = WBA_core._WBA_torus4d_multi
        freq11, freq12 = _freq_WBA(points1, thresh=thresh)
        freq21, freq22 = _freq_WBA(points2, thresh=thresh)
    else:
        print("Standard transformation...")
        _freq_WBA = WBA_core._WBA_multi_parallel_wrapper
        freq11 = _freq_WBA(points1[2, :], points1[0, :], mapMode=mapMode)
        freq12 = _freq_WBA(points1[3, :], points1[1, :], mapMode=mapMode)
        freq21 = _freq_WBA(points2[2, :], points2[0, :], mapMode=mapMode)
        freq22 = _freq_WBA(points2[3, :], points2[1, :], mapMode=mapMode)
    # print([np.mean(val) for val in [freq11, freq12, freq21, freq22]])
    return np.array([np.abs(freq11 - freq21), np.abs(freq12 - freq22)])

def absdiff_grid4d(q10, q20, p1min, p1max, p2min, p2max, p1count=100,
                   p2count=None, k1=0.5, k2=0.7, k=0.01, Npoints=2**10,
                   mapMode='none', mode='WBA', Nlimit=None, thresh=1e-5):    
    if p2count == None:
        p2count = p1count
        
    # mapN = Mapping(k1, k2, k).mapN
    mapNarray = Mapping4dCyl(k1, k2, k).mapNarray
    
    p10 = np.linspace(p1min, p1max, p1count)
    p20 = np.linspace(p2min, p2max, p2count)
    p10arr = np.outer(p10, np.ones(p2count)).flatten()
    p20arr = np.outer(np.ones(p1count), p20).flatten()
    q10arr = np.full(p1count * p2count, q10)
    q20arr = np.full(p1count * p2count, q20)
    # absDiffGrid1 = np.zeros((p1count, p2count))
    # absDiffGrid2 = np.zeros((p1count, p2count))
    absDiffGrid = np.zeros((2, p1count, p2count))
    t1 = time.time()
    # args = [p10, p20, q10, q20, Npoints, mapMode, mode, p2count, mapN]
    # p1range = range(p1count)
    
    # with cft.ProcessPoolExecutor(max_workers=4) as exe:            
    #     for values, i in zip(exe.map(_grid4d_Nlimit_looper, p1range, args),
    #                          p1range):
    #         absDiffGrid[:, i, :] = values
    
    if Nlimit == None:
        p1, p2, q1, q2 = mapNarray(p10arr, p20arr, q10arr, q20arr, 2*Npoints)
        points2 = np.array([p1[Npoints:], p2[Npoints:], 
                            q1[Npoints:], q2[Npoints:]])
        # from multiprocessing import Pool
        # with Pool(processes=4) as pool:
        #     for values, i in pool.imap_unordered(_grid4d_N2N_looper, 
        #                                          (range(p1count), args)):
        #         absDiffGrid[:, i, :] = values
                
        # with cft.ThreadPoolExecutor() as exe:
        #     cftTuples = [exe.submit(_grid4d_N2N_looper, i, args)
        #                     for i in range(p1count)]
        
        # for i in range(p1count):
        #     absDiffGrid[:, i, :] = _grid4d_N2N_looper(i, args)[0]
    else:
        p1, p2, q1, q2 = mapNarray(p10arr, p20arr, q10arr, q20arr, Nlimit)
        points2 = np.array([p1, p2, q1, q2])
    points1 = np.array([p1[:Npoints], p2[:Npoints], 
                        q1[:Npoints], q2[:Npoints]])
    t2 = time.time()
    print(f"Orbits computed in {t2 - t1} seconds.")
    absDiffGrid = _freq_grid4d(points1, points2, thresh, mapMode)
    # return absDiffGrid
    absDiffGrid = absDiffGrid.reshape((2, p1count, p2count))
        # for i in range(p1count):
        #     absDiffGrid[:, i, :] = _grid4d_Nlimit_looper(i, args)[0]
        
        
        # args.append(Nlimit)
        # with cft.ThreadPoolExecutor() as exe:
        #     cftTuples = [exe.submit(_grid4d_Nlimit_looper, i, args)
        #                 for i in range(p1count)]
            
    # for tuples in cft.as_completed(cftTuples):
    #     [values, i] = tuples.result()
    #     absDiffGrid[:, i, :] = values
    
    t3 = time.time()
    print(f"Frequency computation done in {t3 - t2} seconds.")
            
    absDiffGrid[absDiffGrid < 1e-16] = 1e-16
    # absDiffGrid2[absDiffGrid2 < 1e-16] = 1e-16
    absDiffLog = -np.log10(absDiffGrid)
    # absDiffLog2 = -np.log10(absDiffGrid2)
    
    return absDiffLog#,p1, p2, q1, q2#1, absDiffLog2

def _action_plot4d(ax, indx, orbit, Nplot, Narr, args, c=None):
    [thresh, MapToCircle, mapMode, 
     NaffLimit, ShowLegend, ShowMap, ShowTransform] = args
    p1, p2, q1, q2 = orbit[:, 0], orbit[:, 1], orbit[:, 2], orbit[:, 3]
        
    ax[0].plot(q1[:Nplot], (p1[:Nplot] + 0.5) % 1.0 - 0.5, 
               marker='o', ls='', ms=1, mew=0.5, c=c)
    ax[2].plot(q2[:Nplot], (p2[:Nplot] + 0.5) % 1.0 - 0.5, 
               marker='o', ls='', ms=1, mew=0.5, c=c)
    
    if ShowMap:
        show_map_transform(ax, p1, p2, q1-0.5, q2-0.5, Nplot,
                           ShowTransform, c=c)
    # if ShowMap:
        # q1, q2 = q1 - 0.5, q2 - 0.5
    #     p1, p2, q1, q2 = WBA_core.transform_nd_torus(
    #         np.array([p1, p2, q1, q2]))
    #     phi1, r1 = WBA_core.map_arctan2(q1[:Nplot], 
    #                                     (p1[:Nplot] + 0.5) % 1.0 - 0.5)
    #     phi2, r2 = WBA_core.map_arctan2(q2[:Nplot], 
    #                                     (p2[:Nplot] + 0.5) % 1.0 - 0.5)
    #     pd1, pd2 = phi1[1:] - phi1[:-1], phi2[1:] - phi2[:-1]
    #     ax[4].plot(phi1[1:], pd1, ls='', marker='o', ms=1, mew=0.5)
    #     ax[5].plot(phi2[1:], pd2, ls='', marker='o', ms=1, mew=0.5)
    
    compare_conv4d(ax[1::2], orbit, Narr, thresh=thresh,
                   mapMode=mapMode, MapToCircle=MapToCircle,
                   NaffLimit=NaffLimit, ShowLegend=ShowLegend, c=c)
    
def show_map_transform(ax, p1, p2, q1, q2, Nplot, ShowTransform=0, c=None):
    if ShowTransform:
        points =  np.array([p1, p2, q1, q2])
        points = WBA_core.transform_nd_torus(points)
        # x, y, z, w = WBA_core.find_elliptic_sections(points)
        x, y, z, w = WBA_core.sort_by_extent(points, 0.01)
        ax[4].plot(z[:Nplot], x[:Nplot], ls='', marker='o', ms=1, 
                   mew=0.5, c=c)
        ax[5].plot(w[:Nplot], y[:Nplot], ls='', marker='o', ms=1, 
                   mew=0.5, c=c)
    else:
        q1, q2 = q1 - 0.5, q2 - 0.5
        phi1, r1 = WBA_core.map_arctan2(q1[:Nplot], 
                                        (p1[:Nplot] + 0.5) % 1.0 - 0.5)
        phi2, r2 = WBA_core.map_arctan2(q2[:Nplot], 
                                        (p2[:Nplot] + 0.5) % 1.0 - 0.5)
        pd1, pd2 = phi1[1:] - phi1[:-1], phi2[1:] - phi2[:-1]
        ax[4].plot(phi1[1:], pd1, ls='', marker='o', ms=1, mew=0.5, c=c)
        ax[5].plot(phi2[1:], pd2, ls='', marker='o', ms=1, mew=0.5, c=c)
    pass

def interactive_plot4d(k1=1.0, k2=0.0, k=0.0, Nplot=4096, Nmin=5.0, 
                       Nmax=6.0, NN=2, initArray=[], thresh=1e-5,
                       NmaxLimit=None, mapMode='none', MapToCircle=1,
                       NaffLimit=1, ShowLegend=1, ShowMap=0,
                       ShowTransform=0, fs=12, lfs=10, colors=None,
                       UseMarkers=None, SetTitle=1, _dig=16, WBAOnly=0,
                       AssertNaffEqualWBA=0, lw=1.0, lwnaff=1.5, 
                       fscale=None, AssertNu1Nu2Order=0, AssertNuLess05=0,
                       PresentationFlag=0):
    Narr = WBA_tools.N_arr(Nmin, Nmax, NN)
    if NmaxLimit != None:
        Narr = np.append(Narr, np.uint32(2**NmaxLimit))
        
    mapping = Mapping(k1, k2, k)
    
    # # --- use the slice p_1=0.0, i.e. 3D coordinates (p_2, q_1, q_2):
    # slice_info = SliceInfo(indices=[0], values=[0.0], eps=2e-4,
    #                        labels=mapping.region.labels,
    #                        region=mapping.region)
    
    orb_man = OrbitManager(mapping, Narr[-1], Nplot, initArray,
                            # slice_info=slice_info,
                            group_title="Orbits")
    # orb_man = OrbMan(initArray, k1, k2, k, Nplot, Narr[-1])
    
    _dict = {0: {'xlabel' : r'$q_{1,n}$', 'ylabel' : r'$p_{1,n}$', 
                 'title' : f"$K_1={k1}$"},
             1: {'xlabel' : '$N$', 
                 'ylabel' : r"$|\nu_{1,N_\mathrm{max}} - \nu_{1,N}|$", 
                 #'ylabel' : r"$|\nu_{1} - \nu_{1,N}|$", 
                 'title' : ""},
             2: {'xlabel' : r'$q_{2,n}$', 'ylabel' : r'$p_{2,n}$', 
                 'title' : f"$K_2={k2}$"},
             3: {'xlabel' : '$N$', 
                 'ylabel' : r"$|\nu_{2,N_\mathrm{max}} - \nu_{2,N}|$", 
                 #'ylabel' : r"$|\nu_{2} - \nu_{2,N}|$",
                 'title' : ""},
             4: {'xlabel' : r'$\phi_{1,n}$', 
                 'ylabel' : r'$\phi_{1, n+1}-\phi_{1,n}$', 
                 'title' : r"Arctan2-mapping of $(q_1,p_1)$"},
             5: {'xlabel' : r'$\phi_{2,n}$', 
                 'ylabel' : r'$\phi_{2, n+1}-\phi_{2,n}$', 
                 'title' : r"Arctan2-mapping of $(q_2,p_2)$"},
             6: {'xlabel' : r'$x_{1,n}$', 'ylabel' : r'$x_{2,n}$', 
                 'title' : "Elliptic sections of the orbit"},
             7: {'xlabel' : r'$x_{3,n}$', 'ylabel' : r'$x_{4,n}$',  
                 'title' : r"Second elliptic section"}}
    
    fscalemod = 1.0
    if type(fscale) != type(None):
        fscalemod = 16*fscale / (15 + 3*ShowMap)
        
    if PresentationFlag:
        if mapMode == 'arctan2':
            fig = plt.figure(figsize=(13, 9))
            ax0 = plt.subplot2grid((2, 7), (0, 0), rowspan=1, colspan=3)
            ax1 = plt.subplot2grid((2, 7), (0, 3), rowspan=1, colspan=4)
            ax2 = plt.subplot2grid((2, 7), (1, 0), rowspan=1, colspan=3)
            ax3 = plt.subplot2grid((2, 7), (1, 3), rowspan=1, colspan=4)
            ax = np.array([ax0, ax1, ax2, ax3])
        elif mapMode == 'torus4d':
            fig = plt.figure(figsize=(19, 9))
            ax0 = plt.subplot2grid((2, 10), (0, 0), rowspan=1, colspan=3)
            ax1 = plt.subplot2grid((2, 10), (0, 3), rowspan=1, colspan=4)
            ax2 = plt.subplot2grid((2, 10), (1, 0), rowspan=1, colspan=3)
            ax3 = plt.subplot2grid((2, 10), (1, 3), rowspan=1, colspan=4)
            ax4 = plt.subplot2grid((2, 10), (0, 7), rowspan=1, colspan=3)
            ax5 = plt.subplot2grid((2, 10), (1, 7), rowspan=1, colspan=3)
            ax = np.array([ax0, ax1, ax4, ax2, ax3, ax5])
    else:
        fig, ax = plt.subplots(2, 2+ShowMap, figsize=(15 + 3*ShowMap, 10))
        ax = ax.flatten()
        
    if ShowMap:
        ax[2], ax[3], ax[4] = ax[3], ax[4], ax[2]
        if SetTitle:
            ax[4].set_title(_dict[4 + 2*ShowTransform]['title'], fontsize=fs)
    # ax[0].set_title(f"Parameters $k_1={k1}, k_2={k2}$ and coupling $K={k}$"
    #                 + f" for orbit length {Nplot}", fontsize=fs)
    if SetTitle:
        ax[0].set_title(f"$K_1={k1}, K_2={k2}$, $K={k}$ and {Nplot} points",
                        fontsize=fs)
    for i in [0, 2]:
        ax[i].axis([0, 1, -0.5, 0.5])
    for i in [1, 3]:
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
    for i in range(len(ax)):
        BitOffset = ShowTransform & (i in [4, 5])
        ax[i].set_xlabel(_dict[i + 2*BitOffset]['xlabel'], fontsize=fs)
        ax[i].set_ylabel(_dict[i + 2*BitOffset]['ylabel'], fontsize=fs)
        # ax[i].set_title(_dict[i]['title'], fontsize=fs)
    
    colors = Colors(colors)    
    # plot the orbits as projections in (q1, p1), (p2, q2):
    for orb in orb_man.all_orbits():
        c = colors.get_color()
        p1, p2 = orb.points[:Nplot, 0], orb.points[:Nplot, 1]
        q1, q2 = orb.points[:Nplot, 2], orb.points[:Nplot, 3]
        ax[0].plot(q1, p1, marker='o', ms=1, c=c, ls='', mew=0.5)
        ax[2].plot(q2, p2, marker='o', ms=1, c=c, ls='', mew=0.5)
        if ShowMap:
            show_map_transform(ax, p1, p2, q1-0.5, q2-0.5, Nplot,
                               ShowTransform, c=c)
        compare_conv4d(ax[1::2], orb.points, Narr, MapToCircle, 
                       thresh, mapMode, NaffLimit, c=c, ShowLegend=ShowLegend,
                       UseMarkers=UseMarkers, fs=fs, lfs=lfs, _dig=_dig, 
                       SetTitle=SetTitle, lw=lw, lwnaff=lwnaff, 
                       AssertNaffEqualWBA=AssertNaffEqualWBA, 
                       WBAOnly=WBAOnly, AssertNu1Nu2Order=AssertNu1Nu2Order,
                       AssertNuLess05=AssertNuLess05)       
            
    mouse_click = functools.partial(_mouse_click4d, ax=ax, orb_man=orb_man, 
                                    args=(thresh, MapToCircle, mapMode,
                                          NaffLimit, ShowLegend, ShowMap,
                                          ShowTransform),
                                    colors=colors,
                                    Narr=Narr, _action=_action_plot4d)
    keypress = functools.partial(_keypress, ax=ax, orb_man=orb_man)
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    fig.canvas.mpl_connect('key_press_event', keypress)
    fig.tight_layout()
    # plt.show()    
    return fig, ax

from CPG.naff.utils import torus_dynamics
def rotation_matrix(vec):
    vec += np.array([0, 0, 1])
    norm = np.dot(vec, vec)
    if norm == 0:
        return np.eye(3)
    
    return 2 * np.outer(vec, vec) / norm - np.eye(3)
    
def project_3d_to_2d(points, n=np.array([0, 0, 1])):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix
    -to-align-vector-a-to-vector-b-in-3d (Rodriguez formula)
    'points' is a (3,N)-numpy-array.
        1. transform points into the plane, to which 'n' is the normal vector
           by points -= np.outer(n, np.dot(n, points))   
    """
    points = np.dot(rotation_matrix(n), points)
    q, p = points[0, :], points[1, :]
    return q, p

def inertia_tensor_3d(x,y,z):
    xxSum, yySum, zzSum = np.sum(x*x), np.sum(y*y), np.sum(z*z)
    xySum, xzSum, yzSum = np.sum(x*y), np.sum(x*z), np.sum(y*z)
    inertiaTensor = np.array([[yySum + zzSum, -xySum, -xzSum],
                              [-xySum, xxSum + zzSum, -yzSum],
                              [-xzSum, -yzSum, xxSum + yySum]])
    return inertiaTensor / len(x)

def inertia_tensor_nd(r, ndim=4):
    return (np.sum(r*r) * np.eye(ndim) - r @ r.T) / len(r[0])

def transform_nd_torus(r, ndim=4):
    iTensor = inertia_tensor_nd(r, ndim)
    eigVal, eigVec = np.linalg.eigh(iTensor)
    return eigVec.T @ r

def frequency_transformed_torus4d(p10=0.028, p20=0.0, q10=0.5, q20=0.5,
                                  N=2**14, k1=2.25, k2=3.0, k=1.0,
                                  thresh=0.01, ShowConv=0, RetVal=0,
                                  SetTitle=0, SetLabels=1, fs=12,
                                  TightLayout=1, colors=None, fscale=1,
                                  ShowSortByExtent=0):
    """
    r = np.array(Mapping4dCyl().mapN(*initsThresh[:,0],2**12))
r[2:,:]-=0.5
rn3=(r.T*np.array([1.0,2.0,3.0,4.0])).T
rn3t=WBA_core.WBA_transform_nd_torus(rn3)
arr=rn3t
fig,ax=plt.subplots(2,3)
ax=ax.flatten()
ax[0].scatter(arr[0],arr[1],s=2)
ax[3].scatter(arr[2],arr[3],s=2)
ax[1].scatter(arr[0],arr[2],s=2)
ax[4].scatter(arr[1],arr[3],s=2)
ax[2].scatter(arr[0],arr[3],s=2)
ax[5].scatter(arr[1],arr[2],s=2)
    """
    colors = Colors(colors)
    ms = 1  #markersize
    p1, p2, q1, q2 = Mapping4dCyl(k1, k2, k).mapN(p10, p20, q10, q20, N)
    fig, ax = plt.subplots(2, 4, figsize=(16*fscale, 9*fscale))
    ax = ax.flatten()
    def verbose_r_test(p1,p2,q1,q2,thresh):
        r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
        ratio1 = np.min(r1sqr) / np.max(r1sqr)
        ratio2 = np.min(r2sqr) / np.max(r2sqr)
        print("r test ", ratio1, ratio2, (ratio1<thresh and ratio2<thresh))
        pass
            
    def sort_by_extent2(x, y, z, w):
        diff = np.array([np.max(x) - np.min(x), np.max(y) - np.min(y),
                          np.max(z) - np.min(z), np.max(w) - np.min(w)])
        indx = np.argsort(diff)
        return np.array([x, y, z, w])[indx]
    
    def _freq_naff(pqArray, MapToCircle=1):
        p1, p2, q1, q2 = pqArray
        return (compute_freq(q1, p1, MapToCircle), 
                compute_freq(q2, p2, MapToCircle))
    
    def swap_rows(arr, x, y):
        arr[np.array([x, y])] = arr[np.array([y, x])] 
    def min_max_ratio(arr):
        return np.min(arr) / np.max(arr)
    def sort_by_extent_verbose(points):
        p1sqr, q1sqr = points[0, :100]**2, points[2, :100]**2
        p2sqr, q2sqr = points[1, :100]**2, points[3, :100]**2
        r11 = min_max_ratio(p1sqr + q1sqr)
        r12 = min_max_ratio(p2sqr + q2sqr)
        r21 = min_max_ratio(p1sqr + p2sqr)
        r22 = min_max_ratio(q1sqr + q2sqr)
        r31 = min_max_ratio(p1sqr + q2sqr)
        r32 = min_max_ratio(q1sqr + p2sqr)
        ratios = np.array([r11, r12, r21, r22, r31, r32])
        print(ratios)
        argmax = np.argmax(ratios)
        
        if argmax > 3:
            print("argmax greater 3")
            swap_rows(points, 3, 2)
        elif argmax > 1:
            print("argmax greater 1")
            swap_rows(points, 2, 1)
        return points

    _freq_WBA = WBA_core._WBA_single_arctan2

    Narr = WBA_tools.N_arr(5.0, 13.8, 100)
    verbose_r_test(p1,p2,q1,q2,thresh)
    ax[0].scatter(q1, p1, s=ms, c='k')
    ax[4].scatter(q2, p2, s=ms, c='k')
    freq1WBA = _freq_WBA(q1, p1)
    freq2WBA = _freq_WBA(q2, p2)
    freq1Naff, freq2Naff = _freq_naff(np.array([p1, p2, q1, q2]), 0)
    if ShowConv:
        freq1WBAN = WBA_core._WBA_arctan2(Narr, q1, p1)
        freq2WBAN = WBA_core._WBA_arctan2(Narr, q2, p2)
        freq1NaffN = WBA_tools._Naff(Narr, q1, p1, 0)
        freq2NaffN = WBA_tools._Naff(Narr, q2, p2, 0)
    
    # print(freq1WBA, freq2WBA)
    
    p1t,p2t,q1t,q2t = \
        WBA_core.transform_nd_torus(np.array([p1, p2, q1 - 0.5, q2 - 0.5]))
    import WBA_core_for_FreqSpace
    x,y,z,w=WBA_core_for_FreqSpace.sort_by_extent(np.array([p1t,p2t,q1t,q2t]))
    verbose_r_test(x,y,z,w,thresh)
    
    c = colors.get_color()
    if ShowSortByExtent:
        points = sort_by_extent_verbose(np.array([p1t, p2t, q1t, q2t]))
        fig2, ax2 = plt.subplots(1, 2, figsize=(16, 9))
        ax2[0].scatter(points[0, :], points[2, :], s=ms, color=c)
        ax2[1].scatter(points[1, :], points[3, :], s=ms, color=c)
    
    ax[1].scatter(z, x, s=ms, color=c)
    ax[5].scatter(w, y, s=ms, color=c)
    freq1WBAt1 = WBA_core._WBA_single_arctan2(z, x)
    freq2WBAt1 = WBA_core._WBA_single_arctan2(w, y)
    freq1Nafft1 = compute_freq(z, x, 0)
    freq2Nafft1 = compute_freq(w, y, 0)
    if ShowConv:
        freq1WBANt1 = WBA_core._WBA_arctan2(Narr, z, x)
        freq2WBANt1 = WBA_core._WBA_arctan2(Narr, w, y)
        freq1NaffNt1 = WBA_tools._Naff(Narr, z, x, 0)
        freq2NaffNt1 = WBA_tools._Naff(Narr, w, y, 0)
    # print(freq1WBAt1, freq2WBAt1)
    z, y = y, z
    
    verbose_r_test(x,y,z,w,thresh)
    c = colors.get_color()
    ax[2].scatter(z, x, s=ms, color=c)
    ax[6].scatter(w, y, s=ms, color=c)
    freq1WBAt2 = WBA_core._WBA_single_arctan2(z, x)
    freq2WBAt2 = WBA_core._WBA_single_arctan2(w, y)
    freq1Nafft2 = compute_freq(z, x, 0)
    freq2Nafft2 = compute_freq(w, y, 0)
    if ShowConv:
        freq1WBANt2 = WBA_core._WBA_arctan2(Narr, z, x)
        freq2WBANt2 = WBA_core._WBA_arctan2(Narr, w, y)
        freq1NaffNt2 = WBA_tools._Naff(Narr, z, x, 0)
        freq2NaffNt2 = WBA_tools._Naff(Narr, w, y, 0)
    # print(freq1WBAt2, freq2WBAt2)
    w, z = z, w
    
    if ShowConv:
        ax[3].plot(Narr, np.abs(freq1WBAN - freq1WBA), c='k', lw=1)
        ax[3].plot(Narr, np.abs(freq1NaffN - freq1Naff), c='k', lw=1, ls='--')
        ax[3].plot(Narr, np.abs(freq1WBANt1 - freq1WBAt1), c='b', lw=1)
        ax[3].plot(Narr, np.abs(freq1NaffNt1 - freq1Nafft1), 
                   c='b', lw=1, ls='--')
        ax[3].plot(Narr, np.abs(freq1WBANt2 - freq1WBAt2), c='g', lw=1)
        ax[3].plot(Narr, np.abs(freq1NaffNt2 - freq1Nafft2), 
                   c='g', lw=1, ls='--')
        ax[7].plot(Narr, np.abs(freq2WBAN - freq2WBA), c='k', lw=1)
        ax[7].plot(Narr, np.abs(freq2NaffN - freq2Naff), c='k', lw=1, ls='--')
        ax[7].plot(Narr, np.abs(freq2WBANt1 - freq2WBAt1), c='b', lw=1)
        ax[7].plot(Narr, np.abs(freq2NaffNt1 - freq2Nafft1), 
                   c='b', lw=1, ls='--')
        ax[7].plot(Narr, np.abs(freq2WBANt2 - freq2WBAt2), c='g', lw=1)
        ax[7].plot(Narr, np.abs(freq2NaffNt2 - freq2Nafft2), 
                   c='g', lw=1, ls='--')
    else:
        verbose_r_test(x,y,z,w,thresh)
        c = colors.get_color()
        ax[3].scatter(z, x, s=2, color=c)
        ax[7].scatter(w, y, s=2, color=c)
        freq1WBAt3 = WBA_core._WBA_single_arctan2(z, x)
        freq2WBAt3 = WBA_core._WBA_single_arctan2(w, y)
        freq1Nafft3 = compute_freq(z, x, 0)
        freq2Nafft3 = compute_freq(w, y, 0)
    # if ShowConv:
    #     freq1WBANt3 = WBA_core._WBA_arctan2(Narr, z, q)
    #     freq2WBANt3 = WBA_core._WBA_arctan2(Narr, w, y)
    #     freq1NaffNt3 = compute_freq(Narr, z, x)
    #     freq2NaffNt3 = compute_freq(Narr, w, y)
    # print(freq1WBAt3, freq2WBAt3)
    
    # x, y, z, w = sort_by_extent2(p1t, p2t, q1t, q2t)
    # ax[0].plot(x, y, ms=2, c='purple', marker='x', ls='')
    # ax[1].plot(z, w, ms=2, c='purple', marker='x', ls='')
    # print(WBA_core._WBA_single_arctan2(x,y), 
    #       WBA_core._WBA_single_arctan2(z,w))
    
    if SetTitle:
        ax[0].set_title(f"$\\nu_{{1,WBA}}={freq1WBA}$ \
\n $\\nu_{{1,Naff1D}}={freq1Naff}$")
        ax[1].set_title(f"$\\nu_{{1,WBA}}={freq1WBAt1}$ \
\n $\\nu_{{1,Naff1D}}={freq1Nafft1}$")
        ax[2].set_title(f"$\\nu_{{1,WBA}}={freq1WBAt2}$ \
\n $\\nu_{{1,Naff1D}}={freq1Nafft2}$")

        ax[4].set_title(f"$\\nu_{{2,WBA}}={freq2WBA}$ \
\n $\\nu_{{2,Naff1D}}={freq2Naff}$")
        ax[5].set_title(f"$\\nu_{{2,WBA}}={freq2WBAt1}$ \
\n $\\nu_{{2,Naff1D}}={freq2Nafft1}$")
        ax[6].set_title(f"$\\nu_{{2,WBA}}={freq2WBAt2}$ \
\n $\\nu_{{2,Naff1D}}={freq2Nafft2}$")
    if SetLabels:
        labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
        ax[0].set_xlabel(r"$q_1$", fontsize=fs)
        ax[0].set_ylabel(r"$p_1$", fontsize=fs)
        ax[4].set_xlabel(r"$q_2$", fontsize=fs)
        ax[4].set_ylabel(r"$p_2$", fontsize=fs)
        for ctr, indx in enumerate([[0,1,2,3],[3,1,2,0],[2,1,3,0]]):
            ax[ctr+1].set_xlabel(labels[indx[0]], fontsize=fs)
            ax[ctr+1].set_ylabel(labels[indx[1]], fontsize=fs)
            ax[ctr+5].set_xlabel(labels[indx[2]], fontsize=fs)
            ax[ctr+5].set_ylabel(labels[indx[3]], fontsize=fs)
        
    if ShowConv:
        ax[3].set_title("Convergence for WBA (solid) and Naff (dashed)")
        ax[3].set_xscale('log')
        ax[3].set_yscale('log')
        ax[7].set_xscale('log')
        ax[7].set_yscale('log')
    else:
        if SetTitle:
            ax[3].set_title(f"$\\nu_{{1,WBA}}={freq1WBAt3}$ \
    \n $\\nu_{{1,Naff1D}}={freq1Nafft3}$")
            ax[7].set_title(f"$\\nu_{{2,WBA}}={freq2WBAt3}$ \
    \n $\\nu_{{2,Naff1D}}={freq2Nafft3}$")
    if TightLayout: plt.tight_layout()
    return fig, ax

def plot_torus4d_Naff2D(p10=0.028, p20=0.0, q10=0.5, q20=0.5,
                        N=2**14, k1=2.25, k2=3.0, k=1.0, 
                        thresh=0.005, proj=0, timeSeries=0):
    ms = 1  #markersize
    p1, p2, q1, q2 = Mapping4dCyl(k1, k2, k).mapN(p10, p20, q10, q20, N)
    q1 -= 0.5
    q2 -= 0.5
    points = np.array([p1, p2, q1, q2])
    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    ax = ax.flatten()
    # ax = np.array([ax[0], ax[1], ax[0], ax[2], ax[3], ax[4], ax[3], ax[5]])
    def verbose_r_test(p1,p2,q1,q2,thresh):
        r1sqr, r2sqr = p1**2 + q1**2, p2**2 + q2**2
        ratio1 = np.min(r1sqr) / np.max(r1sqr)
        ratio2 = np.min(r2sqr) / np.max(r2sqr)
        print("r test ", ratio1, ratio2, (ratio1 < thresh or ratio2 < thresh))
        pass
    
    _freq_naff_single = naff_4d
    _freq_naff = WBA_tools._Naff4d
    
    def _freq_WBA_single(points):
        p1, p2, q1, q2 = points
        return (WBA_core._WBA_single_arctan2(q1, p1), 
                WBA_core._WBA_single_arctan2(q2, p2))
    def _freq_WBA(N, points):
        p1, p2, q1, q2 = points
        return (WBA_core._WBA_arctan2(N, q1, p1), 
                WBA_core._WBA_arctan2(N, q2, p2))

    Narr = WBA_tools.N_arr(5.0, 13.8, 100)
    verbose_r_test(p1,p2,q1,q2,thresh)
    ax[0].scatter(q1, p1, s=ms, c='k')
    ax[4].scatter(q2, p2, s=ms, c='k')
    freq1WBA, freq2WBA = _freq_WBA_single(points)
    freq1Naff, freq2Naff = _freq_naff_single(points.T, proj=proj)
    freq1WBAN, freq2WBAN = _freq_WBA(Narr, points)
    freq1NaffN, freq2NaffN = _freq_naff(Narr, points.T, proj=proj)
        
    pointsT = WBA_core.transform_nd_torus(points)
    x, y, z, w = WBA_core.sort_by_extent(pointsT, thresh)
    verbose_r_test(z, w, x, y, thresh)
    ax[1].scatter(x, z, s=ms, c='b')
    ax[5].scatter(y, w, s=ms, c='b')
    pointsT = np.array([z, w, x, y])
    freq1WBAT, freq2WBAT = _freq_WBA_single(pointsT)
    freq1NaffT, freq2NaffT = _freq_naff_single(pointsT.T, proj=proj)
    freq1WBANT, freq2WBANT = _freq_WBA(Narr, pointsT)
    freq1NaffNT, freq2NaffNT = _freq_naff(Narr, pointsT.T, proj=proj)
    if timeSeries:
        nfig, nax = plt.subplots(1, 4, figsize=(16, 10))
        # times = np.arange(200)
        # for i in range(4):
        #     nax[i].scatter(times, pointsT[i, :200])
        print(_freq_naff_single())
        nax[0].scatter(pointsT[2, :200], pointsT[0, :200], s=2, c='b')
        nax[1].scatter(pointsT[3, :200], pointsT[1, :200], s=2, c='b')
    
    ax[2].scatter(x, y, s=ms, c='g')
    ax[6].scatter(z, w, s=ms, c='g')
    pointsTb = np.array([y, w, x, z])
    freq1WBATb, freq2WBATb = _freq_WBA_single(pointsTb)
    freq1NaffTb, freq2NaffTb = _freq_naff_single(pointsTb.T, proj=proj)
    freq1WBANTb, freq2WBANTb = _freq_WBA(Narr, pointsTb)
    freq1NaffNTb, freq2NaffNTb = _freq_naff(Narr, pointsTb.T, proj=proj)
    
    ax[3].plot(Narr, np.abs(freq1WBAN - freq1WBA), c='k', lw=1)
    ax[3].plot(Narr, np.abs(freq1NaffN - freq1Naff), c='k', lw=1, ls='--')
    ax[3].plot(Narr, np.abs(freq1WBANT - freq1WBAT), c='b', lw=1)
    ax[3].plot(Narr, np.abs(freq1NaffNT - freq1NaffT), 
               c='b', lw=1, ls='--')
    ax[3].plot(Narr, np.abs(freq1WBANTb - freq1WBATb), c='g', lw=1)
    ax[3].plot(Narr, np.abs(freq1NaffNTb - freq1NaffTb), 
               c='g', lw=1, ls='--')
    ax[7].plot(Narr, np.abs(freq2WBAN - freq2WBA), c='k', lw=1)
    ax[7].plot(Narr, np.abs(freq2NaffN - freq2Naff), c='k', lw=1, ls='--')
    ax[7].plot(Narr, np.abs(freq2WBANT - freq2WBAT), c='b', lw=1)
    ax[7].plot(Narr, np.abs(freq2NaffNT - freq2NaffT), 
               c='b', lw=1, ls='--')
    ax[7].plot(Narr, np.abs(freq2WBANTb - freq2WBATb), c='g', lw=1)
    ax[7].plot(Narr, np.abs(freq2NaffNTb - freq2NaffTb), 
               c='g', lw=1, ls='--')
    
    for i in [3, 7]:
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
    
    ax[3].set_title("Convergence for WBA (solid) and Naff (dashed)")
    ax[0].set_title(f"$\\nu_{{1,WBA}}={freq1WBA}$ \
\n $\\nu_{{1,Naff2D}}={freq1Naff}$")
    ax[1].set_title(f"$\\nu_{{1,WBA}}={abs(freq1WBAT)}$ \
\n $\\nu_{{1,Naff2D}}={freq1NaffT}$")
    ax[2].set_title(f"$\\nu_{{1,WBA}}={abs(freq1WBATb)}$ \
\n $\\nu_{{1,Naff2D}}={freq1NaffTb}$")

    ax[4].set_title(f"$\\nu_{{2,WBA}}={freq2WBA}$ \
\n $\\nu_{{2,Naff2D}}={freq2Naff}$")
    ax[5].set_title(f"$\\nu_{{2,WBA}}={abs(freq2WBAT)}$ \
\n $\\nu_{{2,Naff2D}}={freq2NaffT}$")
    ax[6].set_title(f"$\\nu_{{2,WBA}}={abs(freq2WBATb)}$ \
\n $\\nu_{{2,Naff2D}}={freq2NaffTb}$")
    plt.tight_layout()
    plt.show()
    return

def torus4d_signal(nu1=np.sqrt(2)-1, nu2=1.5-np.sqrt(1.25), N=5000,
                   r=2, R=3, n=np.array([0, 0, 1]), alpha=np.pi/4):
    xt,yt,zt = torus_dynamics(nu1, nu2, N, r, R)
    rvec = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]]) @ np.array([xt,yt,zt])
    xt, yt, zt = rvec
    # q, p = project_3d_to_2d(np.array([xt, yt, zt]), n)
    # phi = np.arctan2(q, p) / (2*np.pi)
    # phiDiff = phi[1:] - phi[:-1]
    # return phi[1:], phiDiff
    # rt = np.array([xt, yt, zt])
    # inertiaTensor = np.dot(rt, rt) * np.eye(3) - np.outer(rt, rt)
    inertiaTensor = inertia_tensor_3d(xt, yt, zt)
    eigVal, eigVec = np.linalg.eigh(inertiaTensor)
    
    return eigVal, eigVec

def arctan2debugger(p10=0.028, p20=0.0, q10=0.5, q20=0.5):
    # figure out how WBA returns values greater 1 for some orbits
    mapN = Mapping().mapN
    #p10, p20, q10, q20 = 0.05, 0.0, 0.5, 0.5
    N = 2**14
    points = mapN([p10, p20, q10, q20], N).points
    p1, p2 = points[:, 0], points[:, 1]
    q1, q2 = points[:, 2], points[:, 3]
    phi1, r1 = WBA_core.map_arctan2(q1 - 0.5, (p1 + 0.5) % 1.0 - 0.5)
    phi2, r2 = WBA_core.map_arctan2(q2 - 0.5, (p2 + 0.5) % 1.0 - 0.5)
    pd1, pd2 = phi1[1:] - phi1[:-1], phi2[1:] - phi2[:-1]
    #sol1 = (pd1 % 1.0 + 0.5) % 1.0 - 0.5
    
    """
    p1,p2,q1,q2,phi1,phi2,pd1, pd2=arctan2debugger()
    arr = pd1+0.0
    plt.plot(t,t,c='k',ls='--')
    arr[((phi1[1:]<0) & (arr<phi1[1:]))] += 1
    plt.scatter(phi1[1:],arr, s=2)
    fwba = WBA_core._WBA_single(arr)
    plt.axhline(fwba, c='k')
    """
    return p1,p2,q1,q2,phi1,phi2,pd1, pd2

def compute_freq4d_grid(minval, maxval, counts, Npoints, k1, k2, k, 
                        thresh=0.005, mode='random'):
    if mode == 'random':
        _freq = WBA_tools.freq4d_grid_random
    else:
        _freq = WBA_tools.freq4d_grid
        
    t1 = time.time()
    f1p1, f2p1, f1p2, f2p2 = _freq(minval, maxval, counts, Npoints, 
                                   k1, k2, k, thresh)
    print("Frequency grid computed in ", time.time() - t1)
    return f1p1, f2p1, f1p2, f2p2

def plot_freq4d_grid(fVals, RetVal=0):
    f1p1, f2p1, f1p2, f2p2 = fVals
    diff1, diff2 = np.abs(f1p1 - f1p2), np.abs(f2p1 - f2p2)
    diff1[diff1 < 1e-16] = 1e-16
    diff2[diff2 < 1e-16] = 1e-16
    abl1, abl2 = -np.log10(diff1), -np.log10(diff2)
    indx = ((abl1 < 5) | (abl2 < 5))
    fig, ax = plt.subplots(figsize=(16, 10))
    # ax.axis([0.0, 0.5, 0.0, 0.5])
    f1plot, f2plot = f1p1[~indx], f2p1[~indx]
    # indx1, indx2 = (f1plot > 0.5), (f2plot > 0.5)
    indx = (f2plot > f1plot)
    f1plot[indx], f2plot[indx] = f2plot[indx], f1plot[indx]
    ax.plot(f1plot, f2plot, c='k', ls='', marker='.', ms=1, mew=0)
    plt.show()
    if RetVal:
        return f1plot, f2plot
    pass

def plot_freq4d_examples(numWBA=29000):
    fw1, fw2, fw1p2, fw2p2 = np.loadtxt(PATHFREQ + "WBA2021_freqs.gz")
    if numWBA > 30000:
        fNameWBA = ("FreqSpaceN11p025p025q075q075k225k30k10_" + 
                    "250x100000_1623003293.gz")
        fw1, fw2, fw1p2, fw2p2 = np.loadtxt(PATHFREQ + fNameWBA)
    fn1, fn2 = np.loadtxt(PATHFREQ + "Naff2021_freqs.gz")
    fold1, fold2 = np.loadtxt(PATHFREQ + "Naff2014_freqs.gz")
    fdiff1, fdiff2 = np.abs(fw1 - fw1p2), np.abs(fw2 - fw2p2)
    indxo = ((fdiff1 > 1e-5) | (fdiff2 > 1e-5))
    fig, ax = plt.subplots(figsize=(12, 9))
    # ax.scatter(fold1, fold2, c='b', s=5, linewidths=0)
    # ax.scatter(fn1, fn2, c='g', s=4, linewidths=0)
    # ax.scatter(fw1[~indxo], fw2[~indxo], c='r', s=3, linewidths=0)
    # ax.axis([0.27, 0.31, 0.06, 0.17])
    indxSwap = (fw2 > fw1)
    fw1[indxSwap], fw2[indxSwap] = fw2[indxSwap], fw1[indxSwap]
    ax.plot(fold1, fold2, c='b', ls='', marker='x', mew=1, ms=6)
    ax.plot(fn1, fn2, c='g', ls='', marker='+', mew=2, ms=8)
    ax.plot(fw1[~indxo], fw2[~indxo], c='r', ls='', marker='o', mew=0, ms=4)
    ax.axis([0.283, 0.293, 0.08, 0.1])
    return fig

def _plot_projection_deltaphi(ax, init, k1, k2, k, N=2**14, proj=1, 
                              fs=12, ms=2, c=None, Embed=0, CompFreq=0,
                              nHarm=1, lfs=16, _dig=5, ShowLegend=0):
    points = np.array(Mapping4dCyl(k1, k2, k).mapN(*init, N))
    points[2:, :] -= 0.5
    transPoints = WBA_core.transform_nd_torus(points)
    sortPoints = WBA_core.sort_by_extent(transPoints)
    if proj == 0:
        x, y = sortPoints[np.array([0, 2]), :]
    elif proj == 1:
        x, y = sortPoints[np.array([1, 3]), :]
        
    phi = np.arctan2(x, y) / (2*np.pi)
    phiDiff = phi[1:] - phi[:-1]
    if Embed == 1:
        WBA_core.embedding(phiDiff)
    elif Embed == 2:
        ### Alternative is the modulo operation with additional shift
        phiDiff = (phiDiff + 0.5) % 1.0 - 0.5
        if np.any(phiDiff < -0.25) and np.any(phiDiff > 0.25):
            phiDiff %= 1.0
    
    labelWBA, labelNaff = "", ""
    if CompFreq:
        fwba = abs(WBA_core._WBA_single(phiDiff))
        if fwba > 0.5: fwba = 1 - fwba
        fwba = np.round(fwba, _dig)
        fnaff = naff_4d(points.T, n_harmonics=nHarm, proj=proj)[proj]
        fnaff = np.round(fnaff, _dig)
        labelWBA = f'$\\nu_\\mathrm{{WBA}}={fwba}$'
        labelNaff = f'$\\nu_\\mathrm{{Naff}}={fnaff}$'
    
    ax[0].set_xlabel(f"$x_{1+2*proj}$", fontsize=fs)
    ax[0].set_ylabel(f"$x_{2+2*proj}$", fontsize=fs)
    ax[1].set_xlabel(f"$\\phi_{1+proj}$", fontsize=fs)
    ax[1].set_ylabel(f"$\\Delta\\phi_{1+proj}$", fontsize=fs)
    ax[0].scatter(x, y, s=ms, c='k', label=labelNaff)
    ax[1].scatter(phi[1:], phiDiff, s=ms, c=c, label=labelWBA)
    ax[1].set_xlim(-0.5, 0.5)
    print("WBA: ", fwba, "Naff:", fnaff)
    if ShowLegend:
        for axis in ax:
            axis.legend(fontsize=lfs, handlelength=-0.5, markerscale=0)
    pass

def plot_projection_deltaphi(inits=None, k1=2.25, k2=3.0, k=1.0, N=2**14,
                             proj=1, colors=None, fs=12, ms=2, Embedding=0,
                             CompFreq=1, lfs=16-3, nHarm=1, _dig=5, 
                             initsIndx=np.array([0, 10]), ShowLegend=0):
    """
    inits == [4, number of inits]-shape numpy-Array containging inital values
    proj == 0 or 1, shows projection for nu1 or nu2 respectively.
    Embedding should be a list with entries being 0,1,2.
        0 == no embedding
        1 == simple embedding variant from DasSaiSanYor2017v3
        2 == shift embedding with moduo arithmetic
    """
    if type(inits) == type(None):
        inits = np.loadtxt(PATHFREQ + "InitsNu1_0285_0085_sorted.gz")
        inits = inits[:, initsIndx]
    numInits = inits.shape[1]
    if type(Embedding) == int:
        Embedding = [Embedding] * numInits
    else:
        if len(Embedding) != numInits: 
            print("Wrong Embedding length"); raise IndexError
    # colors = Colors(colors)
    fig, ax = plt.subplots(2, numInits, figsize=(16, 9))
    for i in range(numInits):
        _plot_projection_deltaphi(ax[:, i], inits[:, i], 
                                  k1, k2, k, N, proj, fs,
                                  ms, c='b', Embed=Embedding[i], _dig=_dig,
                                  CompFreq=CompFreq, lfs=lfs, nHarm=nHarm,
                                  ShowLegend=ShowLegend)
    return fig, ax

def test_fsm_4d(N=2**14, n_harm=1): #FIXME
    inits = [#[0.028, 0.0, 0.5, 0.5], [0.02, -0.02, 0.5, 0.49], 
             [-0.047, -0.02, 0.52, 0.49],]
    Narr = WBA_tools.N_arr(5.0, 14.0, 5)
    colors = Colors()
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    for i, init in enumerate(inits):
        points = np.array(Mapping4dCyl().mapN(*init, N))
        points[2:, :] -= 0.5
        freq_wba = WBA_core.WBA_fsm(Narr, points, n_harm=n_harm)
        freq_naff = WBA_tools._Naff4d(Narr, points.T).T
        diff_wba = np.abs(freq_wba - freq_wba[-1, :])
        diff_naff = np.abs(freq_naff - freq_naff[-1, :])
        diff_wba[diff_wba < 1e-16] = 1e-16
        diff_naff[diff_naff < 1e-16] = 1e-16
        c = colors.get_color()
        print(freq_wba[-1, :], freq_naff[-1, :])
        for j in range(2):
            ax[j].plot(Narr[:-1], diff_wba[:-1, j], 
                       lw=1, ls='-', marker='o', ms=3, mew=0,
                       c=c, label=f"WBA: {freq_wba[-1, j]}")
            ax[j].plot(Narr[:-1], diff_naff[:-1, j], ls='--', lw=1, c=c, 
                       label=f"Naff: {freq_naff[-1, j]}")
    for axis in ax:
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.tick_params(labelsize=12)
        axis.legend(fontsize=12)
    return fig, ax

def fname_gen(mapType, q10, q20, p1min, p1max, p2min, p2max, p1count, 
              p2count, Npoints, Nlimit, k1, k2, k, mode, mapMode):
    if p2count == None:
        p2count = p1count
    fname = f"{mapType}_{p1count}x{p2count}"
    if p1count == None:
        fname = f"{mapType}"
    if Nlimit != None:
        fname += f"_Nl{Nlimit}"
    fname += f"_N{Npoints}_K{k1}K{k2}K{k}Q{q10}Q{q20}P{p1min}to{p1max}\
P{p2min}to{p2max}{mode}_"
    return fname.replace(".", "_")

if __name__ == "__main__":
    print(__doc__)
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    PATHDATA = PATH_TP + "CP_Bachelor\\WBA_Python\\DataFiles\\"
    PATHFREQ = PATH_TP + "CP_Bachelor\\WBA_Python\\FreqSpace\\"
    k1, k2, k = 0.5, 0.7, 0.01
    # initArray = [] #; k2=0.0
    # initArray = [[0.3, 0.2, 0.5, 0.5]])   
    # initArray = [[p1, p2, 0.5, 0.5] for p2 in np.linspace(0.02, 0.2, 2)
    #               for p1 in np.linspace(0.05, 0.17, 3)]
    # initArray.reverse()
    # initArray = [#[0.39, 0.38, 0.0, 0.0], [0.25, 0.322, 0.0, 0.0], 
    #               #[0.35, 0.264, 0.0, 0.0], [0.355, 0.36, 0.145, 0.15], 
    #               #[0.29, 0.235, 0.0, 0.0], 
    #               ]
    # initArray = [[0.3636320489192, 0.35280259676167, 0.0, 0.0],
    #              [-0.2589, -0.2623, 0.0, 0.0],
    #              [0.288, -0.17113, 0.0, 0.0]] ;k1=0.5; k2=0.7; k=0.01
    # initArray = [[0.38, 0.14, 0.5, 0.5], [0.27, 0.08, 0.06, 0.5],
    #               [0.355, 0.11, 0.5, 0.5], [0.44, 0.16, 0.5, 0.5], 
    #               [0.27, 0.04, 0.5, 0.5], [0.25, 0.06, 0.5, 0.5]]
    # initArray = [[0.08, 0.05, 0.5, 0.5], #[0.12, -0.05, 0.5, 0.5],
    #              [0.1, 0.0, 0.6, 0.4], [-0.04, -0.1, 0.56, 0.671]]
    # initArray = [[0.028, 0.0, 0.5, 0.5], [0.0707, 0.0, 0.52, 0.5],
    #               [-0.047, 0.0, 0.52, 0.5], 
    #               [0.0007595, -0.02, 0.498507, 0.49], 
    #               [0.02, -0.02, 0.5, 0.49], 
    #               [-0.047, -0.02, 0.52, 0.49],] ;k1=2.25; k2=3.0; k=1.0
    # initArray = [[0.1849, 0.3548, 0.0, 0.0], 
    #              [-0.27, 0.26, 0.0, 0.0],
    #              [0.2347, 0.2246, 0.0, 0.0],
    #              [0.41, -0.27, 0.0, 0.0],
    #              ]#[-0.226, -0.215, 0.0, 0.0]]  ;k1=0.5; k2=0.7; k=0.02
    # initArray = [[0.184, 0.3548, 0.0, 0.0], [0.4, 0.25, 0.0, 0.0], 
    #              [-0.353343, 0.12409, 0.0, 0.0],
    #              [0.333, -0.246, 0.0, 0.0],]  ;k1=0.5; k2=0.7; k=0.03
    # initArray = [[0.3928, 0.2602, 0.0, 0.0], [-0.4, 0.131, 0.0, 0.0],
    #              [0.42, 0.24, 0.0, 0.0], 
    #              [0.333, -0.246, 0.0, 0.0],]  ;k1=0.5; k2=0.7; k=0.04
    
    #explorator/comp/frequency_comp.py for Naff2D
    
    # CPGRepos\explorator\comp\examples\example_torus_frequencies.py ::
    # initArray = [[0.05,0.05,0.55,0.55]]
    # interactive_plot4d(k1=k1, k2=k2, k=k, Nplot=4096, 
    #                     Nmax=14.0, NN=100,
    #                     mapMode='arctan2', MapToCircle=0, 
    #                     initArray=initArray, ShowLegend=0, ShowMap=0,
    #                     ShowTransform=0, AssertNaffEqualWBA=1) 
    
    # initArray = [[0.08090725947182, -0.07995087108008,
    #               0.44041467529610, 0.59718510271600]]
    # frequency_transformed_torus4d(ShowConv=1)
    # plot_torus4d_Naff2D(*initArray[0])
    # plot_torus4d_Naff2D()
    
    # RUNTIMEWARNING 15 minutes!! (reduced by factor 10 with numba maps)
    q10, q20, p1min, p1max, p2min, p2max = 0.5, 0.5, 0.0, 0.2, 0.0, 0.2
    # q10, q20, p1min, p1max, p2min, p2max = 0.1, 0.1, 0.0, 0.5, 0.0, 0.5
    p1count, p2count, Npoints, Nlimit = 50, None, 10, None
    k1, k2, k = 0.5, 0.7, 0.01
    # WBA/Naff -- none/arctan2/torus4d/(decision) -- FreqGrid, AbsDiffGrid
    mode, mapMode, mapType = 'WBA', 'torus4d', 'AbsDiffGrid'
    cmap = 'viridis_r'#'nipy_spectral_r'#'hsv'
    climits = None#[0.29, 0.56]#None
#     if p2count == None:
#         p2count = p1count
#     fname = f"AbsDiffGrid_{p1count}x{p2count}_Nl{Nlimit}_N{Npoints}_K{k1}\
# K{k2}K{k}Q{q10}Q{q20}P{p1min}to{p1max}P{p2min}to{p2max}{mode}_"
#     fname = fname.replace(".", "_")
    fname = fname_gen(mapType, q10, q20, p1min, p1max, p2min, p2max, p1count, 
                      p2count, Npoints, Nlimit, k1, k2, k, mode, mapMode)
    Npoints = 2**Npoints 
    if Nlimit != None:
        Nlimit = 2**Nlimit
    # data = absdiff_grid4d(q10, q20, p1min, p1max, p2min, p2max, p1count, 
    #                       k1=k1, k2=k2, k=k, Npoints=Npoints, Nlimit=Nlimit,
    #                       mapMode=mapMode, mode=mode, thresh=0.005)
    # data = freqgrid_4d(q10, q20, p1min, p1max, p2min, p2max, p1count, 
    #                     k1=k1, k2=k2, k=k, Npoints=Npoints,
    #                     mapMode=mapMode, mode=mode)
    # np.savetxt(PATHDATA + fname + "nu1.gz", data[0])
    # np.savetxt(PATHDATA + fname + "nu2.gz", data[1])
    
    # abl1 = np.loadtxt(PATHDATA + fname + "nu1.gz")
    # abl2 = np.loadtxt(PATHDATA + fname + "nu2.gz")
    # data = [abl1, abl2]
    # _logImshowWrapper(data, k1, k2, k, Npoints, q10, q20, p1min, p1max,
    #                   p2min, p2max, p1count, Nlimit=Nlimit, mode=mode, 
    #                   cmap=cmap, mapType=mapType, climits=climits)
        
    # minval=np.array([-0.19,-0.19,0.4,0.4])
    # maxval=np.array([0.2,0.2,0.6,0.6])
    # counts=np.array([100,100,10,10])
    # Npoints=1024
    # k1,k2,k=2.25,3.0,1.0
    # fVals = compute_freq4d_grid(minval, maxval, counts, Npoints, k1, k2, k, 
    #                             thresh=0.005, mode='random')
    # plot_freq4d_grid(fVals)
    
    
    """
    p1,p2,q1,q2=Mapping4dCyl(1.0,0.7,0.0).mapN(0.05,0.05,0.55,0.55,1024)
    q1-=0.5
    q2-=0.5
    fig,ax=plt.subplots(2,4,figsize=(16,10))
    ax=ax.flatten()
    ax[0].scatter(q1,p1,s=2,c='k')
    ax[4].scatter(q2,p2,s=2,c='k')
    x,y,z,w = WBA_core.transform_nd_torus(np.array([p1,p2,q1,q2]))
    ax[1].scatter(x,y,s=2,c='r')
    ax[5].scatter(z,w,s=2,c='r')
    ax[2].scatter(x,z,s=2,c='g')
    ax[6].scatter(y,w,s=2,c='g')
    ax[3].scatter(x,w,s=2,c='b')
    ax[7].scatter(z,y,s=2,c='b')
    """
        
    # difference between methods:
#     titlestr=(f'Color corresponds to $\log_{{10}}|\\nu_{{[0,N-1]}}\
# -\\nu_{{[N,2N-1]}}| -\log_{{10}}|\\nu_N\
# -\\nu_{{N_{{lim}}}}|$ with $N_{{lim}}={2**14}$')
#     fname2=fname_gen(mapType,q10,q20,p1min,p1max,p2min,p2max,p1count,
#                      p2count,10,14,k1,k2,k,mode,mapMode)
#     abl4 = np.loadtxt(PATHDATA + fname2 + "nu2.gz")   
#     _logImshowWrapper([abl1-abl3,abl2-abl4], k1, k2, k, Npoints, q10, q20,
#                       p1min, p1max, p2min, p2max, p1count, Nlimit=Nlimit,
#                       mode=mode, titlestr=titlestr)
#     _logImshowWrapper([dl1,dl2], k1, k2, k, Npoints, q10, q20, p1min, p1max,
#                   p2min, p2max, p1count, Nlimit=Nlimit, mode=mode, titlestr=f'Color corresponds to $-\log_{{10}}||\\nu_{{[0,N-1]}}\
# -\\nu_{{[N,2N-1]}}| -|\\nu_N\
# -\\nu_{{N_{{lim}}}}||$ with $N_{{lim}}={2**14}$')
        
    
    # mayavi_plot3d(list_from_orbit_manager(obm), obm.Nplot)
    # t = np.linspace(0, 4 * np.pi, 20)
    
    # x = np.sin(2 * t)
    # y = np.cos(t)
    # z = np.cos(2 * t)
    # s = 2 + np.sin(t)
    
    # mlab.points3d(x, y, z, s, colormap="rainbow", scale_factor=.25, 
    #               scale_mode='none')
    
    fig, ax = test_fsm_4d(n_harm=5)
    
    """
orb_man = OrbitManager.load_h5(PATHFREQ + "RicLanBaeKet2014_freqs.h5")
part = 1000
inits = np.loadtxt(PATHFREQ + "Naff2014_inits.gz")
initsPart = inits[:, :part]
freqs = WBA_tools.freq4d_grid_given_initials(initsPart, 
2048, 2.25, 3.0, 1.0)
indx = ((freqs[0] < 0.05) & (freqs[1] < 0.05))
inits_below_thresh = initsPart[:, indx]
initArray = [inits_below_thresh[:,i] for i in range(5)]
frequency_transformed_torus4d(*initArray[0], ShowSortByExtent=1)
freqsThresh = freqs[:,indx]
freqNaff = [np.array(orb.frequencies.freqs).T 
            for grp in orb_man.groups 
            for orb in grp.orbits]
fn1, fn2 = np.concatenate(freqNaff, axis=1)
fn1part, fn2part = fn1[:1000],fn2[:1000]
fn1thresh, fn2thres = fn1part[indx],fn2part[indx]

indx1 = ((fn1 < 0.297) & (fn1 > 0.295))
indx2 = ((fn2 < 0.11) & (fn2 > 0.09))
indx = np.logical_and(indx1, indx2)
inith = inits[:, indx]
    """
    
    # 4d torus transform debugging
    """
inits = np.loadtxt(PATHDATA + "InitHorseshoeResonance.gz")
inits = np.loadtxt(PATHDATA + "InitsLowEdgeFailure.gz")
inits = np.loadtxt(PATHDATA + "InitsEdgyFailure.gz")
points=np.array(Mapping4dCyl().mapN(*inits[:,0],4096))
points[2:, :] -= 0.5
transp = WBA_core.transform_nd_torus(points)
_=frequency_transformed_torus4d(*inits[:,0], ShowSortByExtent=1)
fw1=fwba[0,:]
fw2=fwba[1,:]
fw1[fw1 > 0.5] = 1 - fw1[fw1 > 0.5]
fw2[fw2 > 0.5] = 1 - fw2[fw2 > 0.5]
indx = (fw1 < fw2)
fw1[indx], fw2[indx] = fw2[indx], fw1[indx]
    """