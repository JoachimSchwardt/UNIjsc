#!/usr/bin/env python

"""GUI for investigating trapping as saved from TrappedOrbits class.

Note: for TrappedOrbitsRnd different code is necessary
(not yet existing).

Three plots are created:
P(t), (q, p) of trapped orbit, frequency plot.

Functionality:
- Left klick in plot 1: select closest time,
  display corresponding orbit in plot 2 (q, p) plot
  and nu(t) in plot 3
- Left klick in plot 2: start trajectory
- Right mouse select rectangle in plot 2:
  Highlight (in black) corresponding frequencies in plot 3.
  FIXME This is not working correctly at the moment!
- Right mouse select rectangle in plot 3:
  Higlight points corresponding to the selected frequencies.

TODO:
- continue trajectory
- ?

"""

from __future__ import division, print_function, absolute_import

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib import rcParams, rc
from WBA_Plotting import embed_labels
rcParams["figure.dpi"] = 100

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

global fs, lfs, lw, lwnaff, ms, mew, _dig, tls, phiGold, phiSilver
global phiGoldLabel, phiSilverLabel, alphaNaff, colors, labels, fscale
fs = 21     # fontsize (titles and labels)
lfs = 16    # legend fontsize
lw = 1.0
lwnaff = 1.3  # lw for naff
ms = 3
mew = 1     
_dig = 16   # digits for nu
tls = 18    # tick label size
alphaNaff = 0.5
fscale = 0.75

# from CPG.color.color_mapper import ColorMapper
from CPG.color.specific import color_mapper_trapping
from CPG.contfrac.stern_brocot_tree import SternBrocotTree

from CPG.naff import Naff1D
from WBA_core import _WBA_single_wrapper

from trapping.trapped_orbits import TrappedOrbits


###############################################################################
# Compute frequency
###############################################################################

def compute_freq(x, y, map_onto_circle=True):
    """Compute frequency of the orbit given by the points (x, y).

    Here we only use the points in x direction and map them
    onto the circle.

    """
    if map_onto_circle:
        naff = Naff1D(np.cos(2.0 * np.pi * x) +
                      1j * np.sin(2.0 * np.pi * x))
    else:
        # With the following choice one gets for y=0.3111 with 4096 points:
        #    freq= 0.31109798119404    Diff= -2.018806e-06
        # which is not good enough.
        # (Origin: presumably that the points lie on a straight line giving
        #  a discontinuity the discontinuity when going from 1.0 to 0.0)
        naff = Naff1D(x + 1j * y)

    nu = naff.compute_frequency()
    if nu > 0.5:
        nu = 1.0 - nu
    return nu


def compute_nu_along_orbit(x, y, freq_pts=4096, offset=100, mapMode='none'):
    """..."""
    freq_pts_2 = freq_pts//2
    """
    nu_values = []
    nu_WBA = []
    t_values = []
    t = freq_pts//2
    while t + freq_pts_2 < len(y):
        t_values.append(t)
        nu_values.append(compute_freq(
            x[t-freq_pts_2:t+freq_pts_2],
            y[t-freq_pts_2:t+freq_pts_2]))
        t = t + offset    
    """
        
    t_values = np.arange(freq_pts_2, len(y) - freq_pts_2, offset)
    t_length = len(t_values)
    nu_values = np.zeros(t_length)
    nu_WBA = np.zeros(t_length)
    for i in range(t_length):
        t = t_values[i]
        qval = x[t-freq_pts_2:t+freq_pts_2]
        pval = y[t-freq_pts_2:t+freq_pts_2]
        
        nu_values[i] = compute_freq(qval, pval)
        nu_WBA[i] = _WBA_single_wrapper(qval - 0.5, pval, mapMode) % 1.0

    return t_values, nu_values, nu_WBA


###############################################################################
# GUI to explore trapping
###############################################################################

class GUITrapping(object):                        # pylint: disable=R0902,R0903
    """GUI to explore trapping in 2D maps."""

    def __init__(self,                            # pylint: disable=R0914,R0915
                 fname, show=False, compute_frequencies=True, 
                 freq_pts=4096, offset=100, min_res_lines=10,
                 alpha=1.0):
        """Initialize GUI."""
        self.alpha = alpha
        self.compute_frequencies = compute_frequencies

        # Shift for time interval
        self.freq_pts = freq_pts
        self.offset = offset

        self.min_res_lines = min_res_lines

        # --- Set in self.update_orbit:
        self.t_nu = None
        self.nu = None
        self.nuWBA = None

        # --- Set up stuff
        color_mapper = color_mapper_trapping()
        self.mpl_color_map = color_mapper.matplotlib_colormap()

        self.trapped_orbits = TrappedOrbits.load_h5(fname)

        self.times = self.trapped_orbits.times
        self.p_t = self.trapped_orbits.p_t

        # For 2D iterator maps we have:
        mapping = self.trapped_orbits.mapping
        xlabel = mapping.xlabel
        ylabel = mapping.ylabel
        xmin, xmax, ymin, ymax = mapping.initial_region.x0x1y0y1

        try:
            mapping.mapN
        except AttributeError:
            # For 2D maps, define the right method for iterating the points
            # which are defined as array of dimension (2).
            mapping.mapN = mapping.mapN_point

        mapping.mapN = mapping.mapN_point

        self.mapping = mapping
        self.trapped_orbits.mapping = mapping

        self.ctr = len(self.trapped_orbits.return_times) - 1

        # self.fig = PlotFigure(figsize=(16, 10))
        self.fig = plt.figure(figsize=(14, 10))
        self.axes1 = plt.subplot(2, 2, 1)      
        self.axes2 = plt.subplot(2, 2, 2)
        self.axes3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
        self.ax = [self.axes1, self.axes2, self.axes3]
        plt.subplots_adjust(left=0.075, right=0.93, top=0.96, bottom=0.08)

        # --- Plot 1: P(t)
        # self.axes1 = self.fig.add_subplot(131)
        # self.axes1.set_xlim((0.25, np.max(self.times)))

        min_p_t = min(self.p_t[:-1])
        if min_p_t < 1e-14:
            # Last value above 1e-14?
            inds = self.p_t > 1e-14
            min_p_t = self.p_t[inds][-1]
        points = self.trapped_orbits.trapped_orbits(self.ctr)
        
        max_length = len(points[:, 0])
        self.axes1.set_xlim((1.0, 10**np.ceil(np.log10(max_length))))
        self.axes1.set_ylim((0.9*min_p_t, 1.0))

        self.axes1.set_xlabel(r"$t$", fontsize=fs)
        self.axes1.set_ylabel(r"$P(t)$", fontsize=fs)
        self.axes1.set_xscale("log")
        self.axes1.set_yscale("log")
        self.axes1.plot(self.times, self.p_t, c='k', lw=lw)

        # --- Plot 2: (longest) trapped orbit.
        # self.axes2 = self.fig.add_subplot(132)
        self.axes2.set_xlim(xmin, xmax)
        self.axes2.set_ylim(ymin, ymax)
        xlabel, ylabel = r"$q_n$", r"$p_n$"
        self.axes2.set_xlabel(xlabel, fontsize=fs)
        self.axes2.set_ylabel(ylabel, fontsize=fs)
        self.axes2.axis([0.0, 1.0, -0.5, 0.5])

        self.points = points
        self.scatter = self.axes2.scatter(
            points[:, 0], points[:, 1], marker='o',
            s=4, c=np.arange(len(points)), #FIXME: colors
            cmap=self.mpl_color_map,
            lw=0)
        # For highlighting selected points
        self.scatter2_selected = None

        # self.axes2.plot()
        # self.axes2.set_title("Longest trapped orbit. Press p/n.", 
        #                      fontsize=fs)

        # --- Addd color bar:
        # cax = self.fig.add_axes([0.3, 0.825, 0.15, 0.05])
        # cax = self.fig.add_axes([0.275, 0.625, 0.03, 0.15])
        ax2_pos = self.axes2.get_position()
        ax2_pos.x1 -= 0.035

        self.axes2.set_position(ax2_pos)
        cax = self.fig.add_axes([ax2_pos.x1 + 0.015, ax2_pos.y0 + 0.04,
                                  0.02, (ax2_pos.y1 - ax2_pos.y0)*0.8])
        self.colorbar = self.fig.colorbar(self.scatter, cax=cax,
                                      orientation='vertical') 
        self.colorbar.ax.tick_params(labelsize=tls)
        # self.colorbar = self.fig.colorbar(
        #     self.scatter, cax=cax, orientation='vertical')

        # from matplotlib import ticker
        # tick_locator = ticker.MaxNLocator(nbins=3)
        # self.colorbar.locator = tick_locator
        # self.colorbar.update_ticks()

        # --- Plot 3: nu(t)
        # self.axes3 = self.fig.add_subplot(133)
        self.axes3.set_xlabel(r"$n$", fontsize=fs)
        self.axes3.set_ylabel(r"$\nu(n)$", fontsize=fs)
        for axis in self.ax:
            axis.tick_params(labelsize=tls)

        # Event connections:
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_event)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_event)

        self.rectangle_selector2 = RectangleSelector(
            self.axes2, self.rect_select_callback2,
            drawtype='box', useblit=True,
            button=[3],  # only for right mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)
        self.rectangle_selector3 = RectangleSelector(
            self.axes3, self.rect_select_callback3,
            drawtype='box', useblit=True,
            button=[3],  # only for right mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)
        
        from matplotlib.ticker import LogLocator, MaxNLocator
    
        self.axes1.yaxis.set_major_locator(LogLocator(numticks=7))
        self.axes2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self.axes3.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self.update_orbit()
        embed_labels(self.ax)
        if show:
            plt.show()

    # -------------------------------------------------------------------------
    # Called from the event:
    def update_orbit(self):
        """Update orbit for new ctr."""
        # Remove any selected points in the middle plot:
        self.axes2.lines = []
        self.scatter2_selected = None

        # Remove any highlighted points in plot 3
        # and FIXME: maybe any frequency lines?
        self.axes3.lines = []

        points = self.trapped_orbits.trapped_orbits(self.ctr)
        self.points = points

        # self.axes2.set_title("Ind=%d, len=%d" % (self.ctr, len(points)))

        # Set x and y data...
        self.scatter.set_offsets(points)
        # Set colors.
        self.scatter.set_array(np.arange(len(points)))

        self.colorbar.mappable.set_clim(vmin=0.0, vmax=len(points))
        self.colorbar.draw_all()

        if not self.compute_frequencies:
            self.fig.canvas.draw()
            return

        # update frequency plot:
        values = compute_nu_along_orbit(points[:, 0], points[:, 1],
                                        freq_pts=self.freq_pts,
                                        offset=self.offset, mapMode='arctan2')
        
        self.t_nu, self.nu, self.nuWBA = values

        self.axes3.collections = []

        if len(self.t_nu) > 0:
            # size = 10
            # self.axes3.scatter(self.t_nu, nu,
            #                    s=size, marker='s', linewidth=0.0,
            #                    c=self.t_nu, cmap=self.mpl_color_map)
            # self.axes3.plot(self.t_nu, self.nu, lw=1, c='r')
            # self.axes3.plot(self.t_nu, self.nuWBA, lw=1, c='b')
            self.axes3.plot(self.t_nu, self.nu, ls='', marker='o', 
                            ms=ms, mew=0, c='r', label='Naff',
                            alpha=self.alpha)
            self.axes3.plot(self.t_nu, self.nuWBA, ls='', marker='o', 
                            ms=ms, mew=0, c='b', label='WBA', 
                            alpha=self.alpha)
            self.axes3.legend(fontsize=lfs, numpoints=3)
    
            min_nu = min(np.min(self.nu), np.min(self.nuWBA))
            max_nu = max(np.max(self.nu), np.max(self.nuWBA))
            print("Nu Range:", min_nu, max_nu)
            min_nu_r = min_nu - 0.05 * (max_nu - min_nu)
            max_nu_r = max_nu + 0.05 * (max_nu - min_nu)
            print("Plot range:", min_nu_r, max_nu_r)
            self.axes3.set_xbound(lower=0.0, upper=len(points))
            self.axes3.set_ybound(lower=min_nu_r, upper=max_nu_r)
            self.axes3.set_xlim(0.0, len(points))
            self.axes3.set_ylim(min_nu_r, max_nu_r)

            self.plot_resonance_lines(min_nu, max_nu)
        else:
            print("Orbit not long enough for frequency analysis.")

        # Update figure
        self.fig.canvas.draw()

    def plot_resonance_lines(self, min_y, max_y):
        """Plot resonance lines."""
        pass
        # sbt = SternBrocotTree(min_y, max_y)
        # while len(sbt.get_fractions_as_array()) < self.min_res_lines:
        #     sbt.next()
        # print("Res: interval:", min_y, max_y)
        # print(sbt.get_fractions_as_array())
        # for frac in sbt.get_fractions_as_array():
        #     self.axes3.axhline(y=frac, color=(0.75, 0.75, 0.75),
        #                        linestyle='-')

    # -------------------------------------------------------------------------
    # Event handling:
    def keypress_event(self, event):
        """Close plot on <space> or quite on 'q'; 'p'rev/'n'ext traj."""
        if event.key == ' ':
            plt.close("all")
        elif event.key == 'q':
            plt.close("all")
            import sys
            sys.exit(0)
        elif event.key == 'R':
            ymin, ymax = self.axes3.get_ylim()
            self.plot_resonance_lines(ymin, ymax)
            self.fig.canvas.draw()

        elif event.key == 'p':
            self.ctr = max(self.ctr - 1, 0)
            print("Prev=", self.ctr)
            self.update_orbit()
        elif event.key == 'n':
            self.ctr = min(self.ctr + 1,
                           len(self.trapped_orbits.return_times) - 1)
            print("Next=", self.ctr)
            self.update_orbit()

        # Some save functionality might be nice:
        # elif event.key == 's':
        #     points = trapped_orbits.trapped_orbits(trapped_orbits.ctr)
        #     data = dict(points=points)
        #     fname = "trapped_grid_standard_%6.5f_border_%6.5f_traj%d" % (
        #         kappa, border, -trapped_orbits.ctr)
        #     fname = fname.replace(".", "_") + ".h5"
        #     CPG.hdf.save_h5_dict(fname, data)
        #     print("Saved", fname)

    def mouse_event(self, event):
        """Mouse-click."""
        mode = plt.get_current_fig_manager().toolbar.mode
        if event.button != 1 or mode != '':
            return

        # Select nearest orbit in P(t) plot:
        if event.inaxes == self.axes1:
            time = event.xdata
            sorted_return_times = np.sort(self.trapped_orbits.return_times)
            # Find index corresponding to closest time:
            ind = np.argmin(np.abs(sorted_return_times - time))

            # Now, self.ctr, counts from the longest (=-1)
            # to the next-longest (=-2) and so on.
            self.ctr = - (len(self.trapped_orbits.return_times) - ind)
            self.ctr = ind

            print("Closest time:", sorted_return_times[ind],
                  " to ", time)
            self.update_orbit()

        # Plot orbit on mous-click in (q, p) plot:
        if event.inaxes == self.axes2:
            point = np.array([event.xdata, event.ydata])
            points = self.trapped_orbits.mapping.mapN(point, 1000)

            self.axes2.plot(points[:, 0], points[:, 1],
                            ls='', marker='o', mew=0, ms=2,
                            color=(0.5, 0.5, 0.5))
            self.fig.canvas.draw()

    # -------------------------------------------------------------------------
    def rect_select_callback2(self, eclick, erelease):
        """On rectangle select in plot 2.

        eclick and erelease are the press and release events.
        """
        print("--------------------------")
        print("FIXME: This is not working correctly right now!")
        print("Rectangle select in plot 2")
        x_1, y_1 = eclick.xdata, eclick.ydata
        x_2, y_2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x_1, y_1, x_2, y_2))
        # print(" The buttons you used were: %s %s" % (
        #     eclick.button, erelease.button))

        if eclick.inaxes != self.axes2:
            return

        # Ensure that x_1 < x_2 and y_1 < y_2:
        if x_1 > x_2:
            x_1, x_2 = x_2, x_1
        if y_1 > y_2:
            y_1, y_2 = y_2, y_1

        if len(self.points) < self.freq_pts:
            print("Orbit not long enough for frequencies.")
            return

        # Remove any highlighted points in plot 2:
        self.scatter2_selected = None
        self.axes2.lines = []

        # Get indices of points in (q, p) inside the rectangle
        inds = np.logical_and(x_1<=self.points[:, 0], self.points[:, 0]<=x_2)
        inds = np.logical_and(inds, y_1 <= self.points[:, 1])
        inds = np.logical_and(inds, self.points[:, 1] <= y_2)

        # Get times corresponding to these points:
        times = np.arange(len(self.points))[inds]

        print("Times=", times)

        # Which frequencies correspond to these times?
        freq_inds = np.int32(times/self.freq_pts)
        print("Number of highlighted frequencies:", len(freq_inds))
        print("INDs=", freq_inds)
        # FIXME compress inds (unique!)

        # --- Highlight in plot 3:
        self.axes3.lines = []
        # self.axes3.plot(
        #     self.t_nu[freq_inds], self.nu[freq_inds],
        #     ls='', marker='o', mew=0, ms=5,
        #     color='black')
        self.axes3.plot(self.t_nu, self.nu, ls='', marker='o', 
                        ms=ms, mew=0, c='r', label='Naff', alpha=self.alpha)
        self.axes3.plot(self.t_nu, self.nuWBA, ls='', marker='o', 
                        ms=ms, mew=0, c='b', label='WBA', alpha=self.alpha)

        self.fig.canvas.draw()
        print("Highlighting in plot 3 done.")

    # -------------------------------------------------------------------------
    def rect_select_callback3(self, eclick, erelease):
        """On rectangle select in plot 3.

        eclick and erelease are the press and release events.
        """
        print("--------------------------")
        print("Rectangle select in plot 3")
        x_1, y_1 = eclick.xdata, eclick.ydata
        x_2, y_2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x_1, y_1, x_2, y_2))
        # print(" The buttons you used were: %s %s" % (
        #     eclick.button, erelease.button))

        if eclick.inaxes != self.axes3:
            return

        # Ensure that x_1 < x_2 and y_1 < y_2:
        if x_1 > x_2:
            x_1, x_2 = x_2, x_1
        if y_1 > y_2:
            y_1, y_2 = y_2, y_1

        # select rectangle of points in (t, nu(t)):
        inds = np.logical_and(x_1 <= self.t_nu, self.t_nu <= x_2)
        inds = np.logical_and(inds, y_1 <= self.nu)
        inds = np.logical_and(inds, self.nu <= y_2)

        # Determine associated orbit points:
        orbit_inds = np.zeros(len(self.points), dtype=bool)

        freq_pts_2 = self.freq_pts//2

        # Loop over selected times
        for t in self.t_nu[inds]:
            orbit_inds[t-freq_pts_2:t+freq_pts_2] = True

        print("Number of orbit points:", np.sum(orbit_inds))

        # Remove any highligted points in plot 3:
        # self.axes3.lines = [] #FIXME: why should you remove anything here?

        if self.scatter2_selected is not None:
            self.axes2.lines = []
        self.scatter2_selected = self.axes2.plot(
            self.points[orbit_inds, 0], self.points[orbit_inds, 1],
            ls='', marker='o', mew=0, ms=4,
            color='black')

        self.fig.canvas.draw()
        print("Highlighting in plot 2 done")
        
def plot_resonance_lines(ax, min_y, max_y, min_res_lines=10):
    sbt = SternBrocotTree(min_y, max_y)
    while len(sbt.get_fractions_as_array()) < min_res_lines:
        sbt.next()
    # print("Res: interval:", min_y, max_y)
    # print(sbt.get_fractions_as_array())
    for frac in sbt.get_fractions_as_array():
        ax.axhline(y=frac, color=(0.75, 0.75, 0.75), ls='-')
    pass    
    
def plotFrequencyAnalysis(TrappingObject, freq_pts=4096, offset=100,
                          min_res_lines=10, alpha=1.0):
    points = TrappingObject.points
    values = compute_nu_along_orbit(points[:, 0], points[:, 1],
                                    freq_pts=freq_pts,
                                    offset=offset, mapMode='arctan2')
    # nuNaff, nuWBA, tval = Trapping.nu, Trapping.nuWBA, Trapping.t_nu
    tval, nuNaff, nuWBA = values
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(tval, nuNaff, ls='', marker='o', ms=4, mew=0, c='r',
            alpha=alpha, label='Naff')
    ax.plot(tval, nuWBA, ls='', marker='o', ms=4, mew=0, c='b',
            alpha=alpha, label='WBA')
    if min_res_lines > 0:
        min_nu = min(np.min(nuNaff), np.min(nuWBA))
        max_nu = max(np.max(nuNaff), np.max(nuWBA))
        plot_resonance_lines(ax, min_nu, max_nu, min_res_lines)
    ax.legend(fontsize=lfs, numpoints=3)
    ax.set_title(f"Orbit segment length of {freq_pts} with offset {offset}",
                 fontsize=14)
    plt.show()
    return

def plot_freq_parts(TrappingObject, start=[0], end=[-1], 
                    freq_pts=1024, offset=50, extent=[[0.0, 1.0, -0.5, 0.5]]):
    # if end - start < freq_pts + offset:
    #     print("start - end must be larger than length + offset!")
    #     end = start + offset + freq_pts
    nrows = len(start)
    points = TrappingObject.points
    values = compute_nu_along_orbit(points[:, 0], points[:, 1],
                                    freq_pts=freq_pts,
                                    offset=offset, mapMode='arctan2')
    tval, nuNaff, nuWBA = values
    
    fig, ax = plt.subplots(nrows, 2, figsize=(14, 6))
    bottom = 0.09
    if nrows == 1:
        ax = np.array([ax, ax])
        bottom = 0.12
    plt.subplots_adjust(left=0.075, bottom=bottom, right=0.93, top=0.983,
                        wspace=0.25)
    xlabel, ylabel = r"$q_n$", r"$p_n$"
    indx_old = np.arange(len(points[:, 0]))
    for i in range(nrows):
        indx = np.arange(start[i], end[i])
        # indx_nu = np.arange(start[i] // offset, end[i] // offset)
        indx_nu = ((tval > start[i]) & (tval < end[i]))
        ax[-1, 1].set_xlabel(xlabel, fontsize=fs)
        ax[i, 1].set_ylabel(ylabel, fontsize=fs)
        ax[i, 1].axis(extent[i])
        ax[i, 0].set_xlim(start[i], end[i])
        ax[i, 1].scatter(points[indx_old, 0], points[indx_old, 1], 
                         s=1, c='k', alpha=0.1)
        scatter_plot = ax[i, 1].scatter(points[indx, 0], points[indx, 1], 
                                        s=5, c=indx, 
                                        cmap=TrappingObject.mpl_color_map)
        ax[i, 0].plot(tval[indx_nu], nuNaff[indx_nu], ls='', 
                   marker='o', ms=4, mew=0, c='r',
                   alpha=alphaNaff, label='Naff')
        ax[i, 0].plot(tval[indx_nu], nuWBA[indx_nu], ls='', 
                   marker='o', ms=4, mew=0, c='b',
                   alpha=alphaNaff, label='WBA')
        
        ax2_pos = ax[i, 1].get_position()
        ax2_pos.x1 -= 0.035
    
        ax[i, 1].set_position(ax2_pos)
        cax = fig.add_axes([ax2_pos.x1 + 0.015, ax2_pos.y0 + 0.04,
                                  0.02, (ax2_pos.y1 - ax2_pos.y0)*0.8])
        colorbar = fig.colorbar(scatter_plot, cax=cax,
                                orientation='vertical') 
        colorbar.ax.tick_params(labelsize=tls)
        ax[-1, 0].set_xlabel(r"$n$", fontsize=fs)
        ax[i, 0].set_ylabel(r"$\nu(n)$", fontsize=fs)
        ax[i, 0].legend(fontsize=lfs, numpoints=3)
        # ax[i, 0].set_title(f"Window length {freq_pts} with offset {offset}",
        #                 fontsize=fs)
        # plt.show()
        indx_old = indx
    for axis in ax.flat:
        axis.tick_params(labelsize=tls)
    fig.canvas.draw()
    if nrows == 1:
        ax = ax[0]
    embed_labels(ax.flat)
    return fig

def plot_freq_parts_v2(TrappingObject, start=[0], end=[-1], 
                       freq_pts=1024, offset=50, jmp=57450, 
                       extent=[[0.0, 1.0, -0.5, 0.5]]):
    # if end - start < freq_pts + offset:
    #     print("start - end must be larger than length + offset!")
    #     end = start + offset + freq_pts
    points = TrappingObject.points
    values = compute_nu_along_orbit(points[:, 0], points[:, 1],
                                    freq_pts=freq_pts,
                                    offset=offset, mapMode='arctan2')
    tval, nuNaff, nuWBA = values
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax = ax.flatten()
    plt.subplots_adjust(left=0.085, bottom=0.09, right=0.93, top=0.983,
                        wspace=0.25)
    ax[0].set_xlim(start[0], end[0])
    ax[0].set_xlabel(r"$n$", fontsize=fs)
    ax[0].set_ylabel(r"$\nu(n)$", fontsize=fs)
    indx = np.arange(start[0], end[0])
    indx_nu = ((tval > start[0]) & (tval < end[0]))
    ax[0].plot(tval[indx_nu], nuNaff[indx_nu], ls='', 
               marker='o', ms=4, mew=0, c='r',
               alpha=alphaNaff, label='Naff')
    ax[0].plot(tval[indx_nu], nuWBA[indx_nu], ls='', 
               marker='o', ms=4, mew=0, c='b',
               alpha=alphaNaff, label='WBA')
    ax[0].legend(fontsize=lfs, numpoints=3)
    ax[1].scatter(points[:, 0], points[:, 1], 
                  s=1, c='k', alpha=0.1)
    scatter_plots = [None]
    scatter_plots.append(ax[1].scatter(points[indx, 0], points[indx, 1], 
                                       s=5, c=indx, 
                                       cmap=TrappingObject.mpl_color_map))
    intvl = 1000
    ax[2].plot(points[jmp-intvl:jmp, 0], points[jmp-intvl:jmp, 1],
               ms=4, mew=0, ls='', marker='o', c='b', 
               label=f"${jmp-intvl}\\le n<{jmp}$")
    ax[2].plot(points[jmp:jmp+intvl, 0], points[jmp:jmp+intvl, 1],
               ms=4, mew=0, ls='', marker='o', c='orange', 
               label=f"${jmp}\\le n<{jmp+intvl}$")
    ax[2].legend(fontsize=lfs, numpoints=3)
    scatter_plots.append(None)
    ax[3].scatter(points[indx, 0], points[indx, 1], 
                  s=1, c='k', alpha=0.1)
    new_indx = np.arange(start[1], end[1])
    scatter_plots.append(ax[3].scatter(points[new_indx, 0], 
                                       points[new_indx, 1], 
                                       s=5, c=new_indx, 
                                       cmap=TrappingObject.mpl_color_map))
        
    for i in range(1, 4):
        try:
            ax[i].axis(extent[i-1])
        except IndexError:
            ax[i].axis(extent[0])
        if i != 2:
            ax_pos = ax[i].get_position()
            ax_pos.x1 -= 0.035
        
            ax[i].set_position(ax_pos)
            cax = fig.add_axes([ax_pos.x1 + 0.015, ax_pos.y0 + 0.04,
                                0.02, (ax_pos.y1 - ax_pos.y0)*0.8])
            colorbar = fig.colorbar(scatter_plots[i], cax=cax,
                                    orientation='vertical') 
            colorbar.ax.tick_params(labelsize=tls)
        ax[i].set_xlabel(r"$q_n$", fontsize=fs)
        ax[i].set_ylabel(r"$p_n$", fontsize=fs)
        # if i != 1: ax[i].set_xlabel(r"$q_n$", fontsize=fs)
        # if i != 3: ax[i].set_ylabel(r"$p_n$", fontsize=fs)
    for axis in ax.flat:
        axis.tick_params(labelsize=tls)
    from matplotlib.ticker import MaxNLocator
    
    for axis in ax.flat:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.canvas.draw()
    embed_labels(ax.flat)
    return fig

###############################################################################

if __name__ == "__main__":
    PATH_TP = "C:\\Users\\Joachim\\Documents\\UNI\\TheoretischePhysik\\" 
    PATH_PIC = "CP_Bachelor\\bachelor_thesis\\pictures\\"
    PATH = PATH_TP + PATH_PIC
    """trapping_std_map for generating orbits"""
    fname = "trapping_std_map_2_3200.h5"
    freq_pts = 1024
    offset = 50
    min_res_lines = 10
    show = True
    # Trapping = GUITrapping(fname, freq_pts=freq_pts, offset=offset, 
    #                         min_res_lines=min_res_lines, show=show,
    #                         alpha=alphaNaff)
    # plt.savefig(PATH + "Trapping2D_K2_32L10Offset50.png", dpi=150)
    ### start, end = [55500], [58500]
    ### fig = plot_freq_parts(Trapping, start, end, freq_pts, offset,
    ###                       extent=[0.25, 0.4, 0.1, 0.25])
    # start, end, jmp = [55500, 57000], [58500, 57700], 57000
    # extent = [[0.25, 0.4, 0.1, 0.25], [0.3, 0.4, 0.05, 0.25], 
    #           [0.3, 0.4, 0.05, 0.25]]
    # fig1 = plot_freq_parts_v2(Trapping, start, end, freq_pts, offset, 
    #                           jmp=jmp, extent=extent)
    # plt.savefig(PATH + "Trapping2D_K2_32L10Offset50_57000.png", dpi=150)
    # start, end, jmp = [15000, 17100], [20000, 18100], 16000
    # start, end = [15000], [20000]
    # extent = [[0.25, 0.4, 0.1, 0.25], [0.3, 0.4, 0.05, 0.25], 
    #           [0.3, 0.4, 0.05, 0.25]]
    # fig2 = plot_freq_parts(Trapping, start, end, freq_pts, offset, 
    #                        extent=extent)
    # plt.savefig(PATH + "Trapping2D_K2_32L10Offset50_16000v1.png", dpi=150)
    # if not show:
    #     plotFrequencyAnalysis(Trapping, freq_pts, offset, alpha=alphaNaff)
        
    # plt.savefig(PATH + "Trapping2D_K2_32L10Offset50.png", dpi=150)