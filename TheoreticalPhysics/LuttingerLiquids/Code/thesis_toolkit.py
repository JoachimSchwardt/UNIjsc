#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:27:53 2023

@author: joachim
"""

import numpy as np
from scipy import integrate
from time import perf_counter as pc
import matplotlib.pyplot as plt
from matplotlib import ticker
from shapely import geometry
import mpl_special


sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
U = 1/np.sqrt(2) * np.array([[1, 1j], [1j, 1]])   # basis transform AB <--> LR

ptr_line_color = 0.3
ptr_label_color = (0, 0, 0, 0.6)
panels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


# FILE AND CODE MANAGEMENT

def parametrized(decorator):
    """To create decorators that take parameters define them using this decorator"""
    def layer(*args, **kwargs):
        def replica(function):
            return decorator(function, *args, **kwargs)
        return replica
    return layer

@parametrized
def with_threshold(function, threshold=10):
    """Decorator that temporarily adjusts the NumPy-Array threshold for printing"""
    def wrapper(*args, **kwargs):
        threshold_temp = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=threshold)
        result = function(*args, **kwargs)
        np.set_printoptions(threshold=threshold_temp)
        return result
    return wrapper

@with_threshold(threshold=10)
def timer(function, *args, **kwargs):
    t_start = pc()
    result = function(*args, **kwargs)
    t_end = pc()
    print(f"{function.__name__} with {args = } and {kwargs = } executed in "
          f"{t_end - t_start:.4f} seconds")
    return result

def get_mmln(num_params):
    return f"{num_params['mmaxp']}.{num_params['mmax']}.{num_params['lmax']}.{num_params['numkp']}"
def get_ep_delta_string():
    return r"$|\Delta E_{\mathrm{EP}}|$"
def get_ep_offset_string():
    return r"$|\langle E_{\mathrm{EP}}\rangle|$"
def get_label(arg):
    if arg is None:
        raise ValueError
    elif arg == "beta":
        return r"$\beta$"
    elif arg == "vf":
        return r"$v_{\mathrm{F}}$"
    elif arg == "u_a":
        return r"$U_{\mathrm{A}}$"
    elif arg.startswith("real"):
        return r"$\mathrm{Re}\," + arg.removeprefix("real") + "$"
    elif arg.startswith("imag"):
        return r"$\mathrm{Im}\," + arg.removeprefix("imag") + "$"
    elif arg == "k ep":
        return r"$k_\mathrm{EP}$"
    elif arg == "arg Delta E":
        return r"$\vartheta$" # r"$\arg(\Delta E)$"
    else:
        return f"${arg}$"

def get_par_string(arg, mod=".3f"):
    if arg is None:
        raise ValueError
    elif isinstance(arg, (np.ndarray, list)):
        if isinstance(arg[-1], dict):
            par = f"{get_mmln(arg[-1])}"
        else:
            par = f"{arg[0]:{mod}}.{arg[-1]:{mod}}"
    elif isinstance(arg, tuple):
        par = f"{arg[0]}.{arg[-1]}"
    elif isinstance(arg, dict):
        par = get_mmln(arg)
    else:
        par = f"{arg:{mod}}"
    return par

def get_pars_string(args, labels, mod=".3f"):
    pars = "_".join([label + get_par_string(arg, mod) for (label, arg) in zip(labels, args)])
    return pars


# QMBP TOOLS


def basis_lr_to_ab(matrix):
    """Transform a 2x2 matrix from LR to AB basis"""
    return U.conj() @ matrix @ U
def basis_ab_to_lr(matrix):
    """Transform a 2x2 matrix from AB to LR basis"""
    return U @ matrix @ U.conj()


def bose_einstein(value):
    """Bose-Einstein distribution function"""
    if value < -50:
        return -1
    if value > 300:
        return 0.0
    return 1 / (np.exp(value) - 1)


def hamilton_from_green_single(omega, green):
    """Convert GF to Hamiltonian at single frequency"""
    return omega * np.eye(2) - np.linalg.inv(green)


def hamilton_from_green(omega, green):
    """Convert Green's function to Hamiltonian for given frequencies"""
    omega = asarray(omega)
    if green.ndim == 2:
        hamilton = hamilton_from_green_single(omega, green)
    if green.ndim == 3:
        if omega.size == 1:
            hamilton = np.array([hamilton_from_green_single(omega, green[i_k])
                                 for i_k in range(green.shape[0])])
        else:
            raise ValueError(f"{omega.size = } not supported for {green.ndim = }")

    elif green.ndim == 4:
        hamilton = np.array([[hamilton_from_green_single(omega_val, green[i_k, i_omega])
                              for i_omega, omega_val in enumerate(omega)]
                             for i_k in range(green.shape[0])])
    else:
        raise NotImplementedError(f"{green.ndim = } not supported for conversion to Hamiltonian")
    return hamilton


def green_from_hamilton(omega_vals, hamilton):
    """Convert Hamiltonian to Green's function for given frequencies"""
    green = np.zeros_like(hamilton)
    for i_omega, omega in enumerate(omega_vals):
        for i_k in range(hamilton.shape[0]):
            try:
                arr = np.linalg.inv(omega * np.eye(2) - hamilton[i_k, i_omega])
            except np.linalg.LinAlgError:
                arr = np.linalg.inv(omega * np.eye(2) - hamilton[i_k, i_omega] + 1e-15)
            green[i_k, i_omega] = arr
    return green


def hamilton_2x2_decomp(hamilton):
    shape = hamilton.shape
    if shape[-2] != 2 or shape[-1] != 2:
        msg = f"Shape should be 2 by 2 in the last two axes but was {shape}"
        raise IndexError(msg)

    d_0 = np.trace(sigma_0 @ hamilton, axis1=-1, axis2=-2)
    d_x = np.trace(sigma_x @ hamilton, axis1=-1, axis2=-2)
    d_y = np.trace(sigma_y @ hamilton, axis1=-1, axis2=-2)
    d_z = np.trace(sigma_z @ hamilton, axis1=-1, axis2=-2)
    return d_0, d_x, d_y, d_z


def hamilton_2x2_ep_eqn(hamilton):
    """Return the equations d_r^2 - d_i^2 and d_r.d_i governing EP's for a given 2x2-Hamilton"""
    _, d_x, d_y, d_z = hamilton_2x2_decomp(hamilton)
    dr2_di2 = d_x.real**2 + d_y.real**2 + d_z.real**2 - d_x.imag**2 - d_y.imag**2 - d_z.imag**2
    dr_di = d_x.real * d_x.imag + d_y.real * d_y.imag + d_z.real * d_z.imag
    return dr2_di2, dr_di


def eigenvalues_2x2(arr):
    """Eigenvalues for a 2x2 matrix"""
    offset = (arr[0, 0] + arr[1, 1]) / 2
    root = np.sqrt(((arr[0, 0] - arr[1, 1]) / 2)**2 + arr[1, 0] * arr[0, 1])
    return offset - root, offset + root


def get_w(w=np.pi, alpha=None, beta=None, v=None):
    """Compute w = pi alpha / (beta v) if parameters are given; else return w"""
    if alpha is not None:
        return np.pi * alpha / (beta * v)
    return w


def c_quad(func, xmin, xmax, args=(), **kwargs):
    def f_real(x, *args):
        return np.real(func(x, *args))
    def f_imag(x, *args):
        return np.imag(func(x, *args))
    r_int = integrate.quad(f_real, xmin, xmax, args, **kwargs)
    i_int = integrate.quad(f_imag, xmin, xmax, args, **kwargs)
    return (r_int[0] + 1j*i_int[0], np.abs(r_int[1] + 1j*i_int[1]))


# TOOLS FOR ARRAY MANIPULATION

def contiguous_arrays(arrays):
    """Given two arrays attempts to make the values inside as contiguous as possible.
    arr1 = np.arange(15)
    arr2 = np.arange(-15, 0)
    arr1[6], arr2[6] = arr2[6], arr1[6]
    convert_to_contiguous_arrays(arr1, arr2) --> recovers original
    """
    arrays = np.asarray(arrays)
    if arrays.ndim != 2:
        raise NotImplementedError(f"{arrays.ndim = } not supported for continuity transformation")

    flag_transpose = False
    if arrays.shape[1] < arrays.shape[0]:
        flag_transpose = True
        arrays = arrays.T
    count, size = np.sort(arrays.shape)
    if count != 2:
        raise NotImplementedError(f"Number of arrays must be 2 but was {count}")

    arr1, arr2 = arrays[0], arrays[1]
    val1, val2 = arr1[0], arr2[0]
    for j in range(1, size):
        if np.abs(arr1[j] - val1) > np.abs(arr1[j] - val2):
            arr1[j], arr2[j] = arr2[j], arr1[j]
        val1, val2 = arr1[j], arr2[j]
    if flag_transpose:
        return np.array([arr1, arr2]).T
    return np.array([arr1, arr2])

def arg_restrict(array, vmin, vmax):
    """(array, vmin, vmax) -> (index array of elements within [vmin, vmax])"""
    return (array >= vmin) & (array <= vmax)

def restrict(array, vmin, vmax):
    """(array, vmin, vmax) -> (array consisting of elements within [vmin, vmax])"""
    return array[arg_restrict(array, vmin, vmax)]

def asarray(data):
    """return the given data as an array with minimal dimension of 1"""
    data = np.asarray(data)
    if data.ndim == 0:
        data = np.array([data])
    return data


def _increase_density(array, pointer, small, large, vmin, vmax, factor=2):
    for val in np.linspace(small, large, factor+1)[1:]:
        if (val >= vmin and val < vmax) or val == large:
            array[pointer] = val
            pointer += 1
    return pointer

def increase_density(array, vmin=-np.inf, vmax=np.inf, factor=2):
    """Increase the density of values within [vmin, vmax] in an array by 'factor'.
    Example: array = [0, 1, 2, 3, 4, 5, 6, 7]
    increase_density(array, 2, 3.6, factor=4)
        -> [0, 1, 2, 2.25, 2.5, 2.75, 3.25, 3.5, 4, 5, 6, 7]
    """
    array = np.asarray(array)
    if array.ndim != 1:
        raise NotImplementedError("Array must be 1-dimensional")
    new_array = np.zeros(array.size * factor)
    new_array[0] = array[0]
    pointer = 1
    for i in range(1, array.size):
        pointer = _increase_density(new_array, pointer, array[i-1], array[i], vmin, vmax, factor)
    return new_array[:pointer]

def convert_vertices_to_shells(path):
    shells = []
    for code, vertex in zip(path.codes, path.vertices):
        if code == 1:
            shells.append([vertex])
        elif code == 2:
            shells[-1].append(vertex)
        elif code == 79:
            shells[-1].append(79)
    return shells

def convert_shells_to_linestrings(shells):
    linestrings = []
    for shell in shells:
        if isinstance(shell[-1], int):
            if shell[-1] == 79:
                linestrings.append(geometry.LinearRing(shell[:-1]))
        else:
            linestrings.append(geometry.LineString(shell))
    # linestrings = [geometry.LinearRing(shell) for shell in shells]
    return linestrings

def convert_shells_to_polygons(shells):
    linestrings = []
    for shell in shells:
        if isinstance(shell[-1], int):
            if shell[-1] == 79:
                linestrings.append(geometry.Polygon(shell[:-1]))
        else:
            linestrings.append(geometry.LineString(shell))
    # linestrings = [geometry.Polygon(shell) for shell in shells]
    return linestrings
    
        

# TOOLS FOR VISUALIZATION

def savefig(fig, name, datatype=".pdf"):
    plt.pause(0.01)                 # TODO: Fix embed_labels to apply once on startup...
    fig.savefig(name + datatype)
    plt.close("all")
    print("Saved figure: ", name + datatype)


def autoformat(axis, **kwargs):
    """Automatically adjust x-limits to fit data and set labels/limits/scaling for a single axis"""
    if kwargs.get("xlim") is None:
        axis.set_xlim(axis.xaxis.get_data_interval())
    for (key, value) in kwargs.items():
        getattr(axis, "set_" + key)(value)

def set_ticks_linear(axis, vmin, vmax, numticks, decimals=7, which='x', dtype=float):
    ticks = np.round(np.linspace(vmin, vmax, numticks), decimals).astype(dtype)
    getattr(axis, f"{which}axis").set_major_locator(ticker.FixedLocator(ticks))

def get_vertex_indices(polygon, atol=1e-2):
    """Returns the indices where a list of values changes discrete slope beyond given tolerance"""
    polygon = np.asarray(polygon)
    indices = [0]
    dx = polygon[1] - polygon[0]
    for i in range(1, polygon.size-1):
        new_dx = polygon[i+1] - polygon[i]
        if np.abs(new_dx - dx) > atol:
            indices.append(i)
            dx = new_dx
    indices.append(polygon.size-1)
    return indices

def get_color_list(cmap="RdBu"):
    """Get list of colors of a given matplotlib colormap"""
    rgba = np.array([np.array(plt.get_cmap(cmap)(x)) for x in np.linspace(0, 1, 128)])
    ir = get_vertex_indices(rgba[:, 0])
    ig = get_vertex_indices(rgba[:, 1])
    ib = get_vertex_indices(rgba[:, 2])
    indices = np.unique(np.concatenate((ir, ig, ib)))
    color_list = [(i / (rgba.shape[0] - 1), rgba[i, :3]) for i in indices]
    return color_list

def convert_color_list_to_string(color_list):
    """Convert list of colors to a string for the online tool at
    https://eltos.github.io/gradient/#0:4C71FF-25:0025B3-50:FFFFFF-75:C7030D-100:FC4A53"""
    string = ""
    for (location, rgb) in color_list:
        hex_val = plt.matplotlib.colors.rgb2hex(rgb)
        loc_val = int(location * 100)
        string += f"-{loc_val}:{hex_val}"
    string = "#" + string.removeprefix("-")
    return string

def convert_string_to_color_list(string):
    """Convert string from online tool to list of colors"""
    segments = string.removeprefix("#").split("-")
    color_list = []
    for segment in segments:
        loc_val, hex_val = segment.split(":")
        location = float(loc_val) / 100
        rgb = plt.matplotlib.colors.hex2color("#" + hex_val)
        color_list.append((location, rgb))
    return color_list

def get_colormap(color_list, cmap_name="custom"):
    """Convert list of colors to colormap. Format of 'color_list'-entries:
        (location, (R, G, B)) where 'location' must start with 0.0 and end with 1.0"""
    colormap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list)
    return colormap

def get_custom_cmap(cmap="RdBu"):
    """Get a precomputed custom colormap"""
    if cmap == "RdBu":
        string = "#0:67001F-10:B3192C-20:D7634F-30:FDDBC7-37:FFE8DC-50:FFFFFF-63:E0F3FF-70:D1E5F0-85:2267AC-100:053061"
        color_list = convert_string_to_color_list(string)
        return get_colormap(color_list)
    else:
        raise NotImplementedError(f"Custom colormap {cmap} does not exist")

def plot_complex(axis, x_vals, y_vals, set_label=True, label_specifier="E", **kwargs):
    y_vals = np.asarray(y_vals)
    if y_vals.ndim == 1:
        y_vals = np.expand_dims(y_vals, axis=0)
    colors = mpl_special.Colors()
    col_real = colors.get_color()
    col_imag = colors.get_color()
    real_label = get_label("real " + label_specifier)
    imag_label = get_label("imag " + label_specifier)
    if set_label:
        for label, color in zip([real_label, imag_label], [col_real, col_imag]):
            axis.plot([], [], c=color, label=label, **kwargs)
    for y_val in y_vals:
        axis.plot(x_vals, y_val.real, c=col_real, **kwargs)
        axis.plot(x_vals, y_val.imag, c=col_imag, **kwargs)

def plot_ptr_axis(axis):
    ax2 = axis.twinx()
    ax2.set_ylabel("$R$", color=ptr_label_color)
    ax2.tick_params(axis='y', labelcolor=ptr_label_color)
    return ax2


def ep_contour_plot(axis, hamilton, k_vals, omega_vals):
    eqn1, eqn2 = hamilton_2x2_ep_eqn(hamilton)
    color_gen = mpl_special.Colors()
    colors = [color_gen.get_color() for _ in range(2)]
    contour1 = axis.contour(k_vals, omega_vals, eqn1.T, [0], colors=colors[0])
    contour2 = axis.contour(k_vals, omega_vals, eqn2.T, [0], colors=colors[1])
    lines = [plt.Line2D([0], [0], color=color) for color in colors]
    labels = [r"$\textbf{d}_\text{r}^2 = \textbf{d}_\text{i}^2$",
              r"$\textbf{d}_\text{r}\cdot \textbf{d}_\text{i} = 0$"]
    axis.legend(lines, labels)
    return contour1, contour2
    
def plot_contour_intersection(axis, contour1, contour2, ls='', marker='o', **kwargs):
    """Plot markers at the intersection of two contours"""
    path1 = contour1.get_paths()[0]
    path2 = contour2.get_paths()[0]
    shells1 = convert_vertices_to_shells(path1)
    shells2 = convert_vertices_to_shells(path2)
    linestrings1 = convert_shells_to_linestrings(shells1)
    linestrings2 = convert_shells_to_linestrings(shells2)
    for linestring1 in linestrings1:
        for linestring2 in linestrings2:
            points = linestring1.intersection(linestring2)
            if not points.is_empty:
                for geom in points.geoms:
                    xp, yp = geom.xy
                    axis.plot(xp, yp, ls=ls, marker=marker, **kwargs)
                    
def plot_fermi_arcs(axis, contour1, contour2, **kwargs):
    """Plot markers at the intersection of two contours"""
    path1 = contour1.get_paths()[0]
    path2 = contour2.get_paths()[0]
    shells1 = convert_vertices_to_shells(path1)
    shells2 = convert_vertices_to_shells(path2)
    polygons1 = convert_shells_to_polygons(shells1)
    linestrings2 = convert_shells_to_linestrings(shells2)
    for polygon1 in polygons1:
        for linestring2 in linestrings2:
            points = polygon1.intersection(linestring2)
            if not points.is_empty:
                try:
                    for geom in points.geoms:
                        xp, yp = geom.xy
                        axis.plot(xp, yp, **kwargs)
                except AttributeError:
                    xp, yp = points.xy
                    axis.plot(xp, yp, **kwargs)

def plot_ep(axis, k_vals, omega_vals, green, cmap=get_custom_cmap("RdBu")):
    """Plot Exceptional Points for a numeric dataset"""
    extent = [k_vals[0], k_vals[-1], omega_vals[0], omega_vals[-1]]
    hamilton = hamilton_from_green(omega_vals, green)
    axis.set_xlabel(r"$k$")
    axis.set_ylabel(r"$\omega$")
    contour1, contour2 = ep_contour_plot(axis, hamilton, k_vals, omega_vals)
    eqn1, eqn2 = hamilton_2x2_ep_eqn(hamilton)
    phase = np.sqrt(eqn1 + 2j*eqn2)
    arg = np.arctan2(phase.imag, phase.real)
    img = axis.imshow(arg.T, cmap=cmap, aspect="auto", extent=extent,
                    origin="lower", vmin=-np.pi/2, vmax=np.pi/2, alpha=0.85)
    cbar = axis.figure.colorbar(img, ax=axis, label=get_label("arg Delta E"))
    mpl_special.format_ticklabels(cbar.ax, which="y")
    return contour1, contour2


# TOOLS FOR LATEX SOURCE MANIPULATION

def get_filenames(indx=None, path=None, file_extension=".tex"):
    """Get filename at 'indx'. If None, return list of all filenames"""
    filenames = ["introduction", "bosonization", "perturbative_gf", "topology", "renormalization",
                  "exceptional_ll", "summary", "appendix"]
    if path is None:
        path = "../MA_Latex/thesis/chapters/"
    if indx is not None:
        return path + filenames[indx] + file_extension
    return [path + filename + file_extension for filename in filenames]

def parse_files(filenames, expression="example"):
    """Parse Latex string"""
    for filename in filenames:
        with open(filename, "r") as file:
            lines = file.readlines()
            for linenumber, line in enumerate(lines):
                if expression in line:
                    print(f"{filename}({linenumber}): {line}")

