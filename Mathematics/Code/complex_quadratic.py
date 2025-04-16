"""Solve (x**2-7*x+11+0j)**(x**2-11*x+30+0j) == 1.0
(https://www.youtube.com/watch?v=XOBKP4Hvpbo,  One of the top five best engineered math questions )"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mpl_special

def f(y):
    xr, xi = y
    x = xr + 1j*xi
    # return np.abs((x**2-7*x+11+0j)**(x**2-11*x+30+0j) - 1.0)
    zval = (x**2-11*x+30+0j) * np.log(x**2-7*x+11+0j)
    return np.abs(zval.real) + np.abs(np.sin(zval.imag / 2))
def g(x):
    return (x**2-7*x+11+0j)**(x**2-11*x+30+0j)

def get_extent(x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    extent = [x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
    return extent
def linspace_center(vmin, vmax, n_val):
    array, d_x = np.linspace(vmin, vmax, n_val, endpoint=False, retstep=True)
    array += d_x / 2
    return array
def imshow(axis, array, x_y=None, norm="auto", dig=2, cmap="RdBu", aspect="auto", **kwargs):
    _array = array.T    # transpose the array for imshow-plot, otherwise axes do not match
    if x_y is not None:
        kwargs["extent"] = get_extent(*x_y)
        if x_y[0].size != array.shape[0]:
            _array = _array.T
    if norm == "auto":
        for key in ["vmin", "vmax"]:
            if key not in kwargs.keys():
                kwargs[key] = round(getattr(array, key[1:])(), dig)
    img = axis.imshow(_array, cmap=cmap, aspect="auto", origin="lower", **kwargs)
    return img

z0 = 2.2918442548 * np.exp(1j*0.4648578738)
result = minimize(f, [z0.real, z0.imag], bounds=[(2.048645855302712, 2.048645873780236),
 (1.0274240531973018, 1.027424077672105)])
print(result)

r = linspace_center(1.9, 7.1, 501)
theta = linspace_center(-np.pi, np.pi, 431)
# extent = (
# ((2.291844250224867, 2.2918442594962105),
#  (0.4648578688737198, 0.46485787879173424))
# )
# r = linspace_center(*extent[0], 301)
# theta = linspace_center(*extent[1], 231)
r2, theta2 = np.meshgrid(r, theta)
z2 = r2 * np.exp(1j*theta2)
f2 = f([z2.real, z2.imag])
fig, axis = plt.subplots()
# img = imshow(axis, np.abs(f2), x_y=(r, theta))
# axis.figure.colorbar(img, ax=axis, label=r"$f(r\mathrm{e}^{\mathrm{i}\theta})$", fraction=0.15, aspect=12)
img = imshow(axis, np.clip(np.log(np.abs(f2)), -6, np.inf), x_y=(r, theta))
axis.figure.colorbar(img, ax=axis, label=r"$\log|f(r\mathrm{e}^{\mathrm{i}\theta})|$", fraction=0.15, aspect=12)
axis.set_xlabel("$r$"); axis.set_ylabel(r"$\theta$")
mpl_special.embed_labels(fig, fig.axes)
print(axis.get_xlim(), axis.get_ylim())

