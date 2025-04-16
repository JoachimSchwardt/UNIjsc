# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# komischer Bug mit dem Spyder Profiler:
# ohne diesen Import kommt ein "scipy.sparse" has no attribute "linalg"
# beim normalen Ausführen funktioniert es aber auch ohne den import...
import scipy.sparse.linalg


# Externe Wärmezufuhr auf den Mittelpunkt mit Radius 0.2^(1/2)
def f(x):
    return 1 if (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.2 else 0


def get_xy(n=30):
    # inkl. Endpunkte
    xs = np.linspace(0, 1, n+1)
    ys = np.linspace(0, 1, n+1)
    return xs, ys


def get_b(xs, ys):
    """n == Punkte pro Achse (bis ca 200 möglich für dichtbesetzte Systeme)"""
    n = xs.shape[0] - 1
    # Gesamtzahl der aktiven Gitterpunkte
    N = (n-1)*(n-1)

    # Bereche den globalen Index des Gitterpunktes (i,j)
    def index(i, j):
        return i*(n-1) + j

    # Berechne die Koordinate des Gitterpunktes (i,j)
    def coord(i, j):
        return xs[i+1], ys[j+1]
    b = np.zeros(N)


    # T_i Blöcke:
    for i in range(n-1):
        # Zeile j in T_i:
        for j in range(n-1):
            idx = index(i,j)

            # Vektor b:
            b[idx] = f(coord(i,j))

    return b


def get_sparse_poisson(size=50):
    """
    Example for 'size = 3':

    d_matrix = matrix([[-2.,  1.],
                        [ 1., -2.]])

    d_eye = matrix([[-2.,  0.,  1.,  0.],
                    [ 0., -2.,  0.,  1.],
                    [ 1.,  0., -2.,  0.],
                    [ 0.,  1.,  0., -2.]])

    eye_d = matrix([[-2.,  1.,  0.,  0.],
                    [ 1., -2.,  0.,  0.],
                    [ 0.,  0., -2.,  1.],
                    [ 0.,  0.,  1., -2.]])

    A = matrix([[-4.,  1.,  1., -0.],
                [ 1., -4., -0.,  1.],
                [ 1., -0., -4.,  1.],
                [-0.,  1.,  1., -4.]]) * (-3**2)
    """
    n = size - 1
    ones = np.ones(n)
    diagonals = np.array([ones, -2*ones, ones])
    d_matrix = sparse.spdiags(diagonals,     # diagonals of 1D-Poisson
                                  [-1, 0, 1],    # diagonal index offset
                                  n, n,          # shape of the result
                                  'csr'          # format
                                  )
    eye = sparse.eye(n)

    # [[D[0, 0] * I, D[0, 1] * I, D[0, 2] * I, ...],
    #  [D[1, 0] * I, D[1, 1] * I, D[1, 2] * I, ...], ...]
    d_eye = sparse.kron(d_matrix, eye)


    # [[D, 0, 0, ...],
    #  [0, D, 0, ...], ...]
    eye_d = sparse.kron(eye, d_matrix)

    A = (d_eye + eye_d) * (-size**2)
    return A.tocsr()


def gradientStep(A, b, xk):
    rk = b - A.dot(xk)

    alphak = rk.dot(rk) / rk.dot(A.dot(rk))
    xk1 = xk + alphak * rk

    return xk1


def cg_step(A, xk, rk, dk):
    alphak = rk.dot(rk)/dk.dot(A.dot(dk))
    xk1 = xk + alphak * dk

    rk1 = rk - alphak * A.dot(dk)

    betak1 = rk1.dot(rk1)/rk.dot(rk)
    dk1 = rk1 + betak1 * dk

    return xk1, rk1, dk1


def get_grad_steps(A, b, x0=None, num_steps=100, num_store=100):
    if x0 is None:
        x0 = b

    num_store = min(num_store, num_steps)


    # storing every step is O(n**2), better reduce to O(store * n)
    store_step = num_steps // num_store

    store = np.arange(0, num_steps, store_step)
    res = np.zeros((store.shape[0], x0.shape[0]))
    ctr = 0
    for i in range(num_steps):
        x0 = gradientStep(A, b, x0)
        if i % store_step == 0:
            res[ctr] = x0
            ctr += 1

    return res, store


def get_cg_steps(A, b, x0=None, num_steps=100, num_store=100):
    if x0 is None:
        x0, r, d = np.zeros_like(b), b, b
    else:
        r = b - A.dot(x0)
        d = r

    num_store = min(num_store, num_steps)

    # storing every step is O(n**2), better reduce to O(store * n)
    store_step = num_steps // num_store

    store = np.arange(0, num_steps, store_step)
    res = np.zeros((store.shape[0], x0.shape[0]))
    ctr = 0
    for i in range(num_steps):
        x0, r, d = cg_step(A, x0, r, d)
        if i % store_step == 0:
            res[ctr] = x0
            ctr += 1

    return res, store


def plot_error(x_vals, y_vals, labels):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$\parallel x_N - x^\ast \parallel$')
    for i in range(len(x_vals)):
        ax.plot(x_vals[i], y_vals[i], label=labels[i])

    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    """
    Um für alle 'n' ein ähnliches Konvergenzverhalten zu sehen, müssen etwa
    'n**2' Schritte durchgeführt werden.
    Das ist allerdings für n >~ 200 bereits sehr teuer
    """
    n = 100
    steps = 1000#n**2

    xs, ys = get_xy(n)
    b = get_b(xs, ys)
    A = get_sparse_poisson(n)

    exact_res = sparse.linalg.spsolve(A, b)
    grad_steps, grad_store = get_grad_steps(A, b, num_steps=steps)
    cg_steps, cg_store = get_cg_steps(A, b, num_steps=steps)

    norm_diff_grad = np.linalg.norm(grad_steps - exact_res, axis=1)
    norm_diff_cg = np.linalg.norm(cg_steps - exact_res, axis=1)

    labels = [r'$\parallel x_{\mathrm{grad}} - x^\ast \parallel$',
              r'$\parallel x_{\mathrm{cg}} - x^\ast \parallel$']
    plot_error([grad_store, cg_store], [norm_diff_grad, norm_diff_cg], labels)
    return 0


if __name__ == '__main__':
    main()
