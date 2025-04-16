# Joachim Schwardt + 4768711
# Julian Fleck + 4759587


import numpy as np

# Für die Plots
import matplotlib.pyplot as plt
from scipy import sparse

# Lade die Scipy-LU-Zerlegung als HACK
from scipy.sparse.linalg import factorized

# COO/CSR Matrix
import scipy


def csr_row_vector_sum(data, indices, indptr, i, vector):
    """
    Für die arrays 'data', 'indices' und 'indptr' einer CSR-matrix 'L' wird die Summe
        \sum_{j=0}^{i-1} L_{ij} L_{kj} vector_{j}
    berechnet. Der Algorithmus bricht also bei der Diagonalen ab.
    """
    _sum = 0.0
    for j in range(indptr[i], indptr[i+1]):
        # abbrechen, wenn der Spaltenindex der 'i'-ten Zeile jenseits der Diagonalen liegt
        if indices[j] >= i:
            break

        _sum += vector[indices[j]] * data[j]**2
    return _sum


def csr_row_vector_row_sum(data, indices, indptr, i, k, vector):
    """
    Für die arrays 'data', 'indices' und 'indptr' einer CSR-matrix 'L' wird die Summe
        \sum_{j=0}^{k-1} L_{ij} L_{kj} vector_{j}
    berechnet. Der Algorithmus bricht also bei der Diagonalen von Zeile 'k' ab.
    """
    j_k = indptr[k]         # index Pointer der 'k'-ten Zeile
    j_i = indptr[i]         # index Pointer der 'i'-ten Zeile
    k_ind = indices[j_k]    # Spaltenindex der 'k'-ten Zeile
    i_ind = indices[j_i]    # Spaltenindex der 'i'-ten Zeile
    _sum = 0.0
    while True:
        # abbrechen, wenn eine der Zeilen am Ende angekommen ist 
        # (evtl. ist das in unserem Fall redundant, weil wir so viele explizite Nullen haben)
        if j_i >= indptr[i+1] or j_k >= indptr[k+1]:
            break
        
        # abbrechen, wenn der Spaltenindex der 'k'-ten Zeile jenseits der Diagonalen liegt
        if k_ind >= k:
            break
        
        if k_ind == i_ind:       # Summe ausführen, wenn die Spaltenindizes gleich sind
            _sum += data[j_k] * data[j_i] * vector[k_ind]
            j_k += 1
            k_ind = indices[j_k]
            j_i += 1
            i_ind = indices[j_i]
            
        elif k_ind < i_ind:      # Element in Zeile 'i' fest, Element in Zeile 'k' vorrücken
            j_k += 1
            k_ind = indices[j_k]
            
        else:                    # Element in Zeile 'k' fest, Element in Zeile 'i' vorrücken
            j_i += 1
            i_ind = indices[j_i]

    return _sum


# a)
def incompleteLU( A ):
    # extrahiere die Daten
    indices = A.indices
    indptr = A.indptr
    data = A.data

    # Größe der Matrix
    n = len(indptr)-1

    # L wird das gleiche Muster haben => Kopie
    lIndices = indices.copy()
    lIndptr = indptr.copy()
    lData = data.copy()

    # L soll zu Beginn I sein
    for i in range(n):
        for index in range(lIndptr[i], lIndptr[i+1]):
            j = lIndices[index]
            lData[ index ] = 1.0 if i==j else 0.0

    # Jetzt hat L das gleiche Muster wie A, aber mit sehr vielen "expliziten" Nullen
    # => am Ende rufen wir "eliminate_zeros()" auf. Das entfernt diese für uns

    # D ist nur eine diag-Matrix => speichere es als Vektor
    d = np.zeros(n)

    for i in range(n):
        for k in range(lIndptr[i], lIndptr[i+1]):
            new_val = data[k]
            col = lIndices[k]     # das ist der Spaltenindex des Elements 'new_val'

            # keine Einträge oberhalb der Diagonalen
            if col > i:
                continue

            # Diagonaleinträge
            elif col == i:
                _sum = csr_row_vector_sum(lData, lIndices, lIndptr, i, d)
                d[i] = new_val - _sum

            # Einträge unterhalb der Diagonalen
            else:
                _sum = csr_row_vector_row_sum(lData, lIndices, lIndptr, i, col, d)
                lData[k] = (new_val - _sum) / d[col]
                

    # aus den 3 Listen wird die CSR Matrix erstellt
    L = scipy.sparse.csr_matrix((lData, lIndices, lIndptr), shape=(n, n))

    # entferne die Nulleinträge auf der rechten Hälfte (damit es eine "richtige" Dreicksmatrix wird)
    # relativ teuer in der Anwendung: Kann man es weglassen?
    L.eliminate_zeros()

    return L, d


# Berechne W^{-1}*r = L^{-T} * diag(d)^{-1} * L^{-1} r
def applyIncompleteCholesky( L, LT, d, r):
    '''
    Hier sollte eigenlich

          spsolve_triangular( L, r )

    etc. stehen, um die Dreieckssysteme aufzulösen.
    Aber das ist unglaublich langsam: https://github.com/scipy/scipy/issues/16131

    Deshalb betrachen wir eine LU-Zerlegung von L und L^T (diese ist quasi gratis, da schon Dreiecksform).
    Diese Zerlegung geben wir als L und LT oben in die Funktion rein.
    Und dann lösen L*a=r und LT*c = b einfach über den ()-Operator folgt auf:
    '''

    # L^{-1}
    a = L(r)

    # D^{-1}
    b = a / d;

    # LT^{-1}
    c = LT(b)

    return c


def get_sparse_poisson(size=4):
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
    d_matrix = sparse.spdiags(diagonals,         # diagonals of 1D-Poisson
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

def cgStep( A, xk, rk, dk ):
    A_dk = A.dot(dk)
    alphak = rk.dot(rk) / dk.dot(A_dk)
    xk1 = xk + alphak * dk
    rk1 = rk - alphak * A_dk

    betak1 = rk1.dot(rk1) / rk.dot(rk)
    dk1 = rk1 + betak1 * dk

    return xk1, rk1, dk1


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
        x0, r, d = cgStep(A, x0, r, d)
        if i % store_step == 0:
            res[ctr] = x0
            ctr += 1

    return res, store


def IcCgStep( A, L, LT, d, xk, rk, dk, W_inv_rk ):
    A_dk = A.dot(dk)
    alphak = rk.dot(W_inv_rk) / dk.dot(A_dk)
    xk1 = xk + alphak * dk

    rk1 = rk - alphak * A_dk
    W_inv_rk1 = applyIncompleteCholesky(L, LT, d, rk1)
    betak1 = rk1.dot(W_inv_rk1) / rk.dot(W_inv_rk)
    dk1 = W_inv_rk1 + betak1 * dk

    return xk1, rk1, dk1, W_inv_rk1


def get_ic_cg_steps(A, L, LT, d, b, x0=None, num_steps=100, num_store=100):
    if x0 is None:
        x0, r0 = np.zeros_like(b), b
    else:
        r0 = b - A.dot(x0)
    d0 = applyIncompleteCholesky(L, LT, d, r0)
    W_inv_r0 = applyIncompleteCholesky(L, LT, d, r0)

    num_store = min(num_store, num_steps)

    # storing every step is O(n**2), better reduce to O(store * n)
    store_step = num_steps // num_store

    store = np.arange(0, num_steps, store_step)
    res = np.zeros((store.shape[0], x0.shape[0]))
    ctr = 0
    for i in range(num_steps):
        x0, r0, d0, W_inv_r0 = IcCgStep(A, L, LT, d, x0, r0, d0, W_inv_r0)
        if i % store_step == 0:
            res[ctr] = x0
            ctr += 1

    return res, store


###################### START ########################


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
    n = 100
    steps = 30

    xs, ys = get_xy(n)
    b = get_b(xs, ys)
    A = get_sparse_poisson(n)

    # Berechne L und LT
    L, d = incompleteLU(A)
    LT = L.transpose()


    # Berechne die LU-Faktorisierung von L und LT, wegen des Scipy-Bugs
    Lfac = factorized(L.tocsc())
    LTfac = factorized(LT)
    # => diese geben wir in die Methoden rein

    exact_res = sparse.linalg.spsolve(A, b)
    cg_steps, cg_store = get_cg_steps(A, b, num_steps=steps)
    ic_cg_steps, ic_cg_store = cg_steps, cg_store
    ic_cg_steps, ic_cg_store = get_ic_cg_steps(A, Lfac, LTfac, d, b, num_steps=steps)

    norm_diff_cg = np.linalg.norm(cg_steps - exact_res, axis=1)
    norm_diff_ic_cg = np.linalg.norm(ic_cg_steps - exact_res, axis=1)

    labels = [r'CG ohne PC', r'CG mit PC']
    plot_error([cg_store, ic_cg_store], [norm_diff_cg, norm_diff_ic_cg], labels)
    return 0

if __name__ == '__main__':
    main()

