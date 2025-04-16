# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np

# Für die Plots
import matplotlib.pyplot as plt

# Lade den Löser für dünnbesetzte Systeme
from scipy.sparse.linalg import factorized

# COO/CSR Matrix
import scipy


# Bereche den globalen Index des Gitterpunktes (i,j)
def index(i, j, n):
    """Konvertieren des Index-Formats"""
    return i*(n-1) + j

def computeCsrA( n ):
    """Poisson-Matrix im CSR-Format"""
    N = (n-1)*(n-1)
    h = 1.0 / n

    # Konstruiere zuerst eine COO Matrix
    row  = []
    col  = []
    data = []

    # T_i Blöcke:
    for i in range(n-1):
        # Zeile j in T_i:
        for j in range(n-1):
            idx = index(i,j,n)
            # Diagonale:
            row.append(idx)
            col.append(idx)
            data.append(4.0/(h**2))

            # Nebendiagonale:
            if j<n-2:
                row.append(idx+1)
                col.append(idx)
                data.append(-1.0/(h**2))
                row.append(idx)
                col.append(idx+1)
                data.append(-1.0/(h**2))

            # -I Blöcke:
            if i<n-2:
                row.append(idx+n-1)
                col.append(idx)
                data.append(-1.0/(h**2))
                row.append(idx)
                col.append(idx+n-1)
                data.append(-1.0/(h**2))

    cooA = scipy.sparse.coo_matrix((data, (row,col)), shape=(N,N))

    return cooA.tocsr()


def doPowerIteration(A, Axk_minus1):
    """Führt einen Schritt der Vektoriteration aus.
    x_{k-1} und Ax_{k-1} nicht notwendigerweise normiert.
    """
    xk = Axk_minus1 / np.linalg.norm(Axk_minus1)
    Axk = A.dot(xk)
    lambda_k = xk.dot(Axk)
    return lambda_k, Axk

# a)
def powerIteration( A, x0, max_iter=300, rtol=1e-15 ):
    """x0 wird hier nicht als normiert angenommen --> Divison durch Norm."""
    Ax0 = A.dot(x0)
    lambda0 = np.linalg.norm(Ax0) / np.linalg.norm(x0)
    for _ in range(1, max_iter):
        lambda1, Ax1 = doPowerIteration(A, Ax0)

        if np.abs(1 - lambda0 / lambda1) < rtol:
            return lambda1

        lambda0 = lambda1
        Ax0 = Ax1

    print(f"Relativer Fehler von {rtol} konnte in {max_iter} Schritten "
          "nicht erreicht werden!")
    print(f"Letzte Änderung war {np.abs(1 - lambda0 / lambda1)}.")
    return lambda1


def doInverseIteration(LR_B, Bxk_minus1):
    """Führt einen Schritt der inversen Vektoriteration aus."""
    xk = Bxk_minus1 / np.linalg.norm(Bxk_minus1)
    Bxk = LR_B(xk)
    mu_k = xk.dot(Bxk)
    return mu_k, Bxk


def inverseIteration( A, x0, ev0, max_iter=300, rtol=1e-15, full_output=False ):
    """Inverse Vektoriteration"""
    # bestimme B = A - ev0*I
    # kopiere die Liste, sonst wird A auch verändert!
    indices = A.indices.copy()
    indptr = A.indptr.copy()
    data = A.data.copy()

    n = A.shape[0]
    for i in range(n):
        for k in range(indptr[i], indptr[i+1]):
            if i == indices[k]:     # Hauptdiagonale, Zeile == Spalte
                data[k] -= ev0

    # Erstelle B als Matrix
    B = scipy.sparse.csr_matrix((data, indices, indptr), shape=A.shape)

    # Berechne die LR-Zerlegung von B mit einer Scipy-Methode
    # 'factorized' benötigt das CSC-Format
    #   ( https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html )
    # Da unser B symmetrisch ist, kann man das billig über das Transponieren
    #   erhalten, ohne weitere Rechnungen durchzuführen ( siehe Teil (f) )
    LRvonB = factorized(B.transpose())
    # ==> jetzt können Sie y = B^{-1}x einfach mit y = LRvonB(x) effizient berechnen

    Bx0 = LRvonB(x0)
    mu0 = np.linalg.norm(Bx0) / np.linalg.norm(x0)
    lambda0 = ev0 + 1 / mu0     # Umrechnung von mu zu lambda
    for i in range(1, max_iter):
        mu1, Bx1 = doInverseIteration(LRvonB, Bx0)

        lambda1 = ev0 + 1 / mu1
        if np.abs(1 - lambda0 / lambda1) < rtol:
            if full_output:
                print(f"Keine Änderung in der angegebenen Toleranz nach {i} Schritten.")
            return lambda1

        lambda0 = lambda1
        Bx0 = Bx1

    print(f"Relativer Fehler von {rtol} konnte in {max_iter} Schritten "
          "nicht erreicht werden!")
    return lambda1


def compare_min_max_eigval():
    n = 10
    A = computeCsrA(n)
    N = A.shape[0]
    x = np.ones(N) # x = ( 1 1 1 ... 1 1 )^T
    print("Größter Eigenwert: ", powerIteration( A, x ) )

    print("Vergleich mit scipy: ",
          scipy.sparse.linalg.eigsh(A, k=1, which="LM",
                                    return_eigenvectors=False)[-1])

    # den Schätzwert haben wir mit scipy bekommen. So richtig nützlich ist die
    #   die inverse Iteration irgendwie nicht, man muss die Eigenwerte ja schon kennen :/
    #   (übersehen wir hier etwas?)
    print("Kleinster Eigenwert: ", inverseIteration(A, x, 20.0))
    print("Vergleich mit scipy: ",
          scipy.sparse.linalg.eigsh(A, k=1, which="SM",
                                    return_eigenvectors=False)[-1])

def compute_condition(n):
    A = computeCsrA(n)
    lmax = scipy.sparse.linalg.eigsh(A, k=1, which="LM",
                                     return_eigenvectors=False)[-1]
    lmin = scipy.sparse.linalg.eigsh(A, k=1, which="SM",
                                     return_eigenvectors=False)[-1]
    return np.abs(lmax / lmin)


def test_condition():
    # Kondition wächst quadratisch an, siehe 'linear_fit_condition_number'
    for n in [3, 10, 100]:
        print(f"Kondition von P_{n} ist: ", compute_condition(n))


def linear_fit_condition_number():
    nvals = np.unique(np.logspace(2, 6, 50, base=2.0, dtype=int))
    kappa = np.array([compute_condition(n) for n in nvals])
    a, b = np.polyfit(np.log(nvals), np.log(kappa), deg=1)
    _, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\kappa(A_n)$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(nvals, kappa, ls='', marker='o', label="num. data")
    ax.plot(nvals, np.exp(b) * nvals**a,
            label=fr"Fit ${np.exp(b):.2f}\cdot n^{{{a:.2f}}}$")
    ax.legend()
    plt.show()


def rate_of_convergence_test():
    for n in [10, 100]:
        A = computeCsrA(n)
        N = A.shape[0]
        x = np.ones(N) # x = ( 1 1 1 ... 1 1 )^T

        # größten und kleinsten Eigenwerte über scipy
        lmin2, lmin1 = scipy.sparse.linalg.eigsh(A, k=2, which="SM", return_eigenvectors=False)

        # Schätzwert für die Anzahl an Iterationen ('1e-15 == (l1 / l2)**(2*num_iter)' auflösen)
        rtol = 1e-15
        num_iter = int(0.5 * np.log(rtol) / np.log(lmin1/lmin2))
        print("Schätzwert für die benötigte Anzahl an Iterationen: ", num_iter)

        lambda2 = inverseIteration(A, x, 50.0, rtol=rtol, full_output=True)

        print("Zweitkleinster Eigenwert: ", lambda2)
        print("Vergleich mit scipy: ", lmin2, f"(lambda_min1 = {lmin1})")



compare_min_max_eigval()

test_condition()

linear_fit_condition_number()

rate_of_convergence_test()
