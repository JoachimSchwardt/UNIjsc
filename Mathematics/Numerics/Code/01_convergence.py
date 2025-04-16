# Joachim Schwardt + 4768711
# Julian Fleck + 4759587


import numpy as np

# Für die Plots
import matplotlib.pyplot as plt

# Lade den Löser für dünnbesetzte Systeme
from scipy.sparse.linalg import spsolve

# COO/CSR Matrix
import scipy

# Zum Zeit messen
import time


# Externe Wärmezufuhr auf den Mittelpunkt mit Radius 0.2^(1/2)
def f(x):
    return 1 if (x[0]-0.5)**2+(x[1]-0.5)**2 < 0.2 else 0

def get_xy(n=30):
    xs = np.linspace(0, 1, n+1) # inkl. Endpunkte
    ys = np.linspace(0, 1, n+1) # inkl. Endpunkte
    return xs, ys


def get_denseA(xs, ys):
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


    # Erzeuge die Matrix A mit dichtbesetzter Struktur
    denseA = np.zeros((N,N))
    b = np.zeros(N)
    
    val = -1.0 * n**2

    # T_i Blöcke:
    for i in range(n-1):
        # Zeile j in T_i:
        for j in range(n-1):
            idx = index(i,j)
            # Diagonale:
            denseA[idx,idx] = -4.0 * val
            # Nebendiagonale:
            if j<n-2:
                denseA[idx+1,idx] = val
                denseA[idx,idx+1] = val
            
            # -I Blöcke:
            if i<n-2:
                denseA[idx+n-1,idx] = val
                denseA[idx,idx+n-1] = val
                
            # Vektor b:
            b[idx] = f(coord(i,j))
            
    return denseA, b


def timer(func, args):
    t0 = time.perf_counter()

    res = func(*args)

    t1 = time.perf_counter()

    return res, t1 - t0

def get_cooA(denseA, n):
    N = (n-1)**2
   
    # Folgendes ist in Python furchtbar langsam (insb. 'append')
    # (Diese Funktion braucht fast 90% der gesamten Rechenzeit in der 
    #  Laufzeitanalyse)
    """
    data = []
    row = []
    col = []

    for i in range(N):
        for j in range(N):
            if denseA[i, j]:
                data.append(denseA[i, j])
                row.append(i)
                col.append(j)
    """
    
    # ... deshalb hier als reine numpy-Version (ist doch auch schöner :) )
    row, col = np.where(denseA != 0.0)
    data = denseA[row, col]
    
    cooA = scipy.sparse.coo_matrix((data, (row, col)), shape=(N,N))
    return cooA

def plot_solution(sol, xs, ys, n):
    # 'n' aus 'N' bestimmen
    n = int(np.sqrt(sol.shape[0])) + 1

    # Wandle den Lösungsvektor in eine Matrix zum Plotten um:
    plotSol = np.zeros((n-1,n-1))
    for i in range(n-1):
        for j in range(n-1):
            plotSol[i,j] = sol[i*(n-1) + j]

    # Plotte die Wärmeverteilung
    plt.figure(figsize=(10,10))
    plt.pcolormesh(ys[1:n], xs[1:n], plotSol, shading="auto")
    plt.show()

    print("-> Für den Plot eventuell auf den Tab Abbildungen/Plots gehen")

def solve_and_plot(n=30):
    xs, ys = get_xy(n)
    denseA, b = get_denseA(xs, ys)
    
    cooA = get_cooA(denseA, n)
    csrA = cooA.tocsr()
    
    
    denseSol, t_dense = timer(np.linalg.solve, (denseA, b))
    print(f"Lösen des dichtbesetzten Systems dauerte {t_dense:.4f} Sekunden.")
    
    sparseSol, t_sparse = timer(spsolve, (csrA, b))
    print(f"Lösen des dünnbesetzten Systems dauerte {t_sparse:.4f} Sekunden.")
    
    absDiff = np.linalg.norm(denseSol - sparseSol, ord=1)
    print(f"1-Norm der Differenz der Lösungen ist {absDiff:.3e}")
    
    plot_solution(denseSol, xs, ys, n)

def plot_runtime_analysis(Nmin, Nmax, NN):
    # logarithmisch äquidistante 'N'-Werte 
    Narr = np.logspace(np.log2(Nmin), np.log2(Nmax), NN, base=2.0, dtype=int)
    
    
    times_dense = np.zeros(NN)
    times_sparse = np.zeros(NN)
    
    for i in range(NN):
        N = Narr[i]
        n = int(np.sqrt(N)) + 1
        
        xs, ys = get_xy(n)
        denseA, b = get_denseA(xs, ys)
        
        cooA = get_cooA(denseA, n)
        csrA = cooA.tocsr()
        
        _, t_dense = timer(np.linalg.solve, (denseA, b))
        _, t_sparse = timer(spsolve, (csrA, b))
        
        times_dense[i] = t_dense
        times_sparse[i] = t_sparse
    
    a_dense, b_dense = np.polyfit(np.log(Narr), np.log(times_dense), deg=1)
    a_sparse, b_sparse = np.polyfit(np.log(Narr), np.log(times_sparse), deg=1)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$t\ /\ $s')
    
    ax.plot(Narr, times_dense, c='k', ls='', marker='o', 
            label=r'$t_\mathrm{dense}$')
    ax.plot(Narr, np.exp(a_dense * np.log(Narr) + b_dense), c='k')
    ax.text(Narr[NN//2], times_dense[NN//2], f"$\\sim N^{{{a_dense:.3f}}}$", 
            c='k', ha='left', va='top')
    
    ax.plot(Narr, times_sparse, c='b', ls='', marker='o', 
            label=r'$t_\mathrm{sparse}$')
    ax.plot(Narr, np.exp(a_sparse * np.log(Narr) + b_sparse), c='b')
    ax.text(Narr[NN//2], times_sparse[NN//2], f"$\\sim N^{{{a_sparse:.3f}}}$", 
            c='b', ha='left', va='top')
    ax.legend(numpoints=3)
    
    print("Lauzeitverhalten für das dichtbesetzte Problem ist etwa "
          + f"$N^{{{a_dense:.3f}}}$")
    print("Lauzeitverhalten für das dünnbesetzte Problem ist etwa "
          + f"$N^{{{a_sparse:.3f}}}$")

def sparseMv(A, x):
    col = A.indices
    indptr = A.indptr
    data = A.data
    
    N = A.shape[0]     
    v = np.zeros(N)

    # Verglichen mit 'A.dot(x)' ist dieser Loop in Python wieder langsam.
    # Am Algorithmus liegt es aber (hoffentlich) nicht. Mit 'numba' kann man 
    #   die eingebaute Funktion für n < 150 sogar schlagen!
    for i in range(N):
        for j in range(indptr[i], indptr[i+1], 1):
            v[i] += data[j] * x[col[j]]
        
    return v


def test_mv_product(n):
    xs, ys = get_xy(n)
    denseA, b = get_denseA(xs, ys)
    
    cooA = get_cooA(denseA, n)
    csrA = cooA.tocsr()
    
    
    print("Laufzeiten des Matrix-Vektor-Produktes:")
    
    x = np.linspace(0.0, 1.0, denseA.shape[1])    # Testvektor
    v_dense, t_dense = timer(denseA.dot, [x])
    print(f"Dense (inbuilt): {t_dense:.3e} Sekunden.")
    
    v_sparse, t_sparse = timer(csrA.dot, [x])
    print(f"Sparse (inbuilt): {t_sparse:.3e} Sekunden.")
    
    v_loops, t_loops = timer(sparseMv, (csrA, x))
    print(f"Sparse (loops): {t_loops:.3e} Sekunden.")
    
    absDiff = np.linalg.norm(v_loops - v_sparse, ord=1)
    print(f"1-Norm der Differenz der Lösungen ist {absDiff:.3e}")

def main():
    solve_and_plot(n=30)
    print("_" * 80 + "\n")
    
    # Achtung, die Laufzeitanalyse dauert recht lange...
    # So ganz ergibt sich das kubische Verhalten nicht, wir kommen mit den
    #   Werten hier auf etwa 'O(N**2.4)' und 'O(N**1.3)'. 
    # Die genauen Exponenten hänge allerdings auch stark von den gewählten
    #   Bereichen für 'N' ab.
    #   Eventuell braucht man deutlich größere 'N'?
    plot_runtime_analysis(Nmin=8000, Nmax=10000, NN=5)    
    
    print("_" * 80 + "\n")
    test_mv_product(n=130)
    
    
    return 0
    

if __name__ == "__main__":
    # main()
    1
