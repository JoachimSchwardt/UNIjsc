# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np

def berechnePQ(x1, x2):
    """
    Gibt für gegebene Nullstellen wieder die Koeffizienten 'p' und 'q' aus, 
    sodass '(x - x1)(x - x2) == x**2 - 2px + q'.
    """
    q = x1 * x2
    p = (x1 + x2) / 2
    return p, q
    


def pqFormel(p, q):
    """Löst die Gleichung 'x**2 - 2px + q = 0' mit der pq-Formel. """
    root = np.sqrt(p**2 - q)
    x1, x2 = p + root, p - root
    return np.sort([x1, x2])

def vieta(p, q):
    """Löst die Gleichung 'x**2 - 2px + q = 0' mit dem Satz von Vieta. """
    x1 = p + np.sign(p) * np.sqrt(p**2 - q)
    x2 = q / x1
    return np.sort([x1, x2])

def absErr(val, true_val):
    return np.abs(val - true_val)

def relErr(val, true_val):
    return absErr(val, true_val) / true_val

def testQuadratischeGleichung(x1arr, x2arr):
    print("x1      | x2      | x1pq_abs | x2pq_abs | x1pq_rel | x2pq_rel | "
          + "x1vie_abs | x2vie_abs | x1vie_rel | x2vie_rel")
    print("-" * 109)
    for x1, x2 in zip(x1arr, x2arr):
        p, q = berechnePQ(x1, x2)
        
        x1pq, x2pq = pqFormel(p, q)
        x1vieta, x2vieta = vieta(p, q)
        
        x1pq_abs_err = absErr(x1pq, x1)
        x2pq_abs_err = absErr(x2pq, x2)
        x1pq_rel_err = relErr(x1pq, x1)
        x2pq_rel_err = relErr(x2pq, x2)
        
        x1vieta_abs_err = absErr(x1vieta, x1)
        x2vieta_abs_err = absErr(x2vieta, x2)
        x1vieta_rel_err = relErr(x1vieta, x1)
        x2vieta_rel_err = relErr(x2vieta, x2)
        
        msg = (f"{x1:.1e} | {x2:.1e} | {x1pq_abs_err:.2e} | "
               + f"{x2pq_abs_err:.2e} | {x1pq_rel_err:.2e} | " 
               + f"{x2pq_rel_err:.2e} | {x1vieta_abs_err:.2e}  | "
               + f"{x2vieta_abs_err:.2e}  | {x1vieta_rel_err:.2e}  | "
               + f"{x2vieta_rel_err:.2e}")
        print(msg)
    print()

def ln2(n):
    """
    Berechnet ln(2) über die n-te Partialsumme von 
        'ln(2) ~ -sum_{i=1}^{n} (-1)**i / i'
    """
    res = np.sum(np.single(np.resize([1, -1], n) / np.arange(1, n + 1, 1)))
    # ### ohne numpy:
    # res = np.single(0.0)
    # for i in range(1, n+1, 1):
    #     res -= np.single((-1)**i / i)
    return res

def ln2Version2(n, ComputePosNegPartials=False):
    """
    Berechnet ln(2) über die n-te Partialsumme von 
        'ln(2) ~ sum_{i=1}^{n} (-1)**i / i',
    sortiert dabei aber die 'i' so, dass erst alle positiven, und dann alle
    negative Terme addiert.
    
    Falls 'ComputePosNegPartials == True' werden die positiven und negativen 
    Partialsummen einzelen berechnet, und erst am Ende addiert.
    """
    pos_vals = np.single(1 / np.arange(1, n + 1, 2))
    neg_vals = -np.single(1 / np.arange(2, n + 1, 2))
    if ComputePosNegPartials:
        pos_val = np.sum(pos_vals)
        neg_val = np.sum(neg_vals)
        res = np.single(pos_val + neg_val)
    else:
        res = np.single(np.sum(np.append(pos_vals, neg_vals)))
        
    # ### ohne numpy:
    # res = np.single(0.0)
    # for i in range(1, n+1, 2):
    #     res += np.single(1 / i)
    # if ComputePosNegPartials:
    #     neg_vals = np.single(0.0)
    #     for i in range(2, n+1, 2):
    #         neg_vals += np.single(1 / i)
    #     res -= neg_vals
    # else:
    #     for i in range(2, n+1, 2):
    #         res -= np.single(1 / i)
    return res

def ln2Version3(n):
    """
    Berechnet ln(2) über die n-te Partialsumme von 
        'ln(2) ~ 2*sum_{i=1}^{n} (1/3)**(2i + 1) / (2i + 1)'
    """
    res = np.single(0.0)
    power = np.single(1/3)
    for i in range(1, 2*n+1, 2):
        res += np.single(power / i)
        power = np.single(power / 9.0)
        # bereits für 40 jenseits von 1e-16 Genauigkeit ...
        # als reiner 'Python-Loop' ist die Rechenzeit für große n etwas lang,
        # deshalb kürzen wir ein wenig ab 
        if i > 60:    
            break
    return 2 * res



def plot_convergence():
    import matplotlib.pyplot as plt
    def ln2Version4(n):
        """ Version 3 in doppelter Genauigkeit """
        res = 0.0
        power = 1/3
        for i in range(1, 2*n+1, 2):
            res += power / i
            power = power / 9.0
            if i > 60: break
        return 2.0 * res
    
    ln2 = np.log(2.0)
    val = np.array([ln2Version4(n) for n in range(1, 50, 1)])
    diff = np.abs(val - ln2)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_yscale('log')
    ax.plot(diff, c='b', label='abs. error')
    ax.legend()
    plt.show()
    

def testLog2(narr):
    print("n       | PartialSum   | Reordered   | +-PartialSum | Log2Variant")
    print("-" * 65)
    
    true_log2 = np.log(2)
    for n in narr:
        log2partial = ln2(n)
        log2reordered = ln2Version2(n, ComputePosNegPartials=False)
        log2reordered_partial = ln2Version2(n, ComputePosNegPartials=True)
        log2variant = ln2Version3(n)
        
        part_err = relErr(log2partial, true_log2)
        reord_err = relErr(log2reordered, true_log2)
        reord_part_err = relErr(log2reordered_partial, true_log2)
        variant_err = relErr(log2variant, true_log2)
        
        msg = (f"{n:<7} | {part_err:.4e}   | {reord_err:.4e}  | "
               + f"{reord_part_err:.4e}   | {variant_err:.4e}  ")
        print(msg)


def main():
    print("Genauigkeit der Nullstellen-Berechnung mit 'pq' vs. 'vieta':\n")
    n = 16
    x1arr = 10**(-np.arange(1, n+1, 1.0))
    x2arr = np.full(n, 1.0)
    testQuadratischeGleichung(x1arr, x2arr)
    
    n = 12
    x1arr = np.full(n, 5.0)
    x2arr = 5.0 + 10**(-np.arange(1, n+1, 1.0))
    testQuadratischeGleichung(x1arr, x2arr)
    
    print("\nRelative Fehler für log(2)-Berechnung:\n")
    narr = 2**np.arange(2, 24, 1)
    testLog2(narr)
    
    
    return 0

if __name__ == "__main__":
    main()
    plot_convergence()
    """
Beim Verfahren von Vieta wird die Auslöschung vermieden.
Dabei nutzt man aus, dass die 'pq-Formel'
    x1,x2 = -p/2 +- sqrt(p**2/4 - q)
numerisch stabil ist, wenn die beiden Terme dasselbe Vorzeichen haben.
Aus 'q = x1*x2' erhält man dann durch eine ebenfalls numerisch stabile 
Division die zweite Nullstelle. 
    

Die gegebene Rehie für ln(2) ist nicht absolut konvergent, eine Umordnung
führt also nicht immer auf dasselbe Ergebnis.
Numerisch können wir allerdings keinen drastischen Anstieg des relativen 
Fehlers beobachten, auch 'ln2Version2' scheint zu konvergieren.
(eine Vermutung war die Verwendung von 'np.sum', aber die Ergebnisse für
 die reinen Python-Loops scheinen nur unwesentlich abzuweichen.)


'ln2Version3' konvergiert so schnell, dass die 'einfache Genauigkeit' bereits
in der zweiten Zeile der Tabelle erreicht ist. 
'plot_convergence' zeigt, dass diese Reihendarstellung exponentiell 
konvergiert, da sich der absolute Fehler im log-Plot linear verhält.
    """
    
    
    """
    Anmerkung:
    
Für die zweite Nullstellenberechnung mit 
    'x1 = 5.0, x2 = 5.0 + 1e-10'
bekomme ich nur 'nan's, da der Term 'p**2 - q' unter der Wurzel negativ wird.
Interessanterweise tritt das Problem weder auf Julians noch meinem Rechner auf,
sondern nur auf meinem Laptop. Hier passiert folgendes:
    p, q = berechnePQ(x1, x2) 
    -> p = 5.0 + 5e-10
    -> q = 25.0 + 5e-10 + 2e-15
    
    -> p**2 = 25.0 + 5e-10
    -> p**2 - q = -3.55...e-15
Hierbei scheint die 1e-10 relevant zu sein, für die anderen Fehler tritt
das Problem nicht auf.
    """
