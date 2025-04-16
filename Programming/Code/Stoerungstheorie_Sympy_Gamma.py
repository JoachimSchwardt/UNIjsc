"""
Zeitabhaengige Stoerungsrechnung. Harmonischer Oszillator mit Stoerung:
    H_1 = alpha * q     (fuer 't in [0, tau]', q = Ortskoordinate)
    Das System sei bei 't=0' im Eigenzustand 'N'. 
    Setze 'x := alpha*tau * sinc(omega*tau / 2)' als die einheitenlose 
    Staerke der Stoerung. Dabei ist 'tau' die Dauer der Stoerung.
Analytische Loesung liefert folgende Uebergangswahrscheinlichkeiten: 
    P_n := |<n|psi>|**2 = N! / n! * exp(-x**2) * |f(x)|**2
    f(x) = sum_k=max(0,n-N)**n [binom(n, k) * (-1)**k * x**2k / (N-n+k)!]
"""

import numpy as np
from scipy.special import binom
from matplotlib import pyplot as plt

def Coefficients(k, n, N):
    # return (-1)**k * binom(n, n-N+k) / np.math.factorial(k)
    return (-1)**k * binom(n, k) / np.math.factorial(N-n+k)

def Transition(x, N, n):
    multiplier = np.exp(-x**2) * np.math.factorial(N) / np.math.factorial(n)
    ans = 0
    # for k in range(int((abs(N-n) + N-n)/2), N+1, 1):
    #     ans += x**(2*k + n- N) * Coefficients(k, n, N)
    # return multiplier * ans**2 
    for k in range(int((abs(n-N) + n-N)/2), n+1, 1):
        ans += x**(2*k + N - n) * Coefficients(k, n, N)
    return multiplier * ans**2 


def main():
    print(__doc__)    
    
    # Parameter fuer Plot festlegen
    N_val = 20                           # Ausgangszustand
    n_chksum = 20                       # Anzahl Terme fuer Summe P_n <= 1
    n_max = 20                           # Abstand Termindizes fuer Plot
    x_max = 3                           # Maximale Stoerungsstaerke fuer Plot
    x_vals = np.linspace(0, x_max, 301)
    
    # Plotbereich erstellen
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([0, x_max, 0, 1])
    ls_array = ['--', '-', ':'] 
    c_array = ['b', 'r', 'g', 'purple', 'cyan', 'orange']
    
    # Funktion plotten
    check_sum = np.zeros_like(x_vals)
    
    for n_val in range(max([0, N_val - n_chksum]), N_val + n_chksum + 1, 1):
        # Array fuer Plot der Ubergangswahrscheinlichkeiten berechnen
        Transition_plot = Transition(x_vals, N_val, n_val)
        # Aufsummieren aller Terme soll konstant 1 ergeben
        check_sum = check_sum + Transition_plot    
        
        # Darstellung nur fuer Terme nahe dem Ausgangszustand 
        if abs(N_val - n_val) <= n_max:
            col_n = c_array[abs(N_val - n_val) % 6]         # Farbcode
            ls_n = ls_array[np.sign(N_val - n_val) + 1]     # Linestyle
            ax.plot(x_vals, Transition_plot, c=col_n, lw=1, ls=ls_n,
                    label=r'$n={}$'.format(n_val))
    ax.plot(x_vals, check_sum, lw=1, c='k')       # Plot etwa konstant 1
    
    # # Naeherungen fuer kleine x
    # ax.plot(x_vals, x_vals**2 * N_val, c='k', lw=0.7)       # N+1 Uebergang
    # ax.plot(x_vals, x_vals**2 * (N_val+1), c='k', lw=0.7)   # N-1 Uebergang
    
    # Grid und Legende hinzufuegen
    ax.grid(True)
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    main()


