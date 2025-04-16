"""
Dynamische Darstellung des Zeitverhaltens eines Gauss'schen Wellenpaketes. 
"""

import numpy as np
import quantenmechanik as qm
import matplotlib.pyplot as plt
import functools

def V_gen(A=0.06, mode=1):
    """
    Erzeugt ein Potential mit gegebenen Parametern.
    Auswahl zwischen voreingestellten Potentialen mit 'mode'.
    """
    def V(x):
        if mode == 0:     # harmonischer Oszillator
            return 0.5 * x**2
        elif mode == 1:   # asymmetrisches Doppelmuldenpotential
            return x**4 - x**2 - A * x
        else:             # Potentialkasten
            return 0*x
    return V
        
def phi_t0(x, x0, p0, Delta_x, h_eff):
    """
    Gauss'sches Wellenpaket bei t=0 als Funktion des Ortes.
        x0 = Mittlere Ort
        p0 = Mittlerer Impuls
        Delta_x = Ausdehnung im Ortsraum
        h_eff = Parameter des Hamiltonian
    """
    norm_fak = (2 * np.pi * Delta_x**2)**(1/4)
    gauss_fak = np.exp(-(x - x0)**2 / (2 * Delta_x)**2)
    return np.exp(1j * p0 * x / h_eff) * gauss_fak / norm_fak

def phi_dynamic(event, ax, ew, ev, fak, x, width, alpha, p0, Delta_x, h_eff,
                t_max, N_t):
    """
    Durch einen Klick mit der linken Maustaste wird die Zeitentwicklung
    eines Gauss'schen Wellenpaketes berechnet und dynamisch auf Hoehe des
    Energieerwartungswertes dargestellt. Der mittlere Ort wird durch die 
    x-Koordinate des Mauszeigers bestimmt. Die Energieeigenwerte des
    Potentials sind als gestrichelte Linien dargestellt, die Eigenfunktionen
    entsprechend auf Hoehe der jeweiligen Energie. 
    """
    # Test, ob Klick im Plotfenster mit linker Maustate und Zoom deaktiviert
    tool_mode = event.canvas.toolbar.mode
    dx = x[1] - x[0]
    if event.button == 1 and event.inaxes and tool_mode == '':
        x0 = event.xdata    # x-Position des Mauszeigers als Startwert 'x0'
    
        # Berechnung der Entwicklungskoeffizienten
        phi_t0_array = phi_t0(x, x0, p0, Delta_x, h_eff)
        c = dx * np.dot(np.conjugate(np.transpose(ev)), phi_t0_array)
        # Energieerwartungswert
        energie_qm = np.dot(abs(c)**2, ew)
        
        # Zeitentwicklung
        t = np.linspace(0, t_max, N_t)
        # outer(N, M) = NxM-Matrix
        phase_t = np.exp(-1j * np.outer(t, ew) / h_eff)
        # NxM1 * M2 = NxM1  mit M1_i *= M2_i  (dim(M1 == M2))
        c_t = c * phase_t
        # dot(a, b) = 'normale' Matrixmultiplikation (Zeile x Spalte)
        phi_t = np.dot(c_t, np.transpose(ev))
        
        # Ausgabe der L2-Norm der Differenz von phi_t0 und der Zerlegung
        # von phi in EF. Beachte Faktor 'dx' (Integral-L2 vs. Summe-l2)
        print('Differenznorm bei x0 = {}: {}'
              .format(round(x0, 3), np.sqrt(dx * 
                  np.sum(np.abs(phi_t0_array - phi_t[0, :])**2))))
        
        # Plot des Erwartungswertes der Energie von 'phi' 
        ax.axhline(energie_qm, c='k', ls='--', lw=width, alpha=alpha)
        # Plot von 'phi' auf Hoehe des Energieerwartungswertes
        phi_plot = ax.plot(x, fak * np.abs(phi_t[0, :])**2 + energie_qm,
                           c='k')
        for count in range(N_t):
            phi_t_plot = fak * np.abs(phi_t[count, :])**2 + energie_qm
            phi_plot[0].set_ydata(phi_t_plot)
            event.canvas.flush_events()
            event.canvas.draw()

def main():
    print(__doc__)
    print(phi_dynamic.__doc__)
    
    # Parameter der Wellenfunktion bei t=0
    p0 = 0.0                  # mittlerer Impuls
    Delta_x = 0.1             # 'Breite' des Anfangszustandes
    
    A = 0.06                  # Parameter des Potentials
    h_eff = 0.07              # Parameter des Hamiltonian
    mode = 1                  # Auswahl des Potentials
    
    xmin, xmax = -1.5, 1.5    # Potential 'unendlich' ausserhalb
    N = 350                   # Anzahl Diskretisierungspunkte
    E_max = 0.15              # obere Grenze fuer dargestellte EW
    fak = 0.01                # Skalierung fuer graphische Darstellung
    width = 1.0               # Linienstaerke der EF im Plot
    alpha = 0.8               # Transparenz der EF im Plot
    
    t_max = 1e1               # maximaler betrachteter Zeitpunkt ab t=0
    N_t = 100                 # Anzahl an Zeitpunkten
    
    suptitle_txt = ("Zeitentwicklung einer Wellenfunktion im Potential: " +
                    r'$V(x)=x^4-x^2-Ax$')
    title_txt = (r'A={}, $p_0$={}, $\Delta x$={}, '.format(A, p0, Delta_x) + 
                 '$\hbar_{eff}$=' + '{}, N={}, '.format(h_eff, N) +
                 r'$x_{min}$=' + '{}, '.format(xmin) + r'$x_{max}$=' + 
                 '{}, '.format(xmax) + r'$t_{max}$=' + 
                 r'{}, $N_t$={}'.format(t_max, N_t))
    
    # Bestimmung der EW und EV mit 'qm'
    V = V_gen(A, mode)         # Erzeugen des Potentials
    x_array, dx = qm.diskretisierung(xmin, xmax, N, retstep=True)
    ew, ev = qm.diagonalisierung(h_eff, x_array, V)
    ev_plot = np.abs(ev)**2    # Absolut-Quadrat der EV fuer Plot
    
    # Plotbereich erstellen
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(suptitle_txt)
    ax = fig.add_subplot(1, 1, 1)
    
    qm.plot_eigenfunktionen(ax, ew, ev_plot, x_array, V, width=width,
                            Emax=E_max, fak=fak, betragsquadrat=False,
                            basislinie=True, alpha=alpha, title=title_txt)
    
    klick_funktion = functools.partial(phi_dynamic, ax=ax, ew=ew, ev=ev,
                                       fak=fak, x=x_array, width=width,
                                       alpha=alpha, p0=p0, Delta_x=Delta_x,
                                       h_eff=h_eff, t_max=t_max, N_t=N_t)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    plt.show()

if __name__ == "__main__":
    main()
    """
a) p0 = 0.0:
    Minimum:
    Der Energieerwartungswert liegt geringfuegig ueber dem Energieeigenwert        
        der Grundschwingung und unterhalb des anderem, energetisch hoeher
        liegendem lokalem Minimum. Die Form des Wellenpaketes sieht der
        zugehoerigen Eigenfunktion bei t=0 sehr aehnlich und bleibt auch
        ueber laengere Zeitraeume (t_max=1000) erhalten. Das heisst,
        abgesehen von kleineren Schwankungen sieht phi(t=t_max) in etwa so
        aus wie der Ausgangszustand.
    Maximum:
    Der Energieerwartungswert liegt knapp unter dem der fuenften Anregung
        und oberhalb des lokalen Maximums des Potentials. Das Wellenpaket
        zerfliesst auf der beim Minimum betrachteten Zeitskala quasi sofort
        und zeigt eine deutlich kompliziertere Bewegung. phi(t=t_max) sieht
        der Eigenfunktion der fuenften Anregung aehnlich (beide haben sechs
        Maxima).
    Bei kleineren Zeitskalen (t_max=5) sieht man, dass das Wellenpaket
        nahezu symmetrisch und gleichmaessig zerfliesst und dann an den
        'Potentialwaenden' reflektiert wird, wodurch eine Art stehende Welle
        entsteht. 
b) p0 = 0.3:
    Minimum:
    Der Energieerwartungswert liegt diesmal etwas ueber dem anderen lokalen
        Minimum. Die 'kleineren Schwankungen' sind hier deutlich staerker
        und es gibt eine nicht verschwindende Wahrscheinlichkeit, das
        Teilchen im anderen Minimum zu finden. 
    Maximum:
    Der Energieerwartungswert liegt ueber dem der fuenften Anregung und das
        Wellenpaket hat bei t=t_max jetzt sieben statt sechs Maxima. Man
        koennte die Hypothese aufstellen, dass das Wellenpaket auf lange
        Sicht so viele Maxima wie die naechsthoehere Eigenfunktion haben
        wird. Auf sehr kleinen Zeitskalen (t_max=1.0) sieht man jetzt
        deutlich, dass sich das Wellenpaket in positive x-Richtung bewegt.
        Da p0 > 0  ist, entspricht das zumindest qualitativ dem Verhalten
        eines klassischen Teilchens.
c) A = 0.00:
Das 'mikroskopische' Verhalten sieht qualitativ genau so aus wie im Fall   
    a), also eine willkuerlich wirkende 'Zitterbewegung'. Bei
    t_max=5500 kann man allerdings beobachten, dass die
    Aufenthaltswahrscheinlichkeit in dem Minimum, in dem das Paket
    gestartet wurde langsam verschwindet und in dem anderen dafuer
    langsam steigt. Bei dem gewaehlten t_max ist der Ausgangszustand in
    etwa wieder erreicht, die Oszillation zwischen den Minima hat also
    eine Periodendauer von etwa T = 5500.
    Setze fuer die Rechnung h_eff = 1. Die Anteile der EF sollen gleich
        sein. Sei A eine Normierungskonstante:
    |psi(x,t)|**2 = A|phi_0(x)*exp(-iE_0 t) + phi_1(x)*exp(-iE_1t)|**2 = 
                  = A|phi_0(x)|**2 + A|phi_1(x)|**2 +
                       + A*phi*_0(x)phi_1(x)*exp(i(E_0-E_1)t) +
                       + A*phi_0(x)phi*_1(x)*exp(-i(E_0-E_1)t)
                  = A|phi_0(x)|**2 + A|phi_1(x)|**2 + 
                       + 2A*phi_0(x)phi_1(x)*cos((E_0-E_1)t)
    Die Periode sollte also T = 2pi * h_eff / |E_0-E_1| sein. 
    Die Eigenwerte sind E_0 = -0.18273...  und E_1 = -0.18265...
    Also folgt T = 5673.08... in guter Uebereinstimmung mit der numerisch
        erhaltenen Zeit T = 5500.
    """