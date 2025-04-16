"""
Visualisierung der Energieeigenwerte in einem periodischen Potential in
  Abhaengigkeit von der Bloch-Phase k (Erster Plotbereich).
Durch einen Klick mit der linken Maustaste im ersten Plotbereich wird eine
  Bloch-Phase ausgewaehlt. Fuer diesen k-Wert werden die Eigenfunktionen
  berechnet. Im zweiten Plotbereich wird dann das Betragsquadrat der EF mit
  einer bestimmten Skalierung ('fak = 0.25') auf Hoehe der jeweiligen
  Eigenenergie dargestellt. Die Darstellung erfolgt ueber 'N_per = 4'
  Perioden, die Berechnung fuer eine Einheitszelle. 
Das Auswaehlen einer weiteren Bloch-Phase loescht alle zuvor gezeichneten EW
  und EF aus dem zweiten Plotfenster. 
Das Potential wird ebenfalls gezeichnet.
"""

import numpy as np
import matplotlib.pyplot as plt
import functools
from scipy.linalg import eigh

def V_gen(A=1.0, mode=1):
    """
    Erzeugt ein Potential mit gegebenen Parametern.
    Auswahl zwischen voreingestellten Potentialen mit 'mode'.
    """
    def V(x):
        if mode == 0:     # periodisches 'Saegezahn' Potential
            return A * (x % A)
        elif mode == 1:   # periodisches Kosinus Potential
            return A * np.cos(2*np.pi*x)
        else:             # periodisches Rechteck Potential
            return A * np.ceil(np.cos(2*np.pi*x))
    return V

def diskretisierung(xmin, xmax, N, retstep=False):
    """Berechne die quantenmechanisch korrekte Ortsdiskretisierung.

    Parameter:
        xmin: unteres Ende des Bereiches
        xmax: oberes Ende des Bereiches
        N: Anzahl der Diskretisierungspunkte
        retstep: entscheidet, ob Schrittweite zurueckgegeben wird
    Rueckgabe:
        x: Array mit diskretisierten Ortspunkten
        delta_x (nur wenn `retstep` True ist): Ortsgitterabstand
    """
    delta_x = (xmax - xmin) / (N + 1)                 # Ortsgitterabstand
    x = np.linspace(xmin+delta_x, xmax-delta_x, N)    # Ortsgitterpunkte

    if retstep:
        return x, delta_x
    else:
        return x

def diagonalisierung(hquer, k, x, V, ew_only=False):
    """Berechne sortierte Eigenwerte und zugehoerige Eigenfunktionen.
    Ist ew_only==True, so werden nur die Eigenwerte berechnet. Es kann dann
    aber auch ein Array an k-Werten uebergeben werden.

    Parameter:
        hquer: effektives hquer
        k: Bloch-Phase
        x: Ortspunkte
        V: Potential als Funktion einer Variable
    Rueckgabe:
        ew: 'sortierte' Eigenwerte (Groesse N*len(k))
        ef: entsprechende Eigenvektoren, ef[:, i] (Groesse N*N)
    """
    delta_x = x[1] - x[0]
    v_werte = V(x)                                  # Werte Potential

    N = len(x)
    z = hquer**2 / (2.0*delta_x**2)                 # Nebendiagonalelem.
    
    h = (np.diag(v_werte + 2.0*z) +
         np.diag(-z*np.ones(N-1), k=-1) +           # Matrix-Darstellung
         np.diag(-z*np.ones(N-1), k=1))             # Hamilton-Operat.
    
    if ew_only == False:       # Nur ein k-Wert 
        phase = np.exp(1j*k)
        # Bloch-Phase einfuegen
        h = h + (np.diag(-z/phase * np.ones(1), k=(N-1)) + 
                 np.diag(-z*phase * np.ones(1), k=-(N-1)))
        
        ew, ef = eigh(h)                            # Diagonalisierung
        ef = ef/np.sqrt(delta_x)                    # WS-Normierung
        return ew, ef
      
    elif ew_only == True:      # Array an k-Werten
        N_k = len(k)
        ew = np.zeros((N_k, N))
        for i in range(N_k):   
            # Berechnung der Ew wie oben, aber fuer alle k-Werte
            phase_i = np.exp(1j*k[i])
            h_i = h + (np.diag(-z/phase_i * np.ones(1), k=(N-1)) + 
                       np.diag(-z*phase_i * np.ones(1), k=-(N-1)))
            ew[i, :] = eigh(h_i, eigvals_only=True)
        return ew

def EwEv_Klickfunktion(event, axis, A, V, V_x, x, h_eff, N, E_max, width,
                       width_V, alpha, alpha_V, fak, colors, x_plot, N_per):
    """
    Durch einen Klick mit der linken Maustaste im ersten Plotbereich wird
    eine Bloch-Phase ausgewaehlt (x-Achse). Fuer diesen k-Wert werden die 
    Eigenfunktionen berechnet. Im zweiten Plotbereich wird dann das
    Betragsquadrat der EF auf Hoehe der jeweiligen Eigenenergie dargestellt.
    Das Auswaehlen einer weiteren Bloch-Phase loescht alle zuvor
    gezeichneten EW und EF aus dem zweiten Plotfenster.
    """
    # Test, ob Klick im ersten Plot mit linker Maustate und Zoom deaktiviert
    tool_mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == axis[0] and tool_mode == '':
        k0 = event.xdata        # x-Position des Mauszeigers als Bloch-Phase
        axis[1].lines = []      # Plotbereich leeren
        ew, ev = diagonalisierung(h_eff, k0, x, V)
        
        anz = np.sum(ew <= E_max)    # Zahl zu plottender Ew
        
        # 'N_per'-fache Kopie von 'ev' fuer den Plot uber 'N_per' Zellen
        ev_plot = np.outer(np.ones(N_per), ev).reshape(N*N_per, N)
        
        # Ausgabe des ausgewaehlten k-Wertes im Plot-Titel
        title_txt_1 = ('Eigenfunktionen und Eigenwerte für k={}'
                       .format(round(k0, 3)))
        axis[1].set_title(title_txt_1, size='small')
        
        # Potential muss erneut gezeichnet werden
        axis[1].plot(x_plot, V_x, lw=width_V, c='k', alpha=alpha_V)
        for i in range(anz):
            # Plot des Erwartungswertes der Energie von 'phi' 
            axis[1].axhline(ew[i], c='k', ls='--', lw=width, alpha=alpha_V)
            # Plot von 'phi' auf Hoehe des Energieerwartungswertes
            axis[1].plot(x_plot, fak * np.abs(ev_plot[:, i])**2 + ew[i],
                         lw=width, c=colors[i % len(colors)], alpha=alpha)
            
    event.canvas.draw()         # Plot aktualisieren 
        
def main():
    print(__doc__)
    
    A = 1.0                        # Parameter des Potentials
    h_eff = 0.2                    # Parameter des Hamiltonian
    mode = 1                       # Auswahl des Potentials
    
    N = 200                        # Anzahl Diskretisierungspunkte
    N_k = 100                      # Anzahl Bloch-Phasen
    N_per = 4                      # Anzahl Perioden Potential
    xmin, xmax = 0, 1              # Einheitszelle
    kmin, kmax = -np.pi, np.pi
    k = np.linspace(kmin, kmax, N_k)           # Bloch-Phasen
    x, delta_x = diskretisierung(xmin, xmax, N, retstep=True)
    # x-Werte fuer den Plot ueber 'N_per' Zellen
    x_plot = np.linspace(xmin, N_per * xmax, N_per * N)
                     
    E_max = 7.0                 # obere Grenze fuer dargestellte EW
    fak = 0.25                  # Skalierung fuer graphische Darstellung
    width = 1.0                 # Linienstaerke der EF im Plot
    width_V = 2.0               # Linienstaerke des Potentials im Plot
    alpha = 0.8                 # Transparenz der EF im Plot
    alpha_V = 0.4               # Transparenz des Potentials im Plot
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']   # feste Farbreihenfolge
    
    V = V_gen(A, mode)               # Erzeugen des Potentials und
    V_x = V(x_plot)                  # Auswertung fuer 'N_per' Zellen
    ew = diagonalisierung(h_eff, k, x, V, ew_only=True) 
    
    # Bestimmung des Index N, sodass E_n(k) > E_max fuer n>N und beliebige k
    E_max_index = max(np.where(ew <= E_max)[1]) + 1
    
    suptitle_txt = ("Eigenwerte und Eigenfunktionen im periodischen " + 
                    "Potential: " + r'$V(x)=A\cdot\cos (2\pi x)$')
    title_txt_0 = (r'A={}, '.format(A) + r'$\hbar_{eff}$=' +
                   '{}'.format(h_eff) + ', $N_k={}$, $N$={}'.format(N_k, N))
    
    # Plotbereiche erstellen
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    fig.suptitle(suptitle_txt)
    
    # Erster Plotbereich
    axis[0].set_title(title_txt_0, size='small')
    axis[0].set_xlabel('k')
    axis[0].set_ylabel(r'$E_n(k)$')
    axis[0].axis([kmin, kmax, min(V_x), E_max])
    for i in range(E_max_index):
        # Plot als kontinuierliche Linien fuer die verschiedenen n 
        # Dadurch gibt es ein paar Werte groesser E_max
        axis[0].plot(k, ew[:, i], c=colors[i % len(colors)], lw=1)
        
    
    # Zweiter Plotbereich
    axis[1].set_xlabel('x')
    axis[1].set_ylabel('V(x), Efkt.')
    axis[1].axis([xmin, N_per * xmax, min(V_x), E_max])
    axis[1].plot(x_plot, V_x, lw=width_V, c='k', alpha=alpha_V,
                 label='Potential')
    axis[1].legend()
    
    # Mausinteraktion
    klick_funktion = functools.partial(EwEv_Klickfunktion, axis=axis, A=A,
                                       V=V, V_x=V_x, x=x, h_eff=h_eff, N=N,
                                       E_max=E_max, width=width,
                                       width_V=width_V, alpha=alpha,
                                       alpha_V=alpha_V, fak=fak,
                                       colors=colors, x_plot=x_plot,
                                       N_per=N_per)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)

    plt.show()

if __name__ == "__main__":
    main()
    """
    Anmerkung: 
    Mit Eigenfunktionen (EF) meine ich hier immer das Betragsquadrat.
a) A = 1:
    Im ersten Plot sieht man auch ohne Zoom, dass die Eigenwerte fuer 
        |k| != 0 oder pi nicht entartet sind. Dementsprechend zeigen die
        zugehoerigen EF auch kein besonderes Verhalten. Fuer groessere
        Eigenwerte naehern sie sich scheinbar sogar konstanten Funktionen an
        (Bei starker Vergroesserung sind trotzdem noch 'Wellen' erkennbar).
    Die EF zu E0 und E1 bleiben genau wie ihre Eigenwerte relativ konstant
        bei einer Variation von k. Ihre Energien sind kleiner als V_max,
        weshalb sie den EF des harmonischen Oszillators aehnlich sehen, da
        das Potential lokal die Form A*x**2 hat.
    In einer relativ kleinen Umgebung (etwa +- 0.1) der k-Werte, bei denen
        sich manche E_n im ersten Plot scheinbar schneiden, waechst die
        Amplitude der entsprechenden EF stark an (bei k = 0 z.B. E3 und E4,
        bei |k| = pi z.B. E2 und E3 bzw. E4 und E5). Nimmt man an, dass die
        k's in gewisser Weise einem Wellenvektor entsprechen, so koennte man
        dieses Verhalten auf die Ausbildung stehender Wellen durch die
        Superposition mit einer Reflektion der k-Welle verstehen. Je kleiner
        die Energiedifferenz zweier Eigenwerte, desto mehr sieht eine der EF
        wie das Negative der anderen aus. Eine der beiden Loesungen hat also
        die groesste Aufenthaltswahrscheinlichkeit bei den Maxima des
        Potentials, die andere bei den Minima. Im Bild der stehenden Wellen
        entspricht das gerade ~sin**2 und ~cos**2. 
    
    
b) A = 1e-5:
    Die EF sind alle nahezu konstant, die Aufenthaltswahrscheinlichkeit ist
        also ueberall quasi gleich. Das entspricht einem freien Teilchen,
        was auch verstaendlich ist, da das Potential bei der gewaehlten
        Amplitude ja vernachlaessigbar klein gegenueber den E_n ist.
    Die Eigenwerte selbst zeigen wieder das 'fast entartende' Verhalten bei
        k = 0 bzw. |k| = pi. Allerdings sind die Energiedifferenzen jetzt
        auch bei E0 und E1 (|k| = pi) sowie E1 und E2 (k = 0) ohne Zoom
        nicht mehr erkennbar. Die Energiedifferenz scheint also in einem
        direkten Zusammenhang zur Staerke des Potentials zu stehen.
    Fuer freie Teilchen gilt E = (hk)**2/2m, was dem parabelfoermigen
        Veraluf der Grundzustandsenergie entspricht (hier haben wir quasi
        freie Teilchen, weil das Potential zwar naeherungsweise 0 ist, aber
        trotzdem die Periodizitaet beruecksichtigt wird). Betrachtet man nun
        die Ueberlagerung aller Parabeln der Form (k + g)**2 fuer g = n*pi
        (n aus Z), so ergibt sich optisch gerade das vorliegende Bild. Die
        Entartung der Energieeigenwerte wird aber durch das Potential
        vermieden, vergleiche die Tunnelaufspaltung bei der Doppelmulde.
    """
    