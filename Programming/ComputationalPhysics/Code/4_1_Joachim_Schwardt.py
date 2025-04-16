"""Fuer die dimensionslose Hamilton-Funktion H(x,p,t) = p**2/2 + V(x,t) mit
dem angetriebenem Doppelmuldenpotential V(x,t) = x**4 - x**2 + x*(A +
B*sin(wt)) werden in zwei Plotbereichen die Trajektorie (x(t), p(t)) und die 
stroboskopische Darstellung (x(t_i), p(t_i)) (mit i=2*pi*i/w) fuer N 
Perioden des Antriebs dargestellt. 
Fuer B=0 werden Hoehenlinien zu geeigneten Energien in beiden Fenstern mit
'contour' dargestellt. Hellere Farben entsprechen dabei hoeheren Energien.

Beim Klick mit der linken Maustaste in einem der beiden Plotfenster werden
Startwerte ausgehend von der Mausposition verwendet. 

Voreingestellt sind A=0.1, B=0.1, w=1, N=200. """

import numpy as np
import functools
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def abl(y, t, A=0.1, B=0.1, w=1):
    """
    Die Hamilton-DGL fuer y = [x, p] lauten hier: 
        x_dot = H_dp = p
        p_dot = -H_dx = 2*x - 4*x**3 - A - B*sin(wt)
    Es ist dabei x=y[0] und p=y[1]. 
    """
    return np.array([y[1], 2*y[0] - 4*y[0]**3 - A - B*np.sin(w*t)])

def wenn_maus_geklickt(event, axes, zeiten, A=0.1, B=0.1, w=1, m=1, N=200):
    """
    Bei einem Klick mit der linken Maustaste in einem der beiden
    Plotbereiche bei deaktiviertem Zoom wird ein Startarray y0 = [x0, p0]
    ausgehend on der Mausposition festgelegt. Die Loesung der Hamilton-DGL
    wird dann in 'axes[0]' als Trajektorie und in 'axes[1]' als diskrete
    Punkte dargestellt.
    """
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        x_pos = event.xdata    # xpos des Mauszeigers
        p_pos = event.ydata    # ypos des Mauszeigers
        y0 = np.array([x_pos, p_pos])   # Startwerte
        
        # ODE berechnen
        y_t = odeint(abl, y0, zeiten, args=(A, B, w))
        x_t, p_t = y_t[:, 0], y_t[:, 1]   # Aufteilen in Arrays mit x und p
        ind = np.arange(0, m*N + 1, m)    # Indizes der 'stroboskop' Zeiten 
        
        # Punkte plotten als Linien (0) und diskret (1)
        axes[0].plot(x_t, p_t, lw=0.6)
        axes[1].plot(x_t[ind], p_t[ind], marker='o', lw=0, ms=2, mew=0)
        
        event.canvas.draw()    # Plot aktualisieren
    
def main():
    print(__doc__)
    N = 200    # Anzahl Perioden
    m = 300    # Anzahl Zeitpunkte pro Periode
    H_par = 200   # Anzahl Punkte in x,y-Richtung fuer Hoehenlinien
    N_lvls = 8    # Anzahl Energieniveaus 
    
    # Parameter des Potentials
    A = 0.1   # Amplitude (-x)-Verscheibung
    B = 0.1   # Amplitude Antrieb
    w = 1     # Frequenz Antrieb
    
    H_txt = r"$V(x,t)=x^4 - x^2 + x\, (A + B\, \sin(\omega t))$ "
    para_txt = (" mit Parametern: A={}, B={}, $\omega$={}"
                .format(round(A, 3), round(B, 3), round(w, 3)))
    
    zeiten = np.linspace(0, 2*N*np.pi / w, m*N + 1)   # m*N + 1 Zeitpunkte
    
    levels = np.linspace(0, 1.5, N_lvls)   # Energiewerte fuer Hoehenlinien
    xmin, xmax = -1.5, 1.5
    pmin, pmax = -2.5, 2.5
    
    x, p = np.linspace(xmin, xmax, H_par), np.linspace(pmin, pmax, H_par)
    x2d, p2d = np.meshgrid(x, p)
    
    # Hamilton fuer B=0 entspricht Energie, da nicht explizit zeitabhaengig
    H_xp = p2d**2/2 + x2d**4 - x2d**2 + A*x2d
        
    # Zwei Plotbereiche erstellen
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axes = [ax1, ax2]   # Vereinfacht das Hinzufuegen von Plotfenstern
    plt.suptitle("Hamilton mit angetriebenem Doppelmuldenpotential\n" +
                 H_txt + para_txt, size='small')
    title_dict = {ax1:'Trajektorie', ax2:'stroboskopische Darstellung'}
    
    for ax in axes:
        # Festlegung Plotbereich (orientiert an den Hoehenlinien)
        ax.axis([xmin, xmax, pmin, pmax])
        ax.set_xlabel("x")   # Beschriftung x,y-Achse
        ax.set_ylabel("p")   
        ax.set_title(title_dict[ax] + " im Phasenraum mit N={} Perioden"
                     .format(N), size='x-small')
        # Plot der Hoehenlinien
        ax.contour(x2d, p2d, H_xp, levels=levels, cmap='plasma')
        
    klick_funktion = functools.partial(wenn_maus_geklickt, axes=axes, N=N,
                                       zeiten=zeiten, A=A, B=B, w=w, m=m)                                 
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    plt.show()
    
if __name__ == "__main__":
    main()
    """
a) B=0 --> Energieerhaltung --> alle Trajektorien entlang Hoehenlinien:
    Bei p=0 gibt es drei x-Werte (Extrema des Potentials), bei denen das
        Teilchen in Ruhe ist. Die Minima sind dabei stabile Fixpunkte und
        das Maximum ein instabiler Fixpunkt.
    Fuer Energien, die kleiner sind als das Potential einer der Mulden
        bleibt das Teilchen im Potentialtopf gefangen. 
    Im Ortsraum entspricht das einer Sinus-artigen Bewegung zwischen den
        beiden Umkehrpunkten. Fuer kleine Werte von 'A' liegen die Bahnen
        jeweils ueber oder unter der x-Achse. 
    Bei Energien die groesser sind als die Potentialbarriere zwischen den
        Mulden, bewegt sich das Teilchen zwischen Umkehrpunkten 'ausserhalb'
        der Mulden. 
    Alle Bewegungen sind regulaer, es tritt keine chaotische Dynamik auf.
    
b1) B=0.1:
    Der instabile Fixpunkt ist verschwunden. Das ist nicht weiter
        verwunderlich, da ja der Antrieb immer fuer einen kleinen Auslenkung
        sorgt. Bei den Minima des Potentials gibt es bei diesem 'B' immer
        noch Bahnen, die im Ortsraum quasi konstant sind. 
    In der Naehe der oben beschriebenen Potentialsbarriere tritt chaotische
        Dynamik auf. In diesem Bereich dienen die Hoehenlinien nicht mehr
        als Orientierung fuer das Verhalten des Systems.
    Oberhalb bestimmter Energien ist die Dynamik dann wieder regulaer. Die
        Trajektorien liegen hier wieder entlang von Hoehenlinien, die
        Bewegung oszilliert allerdings entlang dieser. Beim Plot vieler
        Perioden zeigt sich das durch optisch deutlich 'breitere' Bahnen.
b2) 3  Nachkommastellen Genauigkeit:
    Bei x = 0.031 und p = -0.567 gibt es einen periodischen Orbit mit
        Periode t* = 2pi / w * n = 2pi = 6.283    (n=1) 
    """