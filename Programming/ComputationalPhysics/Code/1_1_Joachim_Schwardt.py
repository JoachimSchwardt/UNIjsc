"""
Created on Tue Mar 31 18:14:20 2020

@author: Joachim
"""
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def standardabb(p0, theta0, K=2.6, n=1000):
    """Erstellt Arrays mit n Elementen für p und theta nach der angegebenen Iterationsvorschrift fuer die Standardabbildung."""
    p, theta = [p0], [theta0]
    for i in range(n):
        # Iterationsformel fuer theta und einordnen in ein 0_2pi Interval mittels modulo
        theta.append((theta[i]+p[i]) % (2*np.pi)) 
        # Iterationsformel fuer p und einordnen in -pi_pi
        p.append(((p[i]+K*np.sin(theta[i+1]))+np.pi) % (2*np.pi) - np.pi) 
    return p, theta # Arrays mit jeweils n+1 Punkten
 
        
def wenn_maus_geklickt(event, ax, K=2.6, n=1000):
    """Plotte Folge mit n Iterationen ausgehend von Mausposition"""
    # Test, ob Klick mit linker Maustaste und im 
    # Koordinatensystem erfolgt sowie ob die Zoomfunktion 
    # des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        p_pos = event.xdata # xpos des Mauszeigers
        theta_pos = event.ydata # ypos des Mauszeigers
        
        # Berechne Iterationen
        p, theta = standardabb(p_pos, theta_pos, K, n)
        # Koordinaten hinzufuegen
        ax.plot(p, theta, marker='.', lw=0, markersize=1, markeredgewidth=0)
        event.canvas.draw() # Plot erstellen

    
def main():
    """Hauptprogramm."""
    K = 2.6 # Kickstaerke
    n = 1000 # Iterationen
    
    fig = plt.figure(figsize=(10, 10)) # Fenstergroesse
    ax = fig.add_subplot(1, 1, 1, aspect=1.0) # quadratisch
    ax.axis([-np.pi, np.pi, 0.0, 2*np.pi])  # Achsengrenzen
    ax.set_xlabel("p")                # Beschriftung x-Achse
    ax.set_ylabel(r"$\dot{\theta}$")  # Beschriftung y-Achse
    plt.suptitle("Standardabbildung")
    plt.title("n = {} und K = {}".format(n, K), size='small') # verwendete Werte fuer n und K
    
    # Achsen in Vielfachen von pi
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))        
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$')) 
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))     

    klick_funktion = functools.partial(wenn_maus_geklickt, ax=ax, K=K, n=n)                                 
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    plt.show()

if __name__ == "__main__":
    main()
    """Eine etwas willkürliche Aufteilung nach Intervallen von Kickstaerken könnte nach meinen Beobachtungen wie folgt aussehen:
0_K_0.5: 
    - abgesehen von K=0 gibt es immer geschlossene Bahnen in der Umgebung vom Zentrum; dieser Bereich dehnt sich für steigende K vor allem in p-Richtung aus.
    - es gibt quasi geradlinige Bahnen mit p ungefaehr konstant. Für kleine K dominiert dieses Verhalten, nimmt aber kontinuierlich ab. Diese Bahnen werden für steigende K zusehends deformiert.  
    - es gibt erste Andeutung chaotischer Dynamik (für K ungefähr 0.5) am Übergang vom Bereich der geschlossenen Bahnen zu den geradlinigen.

0.5_K_1:
    - an den Raendern treten Bereiche geschlossener Bahnen auf, die bei diesen K das Verhalten dominieren. Allerdings sind dabei immer mehrere scheinbar disjunkte Bereiche miteinander verknüpft; setzt man den Startwert in einem beliebigen, so erscheinen Punkte in all diesen Bereichen.
    - die Bereiche chatoischer Dynamik wachsen und werden 'chaotischer', womit ich meine, dass einezelne Startpunkte deutlich größere Bereiche abdecken.
    - die linienartigen Verläufe verschwinden ab K=1 und gehen in geschlossene Bahnen über, die von chaotischen umgeben sind.

1_K_2.1:
    - die Bereiche chaotischer Dynamik fangen an zu dominieren.
    - die Bereiche der zu Beginn linienartigen Verläufe gehen in rein chaotische Dynamik über.
    - die geschlossenen Bahnen in den Randgebieten verschwinden für K>2.1 vollends. 
    - es zeichen sich 6 Bereiche geschlossener Bahnen, die wieder im obigen Sinne miteinander verbunden sind.
    
2.1_K_3:
    - es bewegen sich 4 Bereiche regulärer Dynamik stetig vom Zentrum weg.
    - das Zentrum wird kleiner und schmaler, nimmt aber nach wie vor den Hauptanteil der Bereiche regulärer Dynamik ein.
    - für K>3 verschwinden die 4 Bereiche und weichen chaotischer Dynamik.
    
3_K_4.3:
    - das Zentrum beginnt sich in 2 Bereiche regulärer Dynamik aufzuspalten, die sich etwa in p-Richtung bewegen.
    
4.3_K_6.3:
    - die letzten Bereiche regulärer Dynamik schrumpfen kontinuierlich.
    - ab K=5.6 ist per Mausklick keine nicht-chaotische Bahn mehr aufzufinden.
    - bis zum Intervallende ist nur noch chaotische Dynamik erkennbar.
    
6.3_K_7.6:
    - die Bereiche verändern sich zunächst kaum, verschwinden bei K=7.6 allerdings 'vollständig'. 
    
Zusammenfassung:
    - für wachsende K nimmt die chaotische Dynamik grundsätzlich zu, auch wenn um K=5.6 doch noch einmal Bereiche regulärer Dynamik entstehen.
    - der visuelle Eindruck der Bilder ändert sich im Bereich von 1_K_2.1 drastisch, es verschwinden alle Bereiche komplizierter, aber trotzdem regulärer Dynamik an den Raendern.
    - beim heranzoomen an Uebergaenge von regulärer zu chatoischer Dynamik kann man zumindest eine 'fraktalartige' Struktur erkennen, auf kleineren Skalen gibt es auch immer kleiner werdende Bereiche regulärer Dynamik, die direkt an Chaos grenzen. 
        """
        