""" Husimidarstellung von Gauss-Wellenpaketen
    im asymmetrischen Doppelmuldenpotential $V(x) = x^4 - x^2 + Ax$.

NOTE: Performance Optimierungen mit 'blit'.
"""

import functools
from time import perf_counter
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.linalg import eigh
try:
    import mpl_special
    MPL_SPECIAL_IMPORT = True
except ImportError:
    print("Module 'mpl_special' not available or not in $PATH!")
    MPL_SPECIAL_IMPORT = False


def as_1d_array(value, size):
    """Convert the given input to a 1d array of the given 'size'"""
    value = np.asarray(value)
    if value.ndim >= 2:
        raise ValueError(f"Input {value} must be scalar or 1-dim. but has "
                         f"shape {value.shape} with dimension {value.ndim}")
    if value.ndim == 0:
        value = np.full(size, value)
    return value



def tick_in_limits(tick, limits):
    """Return True if the tick is within the given limits"""
    return limits[0] <= tick.get_loc() <= limits[1]


def diskretisierung(x_min, x_max, num_points, retstep=False):
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
    delta_x = (x_max - x_min) / (num_points + 1)
    x_werte = np.linspace(x_min + delta_x, x_max - delta_x, num_points)

    if retstep:
        return x_werte, delta_x
    return x_werte


def diagonalisierung(h_quer, x_values, potential):
    """Berechne sortierte Eigenwerte und zugehoerige Eigenfunktionen.

    Parameter:
        hquer: effektives hquer
        x: Ortspunkte
        V: Potential als Funktion einer Variable
    Rueckgabe:
        ew: sortierte Eigenwerte (Array der Laenge N)
        ef: entsprechende Eigenvektoren, ef[:, i] (Groesse N*N)
    """
    delta_x = x_values[1] - x_values[0]
    potential_werte = potential(x_values)
    z_parameter = h_quer**2 / (2.0 * delta_x**2)          # Nebendiagonalelem.
    hamilton_matrix = (np.diag(potential_werte + 2.0 * z_parameter) +
                       np.diag(np.full(x_values.size-1, -z_parameter), k=-1) +
                       np.diag(np.full(x_values.size-1, -z_parameter), k=1))

    eigen_werte, eigen_fkt = eigh(hamilton_matrix)
    eigen_fkt /= np.sqrt(delta_x)                         # WS-Normierung
    return eigen_werte, eigen_fkt


def plot_eigenfunktionen(ax, eigen_werte, eigen_fkt, x_werte, potential,
                         width=1, e_max=0.2, fak=0.01,
                         betragsquadrat=False, plot_basislinie=True, alpha=1.0,
                         title=None):
    """Darstellung der Eigenfunktionen.

    Dargestellt werden die niedrigsten Eigenfunktionen 'ef' im Potential 'V'(x)
    auf Hoehe der Eigenwerte 'ew' in den Plotbereich 'ax'
    (Bereitstellung im aufrufenden Programm z.B. durch
    ``ax = fig.add_subplot(111)'').
    Die Eigenwerte werden hierbei als sortiert angenommen.

    Optionale Parameter:
        width: (mit Default-Wert 1) gibt die Linienstaerke beim Plot der
            Eigenfunktionen an. width kann auch ein Array von Linienstaerken
            sein mit einem spezifischen Wert fuer jede Eigenfunktion.
        Emax: (mit Default-Wert 0.15) legt die Energieobergrenze
            fuer den Plot fest.
        fak: ist ein Skalierungsfaktor fuer die graphische Darstellung
            der Eigenfunktionen.
        betragsquadrat: gibt an, ob das Betragsquadrat der Eigenfunktion oder
            die (reelle!) Eigenfunktion selbst dargestellt wird.
        basislinie: gibt an, ob auf Hoehe der jeweiligen Eigenenergie eine
            gestrichelte graue Linie gezeichnet wird.
        alpha: gibt die Transparenz beim Plot der Eigenfunktionen an (siehe
            auch Matplotlib Dokumentation von plot()). alpha kann auch ein
            Array von Transparenzwerten sein mit einem spezifischen Wert
            fuer jede Eigenfunktion.
        title: Titel fuer den Plot.
    """
    if title is None:
        title = "Asymm. Doppelmuldenpotential"

    potential_werte = potential(x_werte)

    # konfiguriere Ortsraumplotfenster
    ax.autoscale(False)
    ax.axis([np.min(x_werte), np.max(x_werte), np.min(potential_werte), e_max])
    ax.set_xlabel(r'$x$')
    if betragsquadrat:
        ax.set_ylabel(r'$V(x)\ \rm{,\ \|Efkt.\|^{2}\ bei\ EW}$')
    else:
        ax.set_ylabel(r'$V(x)\ \rm{,\ Efkt.\ bei\ EW}$')
    ax.set_title(title)

    ax.plot(x_werte, potential_werte, linewidth=2, color='0.7')
    anz_eigen_werte = np.sum(eigen_werte <= e_max)

    if plot_basislinie:
        for i in range(anz_eigen_werte):
            ax.plot(x_werte, eigen_werte[i] + np.zeros(len(x_werte)),
                    ls='--', color='0.7')

    width = as_1d_array(width, anz_eigen_werte)
    alpha = as_1d_array(alpha, anz_eigen_werte)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']           # feste Farbreihenfolge
    for i in range(anz_eigen_werte):
        if betragsquadrat:
            y_werte = eigen_werte[i] + fak * np.abs(eigen_fkt[:, i])**2
        else:
            y_werte = eigen_werte[i] + fak * eigen_fkt[:, i]
        ax.plot(x_werte, y_werte, linewidth=width[i], alpha=alpha[i],
                color=colors[i % len(colors)])


def potential_fkt(x_val, asymmetrie):
    """Potentialfunktion fuer die asymmetrische Doppelmulde mit Parameter A."""
    return x_val**4 - x_val**2 - asymmetrie * x_val


def coherent_state(x_val, p_0, x_0, h_quer, asym):
    """Kohaerenter Zustand im Punkt (x0, p0), berechnet am Ort x."""
    delta_x = asym * np.sqrt(h_quer / 2.0)                # Breite in x
    normalization = 1.0 / np.sqrt(np.sqrt(2.0 * np.pi) * delta_x)
    gaussian = np.exp(-(x_val - x_0)**2/(4*delta_x**2))
    phase = np.exp(1j*p_0 * x_val / h_quer)
    return normalization * gaussian * phase


def husimi_koeffizienten(x_max, p_max, steps, h_quer, asym, x_werte):
    """Berechne Husimi-Koeffizienten auf einem Gitter.

    Das Gitter ist geben durch [-xmax, xmax] \times [-pmax, pmax] und
    wird so gesetzt, dass bei einem imshow mit diesem extent die
    berechneten Punket dem Mittelpunkt der dargestellten Pixel
    entsprechen.
    """
    x_0, dx_0 = np.linspace(-x_max, x_max, steps, retstep=True, endpoint=False)
    p_0, dp_0 = np.linspace(-p_max, p_max, steps, retstep=True, endpoint=False)
    x_0 += dx_0 / 2
    p_0 += dp_0 / 2

    coefficients = np.zeros((steps, steps, x_werte.size), dtype=complex)
    for x_ind in range(steps):
        for p_ind in range(steps):
            coefficient = coherent_state(x_werte, p_0[p_ind], x_0[x_ind], h_quer, asym)
            coefficients[p_ind, x_ind] = coefficient
    return coefficients


def husimi_darstellung(coefficients, phi, x_werte, h_quer):
    """Berechne Husimi-Darstellung von phi mit den Husimi-Koeffizienten c."""
    delta_x = x_werte[1] - x_werte[0]
    return np.abs((coefficients.conj() @ phi))**2 * delta_x / (2*np.pi * h_quer)


def kontur_fkt(x_max, p_max, k_steps, potential):
    """Berechne Funktion fuer Konturplot."""
    x_werte = np.linspace(-x_max, x_max, k_steps)
    p_werte = np.linspace(-p_max, p_max, k_steps)

    x_grid, p_grid = np.meshgrid(x_werte, p_werte)
    hamilton_grid = 0.5*p_grid**2 + potential(x_grid)
    return hamilton_grid, p_grid, x_grid


def husimi_dynamisch(event, ax_phase, ax_ort, eigen_werte, eigen_fkt,
                     x_werte, potential, faktor, h_quer, e_max, asym,
                     koeffizienten, husimi_achsen, zeiten):
    """Auswahl des Schwerpunkts eines Wellenpakets im Phasenraum + Dynamik."""
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax_phase and mode == '':
        # Anmerkung: im Aufgabenblatt wird nur die Husimi-Darstellung
        # gefordert!
        for artist in ax_ort.lines:
            artist.remove()                              # entferne alle Linien

        plot_eigenfunktionen(ax_ort, eigen_werte, eigen_fkt, x_werte, potential,
                             e_max=e_max, width=1, fak=faktor)

        # erzeuge Startwellenpaket, berechne Entwicklungskoeffizienten
        # und den Energieerwartungswert
        delta_x = x_werte[1] - x_werte[0]
        phi0 = coherent_state(x_werte, event.ydata, event.xdata, h_quer, asym)
        coefficients = (eigen_fkt.T.conj() @ phi0) * delta_x
        e_0_qm = np.abs(coefficients)**2 @ eigen_werte

        # Plot von Wellenpaket und Husimidarstellung fuer t = 0
        wellenpaket = ax_ort.plot(x_werte, e_0_qm + faktor * np.abs(phi0)**2,
                                  linewidth=3, color='k')

        # # Note: this one is not really essential.
        # # (Only if the user clicks a couple of 1000 times ... ;)
        # ax_phase.images = []                              # loesche Bilder
        # berechne Husimi
        husimi = husimi_darstellung(koeffizienten, phi0, x_werte, h_quer)
        husimi_plot = ax_phase.imshow(husimi,
                                      extent=husimi_achsen,
                                      interpolation='nearest',
                                      origin='lower',
                                      cmap='cividis')

        # Plot von Wellenpaket und Husimidarstellung fuer t > 0
        fig = plt.gcf()
        axes = [ax_ort, ax_phase]

        # Hintergrund, Konturlinien und Ticks speichern
        # https://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow
        # https://matplotlib.org/stable/tutorials/advanced/blitting.html
        backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
        contour_lines = [child for child in ax_phase.get_children()
                         if isinstance(child, matplotlib.collections.LineCollection)]
        xticks = [tick for tick in ax_phase.xaxis.get_major_ticks()
                  if tick_in_limits(tick, ax_phase.get_xlim())]
        yticks = [tick for tick in ax_phase.yaxis.get_major_ticks()
                  if tick_in_limits(tick, ax_phase.get_ylim())]

        t_start = perf_counter()
        for zeit in zeiten[1:]:
            # Zeitentwicklung Entwicklungskoeff.
            # t1 = perf_counter()
            phase = np.exp(-1j * eigen_werte * zeit / h_quer)
            # t2 = perf_counter()
            phi = eigen_fkt @ (coefficients * phase)
            # t3 = perf_counter()
            wellenpaket[0].set_ydata(e_0_qm + faktor * np.abs(phi)**2)
            ax_ort.draw_artist(wellenpaket[0])
            # t4 = perf_counter()
            husimi = husimi_darstellung(koeffizienten, phi, x_werte, h_quer)
            # t5 = perf_counter()
            husimi_plot.set_data(husimi)
            ax_phase.draw_artist(husimi_plot)
            for line in contour_lines:
                ax_phase.draw_artist(line)
            for tick in xticks:
                ax_phase.draw_artist(tick)
            for tick in yticks:
                ax_phase.draw_artist(tick)
            # t6 = perf_counter()
            for ctr, axis in enumerate(axes):
                fig.canvas.blit(axis.bbox)
                fig.canvas.restore_region(backgrounds[ctr])
            # t7 = perf_counter()
            fig.canvas.flush_events()
            # t8 = perf_counter()
            # print(f"phase: {(t2-t1)*1e6:.6f} us")
            # print(f"eig_f @ coeff: {(t3-t2)*1e6:.6f} us")
            # print(f"wp set_ydata: {(t4-t3)*1e6:.6f} us")
            # print(f"husimi: {(t5-t4)*1e6:.6f} us")
            # print(f"husimi set_data: {(t6-t5)*1e6:.6f} us")
            # print(f"draw: {(t7-t6)*1e6:.6f} us")
            # print(f"flush: {(t8-t7)*1e6:.6f} us")
            # print()
        t_end = perf_counter()
        print(f"Animation completed in {t_end - t_start:.2f} sec.")


def main():
    """Hauptprogramm."""
    asymmetrie_potential = 0.05
    x_extent = 1.5                                 # x-Bereich ist [-L, L]
    num_gitterpunkte = 200
    h_quer = 0.06                                  # effektives hquer
    e_max = 0.15
    asym = 1.0                                     # Asymmetrie der kohaerenten
                                                   #    Zustaende
    num_husimi_grid = 60                           # Anzahl Gitterpunkte
                                                   #    fuer Husimi-Darstellung

    # Zeitentwicklung
    zeiten = np.linspace(0.0, 12.0, 80)            # Zeiten der Zeitentw.

    # Plotparameter
    levels = np.linspace(-0.35, 0.1, 10)           # Levels fuer Konturplot
    faktor = 0.01                                  # Plot-Skalierungsfaktor
    n_contour = 100

    # Festlegung des Potentials
    potential = functools.partial(potential_fkt, asymmetrie=asymmetrie_potential)

    # Diskretisierung und Berechnung der EW/Efkt
    x_werte = diskretisierung(-x_extent, x_extent, num_gitterpunkte)
    eigen_werte, eigen_fkt = diagonalisierung(h_quer, x_werte, potential)

    # Vorberechnung der Kohaerente Zustaende --> 3D Matrix
    koeffizienten = husimi_koeffizienten(x_extent, x_extent, num_husimi_grid,
                                         h_quer, asym, x_werte)

    # Erstelle Plot mit Plotbereich Ortsraum und Phasenraumdarstellung
    fig, [ax_ort, ax_phase] = plt.subplots(1, 2)
    ax_phase.set_aspect(1.0)

    # Plotten Eigenfunktion
    plot_eigenfunktionen(ax_ort, eigen_werte, eigen_fkt, x_werte, potential,
                         e_max=e_max, fak=faktor)

    # Beschriftung Phasenraum
    ax_phase.set_xlabel("$x$")
    ax_phase.set_ylabel("$p$")
    ax_phase.set_title("Phasenraum")

    # Berechnen und plotten der Konturlinien
    hamilton, p_hamilton, x_hamilton = kontur_fkt(x_extent, x_extent,
                                                  n_contour, potential)
    husimi_achsen = [-x_extent, x_extent, -x_extent, x_extent]

    ax_phase.contour(x_hamilton, p_hamilton, hamilton, levels)
    ax_phase.axis(husimi_achsen)

    husimi_dynamisch_partial = functools.partial(husimi_dynamisch,
                                                 eigen_werte=eigen_werte,
                                                 eigen_fkt=eigen_fkt,
                                                 x_werte=x_werte,
                                                 potential=potential,
                                                 faktor=faktor,
                                                 h_quer=h_quer,
                                                 e_max=e_max,
                                                 asym=asym,
                                                 koeffizienten=koeffizienten,
                                                 zeiten=zeiten,
                                                 husimi_achsen=husimi_achsen,
                                                 ax_ort=ax_ort,
                                                 ax_phase=ax_phase,
                                                 )

    # Benutzerfuehrung
    print(__doc__)
    print()
    print("Bitte im Phasenraum durch Klicken mit linker Maustaste den")
    print("Schwerpunkt (x,p) eines Wellenpakets im Phasenraum waehlen.")

    fig.canvas.mpl_connect('button_press_event', husimi_dynamisch_partial)
    if MPL_SPECIAL_IMPORT:
        mpl_special.embed_labels(fig, [ax_ort, ax_phase], embed_ylabels=[0, 1])
    else:
        plt.show()


if __name__ == "__main__":
    main()
