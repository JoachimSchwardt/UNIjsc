"""Hamilton'sche Ratsche"""

import functools
import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    """Simulation der Ratsche"""

    def __init__(self, a, v_0, alpha, num_tau, num_plot, t_aus, num_steps,
                 D, xlim, n_bins, num_p, num_x_vals,
                 FLOAT_ZERO_ATOL=1e-6):
        """Initialisierung"""
        self.a = a
        self.v_0 = v_0
        self.alpha = alpha

        self.num_tau = num_tau
        self.t_aus = t_aus
        self.num_plot = num_plot
        self.num_steps = num_steps
        self.t_vals = np.linspace(0.0, num_tau, num_steps * num_tau, 
                                  endpoint=False)
        self.dt = self.t_vals[1] - self.t_vals[0]

        self.D = D
        self.xlim = xlim
        self.n_bins = n_bins
        self.num_p = num_p
        self.num_x_vals = num_x_vals
        self.x_vals = np.linspace(xlim[0], xlim[1], num_x_vals)

        self.FLOAT_ZERO_ATOL = FLOAT_ZERO_ATOL


    ###########################################################################
    # Routinen für das Potential und seine Ableitung
    ###########################################################################


    def get_xpot_periodic(self, x):
        """Periodischer Teil des ortsabhängigen Teils des Potentials"""
        phi = 2 * np.pi * x
        return self.v_0 * (np.cos(phi) + self.a * np.sin(2 * phi))


    def get_xpot_periodic_prime(self, x):
        """Ableitung des periodischen Teils des ortsabhängigen Teils
        des Potentials
        """
        phi = 2 * np.pi * x
        fac = -np.sin(phi) + 2 * self.a * np.cos(2 * phi)
        return 2 * np.pi * self.v_0 * fac


    def get_xpot_tilt(self, x):
        """Kipp-Anteil des ortsabhängigen Teils des Potentials"""
        return self.alpha * x


    def get_xpot_tilt_prime(self, x):
        """Ableitung des Kipp-Anteils des ortsabhängigen Teils des Potentials"""
        return self.alpha + 0 * x


    def get_xpot(self, x):
        """Ortsabhängiger Anteil des Potentials"""
        return self.get_xpot_periodic(x) + self.get_xpot_tilt(x)


    def get_xpot_prime(self, x):
        """Ableitung des ortsabhängigen Anteils des Potentials"""
        return self.get_xpot_periodic_prime(x) + self.get_xpot_tilt_prime(x)


    def get_tpot(self, t):
        """Zeitabhängiger Anteil des Potentials"""
        if t % 1.0 > self.t_aus:
            return 1.0
        else:
            return 0.0


    def get_pot(self, x, t):
        """Vollständiges Potential"""
        return self.get_tpot(t) * self.get_xpot(x)


    def get_pot_prime(self, x, t):
        """Ortsableitung des vollständigen Potentials"""
        return self.get_tpot(t) * self.get_xpot_prime(x)


    ###########################################################################
    # Routinen für Berechnung und Simulation
    ###########################################################################


    def do_iteration(self, t):
        """Führt eine Iteration der Ratschen-Gleichung aus"""
        x_rand = np.random.normal(0, 1, self.num_p)
        pot_prime = self.get_pot_prime(self.x, t)
        self.x += np.sqrt(2 * self.D * self.dt) * x_rand - self.dt * pot_prime


    def event_loop(self):
        """Führt die Iterationen aus"""
        for step in range(0, self.t_vals.size, self.num_steps // self.num_plot):
            tval = self.t_vals[step]

            self.ax[0].patches = []
            self.ax[0].lines = []

            self.ax[0].hist(self.x, color=self.colors['P'], range=self.xlim,
                            bins=self.n_bins, density=True)
            self.ax[0].plot(self.x_vals, self.get_pot(self.x_vals, tval),
                            c=self.colors['V'])

            if self.get_tpot(tval) < self.FLOAT_ZERO_ATOL:
                key = 'V==0'
            else:
                key = 'V!=0'
            self.ax[1].plot(tval, np.mean(self.x), ls='', marker='o', 
                            ms=3, mew=0, c=self.colors[key])

            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

            for inc in range(step, step + self.num_steps // self.num_plot):
                self.do_iteration(self.t_vals[inc])


    def event_decision(self, event):
        """Entscheidet, was bei einem Mausklick passieren soll"""
        mode = event.canvas.toolbar.mode
        if event.button == 1 and event.inaxes and mode == '':
            self.ax[1].lines.clear()        # Zeitverlauf zurücksetzen/löschen
            self.ax[1].relim()              # Achsenskalierung zurücksetzen
            self.x = np.full(self.num_p, event.xdata)    # alle Teilchen bei x0
            self.event_loop()


    def create_plot(self, figsize=(16, 9)):
        """Erzeugt die Plotfenster mit Labels und Legenden"""
        self.colors = {'V==0' : 'g', 'V!=0' : 'r', 'V' : 'b', 'P' : 'k'}

        self.fig, self.ax = plt.subplots(ncols=2, figsize=figsize)
        self.ax[0].set_xlabel(r"$x$")
        self.ax[0].set_ylabel(r"$V(x,t_n), P(x, t_n)$")
        self.ax[1].set_xlabel(r"$t$")
        self.ax[1].set_ylabel(r"$\langle x \rangle$")

        self.ax[0].set_xlim(self.xlim)
        self.ax[0].hist([], color=self.colors['P'],
                        label=r"Normierte Verteilung $P(x, t_n)$")
        self.ax[0].plot([], [],
                        c=self.colors['V'],
                        label=r"Potential $V(x, t)$")

        self.ax[1].plot([], [], ls='', marker='o', ms=3, mew=0,
                        c=self.colors['V==0'],
                        label=r"$\langle x \rangle_\mathrm{num}$ für $V=0$")
        self.ax[1].plot([], [], ls='', marker='o', ms=3, mew=0,
                        c=self.colors['V!=0'],
                        label=r"$\langle x \rangle_\mathrm{num}$ für $V\neq 0$")

        self.ax[0].legend()
        self.ax[1].legend(numpoints=3)


    def show(self):
        """Startet die Simulation"""
        plt.connect("button_press_event",
                    functools.partial(self.event_decision))
        plt.show()


def main():
    """Hauptprogramm"""
    print(__doc__)
    a = 0.2                         # Asymmetriefaktor
    v_0 = -6.0                   # Potentialhöhe
    alpha = 0.0                     # Kippwinkel

    num_tau = 5                     # Anzahl an Perioden
    num_plot = 30
    theta = 0.8                     # Verhältnis 't_an / t_aus'
    t_aus = 1 / (1 + theta)       # Zeitdauer im Zustand 'Aus' je Periode
    num_steps = 3000                 # Anzahl Schritte pro Zeiteinheit

    D = 0.1067                # Diffusionskonstante
    xlim = (-3.33, 3.33)                  # Räumliche Begrenzung
    n_bins = 50                     # Anzahl an Bins im Histogramm
    num_p = 4096                   # Anzahl an Teilchen/Realisierungen
    num_x_vals = 300                # Anzahl an x-Werte für den Potentialplot

    simulator = Simulator(a=a, v_0=v_0, alpha=alpha, num_plot=num_plot,
                          num_tau=num_tau, t_aus=t_aus,
                          num_steps=num_steps, num_p=num_p,
                          D=D, n_bins=n_bins,
                          xlim=xlim, num_x_vals=num_x_vals)

    simulator.create_plot()
    simulator.show()


if __name__ == "__main__":
    main()
