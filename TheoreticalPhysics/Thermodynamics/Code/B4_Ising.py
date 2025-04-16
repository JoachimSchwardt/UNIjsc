"""
Created on Sat Jan  2 14:32:40 2021

@author: Joachim
"""

import numpy as np
import matplotlib.pyplot as plt
import functools

class IsingState(object):
    def __init__(self, L=50, J=1, b=0.001, beta0=1):
        self.spins = np.ones((L, L))
        self.L = L
        self.J = J
        self.b = b
        self.beta = beta0
        self.m_plot = []
        
    def mean(self):
        return np.sum(self.spins) / self.L**2
    
    def update_beta(self, beta):
        self.beta = beta
        pass
        
    def random_spins_m(self, m=0):
        if abs(m) > 1:
            m = int(m / abs(m))
            
        self.spins = np.ones(self.L**2)
        self.spins[0:int(self.L**2 * (1-m) / 2)] = -1
        np.random.shuffle(self.spins)
        self.spins = np.reshape(self.spins, (self.L, self.L))
        pass
        
    def update_spins(self, m=None, spins=None):
        if m != None:
            self.random_spins_m(m)
        
        if spins != None:
            if np.shape(spins) == (self.L, self.L):
                self.spins = spins
            else:
                print("Spins has wrong shape, defaults to random Arangement!")
                if m != None:
                    self.random_spins_m(m)
                else:
                    self.random_spins_m(0)
        pass
    
    def hamilton(self):
        spins_product = self.spins * (np.roll(self.spins, -1, axis=0) + 
                                      np.roll(self.spins, -1, axis=1))
        return -self.b * self.mean() - self.J * np.sum(spins_product)
    
    def spin_flip(self, size=None):
        if size == None:
            size = self.L**2
            
        # Arrays mit Indizes und Zufallszahlen 
        index_array = np.random.randint(0, self.L, size=(2, size))
        random_array = np.random.uniform(0, 1, size=size)
        
        # L**2 zufaellige Eintraege auf Spinflip pruefen
        for count in range(size):
            i, j = index_array[:, count]
            # Berechnung von 'H' fuer den ausgewaehlten Spin
            S_sum = (self.spins[(i+1) % self.L, j] + 
                     self.spins[i, (j+1) % self.L] + 
                     self.spins[(i-1) % self.L, j] + 
                     self.spins[i, (j-1) % self.L]) 
            
            # Wahrscheinlichkeit fuer Spinflip
            probability = np.exp(-2 * self.beta * self.spins[i, j] * 
                                 (self.J * S_sum + self.b))  
            
            # Spinflip mit 'Wuerfel'
            if probability > random_array[count]:
                self.spins[i, j] *= -1 
        pass
    
def mouse_click(event, axis, state, Nt, L, spins_img, plot_dict, Nflip,
                Nbins):
    """
    Mausklick muss im 'default'-Modus (kein Zoom etc.) erfolgen.
    Klick im ersten Plotbereich:
        Dynamische Darstellung von 10 Monte-Carlo Schritte, ausgehend vom
        aktuellen Zustand. Im zweiten Plotbereich wird bei jedem Schritt der
        Mittelwert gezeichnet.
    Klick im zweiten Plotbereich:
        Auswahl eines neuen Wertepaares (m, tau). Im ersten Plotbereich wird
        ein zufaelliger Zustand erstellt, der dem gewaehlten 'm' entspricht.
    """
    tool_mode = event.canvas.toolbar.mode
    # Mausklick im 'ersten' Plotbereich 'and event.inaxes == axis[0]'
    if event.button == 1 and event.inaxes == axis[0] and tool_mode == '':
        if state.m_plot == []:
            state.m_plot.append(state.mean())
        
        marker = axis[1].plot(state.beta, state.mean(), c='r', ls='',
                              marker='o', mew=0, ms=3)
        
        # Dynamik
        for i in range(Nt):
            # Monte-Carlo Schritt ausfuehren
            state.spin_flip(Nflip)
            
            # Neuen Zustand im ersten Plot darstellen
            spins_img.set_data(state.spins)
            
            
            state.m_plot.append(state.mean())
            marker[0].set_ydata(state.mean())
            axis[2].plot(np.arange(0, len(state.m_plot), 1), 
                         state.m_plot, lw=1, c='b')
            
            # Titel aktualisieren
            axis[1].set_title(plot_dict[1]['title'] + '{}'
                              .format(round(state.mean(), 4)), size='small')
            
            event.canvas.flush_events()
            event.canvas.draw()
        
            
    
    # Mausklick im zweiten Plotbereich
    if event.button == 1 and event.inaxes == axis[1] and tool_mode == '':
        # Plotbereich leeren 
        axis[1].lines = []
        
        # Mauskoordinaten als 'tau' und 'm'
        beta0 = event.xdata
        m0 = event.ydata
        
        state.update_beta(beta0)
        
        # Zufaelligen Zustand mit Mittelwert von etwa m0 erstellen
        state.update_spins(m0)
        
        # Zustand im ersten Plot zeigen
        spins_img.set_data(state.spins)
        
        # Tatsaechliches 'm' als Marker im zweiten Plot
        axis[1].plot(beta0, state.mean(), c='k', ls='', 
                     marker='o', mew=0, ms=3) 
        
        # Titel aktualisieren
        axis[0].set_title(plot_dict[0]['title'] + '{}'
                          .format(round(state.beta, 3)), size='small')
        axis[1].set_title(plot_dict[1]['title'] + '{}'
                          .format(round(state.mean(), 4)), size='small')
        
        event.canvas.flush_events()
        event.canvas.draw()
        
    # Mausklick im dritten Plotbereich
    if event.button == 1 and event.inaxes == axis[2] and tool_mode == '':
        state.m_plot = []
        axis[2].lines = []
        
        # recompute the ax.dataLim
        axis[2].relim()
        # update ax.viewLim using the new dataLim
        axis[2].autoscale_view()
        
        event.canvas.flush_events()
        event.canvas.draw()
        
    # Mausklick im vierten Plotbereich
    if event.button == 1 and event.inaxes == axis[3] and tool_mode == '':
        axis[3].bins, axis[3].bars, axis[3].patches = [], [], []
        
        # recompute the ax.dataLim
        axis[3].relim()
        # update ax.viewLim using the new dataLim
        axis[3].autoscale_view()
        
        axis[3].hist(state.m_plot, color='b', bins=Nbins)
        
        event.canvas.flush_events()
        event.canvas.draw()
    
def main():
    print(__doc__)
    L = 50
    J = 1
    b = 0.001
    beta0 = 0.5
    m0 = 0.8
    
    Nt = 10
    Nflip = None
    Nbins = 20
    beta_max = 2
    
    # Strings fuer Plotueberschriften
    txt1 = (r'Spinstate of a {}$\times${} lattice, $\beta$='.format(L, L)) 
    txt2 = (r'Average magnetization per spin $m$=')
    txt3 = (r'Time evolution of $m(t)$')
    txt4 = (r'Histogram $\rho(m)$')
    
    plot_dict = {0: {'xlabel': 'x', 'ylabel': 'y', 'title': txt1, 
                     'grid': False}, 
                 1: {'xlabel': r'$\beta$', 'ylabel': r'$m$', 'title': txt2, 
                     'grid': True},
                 2: {'xlabel': 't', 'ylabel': r'$m$', 'title': txt3, 
                     'grid': True},
                 3: {'xlabel': r'$m$', 'ylabel': None, 'title': txt4, 
                     'grid': True}}
    
    state = IsingState(L, J, b, beta0)
    state.update_spins(m0)
    
    # Plotbereich
    fig, axis = plt.subplots(2, 2, figsize=(15, 10))
    axis = np.reshape(axis, np.sum(np.shape(axis)))
    
    plt.suptitle('Visualisation of the 2D Ising-model')
    for i in range(len(axis)):
        axis[i].set_title(plot_dict[i]['title'])
        axis[i].set_xlabel(plot_dict[i]['xlabel'])
        axis[i].set_ylabel(plot_dict[i]['ylabel'])
        axis[i].grid(plot_dict[i]['grid'])
        
    axis[0].set_title(plot_dict[0]['title'] + '{}'.format(beta0))
    axis[1].set_title(plot_dict[1]['title'] + '{}'.format(m0))
    
    axis[1].axis([0, beta_max, -1.05, 1.05])
    
    spins_img = axis[0].imshow(state.spins)
    
    
    axis[1].plot(beta0, state.mean(), c='k', ls='', marker='o', mew=0,
                 ms=3, label=r'Initial value $(m,\beta)$')
    axis[1].legend()
    
    # Mausinteraktion
    click_function = functools.partial(mouse_click, axis=axis, Nt=Nt, L=L,
                                       state=state, spins_img=spins_img,
                                       plot_dict=plot_dict, Nflip=Nflip,
                                       Nbins=Nbins)
    fig.canvas.mpl_connect("button_press_event", click_function)
    
    plt.show()

if __name__ == "__main__":
    main()