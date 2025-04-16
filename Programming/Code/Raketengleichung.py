"""
Visualisierung verschiedener Raketengleichungen im Vergleich zur Theorie.
    F0 = m_dot(t)u(t)
    F_gravity = -m(t) * ME * G / x(t)**2
    F_friction = -gamma * x_dot(t)**2
    F_sum = m(t)x_ddot(t)
Damit folgt:
    x_ddot = (m_dot * u + F_gravity + F_friction) / m
"""
import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt

def m_dot(t):
    q = 7          # Treibstoffverbrauch in kg/s
    return q

def u(t):
    v_gas = 2500    # Ausstossgeschwindigkeit in m/s
    return v_gas

def m(t):
    """
    Vorgegebene Funktion fuer den Treibstoffverbrauch. Gibt die Masse als 
    Funktion von der Zeit wieder. Startwert ist immer m0 in kg.
    """
    m0 = 1000        # Startgewicht in kg
    m = m0 - m_dot(t) * t#quad(m_dot, 0, t)[0]
    return m

def g_coeff(x):
    mass_earth = 5.964e24
    rad_earth = 6.371e6
    G = 6.6743e-11
    return G * mass_earth / (rad_earth + x)

def F_grav(x, t):
    """Modell fuer die Gravitationskraft."""
    rad_earth = 6.371e6
    return -m(t) * g_coeff(x) / (rad_earth + x)

def T(x):
    gamma = 1.5                 # Temperaturaenderung pro 1000 m
    return 293 - gamma * x/1000

def rho(x):
    rho0 = 1.225                # Dichte von Luft in kg/m**3 (bei x=0)
    M_mol = 0.0289              # molare Masse von Luft in kg/mol
    NA = 6.02214076 * 10**(23)
    kB = 1.380649 * 10**(-23)
    return rho0 * np.exp(-M_mol * g_coeff(x) / (NA * kB * T(x)))

def F_fric(x, v, t):
    """Modell fuer die Reibungskraft."""
    cw = 1.2
    area = 2**2 * np.pi 
    return -cw * area/2 * rho(x) * v**2


def y_dot(y, t):
    x, v = y
    x_dot = v
    v_dot = (m_dot(t) * u(t) + F_grav(x, t) + F_fric(x, v, t)) / m(t)
    return np.array([x_dot, v_dot])

def velocity_theory(t):
    return u(t) * np.log(m(0) / m(t)) / 1000

def main():
    print(__doc__)
    
    t_end, t_steps = 100, 200    # Flugdauer und Anzahl Iterationsschritte
    t = np.linspace(0, t_end, t_steps) 
    x0, v0 = 0, 0                # Startwerte
                
    y_t = odeint(y_dot, np.array([x0, v0]), t)
    x_t, v_t = y_t[:, 0] / 1000, y_t[:, 1] / 1000     # x in km und v in km/s 
    
        
    # Plotbereich erstellen
    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.suptitle("Simulation der Raketengleichung", fontsize=22)
    ax2 = ax1.twinx()
    ax1.set_xlabel("t / s", fontsize=22)
    ax1.set_ylabel("x(t) / km", fontsize=22, c='b')
    ax2.set_ylabel("v(t) / km/s", fontsize=22, c='g')
    ax1.tick_params(labelsize=16, color='b')
    ax2.tick_params(labelsize=16, color='g')

    # Kurven plotten
    ax1.plot(t, x_t, c='b', label='Altitude')
    ax2.plot(t, v_t, c='g', label='absolute velocity')
    ax2.plot(t, velocity_theory(t), c='g', ls='--', label='Velocity theory')
    
    # Achsen des Koordinatensystems plotten
    ax1.axvline(0, c='k', lw=1)
    ax1.axhline(0, c='k', lw=1)
    
    # Ueberschriften und Legende
    ax1.plot(0, 0, c='g', label='absolute velocity')
    ax1.plot(0, 0, c='g', ls='--', label='Velocity theory')
    ax1.legend(loc=2, prop={'size': 20})
    ax1.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()