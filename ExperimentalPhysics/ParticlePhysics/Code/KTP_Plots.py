"""KTP-Plots und mehr"""

from scipy.integrate import quad as integrate 
from scipy.special import factorial
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

NA = 6.02214076 * 10**(23) # 1/mol


me = 9.1093837015 * 10**(-31) # kg
mp = 1.67262192369 * 10**(-27) # kg
mn = 1.67492749804 * 10**(-27) # kg

c = 299762458 # m/s
u = 1.66 * 10**(-27) # kg
e = 1.602176634 * 10**(-19) # As
mu0 = 1.25663706212 * 10**(-6) # Vs/Am
eps0 = 8.8541878128 * 10**(-12) # As/Vm
G = 6.67430 * 10**(-11) # m^3/kg s^2

T0 = -273.15 # °C
k = 1.380649 * 10**(-23) # J/K
sigma = 5.670374419 * 10**(-8) # W/m^2 K^4
b = 2.897771955 * 10**(-3) # m K
h = 6.62607015 * 10**(-34) # J s
hbar = 1.054571817 * 10**(-34) # J s
g = 9.81 # m/s^2

# zusammengesetzte Größen
R = NA * k # Gaskonstante
alpha = e**2/(4*np.pi*eps0*hbar*c) # Feinstrukturkonstante
ER = alpha**2 * me * c**2 / 2 # Rydberg-Energie in J
a0 = hbar / (me * c * alpha)  # Bohrradius in m
muB = hbar * e / (2 * me)     # Bohrmagneton in T/J

class Molecule(object):
    def __init__(self, m1, m2, Attribute, mode='k'):
        self.m1, self.m2 = m1, m2
        self.mu = m1*m2 / (m1+m2)
        if mode == 'k':
            self.k = Attribute
            self.nu = np.sqrt(self.k / self.mu) / (2*np.pi*c)
        elif mode == 'nu':
            self.nu = Attribute
            self.k =  self.mu * 4*np.pi**2*c**2 * self.nu**2
        self.omega = np.sqrt(self.k / self.mu)
        self.E0 = self.omega * hbar/2
        
class Atom(object):
    def __init__(self, A, Z):
        self.A, self.Z = A, Z
        self.N = A - Z
    def M_bind(self):
        """Bethe-Weizsaecker binding energy formula."""
        a_v = -15.67
        a_s = 17.23
        a_c = 0.714
        a_a = 93.15
        delta = 11.2
        ans = (a_s * (self.A)**(2/3) + a_c * (self.Z)**2 * (self.A)**(-1/3)
               + a_a / (4*self.A) * (self.N - self.Z)**2 + a_v * self.A)
        if int(self.A % 2) == 1 and int(self.N % 2) == 1:
            ans += delta * (self.A)**(-1/2)
        elif int(self.A % 2) == 0 and int(self.N % 2) == 0:
            ans -= delta * (self.A)**(-1/2)
        return ans * 10**6      # given in eV/c**2
        
    def M(self):
        ans = (self.N * mn + self.Z * (mp + me)) * c**2/e + self.M_bind()
        return ans              # given in eV/c**2


def Bethe_Mass_plot(A=108):
    """Optimised for A=108"""
    Z = np.arange(44, 50, 1)
    E_Z = []
    for i in Z:
        E_Z.append(Atom(A, i).M() * e / (c**2 * u))
        
    def f(x, a, b, c):
        return a*x**2 + b*x + c
        
    mmin, mmax = min(E_Z), max(E_Z)
    mdelta = (mmax - mmin)
    
    p0 = (mdelta, -mdelta * min(Z), mmin)
    params1, pcv1 = curve_fit(f, Z[::2], E_Z[::2], p0=p0)
    params2, pcv2 = curve_fit(f, Z[1::2], E_Z[1::2], p0=p0)
    
    
    x = np.linspace(min(Z), max(Z), 200)
    Atoms_dict = {0: {'Id' : '${}_{44}^{108}$Ru', 'x' : -0.5, 'y' : -0.07}, 
                  1: {'Id' : '${}_{45}^{108}$Rh', 'x' : -0.02, 'y' : 0.07},
                  2: {'Id' : '${}_{46}^{108}$Pd', 'x' : -0.1, 'y' : -0.07}, 
                  3: {'Id' : '${}_{47}^{108}$Ar', 'x' : -0.1, 'y' : 0.07},
                  4: {'Id' : '${}_{48}^{108}$Cd', 'x' : 0.02, 'y' : -0.07}, 
                  5: {'Id' : '${}_{49}^{108}$In', 'x' : 0.07, 'y' : 0.03},}
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([43, 50, mmin - 0.1*mdelta, mmax + 0.1*mdelta])
    ax.set_title("Massenzahl A={}".format(A), fontsize=22)
    
    ax.plot(Z, E_Z, ls='', c='b', marker='x', mew=2, ms=12, 
            label="Bethe-Weizsäcker")
    for i in range(len(Z)):
        ax.text(Z[i] + Atoms_dict[i]['x'], E_Z[i] + Atoms_dict[i]['y'] * mdelta, 
                Atoms_dict[i]['Id'], fontsize=20)
    ax.plot(x, f(x, *params1), c='b', ls='--', label="Curve-Fit")
    ax.plot(x, f(x, *params2), c='b', ls='--')
    
    ax.set_ylabel("m(Z) / u", fontsize=22)
    ax.set_xlabel("Z", fontsize=22)
    ax.tick_params(labelsize=16)
    
    ax.grid(True)
    ax.legend(fontsize=20)
    plt.show()
    
def f(x):
    return np.abs(3 * (np.sin(x) / (x**3) - np.cos(x) / (x**2)))
    
def function_plotter(f):
    x = np.linspace(0.01, 4*np.pi, 300)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, yscale='log')
    
    ax.plot(x, f(x), c='b')
    
    ax.axvline(0, lw=0.7, c='k')
    ax.axhline(0, lw=0.7, c='k')
    ax.set_ylabel("$|F(q^2)|$", fontsize=22)
    ax.set_xlabel("x", fontsize=22)
    ax.tick_params(labelsize=16)
    
    ax.grid(True)
    # ax.legend(fontsize=20)
    plt.show()

# rho = 5
# A = 0.5
# n = NA * rho / A
# sigma = 10**(-19) * 10**(-24)
# print(1/(sigma*n))

Q = [4.27, 4.572, 4.858, 5.414, 5.993, 6.804]
gamma = 2*92/np.sqrt(Q[0]) - 1.5*np.sqrt(92*1.25*238**(1/3))
C = np.log10(1.4*10**(17))-2*gamma/np.log(10)
print(C)

for q in Q:
    gamma = 2*92/np.sqrt(q) - 1.5*np.sqrt(92*1.25*238**(1/3))
    print(gamma)
    print(10**(2*gamma/np.log(10) + C))