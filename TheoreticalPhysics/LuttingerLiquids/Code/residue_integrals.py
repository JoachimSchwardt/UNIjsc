"""Numerical verification of residue integral techniques"""

import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from scipy import integrate, special
import mpmath as mp


def integrand_x_zero_t_no_int(x, tau, k, omega, alpha):
    return np.exp(1j*(k*x-omega*tau)) / (tau + 1j*x + alpha)


def integrand_x_with_t_no_int(x, tau, k, omega, alpha, beta):
    return np.exp(1j*(k*x-omega*tau)) / np.sin(np.pi/beta * (tau + 1j*x + alpha))


def integrand_x_zero_t_with_int(x, tau, k, omega, alpha, K):
    interaction = (x**2 + tau**2)**((1 - K) / 2)
    return np.exp(1j*(k*x-omega*tau)) / (tau + 1j*x + alpha) * interaction


def analytical_x_zero_t_no_int(tau, k, omega, alpha):
    prefactor = 2*np.pi * np.heaviside(k, 1)
    phase = np.exp(-1j * omega * tau)
    decay = np.exp(-np.abs(k) * (tau + alpha))
    return prefactor * phase * decay


def analytical_x_with_t_no_int(tau, k, omega, alpha, beta):
    prefactor = 2*beta / (1 + np.exp(-np.abs(k) * beta))
    phase = np.exp(-1j * omega * tau - k * tau) * np.exp(beta*k * np.heaviside(-k, 1))
    return prefactor * phase


def analytical_tau_with_t_no_int(k, omega, alpha, beta):
    # prefactor = 2*beta / (k + 1j * omega) * np.exp(beta*k * np.heaviside(-k, 1))
    # decay = (1 - np.exp(-beta * k)) / (1 + np.exp(-np.abs(k) * beta))
    prefactor = 2*beta
    decay = np.tanh(beta * k / 2) / (k + 1j * omega)
    return prefactor * decay


def analytical_x_zero_t_with_int(tau, k, omega, alpha, K):
    # depending on sign of 'k' different pole within contour --> +-1 in power
    a_val = (K - 1) / 2 + np.heaviside(k, 1)
    prefactor = 2*np.pi / special.gamma(a_val)
    hyper_factor = special.hyperu(1 - a_val, 2 - K, 2*tau * np.abs(k))
    phase = np.exp(-1j * omega * tau)
    decay = np.exp(-np.abs(k) * (tau + alpha))
    power_law = (2*tau + 2*alpha)**(1 - K)
    return prefactor * phase * decay * power_law * hyper_factor


def analytical_tau_zero_t_with_int(k_val, omega, alpha, K):
    # depending on sign of 'k' different pole within contour --> +-1 in power
    a_val = (K + 1) / 2# + np.heaviside(k, 1)
    b_val = (K - 1) / 2# - np.heaviside(k, 1)
    # prefactor = np.pi / special.gamma(a_val) * special.gamma(1 - b_val) / 2**(a_val + b_val - 1)
    # # power_law = (k**2 + omega**2)**(1 - K) / (k + 1j * omega)
    # power_law = (k + 1j * omega)**(b_val - 1) * (k - 1j * omega)**(a_val - 1)
    # return prefactor * power_law
    if isinstance(k_val, np.ndarray):
        raise NotImplementedError("k_val must be scalar")
    eps, tau, k = sy.symbols("epsilon tau k", positive=True)
    n, a, b = sy.symbols("n a b", positive=True, integer=True)
    res = sy.integrate(sy.exp(-(k+1j*omega)*tau) * (2*tau)**(-b-n), (tau, eps, sy.oo))
    lim = sy.limit(res.simplify().args[0][0], eps, 0)
    sum_ = sy.summation(sy.simplify(lim * sy.gamma(b + n)) * k**(a-1-n)
                        * sy.binomial(a-1, n) / sy.gamma(b) / sy.gamma(a),
                        (n, 0, a-1))
    # res = np.prod([sum_.args[i] for i in range(len(sum_.args)) if i != 9]) * sum_.args[9].args[0][0]
    res = sum_.simplify().args[0][0] * 2*np.pi
    return complex(res.subs({a : a_val, b : b_val, k : k_val}).evalf())


def c_quad(func, xmin, xmax, args=(), **kwargs):
    def f_real(x, *args):
        return np.real(func(x, *args))
    def f_imag(x, *args):
        return np.imag(func(x, *args))
    r_int = integrate.quad(f_real, xmin, xmax, args, **kwargs)
    i_int = integrate.quad(f_imag, xmin, xmax, args, **kwargs)
    return (r_int[0] + 1j*i_int[0], np.abs(r_int[1] + 1j*i_int[1]))


def c_dblquad(func, xmin, xmax, ymin, ymax, args=()):
    def f_real(x, y, *args):
        return np.real(func(x, y, *args))
    def f_imag(x, y, *args):
        return np.imag(func(x, y, *args))
    r_int = integrate.dblquad(f_real, xmin, xmax, ymin, ymax, args)
    i_int = integrate.dblquad(f_imag, xmin, xmax, ymin, ymax, args)
    return (r_int[0] + 1j*i_int[0], np.abs(r_int[1] + 1j*i_int[1]))


def int_2d(func, xmin, xmax, ymin, ymax, args=(), n_x=500, n_y=500):
    """Test: L=15; int_2d(lambda x,y: np.exp(-x*x-y*y), -L, L, -L, L) / np.pi"""
    delta_x = (xmax - xmin) / (n_x + 1)
    delta_y = (ymax - ymin) / (n_y + 1)
    x_vals = np.linspace(xmin + delta_x, xmax - delta_x, n_x)
    y_vals = np.linspace(ymin + delta_y, ymax - delta_y, n_y)
    xy_grid = np.meshgrid(x_vals, y_vals)
    f_vals = func(xy_grid[0], xy_grid[1], *args)
    return delta_x * delta_y * np.sum(f_vals)


def test_zero_t_no_int(par):
    """Test zero temperature and no interaction limit (T=0, K=1)"""
    k_vals = np.linspace(-5, 5, 80)
    f_analytic = analytical_x_zero_t_no_int(par["tau"], k_vals, par["omega"], par["alpha"])
    f_numeric = np.array([c_quad(integrand_x_zero_t_no_int, -par["L"], par["L"],
                                 args=(par["tau"], k_val, par["omega"], par["alpha"]))[0]
                          for k_val in k_vals])
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.plot(k_vals, np.abs(f_analytic), label="analytic abs")
    ax1.plot(k_vals, np.abs(f_numeric), ls="--", label="numeric abs")
    ax2.plot(k_vals, np.arctan2(f_analytic.imag, f_analytic.real),
             label="analytic phase")
    ax2.plot(k_vals, np.arctan2(f_numeric.imag, f_numeric.real), ls="--",
             label="numeric phase")
    for axis in [ax1, ax2]:
        axis.set_xlabel("$k$")
        axis.legend()


def test_with_t_no_int(par):
    """Test zero temperature and no interaction limit (T=0, K=1)"""
    k_vals = np.linspace(-5, 5, 80)
    f_analytic = analytical_x_with_t_no_int(par["tau"], k_vals, par["omega"],
                                            par["alpha"], par["beta"])
    f_numeric = np.array([c_quad(integrand_x_with_t_no_int, -par["L"], par["L"],
                                 args=(par["tau"], k_val, par["omega"],
                                       par["alpha"], par["beta"]))[0]
                          for k_val in k_vals])
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.plot(k_vals, np.abs(f_analytic), label="analytic abs")
    ax1.plot(k_vals, np.abs(f_numeric), ls="--", label="numeric abs")
    ax2.plot(k_vals, np.arctan2(f_analytic.imag, f_analytic.real),
             label="analytic phase")
    ax2.plot(k_vals, np.arctan2(f_numeric.imag, f_numeric.real), ls="--",
             label="numeric phase")
    for axis in [ax1, ax2]:
        axis.set_xlabel("$k$")
        axis.legend()


def test_with_t_no_int_full(par):
    """Test zero temperature and no interaction limit (T=0, K=1)"""
    k_vals = np.linspace(-5, 5, 80)
    f_analytic = analytical_tau_with_t_no_int(k_vals, par["omega"],
                                              par["alpha"], par["beta"])
    f_numeric = np.array([c_quad(analytical_x_with_t_no_int, 0, par["beta"],
                                 args=(k_val, par["omega"],
                                       par["alpha"], par["beta"]))[0]
                          for k_val in k_vals])
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.plot(k_vals, np.abs(f_analytic), label="analytic abs")
    ax1.plot(k_vals, np.abs(f_numeric), ls="--", label="numeric abs")
    ax2.plot(k_vals, np.arctan2(f_analytic.imag, f_analytic.real),
             label="analytic phase")
    ax2.plot(k_vals, np.arctan2(f_numeric.imag, f_numeric.real), ls="--",
             label="numeric phase")
    for axis in [ax1, ax2]:
        axis.set_xlabel("$k$")
        axis.legend()


def test_zero_t_with_int(par):
    """Test zero temperature and no interaction limit (T=0, K=1)"""
    k_vals = np.linspace(-5, 5, 80)
    f_analytic = analytical_x_zero_t_with_int(par["tau"], k_vals, par["omega"],
                                              par["alpha"], par["K"])
    f_numeric = np.array([c_quad(integrand_x_zero_t_with_int, -par["L"], par["L"],
                                 args=(par["tau"], k_val, par["omega"],
                                       par["alpha"], par["K"]))[0]
                          for k_val in k_vals])
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.plot(k_vals, np.abs(f_analytic), label="analytic abs")
    ax1.plot(k_vals, np.abs(f_numeric), ls="--", label="numeric abs")
    ax2.plot(k_vals, np.arctan2(f_analytic.imag, f_analytic.real),
             label="analytic phase")
    ax2.plot(k_vals, np.arctan2(f_numeric.imag, f_numeric.real), ls="--",
             label="numeric phase")
    for axis in [ax1, ax2]:
        axis.set_xlabel("$k$")
        axis.legend()


def test_zero_t_with_int_full(par):
    """Test zero temperature and no interaction limit (T=0, K=1)"""
    k_vals = np.linspace(-5, 5, 80)
    f_analytic = analytical_tau_zero_t_with_int(k_vals, par["omega"],
                                                   par["alpha"], par["K"])
    f_numeric = np.array([c_quad(analytical_x_zero_t_with_int, 0, 2e2,
                                 args=(k_val, par["omega"], par["alpha"],
                                       par["K"]))[0]
                          for k_val in k_vals])
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.plot(k_vals, np.abs(f_analytic), label="analytic abs")
    ax1.plot(k_vals, np.abs(f_numeric), ls="--", label="numeric abs")
    ax2.plot(k_vals, np.arctan2(f_analytic.imag, f_analytic.real),
             label="analytic phase")
    ax2.plot(k_vals, np.arctan2(f_numeric.imag, f_numeric.real), ls="--",
             label="numeric phase")
    for axis in [ax1, ax2]:
        axis.set_xlabel("$k$")
        axis.legend()


def special_function_integral():
    """Direct computation of the convolution integral for K=1"""
    y, yp = sy.symbols("y y'", positive=True)
    x, xp = sy.symbols("x x'", real=True)
    x_val = 3.1
    xp_val = 5.6
    y_val = 1.2
    yp_val = 0.7

    yp_vals = np.linspace(0, np.pi, 400)
    ell = 1
    integrand = sy.csc(yp + ell * sy.I * xp) * sy.csc(y-yp - ell * sy.I * (x-xp))
    integrand_np = sy.lambdify([x, y, xp, yp], integrand, "numpy")

    # TODO: WRONG RESULT!! Sympy claims this integral to be exactly 0, which is not true!!
    int_yp = sy.integrate(integrand, (yp, 0, sy.pi))

    # sy.I * (2*x + sy.log(sy.exp(2*sy.I* y - 4*sy.I*yp))) / sy.sin(2 *yp - y + sy.I* x)
    int_xp = sy.integrate(integrand, (xp, -sy.oo, sy.oo))
    int_xp = sy.simplify(int_xp)
    integrand_yp_np = sy.lambdify([x, y, yp], int_xp, "numpy")

    ### c_quad(lambda xp: integrand_np(x_val, y_val, xp, yp_val), -50, 50)
    ### c_quad(lambda yp: integrand_yp_np(0.3, -y_val, yp), 0, np.pi)
    # f_vals = integrand_np(x_val, y_val, xp_val, yp_vals)
    # plt.plot(yp_vals, f_vals.real)
    # plt.plot(yp_vals, f_vals.imag)


def special_function_integral_v2():
    beta, u, tau, taup, alpha = sy.symbols("beta u tau \\tau' alpha", positive=True)
    x, xp = sy.symbols("x x'", real=True)
    j = sy.symbols("j", integer=True, positive=True)
    I = sy.I
    f_1 = sy.csc(sy.pi / (beta*u) * (u*(tau - taup) + alpha*sy.sign(tau - taup) - I*(x - xp)))
    f_2 = sy.csc(sy.pi / (beta*u) * (u * taup + alpha*sy.sign(taup) + I*xp))
    x_res1 = x + I * u * (tau - taup) + I * alpha * sy.sign(tau - taup) + I * j * beta * u
    x_res2 = I * (u*taup + alpha*sy.sign(taup) + j * beta * u)
    f_1_np = sy.lambdify([x, tau, xp, taup, beta, u, alpha], f_1, "numpy")
    f_2_np = sy.lambdify([x, tau, xp, taup, beta, u, alpha], f_2, "numpy")
    def integrand(xp_val, x_val=3.1, tau_val=0.8, taup_val=0.9,
                  beta_val=1.7, u_val=1.2, alpha_val=1e-3):
        first = f_1_np(x_val, tau_val, xp_val, taup_val, beta_val, u_val, alpha_val)
        second = f_2_np(x_val, tau_val, xp_val, taup_val, beta_val, u_val, alpha_val)
        return first * second

    # def magicres(x,y,yp):
    #     return 2*(x+np.pi*1j+(y-2*yp)*1j)/np.sinh(x+(y-2*yp)*1j)##?
    def csc_int(w, z):
        w_count = w.real // np.pi
        z_count = z.real // np.pi
        w = (w.real) % np.pi + 1j * w.imag
        z = (z.real) % np.pi + 1j * z.imag
        return 2 * (w-z) / np.sin(w-z) * (-1)**(w_count + z_count)
    def csc_int_num(w, z):
        return c_quad(lambda x: (np.sin(w+1j*x)*np.sin(z+1j*x))**(-1), -30, 30)
    ### this works for '0 < Re(z, w) < np.pi', otherwise add 'pi' to 'z, w'
    # z = 4.1 + 0.5j
    # w = -3.5 + 3.8j
    # csc_int(w, z), csc_int_num(w, z)

    """
    beta = 5.6
    u = 1.2
    alpha = 0.01
    tau = 0.7
    omega = 2*np.pi/beta * 1
    k = -0.43
    integrand = lambda x: (np.exp(1j*k*x-1j*omega*tau)
                           / np.sin(np.pi/(beta*u)*(u*tau + alpha * np.sign(tau) - 1j*x))/beta)
    c_quad(integrand, -30, 30)
    Out[139]: ((1.1154821317937995-1.1154821317937995j), 4.430177641542743e-10) ???
    """

    """
    beta = 1e4
    u = 1.0
    alpha = 0.0
    tau = 0.7
    omega = 2*np.pi/beta * 1
    k = 1.0
    integrand = lambda x: (np.exp(1j*k*x-1j*omega*tau)
                           / np.sin(np.pi/(beta*u)*(u*tau + alpha * np.sign(tau) + 1j*x))
                           / beta)
    integrand_tau = lambda tau: (2*u/(1 + np.exp(-beta*u*k)) * np.exp(-(k*u + 1j*omega)*tau)
                                 * np.exp(-k*alpha))
    print(2*u/(1 + np.exp(-beta*u*k)) * np.exp(-(k*u + 1j*omega)*tau) * np.exp(-k*alpha),
          c_quad(integrand, -30, 30))
    print(2*u*np.tanh(beta*u*k/2)/(k*u + 1j*omega)*np.exp(-k*alpha),
          c_quad(integrand_tau, 0, beta))
    """

def test_chiral_integral():
    beta = 2.6
    tau = 0.7
    taup = 0.86
    k = 0.76
    L = 50
    omega = 2*np.pi/beta * 1
    integrand1 = lambda x: np.exp(1j*k*x) / beta / np.sin(np.pi/beta * (taup + 1j * x))
    integrand2 = lambda x: np.exp(1j*k*x) / beta / np.sin(np.pi/beta * (tau - 1j * x))
    integral = c_quad(integrand1, -L, L)[0] * c_quad(integrand2, -L, L)[0]
    integral_analytic = 4*np.exp(k*(tau - taup)) / (1 + np.exp(beta*k)) / (1 + np.exp(-beta*k))
    integrand_tau = lambda tau: (4*np.exp(k*(tau - taup)) * np.exp(-1j * omega * (tau + taup))
                                 / (1 + np.exp(beta*k)) / (1 + np.exp(-beta*k)))
    integral_tau = c_quad(integrand_tau, -taup, beta - taup)[0]
    integral_tau_analytic = (4 * np.exp(-2*k*taup) / (k - 1j * omega)
                             * np.tanh(beta*k/2) / (1 + np.exp(-beta*k)))
    integrand_taup = lambda taup: (4 * np.exp(-2*k*taup) / (k - 1j * omega)
                                   * np.tanh(beta*k/2) / (1 + np.exp(-beta*k)))
    integral_taup = c_quad(integrand_taup, 0, beta)[0]
    integral_taup_analytic = (2/k / (k - 1j * omega) * np.tanh(beta*k/2)
                              * (1 - np.exp(-2*beta*k)) / (1 + np.exp(-beta*k)))
    print("x and x' integration:")
    print("NUMERIC: ", integral)
    print("ANALYTIC: ", integral_analytic)
    print("tau integration:")
    print("NUMERIC: ", integral_tau)
    print("ANALYTIC: ", integral_tau_analytic)
    print("tau' integration:")
    print("NUMERIC: ", integral_taup)
    print("ANALYTIC: ", integral_taup_analytic)


def hyp2f1(a, b, c, z, atol=1e-10, max_steps=1000):
    series = 0
    value = 1
    for n in range(max_steps):
        series += value
        value *= z / (n+1) * (a + n) * (b + n) / (c + n)
        if np.abs(value / series) < atol:
            return series
    print(f"Warning, precision of {atol} not reached in {max_steps} "
          f"(estimate: {np.abs(value / series)})")
    return series


def test_f_ell_tau_transform():
    """
    integrate E^(I 0.76 x - I (2 2+1)pi / 2.6 t) csc(pi/2.6 (t + I x)) sign(x)
    from x=-inf to inf, t=0 to 2.6
    result: 0.106587 - 0.847302 i
    analytic: 2*beta / (k + 1j * omega) (--> 't - I x' gives 'k - 1j * omega')
    """
    beta = 2.6
    n = 3
    omega = (2*n + 1) * np.pi/beta
    x = 0.31
    numeric = c_quad(lambda tau: np.exp(-1j * tau * omega) / np.sin(np.pi/beta * (tau + 1j * x)),
                     0, beta)
    analytic = -2j*beta * np.sign(x) * np.exp(-omega * x) * np.heaviside(omega * x, 1)
    print(numeric, analytic)


def test_f_ell_general_tau_transform():
    """
    integrate E^(I (2 2 + 1) pi t / 2.6) csc(pi/2.6 (t + I 0.31))
    / (sin(pi t / 2.6)^2 + sinh(pi 0.31 / 2.6)^2)^3 from t=0 to 2.6
    result: -116.57 i

    TODO: verify this for the full transform
    res = 4**K*beta**2/(2*sy.pi*sy.I) * sy.gamma(n+1+K) * sy.gamma(-K) / sy.gamma(K)
    / sy.gamma(n+2) * sy.gamma((1+K)/2 + z) / sy.gamma((1-K)/2 + z)
    * sy.hyper((1-K, (1+K)/2 + z), (n+2,), 2)

    res = -1j*beta**2 / (2*np.pi) * 4**K*special.binom(n+K, K-1)
    * np.sum([(-1)**j/np.math.factorial(j) * special.gamma(1 - K + j)
              /special.gamma(1 - K)* special.gamma(1 + K + j)
              /special.gamma(1 + K) * special.gamma(n+2)
              / special.gamma(n+2+j) * special.gamma(-K-j)
              * special.gamma((1+K)/2 + j + z) / special.gamma(z + (1-K)/2)
              for j in range(100)])

    DERIVATIVE PRODUCT RULE:
    x, c = sy.symbols("x c", positive=True)
    m, q, p = sy.symbols("m q p", positive=True, integer=True)
    xi = -0.31
    m = 1
    subs = {x : np.exp(-2*xi), c : np.exp(2*xi), q : 5.3, p : 6.1}
    term = sy.diff(x**p * (x-c)**q, x, m)
    term2 = (sy.gamma(p+1)/sy.gamma(p+1-m) * (-c)**q * x**(p-m)
             * sy.hyper((-q, p+1), (p+1-m,), x/c))
    print(term.evalf(subs=subs), term2.evalf(subs=subs))
    """
    beta = 2.6
    n = 2
    omega = (2*n + 1) * np.pi/beta
    x = -0.31
    K = 0.5
    # xi = np.pi * x / beta
    # analytic = (-beta * sy.I * 2**K * sy.binomial(n+K, K-1)
    #             * sy.exp(-x * omega) / sy.sinh(2*xi)**(K+1) * np.sign(x)
    #             * sy.hyper((1-K,1+K), (n+2,), 1 / (1 - sy.exp(4*xi)))).evalf()
    # analytic = (-beta * 1j * 2**K * special.binom(n+K, K-1) * np.exp(-x*omega)
    #             * np.sign(x) / np.sinh(2*np.pi*x/beta +0j)**(K+1)
    #             * special.hyp2f1(1-K, 1+K, n+2, 1/(1 - np.exp(4*np.pi*x/beta))))
    if x >= 0:
        analytic = (-2*beta * 1j * 4**K * special.binom(n+K, K-1) * np.exp(-x*omega)
                    * np.exp(-2*np.pi*x / beta * (K+1))
                    * special.hyp2f1(1+K, n+1+K, n+2, np.exp(-4*np.pi*x/beta)))
    else:
        analytic = (2*beta * 1j * 4**K * special.binom(n+K, K) * np.exp(x*omega)
                    * np.exp(2*np.pi*x / beta * K)
                    * special.hyp2f1(K, n+1+K, n+1, np.exp(4*np.pi*x/beta)))
    numeric = c_quad(lambda tau: (np.exp(1j * omega * tau)
                                  / np.sin(np.pi/beta * (tau + 1j * x))
                                  * (np.sin(np.pi*tau/beta)**2
                                     + np.sinh(np.pi*x/beta)**2)**(-K)),
                     0, beta)
    print(numeric, analytic)


def test_f_ell_general_transform(mode=0):
    """

    """
    beta = 2.6
    n = 2
    omega = (2*n + 1) * np.pi/beta
    k = 0.76
    K = 0.1
    if mode == 1:
        # integrand_x = lambda x: (-beta * 1j * 2**K * special.binom(n+K, K-1) * np.exp(-x*omega)
        #                          * np.sign(x) / np.sinh(2*np.pi*x/beta+0j)**(K+1)
        #                          * special.hyp2f1(1-K, 1+K, n+2, 1/(1 - np.exp(4*np.pi*x/beta)))
        #                          * np.exp(-1j * k * x))
        integrand_x = lambda x: (-2*beta * 1j * 4**K * special.binom(n+K, K-1) * np.exp(-x*omega)
                                 * np.exp(-2*np.pi*x / beta * (K+1))
                                 * special.hyp2f1(1+K, n+1+K, n+2, np.exp(-4*np.pi*x/beta))
                                 * np.exp(-1j * k * x))
        numeric = c_quad(integrand_x, 0, 50)
        z = beta / (4*np.pi) * (omega + 1j * k)
        # analytic = complex((4**K*beta**2/(2*sy.pi*sy.I) * sy.gamma(n+1+K)
        #                     * sy.gamma(-K) / sy.gamma(K)
        #                     / sy.gamma(n+2) * sy.gamma((1+K)/2 + z) / sy.gamma((1-K)/2 + z)
        #                     * sy.hyper((1-K, (1+K)/2 + z), (n+2,), 1)).evalf())
        analytic = complex((4**K*beta**2 / (2*sy.pi*sy.I) * sy.binomial(n+K, K-1)
                            * sy.hyper((1+K, n+1+K, (K+1) / 2 + z),
                                       (n+2, (K+1) / 2 + z + 1), 1)
                            / sy.gamma((K+1) / 2 + z + 1) * sy.gamma((K+1) / 2 + z)).evalf())
    elif mode == 2:
        z_bar = beta / (4*np.pi) * (omega - 1j * k)
        integrand_x = lambda x: (2*beta * 1j * 4**K * special.binom(n+K, K) * np.exp(x*omega)
                                 * np.exp(2*np.pi*x / beta * K)
                                 * special.hyp2f1(K, n+1+K, n+1, np.exp(4*np.pi*x/beta))
                                 * np.exp(-1j * k * x))
        numeric = c_quad(integrand_x, -50, 0)
        analytic = complex((-4**K*beta**2 / (2*sy.pi*sy.I) * sy.binomial(n+K, K)
                            * sy.hyper((K, n+1+K, K / 2 + z_bar),
                                       (n+1, K / 2 + z_bar + 1), 1)
                            / sy.gamma(K / 2 + z_bar + 1) * sy.gamma(K / 2 + z_bar)).evalf())
    else: ### full transformation
        n_val = (beta*omega / (np.pi*1j) - 1) / 2
        z_val = beta / (4*np.pi*1j) * (omega - k)
        z_bar_val = beta / (4*np.pi*1j) * (omega + k)
        part1 = (sy.hyper((K+1, K+1 + n_val, (K+1)/2 + z_val), (n_val + 2, (K+3)/2 + z_val), 1)
                 / (special.gamma(K) * special.gamma(n_val + 2) * ((K+1)/2 + z_val)))
        part2 = (sy.hyper((K, K+1 + n_val, K/2 + z_bar_val), (n_val + 1, K/2 + 1 + z_bar_val), 1)
                 / (special.gamma(K+1) * special.gamma(n_val + 1) * (K/2 + z_bar_val)))
        analytic = 4**K * beta**2/(2*np.pi*1j) * special.gamma(K + n_val + 1) * (part1 - part2)
    print(numeric, analytic)


def general_f_omega_k(omega, k, beta=2.6, K=0.2, digits=8):
    """
    integrate E^(I omega_n^F tau - I k x) csc(pi/beta (tau + I x)) (sin^2 + sinh^2)**(-K)
    """
    n_val = (beta*omega / (np.pi*1j) - 1) / 2
    z_val = beta / (4*np.pi*1j) * (omega - k)
    z_bar_val = beta / (4*np.pi*1j) * (omega + k)
    prefactor = 4**K * beta**2/(2*np.pi*1j) * special.gamma(K + n_val + 1)
    part1 = complex(sy.hyper((K+1, K+1 + n_val, (K+1)/2 + z_val), 
                             (n_val + 2, (K+3)/2 + z_val), 1).evalf(n=digits))
    factor1 = special.gamma(K) * special.gamma(n_val + 2) * ((K+1)/2 + z_val)
    part2 = complex(sy.hyper((K, K+1 + n_val, K/2 + z_bar_val), 
                             (n_val + 1, K/2 + 1 + z_bar_val), 1).evalf(n=digits))
    factor2 = special.gamma(K+1) * special.gamma(n_val + 1) * (K/2 + z_bar_val)
    return prefactor * (part1 / factor1 - part2 / factor2)


def g_func(omega, k, beta=2.6, K=0.2, digits=8):
    """
    integrate E^(I omega_n^F tau - I k x) csc(pi/beta (tau + I x)) (sin^2 + sinh^2)**(-K)
    """
    mp.mp.dps = digits
    n_val = (beta*omega / (np.pi*1j) - 1) / 2
    z_val = beta / (4*np.pi*1j) * (omega - k)
    z_bar_val = beta / (4*np.pi*1j) * (omega + k)
    prefactor = 4**K * beta**2/(2*np.pi*1j) * special.gamma(K + n_val + 1)
    part1 = complex(mp.hyper((K+1, K+1 + n_val, (K+1)/2 + z_val), 
                             (n_val + 2, (K+3)/2 + z_val), 1))
    factor1 = special.gamma(K) * special.gamma(n_val + 2) * ((K+1)/2 + z_val)
    part2 = complex(mp.hyper((K, K+1 + n_val, K/2 + z_bar_val), 
                             (n_val + 1, K/2 + 1 + z_bar_val), 1))
    factor2 = special.gamma(K+1) * special.gamma(n_val + 1) * (K/2 + z_bar_val)
    return prefactor * (part1 / factor1 - part2 / factor2)


def test_f_ell_semi_integral():
    beta = 2.6
    tau = 0.7
    k = 0.76
    L = 50
    omega = 2*np.pi/beta * 1
    integrand_x = lambda x: np.exp(1j*k*x) / (2*beta) / np.sin(np.pi/beta * (tau + 1j * x))
    integral_x = c_quad(integrand_x, 0, L)[0]
    n = np.arange(300)
    # integral_x_analytic = np.sum(np.exp((2*n+1)*1j*np.pi*tau/beta)
    #                              / (2*n+1 - 1j*k*beta/np.pi)) / (1j*np.pi)
    integral_x_analytic = (np.exp(1j*np.pi/beta) / (1j * np.pi) / (1 - 1j * k * beta / np.pi)
                           * hyp2f1(1,
                                    0.5 - 1j * beta * k / (2*np.pi),
                                    1.5 - 1j * beta * k / (2*np.pi),
                                    np.exp(2j * np.pi/beta))
                           )
    integrand_tau = lambda tau: (np.sum(np.exp((2*n+1)*1j*np.pi*tau/beta)
                                        / (2*n+1 - 1j*k*beta/np.pi))
                                 / (1j*np.pi) * np.exp(-1j*omega*tau))
    integral_tau = c_quad(integrand_tau, 0, beta)[0]
    print("x integration:")
    print("NUMERIC: ", integral_x)
    print("ANALYTIC: ", integral_x_analytic)
    print("tau integration:")
    print("NUMERIC: ", integral_tau)
    # print("ANALYTIC: ", integral_tau_analytic)



def main():
    """main"""
    print(__doc__)
    interaction_par = 1.3
    length = 2e2
    alpha = 1e-4
    k_val = 0.43
    beta = 2.1
    omega = 2*np.pi/beta * 2    # second Matsubara freuqency
    tau = beta / 2.4
    par = {"K" : interaction_par,
           "L" : length,
           "alpha" : alpha,
           "k" : k_val,
           "omega" : omega,
           "beta" : beta,
           "tau" : tau,}

    # test_zero_t_no_int(par)
    # test_zero_t_with_int(par)
    # test_zero_t_with_int_full(par)
    # test_with_t_no_int(par)
    # test_with_t_no_int_full(par)
    # test_chiral_integral()
    # test_f_ell_semi_integral()
    # test_f_ell_tau_transform()
    test_f_ell_general_tau_transform()


    return 0

if __name__ == "__main__":
    main()
