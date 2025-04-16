# Joachim Schwardt + 4768711
# Julian Fleck + 4759587

import numpy as np
import matplotlib.pyplot as plt


def function(t, x):
    return -2 * t * x**2


def exact_solution(t):
    return 1 / (t**2 + 1)


def butcherScheme(t, x, func, tau, A, b, c):
    steps = c.size
    kvals = np.zeros(steps)
    for j in range(steps):
        kvals[j] = func(t + c[j] * tau, x + tau * np.sum(A[j-1, :j] * kvals[:j]))
    return x + tau * np.sum(b * kvals)


def explizitRungeKutta(func, n, A, b, c):
    tk = 0
    x0 = 1
    tau = 1 / n

    xn = np.zeros(n+1)
    xn[0] = x0

    for i in range(n):
        # xn[i+1] = scheme(tk, xn[i])
        xn[i+1] = butcherScheme(tk, xn[i], func, tau, A, b, c)
        tk += tau

    return xn


def butcher_heun():
    A = np.array([[1]])
    b = np.array([0.5, 0.5])
    c = np.array([0, 1])
    return A, b, c


def butcher_rk2():
    A = np.array([[0.5]])
    b = np.array([0, 1])
    c = np.array([0, 0.5])
    return A, b, c


def butcher_rk4():
    A = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    b = np.array([1, 2, 2, 1]) / 6
    c = np.array([0, 0.5, 0.5, 1])
    return A, b, c


def test_scheme_convergence(func, exact_func, nvals, scheme):
    A, b, c = scheme()
    errors = np.zeros(nvals.size)
    for i in range(nvals.size):
        xn = explizitRungeKutta(func, nvals[i], A, b, c)
        tvals = np.linspace(0, 1, nvals[i]+1)
        xn_exact = exact_func(tvals)
        errors[i] = np.linalg.norm(xn - xn_exact, ord=np.inf)

    return errors


def regression(nvals, errors):
    log_n = np.log(nvals)
    log_err = np.log(errors)
    slope, offset = np.polyfit(log_n, log_err, deg=1)
    return slope, offset


def plot_convergence(ax, nvals, errors, label):
    slope, offset = regression(nvals, errors)
    col = ax._get_lines.get_next_color()
    ax.plot(nvals, errors, c=col, label=label)
    ax.plot(nvals, np.exp(offset) * nvals**slope, c=col, ls='--',
            label=fr"{label}: ~$\mathcal{{O}}(\tau^{{{-slope:.2f}}})$")


def main():
    # 2**2 = 4, ..., 2**7 = 128 (insgesamt 5 Werte)
    n_all = np.logspace(2, 7, 6, base=2).astype(int)

    heun_error = test_scheme_convergence(function, exact_solution, n_all, butcher_heun)
    rk2_error = test_scheme_convergence(function, exact_solution, n_all, butcher_rk2)
    rk4_error = test_scheme_convergence(function, exact_solution, n_all, butcher_rk4)

    _, ax = plt.subplots(figsize=(16, 9))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$\max_k\ |x_k - x(t_k)|$')
    plot_convergence(ax, n_all, heun_error, label='Heun')
    plot_convergence(ax, n_all, rk2_error, label='RK2')
    plot_convergence(ax, n_all, rk4_error, label='RK4')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
