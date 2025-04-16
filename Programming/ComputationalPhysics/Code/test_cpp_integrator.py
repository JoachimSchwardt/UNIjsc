import rk_integrator_cpp
import numpy as np

# t_values = np.zeros(steps)
# y_values = np.zeros((steps, 1))
# y_current = np.zeros(steps)
# y_next = np.zeros(steps)
# y_previous = np.zeros(steps)
# y_values[0] = np.array([x0])
# temp = np.zeros(steps)
# rk_integrator_cpp.rk_int(fktcpp, t_0_values, t_values, y_values, 
#                          y_current, y_next, y_previous, 0.0, 1e-8, temp)

def rk_int_cpp(func, t_0_values, y_0, tau_init=0.0, atol=1e-8):
    steps = t_0_values.size
    t_values = np.zeros(steps)
    y_current = np.zeros(steps)
    y_next = np.zeros(steps)
    y_previous = np.zeros(steps)
    temp = np.zeros(steps)
    y_values = np.zeros((steps, y_0.size))
    y_values[0] = y_0
    rk_integrator_cpp.rk_int(func, t_0_values, t_values, y_values, 
                             y_current, y_next, y_previous, 0.0, 1e-8, temp)
    return t_values, y_values

def test_cpp():
    x0 = 0.3
    t0 = 0.5
    t_final = 5.0
    steps = 51
    t_0_values = np.linspace(t0, t_final, steps)
    def fktcpp(t, x):
        return t - x
    def y_analytic(t, t0=0, x0=0):
        return (x0 - t0 + 1) * np.exp(-(t-t0)) + t - 1
    t_cpp, y_cpp = rk_int_cpp(fktcpp, t_0_values, np.array([x0]))
    print(y_cpp[-1] - y_analytic(t_cpp[-1], t0, x0))
