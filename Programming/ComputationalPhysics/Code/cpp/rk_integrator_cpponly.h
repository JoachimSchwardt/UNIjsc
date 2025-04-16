/**
Simple implementation of a step-size adaptive RK4 integrator.

compile using ::
    g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-masm=intel main.cpp -lm -lmvec

@author: Joachim Schwardt
*/

#ifndef RK_INTEGRATOR_H_INCLUDED
#define RK_INTEGRATOR_H_INCLUDED

#include<iostream>
#include<chrono>
#include<vector>

/// https://stackoverflow.com/questions/6321839/how-to-disable-warnings-for-particular-include-files
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <pybind11/pybind11.h>      // this library is very evil; need to ignore a lot of errors
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#pragma GCC diagnostic pop


namespace py = pybind11;

using vec_t = std::vector<double>;
using vec_py_t = py::array_t<double>;
using func_t = std::function<vec_t(double, vec_t)>;


struct ODEParameters
{
    ODEParameters(double tau_ = 0.0, const double atol_ = 1e-8)
        : tau{tau_}, atol{atol_}
    { }
    double tau;
    const double atol;
    bool redo_step = true;
    const double min_update_ratio = 2.0;
};

/// Operator overloads for simple array operations
vec_t operator+(const vec_t& vec1, const vec_t& vec2)
{
    /** Overload operator+ for vector addition */
    vec_t result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

vec_t operator*(const vec_t& vec, const double scalar)
{
    /** Overload operator+ for vector * scalar multiplication */
    vec_t result(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
    {
        result[i] = vec[i] * scalar;
    }
    return result;
}
inline vec_t operator*(const double scalar, const vec_t& vec) noexcept
{
    /** Overload operator+ for scalar * vector multiplication */
    return vec * scalar;
}


/// RK4 Integrator
void rk_step(const func_t func, const double t_n, const vec_t& y_n, vec_t& y_n_plus1, const double tau)
{
    /** Execute a single step the the RK4 method */
    vec_t k_1 = func(t_n, y_n);
    vec_t k_2 = func(t_n + tau * 0.5, y_n + tau * k_1 * 0.5);
    vec_t k_3 = func(t_n + tau * 0.5, y_n + tau * k_2 * 0.5);
    vec_t k_4 = func(t_n + tau, y_n + tau * k_3);
    for (size_t i = 0; i < y_n.size(); ++i)
    {
        y_n_plus1[i] = y_n[i] + tau / 6 * (k_1[i] + 2 * k_2[i] + 2 * k_3[i] + k_4[i]);
    }
}

double estimate_error(const func_t func, const vec_t& y_n, const double t_n, const vec_t& y_n_plus1,
                      const double tau)
{
    /** Estimate the error by backward integration */
    vec_t y_n_plus1_minus1(y_n.size());
    rk_step(func, t_n + tau, y_n_plus1, y_n_plus1_minus1, -tau);
    double max_val = 0.0;
    for (size_t i = 0; i < y_n.size(); ++i)
    {
        double abs_val = std::abs(y_n_plus1_minus1[i] - y_n[i]);
        if (abs_val > max_val)
        {
            max_val = abs_val;
        }
    }
    return max_val;
}

double adapt_stepsize(const double err, auto& ode_params)
{
    /** Adapt the stepsize 'tau' based on the current error estimate. */
    double abs_err_ratio = err / ode_params.atol;
    double tau_new = 0.0;
    if (abs_err_ratio < 1 / (4.0 * ode_params.min_update_ratio))
    {
        tau_new = ode_params.min_update_ratio * ode_params.tau;
    }
    else if (abs_err_ratio < 1.0)
    {
        tau_new = ode_params.tau;
    }
    else if (abs_err_ratio < 4.0 * ode_params.min_update_ratio)
    {
        tau_new = ode_params.tau / ode_params.min_update_ratio;
    }
    else
    {
        tau_new = ode_params.tau / (ode_params.min_update_ratio * ode_params.min_update_ratio);
    } 
    ode_params.redo_step = (tau_new < ode_params.tau);
    return tau_new;
}

void _rk_int_step(func_t func, const double t_initial, const double t_final, auto y_values_u, 
                  auto& ode_params, const py::ssize_t index, const int length, const int max_iterations)
{
    /** Integrate an ODE from t_initial to t_final starting from y_initial (SINGLE STEP). */
    double tau_new = t_final - t_initial;    // try integrating in one step
    double t_current = t_initial;
    vec_t y_current(length);
    vec_t y_next(length);
    for (py::ssize_t i = 0; i < length; ++i){
        y_current[i] = y_values_u(index-1, i);
    }
    int iteration = 0;
    do
    {
        ode_params.redo_step = true;
        while (ode_params.redo_step)
        {
            ode_params.tau = tau_new;
            rk_step(func, t_current, y_current, y_next, ode_params.tau);
            double err = estimate_error(func, y_current, t_current, y_next, ode_params.tau);
            tau_new = adapt_stepsize(err, ode_params);
            if (++iteration > max_iterations) {
                std::cout << "Maximal number of iterations " << max_iterations << " reached!\n";
                goto END_LOOP;
            }
        }
        t_current += ode_params.tau;
        for (py::ssize_t i = 0; i < length; ++i){
            y_current[i] = y_next[i];
        }
    } while(t_current < t_final - 1e-4 * tau_new);
END_LOOP:
    for (py::ssize_t i = 0; i < length; ++i){
        y_values_u(index, i) = y_current[i];
    }
}

auto rk_int(func_t func, vec_py_t t_values, vec_py_t y_0, const double atol, const int max_iterations)
{
    /** Integrate an ODE for given initial value y_0 for the times in 't_values'.
        https://www.linyuanshi.me/post/pybind11-array/
        https://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html
        https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
    */
    const int num_t_values = t_values.request().size;
    const int length = y_0.request().size;

    auto y_values = vec_py_t(num_t_values * length);
    y_values.resize({num_t_values, length});

    // allow unrestricted (and unsafe ;) ) access to the memory allocated by NumPy (or Pybind for that matter)
    auto y_0_u = y_0.unchecked<1>();
    auto t_values_u = t_values.unchecked<1>();
    auto y_values_u = y_values.mutable_unchecked<2>();

    ODEParameters ode_params(0.0, atol);
    double t_initial = t_values_u(0);
    for (py::ssize_t i = 0; i < length; ++i)
    {
        y_values_u(0, i) = y_0_u(i);
    }
    for (py::ssize_t i = 1; i < num_t_values; ++i)
    {
        double t_final = t_values_u(i);
        _rk_int_step(func, t_initial, t_final, y_values_u, ode_params, i, length, max_iterations);
        t_initial = t_final;
    }
    return y_values;
}

#endif // RK_INTEGRATOR_H INCLUDED
