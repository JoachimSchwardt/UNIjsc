/**
Simple implementation of a step-size adaptive RK4 integrator.

compile using ::
    g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-masm=intel main.cpp -lm -lmvec

@author: Joachim Schwardt
*/

#include<iostream>
#include<chrono>
#include<vector>
#include<tuple>
//#include<immintrin.h>

/// https://stackoverflow.com/questions/6321839/how-to-disable-warnings-for-particular-include-files
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <pybind11/pybind11.h>      // this library is very evil; need to ignore a lot of errors
#include <pybind11/stl.h>
#pragma GCC diagnostic pop

//#define OMP_NUM_THREADS 16
//#pragma GCC target ("avx2")


//using dtype = double;
//constexpr int SIMD_SIZE = 4;                   // number of 'dtype's fitting into avx2 register
using vec_t = std::vector<double>;
using arr_2d_t = std::vector<std::vector<double>>;
using func_t = std::function<vec_t(double, vec_t)>;

struct ODE_Parameters
{
    ODE_Parameters(const double atol_ = 1e-8)
        : atol{atol_}
    { }
    const double atol;
    double tau = 0.0;
    bool redo_step = true;
    const double min_update_ratio = 2.0;
};

/// Helper functions and overloads
template <typename T>
std::vector<T> linspace(T xmin, T xmax, int num)
{
    T dx = (xmax - xmin) / (num - 1);
    std::vector<T> vals(num);
    for (int i = 0; i < num - 1; ++i)
    {
        vals[i] = xmin;
        xmin += dx;
    }
    vals[num - 1] = xmax;
    return vals;
}

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


static int counter = 0;
/// Hamilton equations
vec_t fkt(double t_val, const vec_t& y_val)
{
    ++counter;
    vec_t result(2);
    result[0] = y_val[1];
    result[1] = -4.0 * std::pow(y_val[0], 3) + 2.0 * y_val[0] - 0.05;
    return result;
}

/// RK4 Integrator
vec_t rk_step(const func_t func, const double t_n, const vec_t& y_n, const double tau)
{
    /** Execute a single step the the RK4 method */
    vec_t k_1 = func(t_n, y_n);
    vec_t k_2 = func(t_n + tau * 0.5, y_n + tau * k_1 * 0.5);
    vec_t k_3 = func(t_n + tau * 0.5, y_n + tau * k_2 * 0.5);
    vec_t k_4 = func(t_n + tau, y_n + tau * k_3);
    vec_t y_new(y_n.size());
    for (size_t i = 0; i < y_n.size(); ++i)
    {
        y_new[i] = y_n[i] + tau / 6 * (k_1[i] + 2 * k_2[i] + 2 * k_3[i] + k_4[i]);
    }
    return y_new;
}

double estimate_error(const func_t func, const vec_t& y_n, const double t_n, const vec_t& y_n_plus1,
                      const double tau)
{
    /** Estimate the error by backward integration */
    vec_t y_n_plus1_minus1(y_n.size());
    y_n_plus1_minus1 = rk_step(func, t_n + tau, y_n_plus1, -tau);
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
    double tau_new;
    if (abs_err_ratio < 1 / ode_params.min_update_ratio)
    {
        tau_new = ode_params.min_update_ratio * ode_params.tau;
    }
    else if (abs_err_ratio < 1.0)
    {
        tau_new = ode_params.tau;
    }
    else if (abs_err_ratio < ode_params.min_update_ratio)
    {
        tau_new = ode_params.tau / ode_params.min_update_ratio;
    }
    else
    {
        tau_new = ode_params.tau / (ode_params.min_update_ratio * std::pow(abs_err_ratio, 0.25));
    }
    ode_params.redo_step = (tau_new < ode_params.tau);
    return tau_new;
}

void _rk_int_step(const func_t func, const double t_initial, const double t_final,
                  const vec_t& y_initial, vec_t& t_values, arr_2d_t& y_values, const size_t i,
                  auto& ode_params)
{
    /** Integrate an ODE from t_initial to t_final starting from y_initial (SINGLE STEP). */
    ode_params.tau = t_final - t_initial;    // try integrating in one step
    double t_current = t_initial;
    vec_t y_current = y_initial;
    vec_t y_next(y_initial.size());

    while (t_current < t_final)
    {
        ode_params.redo_step = true;
        double tau_new = ode_params.tau;
        while (ode_params.redo_step)
        {
            ode_params.tau = tau_new;
            y_next = rk_step(func, t_current, y_current, ode_params.tau);
            double err = estimate_error(func, y_current, t_current, y_next, ode_params.tau);
            tau_new = adapt_stepsize(err, ode_params);
        }
        t_current += ode_params.tau;
        y_current = y_next;
    }
    t_values[i] = t_current;
    y_values[i] = y_current;
}

auto rk_int(const func_t func, const vec_t& t_0_values, const vec_t& y_0, const double atol = 1e-8)
{
    /** Integrate an ODE for given initial value y_0 for the times in 't_values'. */
    ODE_Parameters ode_params(atol);
    vec_t t_values(t_0_values.size());
    arr_2d_t y_values(t_0_values.size(), vec_t(y_0.size()));
    t_values[0] = t_0_values[0];
    y_values[0] = y_0;
    double t_initial = t_values[0];
    for (size_t i = 1; i < t_0_values.size(); ++i)
    {
        double t_final = t_0_values[i];
        vec_t y_initial = y_values[i - 1];
        _rk_int_step(func, t_initial, t_final, y_initial, t_values, y_values, i, ode_params);
        t_initial = t_final;
    }
    return std::make_tuple(t_values, y_values);
}


int main()
{
    /**
    Benchmarks for an AMD Ryzen 7 5800X

    Hamiltonian system:
    t_values = np.linspace(0.0, 100.0, 2000)
    atol : 1e-8
    y0 = [0.5, 0.6]

    %timeit -n 100 y_t = odeint(abl, y0, zeiten, rtol=1e-8, atol=1e-8, tfirst=True)
    2.09 ms ± 33.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    Function calls:  4195

    %timeit -n 100 rk_int(abl, zeiten, y0, atol=absf)
    1.66 ms ± 30.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    Function calls: 25256

    C++:
    ~3 ms
    Function calls: 28136
    */

    const vec_t y_0 = {0.5, 0.6};
    const double t_0 = 0.0;
    const double t_final = 100.0;
    const int n_values = 2000;
    const vec_t t_0_values = linspace(t_0, t_final, n_values);

    auto t1 = std::chrono::steady_clock::now();
    auto result = rk_int(fkt, t_0_values, y_0);
    auto t2 = std::chrono::steady_clock::now();

//    std::cout << abs_rel_error(ode_result.y_values[n_values - 1][0], analytic) << '\n';
//    for (size_t i = 0; i < ode_result.t_values.size(); ++i)
//    {
//        std::cout << ode_result.t_values[i] << ", " << ode_result.y_values[i][0] << '\n';
//    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total runtime: " << duration << " us.\n";
    std::cout<<"COUNTER: "<<counter<<'\n';
    return 0;
}


//namespace py = pybind11;

//PYBIND11_MODULE(rk_integrator_cpp, m) {
//    m.doc() = "pybind11 rk4 integrator";    // optional module docstring
//    py::class_<ODEResult>(m, "ODEResult", py::dynamic_attr())
////        .def(py::init<int, int>())
//        .def_readwrite("t_values", &ODEResult::t_values)
//        .def_readwrite("y_values", &ODEResult::y_values)
//        .def_readonly("atol", &ODEResult::atol);
//
//    m.def("rk_int", &rk_int, "Integrate an ODE for given initial value y_0 for the times in 't_values'.",
//        py::arg("func"), py::arg("t_values"), py::arg("y_0"), py::arg("tau_init"), py::arg("atol"));
//}

/** c++ -Wall -O3 -march=native -mavx2 -ffast-math -fconcepts -fopenmp -shared -std=c++2a -fPIC \
$(/home/joachim/.conda/envs/spyder/bin/python3.9 -m pybind11 --includes) main_pybind.cpp -lm -lmvec  \
-o rk_integrator_cpp$(python3.9-config --extension-suffix)
*/
