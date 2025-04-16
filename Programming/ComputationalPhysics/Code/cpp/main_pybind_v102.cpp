/**
Simple implementation of a step-size adaptive RK4 integrator.

compile using ::
    g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-masm=intel main.cpp -lm -lmvec

@author: Joachim Schwardt


v102:
    changed the logic for adjusting stepsize as a lot of "back-and-forth" happened.
    --> algorithm used to jump between a "small" and "large" tau, essentially redoing a lot of work
    --> issue was identified as a "large" min_update_ratio of 2.0, and errors were ~2e-8 and ~7e-10 etc.
    --> now smaller ratio of 1.4142... to make the jumps smaller and redo less often when close to "atol".
*/

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

using vec_t = py::array_t<double>;
using func_t = std::function<vec_t(double, vec_t)>;


struct ODEParameters
{
    ODEParameters(double tau_ = 0.0, const double atol_ = 1e-8)
        : tau{tau_}, atol{atol_}, tau_temp{tau_}
    { }
    double tau;
    const double atol;
    double tau_temp;
    bool redo_step = true;
    const double min_update_ratio = std::sqrt(2.0); //2.0;
};

static int FUNC_COUNTER = 0;
/// RK4 Integrator
void rk_step(func_t func, const double t_n, vec_t y_n, auto y_n_u, auto y_n_plus1_u, 
             const double tau, const int length, vec_t temp, auto temp_u)
{
    FUNC_COUNTER+=4;
    /** Execute a single step the the RK4 method */
    vec_t k_1 = func(t_n, y_n);
    auto k_1_u = k_1.unchecked<1>();
    for (py::ssize_t i = 0; i < length; ++i)
    {
        temp_u(i) = y_n_u(i) + tau * k_1_u(i) * 0.5;
    }
    vec_t k_2 = func(t_n + tau * 0.5, temp);
    auto k_2_u = k_2.unchecked<1>();
    for (py::ssize_t i = 0; i < length; ++i)
    {
        temp_u(i) = y_n_u(i) + tau * k_2_u(i) * 0.5;
    }
    vec_t k_3 = func(t_n + tau * 0.5, temp);
    auto k_3_u = k_3.unchecked<1>();
    for (py::ssize_t i = 0; i < length; ++i)
    {
        temp_u(i) = y_n_u(i) + tau * k_3_u(i);
    }
    vec_t k_4 = func(t_n + tau, temp);
    auto k_4_u = k_4.unchecked<1>();
    for (py::ssize_t i = 0; i < length; ++i)
    {
        y_n_plus1_u(i) = y_n_u(i) + tau / 6 * (k_1_u(i) + 2 * k_2_u(i) + 2 * k_3_u(i) + k_4_u(i));
    }
}

double estimate_error(func_t func, auto y_n_u, const double t_n, 
                      vec_t y_n_plus1, auto y_n_plus1_u, auto y_n_plus1_minus1_u, 
                      const double tau, const int length, vec_t temp, auto temp_u)
{
    /** Estimate the error by backward integration */
    rk_step(func, t_n + tau, y_n_plus1, y_n_plus1_u, y_n_plus1_minus1_u, -tau, length, temp, temp_u);
    double max_val = 0.0;
    for (py::ssize_t i = 0; i < length; ++i)
    {
        double abs_val = std::abs(y_n_plus1_minus1_u(i) - y_n_u(i));
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
        std::cerr<<"\n\nWARNING! large tau adjust!\n\n";
        tau_new = ode_params.tau / (ode_params.min_update_ratio * ode_params.min_update_ratio);
    } 
//    else
//    {
//        std::cerr<<"\n\nWARNING! LARGE TAU ADJUST!!\n\n";
//        tau_new = ode_params.tau / std::pow(ode_params.min_update_ratio, 4);
//    }
    ode_params.redo_step = (tau_new < ode_params.tau);
    return tau_new;
}


void _rk_int_step(func_t func, const double t_initial, const double t_final, auto t_values_u,
                  auto y_values_u, vec_t y_current, auto y_current_u, vec_t y_next, auto y_next_u, 
                  auto y_previous_u, auto& ode_params, const py::ssize_t index, const int length,
                  const int max_iterations, vec_t temp, auto temp_u)
{
    /** Integrate an ODE from t_initial to t_final starting from y_initial (SINGLE STEP). */
    double tau_new = t_final - t_initial;    // try integrating in one step
    double t_current = t_initial;
    for (py::ssize_t i = 0; i < length; ++i){
        y_current_u(i) = y_values_u(index-1, i);
    }
    int iteration = 0;
    do
    {
        ode_params.redo_step = true;
        while (ode_params.redo_step)
        {
std::cerr<<"y_current, y_next, y_previous, temp:\n";
            ode_params.tau = tau_new;
            rk_step(func, t_current, y_current, y_current_u, y_next_u, ode_params.tau, length, temp, temp_u);
            double err = estimate_error(func, y_current_u, t_current, y_next, y_next_u, 
                                        y_previous_u, ode_params.tau, length, temp, temp_u);
            tau_new = adapt_stepsize(err, ode_params);
for (py::ssize_t i = 0; i < length; ++i){
std::cerr<<y_current_u(i)<<", "<<y_next_u(i)
         <<", "<<y_previous_u(i)<<", "<<temp_u(i)<<", \n";
}
std::cerr<<"TIME: "<<t_current<<", ERROR: "<<err<<", TAU: "<<ode_params.tau<<", NEW_TAU: "<< tau_new
         <<", REDO: "<<ode_params.redo_step<<", ITER: "<<iteration<<"\n\n";
            if (++iteration > max_iterations) {
                std::cout<<"Maximal number of iterations "<<max_iterations<<" reached!\n";
                goto END_LOOP;
            }
        }
        t_current += ode_params.tau;
        for (py::ssize_t i = 0; i < length; ++i){
            y_current_u(i) = y_next_u(i);
        }
    } while(t_current < t_final - 1e-4 * tau_new);
END_LOOP:
    t_values_u(index) = t_current;
    for (py::ssize_t i = 0; i < length; ++i){
        y_values_u(index, i) = y_current_u(i);
    }
std::cerr<<"ITERATIONS: "<<iteration<<", \n";
}


void rk_int(func_t func, vec_t t_0_values, vec_t t_values, 
            vec_t y_values, vec_t y_current, vec_t y_next, vec_t y_previous,
            const double atol, const int max_iterations, vec_t temp)
{
    /** Integrate an ODE for given initial value y_0 for the times in 't_values'.
        https://www.linyuanshi.me/post/pybind11-array/
        https://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html
        https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
    */
    auto t_0_values_u = t_0_values.unchecked<1>();
    auto t_values_u = t_values.mutable_unchecked<1>();
    auto y_values_u = y_values.mutable_unchecked<2>();
    auto y_current_u = y_current.mutable_unchecked<1>();
    auto y_next_u = y_next.mutable_unchecked<1>();
    auto y_previous_u = y_previous.mutable_unchecked<1>();
    const int length = y_current.request().size;
    auto temp_u = temp.mutable_unchecked<1>();

//std::cerr<<"y_current, y_next, y_previous, temp: \n";
//for (py::ssize_t i = 0; i < length; ++i){
//std::cerr<<y_current_u(i)<<", "<<y_next_u(i)
//         <<", "<<y_previous_u(i)<<", "<<temp_u(i)<<", \n";
//}
//std::cerr<<"\n\ny_values: \n";
//
//for (py::ssize_t i = 0; i < t_values.request().size; ++i){
//    for (py::ssize_t j = 0; j < length; ++j){
//    std::cerr<<y_values_u(i,j)<<", ";
//    }
//std::cerr<<" \n";
//}
//std::cerr<<"\n\nlength: "<<length<<" \n\n";
//std::cerr<<"t_0_values, t_values: \n";
//for (py::ssize_t i = 0; i < t_values.request().size; ++i){
//std::cerr<<t_0_values_u(i)<<", "<<t_values_u(i)<<", \n";
//}

    ODEParameters ode_params(0.0, atol);
    t_values_u(0) = t_0_values_u(0);
    for (py::ssize_t i = 1; i < t_values.request().size; ++i)
    {
        double t_initial = t_values_u(i-1);
        double t_final = t_0_values_u(i);
        _rk_int_step(func, t_initial, t_final, t_values_u, 
                     y_values_u, y_current, y_current_u, y_next, y_next_u, y_previous_u, 
                     ode_params, i, length, max_iterations, temp, temp_u);
    }
    std::cerr<<"TOTAL FUNCTION CALLS: "<<FUNC_COUNTER<<'\n';
}


int main()
{
    /**
    Benchmarks for an AMD Ryzen 7 5800X
    */

//    const vec_t y_0 = {0.3};
//    const double t_0 = 0.5;
//    const double t_final = 5.0;
//    const int n_values = 50;
//    const vec_t t_values = linspace(t_0, t_final, n_values);
//
//    auto t1 = std::chrono::steady_clock::now();
//    const ODEResult ode_result = rk_int(&fkt, t_values, y_0);
//    auto t2 = std::chrono::steady_clock::now();
//
//    const double analytic = y_analytic(ode_result.t_values(n_values - 1), t_0, y_0(0));
//    std::cout << abs_rel_error(ode_result.y_values(n_values - 1, 0), analytic) << '\n';
//    for (size_t i = 0; i < ode_result.t_values.size(); ++i)
//    {
//        std::cout << ode_result.t_values(i) << ", " << ode_result.y_values(i, 0) << '\n';
//    }
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
//    std::cout << "Total runtime: " << duration << " ms.\n";
    return 0;
}

PYBIND11_MODULE(rk_integrator_cpp, m) {
    m.doc() = "pybind11 rk4 integrator";    // optional module docstring
    m.def("rk_int", &rk_int, "Integrate an ODE for given initial value y_0 for the times in 't_values'.",
        py::arg("func"), py::arg("t_0_values"), py::arg("t_values"), 
        py::arg("y_values"), py::arg("y_current"), py::arg("y_next"), py::arg("y_previous"), 
        py::arg("atol"), py::arg("max_iterations"), py::arg("temp"));
}


/** c++ -Wall -O3 -march=native -mavx2 -ffast-math -fconcepts -fopenmp -shared -std=c++2a -fPIC \
$(/home/joachim/.conda/envs/spyder/bin/python3.9 -m pybind11 --includes) main_pybind.cpp -lm -lmvec  \
-o rk_integrator_cpp$(python3.9-config --extension-suffix)
*/
