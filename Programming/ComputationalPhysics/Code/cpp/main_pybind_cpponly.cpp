#include <vector>
//#include <cmath>
#include "rk_integrator_cpponly.h"

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


vec_t derivative(double _t_value, vec_t y_value)
{
    /** TODO: This is where you implement your own function */
    return {y_value[1], -4.0 * std::pow(y_value[0], 3) + 2.0 * y_value[0] - 0.05};
}

auto hamilton_odeint(vec_py_t t_values, vec_py_t y_0, const double atol, const int max_iterations)
{
    auto derivative_partial = [](double t_value, vec_t y_value) {
        return derivative(t_value, y_value);
    };
    return rk_int(derivative_partial, t_values, y_0, atol, max_iterations);
}


int main()
{
    return 0;
}

PYBIND11_MODULE(hamilton_integrator_cpp, m) {
    m.doc() = "pybind11 rk4 integrator for specific Hamiltonian";    // optional module docstring
    m.def("hamilton_odeint", &hamilton_odeint, 
          "Integrate the Hamiltonian System for given initial value y_0 for the times in 't_values'.",
        py::arg("t_values"), py::arg("y_0"), py::arg("atol"), py::arg("max_iterations"));
}


/** c++ -Wall -O3 -march=native -mavx2 -ffast-math -fconcepts -fopenmp -shared -std=c++2a -fPIC \
$(/home/joachim/.conda/envs/spyder/bin/python3.9 -m pybind11 --includes) main_pybind_cpponly.cpp -lm -lmvec  \
-o hamilton_integrator_cpp$(python3.9-config --extension-suffix)
*/
