#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <utility>
#include "fftw3.h"
#include "mpreal.h"
#include "cdtype.h"
#include "brentq.h"

using dtype = mpfr::mpreal;
constexpr unsigned int PRECISION = 30;

void initialize_precision(const unsigned int precision)
{
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(precision));
    std::cout.precision(precision);
}



/// ORBIT GENERATION



struct OrbitParameters
{
    int n_points = 4096;
    dtype k1 = "2.25";
    dtype k2 = "3.0";
    dtype k = "1.0";
};


inline dtype mod1(dtype val) noexcept
{
    return val - mpfr::floor(val);
}


void std_map_4d(dtype* q1, dtype* p1, dtype* q2, dtype* p2, const OrbitParameters& orb_par)
{
    /**
    */
    const dtype pi2 = 2 * mpfr::const_pi();
    dtype coupling;

    for (int i = 1; i < orb_par.n_points; ++i)
    {
//        if (i < 3)
//        {
//            std::cout << q1[i - 1] << '\n' << p1[i - 1] << '\n' << q2[i - 1] << '\n' << p2[i - 1] << '\n';
//        }
        q1[i] = mod1(q1[i - 1] + p1[i - 1]);
        q2[i] = mod1(q2[i - 1] + p2[i - 1]);
        coupling = orb_par.k / pi2 * mpfr::sin(pi2 * (q1[i] + q2[i]));
        p1[i] = mod1(p1[i - 1] + coupling + orb_par.k1 / pi2 * mpfr::sin(pi2 * q1[i]) + 0.5) - 0.5;
        p2[i] = mod1(p2[i - 1] + coupling + orb_par.k2 / pi2 * mpfr::sin(pi2 * q2[i]) + 0.5) - 0.5;
    }
}



/// FREQUENCY ANALYSIS


cdtype dft(dtype nu, const std::vector<cdtype>& signal)
{
    cdtype sum = {"0.0", "0.0"};
    for (size_t i = 0; i < signal.size(); i++)
    {
        dtype phase = -2 * i * mpfr::const_pi() * nu;
        cdtype exp = {mpfr::cos(phase), mpfr::sin(phase)};
        sum += signal[i] * exp;
    }
    return sum;
}


cdtype _fourier_j_numeric(const dtype& eps, const std::vector<dtype>& weights)
{
    cdtype sum = {"0.0", "0.0"};
    for (size_t i = 0; i < weights.size(); i++)
    {
        dtype phase = dtype(2 * i) * mpfr::const_pi() * eps;
        cdtype exp = {mpfr::cos(phase), mpfr::sin(phase)};
        sum += weights[i] * exp;
    }
    return sum;
}


dtype _f_epsilon_numeric(const dtype& eps, const std::vector<dtype>& weights, const cdtype& ratio)
{
    dtype first = abs(ratio * _fourier_j_numeric(eps - dtype(1) / weights.size(), weights));
    dtype second = abs(_fourier_j_numeric(eps, weights));
    return first - second;
}


void _remove_peak_num(std::vector<double>& abs_fft, const int ind, const dtype& nu,
                      const std::vector<dtype>& weights, const int num_j = 10)
{
    const int n_points = weights.size();
    cdtype ampl = abs_fft[ind % n_points] / _fourier_j_numeric(nu - dtype(ind) / n_points, weights);
    for (int j = ind - num_j + 1; j < ind + num_j; j++)
    {
        cdtype modifier = _fourier_j_numeric(nu - dtype(j) / n_points, weights);
        abs_fft[j % n_points] -= (double) abs(ampl * modifier);
    }
}


auto get_fftw_plan(fftw_complex* orbit, fftw_complex* fft, const int n_points)
{

    std::string filename = "wisdom_" + std::to_string(n_points) + "_outofplace_fft_patient";
    int success_wisdom = fftw_import_wisdom_from_filename(filename.c_str());
    const fftw_plan plan{fftw_plan_dft_1d(n_points, orbit, fft, FFTW_FORWARD, FFTW_PATIENT)};
    if (!success_wisdom)
    {
        fftw_export_wisdom_to_filename(filename.c_str());
    }
    return plan;
}


std::pair<std::vector<dtype>, std::vector<cdtype>> naffnd_numeric(
            const cdtype* signal,
            const std::vector<dtype>& weights,
            const int num_freq,
            const int num_j)
{
    const int n_points = weights.size();
    fftw_complex* orbit{fftw_alloc_complex(n_points)};
    fftw_complex* fft{fftw_alloc_complex(n_points)};
    const fftw_plan plan = get_fftw_plan(orbit, fft, n_points);

    std::vector<cdtype> weighted_signal(n_points);
    for (int i = 0; i < n_points; ++i)
    {
        weighted_signal[i] = signal[i] * weights[i];
        orbit[i][0] = static_cast<double>(weighted_signal[i].real);
        orbit[i][1] = static_cast<double>(weighted_signal[i].imag);
    }

    fftw_execute_dft(plan, orbit, fft);
    std::vector<double> abs_fft(n_points);
    for (int i = 0; i < n_points; ++i)
    {
        abs_fft[i] = std::sqrt(fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]);
//        std::cout << signal[i].real << ", " << signal[i].imag << '\n';
//        std::cout << abs_fft[i] << '\n';
    }

    std::vector<dtype> nu_arr(num_freq);
    std::vector<cdtype> c_arr(num_freq);

    for (int ctr = 0; ctr < num_freq; ++ctr)
    {
        int ind = std::distance(abs_fft.begin(), std::max_element(abs_fft.begin(), abs_fft.end()));
        if (abs_fft[ind - 1 % n_points] > abs_fft[ind + 1 % n_points])
        {
            --ind;
        }
        const cdtype fft_plus1_value = dft(dtype((ind + 1) % n_points) / n_points, weighted_signal);
        const cdtype fft_value = dft(dtype(ind) / n_points, weighted_signal);
        const cdtype ratio = fft_value / fft_plus1_value;
        const dtype nu_init = dtype(ind) / n_points;
        dtype delta = dtype(1) / n_points;
        dtype eps = newton_approx(_f_epsilon_numeric, delta / 2, 1e-28, weights, ratio);
//        std::cout << ind << ", " << fft_plus1_value << ", " << fft_value << ", " << ratio << '\n';
        std::cout << nu_init << ", " << eps << ", " << delta << '\n';

//        std::cout<<"feps\n";
//        for (int i = 0; i < n_points; ++i){
//        std::cout<<_f_epsilon_numeric(dtype(i) / n_points * delta, weights, ratio)<<'\n';}

        nu_arr[ctr] = nu_init + eps;
        c_arr[ctr] = fft_value / _fourier_j_numeric(eps, weights);
        std::cout<< fft_value<<", "<<_fourier_j_numeric(eps, weights)<<'\n';
        _remove_peak_num(abs_fft, ind, nu_arr[ctr], weights, num_j);
        _remove_peak_num(abs_fft, n_points - ind, 1 - nu_arr[ctr], weights, num_j);
    }
    return std::make_pair(nu_arr, c_arr);
}


/// MAIN

//dtype test(dtype x, dtype a)
//{
//    return x * x - a;
//}

int main()
{
    initialize_precision(PRECISION);
//    const dtype PI = (dtype) 3.1415926535897932384626433832795028841971693993751058209749445923078164;
    const dtype k1 = "2.25";
    const dtype k2 = "3.0";
    const dtype k = "1.0";
    const int n_points = 128;
    const OrbitParameters orb_par = {n_points, k1, k2, k};
    dtype q1[n_points] = {"0.5"};
    dtype p1[n_points] = {"0.05"};
    dtype q2[n_points] = {"0.5"};
    dtype p2[n_points] = {"0.05"};

    std_map_4d(q1, p1, q2, p2, orb_par);
    cdtype signal[n_points];
    for (int i = 0; i < n_points; ++i)
    {
        signal[i] = {q1[i] - 0.5, p1[i]};
    }

    std::vector<dtype> weights(n_points);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = mpfr::exp(-dtype(140) * mpfr::sqr(dtype(i) / n_points - 0.5));
    }

    auto result = naffnd_numeric(signal, weights, 2, 10);
    auto nu_arr = result.first;
    auto c_arr = result.second;

    for (int i = 0; i < 2; ++i)
    {
        std::cout << "Nu: " << nu_arr[i] << ", C: " << c_arr[i] << '\n';
    }


//    const dtype a = 4;
//    newton_approx(test, dtype(10), 1e-30, a);
    return 0;
}
