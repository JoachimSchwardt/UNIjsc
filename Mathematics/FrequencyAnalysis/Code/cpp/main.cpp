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
#include "window_functions.h"

#define _ENABLE_MEASURE_TIME
#include "measure_time.h"

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


void std_map_4d(std::vector<dtype>& q1, std::vector<dtype>& p1,
                std::vector<dtype>& q2, std::vector<dtype>& p2, const OrbitParameters& orb_par)
{
    /**
    */
    const dtype pi2 = 2 * mpfr::const_pi();
    dtype coupling;

    for (int i = 1; i < orb_par.n_points; ++i)
    {
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
//    auto t1 = std::chrono::steady_clock::now();
    dtype phase = dtype(-2) * mpfr::const_pi() * nu;
    cdtype kappa = {mpfr::cos(phase), mpfr::sin(phase)};
    cdtype exp = kappa;
    cdtype sum = signal[0];
    for (size_t i = 1; i < signal.size(); i++)
    {
        sum += signal[i] * exp;
        exp = exp * kappa;
    }
//    auto t2 = std::chrono::steady_clock::now();
//    auto duration = std::chrono::duration_cast < std::chrono::milliseconds > (t2 - t1).count();
//    std::cout<<"DFT:" <<duration<<'\n';
    return sum;
}


cdtype _fourier_j_numeric(const dtype& eps, const std::vector<dtype>& weights)
{
//    auto t1 = std::chrono::steady_clock::now();
    dtype phase = dtype(2) * mpfr::const_pi() * eps;
    cdtype kappa = {mpfr::cos(phase), mpfr::sin(phase)};
    cdtype exp = kappa;
    cdtype sum = {weights[0], 0};
    for (size_t i = 1; i < weights.size(); i++)
    {
        sum += weights[i] * exp;
        exp = exp * kappa;
    }
//    auto t2 = std::chrono::steady_clock::now();
//    auto duration = std::chrono::duration_cast < std::chrono::milliseconds > (t2 - t1).count();
//    std::cout << "FJ NUM: " << duration << "ms.\n";
    return sum;
}
cdtype _fourier_j_gauss(const dtype& eps, const int n_points, const dtype& alpha)
{
    const dtype phase = n_points * eps * mpfr::const_pi();
    cdtype sum = {mpfr::cos(phase), mpfr::sin(phase)};
    return sum * mpfr::exp(-mpfr::sqr(phase) / alpha) * mpfr::sqrt(mpfr::const_pi() / alpha) * n_points;
}


dtype _f_epsilon_numeric(const dtype& eps, const std::vector<dtype>& weights, const cdtype& ratio)
{
    dtype first = abs(ratio * _fourier_j_numeric(eps - dtype(1) / weights.size(), weights));
    dtype second = abs(_fourier_j_numeric(eps, weights));
    return first - second;
}
dtype _f_epsilon_gauss(const dtype& eps, const int n_points, const dtype& alpha, const cdtype& ratio)
{
    dtype first = abs(ratio * _fourier_j_gauss(eps - dtype(1) / n_points, n_points, alpha));
    dtype second = abs(_fourier_j_gauss(eps, n_points, alpha));
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
void _remove_peak_gauss(std::vector<double>& abs_fft, const int ind, const dtype& nu,
                        const int n_points, const dtype& alpha, const int num_j = 10)
{
    cdtype ampl = abs_fft[ind % n_points] / _fourier_j_gauss(nu - dtype(ind) / n_points, n_points, alpha);
    for (int j = ind - num_j + 1; j < ind + num_j; j++)
    {
        cdtype modifier = _fourier_j_gauss(nu - dtype(j) / n_points, n_points, alpha);
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


std::vector<cdtype> get_weighted_signal(const std::vector<cdtype>& signal,
                                        const std::vector<dtype>& weights)
{
    const int n_points = weights.size();
    std::vector<cdtype> weighted_signal(n_points);
    for (int i = 0; i < n_points; ++i)
    {
        weighted_signal[i] = signal[i] * weights[i];
    }
    return weighted_signal;
}


void fill_orbit(fftw_complex* orbit, const std::vector<cdtype>& signal)
{
    for (size_t i = 0; i < signal.size(); ++i)
    {
        orbit[i][0] = static_cast<double>(signal[i].real);
        orbit[i][1] = static_cast<double>(signal[i].imag);
    }
}


std::vector<double> get_abs_fft(fftw_complex* fft, const int n_points)
{
    std::vector<double> abs_fft(n_points);
    for (int i = 0; i < n_points; ++i)
    {
        abs_fft[i] = std::sqrt(fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]);
    }
    return abs_fft;
}

template<typename T>
int argmax (const std::vector<T>& arr)
{
    return std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
}


std::pair<std::vector<dtype>, std::vector<cdtype>> naffnd_numeric(
            const std::vector<cdtype>& signal,
            const std::vector<dtype>& weights,
            const int num_freq,
            const int num_j)
{
    const int n_points = weights.size();
    fftw_complex* orbit{fftw_alloc_complex(n_points)};
    fftw_complex* fft{fftw_alloc_complex(n_points)};
    const fftw_plan plan = measure_time(get_fftw_plan, "PLAN FFTW", orbit, fft, n_points);
    std::vector<cdtype> weighted_signal = measure_time(get_weighted_signal, "WEIGHT SIGNAL", signal, weights);
    measure_time(fill_orbit, "FILL ORBIT", orbit, weighted_signal);
    measure_time(fftw_execute_dft, "FFTW", plan, orbit, fft);
    std::vector<double> abs_fft = measure_time(get_abs_fft, "ABS FFT", fft, n_points);

    std::vector<dtype> nu_arr(num_freq);
    std::vector<cdtype> c_arr(num_freq);

    for (int ctr = 0; ctr < num_freq; ++ctr)
    {
//        int ind = std::distance(abs_fft.begin(), std::max_element(abs_fft.begin(), abs_fft.end()));
        int ind = measure_time(argmax<double>, "ARGMAX", abs_fft);
        if (abs_fft[(ind - 1) % n_points] > abs_fft[(ind + 1) % n_points])
        {
            --ind;
        }
        const cdtype fft_plus1_value = dft(dtype((ind + 1) % n_points) / n_points, weighted_signal);
        const cdtype fft_value = dft(dtype(ind) / n_points, weighted_signal);
        const cdtype ratio = fft_value / fft_plus1_value;
        const dtype nu_init = dtype(ind) / n_points;
        dtype delta = dtype(1) / n_points;
        dtype eps = newton_approx(_f_epsilon_numeric, delta / 2, 1e-28, weights, ratio);
        nu_arr[ctr] = nu_init + eps;
        c_arr[ctr] = fft_value / _fourier_j_numeric(eps, weights);
        if (ctr < num_freq - 1)
        {
            _remove_peak_num(abs_fft, ind, nu_arr[ctr], weights, num_j);
            _remove_peak_num(abs_fft, n_points - ind, 1 - nu_arr[ctr], weights, num_j);
        }
    }
    return std::make_pair(nu_arr, c_arr);
}


std::pair<std::vector<dtype>, std::vector<cdtype>> naffnd_gauss(
            const std::vector<cdtype>& signal,
            const std::vector<dtype>& weights,
            const int num_freq,
            const int num_j,
            const dtype& alpha)
{
    const int n_points = weights.size();
    fftw_complex* orbit{fftw_alloc_complex(n_points)};
    fftw_complex* fft{fftw_alloc_complex(n_points)};
    const fftw_plan plan = get_fftw_plan(orbit, fft, n_points);
    std::vector<cdtype> weighted_signal = measure_time(get_weighted_signal, "WEIGHT SIGNAL", signal, weights);
    fill_orbit(orbit, weighted_signal);
    fftw_execute_dft(plan, orbit, fft);
    std::vector<double> abs_fft = get_abs_fft(fft, n_points);

    std::vector<dtype> nu_arr(num_freq);
    std::vector<cdtype> c_arr(num_freq);

    for (int ctr = 0; ctr < num_freq; ++ctr)
    {
        int ind = argmax<double>(abs_fft);
        if (abs_fft[(ind - 1) % n_points] > abs_fft[(ind + 1) % n_points])
        {
            --ind;
        }
        const cdtype fft_plus1_value = dft(dtype((ind + 1) % n_points) / n_points, weighted_signal);
        const cdtype fft_value = dft(dtype(ind) / n_points, weighted_signal);

        const cdtype ratio = fft_value / fft_plus1_value;
        const dtype nu_init = dtype(ind) / n_points;
        dtype delta = dtype(1) / n_points;
//        dtype eps = newton_approx(_f_epsilon_gauss, delta / 2, 1e-28, n_points, alpha, ratio);
        dtype factor = alpha / (dtype(2) * n_points * mpfr::const_pi() * mpfr::const_pi());
        dtype eps = dtype(1) / (2 * n_points) - mpfr::log(abs(ratio)) * factor;
        std::cout<<"WZ: "<<weighted_signal[3]<<'\n';
        std::cout<<"EPS: " <<eps<<", R: "<<ratio<<", FJ: "<<fft_value<<", ind: "<<ind<<'\n';
        nu_arr[ctr] = nu_init + eps;
        c_arr[ctr] = fft_value / _fourier_j_gauss(eps, n_points, alpha);
        if (ctr < num_freq - 1)
        {
            _remove_peak_gauss(abs_fft, ind, nu_arr[ctr], n_points, alpha, num_j);
            _remove_peak_gauss(abs_fft, n_points - ind, 1 - nu_arr[ctr], n_points, alpha, num_j);
        }
    }
    return std::make_pair(nu_arr, c_arr);
}


std::vector<dtype> gaussian_weights(const int n_points, const dtype alpha = 280.0)
{
    std::vector<dtype> weights(n_points);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = mpfr::exp(-alpha * mpfr::sqr(dtype(i) / n_points - 0.5));
    }
    return weights;
}

std::vector<cdtype> construct_signal(const std::vector<dtype>& q, const std::vector<dtype>& p)
{
    const int n_points = q.size();
    std::vector<cdtype> signal(n_points);
    for (int i = 0; i < n_points; ++i)
    {
        signal[i] = {q[i] - 0.5, p[i]};
    }
    return signal;
}


int main()
{
    initialize_precision(PRECISION);
    const dtype k1 = "2.25";
    const dtype k2 = "3.0";
    const dtype k = "1.0";
    const int n_points = 4096 * 2;
    const OrbitParameters orb_par = {n_points, k1, k2, k};
    std::vector<dtype> q1(n_points);
    std::vector<dtype> p1(n_points);
    std::vector<dtype> q2(n_points);
    std::vector<dtype> p2(n_points);
    q1[0] = "0.5";
    p1[0] = "0.05";
    q2[0] = "0.5";
    p2[0] = "0.05";
    const dtype alpha = 280.0;

    measure_time(std_map_4d, "ORBIT", q1, p1, q2, p2, orb_par);
    std_map_4d(q1, p1, q2, p2, orb_par);

    auto weights = measure_time(gaussian_weights, "GAUSS WEIGHTS", n_points, alpha);
//    std::cout << weights[5000]<<'\n';
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = (dtype) ((double) weights[i]);
    }
//    std::cout << weights[5000]<<'\n';
//    std::cout << weights[5000] + 1e-27<<'\n';
    auto signal = measure_time(construct_signal, "SIGNAL", q1, p1);

//    auto result = measure_time(naffnd_numeric, "NAFFND", signal, weights, 2, 10);
    auto result = measure_time(naffnd_gauss, "NAFFND", signal, weights, 2, 10, alpha);
    auto nu_arr = result.first;
    auto c_arr = result.second;

    for (int i = 0; i < 2; ++i)
    {
        std::cout << "Nu: " << nu_arr[i] << ", C: " << c_arr[i] << '\n';
    }
//    auto chebwin = cheby_win(600, 70);
//    for (auto elem : chebwin){
//    std::cout<<elem<<'\n';}
    return 0;
}
