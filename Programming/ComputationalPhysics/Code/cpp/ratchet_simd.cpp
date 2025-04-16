/** g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-S -masm=intel main.cpp -lm -lmvec -lfftw3  (-S and -masm=intel are for the disassembly output) */
#include<iostream>
#include<cmath>
#include<algorithm>
#include<random>
#include<string>
#include<fstream>
#include<chrono>
#include<array>
#include<immintrin.h>

#define OMP_NUM_THREADS 16

#pragma GCC target ("avx2")

using dtype = double;
constexpr int SIMD_SIZE = 4;                   // number of doubles fitting into avx2 register
constexpr dtype PRECISION = 16;                // number of digits of precision (double)

/// fixed parameters
constexpr int N = { 5000 };                    // number of particles
constexpr int tau = { 30 };                    // period length
constexpr int num_tau = { 5 };                 // number of periods
constexpr dtype delta_t = { 1.0 / 100.0 };     // time discretization
constexpr int num_repeat = {1};               // number of repititions
constexpr dtype D = {8e-3};                    // diffusion constant
constexpr dtype v_0 = {-0.2};                  // potential strength
constexpr dtype L = {1.5};                     // spacial period length of the potential
constexpr dtype x0 = {0.0};                    // initial position of the particles
constexpr dtype alpha = {0.0};                 // skew parameter

/// bounding box of the parameter space
constexpr dtype a_min{0.2};
constexpr dtype a_max{0.21};
constexpr dtype theta_min{0.8};
constexpr dtype theta_max{2.0};
constexpr int num_a = {1};
constexpr int num_theta = {1};

#define _USE_MATH_DEFINES
//constexpr dtype M_PI = {3.14159265358979323846};

inline __m256d _mm256_sin_pd(__m256d __A) noexcept {
    __m256d __B;
    #pragma omp simd
    for (int i = 0; i < SIMD_SIZE; ++i) {
        __B[i] = std::sin(__A[i]);
    }
    return __B;
}

inline __m256d _mm256_cos_pd(__m256d __A) noexcept {
    __m256d __B;
    #pragma omp simd
    for (int i = 0; i < SIMD_SIZE; ++i) {
        __B[i] = std::cos(__A[i]);
    }
    return __B;
}

inline __m256d get_xpot_prime(const __m256d __x, const dtype a) noexcept {
    dtype factor = 2.0 * M_PI / L;
    __m256d __arg = _mm256_mul_pd(_mm256_set1_pd(factor ), __x);
    __m256d __sin = _mm256_sin_pd(__arg);
    __m256d __cos = _mm256_cos_pd(__arg);

    __m256d __res = _mm256_set1_pd(factor * v_0);              // result = factor * v_0 * ...
    __m256d __sum1 = _mm256_set1_pd(2.0 * a);                  // sum1 = 2*a * ...
    __m256d __cos2 = _mm256_sub_pd(_mm256_mul_pd(__cos, __cos), _mm256_mul_pd(__sin, __sin));
    __sum1 = _mm256_mul_pd(__sum1, __cos2);                    // sum1 = 2*a * (cos*cos - sin*sin)
    __res = _mm256_mul_pd(__res, _mm256_sub_pd(__sum1, __sin));
    return __res;
//    return (factor * v_0 * (2.0 * a * (cos * cos - sin * sin) - sin));
}


dtype simulate(const dtype a, const dtype theta) noexcept {
    std::random_device rd{};
    std::mt19937 gen{rd()};
//    std::normal_distribution<> normal{0.0, 1.0};
    std::uniform_real_distribution<> normal{-1.0, 1.0};

    bool State;
    __m256d __xi = {0};
    __m256d __temp;
    std::array<dtype, N> x_vals = {0};
    for (dtype tval = 0; tval < num_tau * tau; tval += delta_t) {
        State = std::fmod(tval, static_cast<dtype>(tau)) > tau / (1 + theta);

#pragma GCC ivdep
        for (int i = 0; i < N; i += SIMD_SIZE) {
//            #pragma omp simd
//            for (int lane = 0; lane < SIMD_SIZE; ++lane) {
//                __xi[lane] = normal(gen);
//            }
//            __xi = _mm256_set1_pd(normal(gen));
            __xi[0] = normal(gen);
//            __xi[1] = normal(gen);
//            __xi[2] = normal(gen);
//            __xi[3] = normal(gen);
            __temp = _mm256_mul_pd(__xi, _mm256_set1_pd(std::sqrt(2 * D * delta_t)));

            #pragma omp simd
            for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                x_vals[i + lane] += __temp[lane];
            }

            if (State) {
                #pragma omp simd
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    __temp[lane] = x_vals[i + lane];
                }
                __temp = _mm256_mul_pd(_mm256_set1_pd(-delta_t), get_xpot_prime(__temp, a));
                #pragma omp simd
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    x_vals[i + lane] += __temp[lane];
                }
            }
        }
    }
    return std::accumulate(std::begin(x_vals), std::end(x_vals), 0.0) / std::size(x_vals);
}


void _parSpace() {
    std::string filename = "results.txt";
    std::ofstream outfile {};
    outfile.open(filename);
    outfile << "N,tau,num_tau,delta_t,D,v_0,L,x_0,alpha,num_repeat\n";
    outfile << N << ',' << tau << ',' << num_tau << ',' << delta_t << ',' << D << ','
            << v_0 << ',' << L << ',' << x0 << ',' << alpha << ',' << num_repeat << '\n';
    outfile << "a,theta,xmean\n";
    outfile.precision(PRECISION);
    outfile << std::fixed;

    constexpr dtype delta_a = {(a_max - a_min) / num_a};
    constexpr dtype delta_theta = {(theta_max - theta_min) / num_theta};
    std::array<dtype, num_a> a_vals = {0};
    std::array<dtype, num_theta> theta_vals = {0};
    dtype a = a_min;
    dtype theta = theta_min;

    for (int i = 0; i < num_a; ++i) {
        a_vals[i] = a;
        a += delta_a;
    }
    for (int i = 0; i < num_theta; ++i) {
        theta_vals[i] = theta;
        theta += delta_theta;
    }

    #pragma omp parallel for
    for (int i_a = 0; i_a < num_a; ++i_a) {
        dtype a = a_vals[i_a];
        for (int i_theta = 0; i_theta < num_theta; ++i_theta ) {
            dtype theta = theta_vals[i_theta];
            for (int i = 0; i < num_repeat; ++i) {
                dtype xmean = simulate(a, theta);
                outfile << a << ',' << theta << ',' << xmean << '\n';
            }
        }
    }
}


int main() {
    /**
    N,     tau,  num_tau,  delta_t,  D,      v_0,   L,    x_0,  alpha,  num_repeat
    5000,  30,   5,        0.01,     0.008,  -0.2,  1.5,  0,    0,      10
        :: 15.8 sec (for 16x1 points)
    */

    auto t1 = std::chrono::steady_clock::now();
    _parSpace();
    auto t2 = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Total runtime: " << duration << " ms.\n";
    return 0;
}
