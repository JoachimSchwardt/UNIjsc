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

#define OMP_NUM_THREADS 16

#pragma GCC target ("avx2")

using dtype = double;
constexpr int SIMD_SIZE = 4;
constexpr dtype PRECISION = 16;
//using dtype = float;
//constexpr int SIMD_SIZE = 8;
//constexpr dtype PRECISION = 7;

/// fixed parameters
constexpr int N = { 5000 };           // number of particles
constexpr int tau = { 30 };           // period length
constexpr int num_tau = { 5 };        // number of periods
constexpr dtype delta_t = { 1.0 / 100.0 };    // time discretization
constexpr int num_repeat = {1};      // number of repititions
constexpr dtype D = {8e-3};          // diffusion constant
constexpr dtype v_0 = {-0.2};        // potential strength
constexpr dtype L = {1.5};           // spacial period length of the potential
constexpr dtype x0 = {0.0};          // initial position of the particles
constexpr dtype alpha = {0.0};       // skew parameter

/// bounding box of the parameter space
constexpr dtype a_min{0.2};
constexpr dtype a_max{0.21};
constexpr dtype theta_min{0.8};
constexpr dtype theta_max{2.0};
constexpr int num_a = {16};
constexpr int num_theta = {1};

constexpr dtype pi = {3.14159265358979323846};


inline dtype get_xpot_prime(const dtype x, const dtype a) noexcept {
    dtype factor = 2 * pi / L;
    dtype arg = factor * x;
    dtype sin = std::sin(arg);
    dtype cos = std::cos(arg);
    return (factor * v_0 * (2 * a * (cos * cos - sin * sin) - sin));
}

dtype simulate(const dtype a, const dtype theta) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal{0.0, 1.0};

    bool State;
    dtype xi;
    std::array<dtype, N> x_vals = {0};
    for (dtype tval = 0; tval < num_tau * tau; tval += delta_t) {
        State = std::fmod(tval, static_cast<dtype>(tau)) > tau / (1 + theta);
//        #pragma omp simd
#pragma GCC ivdep
        for (int i = 0; i < N; ++i) {
            xi = normal(gen);
            x_vals[i] += std::sqrt(2 * D * delta_t) * xi;
            if (State) {
                x_vals[i] -= get_xpot_prime(x_vals[i], a) * delta_t;
            }
        }

//        for (int i = 0; i < N; i += SIMD_SIZE) {
//            #pragma omp simd
//            for (int j = 0; j < SIMD_SIZE; ++j) {
//                xi = normal(gen);
//                x_vals[i + j] += std::sqrt(2 * D * delta_t) * xi;
//                if (State) {
//                    x_vals[i + j] -= get_xpot_prime(x_vals[i + j], a) * delta_t;
//                }
//            }
//        }

//        for (double& xval : x_vals) {
//            xi = normal(gen);
//            xval += std::sqrt(2 * D * delta_t) * xi;
//            if (State) {
//                xval -= get_xpot_prime(xval, a) * delta_t;
//            }
//        }
    }
    return std::accumulate(std::begin(x_vals), std::end(x_vals), 0.0) / std::size(x_vals);
}


void _parSpace() {
    std::string filename = "results.txt";
    std::ofstream outfile {};
    outfile.open(filename);
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
    */

    auto t1 = std::chrono::steady_clock::now();
    _parSpace();
    auto t2 = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Total runtime: " << duration << " ms.\n";
    return 0;
}
