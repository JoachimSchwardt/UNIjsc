/** g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-S -masm=intel main.cpp -lm -lmvec -lfftw3  (-S and -masm=intel are for the disassembly output) */
#include<iostream>
#include<cmath>
#include<algorithm>
#include<random>
#include<string>
#include<fstream>
#include<chrono>
#include<vector>
#include<immintrin.h>

#define _USE_MATH_DEFINES

#define OMP_NUM_THREADS 16
#pragma GCC target ("avx2")


using dtype = double;
constexpr int SIMD_SIZE = 4;                   // number of doubles fitting into avx2 register
constexpr dtype PRECISION = 16;                // number of digits of precision (double)


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


inline __m256d get_xpot_prime(const __m256d __x, const dtype a,
                              const dtype L, const dtype v_0) noexcept {
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


struct Bbox {
    dtype min_val;
    dtype max_val;
    int num;
};


struct RatchetParameters {
    int N;
    int tau;
    int num_tau;
    dtype delta_t;
    int num_repeat;
    dtype D;
    dtype v_0;
    dtype L;
    dtype x0;
    dtype alpha;
};


dtype simulate(const dtype a, const dtype theta, const RatchetParameters& rat_par) noexcept {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal{0.0, 1.0};
//    std::uniform_real_distribution<> normal{-1.0, 1.0};

    bool State;
    const __m256d __diffusion_par = {_mm256_set1_pd(std::sqrt(2 * rat_par.D * rat_par.delta_t))};
    __m256d __xi = {0};
    __m256d __temp;
    std::vector<dtype> x_vals(rat_par.N);
    for (dtype tval = 0; tval < rat_par.num_tau * rat_par.tau; tval += rat_par.delta_t) {
        State = std::fmod(tval, static_cast<dtype>(rat_par.tau)) > rat_par.tau / (1 + theta);

#pragma GCC ivdep
        for (int i = 0; i < rat_par.N; i += SIMD_SIZE) {
            #pragma omp simd
            for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                __xi[lane] = normal(gen);
            }
//            __xi = _mm256_set1_pd(normal(gen));
//            __xi[0] = normal(gen);
//            __xi[1] = normal(gen);
//            __xi[2] = normal(gen);
//            __xi[3] = normal(gen);
            __temp = _mm256_mul_pd(__xi, __diffusion_par);

            #pragma omp simd
            for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                x_vals[i + lane] += __temp[lane];
            }

            if (State) {
                #pragma omp simd
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    __temp[lane] = x_vals[i + lane];
                }
                __temp = _mm256_mul_pd(_mm256_set1_pd(-rat_par.delta_t),
                                       get_xpot_prime(__temp, a, rat_par.L, rat_par.v_0));
                #pragma omp simd
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    x_vals[i + lane] += __temp[lane];
                }
            }
        }
    }
    return std::accumulate(std::begin(x_vals), std::end(x_vals), 0.0) / std::size(x_vals);
}


void _parSpace(const Bbox& a_box, const Bbox& theta_box, const RatchetParameters& rat_par) {
    std::string filename = "results.txt";
    std::ofstream outfile {};
    outfile.open(filename);
    outfile << "N,tau,num_tau,delta_t,D,v_0,L,x_0,alpha,num_repeat\n";
    outfile << rat_par.N << ',' << rat_par.tau << ',' << rat_par.num_tau << ','
            << rat_par.delta_t << ',' << rat_par.D << ','
            << rat_par.v_0 << ',' << rat_par.L << ',' << rat_par.x0 << ','
            << rat_par.alpha << ',' << rat_par.num_repeat << '\n';
    outfile << "a,theta,xmean\n";
    outfile.precision(PRECISION);
    outfile << std::fixed;

    const dtype delta_a = {(a_box.max_val - a_box.min_val) / a_box.num};
    const dtype delta_theta = {(theta_box.max_val - theta_box.min_val) / theta_box.num};
//    std::array<dtype, a_box.num> a_vals = {0};
//    std::array<dtype, theta_box.num> theta_vals = {0};
    std::vector<dtype> a_vals(a_box.num);
    std::vector<dtype> theta_vals(theta_box.num);
    dtype a = a_box.min_val;
    dtype theta = theta_box.min_val;

    for (int i = 0; i < a_box.num; ++i) {
        a_vals[i] = a;
        a += delta_a;
    }
    for (int i = 0; i < theta_box.num; ++i) {
        theta_vals[i] = theta;
        theta += delta_theta;
    }

    #pragma omp parallel for
    for (int i_a = 0; i_a < a_box.num; ++i_a) {
        dtype a = a_vals[i_a];
        for (int i_theta = 0; i_theta < theta_box.num; ++i_theta ) {
            dtype theta = theta_vals[i_theta];
            for (int i = 0; i < rat_par.num_repeat; ++i) {
                dtype xmean = simulate(a, theta, rat_par);
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

    /// fixed parameters
    const int N = { 5000 };                    // number of particles
    const int tau = { 30 };                    // period length
    const int num_tau = { 5 };                 // number of periods
    const dtype delta_t = { 1.0 / 100.0 };     // time discretization
    const int num_repeat = {1};               // number of repititions
    const dtype D = {8e-3};                    // diffusion constant
    const dtype v_0 = {-0.2};                  // potential strength
    const dtype L = {1.5};                     // spacial period length of the potential
    const dtype x0 = {0.0};                    // initial position of the particles
    const dtype alpha = {0.0};                 // skew parameter

    /// bounding box of the parameter space
    const dtype a_min{0.2};
    const dtype a_max{0.21};
    const dtype theta_min{0.8};
    const dtype theta_max{2.0};
    const int num_a = {16};
    const int num_theta = {1};

    const Bbox a_box = {a_min, a_max, num_a};
    const Bbox theta_box = {theta_min, theta_max, num_theta};
    const RatchetParameters rat_par = {N, tau, num_tau, delta_t, num_repeat, D, v_0, L, x0, alpha};

    auto t1 = std::chrono::steady_clock::now();
    _parSpace(a_box, theta_box, rat_par);
    auto t2 = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Total runtime: " << duration << " ms.\n";
    return 0;
}

