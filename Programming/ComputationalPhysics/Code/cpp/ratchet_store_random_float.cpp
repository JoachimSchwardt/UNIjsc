/** g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-S -masm=intel main.cpp -lm -lmvec -lfftw3  (-S and -masm=intel are for the disassembly output) */
#include<iostream>
#include<cmath>
#include<algorithm>
#include<random>
#include<string>
#include<fstream>
#include<sstream>
#include<chrono>
#include<vector>
#include<immintrin.h>

#define _USE_MATH_DEFINES

#define OMP_NUM_THREADS 16
#pragma GCC target ("avx2")


using dtype = float;
constexpr int SIMD_SIZE = 8;                   // number of floats fitting into avx2 register
constexpr dtype PRECISION = 7;                 // number of digits of precision (float)


inline __m256 _mm256_sin_ps(__m256 __A) noexcept {
    __m256 __B;
    #pragma omp simd
    for (int i = 0; i < SIMD_SIZE; ++i) {
        __B[i] = std::sin(__A[i]);
    }
    return __B;
}


inline __m256 _mm256_cos_ps(__m256 __A) noexcept {
    __m256 __B;
    #pragma omp simd
    for (int i = 0; i < SIMD_SIZE; ++i) {
        __B[i] = std::cos(__A[i]);
    }
    return __B;
}


inline __m256 get_xpot_prime(const __m256 __x, const dtype a,
                              const dtype L, const dtype v_0) noexcept {
    dtype factor = 2.0f * M_PI / L;
    __m256 __arg = _mm256_mul_ps(_mm256_set1_ps(factor ), __x);
    __m256 __sin = _mm256_sin_ps(__arg);
    __m256 __cos = _mm256_cos_ps(__arg);

    __m256 __res = _mm256_set1_ps(factor * v_0);               // result = factor * v_0 * ...
    __m256 __sum1 = _mm256_set1_ps(2.0f * a);                  // sum1 = 2*a * ...
    __m256 __cos2 = _mm256_sub_ps(_mm256_mul_ps(__cos, __cos), _mm256_mul_ps(__sin, __sin));
    __sum1 = _mm256_mul_ps(__sum1, __cos2);                    // sum1 = 2*a * (cos*cos - sin*sin)
    __res = _mm256_mul_ps(__res, _mm256_sub_ps(__sum1, __sin));
    return __res;      // factor * v_0 * (2.0 * a * (cos * cos - sin * sin) - sin)
}


struct ResultType {
    dtype a;
    dtype theta;
    dtype xmean;
};


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


dtype simulate(const dtype a, const dtype theta, const RatchetParameters& rat_par,
               const auto& r_tensor) noexcept {
    bool State;
    const __m256 __diffusion_par = {_mm256_set1_ps(std::sqrt(2 * rat_par.D * rat_par.delta_t))};
    __m256 __temp;
    std::vector<dtype> x_vals(rat_par.N);     // particle positions
    std::vector<dtype> r_vals(rat_par.N);     // random values

    dtype tval = 0;
    for (size_t i_t = 0; i_t < r_tensor.size(); ++i_t) {
        State = std::fmod(tval, static_cast<dtype>(rat_par.tau)) > rat_par.tau / (1 + theta);
        r_vals = r_tensor[i_t];
#pragma GCC ivdep
        for (int i = 0; i < rat_par.N; i += SIMD_SIZE) {
            __temp = _mm256_mul_ps(_mm256_loadu_ps(&r_vals[i]), __diffusion_par);

            #pragma omp simd
            for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                x_vals[i + lane] += __temp[lane];
            }

            if (State) {
                #pragma omp simd
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    __temp[lane] = x_vals[i + lane];
                }
                __temp = _mm256_mul_ps(_mm256_set1_ps(-rat_par.delta_t),
                                       get_xpot_prime(__temp, a, rat_par.L, rat_par.v_0));
                #pragma omp simd
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    x_vals[i + lane] += __temp[lane];
                }
            }
        }

        tval += rat_par.delta_t;    // increment the time
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

    // determine the boundix box of the parameter space
    const dtype delta_a = {(a_box.max_val - a_box.min_val) / a_box.num};
    const dtype delta_theta = {(theta_box.max_val - theta_box.min_val) / theta_box.num};
    std::vector<dtype> a_vals(a_box.num);
    std::vector<dtype> theta_vals(theta_box.num);
    dtype a = a_box.min_val;
    dtype theta = theta_box.min_val;

    // create the bounding box of the parameter space
    for (int i = 0; i < a_box.num; ++i) {
        a_vals[i] = a;
        a += delta_a;
    }
    for (int i = 0; i < theta_box.num; ++i) {
        theta_vals[i] = theta;
        theta += delta_theta;
    }

    // generate random numbers for one cycle and reuse for all parameter tuples
    const int num_cycles = rat_par.num_tau * rat_par.tau * static_cast<int>(1 / rat_par.delta_t);
    std::vector<std::vector<dtype>> r_tensor(num_cycles, std::vector<dtype>(rat_par.N));
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution normal{0.0f, 1.0f};

    // temporary arrays to store the data (we want to write in serial, not parallel)
    std::vector<ResultType> results(a_box.num * theta_box.num);

    // main event loop
    for (int i = 0; i < rat_par.num_repeat; ++i) {
        // generate new white noise
        for (int i = 0; i < num_cycles; ++i) {
            for (int p = 0; p < rat_par.N; ++p) {
                r_tensor[i][p] = normal(gen);
            }
        }

        #pragma omp parallel for
        for (int i_a = 0; i_a < a_box.num; ++i_a) {
            dtype a = a_vals[i_a];
            for (int i_theta = 0; i_theta < theta_box.num; ++i_theta ) {
                dtype theta = theta_vals[i_theta];
                dtype xmean = simulate(a, theta, rat_par, r_tensor);

                results[i_a * theta_box.num + i_theta] = ResultType{a, theta, xmean};
            }
        }

        // write the results to file
        for (int i_a = 0; i_a < a_box.num; ++i_a) {
            for (int i_theta = 0; i_theta < theta_box.num; ++i_theta ) {
                ResultType res = results[i_a * theta_box.num + i_theta];
                outfile << res.a << ',' << res.theta << ',' << res.xmean << '\n';
            }
        }
    }
}


int main() {
    /**
    N,     tau,  num_tau,  delta_t,  D,      v_0,   L,    x_0,  alpha,  num_repeat
    5000,  30,   5,        0.01,     0.008,  -0.2,  1.5,  0,    0,      1
        :: 2.7 sec (for 16x16 points) (about 0.6 sec for white noise, leaving 2.1 sec)
    4096
        :: 57.3 sec (for 96x100 points) (theta=0.8 ... 2.0, a=0.0 ... 1.0)
        :: 46.7 sec (for 96x100 points) (theta=0.0 ... 2.0, a=0.0 ... 1.0)
    */

    /// fixed parameters
    const int N = { 4096 };                    // number of particles
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
    const dtype a_min{0.0};
    const dtype a_max{1.0};
    const dtype theta_min{0.0};
    const dtype theta_max{2.0};
    const int num_a = {96};
    const int num_theta = {100};

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
