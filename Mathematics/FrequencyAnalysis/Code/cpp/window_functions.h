#ifndef WINDOW_FUNCTIONS_H_INCLUDED
#define WINDOW_FUNCTIONS_H_INCLUDED

#include <vector>
#include "mpfr.h"
#include "fftw3.h"

using dtype = mpfr::mpreal;
using vec_t = std::vector<dtype>;

//vec_t chebwin(const int n, const dtype at)
// 
//{
//    if (n == 1)
//    {
//        return vec_t("1");
//    }
//    // compute the parameter beta
//    dtype beta = mpfr::cosh(mpfr::acosh(mpfr::pow(at / 20, 10)) / (n - 1));
//    dtype factor = mpfr::const_pi() / n;
//    vec_t dft_coeff(n);
//    for (size_t i = 0; i < n; ++i)
//    {
//        dtype cos_k = beta * mpfr::cos(factor * dtype(i) / n);
//        dft_coeff[i] = cheb(n - 1, beta * cos_k);
//    }
//     
//    vec_t window(n); // the window vector
//    // Appropriate IDFT and filling up depending on even/odd n
//    fftw_complex* out{fftw_alloc_complex(n)};
//    double* in{fftw_alloc_real(n)};
//    for (size_t i = 0; i < n; ++i)
//    {
//        in[i] = dft_coeff[i];
//    }
//    const fftw_plan plan{fftw_plan_dft_r2c_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE)};
//    fftw_execute_dft(plan, in, out);
//
//    if M % 2:
//        w = np.real(sp_fft.fft(p))
//        n = (M + 1) / 2
//        w = w[:n]
//        w = np.concatenate((w[n - 1:0:-1], w))
//    else:
//        p = p * np.exp(1.j * np.pi / M * np.r_[0:M])
//        w = np.real(sp_fft.fft(p))
//        n = M / 2 + 1
//        w = np.concatenate((w[n - 1:0:-1], w[1:n]))
//    if (n % 2)
//    {
//        window = ifft_real(to_cvec(p));
//        int half_length = (n + 1) / 2;
//        window = window.left(half_length) / w(0);
//        window = concat(reverse(window), w.right(n - half_length));
//    }
//    else
//    {
//        window = ifft_real(to_cvec(elem_mult(p, cos_k), elem_mult(p, -sin(k))));
//        int half_length = n / 2 + 1;
//        window = window.left(half_length) / w(1);
//        window = concat(reverse(window), w.right(n - half_length));
//    }
//    return window;
//}
//
//dtype cheb(const int n, const dtype x)
//{
//     
//    if (x < 1.0 && x > -1.0)
//    {
//        return mpfr::cos(dtype(n) * mpfr::acos(x));
//    }
//    else if (x <= -1)
//    {
//        const int sign = (n % 2) ? 1 : -1;
//        return sign * mpfr::cosh(dtype(n) * mpfr::acosh(-x));
//    }
//    return mpfr::cosh(dtype(n) * mpfr::acosh(x));
//}

//dtype cheby_poly(const int n, const dtype x)
//{
//    dtype res;
//    if (mpfr::abs(x) <= 1)
//    {
//        res = mpfr::cos(dtype(n) * mpfr::acos(x));
//    }
//    else
//    {
//        res = mpfr::cosh(dtype(n) * mpfr::acosh(x));
//    }
//    return res;
//}
//
//vec_t cheby_win(const int size, const dtype& atten)
//{
//    auto window = vec_t(size);
//    size_t nn, i;
//    const dtype PI = mpfr::const_pi();
//    dtype M, n, sum = 0;
//    dtype beta = mpfr::pow(dtype(10), atten / dtype(20));
//    dtype x0 = mpfr::cosh((dtype(1) / (size - 1)) * mpfr::acosh(beta));
//    M = (size % 2) ? (size - 1) / dtype(2) : (size / dtype(2));
//    for (nn = 0; nn < static_cast<size_t>(size / 2 + 1); nn++)
//    {
//        n = nn - M;
//        sum = 0;
//        for (i = 1; i <= M; i++)
//        {
//            dtype factor = PI * dtype(i) / size;
//            sum += cheby_poly(size - 1, x0 * mpfr::cos(factor)) * mpfr::cos(dtype(2 * n) * factor);
//        }
//        window[nn] = beta + dtype(2) * sum;
//        window[size - nn - 1] = window[nn];
//    }
//    return window;
//}

#endif // WINDOW_FUNCTIONS_H_INCLUDED
