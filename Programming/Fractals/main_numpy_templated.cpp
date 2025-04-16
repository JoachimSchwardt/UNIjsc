/**
Fast Mandelbrot and Julia set computation using OpenMP and SIMD.

compile using ::
    g++ -Wall -fexpensive-optimizations -O3 -std=c++2a -march=native -mavx2 -ffast-math -fopenmp
-masm=intel main.cpp -lm -lmvec

Make sure to adjust the number of available threads in '#define OMP_NUM_THREADS'.
The length of the 'a' loop (i.e. 'num_a') should be a multiple of OMP_NUM_THREADS to make optimal
use of the parallelization.

WARNING:
This code requires AVX2 intrinsics, check if they are available on your hadware first.
While the simulation should not take more than ~15 minutes, note that the CPU will run on a true
100% during that time. Also, due to the extensive use of SIMD intrinsics (AVX2), it will generate
even more heat than other stress tests.
This is probably not an issue (the CPU will hopefully just slow down to avoid overheating),
but be mindful with the core temperature nonetheless.

@author: Joachim Schwardt
*/

#include<iostream>
#include<chrono>
#include<vector>
#include<immintrin.h>

/// https://stackoverflow.com/questions/6321839/how-to-disable-warnings-for-particular-include-files
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <pybind11/pybind11.h>      // this library is very evil; need to ignore a lot of errors
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop

#define OMP_NUM_THREADS 16
#pragma GCC target ("avx2")


namespace py = pybind11;


using dtype = double;
constexpr int SIMD_SIZE = 4;                   // number of 'dtype's fitting into avx2 register
using ctr_array_t = std::vector<std::vector<long>>;
using col_array_t = std::vector<uint32_t>;
using rgb_array_t = std::vector<col_array_t>;


struct Extent {
    Extent(dtype xmin_, dtype xmax_, dtype ymin_, dtype ymax_)
        : xmin{xmin_}, xmax{xmax_}, ymin{ymin_}, ymax{ymax_}
    { }
    Extent() : Extent(-2.0, 1.0, -1.2, 1.2) { }
    dtype xmin;
    dtype xmax;
    dtype ymin;
    dtype ymax;
};


struct Shape {
    Shape(int width_, int height_) : width{width_}, height{height_} { }
    Shape() : Shape(2560, 1440) { }
    int width;
    int height;
};


template <typename T>
std::vector<T> linspace(T xmin, T xmax, int num) {
    T dx = (xmax - xmin) / (num - 1);
    std::vector<T> vals(num);
    for (int i = 0; i < num - 1; ++i) {
        vals[i] = xmin;
        xmin += dx;
    }
    vals[num - 1] = xmax;
    return vals;
}


__m256i get_counter_mandelbrot(const __m256d cxval, const __m256d cyval,
                               const std::vector<dtype>& z_0, const int n_iter) {
    /** Mandelbrot iteration: z_{n+1} = z_n^2 + c
        Translates to:
            x_{n+1} + iy_{n+1} = (x_n + iy_n)^2 + (cx + icy) =
                               = (x_n^2 - y_n^2 + cx) + i(2x_n * y_n + cy)
    */
    const __m256i __ones = _mm256_set1_epi64x(1);
    const __m256i __n_iter = _mm256_set1_epi64x(n_iter);
    const __m256d __abort_val = _mm256_set1_pd(4.0);
    __m256i __ctr = _mm256_set1_epi64x(0);
    __m256d zxval = _mm256_set1_pd(z_0[0]);
    __m256d zyval = _mm256_set1_pd(z_0[1]);
    __m256d temp_zxval = zxval;
    __m256i mask = __ones;

    do {
        // avoid recomputing the squares in the abort criterion
        __m256d zxval_sqr = _mm256_mul_pd(zxval, zxval);     // zx**2
        __m256d zyval_sqr = _mm256_mul_pd(zyval, zyval);     // zy**2

        // Mandelbrot iteration
        temp_zxval = _mm256_add_pd(_mm256_sub_pd(zxval_sqr, zyval_sqr), cxval);   // zx**2 - zy**2 + cx
        zyval = _mm256_fmadd_pd(2 * zxval, zyval, cyval);                         // 2*zx*zy + cy
        zxval = temp_zxval;

        /* check for the abort criterion (modulus > 2, or equivalently |z|^2 > 4)
           Note that "true" in this mask is represented by "-1", which is due to all 32 bits being "1".
            --> example: mask = [11..11|00..00|11..11|11..11|11..11|00..00|11..11|11..11]
            --> BUT:  "value" = [  -1  |   0  |  -1  |  -1  |  -1  |   0  |  -1  |  -1  ]
        */
        __m256d sqr_sum = _mm256_add_pd(zxval_sqr, zyval_sqr);
        mask = _mm256_and_si256(
                   _mm256_castpd_si256(_mm256_cmp_pd(sqr_sum, __abort_val, _CMP_LT_OQ)),
                   _mm256_cmpgt_epi64(__n_iter, __ctr)
               );

        __ctr = _mm256_sub_epi64(__ctr, mask);     // the mask is [-1, ..., -1] --> subtract to add 1!
    } while (_mm256_movemask_pd(_mm256_castsi256_pd(mask)) > 0);

    return __ctr;
}


__m256i get_counter_julia(const __m256d zxval_0, const __m256d zyval_0,
                          const std::vector<dtype>& c_0, const int n_iter) {
    /** Mandelbrot iteration: z_{n+1} = z_n^2 + c
        Translates to:
            x_{n+1} + iy_{n+1} = (x_n + iy_n)^2 + (cx + icy) =
                               = (x_n^2 - y_n^2 + cx) + i(2x_n * y_n + cy)
        Julia set :: zval corresponds to the z_0's, and c_0 cooresponds to the cval!
                     (reverse of Mandelbrot)
    */
    const __m256i __ones = _mm256_set1_epi64x(1);
    const __m256i __n_iter = _mm256_set1_epi64x(n_iter);
    const __m256d __abort_val = _mm256_set1_pd(4.0);
    __m256i __ctr = _mm256_set1_epi64x(0);
    __m256d zxval = zxval_0;
    __m256d zyval = zyval_0;
    const __m256d cxval = _mm256_set1_pd(c_0[0]);
    const __m256d cyval = _mm256_set1_pd(c_0[1]);
    __m256d temp_zxval = zxval;
    __m256i mask = __ones;

    do {
        // avoid recomputing the squares in the abort criterion
        __m256d zxval_sqr = _mm256_mul_pd(zxval, zxval);     // zx**2
        __m256d zyval_sqr = _mm256_mul_pd(zyval, zyval);     // zy**2

        // Mandelbrot iteration
        temp_zxval = _mm256_add_pd(_mm256_sub_pd(zxval_sqr, zyval_sqr), cxval);   // zx**2 - zy**2 + cx
        zyval = _mm256_fmadd_pd(2 * zxval, zyval, cyval);                         // 2*zx*zy + cy
        zxval = temp_zxval;

        /* check for the abort criterion (modulus > 2, or equivalently |z|^2 > 4)
           Note that "true" in this mask is represented by "-1", which is due to all 64 bits being "1".
            --> example: mask = [11..11|00..00|11..11|11..11]
            --> BUT:  "value" = [  -1  |   0  |  -1  |  -1  ]
        */
        __m256d sqr_sum = _mm256_add_pd(zxval_sqr, zyval_sqr);
        mask = _mm256_and_si256(
                   _mm256_castpd_si256(_mm256_cmp_pd(sqr_sum, __abort_val, _CMP_LT_OQ)),
                   _mm256_cmpgt_epi64(__n_iter, __ctr)
               );

        __ctr = _mm256_sub_epi64(__ctr, mask);     // the mask is [-1, ..., -1] --> subtract to add 1!
    } while (_mm256_movemask_pd(_mm256_castsi256_pd(mask)) > 0);

    return __ctr;
}


template <typename F>
auto get_fractal(const Shape& shape, const Extent& extent_,
                 const std::vector<dtype>& z_0, const int n_iter, F get_counter) {
    auto xvals = linspace(extent_.xmin, extent_.xmax, shape.width);
    auto yvals = linspace(extent_.ymin, extent_.ymax, shape.height);
//    ctr_array_t ctr_array(shape.width, std::vector<long>(shape.height));
    auto ctr_array = py::array_t<long>(shape.height * shape.width);
    ctr_array.resize({shape.width, shape.height});
    auto ctr_array_u = ctr_array.mutable_unchecked<2>();

    #pragma omp parallel for
    for (int row = 0; row < shape.width; ++row) {
        __m256d __xvals = _mm256_set1_pd(xvals[row]);            // all xvals are the same
        for (int col = 0; col < shape.height; col += SIMD_SIZE) {

            // "store" and "load" (without "u") both seg-fault
            // this is most likely due to misalignment, but no idea how that is possible... (or fix-able)
            __m256d __yvals = _mm256_loadu_pd(&yvals[col]);      // load 4 y-values in the same row
            __m256i __counter = get_counter(__xvals, __yvals, z_0, n_iter);

            _mm256_storeu_si256((__m256i*)&ctr_array_u(row, col), __counter);
        }
    }

    return ctr_array;
}

auto get_mandelbrot(const Shape& shape, const Extent& extent_,
                    const std::vector<dtype>& z_0, const int n_iter) {
    return get_fractal(shape, extent_, z_0, n_iter, get_counter_mandelbrot);
}
auto get_julia(const Shape& shape, const Extent& extent_,
                    const std::vector<dtype>& z_0, const int n_iter) {
    return get_fractal(shape, extent_, z_0, n_iter, get_counter_julia);
}


auto ctr2rgb(const py::array_t<int>& ctr_arr, const py::array_t<uint32_t>& col_arr) {
    /** https://www.linyuanshi.me/post/pybind11-array/
        https://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html
    */
    py::buffer_info ctr_info = ctr_arr.request();
    py::buffer_info col_info = col_arr.request();
    int height = ctr_info.shape[0];
    int width = ctr_info.shape[1];

    auto rgb = py::array_t<uint32_t>(height * width);
    rgb.resize({height, width});
    auto rgb_arr = rgb.mutable_unchecked<2>();
    auto ctr_arr_u = ctr_arr.unchecked<2>();
    auto col_arr_u = col_arr.unchecked<1>();

////    #pragma omp parallel for
//    for (int row = 0; row < height; ++row) {
//        for (int col = 0; col < width; ++col) {
//            rgb_arr(row, col) = col_arr_u(ctr_arr_u(row, col));
//        }
//    }
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; col += 8) {
            __m256i indx = _mm256_loadu_si256((__m256i*)&ctr_arr_u(row, col));           // SIMD version
            __m256i rgba_val = _mm256_i32gather_epi32((int*)&col_arr_u(0), indx, 4);     // 4 Byte (uint32)
            _mm256_storeu_si256((__m256i*)&rgb_arr(row, col), rgba_val);
        }
    }
    return rgb;
}


int main() {
    /**
    Benchmarks for an AMD Ryzen 7 5800X.

    n_iter, Width, Height, (xmin, xmax, ymin, ymax),
    100,    2048,  1024,   (-2, 1, -1.2, 1.2),
        :: 0.71 sec (Python)
        :: 0.09 sec (C++ scalar)
    1000,
        :: 0.75 sec (C++ scalar)
        :: 0.70 sec (C++ scalar)         <-- OPT: store squares of x and y
        :: 0.12 sec (C++ scalar MP)
        :: 0.12 sec (C++ SIMD)
        :: 0.02 sec (C++ SIMD MP)
    10000,
        :: 0.29 sec (C++ SIMD MP double) --> roughly 200x speedup!

    */

//    const Shape shape;
//    const Extent extent_;
//    const int n_iter = 100;

//    col_array_t col_arr(n_iter + 1, std::vector<int>(3));
//    std::cout << col_arr[0][0] << '\n';
//
//    int ctr = 0;
//    std::vector<int> rgb_val;
//    for (long red = 0; red < 255; ++red) {
//        ++ctr;
//        if (ctr > n_iter) break;
//        rgb_val = {red, red, 255 - red};
//        col_arr[ctr] = rgb_val;
//        if (red == 250) red = 0;
//    }
//
//
////    for (auto& elem : col_arr){
////    std::cout<<elem[0]<<", "<<elem[1]<<", "<<elem[2]<<'\n';}
//
//    ctr_array_t ctr_arr = get_fractal(shape, extent_, n_iter);
//    auto t1 = std::chrono::steady_clock::now();
//    auto rgb = ctr2rgb(ctr_arr, col_arr);
//    auto t2 = std::chrono::steady_clock::now();
//
//    std::cout << ctr_arr[50][50] << '\n';
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
//    std::cout << "Total runtime: " << duration << " ms.\n";
    return 0;
}


PYBIND11_MODULE(fractal_cpp, m) {
    m.doc() = "pybind11 fractal engine"; // optional module docstring
    py::class_<Shape>(m, "Shape", py::dynamic_attr())
    .def(py::init<int, int>())
    .def_readwrite("width", &Shape::width)
    .def_readwrite("height", &Shape::height);

    py::class_<Extent>(m, "Extent", py::dynamic_attr())
    .def(py::init<dtype, dtype, dtype, dtype>())
    .def_readwrite("xmin", &Extent::xmin)
    .def_readwrite("xmax", &Extent::xmax)
    .def_readwrite("ymin", &Extent::ymin)
    .def_readwrite("ymax", &Extent::ymax);

    m.def("get_mandelbrot", &get_mandelbrot,
          "Return a 2d-array of integers corresponding to the final iteration for each pixel",
          py::arg("shape"), py::arg("extent"), py::arg("z_0"), py::arg("n_iter"));

    m.def("get_julia", &get_julia,
          "Return a 2d-array of integers corresponding to the final iteration for each pixel",
          py::arg("shape"), py::arg("extent"), py::arg("z_0"), py::arg("n_iter"));

    m.def("ctr2rgb", &ctr2rgb,
          "Select RGB values from an array depending on the respective 'ctr'-value",
          py::arg("ctr_arr"), py::arg("col_arr"));
}
/** c++ -Wall -O3 -march=native -mavx2 -ffast-math -fopenmp -shared -std=c++2a -fPIC \
$(/home/jo-cube/miniconda3/envs/spyder/bin/python3.11 -m pybind11 --includes) main_numpy_templated.cpp \
-lm -lmvec -o fractal_cpp$(/home/jo-cube/miniconda3/envs/spyder/bin/python3.11-config --extension-suffix --ldflags)
*/
