/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cholesky.hpp>

#include <common/err_common.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <Array.hpp>
#include <lapack_helper.hpp>
#include <types.hpp>

#include <platform.hpp>
#include <af/dim4.hpp>

#include <optional>
#include <string>
#include <vector>

namespace arrayfire {
namespace metal {

template<typename T>
using potrf_func_def = int (*)(ORDER_TYPE, char, int, T *, int);

template<typename T>
potrf_func_def<T> potrf_func();

#define CH_FUNC(TYPE, PREFIX)                                       \
    template<>                                                      \
    potrf_func_def<TYPE> potrf_func<TYPE>() {                       \
        return &LAPACK_NAME(PREFIX##potrf);                         \
    }

CH_FUNC(float, s)
CH_FUNC(double, d)
CH_FUNC(cfloat, c)
CH_FUNC(cdouble, z)

#undef CH_FUNC

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    Array<T> out = copyToLapackArray(in);
    *info        = cholesky_inplace(out, is_upper);
    out.eval();
    getQueue().sync();
    T* matrix = out.getHostPtr();
    const af::dim4 dims = out.dims();
    const af::dim4 strides = out.strides();
    if (out.elements() > 0 && !matrix) {
        AF_ERROR("Accelerate Cholesky requires CPU-visible Metal storage",
                 AF_ERR_RUNTIME);
    }
    for (dim_t w = 0; w < dims[3]; ++w) {
        for (dim_t z = 0; z < dims[2]; ++z) {
            for (dim_t col = 0; col < dims[1]; ++col) {
                for (dim_t row = 0; row < dims[0]; ++row) {
                    if (is_upper ? row > col : row < col) {
                        matrix[row * strides[0] + col * strides[1] +
                               z * strides[2] + w * strides[3]] = T{};
                    }
                }
            }
        }
    }

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    const int N = static_cast<int>(in.dims()[0]);
    if (in.dims()[1] != N || in.dims()[2] != 1 || in.dims()[3] != 1) {
        AF_ERROR("Accelerate Cholesky requires one square matrix", AF_ERR_ARG);
    }
    if (N == 0) { return 0; }

    in.eval();
    getQueue().sync();
    T *matrix = in.getHostPtr();
    const dim4 inStrides = in.strides();
    std::optional<Array<T>> staged;
    if (!matrix) {
        staged.emplace(copyToLapackArray(in));
        getQueue().sync();
        matrix = staged->getHostPtr();
    }
    if (!matrix) {
        AF_ERROR("Accelerate Cholesky requires CPU-visible Metal storage",
                 AF_ERR_RUNTIME);
    }

    const dim4 matrixStrides = staged ? staged->strides() : inStrides;
    const char uplo = is_upper ? 'U' : 'L';
    int info = 0;
    if (matrixStrides[0] == 1 && matrixStrides[1] >= N) {
        info = potrf_func<T>()(AF_LAPACK_COL_MAJOR, uplo, N, matrix,
                               static_cast<int>(matrixStrides[1]));
    } else {
        std::vector<T> packed(size_t(N) * N);
        for (int col = 0; col < N; ++col) {
            for (int row = 0; row < N; ++row) {
                packed[row + size_t(col) * N] =
                    matrix[row * matrixStrides[0] +
                           col * matrixStrides[1]];
            }
        }
        info = potrf_func<T>()(AF_LAPACK_COL_MAJOR, uplo, N, packed.data(), N);
        for (int col = 0; col < N; ++col) {
            for (int row = 0; row < N; ++row) {
                matrix[row * matrixStrides[0] +
                       col * matrixStrides[1]] =
                    packed[row + size_t(col) * N];
            }
        }
    }
    if (staged) {
        copyFromLapackArray(in, *staged);
    }
    return info;
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
