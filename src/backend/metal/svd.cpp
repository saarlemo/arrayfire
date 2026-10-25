/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/err_common.hpp>
#include <err_metal.hpp>
#include <lapack_helper.hpp>
#include <svd.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <queue.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace arrayfire {
namespace metal {

template<typename T, typename Tr>
using svd_func_def = int (*)(ORDER_TYPE, char, int, int, T *, int, Tr *, T *,
                             int, T *, int);

#define SVD_FUNC_DEF()            \
    template<typename T, typename Tr> \
    svd_func_def<T, Tr> svd_func();

#define SVD_FUNC(T, Tr, PREFIX)                  \
    template<>                                   \
    svd_func_def<T, Tr> svd_func<T, Tr>() {      \
        return &LAPACK_NAME(PREFIX##gesdd);      \
    }

SVD_FUNC_DEF()
SVD_FUNC(float, float, s)
SVD_FUNC(double, double, d)
SVD_FUNC(cfloat, float, c)
SVD_FUNC(cdouble, double, z)

#undef SVD_FUNC_DEF
#undef SVD_FUNC

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    const int M = static_cast<int>(in.dims()[0]);
    const int N = static_cast<int>(in.dims()[1]);
    if (in.dims()[2] != 1 || in.dims()[3] != 1) {
        AF_ERROR("Accelerate SVD currently supports one matrix at a time",
                 AF_ERR_NOT_SUPPORTED);
    }
    if (M == 0 || N == 0) {
        u.eval();
        vt.eval();
        getQueue().sync();
        T *leftVectors  = u.getHostPtr();
        T *rightVectors = vt.getHostPtr();
        if (leftVectors) {
            std::fill(leftVectors, leftVectors + size_t(M) * M, T(0));
            for (int i = 0; i < M; ++i) { leftVectors[i + size_t(i) * M] = T(1); }
        }
        if (rightVectors) {
            std::fill(rightVectors, rightVectors + size_t(N) * N, T(0));
            for (int i = 0; i < N; ++i) { rightVectors[i + size_t(i) * N] = T(1); }
        }
        return;
    }

    // Accelerate LAPACK requires column-major data with unit row stride.
    // Metal arrays normally use CPU-visible shared buffers; materialize views
    // or imported private buffers into an ArrayFire-owned shared allocation.
    in.eval();
    getQueue().sync();
    std::optional<Array<T>> staged;
    Array<T> *matrix = &in;
    T *input        = in.getHostPtr();
    const dim4 inputStrides = in.strides();
    if (inputStrides[0] != 1 || !input || inputStrides[1] < M) {
        staged.emplace(copyToLapackArray(in));
        getQueue().sync();
        matrix = &*staged;
        input  = matrix->getHostPtr();
    }

    s.eval();
    u.eval();
    vt.eval();
    getQueue().sync();
    Tr *singularValues = s.getHostPtr();
    T *leftVectors    = u.getHostPtr();
    T *rightVectors   = vt.getHostPtr();
    if (!input || !singularValues || !leftVectors || !rightVectors) {
        AF_ERROR("Accelerate SVD requires CPU-visible Metal buffers",
                 AF_ERR_RUNTIME);
    }

    const int info = svd_func<T, Tr>()(
        AF_LAPACK_COL_MAJOR, 'A', M, N, input,
        static_cast<int>(matrix->strides()[1]), singularValues, leftVectors,
        static_cast<int>(u.strides()[1]), rightVectors,
        static_cast<int>(vt.strides()[1]));
    if (info != 0) {
        const std::string message =
            "Accelerate LAPACK gesdd failed with info=" + std::to_string(info);
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    if (staged) { copyFromLapackArray(in, *staged); }
}

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    Array<T> in_copy = copyToLapackArray(in);
    svdInPlace(s, u, vt, in_copy);
}

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

#define INSTANTIATE_SVD(T, Tr)                                           \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, \
                             const Array<T> &in);                        \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u,         \
                                    Array<T> & vt, Array<T> & in);

INSTANTIATE_SVD(float, float)
INSTANTIATE_SVD(double, double)
INSTANTIATE_SVD(cfloat, float)
INSTANTIATE_SVD(cdouble, double)

}  // namespace metal
}  // namespace arrayfire
