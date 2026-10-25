/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>

#include <err_metal.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <lapack_helper.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <complex>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename T>
using geqrf_work_func_def = int (*)(ORDER_TYPE, int, int, T *, int, T *, T *,
                                    int);

template<typename T>
using gqr_work_func_def = int (*)(ORDER_TYPE, int, int, int, T *, int,
                                  const T *, T *, int);

template<typename T>
geqrf_work_func_def<T> geqrf_work_func();

#define QR_FUNC(TYPE, PREFIX)                                             \
    template<>                                                            \
    geqrf_work_func_def<TYPE> geqrf_work_func<TYPE>() {                   \
        return &LAPACK_NAME(PREFIX##geqrf_work);                           \
    }

QR_FUNC(float, s)
QR_FUNC(double, d)
QR_FUNC(cfloat, c)
QR_FUNC(cdouble, z)

#define GQR_FUNC(TYPE, PREFIX)                                            \
    template<>                                                            \
    gqr_work_func_def<TYPE> gqr_work_func<TYPE>() {                       \
        return &LAPACK_NAME(PREFIX##_work);                               \
    }

template<typename T>
gqr_work_func_def<T> gqr_work_func();
GQR_FUNC(float, sorgqr)
GQR_FUNC(double, dorgqr)
GQR_FUNC(cfloat, cungqr)
GQR_FUNC(cdouble, zungqr)

#undef QR_FUNC
#undef GQR_FUNC

template<typename T>
int qrWorkSize(const T value) {
    return static_cast<int>(value);
}

template<typename T>
int qrWorkSize(const std::complex<T> value) {
    return static_cast<int>(value.real());
}

template<typename T>
int queryGqrWorkspace(int M, int N, int K, T *matrix, int leadingDimension,
                      const T *tau, int *lwork) {
    T query{};
    const int info = gqr_work_func<T>()(AF_LAPACK_COL_MAJOR, M, N, K, matrix,
                                        leadingDimension, tau, &query, -1);
    if (info != 0) { return info; }
    *lwork = std::max(1, qrWorkSize(query));
    return 0;
}

template<typename T>
int factorQr(int M, int N, T *matrix, int leadingDimension, T *tau) {
    T query{};
    int lwork = -1;
    int info  = geqrf_work_func<T>()(AF_LAPACK_COL_MAJOR, M, N, matrix,
                                     leadingDimension, tau, &query, lwork);
    if (info != 0) { return info; }
    lwork = std::max(1, qrWorkSize(query));
    std::vector<T> work(static_cast<size_t>(lwork));
    return geqrf_work_func<T>()(AF_LAPACK_COL_MAJOR, M, N, matrix,
                                leadingDimension, tau, work.data(), lwork);
}

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    const int paddedColumns = std::max(M, N);
    Array<T> denseInput = copyToLapackArray(in);
    q = createEmptyArray<T>(dim4(M, paddedColumns, iDims[2], iDims[3]));
    q.eval();
    getQueue().sync();
    T* qStorage = q.getHostPtr();
    const T* inputData = denseInput.getHostPtr();
    if (M > 0 && paddedColumns > 0 && (!qStorage || !inputData)) {
        AF_ERROR("Accelerate QR requires CPU-visible Metal storage",
                 AF_ERR_RUNTIME);
    }
    const size_t paddedElements =
        static_cast<size_t>(M) * paddedColumns * iDims[2] * iDims[3];
    if (paddedElements > 0) {
        std::fill(qStorage, qStorage + paddedElements, T{});
    }
    if (M > 0 && N > 0) {
        std::memcpy(qStorage, inputData,
                    static_cast<size_t>(M) * N * sizeof(T));
    }
    q.resetDims(iDims);
    t = qr_inplace(q);

    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);
    r.eval();
    q.eval();
    getQueue().sync();
    const T* packed = q.getHostPtr();
    T* upper = r.getHostPtr();
    if (M > 0 && N > 0 && (!packed || !upper)) {
        AF_ERROR("Accelerate QR requires CPU-visible Metal storage",
                 AF_ERR_RUNTIME);
    }
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < M; ++row) {
            upper[row + static_cast<size_t>(col) * M] =
                row <= col ? packed[row * q.strides()[0] +
                                    col * q.strides()[1]]
                            : T{};
        }
    }

    q.resetDims(dim4(M, M));
    if (M == 0) { return; }

    // R is extracted above. Complete Q before LAPACK overwrites the packed
    // reflectors while generating the explicit orthogonal factor.
    q.eval();
    t.eval();
    getQueue().sync();
    T *qData       = q.getHostPtr();
    const T *tauData = t.getHostPtr();
    if (!qData || !tauData) {
        AF_ERROR("Accelerate QR requires CPU-visible Metal buffers",
                 AF_ERR_RUNTIME);
    }

    const int K = std::min(M, N);
    int lwork = 0;
    const int queryInfo = queryGqrWorkspace<T>(
        M, M, K, qData, static_cast<int>(q.strides()[1]), tauData, &lwork);
    if (queryInfo != 0 || lwork < 1) {
        AF_ERROR("Accelerate LAPACK orgqr workspace query failed",
                 AF_ERR_RUNTIME);
    }
    std::vector<T> work(static_cast<size_t>(lwork));
    const int info = gqr_work_func<T>()(
        AF_LAPACK_COL_MAJOR, M, M, K, qData,
        static_cast<int>(q.strides()[1]), tauData, work.data(), lwork);
    if (info != 0) {
        const std::string message =
            "Accelerate LAPACK orgqr/ungqr failed with info=" +
            std::to_string(info);
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];
    if (iDims[2] != 1 || iDims[3] != 1) {
        AF_ERROR("Accelerate QR currently supports one matrix at a time",
                 AF_ERR_NOT_SUPPORTED);
    }
    Array<T> t = createEmptyArray<T>(af::dim4(std::min(M, N), 1, 1, 1));

    if (M == 0 || N == 0) { return t; }

    in.eval();
    t.eval();
    getQueue().sync();
    T *matrix = in.getHostPtr();
    T *tau    = t.getHostPtr();
    const dim4 inStrides = in.strides();
    std::optional<Array<T>> staged;
    if (!matrix) {
        staged.emplace(copyToLapackArray(in));
        getQueue().sync();
        matrix = staged->getHostPtr();
    }
    if (!matrix || !tau) {
        AF_ERROR("Accelerate QR requires CPU-visible Metal storage",
                 AF_ERR_RUNTIME);
    }

    const dim4 matrixStrides = staged ? staged->strides() : inStrides;
    int info = 0;
    if (matrixStrides[0] == 1 && matrixStrides[1] >= M) {
        info = factorQr<T>(M, N, matrix,
                           static_cast<int>(matrixStrides[1]), tau);
    } else {
        std::vector<T> packed(size_t(M) * N);
        for (int col = 0; col < N; ++col) {
            for (int row = 0; row < M; ++row) {
                packed[row + size_t(col) * M] =
                    matrix[row * matrixStrides[0] +
                           col * matrixStrides[1]];
            }
        }
        info = factorQr<T>(M, N, packed.data(), M, tau);
        if (info == 0) {
            for (int col = 0; col < N; ++col) {
                for (int row = 0; row < M; ++row) {
                    matrix[row * matrixStrides[0] +
                           col * matrixStrides[1]] =
                        packed[row + size_t(col) * M];
                }
            }
        }
    }
    if (info != 0) {
        const std::string message =
            "Accelerate LAPACK geqrf failed with info=" +
            std::to_string(info);
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    if (staged) { copyFromLapackArray(in, *staged); }

    return t;
}

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace metal
}  // namespace arrayfire
