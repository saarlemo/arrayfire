/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>

#include <err_metal.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <kernel/solve.hpp>
#include <lapack_helper.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

namespace arrayfire {
namespace metal {

namespace {

template<typename T>
using gesv_func_def = int (*)(ORDER_TYPE, int, int, T *, int, int *, T *, int);

template<typename T>
using gels_work_func_def = int (*)(ORDER_TYPE, char, int, int, int, T *, int,
                                   T *, int, T *, int);

template<typename T>
gesv_func_def<T> gesv_func();

template<typename T>
gels_work_func_def<T> gels_work_func();

#define SOLVE_LAPACK_FUNC(TYPE, PREFIX)                                      \
    template<>                                                               \
    gesv_func_def<TYPE> gesv_func<TYPE>() {                                  \
        return &LAPACK_NAME(PREFIX##gesv);                                   \
    }                                                                        \
    template<>                                                               \
    gels_work_func_def<TYPE> gels_work_func<TYPE>() {                        \
        return &LAPACK_NAME(PREFIX##gels_work);                              \
    }

SOLVE_LAPACK_FUNC(float, s)
SOLVE_LAPACK_FUNC(double, d)
SOLVE_LAPACK_FUNC(cfloat, c)
SOLVE_LAPACK_FUNC(cdouble, z)

#undef SOLVE_LAPACK_FUNC

template<typename T>
int lapackWorkSize(const T value) {
    return static_cast<int>(value);
}

template<typename T>
int lapackWorkSize(const std::complex<T> value) {
    return static_cast<int>(value.real());
}

template<typename T>
void checkSolveType() {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalSolve(type)) {
        AF_ERROR("Input type is not supported by the Metal solve backend",
                 AF_ERR_NOT_SUPPORTED);
    }
}

template<typename T>
Array<T> triangleSolve(const Array<T> &a, const Array<T> &b,
                       const af_mat_prop options) {
    checkSolveType<T>();
    Array<T> out = copyArray<T>(b);
    getQueue().enqueueNative(kernel::solveTriangularMetal<T>, a, out,
                             bool(options & AF_MAT_UPPER),
                             bool(options & AF_MAT_DIAG_UNIT));
    return out;
}

template<typename T>
Array<T> accelerateSolve(const Array<T> &a, const Array<T> &b) {
    const af::dim4 aDims = a.dims();
    const af::dim4 bDims = b.dims();
    const int M = static_cast<int>(aDims[0]);
    const int N = static_cast<int>(aDims[1]);
    const int NRHS = static_cast<int>(bDims[1]);
    const int batchZ = static_cast<int>(aDims[2]);
    const int batchW = static_cast<int>(aDims[3]);
    const bool square = M == N;
    const int paddedRows = std::max(M, N);

    if (M == 0 || N == 0 || NRHS == 0) {
        return createEmptyArray<T>(af::dim4(N, NRHS, bDims[2], bDims[3]));
    }

    Array<T> factors = copyToLapackArray(a);
    Array<T> output = copyToLapackArray(
        b, af::dim4(paddedRows, NRHS, bDims[2], bDims[3]));

    factors.eval();
    output.eval();
    getQueue().sync();
    T *factorData = factors.getHostPtr();
    T *rhsData = output.getHostPtr();
    if (!factorData || !rhsData) {
        AF_ERROR("Accelerate solve requires CPU-visible Metal storage",
                 AF_ERR_RUNTIME);
    }

    const int lda = static_cast<int>(factors.strides()[1]);
    const int ldb = static_cast<int>(output.strides()[1]);
    const size_t factorBatchStrideZ = factors.strides()[2];
    const size_t factorBatchStrideW = factors.strides()[3];
    const size_t rhsBatchStrideZ = output.strides()[2];
    const size_t rhsBatchStrideW = output.strides()[3];

    int lwork = 0;
    std::vector<T> work;
    if (!square) {
        T workQuery{};
        const int queryInfo = gels_work_func<T>()(
            AF_LAPACK_COL_MAJOR, 'N', M, N, NRHS, factorData, lda, rhsData,
            ldb, &workQuery, -1);
        if (queryInfo != 0) {
            AF_ERROR("Accelerate LAPACK gels workspace query failed",
                     AF_ERR_RUNTIME);
        }
        lwork = std::max(1, lapackWorkSize(workQuery));
        work.resize(static_cast<size_t>(lwork));
    }
    std::vector<int> pivots(square ? static_cast<size_t>(N) : 0);

    for (int w = 0; w < batchW; ++w) {
        for (int z = 0; z < batchZ; ++z) {
            T *batchFactors = factorData + factorBatchStrideZ * z +
                              factorBatchStrideW * w;
            T *batchRhs = rhsData + rhsBatchStrideZ * z + rhsBatchStrideW * w;
            int info = 0;
            if (square) {
                info = gesv_func<T>()(AF_LAPACK_COL_MAJOR, N, NRHS,
                                      batchFactors, lda, pivots.data(),
                                      batchRhs, ldb);
            } else {
                info = gels_work_func<T>()(
                    AF_LAPACK_COL_MAJOR, 'N', M, N, NRHS, batchFactors, lda,
                    batchRhs, ldb, work.data(), lwork);
            }
            if (info != 0) {
                const std::string message =
                    std::string("Accelerate LAPACK ") +
                    (square ? "gesv" : "gels") +
                    " failed with info=" + std::to_string(info) +
                    " at batch (" + std::to_string(z) + ", " +
                    std::to_string(w) + ")";
                AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
            }
        }
    }

    if (!square) {
        output.resetDims(af::dim4(N, NRHS, bDims[2], bDims[3]));
    }
    return output;
}

}  // namespace

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    UNUSED(options);
    checkSolveType<T>();
    Array<T> out = copyArray<T>(b);
    getQueue().enqueueNative(kernel::solveLUMetal<T>, A, pivot, out);
    return out;
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    if (options & AF_MAT_UPPER || options & AF_MAT_LOWER) {
        return triangleSolve<T>(a, b, options);
    }
    return accelerateSolve<T>(a, b);
}

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    UNUSED(A);
    UNUSED(pivot);
    UNUSED(b);
    UNUSED(options);
    AF_ERROR(
        "This version of ArrayFire was built without linear algebra routines",
        AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    UNUSED(a);
    UNUSED(b);
    UNUSED(options);
    AF_ERROR(
        "This version of ArrayFire was built without linear algebra routines",
        AF_ERR_NOT_CONFIGURED);
}

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

#define INSTANTIATE_SOLVE(T)                                                 \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,         \
                               const af_mat_prop options);                   \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b,                          \
                                 const af_mat_prop options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}  // namespace metal
}  // namespace arrayfire
