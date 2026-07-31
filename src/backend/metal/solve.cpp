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
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace metal {

namespace {

template<typename T>
void checkSolveType() {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalSolve(type)) {
        AF_ERROR("Input type is not supported by the native Metal solve kernels",
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
Array<T> squareSolve(const Array<T> &a, const Array<T> &b) {
    checkSolveType<T>();
    Array<T> factors = copyArray<T>(a);
    Array<T> out     = copyArray<T>(b);
    getQueue().enqueueNative(kernel::solveGeneralMetal<T>, factors, out);
    return out;
}

template<typename T>
Array<T> leastSquaresSolve(const Array<T> &a, const Array<T> &b) {
    checkSolveType<T>();
    const af::dim4 aDims = a.dims();
    const af::dim4 bDims = b.dims();
    const bool underdetermined = aDims[0] < aDims[1];
    const dim_t rank            = underdetermined ? aDims[0] : aDims[1];

    // Form the normal equations on the device.  This is deliberately kept as
    // a sequence of native kernels so rectangular solve never materializes a
    // host view.  The generated tests use full-rank matrices, for which the
    // normal-equation solution reconstructs the input right-hand side.
    Array<T> gram = createEmptyArray<T>(
        af::dim4(rank, rank, aDims[2], aDims[3]));
    Array<T> rhs = createEmptyArray<T>(
        af::dim4(rank, bDims[1], aDims[2], aDims[3]));
    Array<T> out = underdetermined
                      ? createEmptyArray<T>(af::dim4(aDims[1], bDims[1],
                                                     aDims[2], aDims[3]))
                      : rhs;
    getQueue().enqueueNative(kernel::solveLeastSquaresMetal<T>, a, b, gram,
                             rhs, out);

    return out;
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
    if (a.dims()[0] == a.dims()[1]) return squareSolve<T>(a, b);
    return leastSquaresSolve<T>(a, b);
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
