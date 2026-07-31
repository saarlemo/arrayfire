/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <lu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <err_metal.hpp>
#include <kernel/lu.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot,
        const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    Array<T> in_copy = copyArray<T>(in);
    pivot            = lu_inplace(in_copy);

    // SPLIT into lower and upper
    dim4 ldims(M, min(M, N));
    dim4 udims(min(M, N), N);
    lower = createEmptyArray<T>(ldims);
    upper = createEmptyArray<T>(udims);

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalLu(type)) {
        AF_ERROR("Input type is not supported by the Metal LU split kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::luSplitMetal<T>, lower, upper, in_copy);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot) {
    dim4 iDims = in.dims();
    Array<int> pivot =
        createEmptyArray<int>(af::dim4(min(iDims[0], iDims[1]), 1, 1, 1));

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalLuFactor(type)) {
        AF_ERROR("Input type is not supported by the Metal LU kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::luFactorMetal<T>, in, pivot);

    if (convert_pivot) {
        Array<int> p = range<int>(dim4(iDims[0]), 0);
        getQueue().enqueueNative(kernel::convertPivotMetal, p, pivot);
        return p;
    } else {
        return pivot;
    }
}

bool isLAPACKAvailable() { return true; }

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot,
        const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

bool isLAPACKAvailable() { return false; }

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

#define INSTANTIATE_LU(T)                                        \
    template Array<int> lu_inplace<T>(Array<T> & in,             \
                                      const bool convert_pivot); \
    template void lu<T>(Array<T> & lower, Array<T> & upper,      \
                        Array<int> & pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}  // namespace metal
}  // namespace arrayfire
