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
#include <copy.hpp>
#include <kernel/cholesky.hpp>
#include <types.hpp>

#include <platform.hpp>
#include <triangle.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    Array<T> out = copyArray<T>(in);
    *info        = cholesky_inplace(out, is_upper);

    triangle<T>(out, out, is_upper, false);

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalCholesky(type)) {
        AF_ERROR("Input type is not supported by the Metal Cholesky kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    int info = 0;
    getQueue().enqueueNative(kernel::choleskyMetal<T>, in, is_upper, &info);
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
