/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/rotate.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <rotate.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const af::dim4 &odims,
                const af_interp_type method) {
    Array<T> out        = createEmptyArray<T>(odims);
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE: break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
    }
    if (!kernel::supportsMetalRotate(type, method)) {
        AF_ERROR("Rotation type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }

    getQueue().enqueueNative(kernel::rotateMetal<T>, out, in, theta, method);
    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> rotate(const Array<T> &in, const float theta, \
                             const af::dim4 &odims,                 \
                             const af_interp_type method);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace metal
}  // namespace arrayfire
