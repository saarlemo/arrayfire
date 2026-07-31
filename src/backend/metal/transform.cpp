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
#include <kernel/transform.hpp>
#include <platform.hpp>
#include <transform.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const af_interp_type method, const bool inverse,
               const bool perspective) {
    out.eval();
    in.eval();

    tf.eval();

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE:
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG); break;
    }

    if (!kernel::supportsMetalTransform(type, method)) {
        AF_ERROR("Metal transform does not support this type",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::transformMetal<T>, out, in, tf, method,
                             inverse, perspective);
}

#define INSTANTIATE(T)                                                       \
    template void transform(Array<T> &out, const Array<T> &in,               \
                            const Array<float> &tf,                          \
                            const af_interp_type method, const bool inverse, \
                            const bool perspective);

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
