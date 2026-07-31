/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/moments.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace metal {

static inline unsigned bitCount(unsigned v) {
    v = v - ((v >> 1U) & 0x55555555U);
    v = (v & 0x33333333U) + ((v >> 2U) & 0x33333333U);
    return (((v + (v >> 4U)) & 0xF0F0F0FU) * 0x1010101U) >> 24U;
}

using af::dim4;

template<typename T>
Array<float> moments(const Array<T> &in, const af_moment_type moment) {
    dim4 odims, idims = in.dims();
    dim_t moments_dim = bitCount(moment);

    odims[0] = moments_dim;
    odims[1] = 1;
    odims[2] = idims[2];
    odims[3] = idims[3];

    Array<float> out    = createValueArray<float>(odims, 0.f);
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalMoments(type)) {
        AF_ERROR("Moments type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::momentsMetal<T>, out, in, moment);
    getQueue().sync();
    return out;
}

#define INSTANTIATE(T)                                   \
    template Array<float> moments<T>(const Array<T> &in, \
                                     const af_moment_type moment);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace metal
}  // namespace arrayfire
