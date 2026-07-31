/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <medfilt.hpp>

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/medfilt.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> medfilt1(const Array<T> &in, const int w_wid,
                  const af::borderType pad) {
    Array<T> out        = createEmptyArray<T>(in.dims());
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalMedfilt(type, w_wid, 1)) {
        AF_ERROR("One-dimensional median-filter type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::medfiltMetal<T>, out, in, w_wid, 1, pad,
                             true);
    return out;
}

template<typename T>
Array<T> medfilt2(const Array<T> &in, const int w_len, const int w_wid,
                  const af::borderType pad) {
    Array<T> out        = createEmptyArray<T>(in.dims());
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalMedfilt(type, w_len, w_wid)) {
        AF_ERROR("Median-filter type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::medfiltMetal<T>, out, in, w_len, w_wid,
                             pad, false);
    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> medfilt1<T>(const Array<T> &in, const int w_wid, \
                                  const af::borderType);               \
    template Array<T> medfilt2<T>(const Array<T> &in, const int w_len, \
                                  const int w_wid, const af::borderType);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace metal
}  // namespace arrayfire
