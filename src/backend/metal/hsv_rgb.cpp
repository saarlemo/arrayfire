/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <hsv_rgb.hpp>
#include <kernel/hsv_rgb.hpp>
#include <metal_compute.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> hsv2rgb(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (kernel::supportsMetalHsvRgb(type)) {
        getQueue().enqueue(kernel::hsvRgbMetal<T>, out, in, true);
    } else {
        getQueue().enqueue(kernel::hsv2rgb<T>, out, in);
    }

    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (kernel::supportsMetalHsvRgb(type)) {
        getQueue().enqueue(kernel::hsvRgbMetal<T>, out, in, false);
    } else {
        getQueue().enqueue(kernel::rgb2hsv<T>, out, in);
    }

    return out;
}

#define INSTANTIATE(T)                                \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in);

INSTANTIATE(double)
INSTANTIATE(float)

}  // namespace metal
}  // namespace arrayfire
