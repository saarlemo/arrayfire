/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/defines.hpp>
#include <common/half.hpp>
#include <convolve.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace metal {

template<typename T, typename accT>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  AF_BATCH_KIND kind, const int rank, const bool expand) {
    auto sDims = signal.dims();
    auto fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (int d = 0; d < AF_MAX_DIMS; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < rank ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fDims[i]; }
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    const af_dtype inputType = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    const af_dtype filterType =
        static_cast<af_dtype>(af::dtype_traits<accT>::af_type);
    if (kernel::supportsMetalConvolve(inputType, filterType)) {
        getQueue().enqueueNative(kernel::convolveMetal<T, accT>, out, signal,
                                 filter, kind, rank, expand);
        return out;
    }

    AF_ERROR("Input type is not supported by the Metal convolution kernel",
             AF_ERR_NOT_SUPPORTED);

    return out;
}

template<typename T, typename accT>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter, const bool expand) {
    const auto &sDims = signal.dims();
    dim4 tDims        = sDims;
    dim4 oDims        = sDims;

    if (expand) {
        auto cfDims = c_filter.dims();
        auto rfDims = r_filter.dims();

        auto cflen = cfDims.elements();
        auto rflen = rfDims.elements();
        // separable convolve only does AF_BATCH_NONE and standard
        // batch(AF_BATCH_LHS)
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> out  = createEmptyArray<T>(oDims);
    Array<T> temp = createEmptyArray<T>(tDims);

    const af_dtype inputType = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    const af_dtype filterType =
        static_cast<af_dtype>(af::dtype_traits<accT>::af_type);
    if (!kernel::supportsMetalSeparableConvolve(inputType, filterType)) {
        AF_ERROR(
            "Input type is not supported by the Metal separable convolution kernel",
            AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::separableConvolveMetal<T, accT>, out, temp,
                             signal, c_filter, r_filter, expand);
    return out;
}

#define INSTANTIATE(T, accT)                                                   \
    template Array<T> convolve<T, accT>(Array<T> const &, Array<accT> const &, \
                                        AF_BATCH_KIND, const int, const bool); \
    template Array<T> convolve2<T, accT>(Array<T> const &,                     \
                                         Array<accT> const &,                  \
                                         Array<accT> const &, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(schar, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)
#undef INSTANTIATE

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    const dim4 sDims = signal.dims();
    const dim4 fDims = filter.dims();
    const dim4 outDims(
        1 + (sDims[0] + 2 * padding[0] -
             (((fDims[0] - 1) * dilation[0]) + 1)) /
                stride[0],
        1 + (sDims[1] + 2 * padding[1] -
             (((fDims[1] - 1) * dilation[1]) + 1)) /
                stride[1],
        fDims[3], sDims[3]);

    Array<T> out = createEmptyArray<T>(outDims);
    const af_dtype type =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalConvolveNN(type)) {
        AF_ERROR("Input type is not supported by the Metal NN convolution kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::convolveNNMetal<T>, out, signal, filter,
                             stride, padding, dilation);
    return out;
}

#define INSTANTIATE(T)                                                        \
    template Array<T> convolve2<T>(Array<T> const &signal,                    \
                                   Array<T> const &filter, const dim4 stride, \
                                   const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

template<typename T>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<T> &original_filter,
                           const Array<T> & /*convolved_output*/,
                           af::dim4 stride, af::dim4 padding,
                           af::dim4 dilation) {
    const af_dtype type =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalConvolveNN(type)) {
        AF_ERROR(
            "Input type is not supported by the Metal NN convolution gradient kernel",
            AF_ERR_NOT_SUPPORTED);
    }
    Array<T> out = createEmptyArray<T>(original_signal.dims());
    getQueue().enqueueNative(kernel::convolveNNDataGradientMetal<T>, out,
                             incoming_gradient, original_filter, stride,
                             padding, dilation);
    return out;
}

template<typename T>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> & /*convolved_output*/,
                             af::dim4 stride, af::dim4 padding,
                             af::dim4 dilation) {
    const af_dtype type =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalConvolveNN(type)) {
        AF_ERROR(
            "Input type is not supported by the Metal NN convolution gradient kernel",
            AF_ERR_NOT_SUPPORTED);
    }
    Array<T> out = createEmptyArray<T>(original_filter.dims());
    getQueue().enqueueNative(kernel::convolveNNFilterGradientMetal<T>, out,
                             original_signal, incoming_gradient, stride,
                             padding, dilation);
    return out;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> conv2DataGradient<T>(                                 \
        Array<T> const &incoming_gradient, Array<T> const &original_signal, \
        Array<T> const &original_filter, Array<T> const &convolved_output,  \
        const dim4 stride, const dim4 padding, const dim4 dilation);        \
    template Array<T> conv2FilterGradient<T>(                               \
        Array<T> const &incoming_gradient, Array<T> const &original_signal, \
        Array<T> const &original_filter, Array<T> const &convolved_output,  \
        const dim4 stride, const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace metal
}  // namespace arrayfire
