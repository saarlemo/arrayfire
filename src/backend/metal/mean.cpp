/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/mean.hpp>
#include <mean.hpp>
#include <queue.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <complex>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace metal {

template<typename Ti, typename Tw, typename To>
Array<To> mean(const Array<Ti>& in, const int dim) {
    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);

    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    if (!kernel::supportsMetalMean(inputType, outputType))
        AF_ERROR("Types are not supported by the Metal mean kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::meanMetal<Ti, To>, out, in, dim, false);
    return out;
}

template<typename T, typename Tw>
Array<T> mean(const Array<T>& in, const Array<Tw>& wt, const int dim) {
    dim4 odims   = in.dims();
    odims[dim]   = 1;
    Array<T> out = createEmptyArray<T>(odims);

    const af_dtype valueType =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    const af_dtype weightType =
        static_cast<af_dtype>(af::dtype_traits<Tw>::af_type);
    if (!kernel::supportsMetalMeanWeighted(valueType, weightType))
        AF_ERROR("Types are not supported by the Metal weighted-mean kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::meanWeightedMetal<T, Tw>, out, in, wt, dim,
                             false);
    return out;
}

template<typename T, typename Tw>
T mean(const Array<T>& in, const Array<Tw>& wt) {
    Array<T> out = createEmptyArray<T>(1);

    const af_dtype valueType =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    const af_dtype weightType =
        static_cast<af_dtype>(af::dtype_traits<Tw>::af_type);
    if (!kernel::supportsMetalMeanWeighted(valueType, weightType))
        AF_ERROR("Types are not supported by the Metal weighted-mean kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::meanWeightedMetal<T, Tw>, out, in, wt, 0,
                             true);
    getQueue().sync();
    return out.getHostPtr()[0];
}

template<typename Ti, typename Tw, typename To>
To mean(const Array<Ti>& in) {
    Array<To> out = createEmptyArray<To>(1);

    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    if (!kernel::supportsMetalMean(inputType, outputType))
        AF_ERROR("Types are not supported by the Metal mean kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::meanMetal<Ti, To>, out, in, 0, true);
    getQueue().sync();
    return out.getHostPtr()[0];
}

#define INSTANTIATE(Ti, Tw, To)                        \
    template To mean<Ti, Tw, To>(const Array<Ti>& in); \
    template Array<To> mean<Ti, Tw, To>(const Array<Ti>& in, const int dim);

INSTANTIATE(double, double, double);
INSTANTIATE(float, float, float);
INSTANTIATE(int, float, float);
INSTANTIATE(unsigned, float, float);
INSTANTIATE(intl, double, double);
INSTANTIATE(uintl, double, double);
INSTANTIATE(short, float, float);
INSTANTIATE(ushort, float, float);
INSTANTIATE(schar, float, float);
INSTANTIATE(uchar, float, float);
INSTANTIATE(char, float, float);
INSTANTIATE(cfloat, float, cfloat);
INSTANTIATE(cdouble, double, cdouble);
INSTANTIATE(half, float, half);
INSTANTIATE(half, float, float);

#define INSTANTIATE_WGT(T, Tw)                                              \
    template T mean<T, Tw>(const Array<T>& in, const Array<Tw>& wts);       \
    template Array<T> mean<T, Tw>(const Array<T>& in, const Array<Tw>& wts, \
                                  const int dim);

INSTANTIATE_WGT(double, double);
INSTANTIATE_WGT(float, float);
INSTANTIATE_WGT(cfloat, float);
INSTANTIATE_WGT(cdouble, double);
INSTANTIATE_WGT(half, float);

}  // namespace metal
}  // namespace arrayfire
