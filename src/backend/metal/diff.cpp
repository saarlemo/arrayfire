/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <diff.hpp>

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/diff.hpp>
#include <platform.hpp>

#include <af/dim4.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> diff1(const Array<T> &in, const int dim) {
    // Decrement dimension of select dimension
    af::dim4 dims = in.dims();
    dims[dim]--;

    Array<T> outArray = createEmptyArray<T>(dims);

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalDiff(type)) {
        AF_ERROR("Diff type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::diff1Metal<T>, outArray, in, dim);

    return outArray;
}

template<typename T>
Array<T> diff2(const Array<T> &in, const int dim) {
    // Decrement dimension of select dimension
    af::dim4 dims = in.dims();
    dims[dim] -= 2;

    Array<T> outArray = createEmptyArray<T>(dims);

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalDiff(type)) {
        AF_ERROR("Diff type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::diff2Metal<T>, outArray, in, dim);

    return outArray;
}

#define INSTANTIATE(T)                                             \
    template Array<T> diff1<T>(const Array<T> &in, const int dim); \
    template Array<T> diff2<T>(const Array<T> &in, const int dim);

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
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace metal
}  // namespace arrayfire
