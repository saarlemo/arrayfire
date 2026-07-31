/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <kernel/range.hpp>
#include <range.hpp>

#include <Array.hpp>
#include <err_metal.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

using arrayfire::common::half;

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> range(const dim4& dims, const int seq_dim) {
    // Set dimension along which the sequence should be
    // Other dimensions are simply tiled
    int _seq_dim = seq_dim;
    if (seq_dim < 0) {
        _seq_dim = 0;  // column wise sequence
    }

    Array<T> out = createEmptyArray<T>(dims);
    if (_seq_dim < 0 || _seq_dim > 3) {
        AF_ERROR("Invalid rep selection", AF_ERR_ARG);
    }

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalRange(type)) {
        AF_ERROR("Range type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::rangeMetal<T>, out,
                             static_cast<unsigned>(_seq_dim));

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> range<T>(const af::dim4& dims, const int seq_dims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

}  // namespace metal
}  // namespace arrayfire
