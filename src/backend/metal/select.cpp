/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <kernel/select.hpp>
#include <select.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <metal_compute.hpp>
#include <platform.hpp>
#include <queue.hpp>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace metal {

template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b) {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (kernel::supportsMetalSelect(type)) {
        getQueue().enqueue(kernel::selectMetal<T>, out, cond, a, b);
    } else {
        // Apple GPUs do not expose FP64 in Metal.
        getQueue().enqueue(kernel::select<T>, out, cond, a, b);
    }
}

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const T &b) {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (kernel::supportsMetalSelect(type)) {
        getQueue().enqueue(kernel::selectScalarMetal<T, flip>, out, cond, a, b);
    } else {
        // Apple GPUs do not expose FP64 in Metal.
        getQueue().enqueue(kernel::select_scalar<T, flip>, out, cond, a, b);
    }
}

#define INSTANTIATE(T)                                                   \
    template void select<T>(Array<T> & out, const Array<char> &cond,     \
                            const Array<T> &a, const Array<T> &b);       \
    template void select_scalar<T, true>(Array<T> & out,                 \
                                         const Array<char> &cond,        \
                                         const Array<T> &a, const T &b); \
    template void select_scalar<T, false>(Array<T> & out,                \
                                          const Array<char> &cond,       \
                                          const Array<T> &a, const T &b);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(char)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace metal
}  // namespace arrayfire
