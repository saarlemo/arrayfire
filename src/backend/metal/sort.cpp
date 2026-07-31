/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <copy.hpp>
#include <kernel/sort.hpp>
#include <queue.hpp>
#include <sort.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> sort(const Array<T>& in, const unsigned dim, bool isAscending) {
    if (dim > 3) AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);

    Array<T> out = copyArray<T>(in);
    const af_dtype type =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalSort(type))
        AF_ERROR("Type is not supported by the Metal sort kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::sortMetal<T>, out, static_cast<int>(dim),
                             isAscending);
    return out;
}

#define INSTANTIATE(T)                                                \
    template Array<T> sort<T>(const Array<T>& in, const unsigned dim, \
                              bool isAscending);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)

}  // namespace metal
}  // namespace arrayfire
