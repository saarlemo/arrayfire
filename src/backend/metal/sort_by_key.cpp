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
#include <kernel/sort_by_key.hpp>
#include <queue.hpp>
#include <sort_by_key.hpp>

namespace arrayfire {
namespace metal {

template<typename Tk, typename Tv>
void sort_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey,
                 const Array<Tv> &ival, const uint dim, bool isAscending) {
    okey = copyArray<Tk>(ikey);
    oval = copyArray<Tv>(ival);

    if (dim > 3) AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
    const af_dtype keyType =
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type);
    const af_dtype valueType =
        static_cast<af_dtype>(af::dtype_traits<Tv>::af_type);
    if (!kernel::supportsMetalSortByKey(keyType, valueType))
        AF_ERROR("Types are not supported by the Metal sort-by-key kernel",
                 AF_ERR_NOT_SUPPORTED);
    getQueue().enqueueNative(kernel::sortByKeyMetal<Tk, Tv>, okey, oval,
                             static_cast<int>(dim), isAscending);
}

#define INSTANTIATE(Tk, Tv)                                        \
    template void sort_by_key<Tk, Tv>(                             \
        Array<Tk> & okey, Array<Tv> & oval, const Array<Tk> &ikey, \
        const Array<Tv> &ival, const uint dim, bool isAscending);

#define INSTANTIATE1(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, int)     \
    INSTANTIATE(Tk, uint)    \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, schar)   \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

INSTANTIATE1(float)
INSTANTIATE1(double)
INSTANTIATE1(int)
INSTANTIATE1(uint)
INSTANTIATE1(char)
INSTANTIATE1(schar)
INSTANTIATE1(uchar)
INSTANTIATE1(short)
INSTANTIATE1(ushort)
INSTANTIATE1(intl)
INSTANTIATE1(uintl)

}  // namespace metal
}  // namespace arrayfire
