/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <identity.hpp>
#include <kernel/identity.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_metal.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

using arrayfire::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> identity(const dim4& dims) {
    Array<T> out = createEmptyArray<T>(dims);

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalIdentity(type)) {
        AF_ERROR("Identity type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::identityMetal<T>, out);

    return out;
}

#define INSTANTIATE_IDENTITY(T) \
    template Array<T> identity<T>(const af::dim4& dims);

INSTANTIATE_IDENTITY(float)
INSTANTIATE_IDENTITY(double)
INSTANTIATE_IDENTITY(cfloat)
INSTANTIATE_IDENTITY(cdouble)
INSTANTIATE_IDENTITY(int)
INSTANTIATE_IDENTITY(uint)
INSTANTIATE_IDENTITY(intl)
INSTANTIATE_IDENTITY(uintl)
INSTANTIATE_IDENTITY(char)
INSTANTIATE_IDENTITY(schar)
INSTANTIATE_IDENTITY(uchar)
INSTANTIATE_IDENTITY(short)
INSTANTIATE_IDENTITY(ushort)
INSTANTIATE_IDENTITY(half)

}  // namespace metal
}  // namespace arrayfire
