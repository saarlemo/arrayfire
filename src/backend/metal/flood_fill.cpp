/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flood_fill.hpp>

#include <err_metal.hpp>
#include <kernel/flood_fill.hpp>

using af::connectivity;

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const af::connectivity nlookup) {
    UNUSED(nlookup);
    auto out = createValueArray(image.dims(), T(0));
    getQueue().enqueueNative(kernel::floodFillMetal<T>, out, image, seedsX,
                             seedsY, newValue, lowValue, highValue);
    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> floodFill(const Array<T>&, const Array<uint>&,           \
                                const Array<uint>&, const T, const T, const T, \
                                const af::connectivity);

INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace metal
}  // namespace arrayfire
