/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/meanshift.hpp>
#include <math.hpp>
#include <meanshift.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>
#include <algorithm>
#include <cmath>

using af::dim4;
using std::vector;

namespace arrayfire {
namespace metal {
template<typename T>
Array<T> meanshift(const Array<T> &in, const float &spatialSigma,
                   const float &chromaticSigma, const unsigned &numIterations,
                   const bool &isColor) {
    Array<T> out = createEmptyArray<T>(in.dims());

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalMeanshift(type)) {
        AF_ERROR("Mean shift type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::meanshiftMetal<T>, out, in, spatialSigma,
                             chromaticSigma, numIterations, isColor);

    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> meanshift<T>(const Array<T> &, const float &, \
                                   const float &, const unsigned &, \
                                   const bool &);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
}  // namespace metal
}  // namespace arrayfire
