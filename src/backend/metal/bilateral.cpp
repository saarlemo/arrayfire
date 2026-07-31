/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <bilateral.hpp>

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/bilateral.hpp>
#include <platform.hpp>

#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename inType, typename outType>
Array<outType> bilateral(const Array<inType> &in, const float &sSigma,
                         const float &cSigma) {
    Array<outType> out = createEmptyArray<outType>(in.dims());
    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<inType>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<outType>::af_type);
    if (!kernel::supportsMetalBilateral(inputType, outputType)) {
        AF_ERROR("Bilateral filter type combination is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::bilateralMetal<inType, outType>, out, in,
                             sSigma, cSigma);
    return out;
}

#define INSTANTIATE(inT, outT)                                    \
    template Array<outT> bilateral<inT, outT>(const Array<inT> &, \
                                              const float &, const float &);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(char, float)
INSTANTIATE(int, float)
INSTANTIATE(uint, float)
INSTANTIATE(schar, float)
INSTANTIATE(uchar, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)

}  // namespace metal
}  // namespace arrayfire
