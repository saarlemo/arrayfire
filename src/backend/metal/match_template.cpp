/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <match_template.hpp>

#include <err_metal.hpp>
#include <kernel/match_template.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename inType, typename outType>
Array<outType> match_template(const Array<inType> &sImg,
                              const Array<inType> &tImg,
                              const af::matchType mType) {
    Array<outType> out = createEmptyArray<outType>(sImg.dims());
    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<inType>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<outType>::af_type);
    if (!kernel::supportsMetalMatchTemplate(inputType, outputType)) {
        AF_ERROR("Template-matching type combination is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::matchTemplateMetal<inType, outType>, out,
                             sImg, tImg, mType);
    return out;
}

#define INSTANTIATE(in_t, out_t)                       \
    template Array<out_t> match_template<in_t, out_t>( \
        const Array<in_t> &, const Array<in_t> &, const af::matchType);

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
