/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalMatchTemplate(af_dtype inputType,
                                af_dtype outputType) noexcept;

void launchMetalMatchTemplate(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam search, size_t searchBytes,
    const af::dim4& searchStrides, BufferParam templ, size_t templateBytes,
    const af::dim4& templateDims, const af::dim4& templateStrides,
    af_match_type matchType, af_dtype inputType, af_dtype outputType);

template<typename Ti, typename To>
void matchTemplateMetal(Param<To> output, CParam<Ti> search, CParam<Ti> templ,
                        const af::matchType matchType) {
    size_t searchElements = 1, templateElements = 1;
    for (int i = 0; i < 4; ++i) {
        searchElements +=
            size_t(search.dims(i) - 1) * size_t(search.strides(i));
        templateElements +=
            size_t(templ.dims(i) - 1) * size_t(templ.strides(i));
    }
    launchMetalMatchTemplate(
        output.bufferParam(), size_t(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), search.bufferParam(),
        searchElements * sizeof(Ti), search.strides(), templ.bufferParam(),
        templateElements * sizeof(Ti), templ.dims(), templ.strides(),
        static_cast<af_match_type>(matchType),
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type),
        static_cast<af_dtype>(af::dtype_traits<To>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
