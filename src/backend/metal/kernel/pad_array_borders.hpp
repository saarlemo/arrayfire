/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/traits.hpp>
#include <utility.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalPadBorders(af_dtype type) noexcept;

void launchMetalPadBorders(BufferParam output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam input,
                           size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& lowerPadding,
                           af_border_type borderType, af_dtype type);

template<typename T>
void padBordersMetal(Param<T> output, CParam<T> input,
                     const af::dim4 lowerPadding, const af::dim4,
                     const af::borderType borderType) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalPadBorders(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.dims(), input.strides(), lowerPadding,
        static_cast<af_border_type>(borderType),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
