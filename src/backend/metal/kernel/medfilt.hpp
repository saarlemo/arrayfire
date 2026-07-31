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

bool supportsMetalMedfilt(af_dtype type, dim_t windowLength,
                          dim_t windowWidth) noexcept;

void launchMetalMedfilt(BufferParam output, size_t outputBytes,
                        const af::dim4& dims,
                        const af::dim4& outputStrides, BufferParam input,
                        size_t inputBytes, const af::dim4& inputStrides,
                        dim_t windowLength, dim_t windowWidth,
                        af_border_type padding, bool oneDimensional,
                        af_dtype type);

template<typename T>
void medfiltMetal(Param<T> output, CParam<T> input, const dim_t windowLength,
                  const dim_t windowWidth, const af::borderType padding,
                  const bool oneDimensional) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalMedfilt(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.strides(), windowLength, windowWidth,
        static_cast<af_border_type>(padding), oneDimensional,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
