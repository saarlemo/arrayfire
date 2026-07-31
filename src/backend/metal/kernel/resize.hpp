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

bool supportsMetalResize(af_dtype type) noexcept;

void launchMetalResize(BufferParam output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, BufferParam input,
                       size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, af_interp_type method,
                       af_dtype type);

template<typename T>
void resizeMetal(Param<T> output, CParam<T> input,
                 const af_interp_type method) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalResize(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.dims(), input.strides(), method,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
