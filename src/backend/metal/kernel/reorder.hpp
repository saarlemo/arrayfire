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

bool supportsMetalReorder(af_dtype type) noexcept;

void launchMetalReorder(BufferParam output, size_t outputBytes,
                        const af::dim4& outputDims,
                        const af::dim4& outputStrides, BufferParam input,
                        size_t inputBytes, const af::dim4& inputStrides,
                        const af::dim4& reorderDims, af_dtype type);

template<typename T>
void reorderMetal(Param<T> output, CParam<T> input, const af::dim4,
                  const af::dim4 reorderDims) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalReorder(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.strides(), reorderDims,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
