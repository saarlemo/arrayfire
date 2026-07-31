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

#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalJoin(af_dtype type) noexcept;

void launchMetalJoinAppend(BufferParam output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam input,
                           size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& outputOffset, af_dtype type);

template<typename T>
void joinMetal(const int dim, Param<T> output,
               const std::vector<CParam<T>> inputs, const int inputCount) {
    af::dim4 outputOffset(0, 0, 0, 0);
    for (int inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
        const CParam<T>& input = inputs[inputIndex];
        if (input.dims().elements() == 0) { continue; }
        size_t inputElements = 1;
        for (int i = 0; i < 4; ++i) {
            inputElements += static_cast<size_t>(input.dims(i) - 1) *
                             static_cast<size_t>(input.strides(i));
        }
        launchMetalJoinAppend(
            output.bufferParam(),
            static_cast<size_t>(output.dims().elements()) * sizeof(T),
            output.dims(), output.strides(), input.bufferParam(),
            inputElements * sizeof(T), input.dims(), input.strides(),
            outputOffset,
            static_cast<af_dtype>(af::dtype_traits<T>::af_type));
        outputOffset[dim] += input.dims(dim);
    }
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
