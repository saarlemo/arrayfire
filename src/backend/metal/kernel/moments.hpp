/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/traits.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalMoments(af_dtype type) noexcept;

void launchMetalMoments(BufferParam output, size_t outputBytes,
                        const af::dim4& outputStrides, BufferParam input,
                        size_t inputBytes, const af::dim4& inputDims,
                        const af::dim4& inputStrides, af_moment_type moment,
                        af_dtype type);

template<typename T>
void momentsMetal(Param<float> output, CParam<T> input,
                  const af_moment_type moment) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalMoments(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(float),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.dims(),
        input.strides(), moment,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
