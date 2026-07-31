/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

bool supportsMetalGradient(af_dtype type) noexcept;

void launchMetalGradient(BufferParam gradient0,
                         const af::dim4& gradient0Strides,
                         BufferParam gradient1,
                         const af::dim4& gradient1Strides,
                         size_t outputBytes, BufferParam input,
                         size_t inputBytes, const af::dim4& inputDims,
                         const af::dim4& inputStrides, af_dtype type);

template<typename T>
void gradientMetal(Param<T> gradient0, Param<T> gradient1, CParam<T> input) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalGradient(
        gradient0.bufferParam(), gradient0.strides(), gradient1.bufferParam(),
        gradient1.strides(),
        static_cast<size_t>(gradient0.dims().elements()) * sizeof(T),
        input.bufferParam(), inputElements * sizeof(T), input.dims(),
        input.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
