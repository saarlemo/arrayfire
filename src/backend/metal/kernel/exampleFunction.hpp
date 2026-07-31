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

#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalExampleFunction(af_dtype type) noexcept;
void launchMetalExampleFunction(
    BufferParam output, size_t outputBytes, const af::dim4& dims,
    const af::dim4& outputStrides, BufferParam left, size_t leftBytes,
    const af::dim4& leftStrides, BufferParam right, size_t rightBytes,
    const af::dim4& rightStrides, af_dtype type);

template<typename T>
void exampleFunctionMetal(Param<T> output, CParam<T> left, CParam<T> right) {
    launchMetalExampleFunction(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), left.bufferParam(), sizeof(T),
        left.strides(), right.bufferParam(), sizeof(T), right.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
