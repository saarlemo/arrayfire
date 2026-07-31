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

bool supportsMetalUnwrap(af_dtype type) noexcept;

void launchMetalUnwrap(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam input, size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides, dim_t windowX,
    dim_t windowY, dim_t strideX, dim_t strideY, dim_t paddingX, dim_t paddingY,
    dim_t dilationX, dim_t dilationY, unsigned columnDimension, af_dtype type);

template<typename T>
void unwrapMetal(Param<T> output, CParam<T> input, const dim_t windowX,
                 const dim_t windowY, const dim_t strideX, const dim_t strideY,
                 const dim_t paddingX, const dim_t paddingY,
                 const dim_t dilationX, const dim_t dilationY,
                 const int columnDimension) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalUnwrap(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.dims(), input.strides(), windowX, windowY, strideX, strideY,
        paddingX, paddingY, dilationX, dilationY,
        static_cast<unsigned>(columnDimension),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
