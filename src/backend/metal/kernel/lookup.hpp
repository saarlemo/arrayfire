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

bool supportsMetalLookup(af_dtype inputType, af_dtype indexType) noexcept;

void launchMetalLookup(BufferParam output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, BufferParam input,
                       size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, BufferParam indices,
                       size_t indexBytes, unsigned dimension,
                       af_dtype inputType, af_dtype indexType);

template<typename InT, typename IndexT>
void lookupMetal(Param<InT> output, CParam<InT> input, CParam<IndexT> indices,
                 const unsigned dimension) {
    size_t inputElements = 1;
    size_t indexElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
        indexElements += static_cast<size_t>(indices.dims(i) - 1) *
                         static_cast<size_t>(indices.strides(i));
    }
    launchMetalLookup(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(InT),
        output.dims(), output.strides(), input.bufferParam(),
        inputElements * sizeof(InT), input.dims(), input.strides(),
        indices.bufferParam(), indexElements * sizeof(IndexT), dimension,
        static_cast<af_dtype>(af::dtype_traits<InT>::af_type),
        static_cast<af_dtype>(af::dtype_traits<IndexT>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
