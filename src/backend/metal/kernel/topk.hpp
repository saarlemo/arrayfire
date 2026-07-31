/*******************************************************
 * Copyright (c) 2026, ArrayFire
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

bool supportsMetalTopK(af_dtype type) noexcept;

void launchMetalTopK(BufferParam values, BufferParam indices,
                     const af::dim4& outputStrides, BufferParam input,
                     const af::dim4& inputDims,
                     const af::dim4& inputStrides, int k, bool ascending,
                     af_dtype type);

template<typename T>
void topKMetal(Param<T> values, Param<uint> indices, CParam<T> input,
               const int k, const bool ascending) {
    launchMetalTopK(
        values.bufferParam(), indices.bufferParam(), values.strides(),
        input.bufferParam(), input.dims(), input.strides(), k, ascending,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
