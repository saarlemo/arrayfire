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

bool supportsMetalRegions(af_dtype type) noexcept;

void launchMetalRegions(BufferParam output, size_t outputBytes,
                        const af::dim4& dims, BufferParam input,
                        const af::dim4& inputStrides,
                        af_connectivity connectivity, af_dtype type);

template<typename T>
void regionsMetal(Param<T> output, CParam<char> input,
                  const af_connectivity connectivity) {
    launchMetalRegions(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), input.bufferParam(), input.strides(), connectivity,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
