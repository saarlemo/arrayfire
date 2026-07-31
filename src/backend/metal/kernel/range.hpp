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

bool supportsMetalRange(af_dtype type) noexcept;

void launchMetalRange(BufferParam output, size_t bytes, const af::dim4& dims,
                      const af::dim4& strides, unsigned sequenceDimension,
                      af_dtype type);

template<typename T>
void rangeMetal(Param<T> output, const unsigned sequenceDimension) {
    launchMetalRange(output.bufferParam(),
                     static_cast<size_t>(output.dims().elements()) * sizeof(T),
                     output.dims(), output.strides(), sequenceDimension,
                     static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
