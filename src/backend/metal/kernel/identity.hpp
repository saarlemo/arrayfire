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

bool supportsMetalIdentity(af_dtype type) noexcept;

void launchMetalIdentity(BufferParam output, size_t bytes,
                         const af::dim4& dims,
                         const af::dim4& strides, af_dtype type);

template<typename T>
void identityMetal(Param<T> output) {
    launchMetalIdentity(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
