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
#include <af/dim4.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalCholesky(af_dtype type) noexcept;

void launchMetalCholesky(BufferParam inout, const af::dim4& dims,
                         const af::dim4& strides, bool upper, af_dtype type,
                         int* info);

template<typename T>
void choleskyMetal(Param<T> inout, const bool upper, int* info) {
    launchMetalCholesky(
        inout.bufferParam(), inout.dims(), inout.strides(), upper,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type), info);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
