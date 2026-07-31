/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/traits.hpp>
#include <cstddef>

using af::dtype_traits;

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalRotate(af_dtype, af_interp_type) noexcept;
void launchMetalRotate(BufferParam, size_t, const af::dim4 &, const af::dim4 &,
                       BufferParam, size_t, const af::dim4 &, const af::dim4 &,
                       float, af_interp_type, af_dtype);
template<typename T>
void rotateMetal(Param<T> out, CParam<T> in, float theta,
                 af_interp_type method) {
    size_t n = 1;
    for (int i = 0; i < 4; ++i)
        n += size_t(in.dims(i) - 1) * size_t(in.strides(i));
    launchMetalRotate(out.bufferParam(),
                      size_t(out.dims().elements()) * sizeof(T), out.dims(),
                      out.strides(), in.bufferParam(), n * sizeof(T),
                      in.dims(), in.strides(), theta, method,
                      static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
