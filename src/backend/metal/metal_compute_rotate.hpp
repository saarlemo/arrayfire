/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/
#pragma once
#include <Param.hpp>
#include <af/traits.hpp>
#include <cstddef>
namespace arrayfire {
namespace metal {
namespace kernel {
bool supportsMetalRotate(af_dtype, af_interp_type) noexcept;
void launchMetalRotate(void *, size_t, const af::dim4 &, const af::dim4 &,
                       const void *, size_t, const af::dim4 &, const af::dim4 &,
                       float, af_interp_type, af_dtype);
template<typename T>
void rotateMetal(Param<T> out, CParam<T> in, float theta,
                 af_interp_type method) {
    size_t n = 1;
    for (int i = 0; i < 4; ++i)
        n += size_t(in.dims(i) - 1) * size_t(in.strides(i));
    launchMetalRotate(out.get(), size_t(out.dims().elements()) * sizeof(T),
                      out.dims(), out.strides(), in.get(), n * sizeof(T),
                      in.dims(), in.strides(), theta, method,
                      static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
