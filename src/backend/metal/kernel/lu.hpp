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

bool supportsMetalLu(af_dtype) noexcept;
bool supportsMetalLuFactor(af_dtype) noexcept;
void launchMetalLuFactor(BufferParam, size_t, const af::dim4 &,
                         const af::dim4 &, BufferParam, size_t,
                         const af::dim4 &, af_dtype);
void launchMetalLuPart(BufferParam, size_t, const af::dim4 &, const af::dim4 &,
                       BufferParam, size_t, const af::dim4 &, const af::dim4 &,
                       bool, af_dtype);
void launchMetalConvertPivot(BufferParam, size_t, const af::dim4 &, BufferParam,
                             size_t, const af::dim4 &);
template<typename T>
void luFactorMetal(Param<T> in, Param<int> pivot) {
    const auto type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    launchMetalLuFactor(
        in.bufferParam(), size_t(in.dims().elements()) * sizeof(T), in.dims(),
        in.strides(), pivot.bufferParam(),
        size_t(pivot.dims().elements()) * sizeof(int), pivot.dims(), type);
}

template<typename T>
void luSplitMetal(Param<T> l, Param<T> u, CParam<T> in) {
    size_t n = 1;
    for (int i = 0; i < 4; ++i)
        n += size_t(in.dims(i) - 1) * size_t(in.strides(i));
    auto t = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    launchMetalLuPart(l.bufferParam(), size_t(l.dims().elements()) * sizeof(T),
                      l.dims(), l.strides(), in.bufferParam(), n * sizeof(T),
                      in.dims(), in.strides(), true, t);
    launchMetalLuPart(u.bufferParam(), size_t(u.dims().elements()) * sizeof(T),
                      u.dims(), u.strides(), in.bufferParam(), n * sizeof(T),
                      in.dims(), in.strides(), false, t);
}
inline void convertPivotMetal(Param<int> p, Param<int> pivot) {
    launchMetalConvertPivot(p.bufferParam(),
                            size_t(p.dims().elements()) * sizeof(int), p.dims(),
                            pivot.bufferParam(),
                            size_t(pivot.dims().elements()) * sizeof(int),
                            pivot.dims());
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
