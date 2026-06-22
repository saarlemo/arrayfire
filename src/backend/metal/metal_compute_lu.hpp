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
bool supportsMetalLu(af_dtype) noexcept;
void launchMetalLuPart(void *, size_t, const af::dim4 &, const af::dim4 &,
                       const void *, size_t, const af::dim4 &, const af::dim4 &,
                       bool, af_dtype);
void launchMetalConvertPivot(void *, size_t, const af::dim4 &, const void *,
                             size_t, const af::dim4 &);
template<typename T>
void luSplitMetal(Param<T> l, Param<T> u, CParam<T> in) {
    size_t n = 1;
    for (int i = 0; i < 4; ++i)
        n += size_t(in.dims(i) - 1) * size_t(in.strides(i));
    auto t = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    launchMetalLuPart(l.get(), size_t(l.dims().elements()) * sizeof(T),
                      l.dims(), l.strides(), in.get(), n * sizeof(T), in.dims(),
                      in.strides(), true, t);
    launchMetalLuPart(u.get(), size_t(u.dims().elements()) * sizeof(T),
                      u.dims(), u.strides(), in.get(), n * sizeof(T), in.dims(),
                      in.strides(), false, t);
}
inline void convertPivotMetal(Param<int> p, Param<int> pivot) {
    launchMetalConvertPivot(p.get(), size_t(p.dims().elements()) * sizeof(int),
                            p.dims(), pivot.get(),
                            size_t(pivot.dims().elements()) * sizeof(int),
                            pivot.dims());
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
