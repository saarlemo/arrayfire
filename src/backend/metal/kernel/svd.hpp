/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalSvd(af_dtype type) noexcept;

void launchMetalSvd(BufferParam singularValues, size_t singularBytes,
                    BufferParam u, size_t uBytes, BufferParam vt,
                    size_t vtBytes, BufferParam input, size_t inputBytes,
                    const af::dim4& inputDims, af_dtype type);

template<typename T, typename Tr>
void svdMetal(Array<Tr>& s, Array<T>& u, Array<T>& vt, Array<T>& input) {
    launchMetalSvd(
        s.bufferParam(), size_t(s.dims().elements()) * sizeof(Tr),
        u.bufferParam(), size_t(u.dims().elements()) * sizeof(T),
        vt.bufferParam(), size_t(vt.dims().elements()) * sizeof(T),
        input.bufferParam(), size_t(input.dims().elements()) * sizeof(T),
        input.dims(), static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T, typename Tr>
void svdInPlaceMetal(Array<Tr>& s, Array<T>& u, Array<T>& vt,
                     Array<T>& input) {
    svdMetal<T, Tr>(s, u, vt, input);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
