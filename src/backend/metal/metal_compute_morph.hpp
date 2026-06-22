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

#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalMorph(af_dtype type) noexcept;

void launchMetalMorph(void* output, size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      const void* input, size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      const void* mask, size_t maskBytes,
                      const af::dim4& maskDims, const af::dim4& maskStrides,
                      bool dilation, bool volume, af_dtype type);

template<typename T>
void morphMetal(Param<T> output, CParam<T> input, CParam<T> mask,
                const bool dilation, const bool volume) {
    size_t inputElements = 1, maskElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
        maskElements += static_cast<size_t>(mask.dims(i) - 1) *
                        static_cast<size_t>(mask.strides(i));
    }
    launchMetalMorph(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), mask.get(), maskElements * sizeof(T),
        mask.dims(), mask.strides(), dilation, volume,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
