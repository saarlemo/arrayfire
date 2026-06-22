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

bool supportsMetalTransform(af_dtype type, af_interp_type method,
                            const af::dim4& inputDims,
                            const af::dim4& transformDims) noexcept;

void launchMetalTransform(void* output, size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, const void* transform,
                          size_t transformBytes, af_interp_type method,
                          bool inverse, bool perspective, af_dtype type);

template<typename T>
void transformMetal(Param<T> output, CParam<T> input, CParam<float> transform,
                    const af_interp_type method, const bool inverse,
                    const bool perspective) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalTransform(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), transform.get(),
        static_cast<size_t>(transform.dims().elements()) * sizeof(float),
        method, inverse, perspective,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
