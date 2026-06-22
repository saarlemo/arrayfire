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

bool supportsMetalHistogram(af_dtype type) noexcept;

void launchMetalHistogram(void* output, size_t outputBytes,
                          const af::dim4& outputStrides, const void* input,
                          size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, unsigned bins,
                          double minValue, double maxValue, bool linear,
                          af_dtype type);

template<typename T>
void histogramMetal(Param<uint> output, CParam<T> input, const unsigned bins,
                    const double minValue, const double maxValue,
                    const bool linear) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalHistogram(
        output.get(),
        static_cast<size_t>(output.dims().elements()) * sizeof(uint),
        output.strides(), input.get(), inputElements * sizeof(T), input.dims(),
        input.strides(), bins, minValue, maxValue, linear,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
