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

bool supportsMetalMeanshift(af_dtype type) noexcept;

void launchMetalMeanshift(BufferParam output, size_t outputBytes,
                          const af::dim4& dims,
                          const af::dim4& outputStrides, BufferParam input,
                          size_t inputBytes, const af::dim4& inputStrides,
                          float spatialSigma, float chromaticSigma,
                          unsigned iterations, bool color, af_dtype type);

template<typename T>
void meanshiftMetal(Param<T> output, CParam<T> input, const float spatialSigma,
                    const float chromaticSigma, const unsigned iterations,
                    const bool color) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalMeanshift(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), input.bufferParam(), inputElements * sizeof(T),
        input.strides(), spatialSigma, chromaticSigma, iterations, color,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
