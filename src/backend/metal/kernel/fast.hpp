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
#include <af/dim4.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

void launchMetalFast(
    BufferParam input, size_t inputBytes, const af::dim4& inputDims,
    BufferParam scoreImage, BufferParam x, BufferParam y, BufferParam scores,
    unsigned* count, float threshold, unsigned arcLength, bool nonmax,
    unsigned maxFeatures, unsigned edge, af_dtype inputType);

template<typename T>
void fastLocateMetal(CParam<T> input, Param<float> scoreImage, Param<float> x,
                     Param<float> y, Param<float> scores, unsigned* count,
                     const float threshold, const unsigned arcLength,
                     const bool nonmax, const unsigned maxFeatures,
                     const unsigned edge) {
    launchMetalFast(
        input.bufferParam(),
        static_cast<size_t>(input.dims().elements()) * sizeof(T), input.dims(),
        scoreImage.bufferParam(), x.bufferParam(), y.bufferParam(),
        scores.bufferParam(), count, threshold, arcLength, nonmax, maxFeatures,
        edge, static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
void fastNonMaxMetal(CParam<float> score, CParam<float> xInput,
                     CParam<float> yInput, Param<float> xOutput,
                     Param<float> yOutput, Param<float> scoreOutput,
                     unsigned* count, unsigned totalFeatures, unsigned edge);
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
