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

namespace arrayfire {
namespace metal {
namespace kernel {

void launchSusanResponse(BufferParam output, size_t outputBytes,
                         BufferParam input, size_t inputBytes,
                         const af::dim4& outputDims, unsigned rows,
                         unsigned columns, unsigned radius,
                         float differenceThreshold, float geometricThreshold,
                         unsigned border, af_dtype type);

void launchSusanNonMax(BufferParam xOutput, BufferParam yOutput,
                       BufferParam responseOutput, BufferParam response,
                       unsigned* count, unsigned rows, unsigned columns,
                       unsigned border, unsigned maxCorners, af_dtype type);

template<typename T>
void susanResponseMetal(Param<T> output, CParam<T> input, unsigned rows,
                        unsigned columns, unsigned radius,
                        float differenceThreshold, float geometricThreshold,
                        unsigned border) {
    launchSusanResponse(
        output.bufferParam(), size_t(output.dims().elements()) * sizeof(T),
        input.bufferParam(), size_t(input.dims().elements()) * sizeof(T),
        output.dims(), rows, columns, radius, differenceThreshold,
        geometricThreshold, border,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void susanNonMax(Param<float> xOutput, Param<float> yOutput,
                 Param<float> responseOutput, CParam<T> response,
                 unsigned* count, unsigned rows, unsigned columns,
                 unsigned border, unsigned maxCorners) {
    launchSusanNonMax(
        xOutput.bufferParam(), yOutput.bufferParam(), responseOutput.bufferParam(),
        response.bufferParam(), count, rows, columns, border, maxCorners,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
