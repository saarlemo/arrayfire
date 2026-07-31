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

bool supportsMetalIir(af_dtype type) noexcept;

void launchMetalIir(BufferParam output, size_t outputBytes,
                    const af::dim4& outputDims,
                    const af::dim4& outputStrides, BufferParam coefficients,
                    size_t coefficientBytes,
                    const af::dim4& coefficientStrides, BufferParam feedback,
                    size_t feedbackBytes, const af::dim4& feedbackDims,
                    const af::dim4& feedbackStrides, bool feedbackBatched,
                    af_dtype type);

template<typename T>
void iirMetal(Param<T> output, Param<T> coefficients, CParam<T> feedback) {
    size_t coefficientElements = 1, feedbackElements = 1;
    for (int i = 0; i < 4; ++i) {
        coefficientElements += static_cast<size_t>(coefficients.dims(i) - 1) *
                               static_cast<size_t>(coefficients.strides(i));
        feedbackElements += static_cast<size_t>(feedback.dims(i) - 1) *
                            static_cast<size_t>(feedback.strides(i));
    }
    const bool feedbackBatched =
        feedback.dims(1) > 1 || feedback.dims(2) > 1 || feedback.dims(3) > 1;
    launchMetalIir(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), coefficients.bufferParam(),
        coefficientElements * sizeof(T), coefficients.strides(),
        feedback.bufferParam(), feedbackElements * sizeof(T), feedback.dims(),
        feedback.strides(), feedbackBatched,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
