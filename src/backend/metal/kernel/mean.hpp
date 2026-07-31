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

bool supportsMetalMean(af_dtype inputType, af_dtype outputType) noexcept;
bool supportsMetalMeanWeighted(af_dtype valueType,
                               af_dtype weightType) noexcept;

void launchMetalMean(BufferParam output, size_t outputBytes,
                     const af::dim4& outputDims,
                     const af::dim4& outputStrides, BufferParam input,
                     size_t inputBytes, const af::dim4& inputDims,
                     const af::dim4& inputStrides, int dimension,
                     bool reduceAll, af_dtype inputType, af_dtype outputType);

void launchMetalMeanWeighted(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam input, size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides,
    BufferParam weights, size_t weightBytes,
    const af::dim4& weightStrides, int dimension, bool reduceAll,
    af_dtype valueType, af_dtype weightType);

template<typename Ti, typename To>
void meanMetal(Param<To> output, CParam<Ti> input, const int dimension,
               const bool reduceAll) {
    launchMetalMean(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), input.bufferParam(), sizeof(Ti),
        input.dims(), input.strides(), dimension, reduceAll,
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type),
        static_cast<af_dtype>(af::dtype_traits<To>::af_type));
}

template<typename T, typename Tw>
void meanWeightedMetal(Param<T> output, CParam<T> input, CParam<Tw> weights,
                       const int dimension, const bool reduceAll) {
    launchMetalMeanWeighted(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.bufferParam(), sizeof(T),
        input.dims(), input.strides(), weights.bufferParam(), sizeof(Tw),
        weights.strides(), dimension, reduceAll,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type),
        static_cast<af_dtype>(af::dtype_traits<Tw>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
