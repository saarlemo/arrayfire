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

bool supportsMetalIReduce(af_dtype type) noexcept;
void launchMetalIReduceAll(BufferParam output, size_t outputBytes,
                           BufferParam locations, size_t locationsBytes,
                           BufferParam input, size_t inputBytes,
                           const af::dim4& inputDims,
                           const af::dim4& inputStrides, bool isMax,
                           af_dtype type);
void launchMetalIReduce(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam locations, size_t locationsBytes,
    BufferParam input, size_t inputBytes, const af::dim4& inputDims,
    const af::dim4& inputStrides, int dimension, bool isMax, af_dtype type);
void launchMetalRReduce(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam locations, size_t locationsBytes,
    BufferParam input, size_t inputBytes, const af::dim4& inputDims,
    const af::dim4& inputStrides, int dimension, BufferParam rlen,
    size_t rlenBytes, const af::dim4& rlenDims, const af::dim4& rlenStrides,
    bool isMax, af_dtype type);

template<typename T>
void ireduceMetalTyped(Param<T> output, Param<uint> locations,
                       CParam<T> input, const int dimension,
                       const bool isMax) {
    launchMetalIReduce(
        output.bufferParam(), static_cast<size_t>(output.dims().elements()) *
                                   sizeof(T),
        output.dims(), output.strides(), locations.bufferParam(),
        static_cast<size_t>(locations.dims().elements()) * sizeof(uint),
        input.bufferParam(), sizeof(T), input.dims(), input.strides(), dimension,
        isMax, static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void ireduceAllMetalTyped(Param<T> output, Param<uint> locations,
                          CParam<T> input, const bool isMax) {
    launchMetalIReduceAll(
        output.bufferParam(), static_cast<size_t>(output.dims().elements()) *
                                   sizeof(T),
        locations.bufferParam(),
        static_cast<size_t>(locations.dims().elements()) * sizeof(uint),
        input.bufferParam(), sizeof(T), input.dims(), input.strides(), isMax,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<af_op_t op, typename T>
void rreduceMetalTyped(Param<T> output, Param<uint> locations,
                       CParam<T> input, const int dimension,
                       CParam<uint> rlen) {
    launchMetalRReduce(
        output.bufferParam(), static_cast<size_t>(output.dims().elements()) *
                                   sizeof(T),
        output.dims(), output.strides(), locations.bufferParam(),
        static_cast<size_t>(locations.dims().elements()) * sizeof(uint),
        input.bufferParam(), sizeof(T), input.dims(), input.strides(), dimension,
        rlen.bufferParam(), static_cast<size_t>(rlen.dims().elements()) *
                                 sizeof(uint),
        rlen.dims(), rlen.strides(), op == af_max_t,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
