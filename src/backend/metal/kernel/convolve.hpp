/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement for ArrayFire can be found at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <af/defines.h>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalConvolve(af_dtype inputType, af_dtype filterType) noexcept;

void launchMetalConvolve(BufferParam output, const af::dim4& outputDims,
                         const af::dim4& outputStrides, BufferParam signal,
                         const af::dim4& signalDims,
                         const af::dim4& signalStrides, BufferParam filter,
                         const af::dim4& filterDims,
                         const af::dim4& filterStrides, AF_BATCH_KIND kind,
                         int rank, bool expand, af_dtype inputType,
                         af_dtype filterType);

template<typename InT, typename AccT>
void convolveMetal(Param<InT> output, CParam<InT> signal,
                   CParam<AccT> filter, const AF_BATCH_KIND kind,
                   const int rank, const bool expand) {
    launchMetalConvolve(
        output.bufferParam(), output.dims(), output.strides(),
        signal.bufferParam(), signal.dims(), signal.strides(),
        filter.bufferParam(), filter.dims(), filter.strides(), kind, rank,
        expand, static_cast<af_dtype>(af::dtype_traits<InT>::af_type),
        static_cast<af_dtype>(af::dtype_traits<AccT>::af_type));
}

bool supportsMetalSeparableConvolve(af_dtype inputType,
                                    af_dtype filterType) noexcept;

void launchMetalSeparableConvolve(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam temp,
    const af::dim4& tempDims, const af::dim4& tempStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam columnFilter, const af::dim4& columnFilterDims,
    const af::dim4& columnFilterStrides, BufferParam rowFilter,
    const af::dim4& rowFilterDims, const af::dim4& rowFilterStrides,
    bool expand, af_dtype inputType, af_dtype filterType);

template<typename InT, typename AccT>
void separableConvolveMetal(Param<InT> output, Param<InT> temp,
                            CParam<InT> signal, CParam<AccT> columnFilter,
                            CParam<AccT> rowFilter, const bool expand) {
    launchMetalSeparableConvolve(
        output.bufferParam(), output.dims(), output.strides(),
        temp.bufferParam(), temp.dims(), temp.strides(), signal.bufferParam(),
        signal.dims(), signal.strides(), columnFilter.bufferParam(),
        columnFilter.dims(), columnFilter.strides(), rowFilter.bufferParam(),
        rowFilter.dims(), rowFilter.strides(), expand,
        static_cast<af_dtype>(af::dtype_traits<InT>::af_type),
        static_cast<af_dtype>(af::dtype_traits<AccT>::af_type));
}

bool supportsMetalConvolveNN(af_dtype type) noexcept;

void launchMetalConvolveNN(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam filter, const af::dim4& filterDims,
    const af::dim4& filterStrides, const af::dim4& stride,
    const af::dim4& padding, const af::dim4& dilation, af_dtype type);

void launchMetalConvolveNNDataGradient(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam incomingGradient,
    const af::dim4& incomingGradientDims,
    const af::dim4& incomingGradientStrides, BufferParam filter,
    const af::dim4& filterDims, const af::dim4& filterStrides,
    const af::dim4& stride, const af::dim4& padding,
    const af::dim4& dilation, af_dtype type);

void launchMetalConvolveNNFilterGradient(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam incomingGradient, const af::dim4& incomingGradientDims,
    const af::dim4& incomingGradientStrides, const af::dim4& stride,
    const af::dim4& padding, const af::dim4& dilation, af_dtype type);

template<typename T>
void convolveNNMetal(Param<T> output, CParam<T> signal, CParam<T> filter,
                     const af::dim4 stride, const af::dim4 padding,
                     const af::dim4 dilation) {
    launchMetalConvolveNN(
        output.bufferParam(), output.dims(), output.strides(),
        signal.bufferParam(), signal.dims(), signal.strides(),
        filter.bufferParam(), filter.dims(), filter.strides(), stride, padding,
        dilation, static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void convolveNNDataGradientMetal(
    Param<T> output, CParam<T> incomingGradient, CParam<T> filter,
    const af::dim4 stride, const af::dim4 padding, const af::dim4 dilation) {
    launchMetalConvolveNNDataGradient(
        output.bufferParam(), output.dims(), output.strides(),
        incomingGradient.bufferParam(), incomingGradient.dims(),
        incomingGradient.strides(), filter.bufferParam(), filter.dims(),
        filter.strides(), stride, padding, dilation,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void convolveNNFilterGradientMetal(
    Param<T> output, CParam<T> signal, CParam<T> incomingGradient,
    const af::dim4 stride, const af::dim4 padding, const af::dim4 dilation) {
    launchMetalConvolveNNFilterGradient(
        output.bufferParam(), output.dims(), output.strides(),
        signal.bufferParam(), signal.dims(), signal.strides(),
        incomingGradient.bufferParam(), incomingGradient.dims(),
        incomingGradient.strides(), stride, padding, dilation,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
