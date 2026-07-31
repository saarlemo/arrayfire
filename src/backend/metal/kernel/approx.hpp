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

#include <af/defines.h>
#include <af/traits.hpp>

#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalApprox(af_dtype valueType,
                         af_dtype positionType) noexcept;

void launchMetalApprox1(BufferParam output, size_t outputBytes,
                        const af::dim4& outputDims,
                        const af::dim4& outputStrides, dim_t outputOffset,
                        BufferParam input,
                        size_t inputBytes, const af::dim4& inputDims,
                        const af::dim4& inputStrides, dim_t inputOffset,
                        BufferParam positions,
                        size_t positionBytes, const af::dim4& positionDims,
                        const af::dim4& positionStrides, dim_t positionOffset,
                        int dimension,
                        float begin, float step, float offGrid,
                        af_interp_type method, af_dtype valueType);

void launchMetalApprox2(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, dim_t outputOffset, BufferParam input,
    size_t inputBytes, const af::dim4& inputDims,
    const af::dim4& inputStrides, dim_t inputOffset, BufferParam x,
    size_t xBytes, const af::dim4& positionDims,
    const af::dim4& xStrides, dim_t xOffset, BufferParam y, size_t yBytes,
    const af::dim4& yStrides, dim_t yOffset, int xDimension, float xBegin,
    float xStep, int yDimension, float yBegin, float yStep, float offGrid,
    af_interp_type method, af_dtype valueType);

template<typename T>
void approx1Metal(Param<T> output, CParam<T> input,
                  CParam<float> positions, const int dimension,
                  const float begin, const float step, const float offGrid,
                  const af_interp_type method) {
    launchMetalApprox1(
        {output.getBuffer(), 0},
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), output.getOffset(),
        {input.getBuffer(), 0},
        static_cast<size_t>(input.dims().elements()) * sizeof(T), input.dims(),
        input.strides(), input.getOffset(), {positions.getBuffer(), 0},
        static_cast<size_t>(positions.dims().elements()) * sizeof(float),
        positions.dims(), positions.strides(), positions.getOffset(),
        dimension, begin, step, offGrid, method,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void approx2Metal(Param<T> output, CParam<T> input, CParam<float> x,
                  const int xDimension, const float xBegin, const float xStep,
                  CParam<float> y, const int yDimension, const float yBegin,
                  const float yStep, const float offGrid,
                  const af_interp_type method) {
    launchMetalApprox2(
        {output.getBuffer(), 0},
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), output.getOffset(),
        {input.getBuffer(), 0},
        static_cast<size_t>(input.dims().elements()) * sizeof(T), input.dims(),
        input.strides(), input.getOffset(), {x.getBuffer(), 0},
        static_cast<size_t>(x.dims().elements()) * sizeof(float), x.dims(),
        x.strides(), x.getOffset(), {y.getBuffer(), 0},
        static_cast<size_t>(y.dims().elements()) * sizeof(float), y.strides(),
        y.getOffset(), xDimension, xBegin, xStep, yDimension, yBegin, yStep,
        offGrid, method,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
