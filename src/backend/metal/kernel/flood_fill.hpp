/*******************************************************
 * Copyright (c) 2019, ArrayFire
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

void launchMetalFloodFill(BufferParam output, size_t outputBytes,
                          const af::dim4& imageDims,
                          const af::dim4& imageStrides, BufferParam image,
                          size_t imageBytes, BufferParam seedX,
                          size_t seedXBytes, const af::dim4& seedDims,
                          const af::dim4& seedXStrides, BufferParam seedY,
                          size_t seedYBytes, const af::dim4& seedYStrides,
                          const void* newValue, const void* lower,
                          const void* upper, size_t valueBytes, af_dtype type);

template<typename T>
void floodFillMetal(Param<T> output, CParam<T> image, CParam<uint> seedX,
                    CParam<uint> seedY, T newValue, T lower, T upper) {
    const auto accessibleElements = [](const af::dim4& dims,
                                       const af::dim4& strides) {
        size_t elements = 1;
        for (int dimension = 0; dimension < 4; ++dimension) {
            elements += static_cast<size_t>(dims[dimension] - 1) *
                        static_cast<size_t>(strides[dimension]);
        }
        return elements;
    };

    launchMetalFloodFill(
        output.bufferParam(),
        accessibleElements(output.dims(), output.strides()) * sizeof(T),
        image.dims(), image.strides(), image.bufferParam(),
        accessibleElements(image.dims(), image.strides()) * sizeof(T),
        seedX.bufferParam(),
        accessibleElements(seedX.dims(), seedX.strides()) * sizeof(uint),
        seedX.dims(), seedX.strides(), seedY.bufferParam(),
        accessibleElements(seedY.dims(), seedY.strides()) * sizeof(uint),
        seedY.strides(), &newValue, &lower, &upper, sizeof(T),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
