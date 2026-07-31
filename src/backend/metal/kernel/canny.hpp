/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {
void launchMetalCannyNonmax(BufferParam output, size_t outputBytes,
                            const af::dim4& dims, const af::dim4& outputStrides,
                            BufferParam magnitude, size_t magnitudeBytes,
                            const af::dim4& magnitudeStrides,
                            BufferParam derivativeX, size_t derivativeXBytes,
                            const af::dim4& derivativeXStrides,
                            BufferParam derivativeY, size_t derivativeYBytes,
                            const af::dim4& derivativeYStrides);

void launchMetalCannyHysteresis(BufferParam output, size_t outputBytes,
                                const af::dim4& dims,
                                const af::dim4& outputStrides,
                                BufferParam strong, size_t strongBytes,
                                const af::dim4& strongStrides, BufferParam weak,
                                size_t weakBytes, const af::dim4& weakStrides);

template<typename T>
size_t cannyAccessibleBytes(CParam<T> input) {
    size_t elements = 1;
    for (int dimension = 0; dimension < 4; ++dimension) {
        elements += static_cast<size_t>(input.dims(dimension) - 1) *
                    static_cast<size_t>(input.strides(dimension));
    }
    return elements * sizeof(T);
}

template<typename T>
size_t cannyAccessibleBytes(Param<T> input) {
    size_t elements = 1;
    for (int dimension = 0; dimension < 4; ++dimension) {
        elements += static_cast<size_t>(input.dims(dimension) - 1) *
                    static_cast<size_t>(input.strides(dimension));
    }
    return elements * sizeof(T);
}

inline void nonMaxSuppressionMetal(Param<float> output, CParam<float> magnitude,
                                   CParam<float> derivativeX,
                                   CParam<float> derivativeY) {
    launchMetalCannyNonmax(output.bufferParam(), cannyAccessibleBytes(output),
                           output.dims(), output.strides(),
                           magnitude.bufferParam(),
                           cannyAccessibleBytes(magnitude), magnitude.strides(),
                           derivativeX.bufferParam(),
                           cannyAccessibleBytes(derivativeX),
                           derivativeX.strides(), derivativeY.bufferParam(),
                           cannyAccessibleBytes(derivativeY),
                           derivativeY.strides());
}

inline void edgeTrackingHysteresisMetal(Param<char> output, CParam<char> strong,
                                        CParam<char> weak) {
    launchMetalCannyHysteresis(output.bufferParam(),
                               cannyAccessibleBytes(output), output.dims(),
                               output.strides(), strong.bufferParam(),
                               cannyAccessibleBytes(strong), strong.strides(),
                               weak.bufferParam(), cannyAccessibleBytes(weak),
                               weak.strides());
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
