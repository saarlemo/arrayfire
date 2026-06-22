/*******************************************************
 * Copyright (c) 2026, ArrayFire
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

void launchMetalCannyNonmax(void* output, size_t outputBytes,
                            const af::dim4& dims, const af::dim4& outputStrides,
                            const void* magnitude, size_t magnitudeBytes,
                            const af::dim4& magnitudeStrides,
                            const void* derivativeX, size_t derivativeXBytes,
                            const af::dim4& derivativeXStrides,
                            const void* derivativeY, size_t derivativeYBytes,
                            const af::dim4& derivativeYStrides);

void launchMetalCannyHysteresis(void* output, size_t outputBytes,
                                const af::dim4& dims,
                                const af::dim4& outputStrides,
                                const void* strong, size_t strongBytes,
                                const af::dim4& strongStrides, const void* weak,
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
    launchMetalCannyNonmax(output.get(), cannyAccessibleBytes(output),
                           output.dims(), output.strides(), magnitude.get(),
                           cannyAccessibleBytes(magnitude), magnitude.strides(),
                           derivativeX.get(), cannyAccessibleBytes(derivativeX),
                           derivativeX.strides(), derivativeY.get(),
                           cannyAccessibleBytes(derivativeY),
                           derivativeY.strides());
}

inline void edgeTrackingHysteresisMetal(Param<char> output, CParam<char> strong,
                                        CParam<char> weak) {
    launchMetalCannyHysteresis(output.get(), cannyAccessibleBytes(output),
                               output.dims(), output.strides(), strong.get(),
                               cannyAccessibleBytes(strong), strong.strides(),
                               weak.get(), cannyAccessibleBytes(weak),
                               weak.strides());
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
