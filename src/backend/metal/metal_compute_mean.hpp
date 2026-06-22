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

void launchMetalMeanFloat(void* output, size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, int dimension);

inline void meanMetal(Param<float> output, CParam<float> input,
                      const int dimension) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalMeanFloat(
        output.get(),
        static_cast<size_t>(output.dims().elements()) * sizeof(float),
        output.dims(), output.strides(), input.get(),
        inputElements * sizeof(float), input.dims(), input.strides(),
        dimension);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
