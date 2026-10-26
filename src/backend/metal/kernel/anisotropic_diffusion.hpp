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

#include <memory>
#include <utility>

namespace arrayfire {
namespace metal {
namespace kernel {

void launchMetalAnisotropicDiffusion(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam input, size_t inputBytes,
    const af::dim4& inputStrides, std::shared_ptr<MTL::Buffer> inputLifetime,
    float dt, float mct,
    af_flux_function flux, bool curvature);

inline void anisotropicDiffusionMetal(Param<float> output,
                                      CParam<float> input,
                                      std::shared_ptr<MTL::Buffer> inputLifetime,
                                      const float dt,
                                      const float mct,
                                      const af_flux_function flux,
                                      const bool curvature) {
    launchMetalAnisotropicDiffusion(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(float),
        output.dims(), output.strides(), input.bufferParam(),
        static_cast<size_t>(input.dims().elements()) * sizeof(float),
        input.strides(), std::move(inputLifetime), dt, mct, flux, curvature);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
