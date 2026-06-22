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

void launchMetalAnisotropicDiffusion(void* inout, size_t bytes,
                                     const af::dim4& dims,
                                     const af::dim4& strides, float dt,
                                     float mct, af_flux_function flux,
                                     bool curvature);

inline void anisotropicDiffusionMetal(Param<float> inout, const float dt,
                                      const float mct,
                                      const af_flux_function flux,
                                      const bool curvature) {
    launchMetalAnisotropicDiffusion(
        inout.get(),
        static_cast<size_t>(inout.dims().elements()) * sizeof(float),
        inout.dims(), inout.strides(), dt, mct, flux, curvature);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
