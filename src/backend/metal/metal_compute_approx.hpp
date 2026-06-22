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

void launchMetalApprox1Float(Param<float> output, CParam<float> input,
                             CParam<float> positions, int dimension,
                             float begin, float step, float offGrid,
                             af_interp_type method);
void launchMetalApprox2Float(Param<float> output, CParam<float> input,
                             CParam<float> x, int xDimension, float xBegin,
                             float xStep, CParam<float> y, int yDimension,
                             float yBegin, float yStep, float offGrid,
                             af_interp_type method);

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
