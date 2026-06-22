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
namespace arrayfire {
namespace metal {
namespace kernel {
void harrisSecondOrderMetal(Param<float> ixx, Param<float> ixy,
                            Param<float> iyy, CParam<float> ix,
                            CParam<float> iy);
void harrisResponseMetal(Param<float> output, unsigned rows, unsigned columns,
                         CParam<float> ixx, CParam<float> ixy,
                         CParam<float> iyy, float k, unsigned border);
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
