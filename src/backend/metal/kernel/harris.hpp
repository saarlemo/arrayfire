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
#include <utility.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

void harrisSecondOrderMetal(Param<float> ixx, Param<float> ixy,
                            Param<float> iyy, CParam<float> ix,
                            CParam<float> iy);
void harrisResponseMetal(Param<float> output, unsigned rows, unsigned columns,
                         CParam<float> ixx, CParam<float> ixy,
                         CParam<float> iyy, float k, unsigned border);
void harrisNonMaxMetal(CParam<float> response, Param<float> xOutput,
                       Param<float> yOutput, Param<float> responseOutput,
                       unsigned* count, unsigned rows, unsigned columns,
                       float minResponse, unsigned border,
                       unsigned maxCorners);
void harrisKeepCornersMetal(Param<float> xOutput, Param<float> yOutput,
                            Param<float> responseOutput, CParam<float> xInput,
                            CParam<float> yInput, CParam<float> responseInput,
                            CParam<unsigned> responseIndex,
                            unsigned corners);
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
