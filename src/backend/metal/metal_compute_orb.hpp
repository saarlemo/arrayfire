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
void orbCentroidMetal(const float* x, const float* y, float* orientation,
                      unsigned features, CParam<float> image,
                      unsigned patchSize);
}
}  // namespace metal
}  // namespace arrayfire
