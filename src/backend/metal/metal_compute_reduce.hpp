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
void reduceAddMetal(Param<float> output, CParam<float> input, int dimension,
                    bool changeNan, double nanValue);
}
}  // namespace metal
}  // namespace arrayfire
