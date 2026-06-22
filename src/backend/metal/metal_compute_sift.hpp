/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
namespace arrayfire {
namespace metal {
namespace kernel {
void siftSubtractMetal(Array<float>& output, const Array<float>& first,
                       const Array<float>& second);
}
}  // namespace metal
}  // namespace arrayfire
