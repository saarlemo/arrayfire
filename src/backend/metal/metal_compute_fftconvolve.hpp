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
void fftConvolveMultiplyMetal(Param<float> packed, const af::dim4& signalDims,
                              const af::dim4& signalStrides,
                              const af::dim4& filterDims,
                              const af::dim4& filterStrides, AF_BATCH_KIND kind,
                              dim_t offset);
}
}  // namespace metal
}  // namespace arrayfire
