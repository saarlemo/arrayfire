/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/traits.hpp>
#include <cstddef>
#include <memory>

namespace arrayfire {
namespace metal {
namespace kernel {

void launchMetalFloodFillStateInit(BufferParam state, size_t stateBytes,
                                   size_t stateElements);
void launchMetalFloodFillSeedState(
    BufferParam state, size_t stateBytes, BufferParam seedX,
    size_t seedXBytes, BufferParam seedY, size_t seedYBytes,
    const af::dim4& imageDims, const af::dim4& imageStrides,
    const af::dim4& seedDims, const af::dim4& seedXStrides,
    const af::dim4& seedYStrides);
void launchMetalFloodFillStep(
    BufferParam state, size_t stateBytes, BufferParam image,
    size_t imageBytes, BufferParam changed, size_t changedBytes,
    const af::dim4& imageDims, const af::dim4& imageStrides, const void* lower,
    const void* upper, size_t valueBytes, af_dtype type);
void launchMetalFloodFillFinalize(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    BufferParam state, size_t stateBytes,
    std::shared_ptr<MTL::Buffer> stateLifetime, const void* newValue,
    size_t valueBytes, af_dtype type);
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
