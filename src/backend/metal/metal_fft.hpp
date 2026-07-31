/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * Distributed under the 3-clause BSD license.
 ********************************************************/

#pragma once

#include <af/dim4.hpp>

namespace MTL {
class Buffer;
}

namespace arrayfire {
namespace metal {

// MPSGraph's FFT operations are exposed through Objective-C. These small
// wrappers keep that boundary out of the backend's normal C++ translation
// units while retaining device-resident input and output buffers.
void fftComplexFloat(MTL::Buffer* output, MTL::Buffer* input,
                     const af::dim4& dims, int rank, bool inverse);

void fftRealToComplexFloat(MTL::Buffer* output, MTL::Buffer* input,
                           const af::dim4& inputDims, int rank);

void fftComplexToRealFloat(MTL::Buffer* output, MTL::Buffer* input,
                           const af::dim4& outputDims, int rank);

}  // namespace metal
}  // namespace arrayfire
