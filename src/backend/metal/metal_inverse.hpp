/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * Distributed under the 3-clause BSD license.
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <af/dim4.hpp>

namespace MTL {
class Buffer;
}

namespace arrayfire {
namespace metal {

void inverseMatrix(MTL::Buffer* output, MTL::Buffer* input,
                   const af::dim4& dims, af_dtype type);

}  // namespace metal
}  // namespace arrayfire
