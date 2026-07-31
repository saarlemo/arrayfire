/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>

namespace arrayfire {
namespace metal {

struct KParam {
    dim_t dims[4];
    dim_t strides[4];
    dim_t offset;

    dim_t *dims_ptr() { return dims; }
    dim_t *strides_ptr() { return strides; }
};

}  // namespace metal
}  // namespace arrayfire
