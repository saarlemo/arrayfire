/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct SiftParams { ulong elements; };

kernel void sift_subtract_float(const device float* first [[buffer(0)]],
                                const device float* second [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                constant SiftParams& p [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < p.elements) output[gid] = first[gid] - second[gid];
}
