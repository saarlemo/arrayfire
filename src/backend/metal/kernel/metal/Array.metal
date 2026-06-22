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

struct ArrayAddParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong leftDims[4];
    ulong leftStrides[4];
    ulong rightDims[4];
    ulong rightStrides[4];
};

kernel void array_add_float(const device float* left [[buffer(0)]],
                            const device float* right [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            constant ArrayAddParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid, c[4];
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];
    c[2] = q % p.outputDims[2]; c[3] = q / p.outputDims[2];
    ulong oi = 0, li = 0, ri = 0;
    for (uint d = 0; d < 4; ++d) {
        oi += c[d] * p.outputStrides[d];
        li += (p.leftDims[d] == 1 ? 0 : c[d]) * p.leftStrides[d];
        ri += (p.rightDims[d] == 1 ? 0 : c[d]) * p.rightStrides[d];
    }
    output[oi] = left[li] + right[ri];
}
