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

struct ReduceParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    uint dimension;
    uint changeNan;
    float nanValue;
};

inline float reduceValue(float value, uint changeNan, float nanValue) {
    return changeNan && isnan(value) ? nanValue : value;
}

kernel void reduce_add_float(const device float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant ReduceParams& p [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid, c[4];
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];
    c[2] = q % p.outputDims[2]; c[3] = q / p.outputDims[2];
    ulong oi = 0, ii = 0;
    for (uint d = 0; d < 4; ++d) {
        oi += c[d] * p.outputStrides[d];
        ii += c[d] * p.inputStrides[d];
    }
    float partials[64];
    ulong count = 0;
    const ulong length = p.inputDims[p.dimension];
    const ulong stride = p.inputStrides[p.dimension];
    for (ulong i = 0; i < length; ++i) {
        float value =
            reduceValue(input[ii + i * stride], p.changeNan, p.nanValue);
        ulong occupied = count++;
        uint level = 0;
        while ((occupied & 1ul) != 0) {
            value = partials[level] + value;
            occupied >>= 1;
            ++level;
        }
        partials[level] = value;
    }
    float sum = 0.0f;
    ulong occupied = count;
    uint level = 0;
    while (occupied != 0) {
        if ((occupied & 1ul) != 0) sum += partials[level];
        occupied >>= 1;
        ++level;
    }
    output[oi] = sum;
}
