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

struct IReduceParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    uint dimension;
};

#define DEFINE_IREDUCE(NAME, INITIAL, BETTER)                                 \
kernel void NAME(const device float* input [[buffer(0)]],                    \
                 device float* output [[buffer(1)]],                         \
                 device uint* locations [[buffer(2)]],                       \
                 constant IReduceParams& p [[buffer(3)]],                    \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                  \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                 \
    ulong q = gid, c[4];                                                      \
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];                        \
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];                        \
    c[2] = q % p.outputDims[2]; c[3] = q / p.outputDims[2];                  \
    ulong oi = 0, ii = 0;                                                     \
    for (uint d = 0; d < 4; ++d) {                                           \
        oi += c[d] * p.outputStrides[d];                                     \
        ii += c[d] * p.inputStrides[d];                                      \
    }                                                                         \
    float best = INITIAL;                                                     \
    uint bestIndex = 0;                                                       \
    for (uint i = 0; i < p.inputDims[p.dimension]; ++i) {                    \
        const float value = input[ii + ulong(i) * p.inputStrides[p.dimension]]; \
        if (BETTER) { best = value; bestIndex = i; }                         \
    }                                                                         \
    output[oi] = best;                                                        \
    locations[oi] = bestIndex;                                               \
}

DEFINE_IREDUCE(ireduce_min_float, INFINITY,
               value < best || (value == best && i > bestIndex))
DEFINE_IREDUCE(ireduce_max_float, -INFINITY, value > best)
