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

struct TopKParams {
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong outputStrides[4];
    ulong k;
    uint ascending;
};

#define DEFINE_TOPK(NAME, TYPE)                                               \
kernel void NAME(const device TYPE* input [[buffer(0)]],                     \
                 device TYPE* outputValues [[buffer(1)]],                    \
                 device uint* outputIndices [[buffer(2)]],                   \
                 constant TopKParams& params [[buffer(3)]],                  \
                 uint vector [[thread_position_in_grid]]) {                  \
    const ulong vectors = params.inputDims[1] * params.inputDims[2] *        \
                          params.inputDims[3];                               \
    if (vector >= vectors) return;                                           \
    ulong q = vector;                                                        \
    const ulong y = q % params.inputDims[1];                                 \
    q /= params.inputDims[1];                                                \
    const ulong z = q % params.inputDims[2];                                 \
    const ulong w = q / params.inputDims[2];                                 \
    const ulong inputBase = y * params.inputStrides[1] +                     \
                            z * params.inputStrides[2] +                     \
                            w * params.inputStrides[3];                      \
    const ulong outputBase = y * params.outputStrides[1] +                   \
                             z * params.outputStrides[2] +                   \
                             w * params.outputStrides[3];                    \
    ulong count = 0;                                                         \
    for (ulong x = 0; x < params.inputDims[0]; ++x) {                        \
        const TYPE candidate =                                               \
            input[inputBase + x * params.inputStrides[0]];                   \
        ulong position = 0;                                                  \
        while (position < count) {                                           \
            const TYPE current = outputValues[                               \
                outputBase + position * params.outputStrides[0]];            \
            const bool better = params.ascending ? candidate < current       \
                                                  : candidate > current;     \
            if (better) break;                                               \
            ++position;                                                      \
        }                                                                    \
        if (position >= params.k) continue;                                  \
        const ulong last = min(count, params.k - 1);                         \
        for (ulong move = last; move > position; --move) {                   \
            outputValues[outputBase + move * params.outputStrides[0]] =      \
                outputValues[outputBase +                                   \
                             (move - 1) * params.outputStrides[0]];           \
            outputIndices[outputBase + move * params.outputStrides[0]] =     \
                outputIndices[outputBase +                                  \
                              (move - 1) * params.outputStrides[0]];          \
        }                                                                    \
        outputValues[outputBase + position * params.outputStrides[0]] =      \
            candidate;                                                       \
        outputIndices[outputBase + position * params.outputStrides[0]] =     \
            uint(x);                                                         \
        if (count < params.k) ++count;                                       \
    }                                                                        \
}

DEFINE_TOPK(topk_float, float)
DEFINE_TOPK(topk_int, int)
DEFINE_TOPK(topk_uint, uint)
DEFINE_TOPK(topk_long, long)
DEFINE_TOPK(topk_ulong, ulong)
DEFINE_TOPK(topk_half, half)
