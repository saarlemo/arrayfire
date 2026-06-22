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

struct MeanParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    uint dimension;
};

kernel void mean_dim_float(const device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant MeanParams& params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    const ulong total = params.outputDims[0] * params.outputDims[1] *
                        params.outputDims[2] * params.outputDims[3];
    if (gid >= total) return;

    ulong q = gid;
    ulong coordinates[4];
    coordinates[0] = q % params.outputDims[0];
    q /= params.outputDims[0];
    coordinates[1] = q % params.outputDims[1];
    q /= params.outputDims[1];
    coordinates[2] = q % params.outputDims[2];
    coordinates[3] = q / params.outputDims[2];

    ulong outputIndex = 0;
    ulong inputBase = 0;
    for (uint d = 0; d < 4; ++d) {
        outputIndex += coordinates[d] * params.outputStrides[d];
        inputBase += coordinates[d] * params.inputStrides[d];
    }

    float partials[64];
    ulong count = 0;
    const ulong length = params.inputDims[params.dimension];
    const ulong stride = params.inputStrides[params.dimension];
    for (ulong i = 0; i < length; ++i) {
        float value = input[inputBase + i * stride];
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
    output[outputIndex] = sum / float(length);
}
