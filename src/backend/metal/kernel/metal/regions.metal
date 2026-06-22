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

struct RegionsParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    uint connectivity;
};

kernel void regions_float(const device char* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant RegionsParams& params [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    const ulong width = params.dims[0];
    const ulong height = params.dims[1];
    const ulong pixels = width * height;

    for (ulong y = 0; y < height; ++y) {
        for (ulong x = 0; x < width; ++x) {
            const ulong inputOffset = x * params.inputStrides[0] +
                                      y * params.inputStrides[1];
            const ulong outputOffset = x * params.outputStrides[0] +
                                       y * params.outputStrides[1];
            output[outputOffset] =
                input[inputOffset] != 0 ? float(y * width + x + 1) : 0.0f;
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (ulong y = 0; y < height; ++y) {
            for (ulong x = 0; x < width; ++x) {
                const ulong offset = x * params.outputStrides[0] +
                                     y * params.outputStrides[1];
                float label = output[offset];
                if (label == 0.0f) continue;
                const long radius = params.connectivity == 8 ? 1 : 0;
                for (long ny = max(long(y) - 1, long(0));
                     ny <= min(long(y) + 1, long(height) - 1); ++ny) {
                    for (long nx = max(long(x) - 1, long(0));
                         nx <= min(long(x) + 1, long(width) - 1); ++nx) {
                        if (radius == 0 && nx != long(x) && ny != long(y))
                            continue;
                        const ulong neighbor =
                            ulong(nx) * params.outputStrides[0] +
                            ulong(ny) * params.outputStrides[1];
                        const float neighborLabel = output[neighbor];
                        if (neighborLabel > 0.0f)
                            label = min(label, neighborLabel);
                    }
                }
                if (label < output[offset]) {
                    output[offset] = label;
                    changed = true;
                }
            }
        }
    }

    float nextLabel = 1.0f;
    for (ulong index = 0; index < pixels; ++index) {
        const ulong x = index % width;
        const ulong y = index / width;
        const ulong offset = x * params.outputStrides[0] +
                             y * params.outputStrides[1];
        if (output[offset] == float(index + 1))
            output[offset] = -nextLabel++;
    }
    for (ulong index = 0; index < pixels; ++index) {
        const ulong x = index % width;
        const ulong y = index / width;
        const ulong offset = x * params.outputStrides[0] +
                             y * params.outputStrides[1];
        const float label = output[offset];
        if (label > 0.0f) {
            const ulong root = ulong(label - 1.0f);
            const ulong rootX = root % width;
            const ulong rootY = root / width;
            const ulong rootOffset = rootX * params.outputStrides[0] +
                                     rootY * params.outputStrides[1];
            output[offset] = -output[rootOffset];
        }
    }
    for (ulong index = 0; index < pixels; ++index) {
        const ulong x = index % width;
        const ulong y = index / width;
        const ulong offset = x * params.outputStrides[0] +
                             y * params.outputStrides[1];
        if (output[offset] < 0.0f) output[offset] = -output[offset];
    }
}
