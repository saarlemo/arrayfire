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

#define DEFINE_REGIONS(NAME, TYPE, IS_UNSIGNED, MAX_VALUE)                 \
kernel void NAME(const device char* input [[buffer(0)]],                   \
                 device TYPE* output [[buffer(1)]],                        \
                 constant RegionsParams& params [[buffer(2)]],            \
                 uint gid [[thread_position_in_grid]]) {                   \
    if (gid != 0) return;                                                  \
    const ulong width = params.dims[0];                                    \
    const ulong height = params.dims[1];                                   \
    const ulong pixels = width * height;                                   \
    for (ulong y = 0; y < height; ++y) {                                   \
        for (ulong x = 0; x < width; ++x) {                                \
            const ulong inputOffset = x * params.inputStrides[0] +         \
                                      y * params.inputStrides[1];          \
            const ulong outputOffset = x * params.outputStrides[0] +       \
                                       y * params.outputStrides[1];        \
            output[outputOffset] = input[inputOffset] != 0                 \
                                       ? TYPE(y * width + x + 1)           \
                                       : TYPE(0);                          \
        }                                                                  \
    }                                                                      \
    bool changed = true;                                                   \
    while (changed) {                                                      \
        changed = false;                                                   \
        for (ulong y = 0; y < height; ++y) {                               \
            for (ulong x = 0; x < width; ++x) {                            \
                const ulong offset = x * params.outputStrides[0] +         \
                                     y * params.outputStrides[1];          \
                TYPE label = output[offset];                               \
                if (label == TYPE(0)) continue;                            \
                const bool diagonal = params.connectivity == 8;           \
                for (long ny = max(long(y) - 1, long(0));                  \
                     ny <= min(long(y) + 1, long(height) - 1); ++ny) {     \
                    for (long nx = max(long(x) - 1, long(0));              \
                         nx <= min(long(x) + 1, long(width) - 1); ++nx) {  \
                        if (!diagonal && nx != long(x) && ny != long(y))   \
                            continue;                                      \
                        const ulong neighbor =                             \
                            ulong(nx) * params.outputStrides[0] +          \
                            ulong(ny) * params.outputStrides[1];           \
                        const TYPE neighborLabel = output[neighbor];       \
                        if (neighborLabel > TYPE(0))                       \
                            label = min(label, neighborLabel);             \
                    }                                                      \
                }                                                          \
                if (label < output[offset]) {                              \
                    output[offset] = label;                                \
                    changed = true;                                       \
                }                                                          \
            }                                                              \
        }                                                                  \
    }                                                                      \
    TYPE nextLabel = TYPE(1);                                              \
    for (ulong index = 0; index < pixels; ++index) {                       \
        const ulong x = index % width;                                     \
        const ulong y = index / width;                                     \
        const ulong offset = x * params.outputStrides[0] +                 \
                             y * params.outputStrides[1];                  \
        if (output[offset] == TYPE(index + 1)) {                           \
            output[offset] = IS_UNSIGNED                                  \
                                 ? TYPE(MAX_VALUE) - nextLabel             \
                                 : -nextLabel;                             \
            nextLabel += TYPE(1);                                         \
        }                                                                  \
    }                                                                      \
    for (long index = long(pixels) - 1; index >= 0; --index) {            \
        const ulong x = ulong(index) % width;                              \
        const ulong y = ulong(index) / width;                              \
        const ulong offset = x * params.outputStrides[0] +                 \
                             y * params.outputStrides[1];                  \
        TYPE label = output[offset];                                       \
        if (label == TYPE(0)) continue;                                    \
        if (IS_UNSIGNED ? ulong(label) > pixels : label < TYPE(0)) {       \
            output[offset] = IS_UNSIGNED ? TYPE(MAX_VALUE) - label         \
                                         : -label;                         \
        } else {                                                           \
            const ulong root = ulong(label - TYPE(1));                     \
            const ulong rootX = root % width;                              \
            const ulong rootY = root / width;                              \
            const ulong rootOffset = rootX * params.outputStrides[0] +     \
                                     rootY * params.outputStrides[1];      \
            const TYPE marker = output[rootOffset];                        \
            output[offset] = IS_UNSIGNED ? TYPE(MAX_VALUE) - marker        \
                                         : -marker;                        \
        }                                                                  \
    }                                                                      \
}

DEFINE_REGIONS(regions_float, float, false, 0)
DEFINE_REGIONS(regions_int, int, false, 0)
DEFINE_REGIONS(regions_uint, uint, true, 0xffffffffu)
DEFINE_REGIONS(regions_short, short, false, 0)
DEFINE_REGIONS(regions_ushort, ushort, true, 65535)
