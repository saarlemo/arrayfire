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

struct FloodFillParams {
    ulong imageDims[4];
    ulong imageStrides[4];
    ulong seedDims[4];
    ulong seedXStrides[4];
    ulong seedYStrides[4];
};

#define DEFINE_FLOOD_FILL(NAME, TYPE)                                          \
kernel void NAME(const device TYPE* image [[buffer(0)]],                       \
                 const device uint* seedX [[buffer(1)]],                       \
                 const device uint* seedY [[buffer(2)]],                       \
                 device TYPE* output [[buffer(3)]],                            \
                 constant FloodFillParams& params [[buffer(4)]],               \
                 constant TYPE& newValue [[buffer(5)]],                        \
                 constant TYPE& lower [[buffer(6)]],                           \
                 constant TYPE& upper [[buffer(7)]],                           \
                 uint gid [[thread_position_in_grid]]) {                       \
    if (gid != 0) return;                                                       \
    const ulong width = params.imageDims[0];                                   \
    const ulong height = params.imageDims[1];                                  \
    const ulong pixels = width * height;                                       \
    for (ulong y = 0; y < height; ++y) {                                       \
        for (ulong x = 0; x < width; ++x) {                                    \
            output[x * params.imageStrides[0] +                                \
                   y * params.imageStrides[1]] = TYPE(0);                      \
        }                                                                       \
    }                                                                           \
    const ulong seedCount = params.seedDims[0] * params.seedDims[1] *           \
                            params.seedDims[2] * params.seedDims[3];            \
    for (ulong seed = 0; seed < seedCount; ++seed) {                           \
        ulong q = seed;                                                        \
        const ulong sx0 = q % params.seedDims[0];                              \
        q /= params.seedDims[0];                                               \
        const ulong sx1 = q % params.seedDims[1];                              \
        q /= params.seedDims[1];                                               \
        const ulong sx2 = q % params.seedDims[2];                              \
        const ulong sx3 = q / params.seedDims[2];                              \
        const ulong xOffset = sx0 * params.seedXStrides[0] +                   \
                              sx1 * params.seedXStrides[1] +                   \
                              sx2 * params.seedXStrides[2] +                   \
                              sx3 * params.seedXStrides[3];                    \
        const ulong yOffset = sx0 * params.seedYStrides[0] +                   \
                              sx1 * params.seedYStrides[1] +                   \
                              sx2 * params.seedYStrides[2] +                   \
                              sx3 * params.seedYStrides[3];                    \
        const uint x = seedX[xOffset];                                         \
        const uint y = seedY[yOffset];                                         \
        if (x < width && y < height) {                                         \
            output[ulong(x) * params.imageStrides[0] +                         \
                   ulong(y) * params.imageStrides[1]] = TYPE(2);               \
        }                                                                       \
    }                                                                           \
    bool changed = true;                                                       \
    while (changed) {                                                          \
        changed = false;                                                       \
        for (ulong y = 0; y < height; ++y) {                                   \
            for (ulong x = 0; x < width; ++x) {                                \
                const ulong offset = x * params.imageStrides[0] +              \
                                     y * params.imageStrides[1];               \
                if (output[offset] != TYPE(2)) continue;                       \
                const long firstY = max(long(y) - 1, long(0));                 \
                const long lastY = min(long(y) + 1, long(height) - 1);         \
                const long firstX = max(long(x) - 1, long(0));                 \
                const long lastX = min(long(x) + 1, long(width) - 1);          \
                for (long ny = firstY; ny <= lastY; ++ny) {                    \
                    for (long nx = firstX; nx <= lastX; ++nx) {                \
                        const ulong neighbor = ulong(nx) *                     \
                                                   params.imageStrides[0] +    \
                                               ulong(ny) *                     \
                                                   params.imageStrides[1];     \
                        if (output[neighbor] != TYPE(0)) continue;              \
                        const TYPE value = image[neighbor];                    \
                        if (value >= lower && value <= upper) {                \
                            output[neighbor] = TYPE(2);                         \
                            changed = true;                                    \
                        } else {                                                \
                            output[neighbor] = TYPE(1);                         \
                        }                                                       \
                    }                                                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    for (ulong index = 0; index < pixels; ++index) {                           \
        const ulong x = index % width;                                         \
        const ulong y = index / width;                                         \
        const ulong offset = x * params.imageStrides[0] +                      \
                             y * params.imageStrides[1];                       \
        output[offset] = output[offset] == TYPE(2) ? newValue : TYPE(0);       \
    }                                                                           \
}

DEFINE_FLOOD_FILL(flood_fill_float, float)
DEFINE_FLOOD_FILL(flood_fill_uint, uint)
DEFINE_FLOOD_FILL(flood_fill_ushort, ushort)
DEFINE_FLOOD_FILL(flood_fill_uchar, uchar)
