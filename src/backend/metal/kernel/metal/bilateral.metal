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

struct BilateralParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
    float spatialSigma, chromaticSigma;
};

#define BILATERAL_TILE_X 16
#define BILATERAL_TILE_Y 8

#define DEFINE_BILATERAL_TILED(NAME, TYPE)                                 \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device float* output [[buffer(1)]],                       \
                 constant BilateralParams& p [[buffer(2)]],                \
                 threadgroup float* shared [[threadgroup(0)]],             \
                 uint3 tid [[thread_position_in_threadgroup]],             \
                 uint3 group [[threadgroup_position_in_grid]]) {           \
    const float space = clamp(p.spatialSigma, 0.0f, 11.5f);                \
    const float color = max(p.chromaticSigma, 0.0f);                       \
    const long radius = max(long(space * 1.5f), 1l);                       \
    const ulong tileWidth = BILATERAL_TILE_X + ulong(2 * radius);          \
    const ulong tileHeight = BILATERAL_TILE_Y + ulong(2 * radius);         \
    const ulong tileElements = tileWidth * tileHeight;                     \
    const ulong windowSize = ulong(2 * radius + 1);                        \
    const ulong weightElements = windowSize * windowSize;                  \
    threadgroup float* tile = shared;                                      \
    threadgroup float* spatialWeights = shared + tileElements;             \
    const long baseX = long(group.x * BILATERAL_TILE_X);                   \
    const long baseY = long(group.y * BILATERAL_TILE_Y);                   \
    const ulong z = group.z % p.dims[2];                                   \
    const ulong w = group.z / p.dims[2];                                   \
    const ulong inputBase = z * p.inputStrides[2] +                        \
                            w * p.inputStrides[3];                         \
    const uint linearThread = tid.y * BILATERAL_TILE_X + tid.x;            \
    const float spatialDen = -2.0f * space * space;                        \
    const float colorDen = -2.0f * color * color;                          \
    for (ulong i = linearThread; i < tileElements;                        \
         i += BILATERAL_TILE_X * BILATERAL_TILE_Y) {                       \
        const ulong tx = i % tileWidth;                                    \
        const ulong ty = i / tileWidth;                                    \
        const long ix = clamp(baseX + long(tx) - radius, 0l,               \
                              long(p.dims[0]) - 1);                        \
        const long iy = clamp(baseY + long(ty) - radius, 0l,               \
                              long(p.dims[1]) - 1);                        \
        const ulong offset = inputBase + ulong(ix) * p.inputStrides[0] +   \
                             ulong(iy) * p.inputStrides[1];                \
        tile[i] = float(input[offset]);                                    \
    }                                                                      \
    for (ulong i = linearThread; i < weightElements;                      \
         i += BILATERAL_TILE_X * BILATERAL_TILE_Y) {                       \
        const long dx = long(i % windowSize) - radius;                    \
        const long dy = long(i / windowSize) - radius;                    \
        spatialWeights[i] =                                               \
            exp(float(dx * dx + dy * dy) / spatialDen);                    \
    }                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    const ulong x = ulong(baseX + tid.x);                                  \
    const ulong y = ulong(baseY + tid.y);                                  \
    if (x >= p.dims[0] || y >= p.dims[1]) return;                          \
    const float center = tile[(tid.y + radius) * tileWidth +               \
                              tid.x + radius];                             \
    float norm = 0.0f, result = 0.0f;                                      \
    for (long j = -radius; j <= radius; ++j)                               \
        for (long i = -radius; i <= radius; ++i) {                         \
            const float value = tile[(tid.y + radius + j) * tileWidth +    \
                                     tid.x + radius + i];                  \
            const float delta = center - value;                            \
            const ulong weightIndex = ulong(j + radius) * windowSize +    \
                                      ulong(i + radius);                   \
            const float weight = spatialWeights[weightIndex] *             \
                                 exp(delta * delta / colorDen);             \
            norm += weight;                                                \
            result += value * weight;                                      \
        }                                                                  \
    const ulong outputOffset = x * p.outputStrides[0] +                    \
                               y * p.outputStrides[1] +                    \
                               z * p.outputStrides[2] +                    \
                               w * p.outputStrides[3];                     \
    output[outputOffset] = result / norm;                                  \
}

DEFINE_BILATERAL_TILED(bilateral_tiled_float, float)
DEFINE_BILATERAL_TILED(bilateral_tiled_int, int)
DEFINE_BILATERAL_TILED(bilateral_tiled_uint, uint)
DEFINE_BILATERAL_TILED(bilateral_tiled_char, char)
DEFINE_BILATERAL_TILED(bilateral_tiled_uchar, uchar)
DEFINE_BILATERAL_TILED(bilateral_tiled_short, short)
DEFINE_BILATERAL_TILED(bilateral_tiled_ushort, ushort)
