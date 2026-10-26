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

struct MorphParams {
    ulong outputDims[4], outputStrides[4];
    ulong inputDims[4], inputStrides[4];
    ulong maskDims[4], maskStrides[4];
    uint dilation, volume;
};

// The 2D fast path cooperatively stages one image tile and its mask halo in
// threadgroup memory.
#define MORPH_TILE_X 16
#define MORPH_TILE_Y 8

#define DEFINE_MORPH_2D_TILED(NAME, TYPE, LOW, HIGH)                       \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 const device TYPE* mask [[buffer(1)]],                    \
                 device TYPE* output [[buffer(2)]],                        \
                 constant MorphParams& p [[buffer(3)]],                    \
                 threadgroup TYPE* tile [[threadgroup(0)]],                \
                 uint3 tid [[thread_position_in_threadgroup]],             \
                 uint3 group [[threadgroup_position_in_grid]]) {           \
    const long rx = long(p.maskDims[0] / 2);                               \
    const long ry = long(p.maskDims[1] / 2);                               \
    const ulong tileWidth = MORPH_TILE_X + p.maskDims[0] - 1;              \
    const ulong tileHeight = MORPH_TILE_Y + p.maskDims[1] - 1;             \
    const ulong tileElements = tileWidth * tileHeight;                     \
    const long baseX = long(group.x * MORPH_TILE_X);                       \
    const long baseY = long(group.y * MORPH_TILE_Y);                       \
    const ulong batch = group.z;                                           \
    const ulong z = batch % p.inputDims[2];                                \
    const ulong w = batch / p.inputDims[2];                                \
    const ulong inputBase = z * p.inputStrides[2] +                        \
                            w * p.inputStrides[3];                         \
    const uint linearThread = tid.y * MORPH_TILE_X + tid.x;                \
    for (ulong i = linearThread; i < tileElements; i += MORPH_TILE_X *     \
                                                               MORPH_TILE_Y) { \
        const ulong tx = i % tileWidth;                                    \
        const ulong ty = i / tileWidth;                                    \
        const long ix = baseX + long(tx) - rx;                             \
        const long iy = baseY + long(ty) - ry;                             \
        TYPE value = p.dilation ? TYPE(LOW) : TYPE(HIGH);                  \
        if (ix >= 0 && iy >= 0 && ix < long(p.inputDims[0]) &&             \
            iy < long(p.inputDims[1])) {                                   \
            const ulong offset = inputBase + ulong(ix) *                   \
                                 p.inputStrides[0] + ulong(iy) *            \
                                 p.inputStrides[1];                        \
            value = input[offset];                                         \
        }                                                                  \
        tile[i] = value;                                                   \
    }                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    const ulong x = ulong(baseX + tid.x);                                  \
    const ulong y = ulong(baseY + tid.y);                                  \
    if (x >= p.outputDims[0] || y >= p.outputDims[1]) return;              \
    TYPE result = p.dilation ? TYPE(LOW) : TYPE(HIGH);                     \
    for (ulong my = 0; my < p.maskDims[1]; ++my)                           \
        for (ulong mx = 0; mx < p.maskDims[0]; ++mx) {                     \
            const ulong maskOffset = mx * p.maskStrides[0] +               \
                                     my * p.maskStrides[1];                \
            if (mask[maskOffset] > TYPE(0)) {                              \
                const TYPE value = tile[(tid.y + my) * tileWidth +         \
                                        tid.x + mx];                        \
                result = p.dilation ? max(result, value)                   \
                                    : min(result, value);                  \
            }                                                              \
        }                                                                  \
    const ulong outputBatch = z * p.outputStrides[2] +                     \
                              w * p.outputStrides[3];                      \
    const ulong outputOffset = outputBatch + x * p.outputStrides[0] +      \
                               y * p.outputStrides[1];                     \
    output[outputOffset] = result;                                         \
}

DEFINE_MORPH_2D_TILED(morph2d_tiled_float, float, -INFINITY, INFINITY)
DEFINE_MORPH_2D_TILED(morph2d_tiled_int, int, (-2147483647-1), 2147483647)
DEFINE_MORPH_2D_TILED(morph2d_tiled_uint, uint, 0u, 0xffffffffu)
DEFINE_MORPH_2D_TILED(morph2d_tiled_char, char, -128, 127)
DEFINE_MORPH_2D_TILED(morph2d_tiled_uchar, uchar, 0, 255)
DEFINE_MORPH_2D_TILED(morph2d_tiled_short, short, -32768, 32767)
DEFINE_MORPH_2D_TILED(morph2d_tiled_ushort, ushort, 0, 65535)

// Chunking bounds shared memory for large 2D masks and all 3D masks. In 3D,
// one mask-z slice is streamed at a time through the same xy tile.
#define MORPH_MASK_CHUNK_X 8
#define MORPH_MASK_CHUNK_Y 8

#define DEFINE_MORPH_CHUNKED(NAME, TYPE, LOW, HIGH)                        \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 const device TYPE* mask [[buffer(1)]],                    \
                 device TYPE* output [[buffer(2)]],                        \
                 constant MorphParams& p [[buffer(3)]],                    \
                 threadgroup TYPE* tile [[threadgroup(0)]],                \
                 uint3 tid [[thread_position_in_threadgroup]],             \
                 uint3 group [[threadgroup_position_in_grid]]) {           \
    const long rx = long(p.maskDims[0] / 2);                               \
    const long ry = long(p.maskDims[1] / 2);                               \
    const long rz = p.volume ? long(p.maskDims[2] / 2) : 0;                \
    const ulong tileWidth = MORPH_TILE_X + MORPH_MASK_CHUNK_X - 1;         \
    const ulong tileHeight = MORPH_TILE_Y + MORPH_MASK_CHUNK_Y - 1;        \
    const ulong tileElements = tileWidth * tileHeight;                     \
    const long baseX = long(group.x * MORPH_TILE_X);                       \
    const long baseY = long(group.y * MORPH_TILE_Y);                       \
    const ulong x = ulong(baseX + tid.x);                                  \
    const ulong y = ulong(baseY + tid.y);                                  \
    const ulong batch = group.z;                                           \
    const ulong z = batch % p.outputDims[2];                               \
    const ulong w = batch / p.outputDims[2];                               \
    const ulong maskZCount = p.volume ? p.maskDims[2] : 1;                 \
    const uint linearThread = tid.y * MORPH_TILE_X + tid.x;                \
    TYPE result = p.dilation ? TYPE(LOW) : TYPE(HIGH);                     \
    for (ulong mz = 0; mz < maskZCount; ++mz) {                            \
        const long iz = p.volume ? long(z) + long(mz) - rz : long(z);      \
        for (ulong maskY = 0; maskY < p.maskDims[1];                      \
             maskY += MORPH_MASK_CHUNK_Y) {                                \
            const ulong chunkHeight = min(                                \
                ulong(MORPH_MASK_CHUNK_Y), p.maskDims[1] - maskY);         \
            for (ulong maskX = 0; maskX < p.maskDims[0];                  \
                 maskX += MORPH_MASK_CHUNK_X) {                            \
                const ulong chunkWidth = min(                              \
                    ulong(MORPH_MASK_CHUNK_X), p.maskDims[0] - maskX);     \
                for (ulong i = linearThread; i < tileElements;            \
                     i += MORPH_TILE_X * MORPH_TILE_Y) {                   \
                    const ulong tx = i % tileWidth;                       \
                    const ulong ty = i / tileWidth;                       \
                    const long ix = baseX + long(maskX) + long(tx) - rx;   \
                    const long iy = baseY + long(maskY) + long(ty) - ry;   \
                    TYPE value = p.dilation ? TYPE(LOW) : TYPE(HIGH);      \
                    if (ix >= 0 && iy >= 0 && iz >= 0 &&                   \
                        ix < long(p.inputDims[0]) &&                       \
                        iy < long(p.inputDims[1]) &&                       \
                        iz < long(p.inputDims[2])) {                       \
                        const ulong offset =                               \
                            ulong(ix) * p.inputStrides[0] +                 \
                            ulong(iy) * p.inputStrides[1] +                 \
                            ulong(iz) * p.inputStrides[2] +                 \
                            w * p.inputStrides[3];                         \
                        value = input[offset];                             \
                    }                                                      \
                    tile[i] = value;                                       \
                }                                                          \
                threadgroup_barrier(mem_flags::mem_threadgroup);           \
                if (x < p.outputDims[0] && y < p.outputDims[1]) {           \
                    for (ulong cy = 0; cy < chunkHeight; ++cy)            \
                        for (ulong cx = 0; cx < chunkWidth; ++cx) {        \
                            const ulong maskOffset =                       \
                                (maskX + cx) * p.maskStrides[0] +           \
                                (maskY + cy) * p.maskStrides[1] +           \
                                mz * p.maskStrides[2];                     \
                            if (mask[maskOffset] > TYPE(0)) {              \
                                const TYPE value =                         \
                                    tile[(tid.y + cy) * tileWidth +        \
                                         tid.x + cx];                       \
                                result = p.dilation ? max(result, value)   \
                                                    : min(result, value);  \
                            }                                              \
                        }                                                  \
                }                                                          \
                threadgroup_barrier(mem_flags::mem_threadgroup);           \
            }                                                              \
        }                                                                  \
    }                                                                      \
    if (x < p.outputDims[0] && y < p.outputDims[1]) {                      \
        const ulong outputOffset = x * p.outputStrides[0] +                \
                                   y * p.outputStrides[1] +                \
                                   z * p.outputStrides[2] +                \
                                   w * p.outputStrides[3];                 \
        output[outputOffset] = result;                                     \
    }                                                                      \
}

DEFINE_MORPH_CHUNKED(morph_chunked_float, float, -INFINITY, INFINITY)
DEFINE_MORPH_CHUNKED(morph_chunked_int, int, (-2147483647-1), 2147483647)
DEFINE_MORPH_CHUNKED(morph_chunked_uint, uint, 0u, 0xffffffffu)
DEFINE_MORPH_CHUNKED(morph_chunked_char, char, -128, 127)
DEFINE_MORPH_CHUNKED(morph_chunked_uchar, uchar, 0, 255)
DEFINE_MORPH_CHUNKED(morph_chunked_short, short, -32768, 32767)
DEFINE_MORPH_CHUNKED(morph_chunked_ushort, ushort, 0, 65535)
