// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct TileParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
};

#define DEFINE_TILE_KERNEL(NAME, TYPE)                                      \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant TileParams& params [[buffer(2)]],                 \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.outputDims[0] * params.outputDims[1] *        \
                        params.outputDims[2] * params.outputDims[3];         \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.outputDims[0];                         \
    position /= params.outputDims[0];                                        \
    const ulong y = position % params.outputDims[1];                         \
    position /= params.outputDims[1];                                        \
    const ulong z = position % params.outputDims[2];                         \
    const ulong w = position / params.outputDims[2];                         \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    const ulong inputOffset = (x % params.inputDims[0]) +                    \
        (y % params.inputDims[1]) * params.inputStrides[1] +                 \
        (z % params.inputDims[2]) * params.inputStrides[2] +                 \
        (w % params.inputDims[3]) * params.inputStrides[3];                  \
    output[outputOffset] = input[inputOffset];                               \
}

DEFINE_TILE_KERNEL(tile_float, float)
DEFINE_TILE_KERNEL(tile_cfloat, float2)
DEFINE_TILE_KERNEL(tile_int, int)
DEFINE_TILE_KERNEL(tile_uint, uint)
DEFINE_TILE_KERNEL(tile_long, long)
DEFINE_TILE_KERNEL(tile_ulong, ulong)
DEFINE_TILE_KERNEL(tile_char, char)
DEFINE_TILE_KERNEL(tile_uchar, uchar)
DEFINE_TILE_KERNEL(tile_short, short)
DEFINE_TILE_KERNEL(tile_ushort, ushort)
DEFINE_TILE_KERNEL(tile_half, half)

