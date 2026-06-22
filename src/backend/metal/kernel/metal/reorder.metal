// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct ReorderParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    ulong reorderDims[4];
};

#define DEFINE_REORDER_KERNEL(NAME, TYPE)                                   \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant ReorderParams& params [[buffer(2)]],              \
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
    ulong inputCoords[4] = {0, 0, 0, 0};                                    \
    inputCoords[params.reorderDims[0]] = x;                                  \
    inputCoords[params.reorderDims[1]] = y;                                  \
    inputCoords[params.reorderDims[2]] = z;                                  \
    inputCoords[params.reorderDims[3]] = w;                                  \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    const ulong inputOffset = inputCoords[0] +                              \
        inputCoords[1] * params.inputStrides[1] +                           \
        inputCoords[2] * params.inputStrides[2] +                           \
        inputCoords[3] * params.inputStrides[3];                            \
    output[outputOffset] = input[inputOffset];                               \
}

DEFINE_REORDER_KERNEL(reorder_float, float)
DEFINE_REORDER_KERNEL(reorder_cfloat, float2)
DEFINE_REORDER_KERNEL(reorder_int, int)
DEFINE_REORDER_KERNEL(reorder_uint, uint)
DEFINE_REORDER_KERNEL(reorder_long, long)
DEFINE_REORDER_KERNEL(reorder_ulong, ulong)
DEFINE_REORDER_KERNEL(reorder_char, char)
DEFINE_REORDER_KERNEL(reorder_uchar, uchar)
DEFINE_REORDER_KERNEL(reorder_short, short)
DEFINE_REORDER_KERNEL(reorder_ushort, ushort)
DEFINE_REORDER_KERNEL(reorder_half, half)

