// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct JoinParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong outputOffset[4];
};

#define DEFINE_JOIN_KERNEL(NAME, TYPE)                                      \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant JoinParams& params [[buffer(2)]],                 \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.inputDims[0] * params.inputDims[1] *          \
                        params.inputDims[2] * params.inputDims[3];           \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.inputDims[0];                          \
    position /= params.inputDims[0];                                         \
    const ulong y = position % params.inputDims[1];                          \
    position /= params.inputDims[1];                                         \
    const ulong z = position % params.inputDims[2];                          \
    const ulong w = position / params.inputDims[2];                          \
    const ulong inputOffset = x + y * params.inputStrides[1] +               \
        z * params.inputStrides[2] + w * params.inputStrides[3];             \
    const ulong outputIndex = x + params.outputOffset[0] +                   \
        (y + params.outputOffset[1]) * params.outputStrides[1] +             \
        (z + params.outputOffset[2]) * params.outputStrides[2] +             \
        (w + params.outputOffset[3]) * params.outputStrides[3];              \
    output[outputIndex] = input[inputOffset];                                \
}

DEFINE_JOIN_KERNEL(join_float, float)
DEFINE_JOIN_KERNEL(join_cfloat, float2)
DEFINE_JOIN_KERNEL(join_int, int)
DEFINE_JOIN_KERNEL(join_uint, uint)
DEFINE_JOIN_KERNEL(join_long, long)
DEFINE_JOIN_KERNEL(join_ulong, ulong)
DEFINE_JOIN_KERNEL(join_char, char)
DEFINE_JOIN_KERNEL(join_uchar, uchar)
DEFINE_JOIN_KERNEL(join_short, short)
DEFINE_JOIN_KERNEL(join_ushort, ushort)
DEFINE_JOIN_KERNEL(join_half, half)

