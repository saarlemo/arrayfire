// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct TriangleParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    uint upper;
    uint unitDiagonal;
};

#define DEFINE_TRIANGLE_KERNEL(NAME, TYPE, ONE, ZERO)                       \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant TriangleParams& params [[buffer(2)]],             \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.dims[0] * params.dims[1] *                   \
                        params.dims[2] * params.dims[3];                     \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.dims[0];                               \
    position /= params.dims[0];                                              \
    const ulong y = position % params.dims[1];                               \
    position /= params.dims[1];                                              \
    const ulong z = position % params.dims[2];                               \
    const ulong w = position / params.dims[2];                               \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    const ulong inputOffset = x + y * params.inputStrides[1] +               \
        z * params.inputStrides[2] + w * params.inputStrides[3];             \
    const bool keep = params.upper ? (y >= x) : (y <= x);                    \
    output[outputOffset] = keep                                              \
        ? ((params.unitDiagonal && x == y) ? ONE : input[inputOffset])       \
        : ZERO;                                                              \
}

DEFINE_TRIANGLE_KERNEL(triangle_float, float, float(1), float(0))
DEFINE_TRIANGLE_KERNEL(triangle_cfloat, float2, float2(1.0f, 0.0f),
                       float2(0.0f, 0.0f))
DEFINE_TRIANGLE_KERNEL(triangle_int, int, int(1), int(0))
DEFINE_TRIANGLE_KERNEL(triangle_uint, uint, uint(1), uint(0))
DEFINE_TRIANGLE_KERNEL(triangle_long, long, long(1), long(0))
DEFINE_TRIANGLE_KERNEL(triangle_ulong, ulong, ulong(1), ulong(0))
DEFINE_TRIANGLE_KERNEL(triangle_char, char, char(1), char(0))
DEFINE_TRIANGLE_KERNEL(triangle_uchar, uchar, uchar(1), uchar(0))
DEFINE_TRIANGLE_KERNEL(triangle_short, short, short(1), short(0))
DEFINE_TRIANGLE_KERNEL(triangle_ushort, ushort, ushort(1), ushort(0))
DEFINE_TRIANGLE_KERNEL(triangle_half, half, half(1), half(0))

