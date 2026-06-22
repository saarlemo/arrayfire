// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct SobelParams {
    ulong dims[4];
    ulong inputStrides[4];
    ulong derivative0Strides[4];
    ulong derivative1Strides[4];
};

#define DEFINE_SOBEL_KERNEL(NAME, INPUT, OUTPUT)                            \
kernel void NAME(const device INPUT* input [[buffer(0)]],                  \
                 device OUTPUT* derivative0 [[buffer(1)]],                 \
                 device OUTPUT* derivative1 [[buffer(2)]],                 \
                 constant SobelParams& params [[buffer(3)]],               \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong total = params.dims[0] * params.dims[1] *                  \
                        params.dims[2] * params.dims[3];                    \
    if (gid >= total) return;                                               \
    ulong position = gid;                                                   \
    const ulong x = position % params.dims[0];                              \
    position /= params.dims[0];                                             \
    const ulong y = position % params.dims[1];                              \
    position /= params.dims[1];                                             \
    const ulong z = position % params.dims[2];                              \
    const ulong w = position / params.dims[2];                              \
    const ulong xm = x == 0 ? min(ulong(1), params.dims[0] - 1) : x - 1;    \
    const ulong xp = x == params.dims[0] - 1                               \
        ? (params.dims[0] > 1 ? x - 1 : x) : x + 1;                        \
    const ulong ym = y == 0 ? min(ulong(1), params.dims[1] - 1) : y - 1;    \
    const ulong yp = y == params.dims[1] - 1                               \
        ? (params.dims[1] > 1 ? y - 1 : y) : y + 1;                        \
    const ulong batchOffset = z * params.inputStrides[2] +                 \
                              w * params.inputStrides[3];                   \
    const OUTPUT nw = OUTPUT(input[batchOffset + ym * params.inputStrides[1] + xm]); \
    const OUTPUT sw = OUTPUT(input[batchOffset + ym * params.inputStrides[1] + xp]); \
    const OUTPUT ne = OUTPUT(input[batchOffset + yp * params.inputStrides[1] + xm]); \
    const OUTPUT se = OUTPUT(input[batchOffset + yp * params.inputStrides[1] + xp]); \
    const OUTPUT n = OUTPUT(input[batchOffset + y * params.inputStrides[1] + xm]);   \
    const OUTPUT s = OUTPUT(input[batchOffset + y * params.inputStrides[1] + xp]);   \
    const OUTPUT west = OUTPUT(input[batchOffset + ym * params.inputStrides[1] + x]); \
    const OUTPUT east = OUTPUT(input[batchOffset + yp * params.inputStrides[1] + x]); \
    const ulong output0Offset = x + y * params.derivative0Strides[1] +      \
        z * params.derivative0Strides[2] + w * params.derivative0Strides[3]; \
    const ulong output1Offset = x + y * params.derivative1Strides[1] +      \
        z * params.derivative1Strides[2] + w * params.derivative1Strides[3]; \
    derivative0[output0Offset] = sw + se - (nw + ne) + OUTPUT(2) * (s - n); \
    derivative1[output1Offset] = ne + se - (nw + sw) +                     \
                                 OUTPUT(2) * (east - west);                 \
}

DEFINE_SOBEL_KERNEL(sobel_float, float, float)
DEFINE_SOBEL_KERNEL(sobel_int, int, int)
DEFINE_SOBEL_KERNEL(sobel_uint, uint, int)
DEFINE_SOBEL_KERNEL(sobel_char, char, int)
DEFINE_SOBEL_KERNEL(sobel_uchar, uchar, int)
DEFINE_SOBEL_KERNEL(sobel_short, short, int)
DEFINE_SOBEL_KERNEL(sobel_ushort, ushort, int)

