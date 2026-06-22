// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct ShiftParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    ulong shifts[4];
};

#define DEFINE_SHIFT_KERNEL(NAME, TYPE)                                     \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant ShiftParams& params [[buffer(2)]],                \
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
    const ulong ix = (x + params.shifts[0]) % params.dims[0];                \
    const ulong iy = (y + params.shifts[1]) % params.dims[1];                \
    const ulong iz = (z + params.shifts[2]) % params.dims[2];                \
    const ulong iw = (w + params.shifts[3]) % params.dims[3];                \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    const ulong inputOffset = ix + iy * params.inputStrides[1] +             \
        iz * params.inputStrides[2] + iw * params.inputStrides[3];           \
    output[outputOffset] = input[inputOffset];                               \
}

DEFINE_SHIFT_KERNEL(shift_float, float)
DEFINE_SHIFT_KERNEL(shift_cfloat, float2)
DEFINE_SHIFT_KERNEL(shift_int, int)
DEFINE_SHIFT_KERNEL(shift_uint, uint)
DEFINE_SHIFT_KERNEL(shift_long, long)
DEFINE_SHIFT_KERNEL(shift_ulong, ulong)
DEFINE_SHIFT_KERNEL(shift_char, char)
DEFINE_SHIFT_KERNEL(shift_uchar, uchar)
DEFINE_SHIFT_KERNEL(shift_short, short)
DEFINE_SHIFT_KERNEL(shift_ushort, ushort)
DEFINE_SHIFT_KERNEL(shift_half, half)

