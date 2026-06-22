// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct RangeParams {
    ulong dims[4];
    ulong strides[4];
    uint sequenceDimension;
};

#define DEFINE_RANGE_KERNEL(NAME, TYPE)                                    \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 constant RangeParams& params [[buffer(1)]],               \
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
    const ulong offset = x + y * params.strides[1] +                        \
                         z * params.strides[2] + w * params.strides[3];      \
    const ulong coordinates[4] = {x, y, z, w};                              \
    output[offset] = static_cast<TYPE>(coordinates[params.sequenceDimension]); \
}

DEFINE_RANGE_KERNEL(range_float, float)
DEFINE_RANGE_KERNEL(range_int, int)
DEFINE_RANGE_KERNEL(range_uint, uint)
DEFINE_RANGE_KERNEL(range_long, long)
DEFINE_RANGE_KERNEL(range_ulong, ulong)
DEFINE_RANGE_KERNEL(range_char, char)
DEFINE_RANGE_KERNEL(range_uchar, uchar)
DEFINE_RANGE_KERNEL(range_short, short)
DEFINE_RANGE_KERNEL(range_ushort, ushort)
DEFINE_RANGE_KERNEL(range_half, half)

