// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct IotaParams {
    ulong dims[4];
    ulong strides[4];
    ulong sourceDims[4];
};

#define DEFINE_IOTA_KERNEL(NAME, TYPE)                                     \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 constant IotaParams& params [[buffer(1)]],                \
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
    const ulong value = (x % params.sourceDims[0]) +                        \
        (y % params.sourceDims[1]) * params.sourceDims[0] +                 \
        (z % params.sourceDims[2]) * params.sourceDims[0] *                 \
            params.sourceDims[1] +                                         \
        (w % params.sourceDims[3]) * params.sourceDims[0] *                 \
            params.sourceDims[1] * params.sourceDims[2];                    \
    output[offset] = static_cast<TYPE>(value);                              \
}

DEFINE_IOTA_KERNEL(iota_float, float)
DEFINE_IOTA_KERNEL(iota_int, int)
DEFINE_IOTA_KERNEL(iota_uint, uint)
DEFINE_IOTA_KERNEL(iota_long, long)
DEFINE_IOTA_KERNEL(iota_ulong, ulong)
DEFINE_IOTA_KERNEL(iota_char, char)
DEFINE_IOTA_KERNEL(iota_uchar, uchar)
DEFINE_IOTA_KERNEL(iota_short, short)
DEFINE_IOTA_KERNEL(iota_ushort, ushort)
DEFINE_IOTA_KERNEL(iota_half, half)

