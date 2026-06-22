// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct IdentityParams {
    ulong dims[4];
    ulong strides[4];
};

#define DEFINE_IDENTITY_KERNEL(NAME, TYPE)                                 \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 constant IdentityParams& params [[buffer(1)]],            \
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
    output[offset] = static_cast<TYPE>(x == y);                             \
}

DEFINE_IDENTITY_KERNEL(identity_float, float)
DEFINE_IDENTITY_KERNEL(identity_int, int)
DEFINE_IDENTITY_KERNEL(identity_uint, uint)
DEFINE_IDENTITY_KERNEL(identity_long, long)
DEFINE_IDENTITY_KERNEL(identity_ulong, ulong)
DEFINE_IDENTITY_KERNEL(identity_char, char)
DEFINE_IDENTITY_KERNEL(identity_uchar, uchar)
DEFINE_IDENTITY_KERNEL(identity_short, short)
DEFINE_IDENTITY_KERNEL(identity_ushort, ushort)
DEFINE_IDENTITY_KERNEL(identity_half, half)

kernel void identity_cfloat(device float2* output [[buffer(0)]],
                            constant IdentityParams& params [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    const ulong total = params.dims[0] * params.dims[1] *
                        params.dims[2] * params.dims[3];
    if (gid >= total) return;
    ulong position = gid;
    const ulong x = position % params.dims[0];
    position /= params.dims[0];
    const ulong y = position % params.dims[1];
    position /= params.dims[1];
    const ulong z = position % params.dims[2];
    const ulong w = position / params.dims[2];
    const ulong offset = x + y * params.strides[1] +
                         z * params.strides[2] + w * params.strides[3];
    output[offset] = float2(x == y ? 1.0f : 0.0f, 0.0f);
}

