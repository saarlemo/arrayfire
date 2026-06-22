// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    uint conjugate;
};

#define TRANSPOSE_OFFSETS                                                   \
    const ulong total = params.outputDims[0] * params.outputDims[1] *       \
                        params.outputDims[2] * params.outputDims[3];         \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.outputDims[0];                         \
    position /= params.outputDims[0];                                        \
    const ulong y = position % params.outputDims[1];                         \
    position /= params.outputDims[1];                                        \
    const ulong z = position % params.outputDims[2];                         \
    const ulong w = position / params.outputDims[2];                         \
    const ulong outputOffset = x * params.outputStrides[0] +                 \
        y * params.outputStrides[1] + z * params.outputStrides[2] +          \
        w * params.outputStrides[3];                                         \
    const ulong inputOffset = y * params.inputStrides[0] +                   \
        x * params.inputStrides[1] + z * params.inputStrides[2] +            \
        w * params.inputStrides[3]

#define DEFINE_TRANSPOSE_KERNEL(NAME, TYPE)                                 \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant TransposeParams& params [[buffer(2)]],            \
                 uint gid [[thread_position_in_grid]]) {                    \
    TRANSPOSE_OFFSETS;                                                       \
    output[outputOffset] = input[inputOffset];                               \
}

DEFINE_TRANSPOSE_KERNEL(transpose_float, float)
DEFINE_TRANSPOSE_KERNEL(transpose_int, int)
DEFINE_TRANSPOSE_KERNEL(transpose_uint, uint)
DEFINE_TRANSPOSE_KERNEL(transpose_long, long)
DEFINE_TRANSPOSE_KERNEL(transpose_ulong, ulong)
DEFINE_TRANSPOSE_KERNEL(transpose_char, char)
DEFINE_TRANSPOSE_KERNEL(transpose_uchar, uchar)
DEFINE_TRANSPOSE_KERNEL(transpose_short, short)
DEFINE_TRANSPOSE_KERNEL(transpose_ushort, ushort)
DEFINE_TRANSPOSE_KERNEL(transpose_half, half)

kernel void transpose_cfloat(const device float2* input [[buffer(0)]],
                             device float2* output [[buffer(1)]],
                             constant TransposeParams& params [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    TRANSPOSE_OFFSETS;
    const float2 value = input[inputOffset];
    output[outputOffset] =
        params.conjugate ? float2(value.x, -value.y) : value;
}

struct TransposeInplaceParams {
    ulong dims[4];
    ulong strides[4];
    uint conjugate;
};

#define INPLACE_COORDINATES                                                 \
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
    const ulong offset = x * params.strides[0] + y * params.strides[1] +    \
        z * params.strides[2] + w * params.strides[3];                      \
    const ulong transposed = y * params.strides[0] +                        \
        x * params.strides[1] + z * params.strides[2] +                     \
        w * params.strides[3]

#define DEFINE_TRANSPOSE_INPLACE_KERNEL(NAME, TYPE)                         \
kernel void NAME(device TYPE* input [[buffer(0)]],                          \
                 constant TransposeInplaceParams& params [[buffer(1)]],     \
                 uint gid [[thread_position_in_grid]]) {                    \
    INPLACE_COORDINATES;                                                     \
    if (x <= y) return;                                                      \
    const TYPE value = input[offset];                                        \
    input[offset] = input[transposed];                                       \
    input[transposed] = value;                                               \
}

DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_float, float)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_int, int)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_uint, uint)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_long, long)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_ulong, ulong)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_char, char)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_uchar, uchar)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_short, short)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_ushort, ushort)
DEFINE_TRANSPOSE_INPLACE_KERNEL(transpose_inplace_half, half)

kernel void transpose_inplace_cfloat(
    device float2* input [[buffer(0)]],
    constant TransposeInplaceParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    INPLACE_COORDINATES;
    if (x < y) return;
    if (x == y) {
        if (params.conjugate) input[offset].y = -input[offset].y;
        return;
    }
    const float2 left = input[offset];
    const float2 right = input[transposed];
    input[offset] =
        params.conjugate ? float2(right.x, -right.y) : right;
    input[transposed] =
        params.conjugate ? float2(left.x, -left.y) : left;
}
