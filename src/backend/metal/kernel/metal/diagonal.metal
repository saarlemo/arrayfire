// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct DiagonalParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    int diagonal;
};

#define DEFINE_DIAG_CREATE_KERNEL(NAME, TYPE)                               \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant DiagonalParams& params [[buffer(2)]],             \
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
    TYPE value = TYPE(0);                                                    \
    if (long(x) == long(y) - params.diagonal) {                              \
        const ulong vectorIndex = params.diagonal > 0 ? x : y;               \
        value = input[vectorIndex + z * params.inputStrides[1]];             \
    }                                                                        \
    output[outputOffset] = value;                                            \
}

#define DEFINE_DIAG_EXTRACT_KERNEL(NAME, TYPE)                              \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant DiagonalParams& params [[buffer(2)]],             \
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
    const ulong inputBase = params.diagonal > 0                              \
        ? ulong(params.diagonal) * params.inputStrides[1]                    \
        : ulong(-params.diagonal);                                           \
    const ulong inputOffset = inputBase + x + x * params.inputStrides[1] +   \
        z * params.inputStrides[2] + w * params.inputStrides[3];             \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    output[outputOffset] = input[inputOffset];                               \
}

#define DEFINE_DIAGONAL_KERNELS(SUFFIX, TYPE)                               \
DEFINE_DIAG_CREATE_KERNEL(diag_create_##SUFFIX, TYPE)                       \
DEFINE_DIAG_EXTRACT_KERNEL(diag_extract_##SUFFIX, TYPE)

DEFINE_DIAGONAL_KERNELS(float, float)
DEFINE_DIAGONAL_KERNELS(cfloat, float2)
DEFINE_DIAGONAL_KERNELS(int, int)
DEFINE_DIAGONAL_KERNELS(uint, uint)
DEFINE_DIAGONAL_KERNELS(long, long)
DEFINE_DIAGONAL_KERNELS(ulong, ulong)
DEFINE_DIAGONAL_KERNELS(char, char)
DEFINE_DIAGONAL_KERNELS(uchar, uchar)
DEFINE_DIAGONAL_KERNELS(short, short)
DEFINE_DIAGONAL_KERNELS(ushort, ushort)
DEFINE_DIAGONAL_KERNELS(half, half)

