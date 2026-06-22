// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct DiffParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    uint dimension;
};

#define DEFINE_DIFF1_KERNEL(NAME, TYPE)                                     \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant DiffParams& params [[buffer(2)]],                 \
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
    const ulong inputOffset = x + y * params.inputStrides[1] +               \
        z * params.inputStrides[2] + w * params.inputStrides[3];             \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    const ulong step = params.inputStrides[params.dimension];                \
    output[outputOffset] = input[inputOffset + step] - input[inputOffset];   \
}

#define DEFINE_DIFF2_KERNEL(NAME, TYPE)                                     \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant DiffParams& params [[buffer(2)]],                 \
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
    const ulong inputOffset = x + y * params.inputStrides[1] +               \
        z * params.inputStrides[2] + w * params.inputStrides[3];             \
    const ulong outputOffset = x + y * params.outputStrides[1] +             \
        z * params.outputStrides[2] + w * params.outputStrides[3];           \
    const ulong step = params.inputStrides[params.dimension];                \
    output[outputOffset] = input[inputOffset + 2 * step] +                   \
        input[inputOffset] - input[inputOffset + step] -                     \
        input[inputOffset + step];                                          \
}

#define DEFINE_DIFF_KERNELS(SUFFIX, TYPE)                                   \
DEFINE_DIFF1_KERNEL(diff1_##SUFFIX, TYPE)                                   \
DEFINE_DIFF2_KERNEL(diff2_##SUFFIX, TYPE)

DEFINE_DIFF_KERNELS(float, float)
DEFINE_DIFF_KERNELS(cfloat, float2)
DEFINE_DIFF_KERNELS(int, int)
DEFINE_DIFF_KERNELS(uint, uint)
DEFINE_DIFF_KERNELS(long, long)
DEFINE_DIFF_KERNELS(ulong, ulong)
DEFINE_DIFF_KERNELS(char, char)
DEFINE_DIFF_KERNELS(uchar, uchar)
DEFINE_DIFF_KERNELS(short, short)
DEFINE_DIFF_KERNELS(ushort, ushort)
DEFINE_DIFF_KERNELS(half, half)

