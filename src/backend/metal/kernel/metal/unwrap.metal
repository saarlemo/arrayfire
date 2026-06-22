// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct UnwrapParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong windowX;
    ulong windowY;
    ulong strideX;
    ulong strideY;
    long paddingX;
    long paddingY;
    ulong dilationX;
    ulong dilationY;
    uint columnDimension;
};

#define DEFINE_UNWRAP_KERNEL(NAME, TYPE, ZERO)                              \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant UnwrapParams& params [[buffer(2)]],               \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.outputDims[0] * params.outputDims[1] *        \
                        params.outputDims[2] * params.outputDims[3];          \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.outputDims[0];                         \
    position /= params.outputDims[0];                                        \
    const ulong y = position % params.outputDims[1];                         \
    position /= params.outputDims[1];                                        \
    const ulong z = position % params.outputDims[2];                         \
    const ulong w = position / params.outputDims[2];                         \
    const ulong column = params.columnDimension == 1 ? y : x;               \
    const ulong element = params.columnDimension == 1 ? x : y;              \
    const ulong windowsX = 1 +                                               \
        (params.inputDims[0] + 2 * params.paddingX -                         \
         ((params.windowX - 1) * params.dilationX + 1)) / params.strideX;    \
    const ulong windowX = column % windowsX;                                \
    const ulong windowY = column / windowsX;                                \
    const ulong localX = element % params.windowX;                          \
    const ulong localY = element / params.windowX;                          \
    const long inputX = static_cast<long>(windowX * params.strideX +         \
        localX * params.dilationX) - params.paddingX;                        \
    const long inputY = static_cast<long>(windowY * params.strideY +         \
        localY * params.dilationY) - params.paddingY;                        \
    const ulong outputOffset = x * params.outputStrides[0] +                 \
        y * params.outputStrides[1] + z * params.outputStrides[2] +          \
        w * params.outputStrides[3];                                         \
    if (inputX >= 0 && inputX < static_cast<long>(params.inputDims[0]) &&    \
        inputY >= 0 && inputY < static_cast<long>(params.inputDims[1])) {    \
        const ulong inputOffset = static_cast<ulong>(inputX) *               \
            params.inputStrides[0] + static_cast<ulong>(inputY) *            \
            params.inputStrides[1] + z * params.inputStrides[2] +            \
            w * params.inputStrides[3];                                      \
        output[outputOffset] = input[inputOffset];                           \
    } else {                                                                 \
        output[outputOffset] = ZERO;                                         \
    }                                                                        \
}

DEFINE_UNWRAP_KERNEL(unwrap_float, float, float(0))
DEFINE_UNWRAP_KERNEL(unwrap_cfloat, float2, float2(0))
DEFINE_UNWRAP_KERNEL(unwrap_int, int, int(0))
DEFINE_UNWRAP_KERNEL(unwrap_uint, uint, uint(0))
DEFINE_UNWRAP_KERNEL(unwrap_long, long, long(0))
DEFINE_UNWRAP_KERNEL(unwrap_ulong, ulong, ulong(0))
DEFINE_UNWRAP_KERNEL(unwrap_char, char, char(0))
DEFINE_UNWRAP_KERNEL(unwrap_uchar, uchar, uchar(0))
DEFINE_UNWRAP_KERNEL(unwrap_short, short, short(0))
DEFINE_UNWRAP_KERNEL(unwrap_ushort, ushort, ushort(0))
DEFINE_UNWRAP_KERNEL(unwrap_half, half, half(0))
