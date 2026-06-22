// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct WrapParams {
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

#define DEFINE_WRAP_KERNEL(NAME, TYPE, ZERO)                                \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant WrapParams& params [[buffer(2)]],                 \
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
    const ulong columns = params.inputDims[params.columnDimension];          \
    const ulong windowsX = 1 +                                               \
        (params.outputDims[0] + 2 * params.paddingX -                        \
         ((params.windowX - 1) * params.dilationX + 1)) / params.strideX;    \
    TYPE sum = ZERO;                                                         \
    for (ulong column = 0; column < columns; ++column) {                     \
        const ulong windowX = column % windowsX;                            \
        const ulong windowY = column / windowsX;                            \
        const long originX = static_cast<long>(windowX * params.strideX) -  \
                             params.paddingX;                               \
        const long originY = static_cast<long>(windowY * params.strideY) -  \
                             params.paddingY;                               \
        const long deltaX = static_cast<long>(x) - originX;                 \
        const long deltaY = static_cast<long>(y) - originY;                 \
        if (deltaX < 0 || deltaY < 0 ||                                     \
            deltaX >= static_cast<long>(params.windowX *                    \
                                        params.dilationX) ||                \
            deltaY >= static_cast<long>(params.windowY *                    \
                                        params.dilationY) ||                \
            deltaX % static_cast<long>(params.dilationX) != 0 ||            \
            deltaY % static_cast<long>(params.dilationY) != 0) {            \
            continue;                                                       \
        }                                                                   \
        const ulong localX = static_cast<ulong>(deltaX) / params.dilationX; \
        const ulong localY = static_cast<ulong>(deltaY) / params.dilationY; \
        const ulong element = localY * params.windowX + localX;             \
        const ulong inputX = params.columnDimension == 1 ? element : column;\
        const ulong inputY = params.columnDimension == 1 ? column : element;\
        const ulong inputOffset = inputX * params.inputStrides[0] +         \
            inputY * params.inputStrides[1] + z * params.inputStrides[2] +  \
            w * params.inputStrides[3];                                     \
        sum += input[inputOffset];                                          \
    }                                                                       \
    const ulong outputOffset = x * params.outputStrides[0] +                \
        y * params.outputStrides[1] + z * params.outputStrides[2] +         \
        w * params.outputStrides[3];                                        \
    output[outputOffset] = sum;                                             \
}

DEFINE_WRAP_KERNEL(wrap_float, float, float(0))
DEFINE_WRAP_KERNEL(wrap_cfloat, float2, float2(0))
DEFINE_WRAP_KERNEL(wrap_int, int, int(0))
DEFINE_WRAP_KERNEL(wrap_uint, uint, uint(0))
DEFINE_WRAP_KERNEL(wrap_long, long, long(0))
DEFINE_WRAP_KERNEL(wrap_ulong, ulong, ulong(0))
DEFINE_WRAP_KERNEL(wrap_char, char, char(0))
DEFINE_WRAP_KERNEL(wrap_uchar, uchar, uchar(0))
DEFINE_WRAP_KERNEL(wrap_short, short, short(0))
DEFINE_WRAP_KERNEL(wrap_ushort, ushort, ushort(0))
DEFINE_WRAP_KERNEL(wrap_half, half, half(0))
