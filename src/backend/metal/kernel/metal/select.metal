// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct SelectParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong aDims[4];
    ulong aStrides[4];
    ulong bDims[4];
    ulong bStrides[4];
    ulong conditionDims[4];
    ulong conditionStrides[4];
    uint flip;
};

#define SELECT_COORDINATES                                                  \
    const ulong total = params.outputDims[0] * params.outputDims[1] *       \
                        params.outputDims[2] * params.outputDims[3];         \
    if (gid >= total) return;                                               \
    ulong position = gid;                                                   \
    const ulong x = position % params.outputDims[0];                        \
    position /= params.outputDims[0];                                       \
    const ulong y = position % params.outputDims[1];                        \
    position /= params.outputDims[1];                                       \
    const ulong z = position % params.outputDims[2];                        \
    const ulong w = position / params.outputDims[2];                        \
    const ulong coordinates[4] = {x, y, z, w};                              \
    ulong outputOffset = 0;                                                 \
    ulong aOffset = 0;                                                      \
    ulong bOffset = 0;                                                      \
    ulong conditionOffset = 0;                                              \
    for (uint i = 0; i < 4; ++i) {                                         \
        outputOffset += coordinates[i] * params.outputStrides[i];           \
        aOffset += (params.aDims[i] == params.outputDims[i]                  \
                        ? coordinates[i]                                    \
                        : 0) * params.aStrides[i];                           \
        bOffset += (params.bDims[i] == params.outputDims[i]                  \
                        ? coordinates[i]                                    \
                        : 0) * params.bStrides[i];                           \
        conditionOffset +=                                                  \
            (params.conditionDims[i] == params.outputDims[i]                \
                 ? coordinates[i]                                          \
                 : 0) * params.conditionStrides[i];                         \
    }

#define DEFINE_SELECT_KERNEL(NAME, TYPE)                                    \
kernel void NAME(const device char* condition [[buffer(0)]],                \
                 const device TYPE* a [[buffer(1)]],                        \
                 const device TYPE* b [[buffer(2)]],                        \
                 device TYPE* output [[buffer(3)]],                         \
                 constant SelectParams& params [[buffer(4)]],               \
                 uint gid [[thread_position_in_grid]]) {                    \
    SELECT_COORDINATES;                                                     \
    output[outputOffset] = condition[conditionOffset] ? a[aOffset]          \
                                                       : b[bOffset];        \
}

#define DEFINE_SELECT_SCALAR_KERNEL(NAME, TYPE)                             \
kernel void NAME(const device char* condition [[buffer(0)]],                \
                 const device TYPE* a [[buffer(1)]],                        \
                 device TYPE* output [[buffer(2)]],                         \
                 constant TYPE& scalar [[buffer(3)]],                       \
                 constant SelectParams& params [[buffer(4)]],               \
                 uint gid [[thread_position_in_grid]]) {                    \
    SELECT_COORDINATES;                                                     \
    const bool chooseArray = bool(params.flip) !=                           \
                             bool(condition[conditionOffset]);              \
    output[outputOffset] = chooseArray ? a[aOffset] : scalar;               \
}

#define DEFINE_SELECT_KERNELS(SUFFIX, TYPE)                                 \
DEFINE_SELECT_KERNEL(select_##SUFFIX, TYPE)                                 \
DEFINE_SELECT_SCALAR_KERNEL(select_scalar_##SUFFIX, TYPE)

DEFINE_SELECT_KERNELS(float, float)
DEFINE_SELECT_KERNELS(cfloat, float2)
DEFINE_SELECT_KERNELS(int, int)
DEFINE_SELECT_KERNELS(uint, uint)
DEFINE_SELECT_KERNELS(long, long)
DEFINE_SELECT_KERNELS(ulong, ulong)
DEFINE_SELECT_KERNELS(char, char)
DEFINE_SELECT_KERNELS(uchar, uchar)
DEFINE_SELECT_KERNELS(short, short)
DEFINE_SELECT_KERNELS(ushort, ushort)
DEFINE_SELECT_KERNELS(half, half)
