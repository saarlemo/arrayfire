/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <metal_stdlib>

using namespace metal;

struct AssignParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong rhsDims[4];
    ulong rhsStrides[4];
    long offsets[4];
};

#define DEFINE_ASSIGN(NAME, TYPE)                                             \
kernel void NAME(const device TYPE* original [[buffer(0)]],                   \
                 const device TYPE* rhs [[buffer(1)]],                        \
                 device TYPE* output [[buffer(2)]],                           \
                 constant AssignParams& params [[buffer(3)]],                 \
                 uint gid [[thread_position_in_grid]]) {                      \
    const ulong total = params.outputDims[0] * params.outputDims[1] *          \
                        params.outputDims[2] * params.outputDims[3];           \
    if (gid >= total) return;                                                  \
    ulong q = gid;                                                             \
    const ulong x = q % params.outputDims[0];                                 \
    q /= params.outputDims[0];                                                 \
    const ulong y = q % params.outputDims[1];                                 \
    q /= params.outputDims[1];                                                 \
    const ulong z = q % params.outputDims[2];                                 \
    const ulong w = q / params.outputDims[2];                                 \
    const ulong coordinates[4] = {x, y, z, w};                                \
    const ulong outputOffset = x * params.outputStrides[0] +                  \
                               y * params.outputStrides[1] +                  \
                               z * params.outputStrides[2] +                  \
                               w * params.outputStrides[3];                   \
    bool selected = true;                                                      \
    ulong rhsOffset = 0;                                                       \
    for (uint dimension = 0; dimension < 4; ++dimension) {                    \
        const long sourceCoordinate =                                         \
            long(coordinates[dimension]) - params.offsets[dimension];         \
        selected = selected && sourceCoordinate >= 0 &&                       \
                   sourceCoordinate < long(params.rhsDims[dimension]);        \
        if (sourceCoordinate >= 0)                                             \
            rhsOffset += ulong(sourceCoordinate) * params.rhsStrides[dimension];\
    }                                                                          \
    output[outputOffset] = selected ? rhs[rhsOffset] : original[outputOffset]; \
}

DEFINE_ASSIGN(assign_float, float)
DEFINE_ASSIGN(assign_cfloat, float2)
DEFINE_ASSIGN(assign_int, int)
DEFINE_ASSIGN(assign_uint, uint)
DEFINE_ASSIGN(assign_long, long)
DEFINE_ASSIGN(assign_ulong, ulong)
DEFINE_ASSIGN(assign_char, char)
DEFINE_ASSIGN(assign_uchar, uchar)
DEFINE_ASSIGN(assign_short, short)
DEFINE_ASSIGN(assign_ushort, ushort)
DEFINE_ASSIGN(assign_half, half)
