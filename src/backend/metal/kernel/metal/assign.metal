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
    long destinationStrides[4];
    ulong rhsDims[4];
    long rhsStrides[4];
    long offsets[4];
    long indexOffsets[4];
    long indexStrides[4];
    long outputOffset;
    long rhsOffset;
    uint isSequence[4];
};

long assign_trim_index(long index, long length) {
    if (index < 0) return (-index - 1) % length;
    if (index >= length) return length - (index % length) - 1;
    return index;
}

#define DEFINE_ASSIGN(NAME, TYPE)                                            \
kernel void NAME(const device TYPE* rhs [[buffer(0)]],                       \
                  const device uint* index0 [[buffer(1)]],                   \
                  const device uint* index1 [[buffer(2)]],                   \
                  const device uint* index2 [[buffer(3)]],                   \
                  const device uint* index3 [[buffer(4)]],                   \
                  device TYPE* output [[buffer(5)]],                         \
                  constant AssignParams& p [[buffer(6)]],                    \
                  uint gid [[thread_position_in_grid]]) {                    \
    const ulong total =                                                      \
        p.rhsDims[0] * p.rhsDims[1] * p.rhsDims[2] * p.rhsDims[3];          \
    if (gid >= total) return;                                                \
    ulong q = gid, coordinate[4];                                            \
    coordinate[0] = q % p.rhsDims[0];                                       \
    q /= p.rhsDims[0];                                                       \
    coordinate[1] = q % p.rhsDims[1];                                       \
    q /= p.rhsDims[1];                                                       \
    coordinate[2] = q % p.rhsDims[2];                                       \
    coordinate[3] = q / p.rhsDims[2];                                       \
    long rhsIndex = p.rhsOffset;                                             \
    long outputIndex = p.outputOffset;                                       \
    for (uint dimension = 0; dimension < 4; ++dimension) {                   \
        const long sourceCoordinate = long(coordinate[dimension]);           \
        rhsIndex += sourceCoordinate * p.rhsStrides[dimension];              \
        long destinationCoordinate;                                          \
        if (p.isSequence[dimension] != 0) {                                  \
            destinationCoordinate = sourceCoordinate + p.offsets[dimension]; \
        } else {                                                             \
            const long indexAddress = p.indexOffsets[dimension] +            \
                                      sourceCoordinate *                     \
                                          p.indexStrides[dimension];         \
            switch (dimension) {                                             \
                case 0:                                                      \
                    destinationCoordinate = long(index0[indexAddress]);      \
                    break;                                                   \
                case 1:                                                      \
                    destinationCoordinate = long(index1[indexAddress]);      \
                    break;                                                   \
                case 2:                                                      \
                    destinationCoordinate = long(index2[indexAddress]);      \
                    break;                                                   \
                default: destinationCoordinate = long(index3[indexAddress]);\
            }                                                                \
        }                                                                    \
        outputIndex +=                                                       \
            assign_trim_index(destinationCoordinate,                         \
                              long(p.outputDims[dimension])) *                \
            p.destinationStrides[dimension];                                 \
    }                                                                        \
    output[outputIndex] = rhs[rhsIndex];                                     \
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
