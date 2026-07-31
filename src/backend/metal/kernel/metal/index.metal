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

struct IndexParams {
    ulong outputDims[4];
    long outputStrides[4];
    ulong inputDims[4];
    long inputStrides[4];
    long offsets[4];
    long steps[4];
    long indexOffsets[4];
    long indexStrides[4];
    long outputOffset;
    long inputOffset;
    uint isSequence[4];
};

long trim_index(long index, long length) {
    if (index < 0) return (-index - 1) % length;
    if (index >= length) return length - (index % length) - 1;
    return index;
}

#define DEFINE_INDEX(NAME, TYPE)                                             \
kernel void NAME(const device TYPE* input [[buffer(0)]],                     \
                 const device uint* index0 [[buffer(1)]],                    \
                 const device uint* index1 [[buffer(2)]],                    \
                 const device uint* index2 [[buffer(3)]],                    \
                 const device uint* index3 [[buffer(4)]],                    \
                 device TYPE* output [[buffer(5)]],                          \
                 constant IndexParams& p [[buffer(6)]],                      \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                  \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                \
    ulong q = gid, coordinate[4];                                            \
    coordinate[0] = q % p.outputDims[0];                                     \
    q /= p.outputDims[0];                                                     \
    coordinate[1] = q % p.outputDims[1];                                     \
    q /= p.outputDims[1];                                                     \
    coordinate[2] = q % p.outputDims[2];                                     \
    coordinate[3] = q / p.outputDims[2];                                     \
    long inputIndex = p.inputOffset;                                         \
    long outputIndex = p.outputOffset;                                       \
    for (uint dimension = 0; dimension < 4; ++dimension) {                   \
        const long outputCoordinate = long(coordinate[dimension]);           \
        long sourceCoordinate;                                               \
        if (p.isSequence[dimension] != 0) {                                  \
            sourceCoordinate = p.offsets[dimension] +                        \
                               outputCoordinate * p.steps[dimension];         \
        } else {                                                             \
            const long indexAddress = p.indexOffsets[dimension] +            \
                                      outputCoordinate *                     \
                                          p.indexStrides[dimension];         \
            switch (dimension) {                                             \
                case 0: sourceCoordinate = long(index0[indexAddress]); break;\
                case 1: sourceCoordinate = long(index1[indexAddress]); break;\
                case 2: sourceCoordinate = long(index2[indexAddress]); break;\
                default: sourceCoordinate = long(index3[indexAddress]);      \
            }                                                                \
        }                                                                    \
        inputIndex +=                                                        \
            trim_index(sourceCoordinate, long(p.inputDims[dimension])) *     \
            p.inputStrides[dimension];                                       \
        outputIndex += outputCoordinate * p.outputStrides[dimension];        \
    }                                                                        \
    output[outputIndex] = input[inputIndex];                                 \
}

DEFINE_INDEX(index_float, float)
DEFINE_INDEX(index_cfloat, float2)
DEFINE_INDEX(index_int, int)
DEFINE_INDEX(index_uint, uint)
DEFINE_INDEX(index_long, long)
DEFINE_INDEX(index_ulong, ulong)
DEFINE_INDEX(index_char, char)
DEFINE_INDEX(index_uchar, uchar)
DEFINE_INDEX(index_short, short)
DEFINE_INDEX(index_ushort, ushort)
DEFINE_INDEX(index_half, half)
