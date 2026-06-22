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
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    long offsets[4];
    long steps[4];
};

long trim_index(long index, long length) {
    if (index < 0) return (-index - 1) % length;
    if (index >= length) return length - (index % length) - 1;
    return index;
}

#define DEFINE_INDEX(NAME, TYPE)                                              \
kernel void NAME(const device TYPE* input [[buffer(0)]],                      \
                 device TYPE* output [[buffer(1)]],                           \
                 constant IndexParams& params [[buffer(2)]],                  \
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
    const long ix = trim_index(params.offsets[0] + long(x) * params.steps[0], \
                               long(params.inputDims[0]));                     \
    const long iy = trim_index(params.offsets[1] + long(y) * params.steps[1], \
                               long(params.inputDims[1]));                     \
    const long iz = trim_index(params.offsets[2] + long(z) * params.steps[2], \
                               long(params.inputDims[2]));                     \
    const long iw = trim_index(params.offsets[3] + long(w) * params.steps[3], \
                               long(params.inputDims[3]));                     \
    const ulong inputOffset = ulong(ix) * params.inputStrides[0] +            \
                              ulong(iy) * params.inputStrides[1] +            \
                              ulong(iz) * params.inputStrides[2] +            \
                              ulong(iw) * params.inputStrides[3];             \
    const ulong outputOffset = x * params.outputStrides[0] +                  \
                               y * params.outputStrides[1] +                  \
                               z * params.outputStrides[2] +                  \
                               w * params.outputStrides[3];                   \
    output[outputOffset] = input[inputOffset];                                \
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
