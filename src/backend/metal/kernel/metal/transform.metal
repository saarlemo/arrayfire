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

struct TransformParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    uint method;
    uint inverse;
    uint perspective;
};

void invert_transform(thread float* output, const device float* input,
                      bool perspective) {
    if (perspective) {
        output[0] = input[4] * input[8] - input[5] * input[7];
        output[1] = -(input[1] * input[8] - input[2] * input[7]);
        output[2] = input[1] * input[5] - input[2] * input[4];
        output[3] = -(input[3] * input[8] - input[5] * input[6]);
        output[4] = input[0] * input[8] - input[2] * input[6];
        output[5] = -(input[0] * input[5] - input[2] * input[3]);
        output[6] = input[3] * input[7] - input[4] * input[6];
        output[7] = -(input[0] * input[7] - input[1] * input[6]);
        output[8] = input[0] * input[4] - input[1] * input[3];
        const float determinant = input[0] * output[0] +
                                  input[1] * output[3] +
                                  input[2] * output[6];
        for (uint i = 0; i < 9; ++i) output[i] /= determinant;
    } else {
        const float determinant = input[0] * input[4] - input[1] * input[3];
        output[0] = input[4] / determinant;
        output[1] = input[3] / determinant;
        output[3] = input[1] / determinant;
        output[4] = input[0] / determinant;
        output[2] = input[2] * -output[0] + input[5] * -output[1];
        output[5] = input[2] * -output[3] + input[5] * -output[4];
    }
}

#define DEFINE_TRANSFORM(NAME, TYPE)                                          \
kernel void NAME(const device TYPE* input [[buffer(0)]],                      \
                 const device float* transform [[buffer(1)]],                 \
                 device TYPE* output [[buffer(2)]],                           \
                 constant TransformParams& params [[buffer(3)]],              \
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
    float matrix[9] = {0.0f};                                                  \
    const uint length = params.perspective != 0 ? 9 : 6;                      \
    if (params.inverse != 0) {                                                 \
        for (uint i = 0; i < length; ++i) matrix[i] = transform[i];           \
    } else {                                                                   \
        invert_transform(matrix, transform, params.perspective != 0);         \
    }                                                                          \
    float sourceX = float(x) * matrix[0] + float(y) * matrix[1] + matrix[2];  \
    float sourceY = float(x) * matrix[3] + float(y) * matrix[4] + matrix[5];  \
    if (params.perspective != 0) {                                             \
        const float scale = float(x) * matrix[6] + float(y) * matrix[7] +     \
                            matrix[8];                                         \
        sourceX /= scale;                                                      \
        sourceY /= scale;                                                      \
    }                                                                          \
    const ulong outputOffset = x * params.outputStrides[0] +                  \
                               y * params.outputStrides[1] +                  \
                               z * params.outputStrides[2] +                  \
                               w * params.outputStrides[3];                   \
    if (sourceX >= -0.0001f && sourceY >= -0.0001f &&                         \
        sourceX < float(params.inputDims[0]) &&                               \
        sourceY < float(params.inputDims[1])) {                               \
        const long ix = params.method == 4 ? long(floor(sourceX))             \
                                            : long(round(sourceX));           \
        const long iy = params.method == 4 ? long(floor(sourceY))             \
                                            : long(round(sourceY));           \
        output[outputOffset] =                                                 \
            input[ulong(ix) * params.inputStrides[0] +                        \
                  ulong(iy) * params.inputStrides[1] +                        \
                  z * params.inputStrides[2] + w * params.inputStrides[3]];   \
    } else {                                                                   \
        output[outputOffset] = TYPE(0);                                        \
    }                                                                          \
}

DEFINE_TRANSFORM(transform_float, float)
DEFINE_TRANSFORM(transform_cfloat, float2)
DEFINE_TRANSFORM(transform_int, int)
DEFINE_TRANSFORM(transform_uint, uint)
DEFINE_TRANSFORM(transform_long, long)
DEFINE_TRANSFORM(transform_ulong, ulong)
DEFINE_TRANSFORM(transform_char, char)
DEFINE_TRANSFORM(transform_uchar, uchar)
DEFINE_TRANSFORM(transform_short, short)
DEFINE_TRANSFORM(transform_ushort, ushort)
