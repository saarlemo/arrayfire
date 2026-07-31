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
    long outputStrides[4];
    ulong inputDims[4];
    long inputStrides[4];
    ulong transformDims[4];
    long transformStrides[4];
    long outputOffset;
    long inputOffset;
    long transformOffset;
    uint method;
    uint inverse;
    uint perspective;
};

float transformCoefficient(const device float* input, long columnStride,
                           long rowStride, ulong columns, uint index) {
    const ulong column = ulong(index) % columns;
    const ulong row = ulong(index) / columns;
    return input[long(column) * columnStride + long(row) * rowStride];
}

void transformInvert(thread float* output, const device float* input,
                     long columnStride, long rowStride, ulong columns,
                     bool perspective) {
    const float i0 =
        transformCoefficient(input, columnStride, rowStride, columns, 0);
    const float i1 =
        transformCoefficient(input, columnStride, rowStride, columns, 1);
    const float i2 =
        transformCoefficient(input, columnStride, rowStride, columns, 2);
    const float i3 =
        transformCoefficient(input, columnStride, rowStride, columns, 3);
    const float i4 =
        transformCoefficient(input, columnStride, rowStride, columns, 4);
    const float i5 =
        transformCoefficient(input, columnStride, rowStride, columns, 5);
    if (perspective) {
        const float i6 =
            transformCoefficient(input, columnStride, rowStride, columns, 6);
        const float i7 =
            transformCoefficient(input, columnStride, rowStride, columns, 7);
        const float i8 =
            transformCoefficient(input, columnStride, rowStride, columns, 8);
        output[0] = i4 * i8 - i5 * i7;
        output[1] = -(i1 * i8 - i2 * i7);
        output[2] = i1 * i5 - i2 * i4;
        output[3] = -(i3 * i8 - i5 * i6);
        output[4] = i0 * i8 - i2 * i6;
        output[5] = -(i0 * i5 - i2 * i3);
        output[6] = i3 * i7 - i4 * i6;
        output[7] = -(i0 * i7 - i1 * i6);
        output[8] = i0 * i4 - i1 * i3;
        const float determinant =
            i0 * output[0] + i1 * output[3] + i2 * output[6];
        for (uint i = 0; i < 9; ++i) output[i] /= determinant;
    } else {
        const float determinant = i0 * i4 - i1 * i3;
        output[0] = i4 / determinant;
        output[1] = i3 / determinant;
        output[3] = i1 / determinant;
        output[4] = i0 / determinant;
        output[2] = i2 * -output[0] + i5 * -output[1];
        output[5] = i2 * -output[3] + i5 * -output[4];
    }
}

template<typename T>
T transformLinear(T left, T right, float ratio) {
    return fma(T(ratio), right - left, left);
}

template<typename T>
T transformCubic(T v0, T v1, T v2, T v3, float ratio, bool spline) {
    T a0, a1, a2, a3;
    if (spline) {
        a0 = -0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3;
        a1 = v0 - 2.5f * v1 + 2.0f * v2 - 0.5f * v3;
        a2 = -0.5f * v0 + 0.5f * v2;
        a3 = v1;
    } else {
        a0 = v3 - v2 - v0 + v1;
        a1 = v0 - v1 - a0;
        a2 = v2 - v0;
        a3 = v1;
    }
    const float ratio2 = ratio * ratio;
    return a0 * ratio2 * ratio + a1 * ratio2 + a2 * ratio + a3;
}

#define DEFINE_TRANSFORM(NAME, TYPE, WORK_TYPE)                               \
kernel void NAME(const device TYPE* input [[buffer(0)]],                      \
                 const device float* transform [[buffer(1)]],                 \
                 device TYPE* output [[buffer(2)]],                           \
                 constant TransformParams& p [[buffer(3)]],                   \
                 uint gid [[thread_position_in_grid]]) {                      \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                   \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                  \
    ulong q = gid;                                                             \
    const ulong x = q % p.outputDims[0];                                      \
    q /= p.outputDims[0];                                                      \
    const ulong y = q % p.outputDims[1];                                      \
    q /= p.outputDims[1];                                                      \
    const ulong z = q % p.outputDims[2];                                      \
    const ulong w = q / p.outputDims[2];                                      \
    const ulong inputZ = p.inputDims[2] > 1 ? z : 0;                          \
    const ulong inputW = p.inputDims[3] > 1 ? w : 0;                          \
    const ulong transformZ = p.transformDims[2] > 1 ? z : 0;                 \
    const ulong transformW = p.transformDims[3] > 1 ? w : 0;                 \
    const long outputIndex =                                                   \
        p.outputOffset + long(x) * p.outputStrides[0] +                       \
        long(y) * p.outputStrides[1] + long(z) * p.outputStrides[2] +         \
        long(w) * p.outputStrides[3];                                         \
    const long inputBatch =                                                    \
        p.inputOffset + long(inputZ) * p.inputStrides[2] +                    \
        long(inputW) * p.inputStrides[3];                                     \
    const long transformIndex =                                                \
        p.transformOffset + long(transformZ) * p.transformStrides[2] +        \
        long(transformW) * p.transformStrides[3];                             \
    const device float* transformMatrix = transform + transformIndex;          \
    float matrix[9] = {0.0f};                                                  \
    const uint matrixLength = p.perspective != 0 ? 9 : 6;                     \
    if (p.inverse != 0) {                                                      \
        for (uint i = 0; i < matrixLength; ++i)                               \
            matrix[i] = transformCoefficient(                                 \
                transformMatrix, p.transformStrides[0],                       \
                p.transformStrides[1], p.transformDims[0], i);                \
    } else {                                                                   \
        transformInvert(matrix, transformMatrix, p.transformStrides[0],       \
                        p.transformStrides[1], p.transformDims[0],             \
                        p.perspective != 0);                                   \
    }                                                                          \
    float sourceX = float(x) * matrix[0] + float(y) * matrix[1] + matrix[2];  \
    float sourceY = float(x) * matrix[3] + float(y) * matrix[4] + matrix[5];  \
    if (p.perspective != 0) {                                                  \
        const float scale = float(x) * matrix[6] + float(y) * matrix[7] +     \
                            matrix[8];                                         \
        sourceX /= scale;                                                      \
        sourceY /= scale;                                                      \
    }                                                                          \
    const bool inside = sourceX >= -0.0001f && sourceY >= -0.0001f &&         \
                        sourceX < float(p.inputDims[0]) &&                     \
                        sourceY < float(p.inputDims[1]);                       \
    if (!inside) {                                                             \
        output[outputIndex] = TYPE(0);                                         \
        return;                                                                \
    }                                                                          \
    if (p.method == 0 || p.method == 4) {                                     \
        const long ix = p.method == 4 ? long(floor(sourceX))                  \
                                      : long(round(sourceX));                  \
        const long iy = p.method == 4 ? long(floor(sourceY))                  \
                                      : long(round(sourceY));                  \
        if (ix >= 0 && iy >= 0 && ix < long(p.inputDims[0]) &&                \
            iy < long(p.inputDims[1])) {                                      \
            output[outputIndex] =                                              \
                input[inputBatch + ix * p.inputStrides[0] +                   \
                      iy * p.inputStrides[1]];                                 \
        } else {                                                               \
            output[outputIndex] = TYPE(0);                                     \
        }                                                                      \
        return;                                                                \
    }                                                                          \
    const long gridX = long(floor(sourceX));                                  \
    const long gridY = long(floor(sourceY));                                  \
    const float xFraction = sourceX - floor(sourceX);                         \
    const float yFraction = sourceY - floor(sourceY);                         \
    if (p.method == 2 || p.method == 6) {                                     \
        const long x0 = clamp(gridX, 0l, long(p.inputDims[0]) - 1);           \
        const long x1 = clamp(gridX + 1, 0l, long(p.inputDims[0]) - 1);       \
        const long y0 = clamp(gridY, 0l, long(p.inputDims[1]) - 1);           \
        const long y1 = clamp(gridY + 1, 0l, long(p.inputDims[1]) - 1);       \
        float xRatio = xFraction;                                              \
        float yRatio = yFraction;                                              \
        if (p.method == 6) {                                                   \
            xRatio = (1.0f - cos(xRatio * M_PI_F)) * 0.5f;                   \
            yRatio = (1.0f - cos(yRatio * M_PI_F)) * 0.5f;                   \
        }                                                                      \
        const WORK_TYPE v00 = WORK_TYPE(                                      \
            input[inputBatch + x0 * p.inputStrides[0] +                       \
                  y0 * p.inputStrides[1]]);                                   \
        const WORK_TYPE v01 = WORK_TYPE(                                      \
            input[inputBatch + x1 * p.inputStrides[0] +                       \
                  y0 * p.inputStrides[1]]);                                   \
        const WORK_TYPE v10 = WORK_TYPE(                                      \
            input[inputBatch + x0 * p.inputStrides[0] +                       \
                  y1 * p.inputStrides[1]]);                                   \
        const WORK_TYPE v11 = WORK_TYPE(                                      \
            input[inputBatch + x1 * p.inputStrides[0] +                       \
                  y1 * p.inputStrides[1]]);                                   \
        output[outputIndex] = TYPE(transformLinear(                           \
            transformLinear(v00, v01, xRatio),                               \
            transformLinear(v10, v11, xRatio), yRatio));                     \
        return;                                                                \
    }                                                                          \
    const bool spline = p.method == 9;                                        \
    WORK_TYPE rows[4];                                                         \
    for (long j = 0; j < 4; ++j) {                                           \
        const long sourceRow =                                                 \
            clamp(gridY + j - 1, 0l, long(p.inputDims[1]) - 1);              \
        WORK_TYPE values[4];                                                   \
        for (long i = 0; i < 4; ++i) {                                       \
            const long sourceColumn =                                         \
                clamp(gridX + i - 1, 0l, long(p.inputDims[0]) - 1);          \
            values[i] = WORK_TYPE(                                            \
                input[inputBatch + sourceColumn * p.inputStrides[0] +         \
                      sourceRow * p.inputStrides[1]]);                         \
        }                                                                      \
        rows[j] = transformCubic(values[0], values[1], values[2], values[3],  \
                                 xFraction, spline);                           \
    }                                                                          \
    output[outputIndex] = TYPE(transformCubic(                                \
        rows[0], rows[1], rows[2], rows[3], yFraction, spline));              \
}

DEFINE_TRANSFORM(transform_float, float, float)
DEFINE_TRANSFORM(transform_cfloat, float2, float2)
DEFINE_TRANSFORM(transform_int, int, float)
DEFINE_TRANSFORM(transform_uint, uint, float)
DEFINE_TRANSFORM(transform_long, long, float)
DEFINE_TRANSFORM(transform_ulong, ulong, float)
DEFINE_TRANSFORM(transform_char, char, float)
DEFINE_TRANSFORM(transform_uchar, uchar, float)
DEFINE_TRANSFORM(transform_short, short, float)
DEFINE_TRANSFORM(transform_ushort, ushort, float)
