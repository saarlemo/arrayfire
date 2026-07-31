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

struct ApproxParams {
    ulong outputDims[4];
    long outputStrides[4];
    ulong inputDims[4];
    long inputStrides[4];
    ulong xDims[4];
    long xStrides[4];
    long yStrides[4];
    long outputOffset;
    long inputOffset;
    long xOffset;
    long yOffset;
    uint xDimension;
    uint yDimension;
    uint method;
    float xBegin;
    float xStep;
    float yBegin;
    float yStep;
    float offGrid;
};

float approxLinear(float left, float right, float ratio) {
    if (ratio == 0.0f) return left;
    if (ratio == 1.0f) return right;
    if (isinf(left) && !isinf(right)) return left;
    if (!isinf(left) && isinf(right)) return right;
    if (isinf(left) && isinf(right) && signbit(left) == signbit(right))
        return left;
    return (1.0f - ratio) * left + ratio * right;
}

float2 approxLinear(float2 left, float2 right, float ratio) {
    return float2(approxLinear(left.x, right.x, ratio),
                  approxLinear(left.y, right.y, ratio));
}

template<typename T>
T approxCubic(T v0, T v1, T v2, T v3, float ratio, bool spline) {
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

#define DEFINE_APPROX1(NAME, TYPE, OFF_GRID)                                 \
kernel void NAME(const device TYPE* input [[buffer(0)]],                     \
                 const device float* positions [[buffer(1)]],                \
                 device TYPE* output [[buffer(2)]],                          \
                 constant ApproxParams& p [[buffer(3)]],                     \
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
    long outputIndex = p.outputOffset, inputBase = p.inputOffset;            \
    long positionIndex = p.xOffset;                                          \
    for (uint dimension = 0; dimension < 4; ++dimension) {                   \
        outputIndex +=                                                        \
            long(coordinate[dimension]) * p.outputStrides[dimension];        \
        if (dimension != p.xDimension)                                       \
            inputBase +=                                                      \
                long(coordinate[dimension]) * p.inputStrides[dimension];     \
        if (p.xDims[dimension] > 1)                                          \
            positionIndex +=                                                 \
                long(coordinate[dimension]) * p.xStrides[dimension];         \
    }                                                                        \
    const float x =                                                          \
        (positions[positionIndex] - p.xBegin) / p.xStep;                     \
    if (!isfinite(x) || x < 0.0f ||                                         \
        float(p.inputDims[p.xDimension]) < x + 1.0f) {                       \
        output[outputIndex] = OFF_GRID;                                      \
        return;                                                              \
    }                                                                        \
    const long stride = p.inputStrides[p.xDimension];                        \
    if (p.method == 0 || p.method == 4) {                                   \
        const long index = p.method == 4 ? long(floor(x))                    \
                                          : long(floor(x + 0.5f));           \
        output[outputIndex] = input[inputBase + index * stride];             \
        return;                                                              \
    }                                                                        \
    const long grid = long(floor(x));                                        \
    float ratio = x - floor(x);                                              \
    if (p.method == 1 || p.method == 5) {                                   \
        if (p.method == 5)                                                   \
            ratio = (1.0f - cos(ratio * M_PI_F)) * 0.5f;                    \
        const long next = min(grid + 1, long(p.inputDims[p.xDimension]) - 1);\
        output[outputIndex] = approxLinear(                                  \
            input[inputBase + grid * stride],                                \
            input[inputBase + next * stride], ratio);                        \
        return;                                                              \
    }                                                                        \
    const long limit = long(p.inputDims[p.xDimension]) - 1;                 \
    const long i0 = clamp(grid - 1, 0l, limit);                              \
    const long i1 = clamp(grid, 0l, limit);                                  \
    const long i2 = clamp(grid + 1, 0l, limit);                              \
    const long i3 = clamp(grid + 2, 0l, limit);                              \
    output[outputIndex] = approxCubic(                                       \
        input[inputBase + i0 * stride], input[inputBase + i1 * stride],      \
        input[inputBase + i2 * stride], input[inputBase + i3 * stride],      \
        ratio, p.method == 8);                                               \
}

#define DEFINE_APPROX2(NAME, TYPE, OFF_GRID)                                 \
kernel void NAME(const device TYPE* input [[buffer(0)]],                     \
                 const device float* xPositions [[buffer(1)]],               \
                 const device float* yPositions [[buffer(2)]],               \
                 device TYPE* output [[buffer(3)]],                          \
                 constant ApproxParams& p [[buffer(4)]],                     \
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
    long outputIndex = p.outputOffset, inputBase = p.inputOffset;            \
    long xIndex = p.xOffset, yIndex = p.yOffset;                             \
    for (uint dimension = 0; dimension < 4; ++dimension) {                   \
        outputIndex +=                                                        \
            long(coordinate[dimension]) * p.outputStrides[dimension];        \
        if (dimension != p.xDimension && dimension != p.yDimension)         \
            inputBase +=                                                      \
                long(coordinate[dimension]) * p.inputStrides[dimension];     \
        if (p.xDims[dimension] > 1) {                                        \
            xIndex += long(coordinate[dimension]) * p.xStrides[dimension];   \
            yIndex += long(coordinate[dimension]) * p.yStrides[dimension];   \
        }                                                                    \
    }                                                                        \
    const float x = (xPositions[xIndex] - p.xBegin) / p.xStep;              \
    const float y = (yPositions[yIndex] - p.yBegin) / p.yStep;              \
    if (!isfinite(x) || !isfinite(y) || x < 0.0f || y < 0.0f ||            \
        float(p.inputDims[p.xDimension]) < x + 1.0f ||                       \
        float(p.inputDims[p.yDimension]) < y + 1.0f) {                       \
        output[outputIndex] = OFF_GRID;                                      \
        return;                                                              \
    }                                                                        \
    const long xStride = p.inputStrides[p.xDimension];                       \
    const long yStride = p.inputStrides[p.yDimension];                       \
    if (p.method == 0 || p.method == 4) {                                   \
        const long ix = p.method == 4 ? long(floor(x))                      \
                                       : long(floor(x + 0.5f));              \
        const long iy = p.method == 4 ? long(floor(y))                      \
                                       : long(floor(y + 0.5f));              \
        output[outputIndex] =                                                \
            input[inputBase + ix * xStride + iy * yStride];                  \
        return;                                                              \
    }                                                                        \
    const long gx = long(floor(x)), gy = long(floor(y));                    \
    float xRatio = x - floor(x), yRatio = y - floor(y);                     \
    if (p.method == 1 || p.method == 2 || p.method == 5 ||                  \
        p.method == 6) {                                                     \
        if (p.method == 5 || p.method == 6) {                               \
            xRatio = (1.0f - cos(xRatio * M_PI_F)) * 0.5f;                  \
            yRatio = (1.0f - cos(yRatio * M_PI_F)) * 0.5f;                  \
        }                                                                    \
        const long nx = min(gx + 1, long(p.inputDims[p.xDimension]) - 1);   \
        const long ny = min(gy + 1, long(p.inputDims[p.yDimension]) - 1);   \
        const TYPE top = approxLinear(                                       \
            input[inputBase + gx * xStride + gy * yStride],                 \
            input[inputBase + nx * xStride + gy * yStride],                 \
            xRatio);                                                         \
        const TYPE bottom = approxLinear(                                    \
            input[inputBase + gx * xStride + ny * yStride],                 \
            input[inputBase + nx * xStride + ny * yStride],                 \
            xRatio);                                                         \
        output[outputIndex] = approxLinear(top, bottom, yRatio);             \
        return;                                                              \
    }                                                                        \
    const long xLimit = long(p.inputDims[p.xDimension]) - 1;                \
    const long yLimit = long(p.inputDims[p.yDimension]) - 1;                \
    TYPE rows[4];                                                            \
    for (long j = 0; j < 4; ++j) {                                          \
        const long sy = clamp(gy + j - 1, 0l, yLimit);                      \
        TYPE values[4];                                                      \
        for (long i = 0; i < 4; ++i) {                                      \
            const long sx = clamp(gx + i - 1, 0l, xLimit);                  \
            values[i] = input[inputBase + sx * xStride + sy * yStride];     \
        }                                                                    \
        rows[j] = approxCubic(values[0], values[1], values[2], values[3],    \
                              xRatio, p.method == 8 || p.method == 9);       \
    }                                                                        \
    output[outputIndex] =                                                    \
        approxCubic(rows[0], rows[1], rows[2], rows[3], yRatio,              \
                    p.method == 8 || p.method == 9);                         \
}

DEFINE_APPROX1(approx1_float, float, p.offGrid)
DEFINE_APPROX1(approx1_cfloat, float2, float2(p.offGrid, 0.0f))
DEFINE_APPROX2(approx2_float, float, p.offGrid)
DEFINE_APPROX2(approx2_cfloat, float2, float2(p.offGrid, 0.0f))
