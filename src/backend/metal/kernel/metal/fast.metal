/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct FastParams {
    ulong rows;
    ulong columns;
    float threshold;
    uint arcLength;
    uint nonmax;
    uint maxFeatures;
    uint edge;
};

inline int fastDy(int i) {
    return i >= 8 ? clamp(12 - i, -3, 3) : clamp(i - 4, -3, 3);
}
inline int fastDx(int i) { return i < 12 ? fastDy(i + 4) : fastDy(i - 12); }

template<typename T>
inline int fastTest(const device T* image, float center, float threshold,
                    int row, int column, ulong rows) {
    const float value = float(image[ulong(column) * rows + ulong(row)]);
    return -(value < center - threshold) + (value > center + threshold);
}

template<typename T>
inline void fastLocatePixel(
    const device T* image, device float* scoreImage, device float* xOutput,
    device float* yOutput, device float* scoreOutput,
    device atomic_uint* count, constant FastParams& p, uint gid) {
    const ulong validRows = p.rows > 2ul * p.edge ? p.rows - 2ul * p.edge : 0;
    const ulong validColumns =
        p.columns > 2ul * p.edge ? p.columns - 2ul * p.edge : 0;
    const ulong candidateCount = validRows * validColumns;
    if (validRows == 0 || ulong(gid) >= candidateCount) return;

    const int row = int(p.edge + ulong(gid) % validRows);
    const int column = int(p.edge + ulong(gid) / validRows);
    // The public API documents edge >= 3 but does not enforce it. Keep the
    // circle reads in bounds for smaller edge values as well.
    if (row < 3 || column < 3 || ulong(row + 3) >= p.rows ||
        ulong(column + 3) >= p.columns)
        return;

    const ulong centerIndex = ulong(column) * p.rows + ulong(row);
    const float center = float(image[centerIndex]);
    int d = fastTest(image, center, p.threshold, row - 3, column, p.rows) |
            fastTest(image, center, p.threshold, row + 3, column, p.rows);
    if (d == 0) return;
    d &= fastTest(image, center, p.threshold, row - 2, column + 2, p.rows) |
         fastTest(image, center, p.threshold, row + 2, column - 2, p.rows);
    d &= fastTest(image, center, p.threshold, row, column + 3, p.rows) |
         fastTest(image, center, p.threshold, row, column - 3, p.rows);
    d &= fastTest(image, center, p.threshold, row + 2, column + 2, p.rows) |
         fastTest(image, center, p.threshold, row - 2, column - 2, p.rows);
    if (d == 0) return;
    d &= fastTest(image, center, p.threshold, row - 3, column + 1, p.rows) |
         fastTest(image, center, p.threshold, row + 3, column - 1, p.rows);
    d &= fastTest(image, center, p.threshold, row - 1, column + 3, p.rows) |
         fastTest(image, center, p.threshold, row + 1, column - 3, p.rows);
    d &= fastTest(image, center, p.threshold, row + 1, column + 3, p.rows) |
         fastTest(image, center, p.threshold, row - 1, column - 3, p.rows);
    d &= fastTest(image, center, p.threshold, row + 3, column + 1, p.rows) |
         fastTest(image, center, p.threshold, row - 3, column - 1, p.rows);
    if (d == 0) return;

    int sum = 0;
    for (uint i = 0; i < p.arcLength; ++i)
        sum += fastTest(image, center, p.threshold, row + fastDy(i),
                        column + fastDx(i), p.rows);
    int maxSum = max(0, sum), minSum = min(0, sum);
    for (uint i = p.arcLength; i < 16; ++i) {
        sum -= fastTest(image, center, p.threshold,
                        row + fastDy(i - p.arcLength),
                        column + fastDx(i - p.arcLength), p.rows);
        sum += fastTest(image, center, p.threshold, row + fastDy(i),
                        column + fastDx(i), p.rows);
        maxSum = max(maxSum, sum);
        minSum = min(minSum, sum);
    }
    for (uint i = 0; i < p.arcLength - 1; ++i) {
        sum -= fastTest(image, center, p.threshold,
                        row + fastDy(16 - p.arcLength + i),
                        column + fastDx(16 - p.arcLength + i), p.rows);
        sum += fastTest(image, center, p.threshold, row + fastDy(i),
                        column + fastDx(i), p.rows);
        maxSum = max(maxSum, sum);
        minSum = min(minSum, sum);
    }

    if (maxSum != int(p.arcLength) && minSum != -int(p.arcLength)) return;

    float bright = 0.0f, dark = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        const float value =
            float(image[ulong(column + fastDx(i)) * p.rows +
                        ulong(row + fastDy(i))]);
        bright += (value > center + p.threshold) *
                  (fabs(value - center) - p.threshold);
        dark += (value < center - p.threshold) *
                (fabs(center - value) - p.threshold);
    }

    const uint index =
        atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    if (index < p.maxFeatures) {
        const float response = max(bright, dark);
        xOutput[index] = float(column);
        yOutput[index] = float(row);
        scoreOutput[index] = response;
        if (p.nonmax) scoreImage[centerIndex] = response;
    }
}

#define DEFINE_FAST_LOCATE(NAME, TYPE)                                       \
kernel void NAME(const device TYPE* image [[buffer(0)]],                    \
                 device float* scoreImage [[buffer(1)]],                    \
                 device float* xOutput [[buffer(2)]],                       \
                 device float* yOutput [[buffer(3)]],                       \
                 device float* scoreOutput [[buffer(4)]],                   \
                 device atomic_uint* count [[buffer(5)]],                   \
                 constant FastParams& p [[buffer(6)]],                      \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong validRows = p.rows > 2ul * p.edge ? p.rows - 2ul * p.edge : 0; \
    const ulong validColumns = p.columns > 2ul * p.edge                  \
                                   ? p.columns - 2ul * p.edge               \
                                   : 0;                                    \
    if (ulong(gid) < validRows * validColumns)                             \
        fastLocatePixel(image, scoreImage, xOutput, yOutput, scoreOutput,  \
                        count, p, gid);                                     \
}

DEFINE_FAST_LOCATE(fast_locate_float, float)
DEFINE_FAST_LOCATE(fast_locate_half, half)
DEFINE_FAST_LOCATE(fast_locate_int, int)
DEFINE_FAST_LOCATE(fast_locate_uint, uint)
DEFINE_FAST_LOCATE(fast_locate_char, char)
DEFINE_FAST_LOCATE(fast_locate_uchar, uchar)
DEFINE_FAST_LOCATE(fast_locate_short, short)
DEFINE_FAST_LOCATE(fast_locate_ushort, ushort)

kernel void fast_nonmax_float(
    const device float* score [[buffer(0)]],
    const device float* xInput [[buffer(1)]],
    const device float* yInput [[buffer(2)]],
    device float* xOutput [[buffer(3)]],
    device float* yOutput [[buffer(4)]],
    device float* scoreOutput [[buffer(5)]],
    device atomic_uint* count [[buffer(6)]],
    constant FastParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.maxFeatures) return;
    const uint x = uint(round(xInput[gid]));
    const uint y = uint(round(yInput[gid]));
    if (x <= p.edge + 1 || y <= p.edge + 1 ||
        x >= p.columns - p.edge - 1 || y >= p.rows - p.edge - 1)
        return;
    const ulong center = ulong(y) + ulong(p.rows) * ulong(x);
    const float value = score[center];
    float maxValue = score[center - 1 - p.rows];
    maxValue = max(maxValue, score[center - 1]);
    maxValue = max(maxValue, score[center - 1 + p.rows]);
    maxValue = max(maxValue, score[center - p.rows]);
    maxValue = max(maxValue, score[center + p.rows]);
    maxValue = max(maxValue, score[center + 1 - p.rows]);
    maxValue = max(maxValue, score[center + 1]);
    maxValue = max(maxValue, score[center + 1 + p.rows]);
    if (value > maxValue) {
        const uint outputIndex = atomic_fetch_add_explicit(
            count, 1u, memory_order_relaxed);
        if (outputIndex < p.maxFeatures) {
            xOutput[outputIndex] = float(x);
            yOutput[outputIndex] = float(y);
            scoreOutput[outputIndex] = value;
        }
    }
}
