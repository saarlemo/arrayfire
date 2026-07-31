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
inline void fastLocateImpl(
    const device T* image, device float* scoreImage, device float* xOutput,
    device float* yOutput, device float* scoreOutput, device uint* count,
    constant FastParams& p) {
    uint found = 0;
    for (int row = int(p.edge); row < int(p.rows - p.edge); ++row) {
        for (int column = int(p.edge); column < int(p.columns - p.edge);
             ++column) {
            const float center =
                float(image[ulong(column) * p.rows + ulong(row)]);
            int d = fastTest(image, center, p.threshold, row - 3, column, p.rows) |
                    fastTest(image, center, p.threshold, row + 3, column, p.rows);
            if (d == 0) continue;
            d &= fastTest(image, center, p.threshold, row - 2, column + 2,
                          p.rows) |
                 fastTest(image, center, p.threshold, row + 2, column - 2,
                          p.rows);
            d &= fastTest(image, center, p.threshold, row, column + 3, p.rows) |
                 fastTest(image, center, p.threshold, row, column - 3, p.rows);
            d &= fastTest(image, center, p.threshold, row + 2, column + 2,
                          p.rows) |
                 fastTest(image, center, p.threshold, row - 2, column - 2,
                          p.rows);
            if (d == 0) continue;
            d &= fastTest(image, center, p.threshold, row - 3, column + 1,
                          p.rows) |
                 fastTest(image, center, p.threshold, row + 3, column - 1,
                          p.rows);
            d &= fastTest(image, center, p.threshold, row - 1, column + 3,
                          p.rows) |
                 fastTest(image, center, p.threshold, row + 1, column - 3,
                          p.rows);
            d &= fastTest(image, center, p.threshold, row + 1, column + 3,
                          p.rows) |
                 fastTest(image, center, p.threshold, row - 1, column - 3,
                          p.rows);
            d &= fastTest(image, center, p.threshold, row + 3, column + 1,
                          p.rows) |
                 fastTest(image, center, p.threshold, row - 3, column - 1,
                          p.rows);
            if (d == 0) continue;

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

            float bright = 0.0f, dark = 0.0f;
            for (uint i = 0; i < 16; ++i) {
                const float value = float(
                    image[ulong(column + fastDx(i)) * p.rows +
                          ulong(row + fastDy(i))]);
                bright += (value > center + p.threshold) *
                          (fabs(value - center) - p.threshold);
                dark += (value < center - p.threshold) *
                        (fabs(center - value) - p.threshold);
            }
            if (maxSum == int(p.arcLength) || minSum == -int(p.arcLength)) {
                const uint index = found++;
                if (index < p.maxFeatures) {
                    const float response = max(bright, dark);
                    xOutput[index] = float(column);
                    yOutput[index] = float(row);
                    scoreOutput[index] = response;
                    if (p.nonmax)
                        scoreImage[ulong(column) * p.rows + ulong(row)] = response;
                }
            }
        }
    }
    *count = found;
}

#define DEFINE_FAST_LOCATE(NAME, TYPE)                                       \
kernel void NAME(const device TYPE* image [[buffer(0)]],                    \
                 device float* scoreImage [[buffer(1)]],                    \
                 device float* xOutput [[buffer(2)]],                       \
                 device float* yOutput [[buffer(3)]],                       \
                 device float* scoreOutput [[buffer(4)]],                   \
                 device uint* count [[buffer(5)]],                          \
                 constant FastParams& p [[buffer(6)]],                      \
                 uint gid [[thread_position_in_grid]]) {                    \
    if (gid == 0)                                                           \
        fastLocateImpl(image, scoreImage, xOutput, yOutput, scoreOutput,    \
                       count, p);                                           \
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
