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
inline int fastTest(const device float* image, float center, float threshold,
                    int row, int column, ulong rows) {
    const float value = image[ulong(column) * rows + ulong(row)];
    return -(value < center - threshold) + (value > center + threshold);
}

kernel void fast_locate_float(
    const device float* image [[buffer(0)]], device float* scoreImage [[buffer(1)]],
    device float* xOutput [[buffer(2)]], device float* yOutput [[buffer(3)]],
    device float* scoreOutput [[buffer(4)]], device uint* count [[buffer(5)]],
    constant FastParams& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    uint found = 0;
    for (int row = int(p.edge); row < int(p.rows - p.edge); ++row) {
        for (int column = int(p.edge); column < int(p.columns - p.edge); ++column) {
            const float center = image[ulong(column) * p.rows + ulong(row)];
            int d = fastTest(image, center, p.threshold, row - 3, column, p.rows) |
                    fastTest(image, center, p.threshold, row + 3, column, p.rows);
            if (d == 0) continue;
            d &= fastTest(image, center, p.threshold, row - 2, column + 2, p.rows) |
                 fastTest(image, center, p.threshold, row + 2, column - 2, p.rows);
            d &= fastTest(image, center, p.threshold, row, column + 3, p.rows) |
                 fastTest(image, center, p.threshold, row, column - 3, p.rows);
            d &= fastTest(image, center, p.threshold, row + 2, column + 2, p.rows) |
                 fastTest(image, center, p.threshold, row - 2, column - 2, p.rows);
            if (d == 0) continue;
            d &= fastTest(image, center, p.threshold, row - 3, column + 1, p.rows) |
                 fastTest(image, center, p.threshold, row + 3, column - 1, p.rows);
            d &= fastTest(image, center, p.threshold, row - 1, column + 3, p.rows) |
                 fastTest(image, center, p.threshold, row + 1, column - 3, p.rows);
            d &= fastTest(image, center, p.threshold, row + 1, column + 3, p.rows) |
                 fastTest(image, center, p.threshold, row - 1, column - 3, p.rows);
            d &= fastTest(image, center, p.threshold, row + 3, column + 1, p.rows) |
                 fastTest(image, center, p.threshold, row - 3, column - 1, p.rows);
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
                maxSum = max(maxSum, sum); minSum = min(minSum, sum);
            }
            for (uint i = 0; i < p.arcLength - 1; ++i) {
                sum -= fastTest(image, center, p.threshold,
                                row + fastDy(16 - p.arcLength + i),
                                column + fastDx(16 - p.arcLength + i), p.rows);
                sum += fastTest(image, center, p.threshold, row + fastDy(i),
                                column + fastDx(i), p.rows);
                maxSum = max(maxSum, sum); minSum = min(minSum, sum);
            }
            float bright = 0.0f, dark = 0.0f;
            for (uint i = 0; i < 16; ++i) {
                const float v = image[ulong(column + fastDx(i)) * p.rows +
                                      ulong(row + fastDy(i))];
                bright += (v > center + p.threshold) * (fabs(v - center) - p.threshold);
                dark += (v < center - p.threshold) * (fabs(center - v) - p.threshold);
            }
            if (maxSum == int(p.arcLength) || minSum == -int(p.arcLength)) {
                const uint index = found++;
                if (index < p.maxFeatures) {
                    const float response = max(bright, dark);
                    xOutput[index] = float(column); yOutput[index] = float(row);
                    scoreOutput[index] = response;
                    if (p.nonmax) scoreImage[ulong(column) * p.rows + ulong(row)] = response;
                }
            }
        }
    }
    *count = found;
}
