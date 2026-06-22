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

struct SusanParams {
    uint rows;
    uint columns;
    uint radius;
    uint border;
    float differenceThreshold;
    float geometricThreshold;
};

kernel void susan_response_float(const device float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant SusanParams& p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    const uint total = p.rows * p.columns;
    if (gid >= total) return;
    const uint row = gid % p.rows;
    const uint column = gid / p.rows;
    if (row < p.border || row >= p.rows - p.border || column < p.border ||
        column >= p.columns - p.border) return;
    const float center = input[gid];
    float area = 0.0f;
    const int radius = int(p.radius);
    const int radiusSquared = radius * radius;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y >= radiusSquared) continue;
            const ulong index = ulong(int(row) + x) +
                                ulong(int(column) + y) * p.rows;
            const float ratio = (input[index] - center) /
                                p.differenceThreshold;
            const float squared = ratio * ratio;
            area += exp(-(squared * squared * squared));
        }
    }
    output[gid] = area < p.geometricThreshold
                      ? p.geometricThreshold - area
                      : 0.0f;
}
