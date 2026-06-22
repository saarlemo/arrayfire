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

struct OrbParams {
    uint features;
    uint rows;
    uint columns;
    uint patchSize;
};

kernel void orb_centroid_float(
    const device float* xInput [[buffer(0)]],
    const device float* yInput [[buffer(1)]],
    const device float* image [[buffer(2)]],
    device float* orientation [[buffer(3)]],
    constant OrbParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.features) return;
    const uint x = uint(floor(xInput[gid] + 0.5f));
    const uint y = uint(floor(yInput[gid] + 0.5f));
    const uint radius = p.patchSize / 2;
    if (x < radius || y < radius || x > p.columns - radius ||
        y > p.rows - radius) return;
    float m01 = 0.0f, m10 = 0.0f;
    for (uint k = 0; k < p.patchSize * p.patchSize; ++k) {
        const int i = int(k / p.patchSize) - int(radius);
        const int j = int(k % p.patchSize) - int(radius);
        const float value = image[uint(int(x) + i) * p.rows +
                                  uint(int(y) + j)];
        m01 += float(j) * value;
        m10 += float(i) * value;
    }
    orientation[gid] = atan2(m01, m10);
}
