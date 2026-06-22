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

struct HarrisParams {
    ulong elements;
    uint rows;
    uint columns;
    uint border;
    float k;
};

kernel void harris_second_order_float(
    const device float* ix [[buffer(0)]], const device float* iy [[buffer(1)]],
    device float* ixx [[buffer(2)]], device float* ixy [[buffer(3)]],
    device float* iyy [[buffer(4)]], constant HarrisParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    ixx[gid] = ix[gid] * ix[gid];
    ixy[gid] = ix[gid] * iy[gid];
    iyy[gid] = iy[gid] * iy[gid];
}

kernel void harris_response_float(
    const device float* ixx [[buffer(0)]], const device float* ixy [[buffer(1)]],
    const device float* iyy [[buffer(2)]], device float* output [[buffer(3)]],
    constant HarrisParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    const uint row = gid % p.rows;
    const uint column = gid / p.rows;
    if (row < p.border || row >= p.rows - p.border || column < p.border ||
        column >= p.columns - p.border) return;
    const float trace = ixx[gid] + iyy[gid];
    const float determinant = ixx[gid] * iyy[gid] - ixy[gid] * ixy[gid];
    output[gid] = determinant - p.k * trace * trace;
}
