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

struct HarrisNonMaxParams {
    uint rows;
    uint columns;
    uint border;
    uint maxCorners;
    float minResponse;
};

struct HarrisKeepParams {
    uint corners;
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

kernel void harris_nonmax_float(
    const device float* response [[buffer(0)]],
    device float* xOutput [[buffer(1)]], device float* yOutput [[buffer(2)]],
    device float* responseOutput [[buffer(3)]],
    device atomic_uint* count [[buffer(4)]],
    constant HarrisNonMaxParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const uint elements = p.rows * p.columns;
    if (gid >= elements) return;

    const uint row = gid % p.rows;
    const uint column = gid / p.rows;
    const uint border = p.border + 1;
    if (row < border || row >= p.rows - border ||
        column < border || column >= p.columns - border)
        return;

    const float value = response[gid];
    float maxValue = response[(column - 1) * p.rows + row - 1];
    maxValue = max(maxValue, response[column * p.rows + row - 1]);
    maxValue = max(maxValue, response[(column + 1) * p.rows + row - 1]);
    maxValue = max(maxValue, response[(column - 1) * p.rows + row]);
    maxValue = max(maxValue, response[(column + 1) * p.rows + row]);
    maxValue = max(maxValue, response[(column - 1) * p.rows + row + 1]);
    maxValue = max(maxValue, response[column * p.rows + row + 1]);
    maxValue = max(maxValue, response[(column + 1) * p.rows + row + 1]);

    if (value > maxValue && value >= p.minResponse) {
        const uint outputIndex = atomic_fetch_add_explicit(
            count, 1u, memory_order_relaxed);
        if (outputIndex < p.maxCorners) {
            xOutput[outputIndex] = float(column);
            yOutput[outputIndex] = float(row);
            responseOutput[outputIndex] = value;
        }
    }
}

kernel void harris_keep_corners(
    const device float* xInput [[buffer(0)]],
    const device float* yInput [[buffer(1)]],
    const device float* responseInput [[buffer(2)]],
    const device uint* responseIndex [[buffer(3)]],
    device float* xOutput [[buffer(4)]], device float* yOutput [[buffer(5)]],
    device float* responseOutput [[buffer(6)]],
    constant HarrisKeepParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.corners) return;
    const uint index = responseIndex[gid];
    xOutput[gid] = xInput[index];
    yOutput[gid] = yInput[index];
    // The response array is already sorted; the index only selects the
    // corresponding x/y corner.  This matches the CUDA/OpenCL keep_corners
    // kernels, which write responseInput[gid].
    responseOutput[gid] = responseInput[gid];
}
