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

struct OrbHarrisParams {
    uint features;
    uint rows;
    uint columns;
    uint blockSize;
    uint patchSize;
    float kThreshold;
};

struct OrbExtractParams {
    uint features;
    uint rows;
    uint columns;
    uint patchSize;
    float scale;
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

kernel void orb_harris_float(
    const device float* xInput [[buffer(0)]],
    const device float* yInput [[buffer(1)]],
    const device float* image [[buffer(2)]],
    device float* xOutput [[buffer(3)]],
    device float* yOutput [[buffer(4)]],
    device float* scoreOutput [[buffer(5)]],
    device atomic_uint* count [[buffer(6)]],
    constant OrbHarrisParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.features) return;

    const uint x = uint(floor(xInput[gid] + 0.5f));
    const uint y = uint(floor(yInput[gid] + 0.5f));
    const float size = float(p.patchSize);
    const uint patchRadius = uint(ceil(size * sqrt(2.0f) / 2.0f));
    if (x < patchRadius || y < patchRadius ||
        x >= p.columns - patchRadius || y >= p.rows - patchRadius)
        return;

    const int radius = int(p.blockSize / 2);
    float ixx = 0.0f;
    float iyy = 0.0f;
    float ixy = 0.0f;
    for (uint k = 0; k < p.blockSize * p.blockSize; ++k) {
        const int i = int(k / p.blockSize) - radius;
        const int j = int(k % p.blockSize) - radius;
        const ulong center = ulong(x + i) * p.rows + ulong(y + j);
        const float ix = image[center + p.rows] - image[center - p.rows];
        const float iy = image[center + 1] - image[center - 1];
        ixx += ix * ix;
        iyy += iy * iy;
        ixy += ix * iy;
    }

    const float trace = ixx + iyy;
    const float determinant = ixx * iyy - ixy * ixy;
    const float response =
        (determinant - p.kThreshold * trace * trace) * 1.0e-12f;
    const uint outputIndex =
        atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    xOutput[outputIndex] = float(x);
    yOutput[outputIndex] = float(y);
    scoreOutput[outputIndex] = response;
}

kernel void orb_extract_float(
    device uint* descriptor [[buffer(0)]],
    device float* xInputOutput [[buffer(1)]],
    device float* yInputOutput [[buffer(2)]],
    const device float* orientation [[buffer(3)]],
    device float* sizeOutput [[buffer(4)]],
    const device int* pattern [[buffer(5)]],
    const device float* image [[buffer(6)]],
    constant OrbExtractParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.features) return;

    const int x = int(floor(xInputOutput[gid] + 0.5f));
    const int y = int(floor(yInputOutput[gid] + 0.5f));
    const uint radius = uint(ceil(float(p.patchSize) * sqrt(2.0f) / 2.0f));
    if (x < int(radius) || y < int(radius) ||
        x >= int(p.columns) - int(radius) ||
        y >= int(p.rows) - int(radius))
        return;

    const float angle = orientation[gid];
    const float sine = sin(angle);
    const float cosine = cos(angle);
    const float patchScale = 1.0f;
    for (uint word = 0; word < 8; ++word) {
        uint value = 0;
        for (uint bit = 0; bit < 32; ++bit) {
            const uint base = (word * 32 + bit) * 4;
            const int dx1 = pattern[base];
            const int dy1 = pattern[base + 1];
            const int dx2 = pattern[base + 2];
            const int dy2 = pattern[base + 3];
            const int px1 = x + int(round(
                float(dx1) * patchScale * cosine -
                float(dy1) * patchScale * sine));
            const int py1 = y + int(round(
                float(dx1) * patchScale * sine +
                float(dy1) * patchScale * cosine));
            const int px2 = x + int(round(
                float(dx2) * patchScale * cosine -
                float(dy2) * patchScale * sine));
            const int py2 = y + int(round(
                float(dx2) * patchScale * sine +
                float(dy2) * patchScale * cosine));
            const float first = image[ulong(px1) * p.rows + ulong(py1)];
            const float second = image[ulong(px2) * p.rows + ulong(py2)];
            value |= (first < second ? 1u : 0u) << bit;
        }
        descriptor[gid * 8 + word] = value;
    }

    xInputOutput[gid] = round(float(x) * p.scale);
    yInputOutput[gid] = round(float(y) * p.scale);
    sizeOutput[gid] = float(p.patchSize) * p.scale;
}
