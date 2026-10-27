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

struct SiftParams { ulong elements; };

struct SiftDescriptorParams {
    uint featureCount;
    uint rows;
    uint columns;
    float scale;
};

struct SiftExtremaParams {
    ulong elements;
    ulong width;
    ulong height;
    float threshold;
};

static void siftAtomicAddFloat(threadgroup atomic_uint* destination,
                               const float value) {
    uint expected = atomic_load_explicit(destination, memory_order_relaxed);
    while (true) {
        const uint desired =
            as_type<uint>(as_type<float>(expected) + value);
        if (atomic_compare_exchange_weak_explicit(
                destination, &expected, desired, memory_order_relaxed,
                memory_order_relaxed)) {
            return;
        }
    }
}

kernel void sift_subtract_float(const device float* first [[buffer(0)]],
                                const device float* second [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                constant SiftParams& p [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < p.elements) output[gid] = first[gid] - second[gid];
}

// Emits one candidate flag per pixel. The host compacts these flags in the
// original layer/y/x order so the per-octave feature cap remains deterministic.
kernel void sift_extrema_flags_float(
    const device float* previous [[buffer(0)]],
    const device float* center [[buffer(1)]],
    const device float* next [[buffer(2)]],
    device uchar* flags [[buffer(3)]],
    constant SiftExtremaParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong index = ulong(gid);
    if (index >= p.elements) return;

    const ulong x = index % p.width;
    const ulong y = index / p.width;
    constexpr ulong border = 5;
    if (x < border || x >= p.width - border || y < border ||
        y >= p.height - border) {
        flags[index] = 0;
        return;
    }

    const float value = center[index];
    if (!(abs(value) > p.threshold)) {
        flags[index] = 0;
        return;
    }

    bool isMaximum = value > 0.f;
    bool isMinimum = value < 0.f;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const long neighborX = long(x) + dx;
                const long neighborY = long(y) + dy;
                const ulong neighbor = ulong(neighborY) * p.width +
                                       ulong(neighborX);
                const float other =
                    dz < 0 ? previous[neighbor]
                           : (dz > 0 ? next[neighbor] : center[neighbor]);
                isMaximum = isMaximum && value > other;
                isMinimum = isMinimum && value < other;
            }
        }
    }
    flags[index] = (isMaximum || isMinimum) ? 1 : 0;
}

// Each threadgroup computes one feature descriptor. Threads cooperatively
// accumulate the oriented gradient samples into eight private histograms, then
// combine and normalize them before writing to that feature's fixed output slot.
kernel void sift_descriptor_float(
    const device float* image [[buffer(0)]],
    const device float* featureData [[buffer(1)]],
    const device uint* featureIndices [[buffer(2)]],
    device float* descriptors [[buffer(3)]],
    constant SiftDescriptorParams& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr uint descLength = 128;
    constexpr uint histCount = 8;
    constexpr uint threadgroupSize = 128;
    constexpr float pi = 3.14159265358979323846f;

    const uint feature = group.y;
    if (feature >= p.featureCount) return;

    threadgroup atomic_uint histograms[histCount * descLength];
    threadgroup float descriptor[descLength];
    threadgroup float norm[descLength];

    for (uint i = tid; i < histCount * descLength; i += threadgroupSize) {
        atomic_store_explicit(&histograms[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint featureOffset = feature * 4u;
    const float xIn = featureData[featureOffset];
    const float yIn = featureData[featureOffset + 1u];
    const float size = featureData[featureOffset + 2u];
    float ori = (360.f - featureData[featureOffset + 3u]) * pi / 180.f;
    ori = ori > pi ? ori - pi * 2.f : ori;

    const int fx = int(round(xIn * p.scale));
    const int fy = int(round(yIn * p.scale));
    const float cosT = cos(ori);
    const float sinT = sin(ori);
    const float binsPerRad = 8.f / (pi * 2.f);
    const float expDenom = 4.f * 4.f * 0.5f;
    const float histWidth = 3.f * size * p.scale * 0.5f;
    const int radius = int(histWidth * sqrt(2.f) * 5.f * 0.5f + 0.5f);
    const int len = radius * 2 + 1;
    const uint histOffset = (tid % histCount) * descLength;

    for (int l = int(tid); l < len * len; l += int(threadgroupSize)) {
        const int i = l / len - radius;
        const int j = l % len - radius;
        const int y = fy + i;
        const int x = fx + j;

        float xRot = (j * cosT - i * sinT) / histWidth;
        float yRot = (j * sinT + i * cosT) / histWidth;
        float xbin = xRot + 4 / 2 - 0.5f;
        float ybin = yRot + 4 / 2 - 0.5f;

        if (ybin > -1.f && ybin < 4.f && xbin > -1.f && xbin < 4.f &&
            y > 0 && y < int(p.rows) - 1 && x > 0 &&
            x < int(p.columns) - 1) {
            const float dx = image[(x + 1) * int(p.rows) + y] -
                             image[(x - 1) * int(p.rows) + y];
            const float dy = image[x * int(p.rows) + y - 1] -
                             image[x * int(p.rows) + y + 1];
            const float gradMag = sqrt(dx * dx + dy * dy);
            float gradOri = atan2(dy, dx) - ori;
            while (gradOri < 0.f) gradOri += pi * 2.f;
            while (gradOri >= pi * 2.f) gradOri -= pi * 2.f;

            const float weight =
                exp(-(xRot * xRot + yRot * yRot) / expDenom);
            float obin = gradOri * binsPerRad;
            const float mag = gradMag * weight;

            const int x0 = int(floor(xbin));
            const int y0 = int(floor(ybin));
            const int o0 = int(floor(obin));
            xbin -= x0;
            ybin -= y0;
            obin -= o0;

            for (int yl = 0; yl <= 1; yl++) {
                const int yb = y0 + yl;
                if (yb >= 0 && yb < 4) {
                    const float vy = mag * (yl == 0 ? 1.f - ybin : ybin);
                    for (int xl = 0; xl <= 1; xl++) {
                        const int xb = x0 + xl;
                        if (xb >= 0 && xb < 4) {
                            const float vx =
                                vy * (xl == 0 ? 1.f - xbin : xbin);
                            for (int ol = 0; ol <= 1; ol++) {
                                const int ob = (o0 + ol) % 8;
                                const float value =
                                    vx * (ol == 0 ? 1.f - obin : obin);
                                const uint bin =
                                    uint((yb * 4 + xb) * 8 + ob);
                                siftAtomicAddFloat(
                                    &histograms[histOffset + bin], value);
                            }
                        }
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float h0 = as_type<float>(atomic_load_explicit(
        &histograms[tid], memory_order_relaxed));
    const float h1 = as_type<float>(atomic_load_explicit(
        &histograms[descLength + tid], memory_order_relaxed));
    const float h2 = as_type<float>(atomic_load_explicit(
        &histograms[2u * descLength + tid], memory_order_relaxed));
    const float h3 = as_type<float>(atomic_load_explicit(
        &histograms[3u * descLength + tid], memory_order_relaxed));
    const float h4 = as_type<float>(atomic_load_explicit(
        &histograms[4u * descLength + tid], memory_order_relaxed));
    const float h5 = as_type<float>(atomic_load_explicit(
        &histograms[5u * descLength + tid], memory_order_relaxed));
    const float h6 = as_type<float>(atomic_load_explicit(
        &histograms[6u * descLength + tid], memory_order_relaxed));
    const float h7 = as_type<float>(atomic_load_explicit(
        &histograms[7u * descLength + tid], memory_order_relaxed));
    descriptor[tid] = ((h0 + h4) + (h2 + h6)) + ((h1 + h5) + (h3 + h7));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    norm[tid] = descriptor[tid] * descriptor[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = descLength / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) norm[tid] += norm[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float firstInvNorm = 1.f / sqrt(norm[0]);
    descriptor[tid] *= firstInvNorm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    descriptor[tid] = min(descriptor[tid], 0.2f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    norm[tid] = descriptor[tid] * descriptor[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = descLength / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) norm[tid] += norm[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float secondInvNorm = 1.f / sqrt(norm[0]);
    descriptor[tid] *= secondInvNorm;

    if (tid < descLength) {
        const uint outputIndex = featureIndices[feature];
        descriptors[outputIndex * descLength + tid] =
            round(min(255.f, descriptor[tid] * 512.f));
    }
}
