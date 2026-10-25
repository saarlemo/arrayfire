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

struct HistogramParams {
    ulong inputDims[4], outputStrides[4], inputStrides[4];
    uint bins;
    float minValue, maxValue;
    uint linear, threadgroupSize;
};

struct HistogramZeroParams {
    ulong count;
};

kernel void histogram_zero(device uint* output [[buffer(0)]],
                           constant HistogramZeroParams& p [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) < p.count) output[gid] = 0;
}

#define DEFINE_HISTOGRAM(NAME, TYPE)                                       \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device atomic_uint* output [[buffer(1)]],                 \
                 constant HistogramParams& p [[buffer(2)]],                \
                 uint group [[threadgroup_position_in_grid]],              \
                 uint tid [[thread_index_in_threadgroup]]) {               \
    const ulong planeCount = p.inputDims[0] * p.inputDims[1];              \
    const ulong batchCount = p.inputDims[2] * p.inputDims[3];              \
    const ulong groupsPerBatch =                                           \
        (planeCount + p.threadgroupSize - 1) / p.threadgroupSize;           \
    const ulong batch = ulong(group) / groupsPerBatch;                     \
    const ulong pixel = (ulong(group) % groupsPerBatch) *                  \
                        p.threadgroupSize + tid;                            \
    const bool validBatch = batch < batchCount;                            \
    const ulong z = batch % p.inputDims[2];                                \
    const ulong w = batch / p.inputDims[2];                                \
    const ulong inputBase = z * p.inputStrides[2] +                         \
                            w * p.inputStrides[3];                         \
    const ulong outputBase = z * p.outputStrides[2] +                      \
                             w * p.outputStrides[3];                      \
    threadgroup atomic_uint localBins[256];                                \
    if (p.bins <= 256) {                                                   \
        for (uint bin = tid; bin < p.bins; bin += p.threadgroupSize)       \
            atomic_store_explicit(&localBins[bin], 0u, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        if (validBatch && pixel < planeCount) {                            \
            const ulong index = p.linear ? pixel : (pixel % p.inputDims[0]) * \
                p.inputStrides[0] + (pixel / p.inputDims[0]) *             \
                    p.inputStrides[1];                                     \
            const float scaled = (float(input[inputBase + index]) -        \
                                  p.minValue) * float(p.bins) /            \
                                 (p.maxValue - p.minValue);                 \
            int bin = int(floor(nextafter(scaled, INFINITY)));             \
            bin = clamp(bin, 0, int(p.bins) - 1);                          \
            atomic_fetch_add_explicit(&localBins[uint(bin)], 1u,           \
                                      memory_order_relaxed);                \
        }                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        if (validBatch) {                                                   \
            for (uint bin = tid; bin < p.bins; bin += p.threadgroupSize) { \
                const uint count = atomic_load_explicit(&localBins[bin],   \
                                                        memory_order_relaxed); \
                if (count != 0)                                            \
                    atomic_fetch_add_explicit(                             \
                        &output[outputBase + bin], count,                  \
                        memory_order_relaxed);                              \
            }                                                               \
        }                                                                   \
    } else if (validBatch && pixel < planeCount) {                         \
        const ulong index = p.linear ? pixel : (pixel % p.inputDims[0]) *  \
            p.inputStrides[0] + (pixel / p.inputDims[0]) * p.inputStrides[1]; \
        const float scaled = (float(input[inputBase + index]) -            \
                              p.minValue) * float(p.bins) /                \
                             (p.maxValue - p.minValue);                     \
        int bin = int(floor(nextafter(scaled, INFINITY)));                 \
        bin = clamp(bin, 0, int(p.bins) - 1);                              \
        atomic_fetch_add_explicit(&output[outputBase + ulong(bin)], 1u,    \
                                  memory_order_relaxed);                    \
    }                                                                       \
}

DEFINE_HISTOGRAM(histogram_float, float)
DEFINE_HISTOGRAM(histogram_int, int)
DEFINE_HISTOGRAM(histogram_uint, uint)
DEFINE_HISTOGRAM(histogram_long, long)
DEFINE_HISTOGRAM(histogram_ulong, ulong)
DEFINE_HISTOGRAM(histogram_char, char)
DEFINE_HISTOGRAM(histogram_uchar, uchar)
DEFINE_HISTOGRAM(histogram_short, short)
DEFINE_HISTOGRAM(histogram_ushort, ushort)
DEFINE_HISTOGRAM(histogram_half, half)
