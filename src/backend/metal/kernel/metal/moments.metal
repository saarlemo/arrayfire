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

struct MomentsParams {
    ulong inputDims[4], outputStrides[4], inputStrides[4];
    ulong partialGroups;
    uint moment, threadgroupSize;
};

#define DEFINE_MOMENTS(NAME, TYPE)                                         \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device float* partials [[buffer(1)]],                    \
                 constant MomentsParams& p [[buffer(2)]],                 \
                 uint group [[threadgroup_position_in_grid]],             \
                 uint tid [[thread_index_in_threadgroup]]) {              \
    const ulong plane = p.inputDims[0] * p.inputDims[1];                   \
    const ulong batches = p.inputDims[2] * p.inputDims[3];                 \
    const ulong batch = ulong(group) / p.partialGroups;                   \
    const ulong tile = ulong(group) % p.partialGroups;                    \
    const ulong pixel = tile * p.threadgroupSize + tid;                    \
    threadgroup float sums[4][256];                                        \
    for (uint m = 0; m < 4; ++m) sums[m][tid] = 0.0f;                     \
    if (batch < batches && pixel < plane) {                                \
        const ulong x = pixel % p.inputDims[0];                            \
        const ulong y = pixel / p.inputDims[0];                            \
        const ulong z = batch % p.inputDims[2];                            \
        const ulong w = batch / p.inputDims[2];                            \
        const ulong base = z * p.inputStrides[2] +                         \
                           w * p.inputStrides[3];                          \
        const float value = float(input[base + x * p.inputStrides[0] +     \
                                        y * p.inputStrides[1]]);           \
        sums[0][tid] = value;                                               \
        sums[1][tid] = float(x) * value;                                   \
        sums[2][tid] = float(y) * value;                                   \
        sums[3][tid] = float(x) * float(y) * value;                        \
    }                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    uint active = p.threadgroupSize;                                       \
    while (active > 1) {                                                    \
        const uint upper = (active + 1) / 2;                               \
        if (tid < active - upper) {                                        \
            for (uint m = 0; m < 4; ++m)                                   \
                sums[m][tid] += sums[m][tid + upper];                      \
        }                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        active = upper;                                                     \
    }                                                                       \
    if (tid == 0) {                                                         \
        const uint selectedCount = uint((p.moment & 1u) != 0) +            \
                                    uint((p.moment & 2u) != 0) +            \
                                    uint((p.moment & 4u) != 0) +            \
                                    uint((p.moment & 8u) != 0);             \
        const ulong offset =                                               \
            (batch * p.partialGroups + tile) * selectedCount;              \
        uint m = 0;                                                         \
        if (p.moment & 1u) partials[offset + m++] = sums[0][0];            \
        if (p.moment & 2u) partials[offset + m++] = sums[1][0];            \
        if (p.moment & 4u) partials[offset + m++] = sums[2][0];            \
        if (p.moment & 8u) partials[offset + m] = sums[3][0];               \
    }                                                                       \
}

kernel void moments_reduce(const device float* partials [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant MomentsParams& p [[buffer(2)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    const ulong batches = p.inputDims[2] * p.inputDims[3];
    const uint selectedCount = uint((p.moment & 1u) != 0) +
                               uint((p.moment & 2u) != 0) +
                               uint((p.moment & 4u) != 0) +
                               uint((p.moment & 8u) != 0);
    const ulong batch = ulong(group) / selectedCount;
    const uint selectedMoment = group % selectedCount;
    threadgroup float sums[256];
    float sum = 0.0f;
    for (ulong tile = tid; tile < p.partialGroups;
         tile += p.threadgroupSize) {
        const ulong index =
            (batch * p.partialGroups + tile) * selectedCount +
            selectedMoment;
        sum += partials[index];
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint active = p.threadgroupSize;
    while (active > 1) {
        const uint upper = (active + 1) / 2;
        if (tid < active - upper) sums[tid] += sums[tid + upper];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = upper;
    }
    if (tid == 0 && batch < batches) {
        const ulong z = batch % p.inputDims[2];
        const ulong w = batch / p.inputDims[2];
        const ulong outputBase = z * p.outputStrides[2] +
                                 w * p.outputStrides[3];
        output[outputBase + selectedMoment * p.outputStrides[0]] = sums[0];
    }
}

DEFINE_MOMENTS(moments_float, float)
DEFINE_MOMENTS(moments_int, int)
DEFINE_MOMENTS(moments_uint, uint)
DEFINE_MOMENTS(moments_char, char)
DEFINE_MOMENTS(moments_uchar, uchar)
DEFINE_MOMENTS(moments_short, short)
DEFINE_MOMENTS(moments_ushort, ushort)
