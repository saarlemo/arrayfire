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

struct RegionsParams {
    ulong width;
    ulong height;
    ulong inputStrideX;
    ulong inputStrideY;
    uint connectivity;
};

struct RegionsScanParams {
    ulong count;
    uint rootFlags;
};

struct RegionsCountParams {
    ulong count;
};

struct RegionsValidationParams {
    uint maxLabels;
};

static uint regions_find_root(device atomic_uint* parent, uint label) {
    while (label != 0u) {
        const uint next =
            atomic_load_explicit(parent + label - 1u, memory_order_relaxed);
        if (next == label) return label;
        const uint grand =
            atomic_load_explicit(parent + next - 1u, memory_order_relaxed);
        uint expected = next;
        atomic_compare_exchange_weak_explicit(
            parent + label - 1u, &expected, grand, memory_order_relaxed,
            memory_order_relaxed);
        label = grand;
    }
    return 0u;
}

static void regions_join(device atomic_uint* parent, const uint first,
                         const uint second) {
    while (true) {
        const uint rootA = regions_find_root(parent, first);
        const uint rootB = regions_find_root(parent, second);
        if (rootA == rootB) return;

        const uint high = max(rootA, rootB);
        const uint low  = min(rootA, rootB);
        uint expected   = high;
        if (atomic_compare_exchange_weak_explicit(
                parent + high - 1u, &expected, low, memory_order_relaxed,
                memory_order_relaxed)) {
            return;
        }
    }
}

kernel void regions_init_uint(const device char* input [[buffer(0)]],
                              device uint* parent [[buffer(1)]],
                              constant RegionsParams& params [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    const ulong count = params.width * params.height;
    if (ulong(gid) >= count) return;
    const ulong x = ulong(gid) % params.width;
    const ulong y = ulong(gid) / params.width;
    const ulong inputOffset = x * params.inputStrideX +
                              y * params.inputStrideY;
    parent[gid] = input[inputOffset] != 0 ? gid + 1u : 0u;
}

kernel void regions_union_uint(const device char* input [[buffer(0)]],
                               device atomic_uint* parent [[buffer(1)]],
                               constant RegionsParams& params [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    const ulong count = params.width * params.height;
    if (ulong(gid) >= count) return;
    const ulong x = ulong(gid) % params.width;
    const ulong y = ulong(gid) / params.width;
    const ulong inputOffset = x * params.inputStrideX +
                              y * params.inputStrideY;
    if (input[inputOffset] == 0) return;

    const uint label = gid + 1u;
    if (x + 1u < params.width) {
        const ulong rightOffset = (x + 1u) * params.inputStrideX +
                                  y * params.inputStrideY;
        if (input[rightOffset] != 0) regions_join(parent, label, label + 1u);
    }
    if (y + 1u < params.height) {
        const uint down = label + uint(params.width);
        const ulong downOffset = x * params.inputStrideX +
                                 (y + 1u) * params.inputStrideY;
        if (input[downOffset] != 0) regions_join(parent, label, down);

        if (params.connectivity == 8u) {
            if (x > 0u) {
                const ulong downLeftOffset = (x - 1u) * params.inputStrideX +
                                            (y + 1u) * params.inputStrideY;
                if (input[downLeftOffset] != 0)
                    regions_join(parent, label, down - 1u);
            }
            if (x + 1u < params.width) {
                const ulong downRightOffset = (x + 1u) * params.inputStrideX +
                                             (y + 1u) * params.inputStrideY;
                if (input[downRightOffset] != 0)
                    regions_join(parent, label, down + 1u);
            }
        }
    }
}

kernel void regions_roots_uint(device atomic_uint* parent [[buffer(0)]],
                               device uint* roots [[buffer(1)]],
                               constant RegionsCountParams& params [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= params.count) return;
    const uint label =
        atomic_load_explicit(parent + gid, memory_order_relaxed);
    roots[gid] = label == 0u ? 0u : regions_find_root(parent, label);
}

kernel void regions_scan_uint(const device uint* input [[buffer(0)]],
                              device uint* prefix [[buffer(1)]],
                              device uint* blockSums [[buffer(2)]],
                              constant RegionsScanParams& params [[buffer(3)]],
                              uint gid [[thread_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint values[256];
    const bool inBounds = ulong(gid) < params.count;
    const uint inputValue =
        inBounds ? (params.rootFlags != 0u
                        ? uint(input[gid] == gid + 1u)
                        : input[gid])
                 : 0u;
    values[lane] = inputValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        const uint addend = lane >= offset ? values[lane - offset] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[lane] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) prefix[gid] = values[lane] - inputValue;
    if (lane == 255u) blockSums[group] = values[lane];
}

kernel void regions_add_offsets_uint(device uint* prefix [[buffer(0)]],
                                     const device uint* blockPrefix [[buffer(1)]],
                                     constant RegionsCountParams& params [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= params.count) return;
    prefix[gid] += blockPrefix[gid / 256u];
}

kernel void regions_validate_count_uint(
    const device uint* componentCount [[buffer(0)]],
    device uint* invalidCount [[buffer(1)]],
    constant RegionsValidationParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid == 0u) invalidCount[0] = componentCount[0] > params.maxLabels;
}

#define DEFINE_REGIONS_FINALIZE(NAME, TYPE)                              \
kernel void NAME(const device uint* roots [[buffer(0)]],                  \
                 const device uint* prefix [[buffer(1)]],                \
                 device TYPE* output [[buffer(2)]],                      \
                 constant RegionsCountParams& params [[buffer(3)]],      \
                 uint gid [[thread_position_in_grid]]) {                 \
    if (ulong(gid) >= params.count) return;                              \
    const uint root = roots[gid];                                        \
    output[gid] = root == 0u ? TYPE(0) : TYPE(prefix[root - 1u] + 1u);   \
}

DEFINE_REGIONS_FINALIZE(regions_finalize_float, float)
DEFINE_REGIONS_FINALIZE(regions_finalize_int, int)
DEFINE_REGIONS_FINALIZE(regions_finalize_uint, uint)
DEFINE_REGIONS_FINALIZE(regions_finalize_short, short)
DEFINE_REGIONS_FINALIZE(regions_finalize_ushort, ushort)
