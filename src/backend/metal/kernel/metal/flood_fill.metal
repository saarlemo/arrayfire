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

struct FloodFillParams {
    ulong imageDims[4];
    ulong imageStrides[4];
    ulong seedDims[4];
    ulong seedXStrides[4];
    ulong seedYStrides[4];
};

kernel void flood_fill_state_init(device atomic_uint* state [[buffer(0)]],
                                  constant ulong& count [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid < count) atomic_store_explicit(state + gid, 0u,
                                           memory_order_relaxed);
}

kernel void flood_fill_seed_state(const device uint* seedX [[buffer(0)]],
                                  const device uint* seedY [[buffer(1)]],
                                  device atomic_uint* state [[buffer(2)]],
                                  constant FloodFillParams& params
                                      [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    const ulong seedCount = params.seedDims[0] * params.seedDims[1] *
                            params.seedDims[2] * params.seedDims[3];
    if (gid >= seedCount) return;

    ulong q = gid;
    const ulong s0 = q % params.seedDims[0];
    q /= params.seedDims[0];
    const ulong s1 = q % params.seedDims[1];
    q /= params.seedDims[1];
    const ulong s2 = q % params.seedDims[2];
    const ulong s3 = q / params.seedDims[2];
    const ulong seedOffset = s0 * params.seedXStrides[0] +
                             s1 * params.seedXStrides[1] +
                             s2 * params.seedXStrides[2] +
                             s3 * params.seedXStrides[3];
    const ulong seedYOffset = s0 * params.seedYStrides[0] +
                              s1 * params.seedYStrides[1] +
                              s2 * params.seedYStrides[2] +
                              s3 * params.seedYStrides[3];
    const uint x = seedX[seedOffset];
    const uint y = seedY[seedYOffset];
    if (x < params.imageDims[0] && y < params.imageDims[1]) {
        const ulong index = ulong(x) + ulong(y) * params.imageDims[0];
        atomic_store_explicit(state + index, 2u, memory_order_relaxed);
    }
}

#define DEFINE_FLOOD_FILL_STEP(NAME, TYPE)                                  \
    kernel void NAME(                                                       \
        const device TYPE* image [[buffer(0)]],                             \
        device atomic_uint* changed [[buffer(1)]],                          \
        device atomic_uint* state [[buffer(2)]],                             \
        constant FloodFillParams& params [[buffer(3)]],                     \
        constant TYPE& lower [[buffer(4)]],                                 \
        constant TYPE& upper [[buffer(5)]],                                 \
        uint gid [[thread_position_in_grid]]) {                             \
        const ulong width = params.imageDims[0];                            \
        const ulong height = params.imageDims[1];                           \
        const ulong total = width * height;                                 \
        if (gid >= total ||                                                  \
            atomic_load_explicit(state + gid, memory_order_relaxed) != 2u)  \
            return;                                                          \
        const ulong x = gid % width;                                        \
        const ulong y = gid / width;                                        \
        for (long ny = max(long(y) - 1, 0L);                                \
             ny <= min(long(y) + 1, long(height) - 1); ++ny) {             \
            for (long nx = max(long(x) - 1, 0L);                            \
                 nx <= min(long(x) + 1, long(width) - 1); ++nx) {           \
                const ulong neighborIndex = ulong(nx) + ulong(ny) * width;  \
                uint expected = 0u;                                         \
                const ulong imageOffset = ulong(nx) *                      \
                                              params.imageStrides[0] +       \
                                          ulong(ny) *                       \
                                              params.imageStrides[1];        \
                const TYPE value = image[imageOffset];                      \
                const uint next = value >= lower && value <= upper ? 2u     \
                                                                   : 1u;    \
                bool claimed = atomic_compare_exchange_weak_explicit(       \
                    state + neighborIndex, &expected, next,                 \
                    memory_order_relaxed, memory_order_relaxed);            \
                while (!claimed && expected == 0u) {                        \
                    expected = 0u;                                          \
                    claimed = atomic_compare_exchange_weak_explicit(        \
                        state + neighborIndex, &expected, next,             \
                        memory_order_relaxed, memory_order_relaxed);        \
                }                                                           \
                if (claimed && next == 2u) {                                \
                    atomic_store_explicit(changed, 1u, memory_order_relaxed);\
                }                                                           \
            }                                                               \
        }                                                                   \
    }

#define DEFINE_FLOOD_FILL_FINALIZE(NAME, TYPE)                              \
    kernel void NAME(                                                       \
        const device atomic_uint* state [[buffer(0)]],                      \
        device TYPE* output [[buffer(1)]],                                  \
        constant TYPE& newValue [[buffer(2)]],                              \
        uint gid [[thread_position_in_grid]]) {                             \
        output[gid] = atomic_load_explicit(state + gid, memory_order_relaxed)\
                           == 2u                                             \
                       ? newValue                                           \
                       : TYPE(0);                                           \
    }

DEFINE_FLOOD_FILL_STEP(flood_fill_step_float, float)
DEFINE_FLOOD_FILL_STEP(flood_fill_step_uint, uint)
DEFINE_FLOOD_FILL_STEP(flood_fill_step_ushort, ushort)
DEFINE_FLOOD_FILL_STEP(flood_fill_step_uchar, uchar)

DEFINE_FLOOD_FILL_FINALIZE(flood_fill_finalize_float, float)
DEFINE_FLOOD_FILL_FINALIZE(flood_fill_finalize_uint, uint)
DEFINE_FLOOD_FILL_FINALIZE(flood_fill_finalize_ushort, ushort)
DEFINE_FLOOD_FILL_FINALIZE(flood_fill_finalize_uchar, uchar)
