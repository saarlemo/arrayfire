/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
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

struct SusanNonMaxParams {
    uint rows;
    uint columns;
    uint border;
    uint maxCorners;
};

#define DEFINE_SUSAN(SUFFIX, TYPE)                                           \
kernel void susan_response_##SUFFIX(                                         \
                            const device TYPE* input [[buffer(0)]],          \
                            device TYPE* output [[buffer(1)]],               \
                            constant SusanParams& p [[buffer(2)]],            \
                            uint gid [[thread_position_in_grid]]) {          \
    const uint total = p.rows * p.columns;                                   \
    if (gid >= total) return;                                                 \
    const uint row = gid % p.rows;                                            \
    const uint column = gid / p.rows;                                         \
    if (row < p.border || row >= p.rows - p.border ||                        \
        column < p.border || column >= p.columns - p.border) return;          \
    const TYPE center = input[gid];                                           \
    float area = 0.0f;                                                        \
    const int radius = int(p.radius);                                         \
    const int radiusSquared = radius * radius;                                \
    for (int y = -radius; y <= radius; ++y) {                                 \
        for (int x = -radius; x <= radius; ++x) {                             \
            if (x * x + y * y >= radiusSquared) continue;                    \
            const ulong index = ulong(int(row) + x) +                       \
                                ulong(int(column) + y) * p.rows;              \
            const float difference = float(input[index]) - float(center);    \
            const float ratio = difference / p.differenceThreshold;           \
            const float squared = ratio * ratio;                              \
            area += exp(-(squared * squared * squared));                      \
        }                                                                      \
    }                                                                          \
    output[gid] = area < p.geometricThreshold                                \
                      ? TYPE(p.geometricThreshold - area)                    \
                      : TYPE(0);                                              \
}                                                                              \
kernel void susan_nonmax_##SUFFIX(                                            \
                          const device TYPE* response [[buffer(0)]],         \
                          device float* xOutput [[buffer(1)]],               \
                          device float* yOutput [[buffer(2)]],               \
                          device float* scoreOutput [[buffer(3)]],            \
                          device atomic_uint* count [[buffer(4)]],            \
                          constant SusanNonMaxParams& p [[buffer(5)]],        \
                          uint gid [[thread_position_in_grid]]) {             \
    const uint x = gid % p.rows;                                              \
    const uint y = gid / p.rows;                                              \
    const uint border = p.border + 1;                                         \
    if (x < border || x >= p.rows - border ||                                 \
        y < border || y >= p.columns - border) return;                        \
    const TYPE value = response[gid];                                         \
    TYPE maximum = response[(y - 1) * p.rows + x - 1];                       \
    maximum = max(maximum, response[y * p.rows + x - 1]);                    \
    maximum = max(maximum, response[(y + 1) * p.rows + x - 1]);              \
    maximum = max(maximum, response[(y - 1) * p.rows + x]);                  \
    maximum = max(maximum, response[(y + 1) * p.rows + x]);                  \
    maximum = max(maximum, response[(y - 1) * p.rows + x + 1]);              \
    maximum = max(maximum, response[y * p.rows + x + 1]);                    \
    maximum = max(maximum, response[(y + 1) * p.rows + x + 1]);              \
    if (value > maximum) {                                                    \
        const uint outputIndex = atomic_fetch_add_explicit(                  \
            count, 1u, memory_order_relaxed);                                \
        if (outputIndex < p.maxCorners) {                                     \
            xOutput[outputIndex] = float(x);                                  \
            yOutput[outputIndex] = float(y);                                  \
            scoreOutput[outputIndex] = float(value);                         \
        }                                                                      \
    }                                                                          \
}

DEFINE_SUSAN(float, float)
DEFINE_SUSAN(int, int)
DEFINE_SUSAN(uint, uint)
DEFINE_SUSAN(char, char)
DEFINE_SUSAN(uchar, uchar)
DEFINE_SUSAN(short, short)
DEFINE_SUSAN(ushort, ushort)
