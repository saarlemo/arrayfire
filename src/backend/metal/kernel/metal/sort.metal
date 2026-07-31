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

struct SortParams {
    ulong dims[4];
    ulong strides[4];
    uint ascending;
    uint dimension;
};

#define DEFINE_SORT(NAME, TYPE)                                               \
kernel void NAME(const device TYPE* input [[buffer(0)]],                      \
                 device TYPE* output [[buffer(1)]],                           \
                 constant SortParams& params [[buffer(2)]],                   \
                 uint vector [[thread_position_in_grid]]) {                   \
    const ulong vectors = (params.dims[0] * params.dims[1] * params.dims[2] * \
                           params.dims[3]) / params.dims[params.dimension];    \
    if (vector >= vectors) return;                                             \
    ulong q = vector;                                                          \
    ulong base = 0;                                                            \
    for (uint d = 0; d < 4; ++d) {                                           \
        if (d == params.dimension) continue;                                  \
        const ulong coordinate = q % params.dims[d];                          \
        q /= params.dims[d];                                                   \
        base += coordinate * params.strides[d];                               \
    }                                                                          \
    const ulong length = params.dims[params.dimension];                       \
    const ulong stride = params.strides[params.dimension];                    \
    for (ulong x = 0; x < length; ++x)                                       \
        output[base + x * stride] = input[base + x * stride];                 \
    for (ulong x = 1; x < length; ++x) {                                     \
        const TYPE value = output[base + x * stride];                         \
        ulong position = x;                                                    \
        while (position > 0) {                                                 \
            const TYPE previous =                                              \
                output[base + (position - 1) * stride];                       \
            const bool ordered =                                               \
                sortOrdered(previous, value, params.ascending);                \
            if (ordered) break;                                                \
            output[base + position * stride] = previous;                      \
            --position;                                                        \
        }                                                                       \
        output[base + position * stride] = value;                             \
    }                                                                           \
}

DEFINE_SORT(sort_float, float)
DEFINE_SORT(sort_int, int)
DEFINE_SORT(sort_uint, uint)
DEFINE_SORT(sort_long, long)
DEFINE_SORT(sort_ulong, ulong)
DEFINE_SORT(sort_char, char)
DEFINE_SORT(sort_uchar, uchar)
DEFINE_SORT(sort_short, short)
DEFINE_SORT(sort_ushort, ushort)
