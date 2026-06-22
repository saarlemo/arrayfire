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
};

#define DEFINE_SORT(NAME, TYPE)                                               \
kernel void NAME(const device TYPE* input [[buffer(0)]],                      \
                 device TYPE* output [[buffer(1)]],                           \
                 constant SortParams& params [[buffer(2)]],                   \
                 uint vector [[thread_position_in_grid]]) {                   \
    const ulong vectors = params.dims[1] * params.dims[2] * params.dims[3];   \
    if (vector >= vectors) return;                                             \
    ulong q = vector;                                                          \
    const ulong y = q % params.dims[1];                                       \
    q /= params.dims[1];                                                       \
    const ulong z = q % params.dims[2];                                       \
    const ulong w = q / params.dims[2];                                       \
    const ulong base = y * params.strides[1] + z * params.strides[2] +        \
                       w * params.strides[3];                                 \
    for (ulong x = 0; x < params.dims[0]; ++x)                               \
        output[base + x * params.strides[0]] =                                \
            input[base + x * params.strides[0]];                              \
    for (ulong x = 1; x < params.dims[0]; ++x) {                              \
        const TYPE value = output[base + x * params.strides[0]];              \
        ulong position = x;                                                    \
        while (position > 0) {                                                 \
            const TYPE previous =                                              \
                output[base + (position - 1) * params.strides[0]];            \
            const bool ordered =                                               \
                sortOrdered(previous, value, params.ascending);                \
            if (ordered) break;                                                \
            output[base + position * params.strides[0]] = previous;           \
            --position;                                                        \
        }                                                                       \
        output[base + position * params.strides[0]] = value;                  \
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
