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

struct SortByKeyParams {
    ulong dims[4];
    ulong keyStrides[4];
    ulong valueStrides[4];
    uint ascending;
};

#define DEFINE_SORT_BY_KEY(NAME, TYPE)                                        \
kernel void NAME(const device TYPE* inputKeys [[buffer(0)]],                  \
                 const device TYPE* inputValues [[buffer(1)]],                \
                 device TYPE* outputKeys [[buffer(2)]],                       \
                 device TYPE* outputValues [[buffer(3)]],                     \
                 constant SortByKeyParams& params [[buffer(4)]],              \
                 uint vector [[thread_position_in_grid]]) {                   \
    const ulong vectors = params.dims[1] * params.dims[2] * params.dims[3];   \
    if (vector >= vectors) return;                                             \
    ulong q = vector;                                                          \
    const ulong y = q % params.dims[1];                                       \
    q /= params.dims[1];                                                       \
    const ulong z = q % params.dims[2];                                       \
    const ulong w = q / params.dims[2];                                       \
    const ulong keyBase = y * params.keyStrides[1] +                          \
                          z * params.keyStrides[2] +                          \
                          w * params.keyStrides[3];                           \
    const ulong valueBase = y * params.valueStrides[1] +                      \
                            z * params.valueStrides[2] +                      \
                            w * params.valueStrides[3];                       \
    for (ulong x = 0; x < params.dims[0]; ++x) {                              \
        outputKeys[keyBase + x * params.keyStrides[0]] =                      \
            inputKeys[keyBase + x * params.keyStrides[0]];                    \
        outputValues[valueBase + x * params.valueStrides[0]] =                \
            inputValues[valueBase + x * params.valueStrides[0]];              \
    }                                                                           \
    for (ulong x = 1; x < params.dims[0]; ++x) {                              \
        const TYPE key = outputKeys[keyBase + x * params.keyStrides[0]];       \
        const TYPE value =                                                      \
            outputValues[valueBase + x * params.valueStrides[0]];             \
        ulong position = x;                                                    \
        while (position > 0) {                                                 \
            const TYPE previous = outputKeys[                                 \
                keyBase + (position - 1) * params.keyStrides[0]];             \
            const bool ordered =                                               \
                sortOrdered(previous, key, params.ascending);                  \
            if (ordered) break;                                                \
            outputKeys[keyBase + position * params.keyStrides[0]] = previous; \
            outputValues[valueBase + position * params.valueStrides[0]] =     \
                outputValues[valueBase +                                      \
                             (position - 1) * params.valueStrides[0]];         \
            --position;                                                        \
        }                                                                       \
        outputKeys[keyBase + position * params.keyStrides[0]] = key;           \
        outputValues[valueBase + position * params.valueStrides[0]] = value;   \
    }                                                                           \
}

DEFINE_SORT_BY_KEY(sort_by_key_float, float)
DEFINE_SORT_BY_KEY(sort_by_key_int, int)
DEFINE_SORT_BY_KEY(sort_by_key_uint, uint)
DEFINE_SORT_BY_KEY(sort_by_key_long, long)
DEFINE_SORT_BY_KEY(sort_by_key_ulong, ulong)
DEFINE_SORT_BY_KEY(sort_by_key_char, char)
DEFINE_SORT_BY_KEY(sort_by_key_uchar, uchar)
DEFINE_SORT_BY_KEY(sort_by_key_short, short)
DEFINE_SORT_BY_KEY(sort_by_key_ushort, ushort)
