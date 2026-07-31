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
    uint dimension;
};

#define SORT_BASES()                                                          \
    const ulong vectors = (params.dims[0] * params.dims[1] * params.dims[2] * \
                           params.dims[3]) / params.dims[params.dimension];    \
    if (vector >= vectors) return;                                             \
    ulong q = vector;                                                          \
    ulong keyBase = 0;                                                         \
    ulong valueBase = 0;                                                       \
    for (uint d = 0; d < 4; ++d) {                                           \
        if (d == params.dimension) continue;                                  \
        const ulong coordinate = q % params.dims[d];                          \
        q /= params.dims[d];                                                   \
        keyBase += coordinate * params.keyStrides[d];                         \
        valueBase += coordinate * params.valueStrides[d];                     \
    }                                                                          \
    const ulong length = params.dims[params.dimension];                       \
    const ulong keyStride = params.keyStrides[params.dimension];              \
    const ulong valueStride = params.valueStrides[params.dimension];

#define DEFINE_SORT_BY_KEY(NAME, KEY_TYPE, VALUE_TYPE)                        \
kernel void NAME(const device KEY_TYPE* inputKeys [[buffer(0)]],              \
                 const device VALUE_TYPE* inputValues [[buffer(1)]],          \
                 device KEY_TYPE* outputKeys [[buffer(2)]],                   \
                 device VALUE_TYPE* outputValues [[buffer(3)]],               \
                 constant SortByKeyParams& params [[buffer(4)]],              \
                 uint vector [[thread_position_in_grid]]) {                   \
    SORT_BASES()                                                               \
    for (ulong x = 0; x < length; ++x) {                                      \
        outputKeys[keyBase + x * keyStride] =                                 \
            inputKeys[keyBase + x * keyStride];                               \
        outputValues[valueBase + x * valueStride] =                           \
            inputValues[valueBase + x * valueStride];                         \
    }                                                                          \
    for (ulong x = 1; x < length; ++x) {                                      \
        const KEY_TYPE key = outputKeys[keyBase + x * keyStride];             \
        const VALUE_TYPE value = outputValues[valueBase + x * valueStride];   \
        ulong position = x;                                                    \
        while (position > 0) {                                                 \
            const KEY_TYPE previous =                                         \
                outputKeys[keyBase + (position - 1) * keyStride];             \
            if (sortOrdered(previous, key, params.ascending)) break;           \
            outputKeys[keyBase + position * keyStride] = previous;            \
            outputValues[valueBase + position * valueStride] =                \
                outputValues[valueBase + (position - 1) * valueStride];       \
            --position;                                                        \
        }                                                                      \
        outputKeys[keyBase + position * keyStride] = key;                     \
        outputValues[valueBase + position * valueStride] = value;             \
    }                                                                          \
}

#define DEFINE_SORT_BY_KEY_VALUES(KEY_NAME, KEY_TYPE)                        \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_float, KEY_TYPE, float)      \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_cfloat, KEY_TYPE, float2)     \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_int, KEY_TYPE, int)          \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_uint, KEY_TYPE, uint)        \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_long, KEY_TYPE, long)        \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_ulong, KEY_TYPE, ulong)      \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_char, KEY_TYPE, char)        \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_uchar, KEY_TYPE, uchar)      \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_short, KEY_TYPE, short)      \
    DEFINE_SORT_BY_KEY(sort_by_key_##KEY_NAME##_ushort, KEY_TYPE, ushort)

DEFINE_SORT_BY_KEY_VALUES(float, float)
DEFINE_SORT_BY_KEY_VALUES(int, int)
DEFINE_SORT_BY_KEY_VALUES(uint, uint)
DEFINE_SORT_BY_KEY_VALUES(long, long)
DEFINE_SORT_BY_KEY_VALUES(ulong, ulong)
DEFINE_SORT_BY_KEY_VALUES(char, char)
DEFINE_SORT_BY_KEY_VALUES(uchar, uchar)
DEFINE_SORT_BY_KEY_VALUES(short, short)
DEFINE_SORT_BY_KEY_VALUES(ushort, ushort)

#define DEFINE_SORT_INDEX(NAME, TYPE)                                         \
kernel void NAME(const device TYPE* inputKeys [[buffer(0)]],                  \
                 const device uint* inputValues [[buffer(1)]],                \
                 device TYPE* outputKeys [[buffer(2)]],                       \
                 device uint* outputValues [[buffer(3)]],                     \
                 constant SortByKeyParams& params [[buffer(4)]],              \
                 uint vector [[thread_position_in_grid]]) {                   \
    SORT_BASES()                                                               \
    for (ulong x = 0; x < length; ++x) {                                      \
        outputKeys[keyBase + x * keyStride] =                                 \
            inputKeys[keyBase + x * keyStride];                               \
        outputValues[valueBase + x * valueStride] =                           \
            inputValues[valueBase + x * valueStride];                         \
    }                                                                          \
    for (ulong x = 1; x < length; ++x) {                                      \
        const TYPE key = outputKeys[keyBase + x * keyStride];                 \
        const uint value = outputValues[valueBase + x * valueStride];         \
        ulong position = x;                                                    \
        while (position > 0) {                                                 \
            const TYPE previous =                                             \
                outputKeys[keyBase + (position - 1) * keyStride];             \
            if (sortOrdered(previous, key, params.ascending)) break;           \
            outputKeys[keyBase + position * keyStride] = previous;            \
            outputValues[valueBase + position * valueStride] =                \
                outputValues[valueBase + (position - 1) * valueStride];       \
            --position;                                                        \
        }                                                                      \
        outputKeys[keyBase + position * keyStride] = key;                     \
        outputValues[valueBase + position * valueStride] = value;             \
    }                                                                          \
}

DEFINE_SORT_INDEX(sort_index_float, float)
DEFINE_SORT_INDEX(sort_index_int, int)
DEFINE_SORT_INDEX(sort_index_uint, uint)
DEFINE_SORT_INDEX(sort_index_long, long)
DEFINE_SORT_INDEX(sort_index_ulong, ulong)
DEFINE_SORT_INDEX(sort_index_char, char)
DEFINE_SORT_INDEX(sort_index_uchar, uchar)
DEFINE_SORT_INDEX(sort_index_short, short)
DEFINE_SORT_INDEX(sort_index_ushort, ushort)
