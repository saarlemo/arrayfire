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

struct ScanByKeyParams {
    ulong dims[4];
    long outputStrides[4];
    long keyStrides[4];
    long inputStrides[4];
    uint dimension;
    uint operation;
    uint inclusive;
};

constant uint SBK_ADD = 0;
constant uint SBK_MUL = 1;
constant uint SBK_MIN = 2;
constant uint SBK_MAX = 3;

template<typename T>
bool sbkIsNan(const T) {
    return false;
}

bool sbkIsNan(const float value) { return isnan(value); }

bool sbkIsNan(const float2 value) { return any(isnan(value)); }

template<typename T>
T sbkCombine(const T lhs, const T rhs, const uint operation) {
    if (operation == SBK_MUL) return lhs * rhs;
    if (operation == SBK_MIN) return min(lhs, rhs);
    if (operation == SBK_MAX) return max(lhs, rhs);
    return lhs + rhs;
}

float2 sbkCombine(const float2 lhs, const float2 rhs, const uint operation) {
    if (operation == SBK_MUL) {
        return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                      lhs.x * rhs.y + lhs.y * rhs.x);
    }
    if (operation == SBK_MIN)
        return dot(lhs, lhs) < dot(rhs, rhs) ? lhs : rhs;
    if (operation == SBK_MAX)
        return dot(lhs, lhs) > dot(rhs, rhs) ? lhs : rhs;
    return lhs + rhs;
}

#define DEFINE_SCAN_BY_KEY(NAME, KEY_TYPE, VALUE_TYPE, ONE, MINIMUM, MAXIMUM) \
kernel void NAME(const device KEY_TYPE* keys [[buffer(0)]],                  \
                 const device VALUE_TYPE* input [[buffer(1)]],              \
                 device VALUE_TYPE* output [[buffer(2)]],                   \
                 constant ScanByKeyParams& params [[buffer(3)]],            \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *  \
                        params.dims[3];                                     \
    if (gid >= total) return;                                                \
    ulong q = gid;                                                          \
    ulong coordinates[4];                                                   \
    coordinates[0] = q % params.dims[0];                                   \
    q /= params.dims[0];                                                    \
    coordinates[1] = q % params.dims[1];                                   \
    q /= params.dims[1];                                                    \
    coordinates[2] = q % params.dims[2];                                   \
    coordinates[3] = q / params.dims[2];                                   \
    if (coordinates[params.dimension] != 0) return;                         \
    long outputBase = 0;                                                    \
    long keyBase = 0;                                                       \
    long inputBase = 0;                                                     \
    for (uint d = 0; d < 4; ++d) {                                         \
        outputBase += long(coordinates[d]) * params.outputStrides[d];       \
        keyBase += long(coordinates[d]) * params.keyStrides[d];             \
        inputBase += long(coordinates[d]) * params.inputStrides[d];         \
    }                                                                       \
    VALUE_TYPE accumulated = VALUE_TYPE(0);                                 \
    if (params.operation == SBK_MUL) accumulated = (ONE);                   \
    if (params.operation == SBK_MIN) accumulated = (MINIMUM);               \
    if (params.operation == SBK_MAX) accumulated = (MAXIMUM);               \
    KEY_TYPE previousKey = KEY_TYPE(0);                                     \
    const ulong length = params.dims[params.dimension];                      \
    for (ulong i = 0; i < length; ++i) {                                   \
        const long keyIndex =                                               \
            keyBase + long(i) * params.keyStrides[params.dimension];        \
        const long inputIndex =                                             \
            inputBase + long(i) * params.inputStrides[params.dimension];    \
        const long outputIndex =                                            \
            outputBase + long(i) * params.outputStrides[params.dimension];  \
        const KEY_TYPE key = keys[keyIndex];                                \
        if (i == 0 || key != previousKey) {                                 \
            accumulated = VALUE_TYPE(0);                                    \
            if (params.operation == SBK_MUL) accumulated = (ONE);           \
            if (params.operation == SBK_MIN) accumulated = (MINIMUM);       \
            if (params.operation == SBK_MAX) accumulated = (MAXIMUM);       \
        }                                                                   \
        VALUE_TYPE value = input[inputIndex];                               \
        if ((params.operation == SBK_MIN ||                                 \
             params.operation == SBK_MAX) &&                               \
            sbkIsNan(value))                                                \
            value = params.operation == SBK_MIN ? (MINIMUM) : (MAXIMUM);    \
        if (!params.inclusive) output[outputIndex] = accumulated;            \
        accumulated = sbkCombine(accumulated, value, params.operation);     \
        if (params.inclusive) output[outputIndex] = accumulated;             \
        previousKey = key;                                                  \
    }                                                                       \
}

#define DEFINE_SCAN_BY_KEY_VALUES(KEY_NAME, KEY_TYPE)                       \
    DEFINE_SCAN_BY_KEY(scan_by_key_##KEY_NAME##_float, KEY_TYPE, float,    \
                       1.0f, INFINITY, -INFINITY)                           \
    DEFINE_SCAN_BY_KEY(scan_by_key_##KEY_NAME##_cfloat, KEY_TYPE, float2,   \
                       (float2(1.0f, 0.0f)),                                \
                       (float2(INFINITY, 0.0f)), (float2(0.0f)))             \
    DEFINE_SCAN_BY_KEY(scan_by_key_##KEY_NAME##_int, KEY_TYPE, int, 1,     \
                       0x7fffffff, (-0x7fffffff - 1))                        \
    DEFINE_SCAN_BY_KEY(scan_by_key_##KEY_NAME##_uint, KEY_TYPE, uint, 1u,  \
                       0xffffffffu, 0u)                                     \
    DEFINE_SCAN_BY_KEY(scan_by_key_##KEY_NAME##_long, KEY_TYPE, long, 1L,  \
                       0x7fffffffffffffffL,                                 \
                       (-0x7fffffffffffffffL - 1L))                         \
    DEFINE_SCAN_BY_KEY(scan_by_key_##KEY_NAME##_ulong, KEY_TYPE, ulong,     \
                       1UL, 0xffffffffffffffffUL, 0UL)

DEFINE_SCAN_BY_KEY_VALUES(uint, uint)
DEFINE_SCAN_BY_KEY_VALUES(ulong, ulong)
