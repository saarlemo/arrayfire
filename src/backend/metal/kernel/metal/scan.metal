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

struct ScanParams {
    ulong dims[4];
    long outputStrides[4];
    long inputStrides[4];
    uint dimension;
    uint operation;
    uint inclusive;
};

constant uint SCAN_ADD = 0;
constant uint SCAN_MUL = 1;
constant uint SCAN_MIN = 2;
constant uint SCAN_MAX = 3;
constant uint SCAN_NOTZERO = 4;

template<typename T>
bool scanIsNan(const T) {
    return false;
}

bool scanIsNan(const float value) { return isnan(value); }

bool scanIsNan(const float2 value) { return any(isnan(value)); }

template<typename T>
bool scanNonzero(const T value) {
    return value != T(0);
}

bool scanNonzero(const float2 value) {
    return any(value != float2(0.0f));
}

template<typename T>
T scanCombine(const T lhs, const T rhs, const uint operation) {
    if (operation == SCAN_MUL) return lhs * rhs;
    if (operation == SCAN_MIN) return min(lhs, rhs);
    if (operation == SCAN_MAX) return max(lhs, rhs);
    return lhs + rhs;
}

float2 scanCombine(const float2 lhs, const float2 rhs,
                   const uint operation) {
    if (operation == SCAN_MUL) {
        return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                      lhs.x * rhs.y + lhs.y * rhs.x);
    }
    if (operation == SCAN_MIN)
        return dot(lhs, lhs) < dot(rhs, rhs) ? lhs : rhs;
    if (operation == SCAN_MAX)
        return dot(lhs, lhs) > dot(rhs, rhs) ? lhs : rhs;
    return lhs + rhs;
}

#define DEFINE_SCAN(NAME, INPUT_TYPE, OUTPUT_TYPE, ONE, MINIMUM, MAXIMUM)    \
kernel void NAME(const device INPUT_TYPE* input [[buffer(0)]],              \
                 device OUTPUT_TYPE* output [[buffer(1)]],                  \
                 constant ScanParams& params [[buffer(2)]],                 \
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
    long inputBase = 0;                                                     \
    long outputBase = 0;                                                    \
    for (uint d = 0; d < 4; ++d) {                                         \
        inputBase += long(coordinates[d]) * params.inputStrides[d];         \
        outputBase += long(coordinates[d]) * params.outputStrides[d];       \
    }                                                                       \
    OUTPUT_TYPE accumulated = OUTPUT_TYPE(0);                               \
    if (params.operation == SCAN_MUL) accumulated = (ONE);                  \
    if (params.operation == SCAN_MIN) accumulated = (MINIMUM);              \
    if (params.operation == SCAN_MAX) accumulated = (MAXIMUM);              \
    const ulong length = params.dims[params.dimension];                      \
    for (ulong i = 0; i < length; ++i) {                                   \
        const long inputIndex =                                             \
            inputBase + long(i) * params.inputStrides[params.dimension];    \
        const long outputIndex =                                            \
            outputBase + long(i) * params.outputStrides[params.dimension];  \
        const INPUT_TYPE inputValue = input[inputIndex];                    \
        OUTPUT_TYPE value = OUTPUT_TYPE(inputValue);                        \
        if (params.operation == SCAN_NOTZERO)                               \
            value = OUTPUT_TYPE(scanNonzero(inputValue));                   \
        if ((params.operation == SCAN_MIN ||                                \
             params.operation == SCAN_MAX) &&                              \
            scanIsNan(inputValue))                                          \
            value = params.operation == SCAN_MIN ? (MINIMUM) : (MAXIMUM);   \
        if (!params.inclusive) output[outputIndex] = accumulated;            \
        accumulated = scanCombine(accumulated, value, params.operation);    \
        if (params.inclusive) output[outputIndex] = accumulated;             \
    }                                                                       \
}

DEFINE_SCAN(scan_float_float, float, float, 1.0f, INFINITY, -INFINITY)
DEFINE_SCAN(scan_cfloat_cfloat, float2, float2, (float2(1.0f, 0.0f)),
            (float2(INFINITY, 0.0f)), (float2(0.0f)))
DEFINE_SCAN(scan_int_int, int, int, 1, 0x7fffffff, (-0x7fffffff - 1))
DEFINE_SCAN(scan_uint_uint, uint, uint, 1u, 0xffffffffu, 0u)
DEFINE_SCAN(scan_long_long, long, long, 1L, 0x7fffffffffffffffL,
            (-0x7fffffffffffffffL - 1L))
DEFINE_SCAN(scan_ulong_ulong, ulong, ulong, 1UL, 0xffffffffffffffffUL, 0UL)
DEFINE_SCAN(scan_char_int, char, int, 1, 0x7fffffff, (-0x7fffffff - 1))
DEFINE_SCAN(scan_uchar_uint, uchar, uint, 1u, 0xffffffffu, 0u)
DEFINE_SCAN(scan_uchar_int, uchar, int, 1, 0x7fffffff, (-0x7fffffff - 1))
DEFINE_SCAN(scan_short_int, short, int, 1, 0x7fffffff, (-0x7fffffff - 1))
DEFINE_SCAN(scan_ushort_uint, ushort, uint, 1u, 0xffffffffu, 0u)
