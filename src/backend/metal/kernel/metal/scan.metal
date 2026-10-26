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

struct ScanTotalsParams {
    ulong tileCount;
    ulong blockCount;
    uint operation;
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

#define DEFINE_SCAN_PARALLEL(NAME, INPUT_TYPE, OUTPUT_TYPE, ONE, MINIMUM,    \
                             MAXIMUM)                                         \
kernel void NAME##_tile(                                                    \
    const device INPUT_TYPE* input [[buffer(0)]],                            \
    device OUTPUT_TYPE* output [[buffer(1)]],                                \
    device OUTPUT_TYPE* tileTotals [[buffer(2)]],                            \
    constant ScanParams& params [[buffer(3)]],                               \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    constexpr ulong tileLength = 4096;                                       \
    constexpr uint lanes = 256;                                              \
    constexpr uint valuesPerLane = 16;                                      \
    const ulong length = params.dims[params.dimension];                     \
    const ulong tileCount = (length + tileLength - 1) / tileLength;         \
    const ulong tileBegin = ulong(group.x) * tileLength;                    \
    ulong coordinates[4];                                                   \
    ulong line = ulong(group.y);                                            \
    for (uint d = 0; d < 4; ++d) {                                         \
        coordinates[d] = 0;                                                \
        if (d != params.dimension) {                                        \
            coordinates[d] = line % params.dims[d];                         \
            line /= params.dims[d];                                        \
        }                                                                   \
    }                                                                       \
    long inputBase = 0;                                                     \
    long outputBase = 0;                                                    \
    for (uint d = 0; d < 4; ++d) {                                         \
        inputBase += long(coordinates[d]) * params.inputStrides[d];         \
        outputBase += long(coordinates[d]) * params.outputStrides[d];       \
    }                                                                       \
    OUTPUT_TYPE localPrefixes[valuesPerLane];                              \
    OUTPUT_TYPE laneAggregate = OUTPUT_TYPE(0);                             \
    if (params.operation == SCAN_MUL) laneAggregate = (ONE);                \
    if (params.operation == SCAN_MIN) laneAggregate = (MINIMUM);            \
    if (params.operation == SCAN_MAX) laneAggregate = (MAXIMUM);            \
    for (uint j = 0; j < valuesPerLane; ++j) {                             \
        const ulong position = tileBegin + ulong(lane) * valuesPerLane + j; \
        OUTPUT_TYPE value = OUTPUT_TYPE(0);                                 \
        if (params.operation == SCAN_MUL) value = (ONE);                    \
        if (params.operation == SCAN_MIN) value = (MINIMUM);                \
        if (params.operation == SCAN_MAX) value = (MAXIMUM);                \
        if (position < length) {                                            \
            const long inputIndex =                                         \
                inputBase + long(position) *                                \
                                params.inputStrides[params.dimension];       \
            const INPUT_TYPE inputValue = input[inputIndex];                \
            value = OUTPUT_TYPE(inputValue);                                \
            if (params.operation == SCAN_NOTZERO)                           \
                value = OUTPUT_TYPE(scanNonzero(inputValue));               \
            if ((params.operation == SCAN_MIN ||                            \
                 params.operation == SCAN_MAX) &&                          \
                scanIsNan(inputValue))                                      \
                value = params.operation == SCAN_MIN ? (MINIMUM)            \
                                                       : (MAXIMUM);          \
        }                                                                   \
        if (!params.inclusive) localPrefixes[j] = laneAggregate;            \
        laneAggregate = scanCombine(laneAggregate, value, params.operation);\
        if (params.inclusive) localPrefixes[j] = laneAggregate;             \
    }                                                                       \
    threadgroup OUTPUT_TYPE aggregates[lanes];                              \
    aggregates[lane] = laneAggregate;                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    for (uint offset = 1; offset < lanes; offset <<= 1) {                   \
        OUTPUT_TYPE left = OUTPUT_TYPE(0);                                  \
        if (params.operation == SCAN_MUL) left = (ONE);                     \
        if (params.operation == SCAN_MIN) left = (MINIMUM);                 \
        if (params.operation == SCAN_MAX) left = (MAXIMUM);                 \
        if (lane >= offset) left = aggregates[lane - offset];               \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        if (lane >= offset)                                                 \
            aggregates[lane] =                                              \
                scanCombine(left, aggregates[lane], params.operation);      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                       \
    OUTPUT_TYPE laneCarry = OUTPUT_TYPE(0);                                 \
    if (params.operation == SCAN_MUL) laneCarry = (ONE);                    \
    if (params.operation == SCAN_MIN) laneCarry = (MINIMUM);                \
    if (params.operation == SCAN_MAX) laneCarry = (MAXIMUM);                \
    if (lane > 0) laneCarry = aggregates[lane - 1];                         \
    for (uint j = 0; j < valuesPerLane; ++j) {                             \
        const ulong position = tileBegin + ulong(lane) * valuesPerLane + j; \
        if (position < length) {                                            \
            const long outputIndex =                                        \
                outputBase + long(position) *                               \
                                 params.outputStrides[params.dimension];    \
            output[outputIndex] = scanCombine(laneCarry, localPrefixes[j],  \
                                              params.operation);             \
        }                                                                   \
    }                                                                       \
    if (lane == 0)                                                          \
        tileTotals[ulong(group.y) * tileCount + group.x] = aggregates[255];  \
}                                                                           \
kernel void NAME##_apply(                                                  \
    const device OUTPUT_TYPE* tilePrefixes [[buffer(0)]],                   \
    const device OUTPUT_TYPE* blockTotals [[buffer(1)]],                     \
    device OUTPUT_TYPE* output [[buffer(2)]],                                \
    constant ScanParams& params [[buffer(3)]],                               \
    constant ScanTotalsParams& totalsParams [[buffer(4)]],                   \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    constexpr ulong tileLength = 4096;                                       \
    constexpr ulong totalsPerBlock = 4096;                                   \
    constexpr uint valuesPerLane = 16;                                      \
    const ulong length = params.dims[params.dimension];                     \
    const ulong tileCount = totalsParams.tileCount;                         \
    const ulong blockCount = totalsParams.blockCount;                       \
    const ulong tileBegin = ulong(group.x) * tileLength;                    \
    ulong coordinates[4];                                                   \
    ulong line = ulong(group.y);                                            \
    for (uint d = 0; d < 4; ++d) {                                         \
        coordinates[d] = 0;                                                \
        if (d != params.dimension) {                                        \
            coordinates[d] = line % params.dims[d];                         \
            line /= params.dims[d];                                        \
        }                                                                   \
    }                                                                       \
    long outputBase = 0;                                                    \
    for (uint d = 0; d < 4; ++d)                                           \
        outputBase += long(coordinates[d]) * params.outputStrides[d];       \
    OUTPUT_TYPE identity = OUTPUT_TYPE(0);                                 \
    if (params.operation == SCAN_MUL) identity = (ONE);                     \
    if (params.operation == SCAN_MIN) identity = (MINIMUM);                 \
    if (params.operation == SCAN_MAX) identity = (MAXIMUM);                 \
    const ulong totalsBlock = ulong(group.x) / totalsPerBlock;              \
    const ulong tileInBlock = ulong(group.x) % totalsPerBlock;              \
    const ulong prefixBase = ulong(group.y) * tileCount +                   \
                             totalsBlock * totalsPerBlock;                   \
    const ulong blockTotalsBase = ulong(group.y) * blockCount;               \
    threadgroup OUTPUT_TYPE carry;                                          \
    if (lane == 0) {                                                        \
        carry = identity;                                                   \
        for (ulong block = 0; block < totalsBlock; ++block)                 \
            carry = scanCombine(carry, blockTotals[blockTotalsBase + block],\
                                totalsParams.operation);                     \
        if (tileInBlock > 0)                                                 \
            carry = scanCombine(                                             \
                carry, tilePrefixes[prefixBase + tileInBlock - 1],          \
                totalsParams.operation);                                     \
    }                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    for (uint j = 0; j < valuesPerLane; ++j) {                             \
        const ulong position = tileBegin + ulong(lane) * valuesPerLane + j; \
        if (position < length) {                                            \
            const long outputIndex =                                        \
                outputBase + long(position) *                               \
                                 params.outputStrides[params.dimension];    \
            output[outputIndex] =                                           \
                scanCombine(carry, output[outputIndex], params.operation);  \
        }                                                                   \
    }                                                                       \
}                                                                           \
kernel void NAME##_totals_scan(                                            \
    const device OUTPUT_TYPE* tileTotals [[buffer(0)]],                     \
    device OUTPUT_TYPE* tilePrefixes [[buffer(1)]],                          \
    device OUTPUT_TYPE* blockTotals [[buffer(2)]],                           \
    constant ScanTotalsParams& params [[buffer(3)]],                         \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    constexpr ulong totalsPerBlock = 4096;                                   \
    constexpr uint lanes = 256;                                              \
    constexpr uint valuesPerLane = 16;                                      \
    const ulong blockBegin = ulong(group.x) * totalsPerBlock;               \
    const ulong blockLength = min(totalsPerBlock,                           \
                                  params.tileCount - blockBegin);             \
    const ulong inputBase = ulong(group.y) * params.tileCount;              \
    const ulong prefixBase = inputBase + blockBegin;                        \
    const ulong blockTotalsBase = ulong(group.y) * params.blockCount;       \
    OUTPUT_TYPE localPrefixes[valuesPerLane];                              \
    OUTPUT_TYPE laneAggregate = OUTPUT_TYPE(0);                             \
    if (params.operation == SCAN_MUL) laneAggregate = (ONE);                \
    if (params.operation == SCAN_MIN) laneAggregate = (MINIMUM);            \
    if (params.operation == SCAN_MAX) laneAggregate = (MAXIMUM);            \
    for (uint j = 0; j < valuesPerLane; ++j) {                             \
        const ulong position = ulong(lane) * valuesPerLane + j;             \
        OUTPUT_TYPE value = OUTPUT_TYPE(0);                                 \
        if (params.operation == SCAN_MUL) value = (ONE);                    \
        if (params.operation == SCAN_MIN) value = (MINIMUM);                \
        if (params.operation == SCAN_MAX) value = (MAXIMUM);                \
        if (position < blockLength)                                          \
            value = tileTotals[inputBase + blockBegin + position];          \
        laneAggregate = scanCombine(laneAggregate, value, params.operation);\
        localPrefixes[j] = laneAggregate;                                   \
    }                                                                       \
    threadgroup OUTPUT_TYPE aggregates[lanes];                              \
    aggregates[lane] = laneAggregate;                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    for (uint offset = 1; offset < lanes; offset <<= 1) {                   \
        OUTPUT_TYPE left = OUTPUT_TYPE(0);                                  \
        if (params.operation == SCAN_MUL) left = (ONE);                     \
        if (params.operation == SCAN_MIN) left = (MINIMUM);                 \
        if (params.operation == SCAN_MAX) left = (MAXIMUM);                 \
        if (lane >= offset) left = aggregates[lane - offset];               \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        if (lane >= offset)                                                 \
            aggregates[lane] =                                              \
                scanCombine(left, aggregates[lane], params.operation);      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                       \
    OUTPUT_TYPE laneCarry = OUTPUT_TYPE(0);                                 \
    if (params.operation == SCAN_MUL) laneCarry = (ONE);                    \
    if (params.operation == SCAN_MIN) laneCarry = (MINIMUM);                \
    if (params.operation == SCAN_MAX) laneCarry = (MAXIMUM);                \
    if (lane > 0) laneCarry = aggregates[lane - 1];                         \
    for (uint j = 0; j < valuesPerLane; ++j) {                             \
        const ulong position = ulong(lane) * valuesPerLane + j;             \
        if (position < blockLength)                                          \
            tilePrefixes[prefixBase + position] =                          \
                scanCombine(laneCarry, localPrefixes[j],                    \
                           params.operation);                               \
    }                                                                       \
    if (lane == 0)                                                          \
        blockTotals[blockTotalsBase + group.x] = aggregates[255];            \
}

DEFINE_SCAN_PARALLEL(scan_float_float, float, float, 1.0f, INFINITY,
                     -INFINITY)
DEFINE_SCAN_PARALLEL(scan_cfloat_cfloat, float2, float2,
                     (float2(1.0f, 0.0f)), (float2(INFINITY, 0.0f)),
                     (float2(0.0f)))
DEFINE_SCAN_PARALLEL(scan_int_int, int, int, 1, 0x7fffffff,
                     (-0x7fffffff - 1))
DEFINE_SCAN_PARALLEL(scan_uint_uint, uint, uint, 1u, 0xffffffffu, 0u)
DEFINE_SCAN_PARALLEL(scan_long_long, long, long, 1L, 0x7fffffffffffffffL,
                     (-0x7fffffffffffffffL - 1L))
DEFINE_SCAN_PARALLEL(scan_ulong_ulong, ulong, ulong, 1UL,
                     0xffffffffffffffffUL, 0UL)
DEFINE_SCAN_PARALLEL(scan_char_int, char, int, 1, 0x7fffffff,
                     (-0x7fffffff - 1))
DEFINE_SCAN_PARALLEL(scan_uchar_uint, uchar, uint, 1u, 0xffffffffu, 0u)
DEFINE_SCAN_PARALLEL(scan_uchar_int, uchar, int, 1, 0x7fffffff,
                     (-0x7fffffff - 1))
DEFINE_SCAN_PARALLEL(scan_short_int, short, int, 1, 0x7fffffff,
                     (-0x7fffffff - 1))
DEFINE_SCAN_PARALLEL(scan_ushort_uint, ushort, uint, 1u, 0xffffffffu, 0u)
