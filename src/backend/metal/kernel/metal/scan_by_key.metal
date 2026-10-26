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

struct ScanByKeyTotalsParams {
    ulong tileCount;
    ulong blockCount;
    uint operation;
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

template<typename KEY_TYPE, typename VALUE_TYPE>
struct ScanByKeySummary {
    KEY_TYPE firstKey;
    KEY_TYPE lastKey;
    VALUE_TYPE trailingAggregate;
    uint leadingLength;
    uint allSameKey;
    uint valid;
};

template<typename KEY_TYPE, typename VALUE_TYPE>
ScanByKeySummary<KEY_TYPE, VALUE_TYPE> sbkInvalidSummary(
    const VALUE_TYPE identity) {
    ScanByKeySummary<KEY_TYPE, VALUE_TYPE> summary;
    summary.firstKey = KEY_TYPE(0);
    summary.lastKey = KEY_TYPE(0);
    summary.trailingAggregate = identity;
    summary.leadingLength = 0;
    summary.allSameKey = 0;
    summary.valid = 0;
    return summary;
}

template<typename KEY_TYPE, typename VALUE_TYPE>
ScanByKeySummary<KEY_TYPE, VALUE_TYPE> sbkConcatSummaries(
    const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> left,
    const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> right,
    const uint operation) {
    if (!left.valid) return right;
    if (!right.valid) return left;

    ScanByKeySummary<KEY_TYPE, VALUE_TYPE> result;
    result.firstKey = left.firstKey;
    result.lastKey = right.lastKey;
    result.allSameKey = left.allSameKey && right.allSameKey &&
                        left.lastKey == right.firstKey;
    result.leadingLength = left.leadingLength;
    if (left.allSameKey && left.lastKey == right.firstKey)
        result.leadingLength += right.leadingLength;
    result.trailingAggregate =
        right.allSameKey && left.lastKey == right.firstKey
            ? sbkCombine(left.trailingAggregate, right.trailingAggregate,
                         operation)
            : right.trailingAggregate;
    result.valid = 1;
    return result;
}

template<typename KEY_TYPE, typename VALUE_TYPE>
ScanByKeySummary<KEY_TYPE, VALUE_TYPE> sbkLoadSummary(
    const device KEY_TYPE* firstKeys, const device KEY_TYPE* lastKeys,
    const device VALUE_TYPE* trailingAggregates,
    const device uint* leadingLengths, const device uint* flags,
    const ulong index) {
    ScanByKeySummary<KEY_TYPE, VALUE_TYPE> summary;
    summary.firstKey = firstKeys[index];
    summary.lastKey = lastKeys[index];
    summary.trailingAggregate = trailingAggregates[index];
    summary.leadingLength = leadingLengths[index];
    summary.allSameKey = flags[index] & 1u;
    summary.valid = (flags[index] >> 1) & 1u;
    return summary;
}

template<typename KEY_TYPE, typename VALUE_TYPE>
void sbkStoreSummary(
    device KEY_TYPE* firstKeys, device KEY_TYPE* lastKeys,
    device VALUE_TYPE* trailingAggregates, device uint* leadingLengths,
    device uint* flags, const ulong index,
    const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> summary) {
    firstKeys[index] = summary.firstKey;
    lastKeys[index] = summary.lastKey;
    trailingAggregates[index] = summary.trailingAggregate;
    leadingLengths[index] = summary.leadingLength;
    flags[index] = (summary.allSameKey ? 1u : 0u) |
                   (summary.valid ? 2u : 0u);
}

#define DEFINE_SCAN_BY_KEY_PARALLEL(NAME, KEY_TYPE, VALUE_TYPE, ONE, MINIMUM, \
                                    MAXIMUM)                                \
kernel void NAME##_tile_scan(                                               \
    const device KEY_TYPE* keys [[buffer(0)]],                              \
    const device VALUE_TYPE* input [[buffer(1)]],                           \
    device VALUE_TYPE* output [[buffer(2)]],                                \
    device KEY_TYPE* tileFirstKeys [[buffer(3)]],                           \
    device KEY_TYPE* tileLastKeys [[buffer(4)]],                            \
    device VALUE_TYPE* tileTrailingAggregates [[buffer(5)]],                \
    device uint* tileLeadingLengths [[buffer(6)]],                          \
    device uint* tileFlags [[buffer(7)]],                                    \
    constant ScanByKeyParams& params [[buffer(8)]],                         \
    uint3 group [[threadgroup_position_in_grid]],                           \
    uint lane [[thread_index_in_threadgroup]]) {                            \
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *  \
                        params.dims[3];                                     \
    const ulong length = params.dims[params.dimension];                     \
    if (length == 0) return;                                                \
    const ulong lineCount = total / length;                                \
    const ulong tileCount = (length + 4095) / 4096;                        \
    if (ulong(group.x) >= tileCount || ulong(group.y) >= lineCount) return; \
    ulong coordinates[4] = {0, 0, 0, 0};                                    \
    ulong coordinateQ = ulong(group.y);                                     \
    for (uint d = 0; d < 4; ++d) {                                         \
        if (d != params.dimension) {                                        \
            coordinates[d] = coordinateQ % params.dims[d];                 \
            coordinateQ /= params.dims[d];                                 \
        }                                                                   \
    }                                                                       \
    long keyBase = 0;                                                       \
    long inputBase = 0;                                                     \
    long outputBase = 0;                                                    \
    for (uint d = 0; d < 4; ++d) {                                         \
        keyBase += long(coordinates[d]) * params.keyStrides[d];            \
        inputBase += long(coordinates[d]) * params.inputStrides[d];        \
        outputBase += long(coordinates[d]) * params.outputStrides[d];      \
    }                                                                       \
    const ulong tileBegin = ulong(group.x) * 4096;                         \
    const ulong tileEnd = min(tileBegin + 4096, length);                   \
    const ulong laneBegin = tileBegin + ulong(lane) * 16;                  \
    VALUE_TYPE identity = VALUE_TYPE(0);                                   \
    if (params.operation == SBK_MUL) identity = (ONE);                     \
    if (params.operation == SBK_MIN) identity = (MINIMUM);                 \
    if (params.operation == SBK_MAX) identity = (MAXIMUM);                 \
    VALUE_TYPE accumulated = identity;                                     \
    ScanByKeySummary<KEY_TYPE, VALUE_TYPE> localSummary =                   \
        sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity);                 \
    KEY_TYPE previousKey = KEY_TYPE(0);                                    \
    bool leadingRunOpen = true;                                            \
    for (uint j = 0; j < 16; ++j) {                                       \
        const ulong position = laneBegin + ulong(j);                       \
        if (position >= tileEnd) break;                                    \
        const long keyIndex =                                              \
            keyBase + long(position) * params.keyStrides[params.dimension];\
        const long inputIndex =                                            \
            inputBase + long(position) * params.inputStrides[params.dimension]; \
        const long outputIndex =                                           \
            outputBase + long(position) * params.outputStrides[params.dimension]; \
        const KEY_TYPE key = keys[keyIndex];                               \
        VALUE_TYPE value = input[inputIndex];                              \
        if ((params.operation == SBK_MIN ||                                \
             params.operation == SBK_MAX) &&                               \
            sbkIsNan(value))                                                \
            value = params.operation == SBK_MIN ? (MINIMUM) : (MAXIMUM);   \
        if (!localSummary.valid) {                                         \
            localSummary.firstKey = key;                                   \
            localSummary.lastKey = key;                                    \
            localSummary.trailingAggregate = value;                        \
            localSummary.leadingLength = 1;                                \
            localSummary.allSameKey = 1;                                   \
            localSummary.valid = 1;                                        \
        } else {                                                            \
            if (key != localSummary.lastKey) {                             \
                localSummary.trailingAggregate = value;                    \
                localSummary.allSameKey = 0;                               \
            } else {                                                        \
                localSummary.trailingAggregate = sbkCombine(               \
                    localSummary.trailingAggregate, value, params.operation); \
            }                                                               \
            if (leadingRunOpen && key == localSummary.firstKey)            \
                ++localSummary.leadingLength;                              \
            else                                                           \
                leadingRunOpen = false;                                    \
            localSummary.lastKey = key;                                    \
        }                                                                   \
        if (j == 0 || key != previousKey) accumulated = identity;          \
        if (!params.inclusive) output[outputIndex] = accumulated;          \
        accumulated = sbkCombine(accumulated, value, params.operation);    \
        if (params.inclusive) output[outputIndex] = accumulated;           \
        previousKey = key;                                                 \
    }                                                                       \
    threadgroup ScanByKeySummary<KEY_TYPE, VALUE_TYPE> laneSummaries[256]; \
    laneSummaries[lane] = localSummary;                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    for (uint offset = 1; offset < 256; offset <<= 1) {                    \
        ScanByKeySummary<KEY_TYPE, VALUE_TYPE> left =                       \
            sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity);             \
        if (lane >= offset) left = laneSummaries[lane - offset];           \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> right =                \
            laneSummaries[lane];                                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        if (lane >= offset)                                                \
            laneSummaries[lane] = sbkConcatSummaries(                      \
                left, right, params.operation);                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                       \
    if (localSummary.valid && lane > 0) {                                  \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> previousLane =         \
            laneSummaries[lane - 1];                                       \
        if (previousLane.valid &&                                          \
            previousLane.lastKey == localSummary.firstKey) {              \
            for (uint j = 0; j < localSummary.leadingLength; ++j) {         \
                const ulong position = laneBegin + ulong(j);               \
                const long outputIndex =                                   \
                    outputBase + long(position) *                          \
                                     params.outputStrides[params.dimension];\
                output[outputIndex] = sbkCombine(                          \
                    previousLane.trailingAggregate, output[outputIndex],   \
                    params.operation);                                     \
            }                                                               \
        }                                                                   \
    }                                                                       \
    if (lane == 255) {                                                      \
        const ulong summaryIndex = ulong(group.y) * tileCount + group.x;   \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> tileSummary =          \
            laneSummaries[255];                                            \
        tileFirstKeys[summaryIndex] = tileSummary.firstKey;                \
        tileLastKeys[summaryIndex] = tileSummary.lastKey;                  \
        tileTrailingAggregates[summaryIndex] =                              \
            tileSummary.trailingAggregate;                                  \
        tileLeadingLengths[summaryIndex] = tileSummary.leadingLength;      \
        tileFlags[summaryIndex] = (tileSummary.allSameKey ? 1u : 0u) |    \
                                  (tileSummary.valid ? 2u : 0u);           \
    }                                                                       \
}                                                                           \
kernel void NAME##_totals_scan(                                            \
    const device KEY_TYPE* tileFirstKeys [[buffer(0)]],                     \
    const device KEY_TYPE* tileLastKeys [[buffer(1)]],                      \
    const device VALUE_TYPE* tileTrailingAggregates [[buffer(2)]],          \
    const device uint* tileLeadingLengths [[buffer(3)]],                    \
    const device uint* tileFlags [[buffer(4)]],                             \
    device KEY_TYPE* prefixFirstKeys [[buffer(5)]],                         \
    device KEY_TYPE* prefixLastKeys [[buffer(6)]],                          \
    device VALUE_TYPE* prefixTrailingAggregates [[buffer(7)]],              \
    device uint* prefixLeadingLengths [[buffer(8)]],                        \
    device uint* prefixFlags [[buffer(9)]],                                 \
    device KEY_TYPE* blockFirstKeys [[buffer(10)]],                         \
    device KEY_TYPE* blockLastKeys [[buffer(11)]],                          \
    device VALUE_TYPE* blockTrailingAggregates [[buffer(12)]],              \
    device uint* blockLeadingLengths [[buffer(13)]],                        \
    device uint* blockFlags [[buffer(14)]],                                 \
    constant ScanByKeyTotalsParams& params [[buffer(15)]],                  \
    uint3 group [[threadgroup_position_in_grid]],                           \
    uint lane [[thread_index_in_threadgroup]]) {                            \
    if (ulong(group.x) >= params.blockCount) return;                        \
    VALUE_TYPE identity = VALUE_TYPE(0);                                   \
    if (params.operation == SBK_MUL) identity = (ONE);                     \
    if (params.operation == SBK_MIN) identity = (MINIMUM);                 \
    if (params.operation == SBK_MAX) identity = (MAXIMUM);                 \
    const ulong summaryBase = ulong(group.y) * params.tileCount;            \
    const ulong blockBegin = ulong(group.x) * 4096;                         \
    ScanByKeySummary<KEY_TYPE, VALUE_TYPE> laneSummary =                    \
        sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity);                 \
    for (uint j = 0; j < 16; ++j) {                                       \
        const ulong tile = blockBegin + ulong(lane) * 16 + j;              \
        if (tile >= params.tileCount) break;                               \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> next =                 \
            sbkLoadSummary(tileFirstKeys, tileLastKeys,                    \
                           tileTrailingAggregates, tileLeadingLengths,      \
                           tileFlags, summaryBase + tile);                 \
        laneSummary = sbkConcatSummaries(laneSummary, next,                 \
                                         params.operation);                 \
    }                                                                       \
    threadgroup ScanByKeySummary<KEY_TYPE, VALUE_TYPE> laneSummaries[256]; \
    laneSummaries[lane] = laneSummary;                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    for (uint offset = 1; offset < 256; offset <<= 1) {                    \
        ScanByKeySummary<KEY_TYPE, VALUE_TYPE> left =                       \
            sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity);             \
        if (lane >= offset) left = laneSummaries[lane - offset];           \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> right =                \
            laneSummaries[lane];                                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        if (lane >= offset)                                                \
            laneSummaries[lane] = sbkConcatSummaries(                      \
                left, right, params.operation);                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                       \
    const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> previousLane =             \
        lane == 0 ? sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity)       \
                  : laneSummaries[lane - 1];                                \
    ScanByKeySummary<KEY_TYPE, VALUE_TYPE> localPrefix =                    \
        sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity);                 \
    for (uint j = 0; j < 16; ++j) {                                       \
        const ulong tile = blockBegin + ulong(lane) * 16 + j;              \
        if (tile >= params.tileCount) break;                               \
        const ulong index = summaryBase + tile;                            \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> next =                 \
            sbkLoadSummary(tileFirstKeys, tileLastKeys,                    \
                           tileTrailingAggregates, tileLeadingLengths,      \
                           tileFlags, index);                               \
        localPrefix = sbkConcatSummaries(localPrefix, next,                 \
                                         params.operation);                 \
        const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> tilePrefix =            \
            sbkConcatSummaries(previousLane, localPrefix,                   \
                               params.operation);                           \
        sbkStoreSummary(prefixFirstKeys, prefixLastKeys,                   \
                        prefixTrailingAggregates, prefixLeadingLengths,     \
                        prefixFlags, index, tilePrefix);                    \
    }                                                                       \
    if (lane == 255) {                                                      \
        const ulong blockIndex = ulong(group.y) * params.blockCount +       \
                                 group.x;                                   \
        sbkStoreSummary(blockFirstKeys, blockLastKeys,                     \
                        blockTrailingAggregates, blockLeadingLengths,       \
                        blockFlags, blockIndex, laneSummaries[255]);        \
    }                                                                       \
}                                                                           \
kernel void NAME##_carry_apply(                                            \
    device VALUE_TYPE* output [[buffer(0)]],                               \
    const device KEY_TYPE* tileFirstKeys [[buffer(1)]],                     \
    const device KEY_TYPE* tileLastKeys [[buffer(2)]],                      \
    const device VALUE_TYPE* tileTrailingAggregates [[buffer(3)]],          \
    const device uint* tileLeadingLengths [[buffer(4)]],                    \
    const device uint* tileFlags [[buffer(5)]],                             \
    const device KEY_TYPE* prefixFirstKeys [[buffer(6)]],                   \
    const device KEY_TYPE* prefixLastKeys [[buffer(7)]],                    \
    const device VALUE_TYPE* prefixTrailingAggregates [[buffer(8)]],        \
    const device uint* prefixLeadingLengths [[buffer(9)]],                  \
    const device uint* prefixFlags [[buffer(10)]],                           \
    const device KEY_TYPE* blockFirstKeys [[buffer(11)]],                   \
    const device KEY_TYPE* blockLastKeys [[buffer(12)]],                    \
    const device VALUE_TYPE* blockTrailingAggregates [[buffer(13)]],        \
    const device uint* blockLeadingLengths [[buffer(14)]],                  \
    const device uint* blockFlags [[buffer(15)]],                            \
    constant ScanByKeyTotalsParams& totals [[buffer(16)]],                  \
    constant ScanByKeyParams& params [[buffer(17)]],                        \
    uint3 group [[threadgroup_position_in_grid]],                           \
    uint lane [[thread_index_in_threadgroup]]) {                            \
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *  \
                        params.dims[3];                                     \
    const ulong length = params.dims[params.dimension];                     \
    if (length == 0 || group.x == 0) return;                                \
    const ulong lineCount = total / length;                                \
    if (ulong(group.x) >= totals.tileCount ||                             \
        ulong(group.y) >= lineCount) return;                               \
    ulong coordinates[4] = {0, 0, 0, 0};                                    \
    ulong coordinateQ = ulong(group.y);                                     \
    for (uint d = 0; d < 4; ++d) {                                         \
        if (d != params.dimension) {                                        \
            coordinates[d] = coordinateQ % params.dims[d];                 \
            coordinateQ /= params.dims[d];                                 \
        }                                                                   \
    }                                                                       \
    long outputBase = 0;                                                    \
    for (uint d = 0; d < 4; ++d)                                           \
        outputBase += long(coordinates[d]) * params.outputStrides[d];      \
    const ulong summaryBase = ulong(group.y) * totals.tileCount;           \
    const ulong currentIndex = summaryBase + group.x;                       \
    const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> current =                  \
        sbkLoadSummary(tileFirstKeys, tileLastKeys,                         \
                       tileTrailingAggregates, tileLeadingLengths, tileFlags, \
                       currentIndex);                                       \
    threadgroup VALUE_TYPE carry;                                          \
    threadgroup uint carryValid;                                           \
    threadgroup uint leadingLength;                                        \
    if (lane == 0) {                                                       \
        VALUE_TYPE identity = VALUE_TYPE(0);                               \
        if (params.operation == SBK_MUL) identity = (ONE);                 \
        if (params.operation == SBK_MIN) identity = (MINIMUM);             \
        if (params.operation == SBK_MAX) identity = (MAXIMUM);             \
        ScanByKeySummary<KEY_TYPE, VALUE_TYPE> prefix =                    \
            sbkInvalidSummary<KEY_TYPE, VALUE_TYPE>(identity);             \
        const ulong blockIndex = ulong(group.x) / 4096;                    \
        const ulong tileInBlock = ulong(group.x) % 4096;                    \
        const ulong blockBase = ulong(group.y) * totals.blockCount;          \
        for (ulong priorBlock = 0; priorBlock < blockIndex; ++priorBlock) { \
            const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> prior =            \
                sbkLoadSummary(blockFirstKeys, blockLastKeys,               \
                               blockTrailingAggregates,                     \
                               blockLeadingLengths, blockFlags,             \
                               blockBase + priorBlock);                     \
            prefix = sbkConcatSummaries(prefix, prior, totals.operation);  \
        }                                                                   \
        if (tileInBlock > 0) {                                              \
            const ulong priorTileIndex = currentIndex - 1;                 \
            const ScanByKeySummary<KEY_TYPE, VALUE_TYPE> withinBlock =      \
                sbkLoadSummary(prefixFirstKeys, prefixLastKeys,             \
                               prefixTrailingAggregates,                    \
                               prefixLeadingLengths, prefixFlags,           \
                               priorTileIndex);                            \
            prefix = sbkConcatSummaries(prefix, withinBlock,                \
                                        totals.operation);                  \
        }                                                                   \
        carryValid = prefix.valid && prefix.lastKey == current.firstKey;   \
        carry = carryValid ? prefix.trailingAggregate : identity;          \
        leadingLength = current.leadingLength;                              \
    }                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    if (carryValid) {                                                       \
        const ulong tileBegin = ulong(group.x) * 4096;                     \
        const ulong tileEnd = min(tileBegin + 4096, length);               \
        for (uint j = 0; j < 16; ++j) {                                    \
            const ulong tileOffset = ulong(lane) * 16 + j;                 \
            const ulong position = tileBegin + tileOffset;                 \
            if (tileOffset >= leadingLength || position >= tileEnd) break; \
            const long outputIndex =                                       \
                outputBase + long(position) *                              \
                                 params.outputStrides[params.dimension];   \
            output[outputIndex] = sbkCombine(carry, output[outputIndex],   \
                                             params.operation);            \
        }                                                                   \
    }                                                                       \
}

#define DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, VALUE_NAME, VALUE_TYPE, \
                                ONE, MINIMUM, MAXIMUM)                      \
    DEFINE_SCAN_BY_KEY_PARALLEL(                                           \
        scan_by_key_##KEY_NAME##_##VALUE_NAME, KEY_TYPE, VALUE_TYPE, ONE,  \
        MINIMUM, MAXIMUM)

#define DEFINE_SCAN_BY_KEY_VALUES(KEY_NAME, KEY_TYPE)                      \
    DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, float, float, 1.0f,       \
                            INFINITY, -INFINITY)                            \
    DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, cfloat, float2,            \
                            (float2(1.0f, 0.0f)),                           \
                            (float2(INFINITY, 0.0f)), (float2(0.0f)))      \
    DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, int, int, 1,               \
                            0x7fffffff, (-0x7fffffff - 1))                  \
    DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, uint, uint, 1u,            \
                            0xffffffffu, 0u)                                \
    DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, long, long, 1L,            \
                            0x7fffffffffffffffL,                           \
                            (-0x7fffffffffffffffL - 1L))                    \
    DEFINE_SCAN_BY_KEY_PAIR(KEY_NAME, KEY_TYPE, ulong, ulong, 1UL,         \
                            0xffffffffffffffffUL, 0UL)

DEFINE_SCAN_BY_KEY_VALUES(uint, uint)
DEFINE_SCAN_BY_KEY_VALUES(ulong, ulong)
