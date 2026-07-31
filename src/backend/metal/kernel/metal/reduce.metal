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

struct ReduceParams {
    ulong outputDims[4];
    long outputStrides[4];
    ulong inputDims[4];
    long inputStrides[4];
    uint dimension;
    uint operation;
    uint reduceAll;
    uint changeNan;
    float nanValue;
};

struct ReduceByKeyCompactParams {
    ulong length;
    long inputStride;
};

struct ReduceByKeyParams {
    ulong outputDims[4];
    long outputStrides[4];
    ulong inputDims[4];
    long inputStrides[4];
    long keyStride;
    uint dimension;
    uint operation;
    uint nReduced;
    uint changeNan;
    float nanValue;
};

constant uint REDUCE_ADD = 0;
constant uint REDUCE_MUL = 1;
constant uint REDUCE_MIN = 2;
constant uint REDUCE_MAX = 3;
constant uint REDUCE_NOTZERO = 4;
constant uint REDUCE_OR = 5;
constant uint REDUCE_AND = 6;

template<typename T>
bool reduceIsNan(const T) {
    return false;
}

bool reduceIsNan(const half value) { return isnan(float(value)); }
bool reduceIsNan(const float value) { return isnan(value); }
bool reduceIsNan(const float2 value) { return any(isnan(value)); }

template<typename T>
bool reduceNonzero(const T value) {
    return value != T(0);
}

bool reduceNonzero(const float2 value) {
    return any(value != float2(0.0f));
}

template<typename Ti, typename To>
void reduceAssign(thread To& output, const Ti input) {
    output = To(input);
}

template<typename To>
void reduceAssign(thread To& output, const float2 input) {
    output = To(input.x);
}

void reduceAssign(thread float2& output, const float2 input) {
    output = input;
}

template<typename T>
void reduceSetNan(thread T& output, const float value) {
    output = T(value);
}

void reduceSetNan(thread float2& output, const float value) {
    output = float2(value, 0.0f);
}

template<typename T>
T reduceCombine(const T lhs, const T rhs, const uint operation) {
    if (operation == REDUCE_MUL) return lhs * rhs;
    if (operation == REDUCE_MIN) return min(lhs, rhs);
    if (operation == REDUCE_MAX) return max(lhs, rhs);
    if (operation == REDUCE_OR) return T(lhs != T(0) || rhs != T(0));
    if (operation == REDUCE_AND) return T(lhs != T(0) && rhs != T(0));
    return lhs + rhs;
}

float2 reduceCombine(const float2 lhs, const float2 rhs,
                     const uint operation) {
    if (operation == REDUCE_MUL) {
        return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                      lhs.x * rhs.y + lhs.y * rhs.x);
    }
    if (operation == REDUCE_MIN)
        return dot(lhs, lhs) < dot(rhs, rhs) ? lhs : rhs;
    if (operation == REDUCE_MAX)
        return dot(lhs, lhs) > dot(rhs, rhs) ? lhs : rhs;
    return lhs + rhs;
}

#define REDUCE_VALUE(INPUT_TYPE, COMPUTE_TYPE, INPUT_INDEX, MINIMUM, MAXIMUM) \
    const INPUT_TYPE inputValue = input[INPUT_INDEX];                        \
    COMPUTE_TYPE value = COMPUTE_TYPE(0);                                    \
    reduceAssign(value, inputValue);                                         \
    if (p.operation == REDUCE_NOTZERO || p.operation == REDUCE_OR ||         \
        p.operation == REDUCE_AND)                                           \
        value = COMPUTE_TYPE(reduceNonzero(inputValue));                     \
    if ((p.operation == REDUCE_MIN || p.operation == REDUCE_MAX) &&          \
        reduceIsNan(inputValue))                                             \
        value = p.operation == REDUCE_MIN ? (MINIMUM) : (MAXIMUM);           \
    if (p.changeNan && reduceIsNan(value)) reduceSetNan(value, p.nanValue);

#define DEFINE_REDUCE(NAME, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE, ONE,       \
                      MINIMUM, MAXIMUM)                                      \
kernel void NAME(const device INPUT_TYPE* input [[buffer(0)]],               \
                 device OUTPUT_TYPE* output [[buffer(1)]],                   \
                 constant ReduceParams& p [[buffer(2)]],                     \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong outputTotal = p.outputDims[0] * p.outputDims[1] *            \
                              p.outputDims[2] * p.outputDims[3];              \
    if (gid >= outputTotal) return;                                           \
    COMPUTE_TYPE result = COMPUTE_TYPE(0);                                   \
    if (p.operation == REDUCE_MUL || p.operation == REDUCE_AND)              \
        result = (ONE);                                                       \
    if (p.operation == REDUCE_MIN) result = (MINIMUM);                       \
    if (p.operation == REDUCE_MAX) result = (MAXIMUM);                       \
    if (p.reduceAll) {                                                        \
        const ulong inputTotal = p.inputDims[0] * p.inputDims[1] *           \
                                 p.inputDims[2] * p.inputDims[3];             \
        for (ulong linear = 0; linear < inputTotal; ++linear) {              \
            ulong q = linear;                                                \
            long inputIndex = 0;                                             \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.inputDims[d];                 \
                q /= p.inputDims[d];                                         \
                inputIndex += long(coordinate) * p.inputStrides[d];          \
            }                                                                \
            REDUCE_VALUE(INPUT_TYPE, COMPUTE_TYPE, inputIndex, MINIMUM,      \
                         MAXIMUM)                                             \
            result = reduceCombine(result, value, p.operation);              \
        }                                                                    \
        output[0] = OUTPUT_TYPE(result);                                     \
        return;                                                              \
    }                                                                        \
    ulong q = gid;                                                           \
    long outputIndex = 0;                                                    \
    long inputBase = 0;                                                      \
    for (uint d = 0; d < 4; ++d) {                                          \
        const ulong coordinate = q % p.outputDims[d];                        \
        q /= p.outputDims[d];                                                \
        outputIndex += long(coordinate) * p.outputStrides[d];                \
        inputBase += long(coordinate) * p.inputStrides[d];                   \
    }                                                                        \
    const ulong length = p.inputDims[p.dimension];                           \
    const long stride = p.inputStrides[p.dimension];                         \
    for (ulong i = 0; i < length; ++i) {                                    \
        const long inputIndex = inputBase + long(i) * stride;                \
        REDUCE_VALUE(INPUT_TYPE, COMPUTE_TYPE, inputIndex, MINIMUM, MAXIMUM) \
        result = reduceCombine(result, value, p.operation);                  \
    }                                                                        \
    output[outputIndex] = OUTPUT_TYPE(result);                               \
}

#define REDUCE_FLOAT(NAME, INPUT, OUTPUT)                                    \
    DEFINE_REDUCE(NAME, INPUT, OUTPUT, float, 1.0f, INFINITY, -INFINITY)
#define REDUCE_INT(NAME, INPUT, OUTPUT)                                      \
    DEFINE_REDUCE(NAME, INPUT, OUTPUT, int, 1, 0x7fffffff,                   \
                  (-0x7fffffff - 1))
#define REDUCE_UINT(NAME, INPUT, OUTPUT)                                     \
    DEFINE_REDUCE(NAME, INPUT, OUTPUT, uint, 1u, 0xffffffffu, 0u)
#define REDUCE_BOOL(NAME, INPUT)                                             \
    DEFINE_REDUCE(NAME, INPUT, char, char, char(1), char(1), char(0))

REDUCE_FLOAT(reduce_float_float, float, float)
DEFINE_REDUCE(reduce_cfloat_cfloat, float2, float2, float2,
              (float2(1.0f, 0.0f)), (float2(INFINITY, 0.0f)),
              (float2(0.0f)))
REDUCE_INT(reduce_int_int, int, int)
REDUCE_UINT(reduce_uint_uint, uint, uint)
DEFINE_REDUCE(reduce_long_long, long, long, long, 1L, 0x7fffffffffffffffL,
              (-0x7fffffffffffffffL - 1L))
DEFINE_REDUCE(reduce_ulong_ulong, ulong, ulong, ulong, 1UL,
              0xffffffffffffffffUL, 0UL)
DEFINE_REDUCE(reduce_bool_bool, char, char, char, char(1), char(1), char(0))
DEFINE_REDUCE(reduce_char_char, char, char, char, char(1), char(127),
              char(-128))
DEFINE_REDUCE(reduce_uchar_uchar, uchar, uchar, uchar, uchar(1), uchar(255),
              uchar(0))
DEFINE_REDUCE(reduce_short_short, short, short, short, short(1), short(32767),
              short(-32768))
DEFINE_REDUCE(reduce_ushort_ushort, ushort, ushort, ushort, ushort(1),
              ushort(65535), ushort(0))
DEFINE_REDUCE(reduce_half_half, half, half, float, 1.0f, INFINITY, -INFINITY)

REDUCE_INT(reduce_bool_int, char, int)
REDUCE_INT(reduce_char_int, char, int)
REDUCE_UINT(reduce_uchar_uint, uchar, uint)
REDUCE_INT(reduce_short_int, short, int)
REDUCE_UINT(reduce_ushort_uint, ushort, uint)

REDUCE_FLOAT(reduce_int_float, int, float)
REDUCE_FLOAT(reduce_uint_float, uint, float)
REDUCE_FLOAT(reduce_bool_float, char, float)
REDUCE_FLOAT(reduce_char_float, char, float)
REDUCE_FLOAT(reduce_uchar_float, uchar, float)
REDUCE_FLOAT(reduce_short_float, short, float)
REDUCE_FLOAT(reduce_ushort_float, ushort, float)
REDUCE_FLOAT(reduce_half_float, half, float)

REDUCE_UINT(reduce_float_uint, float, uint)
REDUCE_UINT(reduce_cfloat_uint, float2, uint)
REDUCE_UINT(reduce_int_uint, int, uint)
REDUCE_UINT(reduce_long_uint, long, uint)
REDUCE_UINT(reduce_ulong_uint, ulong, uint)
REDUCE_UINT(reduce_bool_uint, char, uint)
REDUCE_UINT(reduce_char_uint, char, uint)
REDUCE_UINT(reduce_short_uint, short, uint)
REDUCE_UINT(reduce_half_uint, half, uint)

REDUCE_BOOL(reduce_float_bool, float)
REDUCE_BOOL(reduce_cfloat_bool, float2)
REDUCE_BOOL(reduce_int_bool, int)
REDUCE_BOOL(reduce_uint_bool, uint)
REDUCE_BOOL(reduce_long_bool, long)
REDUCE_BOOL(reduce_ulong_bool, ulong)
REDUCE_BOOL(reduce_char_bool, char)
REDUCE_BOOL(reduce_uchar_bool, uchar)
REDUCE_BOOL(reduce_short_bool, short)
REDUCE_BOOL(reduce_ushort_bool, ushort)
REDUCE_BOOL(reduce_half_bool, half)

#define DEFINE_REDUCE_BY_KEY_COMPACT(NAME, KEY_TYPE)                         \
kernel void NAME(const device KEY_TYPE* input [[buffer(0)]],                 \
                 device KEY_TYPE* output [[buffer(1)]],                      \
                 device int* count [[buffer(2)]],                            \
                 constant ReduceByKeyCompactParams& p [[buffer(3)]],         \
                 uint gid [[thread_position_in_grid]]) {                     \
    if (gid != 0 || p.length == 0) return;                                   \
    uint outputIndex = 0;                                                    \
    KEY_TYPE current = input[0];                                             \
    for (ulong i = 1; i < p.length; ++i) {                                  \
        const KEY_TYPE next = input[long(i) * p.inputStride];                \
        if (next != current) {                                               \
            output[outputIndex++] = current;                                 \
            current = next;                                                  \
        }                                                                    \
    }                                                                        \
    output[outputIndex] = current;                                           \
    count[0] = int(outputIndex + 1);                                         \
}

DEFINE_REDUCE_BY_KEY_COMPACT(reduce_by_key_compact_int, int)
DEFINE_REDUCE_BY_KEY_COMPACT(reduce_by_key_compact_uint, uint)

#define DEFINE_REDUCE_BY_KEY(NAME, KEY_TYPE, INPUT_TYPE, OUTPUT_TYPE,        \
                             COMPUTE_TYPE, ONE, MINIMUM, MAXIMUM)            \
kernel void NAME(const device KEY_TYPE* keys [[buffer(0)]],                  \
                 const device INPUT_TYPE* input [[buffer(1)]],               \
                 device OUTPUT_TYPE* output [[buffer(2)]],                   \
                 constant ReduceByKeyParams& p [[buffer(3)]],                \
                 uint gid [[thread_position_in_grid]]) {                     \
    ulong sliceCount = 1;                                                    \
    for (uint d = 0; d < 4; ++d)                                            \
        if (d != p.dimension) sliceCount *= p.inputDims[d];                  \
    if (gid >= sliceCount) return;                                           \
    ulong q = gid;                                                           \
    long inputBase = 0;                                                      \
    long outputBase = 0;                                                     \
    for (uint d = 0; d < 4; ++d) {                                          \
        if (d == p.dimension) continue;                                      \
        const ulong coordinate = q % p.inputDims[d];                         \
        q /= p.inputDims[d];                                                 \
        inputBase += long(coordinate) * p.inputStrides[d];                   \
        outputBase += long(coordinate) * p.outputStrides[d];                 \
    }                                                                        \
    COMPUTE_TYPE result = COMPUTE_TYPE(0);                                   \
    if (p.operation == REDUCE_MUL || p.operation == REDUCE_AND)              \
        result = (ONE);                                                       \
    if (p.operation == REDUCE_MIN) result = (MINIMUM);                       \
    if (p.operation == REDUCE_MAX) result = (MAXIMUM);                       \
    uint keyIndex = 0;                                                       \
    KEY_TYPE currentKey = keys[0];                                           \
    const ulong length = p.inputDims[p.dimension];                           \
    for (ulong i = 0; i < length; ++i) {                                    \
        const KEY_TYPE key = keys[long(i) * p.keyStride];                    \
        if (i != 0 && key != currentKey) {                                   \
            output[outputBase + long(keyIndex) *                             \
                                  p.outputStrides[p.dimension]] =            \
                OUTPUT_TYPE(result);                                         \
            ++keyIndex;                                                      \
            currentKey = key;                                                \
            result = COMPUTE_TYPE(0);                                        \
            if (p.operation == REDUCE_MUL || p.operation == REDUCE_AND)      \
                result = (ONE);                                               \
            if (p.operation == REDUCE_MIN) result = (MINIMUM);               \
            if (p.operation == REDUCE_MAX) result = (MAXIMUM);               \
        }                                                                    \
        const long inputIndex =                                              \
            inputBase + long(i) * p.inputStrides[p.dimension];               \
        REDUCE_VALUE(INPUT_TYPE, COMPUTE_TYPE, inputIndex, MINIMUM, MAXIMUM) \
        result = reduceCombine(result, value, p.operation);                  \
    }                                                                        \
    if (keyIndex < p.nReduced)                                               \
        output[outputBase + long(keyIndex) *                                 \
                              p.outputStrides[p.dimension]] =                \
            OUTPUT_TYPE(result);                                             \
}

#define REDUCE_BY_KEY_PAIR(INPUT_NAME, OUTPUT_NAME, INPUT_TYPE, OUTPUT_TYPE, \
                           COMPUTE_TYPE, ONE, MINIMUM, MAXIMUM)              \
    DEFINE_REDUCE_BY_KEY(reduce_by_key_int_##INPUT_NAME##_##OUTPUT_NAME,     \
                         int, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE, ONE,     \
                         MINIMUM, MAXIMUM)                                   \
    DEFINE_REDUCE_BY_KEY(reduce_by_key_uint_##INPUT_NAME##_##OUTPUT_NAME,    \
                         uint, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE, ONE,    \
                         MINIMUM, MAXIMUM)

#define RBK_FLOAT(INPUT_NAME, OUTPUT_NAME, INPUT, OUTPUT)                    \
    REDUCE_BY_KEY_PAIR(INPUT_NAME, OUTPUT_NAME, INPUT, OUTPUT, float, 1.0f,  \
                       INFINITY, -INFINITY)
#define RBK_INT(INPUT_NAME, OUTPUT_NAME, INPUT, OUTPUT)                      \
    REDUCE_BY_KEY_PAIR(INPUT_NAME, OUTPUT_NAME, INPUT, OUTPUT, int, 1,       \
                       0x7fffffff, (-0x7fffffff - 1))
#define RBK_UINT(INPUT_NAME, OUTPUT_NAME, INPUT, OUTPUT)                     \
    REDUCE_BY_KEY_PAIR(INPUT_NAME, OUTPUT_NAME, INPUT, OUTPUT, uint, 1u,     \
                       0xffffffffu, 0u)
#define RBK_BOOL(INPUT_NAME, INPUT)                                          \
    REDUCE_BY_KEY_PAIR(INPUT_NAME, bool, INPUT, char, char, char(1),         \
                       char(1), char(0))

RBK_FLOAT(float, float, float, float)
REDUCE_BY_KEY_PAIR(cfloat, cfloat, float2, float2, float2,
                   (float2(1.0f, 0.0f)), (float2(INFINITY, 0.0f)),
                   (float2(0.0f)))
RBK_INT(int, int, int, int)
RBK_UINT(uint, uint, uint, uint)
REDUCE_BY_KEY_PAIR(long, long, long, long, long, 1L, 0x7fffffffffffffffL,
                   (-0x7fffffffffffffffL - 1L))
REDUCE_BY_KEY_PAIR(ulong, ulong, ulong, ulong, ulong, 1UL,
                   0xffffffffffffffffUL, 0UL)
REDUCE_BY_KEY_PAIR(bool, bool, char, char, char, char(1), char(1), char(0))
REDUCE_BY_KEY_PAIR(char, char, char, char, char, char(1), char(127),
                   char(-128))
REDUCE_BY_KEY_PAIR(uchar, uchar, uchar, uchar, uchar, uchar(1), uchar(255),
                   uchar(0))
REDUCE_BY_KEY_PAIR(short, short, short, short, short, short(1), short(32767),
                   short(-32768))
REDUCE_BY_KEY_PAIR(ushort, ushort, ushort, ushort, ushort, ushort(1),
                   ushort(65535), ushort(0))
REDUCE_BY_KEY_PAIR(half, half, half, half, float, 1.0f, INFINITY, -INFINITY)

RBK_INT(bool, int, char, int)
RBK_INT(char, int, char, int)
RBK_UINT(uchar, uint, uchar, uint)
RBK_INT(short, int, short, int)
RBK_UINT(ushort, uint, ushort, uint)

RBK_FLOAT(int, float, int, float)
RBK_FLOAT(uint, float, uint, float)
RBK_FLOAT(bool, float, char, float)
RBK_FLOAT(char, float, char, float)
RBK_FLOAT(uchar, float, uchar, float)
RBK_FLOAT(short, float, short, float)
RBK_FLOAT(ushort, float, ushort, float)
RBK_FLOAT(half, float, half, float)

RBK_UINT(float, uint, float, uint)
RBK_UINT(cfloat, uint, float2, uint)
RBK_UINT(int, uint, int, uint)
RBK_UINT(long, uint, long, uint)
RBK_UINT(ulong, uint, ulong, uint)
RBK_UINT(bool, uint, char, uint)
RBK_UINT(char, uint, char, uint)
RBK_UINT(short, uint, short, uint)
RBK_UINT(half, uint, half, uint)

RBK_BOOL(float, float)
RBK_BOOL(cfloat, float2)
RBK_BOOL(int, int)
RBK_BOOL(uint, uint)
RBK_BOOL(long, long)
RBK_BOOL(ulong, ulong)
RBK_BOOL(char, char)
RBK_BOOL(uchar, uchar)
RBK_BOOL(short, short)
RBK_BOOL(ushort, ushort)
RBK_BOOL(half, half)
