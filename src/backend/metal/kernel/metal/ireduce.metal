/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct IReduceParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong rlenDims[4];
    ulong rlenStrides[4];
    uint dimension;
    uint isMax;
};

struct IReduceAllParams {
    ulong inputDims[4];
    ulong inputStrides[4];
    uint isMax;
};

template<typename T>
bool ireduceIsNan(const T) { return false; }
bool ireduceIsNan(const half v) { return isnan(float(v)); }
bool ireduceIsNan(const float v) { return isnan(v); }
bool ireduceIsNan(const float2 v) { return any(isnan(v)); }

template<typename T>
float ireduceScore(const T v) { return float(v); }
float ireduceScore(const float2 v) { return dot(v, v); }

template<typename T>
T ireduceConvert(const T v) { return v; }
float ireduceConvert(const half v) { return float(v); }
float2 ireduceConvert(const float2 v) { return v; }

#define DEFINE_IREDUCE(NAME, INPUT_TYPE, COMPUTE_TYPE, MINIMUM, MAXIMUM)     \
kernel void NAME(const device INPUT_TYPE* input [[buffer(0)]],               \
                 device INPUT_TYPE* output [[buffer(1)]],                   \
                 device uint* locations [[buffer(2)]],                       \
                 constant IReduceParams& p [[buffer(3)]],                    \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                  \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                 \
    ulong q = gid, c[4];                                                      \
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];                         \
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];                         \
    c[2] = q % p.outputDims[2]; q /= p.outputDims[2];                         \
    c[3] = q;                                                                \
    ulong oi = 0, ii = 0;                                                     \
    for (uint d = 0; d < 4; ++d) {                                            \
        oi += c[d] * p.outputStrides[d];                                      \
        ii += c[d] * p.inputStrides[d];                                       \
    }                                                                         \
    COMPUTE_TYPE best = p.isMax ? COMPUTE_TYPE(MAXIMUM) :                    \
                                     COMPUTE_TYPE(MINIMUM);                   \
    uint bestIndex = 0;                                                       \
    bool found = false;                                                       \
    for (uint i = 0; i < p.inputDims[p.dimension]; ++i) {                     \
        const INPUT_TYPE raw =                                               \
            input[ii + ulong(i) * p.inputStrides[p.dimension]];               \
        if (ireduceIsNan(raw)) continue;                                      \
        const COMPUTE_TYPE value = COMPUTE_TYPE(ireduceConvert(raw));        \
        const bool better =                                                   \
            !found || (p.isMax ? (ireduceScore(value) > ireduceScore(best) || \
                                  (ireduceScore(value) == ireduceScore(best) && \
                                   i <= bestIndex))                          \
                              : (ireduceScore(value) < ireduceScore(best) ||  \
                                 (ireduceScore(value) == ireduceScore(best) && \
                                  i > bestIndex)));                          \
        if (better) { best = value; bestIndex = i; found = true; }             \
    }                                                                         \
    output[oi] = INPUT_TYPE(best);                                             \
    locations[oi] = bestIndex;                                                 \
}

DEFINE_IREDUCE(ireduce_float, float, float, INFINITY, -INFINITY)
DEFINE_IREDUCE(ireduce_cfloat, float2, float2, float2(INFINITY), float2(-INFINITY))
DEFINE_IREDUCE(ireduce_int, int, int, 0x7fffffff, (-0x7fffffff - 1))
DEFINE_IREDUCE(ireduce_uint, uint, uint, 0xffffffffu, 0u)
DEFINE_IREDUCE(ireduce_long, long, long, 0x7fffffffffffffffL,
               (-0x7fffffffffffffffL - 1L))
DEFINE_IREDUCE(ireduce_ulong, ulong, ulong, 0xffffffffffffffffUL, 0UL)
DEFINE_IREDUCE(ireduce_char, char, char, 127, -128)
DEFINE_IREDUCE(ireduce_uchar, uchar, uchar, 255, 0)
DEFINE_IREDUCE(ireduce_short, short, short, 32767, -32768)
DEFINE_IREDUCE(ireduce_ushort, ushort, ushort, 65535, 0)
DEFINE_IREDUCE(ireduce_half, half, float, INFINITY, -INFINITY)

#define DEFINE_IREDUCE_ALL_PARALLEL(NAME, INPUT_TYPE, COMPUTE_TYPE, MINIMUM, \
                                    MAXIMUM)                                  \
kernel void NAME##_partial(                                                  \
    const device INPUT_TYPE* input [[buffer(0)]],                            \
    device COMPUTE_TYPE* partialValues [[buffer(1)]],                        \
    device uint* partialIndices [[buffer(2)]],                               \
    device uint* partialValid [[buffer(3)]],                                 \
    constant IReduceAllParams& p [[buffer(4)]],                              \
    uint group [[threadgroup_position_in_grid]],                             \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    threadgroup COMPUTE_TYPE values[256];                                    \
    threadgroup uint indices[256];                                           \
    threadgroup uchar valid[256];                                            \
    const ulong total = p.inputDims[0] * p.inputDims[1] *                    \
                        p.inputDims[2] * p.inputDims[3];                      \
    const ulong linear = ulong(group) * 256 + lane;                          \
    bool found = false;                                                      \
    COMPUTE_TYPE best = p.isMax ? COMPUTE_TYPE(MAXIMUM) :                    \
                                  COMPUTE_TYPE(MINIMUM);                      \
    uint bestIndex = 0;                                                      \
    if (linear < total) {                                                    \
        ulong q = linear, c[4];                                              \
        c[0] = q % p.inputDims[0]; q /= p.inputDims[0];                     \
        c[1] = q % p.inputDims[1]; q /= p.inputDims[1];                     \
        c[2] = q % p.inputDims[2]; q /= p.inputDims[2];                     \
        c[3] = q;                                                            \
        ulong offset = 0;                                                    \
        for (uint d = 0; d < 4; ++d)                                         \
            offset += c[d] * p.inputStrides[d];                              \
        const INPUT_TYPE raw = input[offset];                                \
        if (!ireduceIsNan(raw)) {                                            \
            best = COMPUTE_TYPE(ireduceConvert(raw));                        \
            bestIndex = uint(linear);                                        \
            found = true;                                                    \
        }                                                                    \
    }                                                                        \
    values[lane] = best;                                                    \
    indices[lane] = bestIndex;                                              \
    valid[lane] = found;                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint stride = 128; stride > 0; stride >>= 1) {                     \
        if (lane < stride && valid[lane + stride]) {                         \
            const COMPUTE_TYPE rhs = values[lane + stride];                  \
            const uint rhsIndex = indices[lane + stride];                    \
            const bool better = !valid[lane] ||                             \
                (p.isMax ? (ireduceScore(rhs) > ireduceScore(values[lane]) || \
                            (ireduceScore(rhs) == ireduceScore(values[lane]) &&\
                             rhsIndex < indices[lane]))                      \
                         : (ireduceScore(rhs) < ireduceScore(values[lane]) ||\
                            (ireduceScore(rhs) == ireduceScore(values[lane]) &&\
                             rhsIndex > indices[lane])));                    \
            if (better) {                                                     \
                values[lane] = rhs;                                          \
                indices[lane] = rhsIndex;                                    \
            }                                                                \
            valid[lane] = 1;                                                 \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (lane == 0) {                                                         \
        partialValues[group] = values[0];                                    \
        partialIndices[group] = indices[0];                                  \
        partialValid[group] = valid[0];                                      \
    }                                                                        \
}                                                                            \
kernel void NAME##_final(                                                   \
    const device COMPUTE_TYPE* partialValues [[buffer(0)]],                 \
    const device uint* partialIndices [[buffer(1)]],                         \
    const device uint* partialValid [[buffer(2)]],                           \
    device INPUT_TYPE* output [[buffer(3)]],                                 \
    device uint* locations [[buffer(4)]],                                    \
    constant IReduceAllParams& p [[buffer(5)]],                              \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    threadgroup COMPUTE_TYPE values[256];                                    \
    threadgroup uint indices[256];                                           \
    threadgroup uchar valid[256];                                            \
    const ulong total = p.inputDims[0] * p.inputDims[1] *                    \
                        p.inputDims[2] * p.inputDims[3];                      \
    const ulong chunks = max((total + 255) / 256, 1UL);                     \
    bool found = false;                                                      \
    COMPUTE_TYPE best = p.isMax ? COMPUTE_TYPE(MAXIMUM) :                    \
                                  COMPUTE_TYPE(MINIMUM);                      \
    uint bestIndex = 0;                                                      \
    for (ulong i = lane; i < chunks; i += 256) {                             \
        const uint candidateIndex = partialIndices[i];                       \
        if (partialValid[i] != 0) {                                          \
            const COMPUTE_TYPE candidate = partialValues[i];                 \
            const bool better = !found ||                                    \
                (p.isMax ? (ireduceScore(candidate) > ireduceScore(best) ||  \
                            (ireduceScore(candidate) == ireduceScore(best) &&\
                             candidateIndex < bestIndex))                     \
                         : (ireduceScore(candidate) < ireduceScore(best) ||  \
                            (ireduceScore(candidate) == ireduceScore(best) &&\
                             candidateIndex > bestIndex)));                  \
            if (better) {                                                     \
                best = candidate;                                            \
                bestIndex = candidateIndex;                                  \
            }                                                                \
            found = true;                                                    \
        }                                                                    \
    }                                                                        \
    values[lane] = best;                                                    \
    indices[lane] = bestIndex;                                              \
    valid[lane] = found;                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint stride = 128; stride > 0; stride >>= 1) {                     \
        if (lane < stride && valid[lane + stride]) {                         \
            const COMPUTE_TYPE rhs = values[lane + stride];                  \
            const uint rhsIndex = indices[lane + stride];                    \
            const bool better = !valid[lane] ||                             \
                (p.isMax ? (ireduceScore(rhs) > ireduceScore(values[lane]) || \
                            (ireduceScore(rhs) == ireduceScore(values[lane]) &&\
                             rhsIndex < indices[lane]))                      \
                         : (ireduceScore(rhs) < ireduceScore(values[lane]) ||\
                            (ireduceScore(rhs) == ireduceScore(values[lane]) &&\
                             rhsIndex > indices[lane])));                    \
            if (better) {                                                     \
                values[lane] = rhs;                                          \
                indices[lane] = rhsIndex;                                    \
            }                                                                \
            valid[lane] = 1;                                                 \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (lane == 0) {                                                         \
        output[0] = INPUT_TYPE(valid[0] ? values[0] :                        \
            (p.isMax ? COMPUTE_TYPE(MAXIMUM) : COMPUTE_TYPE(MINIMUM)));      \
        locations[0] = valid[0] ? indices[0] : 0;                             \
    }                                                                        \
}

DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_float, float, float, INFINITY, -INFINITY)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_cfloat, float2, float2,
                            float2(INFINITY), float2(-INFINITY))
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_int, int, int, 0x7fffffff,
                            (-0x7fffffff - 1))
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_uint, uint, uint, 0xffffffffu, 0u)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_long, long, long, 0x7fffffffffffffffL,
                            (-0x7fffffffffffffffL - 1L))
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_ulong, ulong, ulong,
                            0xffffffffffffffffUL, 0UL)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_char, char, char, 127, -128)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_uchar, uchar, uchar, 255, 0)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_short, short, short, 32767, -32768)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_ushort, ushort, ushort, 65535, 0)
DEFINE_IREDUCE_ALL_PARALLEL(ireduce_all_half, half, float, INFINITY, -INFINITY)

#define DEFINE_RREDUCE(NAME, INPUT_TYPE, COMPUTE_TYPE, MINIMUM, MAXIMUM)     \
kernel void NAME(const device INPUT_TYPE* input [[buffer(0)]],               \
                 const device uint* rlen [[buffer(1)]],                     \
                 device uint* locations [[buffer(2)]],                       \
                 device INPUT_TYPE* output [[buffer(3)]],                   \
                 constant IReduceParams& p [[buffer(4)]],                    \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                  \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                 \
    ulong q = gid, c[4];                                                      \
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];                         \
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];                         \
    c[2] = q % p.outputDims[2]; q /= p.outputDims[2];                         \
    c[3] = q;                                                                \
    ulong oi = 0, ii = 0, ri = 0;                                             \
    bool rlenValid = true;                                                    \
    for (uint d = 0; d < 4; ++d) {                                            \
        oi += c[d] * p.outputStrides[d];                                      \
        ii += c[d] * p.inputStrides[d];                                       \
        rlenValid = rlenValid && c[d] < p.rlenDims[d];                        \
        if (rlenValid) ri += c[d] * p.rlenStrides[d];                         \
    }                                                                         \
    uint length = p.inputDims[p.dimension];                                   \
    if (rlenValid) length = min(length, rlen[ri]);                            \
    COMPUTE_TYPE best = p.isMax ? COMPUTE_TYPE(MAXIMUM) :                    \
                                     COMPUTE_TYPE(MINIMUM);                   \
    uint bestIndex = 0;                                                       \
    bool found = false;                                                       \
    for (uint i = 0; i < length; ++i) {                                       \
        const INPUT_TYPE raw = input[ii + ulong(i) * p.inputStrides[p.dimension]]; \
        if (ireduceIsNan(raw)) continue;                                      \
        const COMPUTE_TYPE value = COMPUTE_TYPE(ireduceConvert(raw));        \
        const bool better =                                                   \
            !found || (p.isMax ? (ireduceScore(value) > ireduceScore(best) || \
                                  (ireduceScore(value) == ireduceScore(best) && \
                                   i <= bestIndex))                          \
                              : (ireduceScore(value) < ireduceScore(best) ||  \
                                 (ireduceScore(value) == ireduceScore(best) && \
                                  i > bestIndex)));                          \
        if (better) { best = value; bestIndex = i; found = true; }             \
    }                                                                         \
    output[oi] = INPUT_TYPE(best);                                             \
    locations[oi] = bestIndex;                                                 \
}

DEFINE_RREDUCE(rreduce_float, float, float, INFINITY, -INFINITY)
DEFINE_RREDUCE(rreduce_cfloat, float2, float2, float2(INFINITY), float2(-INFINITY))
DEFINE_RREDUCE(rreduce_int, int, int, 0x7fffffff, (-0x7fffffff - 1))
DEFINE_RREDUCE(rreduce_uint, uint, uint, 0xffffffffu, 0u)
DEFINE_RREDUCE(rreduce_long, long, long, 0x7fffffffffffffffL,
               (-0x7fffffffffffffffL - 1L))
DEFINE_RREDUCE(rreduce_ulong, ulong, ulong, 0xffffffffffffffffUL, 0UL)
DEFINE_RREDUCE(rreduce_char, char, char, 127, -128)
DEFINE_RREDUCE(rreduce_uchar, uchar, uchar, 255, 0)
DEFINE_RREDUCE(rreduce_short, short, short, 32767, -32768)
DEFINE_RREDUCE(rreduce_ushort, ushort, ushort, 65535, 0)
DEFINE_RREDUCE(rreduce_half, half, float, INFINITY, -INFINITY)
