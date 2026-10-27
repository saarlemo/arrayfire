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

struct DotParams {
    ulong length, lhsStride, rhsStride;
    uint conjugateLhs, conjugateRhs;
};

inline float2 dotConjugate(float2 value) { return float2(value.x,-value.y); }
inline float2 dotMultiply(float2 a,float2 b) {
    return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}

kernel void dot_float_partial(const device float* lhs [[buffer(0)]],
                              const device float* rhs [[buffer(1)]],
                              device float* partials [[buffer(2)]],
                              constant DotParams& p [[buffer(3)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    const ulong i = ulong(group) * 256 + lane;
    values[lane] = i < p.length
                       ? lhs[i * p.lhsStride] * rhs[i * p.rhsStride]
                       : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partials[group] = values[0];
}

kernel void dot_float_final(const device float* partials [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant DotParams& p [[buffer(2)]],
                            uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    const ulong chunks = (p.length + 255) / 256;
    float value = 0.0f;
    for (ulong i = lane; i < chunks; i += 256) value += partials[i];
    values[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[0] = values[0];
}

kernel void dot_cfloat_partial(const device float2* lhs [[buffer(0)]],
                               const device float2* rhs [[buffer(1)]],
                               device float2* partials [[buffer(2)]],
                               constant DotParams& p [[buffer(3)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 values[256];
    const ulong i = ulong(group) * 256 + lane;
    float2 value = float2(0.0f);
    if (i < p.length) {
        float2 a = lhs[i * p.lhsStride], b = rhs[i * p.rhsStride];
        if (p.conjugateLhs) a = dotConjugate(a);
        if (p.conjugateRhs) b = dotConjugate(b);
        value = dotMultiply(a, b);
    }
    values[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partials[group] = values[0];
}

kernel void dot_cfloat_final(const device float2* partials [[buffer(0)]],
                             device float2* output [[buffer(1)]],
                             constant DotParams& p [[buffer(2)]],
                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 values[256];
    const ulong chunks = (p.length + 255) / 256;
    float2 value = float2(0.0f);
    for (ulong i = lane; i < chunks; i += 256) value += partials[i];
    values[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[0] = values[0];
}

struct GemmParams {
    ulong outputDims[4];
    long outputStrides[4];
    ulong lhsDims[4];
    long lhsStrides[4];
    ulong rhsDims[4];
    long rhsStrides[4];
    uint transposeLhs;
    uint conjugateLhs;
    uint transposeRhs;
    uint conjugateRhs;
    uint m;
    uint n;
    uint k;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

template<typename T>
inline T gemmConjugate(const T value, const bool) { return value; }
inline float2 gemmConjugate(const float2 value, const bool conjugate) {
    return conjugate ? dotConjugate(value) : value;
}

template<typename T>
inline T gemmMultiply(const T lhs, const T rhs) {
    return lhs * rhs;
}
inline float2 gemmMultiply(const float2 lhs, const float2 rhs) {
    return dotMultiply(lhs, rhs);
}

template<typename T>
inline T gemmScale(const T accum, const T oldValue, const float alphaReal,
                  const float, const float betaReal, const float) {
    return alphaReal * accum + betaReal * oldValue;
}
inline float2 gemmScale(const float2 accum, const float2 oldValue,
                        const float alphaReal, const float alphaImag,
                        const float betaReal, const float betaImag) {
    return dotMultiply(float2(alphaReal, alphaImag), accum) +
           dotMultiply(float2(betaReal, betaImag), oldValue);
}

constant uint GEMM_TILE_M = 16;
constant uint GEMM_TILE_N = 16;
constant uint GEMM_TILE_K = 16;

template<typename T>
inline void gemmTiledImpl(
    const device T* lhs, const device T* rhs, device T* output,
    constant GemmParams& p, threadgroup T* lhsTile, threadgroup T* rhsTile,
    const uint3 tid, const uint3 group) {
    const ulong row = ulong(group.x) * GEMM_TILE_M + tid.x;
    const ulong col = ulong(group.y) * GEMM_TILE_N + tid.y;
    const ulong batch = group.z;
    const ulong batch2 = batch % p.outputDims[2];
    const ulong batch3 = batch / p.outputDims[2];
    const ulong lhsB2 = p.lhsDims[2] == 1 ? 0 : batch2;
    const ulong lhsB3 = p.lhsDims[3] == 1 ? 0 : batch3;
    const ulong rhsB2 = p.rhsDims[2] == 1 ? 0 : batch2;
    const ulong rhsB3 = p.rhsDims[3] == 1 ? 0 : batch3;
    const long lhsBatch = long(lhsB2) * p.lhsStrides[2] +
                          long(lhsB3) * p.lhsStrides[3];
    const long rhsBatch = long(rhsB2) * p.rhsStrides[2] +
                          long(rhsB3) * p.rhsStrides[3];
    const uint linearThread = tid.y * GEMM_TILE_M + tid.x;
    T accum = T(0);

    for (uint kBase = 0; kBase < p.k; kBase += GEMM_TILE_K) {
        const uint lhsRow = p.transposeLhs
                                ? linearThread / GEMM_TILE_K
                                : linearThread % GEMM_TILE_M;
        const uint lhsK = p.transposeLhs
                              ? linearThread % GEMM_TILE_K
                              : linearThread / GEMM_TILE_M;
        const ulong globalLhsRow = ulong(group.x) * GEMM_TILE_M + lhsRow;
        const uint globalLhsK = kBase + lhsK;
        T lhsValue = T(0);
        if (globalLhsRow < p.m && globalLhsK < p.k) {
            const long lhsIndex =
                lhsBatch + (p.transposeLhs
                                ? long(globalLhsK) * p.lhsStrides[0] +
                                      long(globalLhsRow) * p.lhsStrides[1]
                                : long(globalLhsRow) * p.lhsStrides[0] +
                                      long(globalLhsK) * p.lhsStrides[1]);
            lhsValue = gemmConjugate(lhs[lhsIndex], p.conjugateLhs);
        }
        lhsTile[lhsRow * GEMM_TILE_K + lhsK] = lhsValue;

        const uint rhsK = p.transposeRhs
                              ? linearThread / GEMM_TILE_N
                              : linearThread % GEMM_TILE_K;
        const uint rhsCol = p.transposeRhs
                                ? linearThread % GEMM_TILE_N
                                : linearThread / GEMM_TILE_K;
        const uint globalRhsK = kBase + rhsK;
        const ulong globalRhsCol = ulong(group.y) * GEMM_TILE_N + rhsCol;
        T rhsValue = T(0);
        if (globalRhsK < p.k && globalRhsCol < p.n) {
            const long rhsIndex =
                rhsBatch + (p.transposeRhs
                                ? long(globalRhsCol) * p.rhsStrides[0] +
                                      long(globalRhsK) * p.rhsStrides[1]
                                : long(globalRhsK) * p.rhsStrides[0] +
                                      long(globalRhsCol) * p.rhsStrides[1]);
            rhsValue = gemmConjugate(rhs[rhsIndex], p.conjugateRhs);
        }
        rhsTile[rhsK * GEMM_TILE_N + rhsCol] = rhsValue;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < p.m && col < p.n) {
            for (uint k = 0; k < GEMM_TILE_K; ++k) {
                accum += gemmMultiply(lhsTile[tid.x * GEMM_TILE_K + k],
                                      rhsTile[k * GEMM_TILE_N + tid.y]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < p.m && col < p.n) {
        const long outputIndex = long(row) * p.outputStrides[0] +
                                 long(col) * p.outputStrides[1] +
                                 long(batch2) * p.outputStrides[2] +
                                 long(batch3) * p.outputStrides[3];
        output[outputIndex] = gemmScale(
            accum, output[outputIndex], p.alphaReal, p.alphaImag, p.betaReal,
            p.betaImag);
    }
}

#define DEFINE_GEMM_TILED_KERNEL(NAME, TYPE)                                \
kernel void NAME(const device TYPE* lhs [[buffer(0)]],                      \
                 const device TYPE* rhs [[buffer(1)]],                      \
                 device TYPE* output [[buffer(2)]],                         \
                 constant GemmParams& p [[buffer(3)]],                      \
                 threadgroup TYPE* lhsTile [[threadgroup(0)]],              \
                 threadgroup TYPE* rhsTile [[threadgroup(1)]],              \
                 uint3 tid [[thread_position_in_threadgroup]],              \
                 uint3 group [[threadgroup_position_in_grid]]) {            \
    gemmTiledImpl(lhs, rhs, output, p, lhsTile, rhsTile, tid, group);       \
}

DEFINE_GEMM_TILED_KERNEL(gemm_float, float)
DEFINE_GEMM_TILED_KERNEL(gemm_cfloat, float2)

#undef DEFINE_GEMM_TILED_KERNEL
