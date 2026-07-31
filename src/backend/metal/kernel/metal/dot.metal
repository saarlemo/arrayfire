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

kernel void dot_float(const device float* lhs [[buffer(0)]],
                      const device float* rhs [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant DotParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid) return;
    float value=0.0f;
    for (ulong i=0;i<p.length;++i)
        value+=lhs[i*p.lhsStride]*rhs[i*p.rhsStride];
    output[0]=value;
}

kernel void dot_cfloat(const device float2* lhs [[buffer(0)]],
                       const device float2* rhs [[buffer(1)]],
                       device float2* output [[buffer(2)]],
                       constant DotParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid) return;
    float2 value=float2(0.0f);
    for (ulong i=0;i<p.length;++i) {
        float2 a=lhs[i*p.lhsStride],b=rhs[i*p.rhsStride];
        if (p.conjugateLhs) a=dotConjugate(a);
        if (p.conjugateRhs) b=dotConjugate(b);
        value+=dotMultiply(a,b);
    }
    output[0]=value;
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

template<typename T>
inline void gemmImpl(const device T* lhs, const device T* rhs,
                     device T* output, constant GemmParams& p, const uint gid) {
    const ulong total = p.outputDims[0] * p.outputDims[1] * p.outputDims[2] *
                        p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid, c[4];
    for (uint d = 0; d < 4; ++d) {
        c[d] = q % p.outputDims[d];
        q /= p.outputDims[d];
    }
    const ulong row = c[0];
    const ulong col = c[1];
    const ulong lhsB2 = p.lhsDims[2] == 1 ? 0 : c[2];
    const ulong lhsB3 = p.lhsDims[3] == 1 ? 0 : c[3];
    const ulong rhsB2 = p.rhsDims[2] == 1 ? 0 : c[2];
    const ulong rhsB3 = p.rhsDims[3] == 1 ? 0 : c[3];
    const long lhsBatch = long(lhsB2) * p.lhsStrides[2] +
                          long(lhsB3) * p.lhsStrides[3];
    const long rhsBatch = long(rhsB2) * p.rhsStrides[2] +
                          long(rhsB3) * p.rhsStrides[3];
    T accum = T(0);
    for (uint i = 0; i < p.k; ++i) {
        const long lhsIndex =
            lhsBatch + (p.transposeLhs
                            ? long(i) * p.lhsStrides[0] +
                                  long(row) * p.lhsStrides[1]
                            : long(row) * p.lhsStrides[0] +
                                  long(i) * p.lhsStrides[1]);
        const long rhsIndex =
            rhsBatch + (p.transposeRhs
                            ? long(col) * p.rhsStrides[0] +
                                  long(i) * p.rhsStrides[1]
                            : long(i) * p.rhsStrides[0] +
                                  long(col) * p.rhsStrides[1]);
        accum += gemmMultiply(gemmConjugate(lhs[lhsIndex], p.conjugateLhs),
                              gemmConjugate(rhs[rhsIndex], p.conjugateRhs));
    }
    const long outputIndex = long(row) * p.outputStrides[0] +
                             long(col) * p.outputStrides[1] +
                             long(c[2]) * p.outputStrides[2] +
                             long(c[3]) * p.outputStrides[3];
    output[outputIndex] = gemmScale(
        accum, output[outputIndex], p.alphaReal, p.alphaImag, p.betaReal,
        p.betaImag);
}

kernel void gemm_float(const device float* lhs [[buffer(0)]],
                       const device float* rhs [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant GemmParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    gemmImpl(lhs, rhs, output, p, gid);
}

kernel void gemm_cfloat(const device float2* lhs [[buffer(0)]],
                        const device float2* rhs [[buffer(1)]],
                        device float2* output [[buffer(2)]],
                        constant GemmParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    gemmImpl(lhs, rhs, output, p, gid);
}
