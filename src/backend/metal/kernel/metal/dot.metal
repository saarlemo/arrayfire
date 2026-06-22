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
