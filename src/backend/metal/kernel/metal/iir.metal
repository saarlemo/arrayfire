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

struct IirParams {
    ulong outputDims[4], outputStrides[4];
    ulong coefficientStrides[4], feedbackDims[4], feedbackStrides[4];
    uint feedbackBatched;
};

inline float2 complexMultiply(float2 a, float2 b) {
    return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

inline float2 complexDivide(float2 a, float2 b) {
    const float scale=b.x*b.x+b.y*b.y;
    return float2((a.x*b.x+a.y*b.y)/scale,
                  (a.y*b.x-a.x*b.y)/scale);
}

kernel void iir_float(const device float* coefficients [[buffer(0)]],
                      const device float* feedback [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant IirParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    const ulong series=p.outputDims[1]*p.outputDims[2]*p.outputDims[3];
    if (gid>=series) return;
    ulong q=gid; const ulong y=q%p.outputDims[1]; q/=p.outputDims[1];
    const ulong z=q%p.outputDims[2], w=q/p.outputDims[2];
    const ulong oo=y*p.outputStrides[1]+z*p.outputStrides[2]+w*p.outputStrides[3];
    const ulong co=y*p.coefficientStrides[1]+z*p.coefficientStrides[2]+w*p.coefficientStrides[3];
    const ulong ao=p.feedbackBatched ? y*p.feedbackStrides[1]+z*p.feedbackStrides[2]+w*p.feedbackStrides[3] : 0;
    for (ulong x=0; x<p.outputDims[0]; ++x) {
        float value=coefficients[co+x*p.coefficientStrides[0]];
        const ulong count=min(x,p.feedbackDims[0]-1);
        for (ulong k=1; k<=count; ++k)
            value-=feedback[ao+k*p.feedbackStrides[0]]*output[oo+(x-k)*p.outputStrides[0]];
        output[oo+x*p.outputStrides[0]]=value/feedback[ao];
    }
}

kernel void iir_cfloat(const device float2* coefficients [[buffer(0)]],
                       const device float2* feedback [[buffer(1)]],
                       device float2* output [[buffer(2)]],
                       constant IirParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    const ulong series=p.outputDims[1]*p.outputDims[2]*p.outputDims[3];
    if (gid>=series) return;
    ulong q=gid; const ulong y=q%p.outputDims[1]; q/=p.outputDims[1];
    const ulong z=q%p.outputDims[2], w=q/p.outputDims[2];
    const ulong oo=y*p.outputStrides[1]+z*p.outputStrides[2]+w*p.outputStrides[3];
    const ulong co=y*p.coefficientStrides[1]+z*p.coefficientStrides[2]+w*p.coefficientStrides[3];
    const ulong ao=p.feedbackBatched ? y*p.feedbackStrides[1]+z*p.feedbackStrides[2]+w*p.feedbackStrides[3] : 0;
    for (ulong x=0; x<p.outputDims[0]; ++x) {
        float2 value=coefficients[co+x*p.coefficientStrides[0]];
        const ulong count=min(x,p.feedbackDims[0]-1);
        for (ulong k=1; k<=count; ++k)
            value-=complexMultiply(feedback[ao+k*p.feedbackStrides[0]],output[oo+(x-k)*p.outputStrides[0]]);
        output[oo+x*p.outputStrides[0]]=complexDivide(value,feedback[ao]);
    }
}
