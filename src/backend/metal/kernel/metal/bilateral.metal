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

struct BilateralParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
    float spatialSigma, chromaticSigma;
};

#define DEFINE_BILATERAL(NAME, TYPE)                                       \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device float* output [[buffer(1)]],                       \
                 constant BilateralParams& p [[buffer(2)]],                \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong total=p.dims[0]*p.dims[1]*p.dims[2]*p.dims[3];            \
    if (gid>=total) return;                                                \
    ulong q=gid; const long x=long(q%p.dims[0]); q/=p.dims[0];            \
    const long y=long(q%p.dims[1]); q/=p.dims[1];                         \
    const ulong z=q%p.dims[2],w=q/p.dims[2];                              \
    const ulong base=z*p.inputStrides[2]+w*p.inputStrides[3];             \
    const float center=float(input[base+ulong(x)*p.inputStrides[0]+       \
                                   ulong(y)*p.inputStrides[1]]);           \
    const float space=clamp(p.spatialSigma,0.0f,11.5f);                   \
    const float color=max(p.chromaticSigma,0.0f);                         \
    const long radius=max(long(space*1.5f),1l);                           \
    const float spatialDen=-2.0f*space*space;                             \
    const float colorDen=-2.0f*color*color;                               \
    float norm=0.0f,result=0.0f;                                          \
    for (long j=-radius;j<=radius;++j)                                    \
      for (long i=-radius;i<=radius;++i) {                                \
        const ulong ix=ulong(clamp(x+i,0l,long(p.dims[0])-1));            \
        const ulong iy=ulong(clamp(y+j,0l,long(p.dims[1])-1));            \
        const float value=float(input[base+ix*p.inputStrides[0]+          \
                                      iy*p.inputStrides[1]]);              \
        const float delta=center-value;                                   \
        const float weight=exp(float(i*i+j*j)/spatialDen+                 \
                               delta*delta/colorDen);                      \
        norm+=weight; result+=value*weight;                               \
      }                                                                    \
    const ulong oo=ulong(x)*p.outputStrides[0]+ulong(y)*p.outputStrides[1]+\
        z*p.outputStrides[2]+w*p.outputStrides[3];                         \
    output[oo]=result/norm;                                                \
}

DEFINE_BILATERAL(bilateral_float, float)
DEFINE_BILATERAL(bilateral_int, int)
DEFINE_BILATERAL(bilateral_uint, uint)
DEFINE_BILATERAL(bilateral_char, char)
DEFINE_BILATERAL(bilateral_uchar, uchar)
DEFINE_BILATERAL(bilateral_short, short)
DEFINE_BILATERAL(bilateral_ushort, ushort)
