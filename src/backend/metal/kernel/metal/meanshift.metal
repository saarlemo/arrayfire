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

struct MeanshiftParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
    float spatialSigma, chromaticSigma;
    uint iterations, color;
};

#define DEFINE_MEANSHIFT(NAME, TYPE)                                       \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device TYPE* output [[buffer(1)]],                        \
                 constant MeanshiftParams& p [[buffer(2)]],                \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong planes=p.color ? 1 : p.dims[2];                            \
    const ulong total=p.dims[0]*p.dims[1]*planes*p.dims[3];               \
    if (gid>=total) return;                                                \
    ulong q=gid; const long x=long(q%p.dims[0]); q/=p.dims[0];            \
    const long y=long(q%p.dims[1]); q/=p.dims[1];                         \
    const ulong z=q%planes,w=q/planes;                                    \
    const ulong channels=p.color ? p.dims[2] : 1;                         \
    const ulong base=z*p.inputStrides[2]+w*p.inputStrides[3];             \
    float center[4]={0.0f,0.0f,0.0f,0.0f};                               \
    for (ulong ch=0;ch<channels;++ch)                                     \
        center[ch]=float(input[base+ulong(x)*p.inputStrides[0]+           \
                               ulong(y)*p.inputStrides[1]+                \
                               ch*p.inputStrides[2]]);                     \
    long mx=x,my=y;                                                        \
    const long radius=max(long(p.spatialSigma*1.5f),1l);                  \
    const float cvar=p.chromaticSigma*p.chromaticSigma;                   \
    for (uint it=0;it<p.iterations;++it) {                                \
        const long oldx=mx,oldy=my; long sx=0,sy=0; uint count=0;         \
        float mean[4]={0.0f,0.0f,0.0f,0.0f};                             \
        for (long j=-radius;j<=radius;++j) {                              \
          const long iy=my+j; if (iy<0||iy>=long(p.dims[1])) continue;    \
          for (long i=-radius;i<=radius;++i) {                            \
            const long ix=mx+i; if (ix<0||ix>=long(p.dims[0])) continue;  \
            float values[4]; float norm=0.0f;                             \
            for (ulong ch=0;ch<channels;++ch) {                           \
              values[ch]=float(input[base+ulong(ix)*p.inputStrides[0]+   \
                ulong(iy)*p.inputStrides[1]+ch*p.inputStrides[2]]);       \
              const float d=center[ch]-values[ch]; norm+=d*d;            \
            }                                                             \
            if (norm<=cvar) { for (ulong ch=0;ch<channels;++ch)           \
                mean[ch]+=values[ch]; sx+=ix; sy+=iy; ++count; }          \
          }                                                               \
        }                                                                 \
        if (!count) break;                                                \
        const float inv=1.0f/float(count); mx=long(trunc(float(sx)*inv)); \
        my=long(trunc(float(sy)*inv)); float norm=0.0f;                    \
        for (ulong ch=0;ch<channels;++ch) {                               \
          mean[ch]=trunc(mean[ch]*inv); const float d=mean[ch]-center[ch];\
          norm+=d*d;                                                      \
        }                                                                 \
        const bool stop=(mx==oldx&&my==oldy)||                            \
            (float(abs(mx-oldx)+abs(my-oldy))+norm<=1.0f);                \
        for (ulong ch=0;ch<channels;++ch) center[ch]=mean[ch];            \
        if (stop) break;                                                  \
    }                                                                     \
    const ulong oo=ulong(x)*p.outputStrides[0]+ulong(y)*p.outputStrides[1]+\
        z*p.outputStrides[2]+w*p.outputStrides[3];                         \
    for (ulong ch=0;ch<channels;++ch) output[oo+ch*p.outputStrides[2]]=TYPE(center[ch]);\
}

DEFINE_MEANSHIFT(meanshift_float, float)
DEFINE_MEANSHIFT(meanshift_int, int)
DEFINE_MEANSHIFT(meanshift_uint, uint)
DEFINE_MEANSHIFT(meanshift_long, long)
DEFINE_MEANSHIFT(meanshift_ulong, ulong)
DEFINE_MEANSHIFT(meanshift_char, char)
DEFINE_MEANSHIFT(meanshift_uchar, uchar)
DEFINE_MEANSHIFT(meanshift_short, short)
DEFINE_MEANSHIFT(meanshift_ushort, ushort)
