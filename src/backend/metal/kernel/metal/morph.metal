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

struct MorphParams {
    ulong outputDims[4], outputStrides[4];
    ulong inputDims[4], inputStrides[4];
    ulong maskDims[4], maskStrides[4];
    uint dilation, volume;
};

#define DEFINE_MORPH(NAME, TYPE, LOW, HIGH)                                \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 const device TYPE* mask [[buffer(1)]],                    \
                 device TYPE* output [[buffer(2)]],                        \
                 constant MorphParams& p [[buffer(3)]],                    \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong total=p.outputDims[0]*p.outputDims[1]*                     \
        p.outputDims[2]*p.outputDims[3];                                   \
    if (gid>=total) return;                                                \
    ulong q=gid; const long x=long(q%p.outputDims[0]); q/=p.outputDims[0]; \
    const long y=long(q%p.outputDims[1]); q/=p.outputDims[1];              \
    const long z=long(q%p.outputDims[2]); const ulong w=q/p.outputDims[2]; \
    TYPE result=p.dilation ? TYPE(LOW) : TYPE(HIGH);                       \
    const long rx=long(p.maskDims[0]/2), ry=long(p.maskDims[1]/2);         \
    const long rz=p.volume ? long(p.maskDims[2]/2) : 0;                   \
    const ulong mzCount=p.volume ? p.maskDims[2] : 1;                     \
    for (ulong mz=0; mz<mzCount; ++mz)                                    \
      for (ulong my=0; my<p.maskDims[1]; ++my)                            \
        for (ulong mx=0; mx<p.maskDims[0]; ++mx) {                        \
            const ulong mo=mx*p.maskStrides[0]+my*p.maskStrides[1]+       \
                           mz*p.maskStrides[2];                            \
            if (!(mask[mo]>TYPE(0))) continue;                            \
            const long ix=x+long(mx)-rx, iy=y+long(my)-ry;                \
            const long iz=p.volume ? z+long(mz)-rz : z;                   \
            if (ix<0||iy<0||iz<0||ix>=long(p.inputDims[0])||              \
                iy>=long(p.inputDims[1])||iz>=long(p.inputDims[2]))       \
                continue;                                                 \
            const ulong io=ulong(ix)*p.inputStrides[0]+                   \
                ulong(iy)*p.inputStrides[1]+ulong(iz)*p.inputStrides[2]+  \
                w*p.inputStrides[3];                                      \
            result=p.dilation ? max(result,input[io]) :                   \
                                min(result,input[io]);                     \
        }                                                                  \
    const ulong oo=ulong(x)*p.outputStrides[0]+ulong(y)*p.outputStrides[1]+\
        ulong(z)*p.outputStrides[2]+w*p.outputStrides[3];                  \
    output[oo]=result;                                                     \
}

DEFINE_MORPH(morph_float, float, -INFINITY, INFINITY)
DEFINE_MORPH(morph_int, int, (-2147483647-1), 2147483647)
DEFINE_MORPH(morph_uint, uint, 0u, 0xffffffffu)
DEFINE_MORPH(morph_char, char, -128, 127)
DEFINE_MORPH(morph_uchar, uchar, 0, 255)
DEFINE_MORPH(morph_short, short, -32768, 32767)
DEFINE_MORPH(morph_ushort, ushort, 0, 65535)
